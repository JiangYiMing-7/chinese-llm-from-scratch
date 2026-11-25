#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2: 数据混合采样。

- 从 BAAI/CCI3-HQ（HuggingFace streaming）采样 70%
- 从 Task 1 清洗后的自建语料采样 30%
- 按字符数控制总规模（默认约 60 亿字，对应约 2B tokens）
- 用 shuffle buffer 做近似全局打散，输出到 mixed_corpus.jsonl

数据质量保证（两阶段去重）：
  Phase 1：扫描已清洗的本地语料，把每篇文档的 SHA-256 + MinHash
           全部加入全局去重索引。
  Phase 2：CCI3-HQ streaming 时对每篇文档：
           ① 轻量质量过滤（长度 + CJK 比例）
           ② SHA-256 精确去重（跨源，拒绝与本地语料完全相同的文档）
           ③ MinHash 近似去重（跨源，拒绝与本地语料 Jaccard ≥ 阈值的文档）
  上述流程保证混合语料在两个来源之间无重复，同时 CCI3-HQ 内部也不引入
  新的近重复文档。

全流程流式处理：CCI3-HQ 用 datasets streaming，自建语料逐行读，
打散用固定大小的 shuffle buffer，不会一次性把全部语料读进内存。
"""

import argparse
import hashlib
import json
import logging
import os
import random
import re
import time

from dedup_utils import (
    ExactHashStore,
    ensure_distinct_paths,
    ensure_parent_dir,
    reset_redis_prefix,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mix_datasets")

_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(_CJK_RE.findall(text)) / len(text)


def _make_minhash(text: str, num_perm: int):
    from datasketch import MinHash
    mh = MinHash(num_perm=num_perm)
    for i in range(max(1, len(text) - 2)):
        mh.update(text[i: i + 3].encode("utf-8"))
    return mh


def _build_lsh(num_perm: int, threshold: float,
               backend: str, redis_host: str, redis_port: int, redis_prefix: str):
    from datasketch import MinHashLSH
    if backend == "redis":
        storage_config = {
            "type": "redis",
            "redis": {"host": redis_host, "port": redis_port},
            "basename": redis_prefix.encode("utf-8"),
        }
        logger.info("去重后端：redis (%s:%d)", redis_host, redis_port)
        return MinHashLSH(threshold=threshold, num_perm=num_perm,
                          storage_config=storage_config)
    logger.info("去重后端：memory（大规模语料请改用 --dedup-backend redis）")
    return MinHashLSH(threshold=threshold, num_perm=num_perm)


# --------------------------------------------------------------------------
# 数据源读取
# --------------------------------------------------------------------------

def iter_local_jsonl(path: str, text_field: str):
    """流式读取本地 jsonl，yield 文本字符串。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field, "")
            if isinstance(text, str) and text:
                yield text


def iter_cci3_hq(text_field: str, hf_token: str | None = None):
    """流式读取 BAAI/CCI3-HQ，yield 文本字符串。"""
    from datasets import load_dataset

    ds = load_dataset(
        "BAAI/CCI3-HQ",
        split="train",
        streaming=True,
        token=hf_token,
    )
    for ex in ds:
        text = ex.get(text_field) or ex.get("text") or ""
        if isinstance(text, str) and text:
            yield text


# --------------------------------------------------------------------------
# 两阶段去重
# --------------------------------------------------------------------------

def build_index_from_local(
    local_path: str, text_field: str,
    exact_hashes: ExactHashStore, lsh, num_perm: int,
) -> int:
    """Phase 1：把已清洗本地语料的指纹全部加入全局去重索引。

    Phase 2 中 CCI3-HQ 的每条文档都会和这个索引比对，
    确保跨源无精确/近似重复。
    """
    t0 = time.time()
    logger.info("Phase 1：构建本地语料去重索引（%s）...", local_path)
    n = 0
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field, "")
            if not isinstance(text, str) or not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if exact_hashes.contains(digest):
                continue
            lsh.insert(f"local-{n}", _make_minhash(text, num_perm))
            exact_hashes.add(digest)
            n += 1
            if n % 50000 == 0:
                logger.info("  已索引 %d 篇本地文档...", n)
    logger.info("Phase 1 完成：%d 篇本地文档加入索引，用时 %.1fs", n, time.time() - t0)
    return n


def iter_cci3_with_filter(
    cci_raw, exact_hashes: ExactHashStore, lsh, num_perm: int,
    min_len: int, max_len: int, cjk_threshold: float, stats: dict,
):
    """Phase 2：对 CCI3-HQ 做轻量过滤 + 跨源去重后 yield 通过的文本。

    CCI3-HQ 是 BAAI 精选语料，只做必要检查：
      1. 长度过滤（排除极短/极长异常文档）
      2. CJK 比例检查（排除偶发的非中文条目）
      3. SHA-256 精确去重（全局，含跨源）
      4. MinHash 近似去重（全局，含跨源）
    跳过标点/数字/重复子串检测（HQ 语料无需，且耗时高）。
    """
    for text in cci_raw:
        stats["cci_total"] += 1

        n = len(text)
        if n < min_len or n > max_len:
            stats["cci_drop_length"] += 1
            continue

        if _cjk_ratio(text[:500]) < cjk_threshold:
            stats["cci_drop_language"] += 1
            continue

        h = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if exact_hashes.contains(h):
            stats["cci_drop_exact_dup"] += 1
            continue

        mh = _make_minhash(text, num_perm)
        if lsh.query(mh):
            stats["cci_drop_dedup"] += 1
            continue
        lsh.insert(f"cci-{stats['cci_kept']}", mh)
        exact_hashes.add(h)

        stats["cci_kept"] += 1
        yield text


# --------------------------------------------------------------------------
# 混合主逻辑
# --------------------------------------------------------------------------

def shuffle_buffer(source, buffer_size: int, rng: random.Random):
    """对迭代器做 shuffle buffer 近似打散（类似 tf.data.shuffle）。"""
    buf = []
    for item in source:
        if len(buf) < buffer_size:
            buf.append(item)
        else:
            idx = rng.randint(0, buffer_size - 1)
            yield buf[idx]
            buf[idx] = item
    rng.shuffle(buf)
    yield from buf


def take_until_chars(source, char_budget: int, source_name: str,
                     max_docs: int | None = None):
    """从 source 取文档，直到累计字符数达到 char_budget 或文档数达到 max_docs。"""
    used = 0
    n_docs = 0
    for text in source:
        if used >= char_budget:
            break
        if max_docs and n_docs >= max_docs:
            logger.info("[%s] 已达 --max-docs=%d 上限", source_name, max_docs)
            break
        yield text
        used += len(text)
        n_docs += 1
        if n_docs % 50000 == 0:
            logger.info("[%s] 已采样 %d 篇 / %.2f 亿字",
                        source_name, n_docs, used / 1e8)
    logger.info("[%s] 采样完成：%d 篇，共 %.3f 亿字",
                source_name, n_docs, used / 1e8)


def mix(args):
    ensure_distinct_paths(args.local_corpus, args.cci_local, args.output)
    ensure_parent_dir(args.output)

    if args.dedup_backend == "redis" and args.reset_dedup_store:
        reset_redis_prefix(args.redis_host, args.redis_port, args.redis_prefix)

    rng = random.Random(args.seed)

    total_chars = int(args.total_chars)
    cci_budget = int(total_chars * args.cci_ratio)
    local_budget = total_chars - cci_budget
    logger.info("目标总字符数：%.2f 亿", total_chars / 1e8)
    logger.info("  CCI3-HQ  预算：%.2f 亿字 (%.0f%%)",
                cci_budget / 1e8, 100 * args.cci_ratio)
    logger.info("  自建语料 预算：%.2f 亿字 (%.0f%%)",
                local_budget / 1e8, 100 * (1 - args.cci_ratio))

    cci_stats = {
        "cci_total": 0, "cci_drop_length": 0, "cci_drop_language": 0,
        "cci_drop_exact_dup": 0, "cci_drop_dedup": 0, "cci_kept": 0,
    }
    if args.cross_dedup:
        lsh = _build_lsh(
            args.num_perm, args.dedup_threshold,
            args.dedup_backend, args.redis_host, args.redis_port, args.redis_prefix,
        )
        exact_hashes = ExactHashStore(
            args.dedup_backend,
            args.redis_host,
            args.redis_port,
            args.redis_prefix,
        )
        build_index_from_local(args.local_corpus, args.text_field,
                               exact_hashes, lsh, args.num_perm)
    else:
        logger.info("跨源去重已关闭（--no-cross-dedup）")

    t0 = time.time()

    # CCI 来源
    if args.cci_local:
        logger.info("CCI 来源：本地文件 %s（离线模式）", args.cci_local)
        cci_raw = iter_local_jsonl(args.cci_local, args.text_field)
    else:
        logger.info("CCI 来源：HuggingFace streaming BAAI/CCI3-HQ")
        cci_raw = iter_cci3_hq(args.text_field, args.hf_token)

    # Phase 2：CCI3-HQ 轻量过滤 + 跨源去重
    if args.cross_dedup:
        cci_filtered = iter_cci3_with_filter(
            cci_raw, exact_hashes, lsh, args.num_perm,
            min_len=args.cci_min_len, max_len=args.max_len,
            cjk_threshold=args.cci_cjk_threshold,
            stats=cci_stats,
        )
    else:
        cci_filtered = cci_raw

    cci_stream = take_until_chars(cci_filtered, cci_budget, "CCI3-HQ", args.max_docs)
    local_stream = take_until_chars(
        iter_local_jsonl(args.local_corpus, args.text_field),
        local_budget, "自建语料", args.max_docs,
    )

    def tagged(stream, tag):
        for text in stream:
            yield {"text": text, "source": tag}

    def interleave(a, b):
        ia, ib = iter(a), iter(b)
        a_done = b_done = False
        while not (a_done and b_done):
            if not a_done:
                try:
                    yield next(ia)
                except StopIteration:
                    a_done = True
            if not b_done:
                try:
                    yield next(ib)
                except StopIteration:
                    b_done = True

    merged = interleave(
        tagged(cci_stream, "cci3-hq"),
        tagged(local_stream, "local"),
    )

    n_written = 0
    chars_written = 0
    src_counter = {"cci3-hq": 0, "local": 0}
    tmp_output = args.output + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as out:
        for obj in shuffle_buffer(merged, args.shuffle_buffer, rng):
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1
            chars_written += len(obj["text"])
            src_counter[obj["source"]] += 1
            if n_written % 50000 == 0:
                logger.info("已写出 %d 篇 / %.2f 亿字",
                            n_written, chars_written / 1e8)
    os.replace(tmp_output, args.output)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("混合完成：%d 篇，共 %.3f 亿字，约 %.0f M tokens（估算），用时 %.1fs",
                n_written, chars_written / 1e8, chars_written * 0.8 / 1e6, elapsed)
    logger.info("来源：CCI3-HQ %d 篇 (%.1f%%) / 自建 %d 篇 (%.1f%%)",
                src_counter["cci3-hq"], 100 * src_counter["cci3-hq"] / max(1, n_written),
                src_counter["local"], 100 * src_counter["local"] / max(1, n_written))

    if args.cross_dedup and cci_stats["cci_total"] > 0:
        total = cci_stats["cci_total"]
        logger.info("-" * 60)
        logger.info("CCI3-HQ 质量过滤 + 跨源去重（共扫描 %d 篇）：", total)
        for key, label in [
            ("cci_drop_length",    "长度过滤"),
            ("cci_drop_language",  "语言过滤"),
            ("cci_drop_exact_dup", "精确去重"),
            ("cci_drop_dedup",     "近似去重"),
            ("cci_kept",           "通过保留"),
        ]:
            v = cci_stats[key]
            logger.info("  %-8s %6d  (%.2f%%)", label, v, 100 * v / total)

    logger.info("输出文件：%s", args.output)
    return {
        "n_written": n_written,
        "chars_written": chars_written,
        "source_counts": src_counter,
        "cci_stats": cci_stats,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="数据混合采样（Task 2）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--local-corpus", required=True,
                   help="Task 1 清洗后的自建语料 jsonl 路径")
    p.add_argument("--output", default="mixed_corpus.jsonl",
                   help="混合后输出 jsonl 路径")
    p.add_argument("--text-field", default="text", help="文本字段名")

    p.add_argument("--total-chars", type=float, default=6e9,
                   help="目标总字符数（默认 60 亿字，约对应 2B tokens）")
    p.add_argument("--cci-ratio", type=float, default=0.70,
                   help="CCI3-HQ 占比，其余为自建语料")
    p.add_argument("--shuffle-buffer", type=int, default=100000,
                   help="shuffle buffer 大小（越大打散越均匀，越占内存）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token（CCI3-HQ 需要登录访问时填）")
    p.add_argument("--cci-local", default=None,
                   help="从本地 jsonl 读取 CCI（离线/验证用，跳过 HF 下载）")
    p.add_argument("--max-docs", type=int, default=None,
                   help="每个来源最多采样的文档数（调试用）")

    # 跨源去重
    p.add_argument("--cross-dedup", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="开启 CCI3-HQ 轻量过滤 + 跨源去重（默认开启）")
    p.add_argument("--cci-min-len", type=int, default=50,
                   help="CCI3-HQ 文档最短字数")
    p.add_argument("--max-len", type=int, default=100000,
                   help="文档最长字数上限（与 clean_corpus 保持一致）")
    p.add_argument("--cci-cjk-threshold", type=float, default=0.30,
                   help="CCI3-HQ CJK 字符比例下限（轻量语言过滤）")
    p.add_argument("--dedup-threshold", type=float, default=0.85,
                   help="MinHash 去重 Jaccard 阈值")
    p.add_argument("--num-perm", type=int, default=128,
                   help="MinHash 排列数（精度/速度权衡）")
    p.add_argument("--dedup-backend", choices=["memory", "redis"],
                   default="memory",
                   help="MinHashLSH 后端；大规模语料建议 redis")
    p.add_argument("--redis-host", default="localhost", help="Redis 主机")
    p.add_argument("--redis-port", type=int, default=6379, help="Redis 端口")
    p.add_argument("--redis-prefix", default="mixcorpus",
                   help="Redis 键前缀（隔离多次任务）")
    p.add_argument("--reset-dedup-store", action="store_true",
                   help="运行前清理当前 redis-prefix 下的旧去重索引；仅 redis 后端有效")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mix(args)
