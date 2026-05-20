#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1: 自建语料清洗流水线（中文维基百科 + CLUECorpus2020）。

按顺序执行的过滤器：
  1. 编码统一（确保 UTF-8、去 BOM、规范化空白）
  2. 长度过滤（去掉 < min_len 或 > max_len 字的文档）
  3. 语言过滤（fasttext-langid 优先，回退 langdetect，保留中文文档）
  4. 质量过滤：
       - 标点符号比例 > punct_ratio 的文档过滤
       - 数字字符比例 > digit_ratio 的文档过滤
       - 最长重复子串长度 > repeat_ratio * 文档长度 的文档过滤
  4.5 SHA-256 精确去重（对完全相同的文档先做低成本精确哈希过滤）
  5. 文档级 MinHash 去重（datasketch，Jaccard 阈值 dedup_threshold，过滤近似重复）

输入：jsonl，每行一个对象，含 "text" 字段。
输出：清洗后的 jsonl + 清洗报告（json）。

全流程流式处理，不会一次性把语料读进内存（去重的 LSH 索引除外）。
"""

import argparse
import hashlib
import json
import logging
import re
import os
import sys
import time
import unicodedata

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
logger = logging.getLogger("clean_corpus")


# --------------------------------------------------------------------------
# 字符类别判定
# --------------------------------------------------------------------------
# 中文字符（CJK 统一表意文字主区 + 扩展 A）
_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
# 标点：ASCII 标点 + 中文常用标点
_PUNCT_RE = re.compile(
    r"[　-〿＀-￯!-/:-@\[-`{-~，。！？、；：“”‘’（）《》【】—…·]"
)
_DIGIT_RE = re.compile(r"[0-9０-９]")
# 行内空白（空格/制表符等，不含换行）压成单空格；换行单独保留作段落边界
_INLINE_WS_RE = re.compile(r"[^\S\n]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")


def normalize_text(text):
    """编码统一：去 BOM、NFKC 规范化、压缩行内空白、保留段落换行。

    保留换行的原因：换行是天然的段落/句子边界。若把换行也压成空格，每篇
    文档会塌成一整行——训练 tokenizer 时约 77% 的中文维基文档会超过
    SentencePiece 的 max_sentence_length(4192 字节)，被整条丢弃，造成大量
    训练语料损失。保留 \\n 后 train_tokenizer.py 可按段落切句。
    """
    if not isinstance(text, str):
        text = str(text)
    # 去 BOM / 零宽字符
    text = text.replace("﻿", "").replace("​", "")
    # Unicode 规范化（全角->半角、兼容字符统一）
    text = unicodedata.normalize("NFKC", text)
    # 行内空白压成单空格，但保留换行
    text = _INLINE_WS_RE.sub(" ", text)
    # 连续 3+ 换行压成段落分隔（双换行），逐行去首尾空白
    text = _MULTI_NL_RE.sub("\n\n", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def char_ratio(text, pattern):
    """返回 text 中匹配 pattern 的字符占比。"""
    if not text:
        return 0.0
    return len(pattern.findall(text)) / len(text)


def longest_repeated_substring_len(text, max_check_len=10000):
    """返回最长重复子串（出现 >= 2 次，可重叠）的长度。

    二分长度 + 滚动哈希：在长度 L 上检查是否存在出现两次的子串。
    复杂度约 O(n log n)，对 <= max_check_len 的文档足够快。
    """
    n = len(text)
    if n < 2:
        return 0
    if n > max_check_len:
        text = text[:max_check_len]
        n = max_check_len

    data = [ord(c) for c in text]
    base = 131542391
    mod = (1 << 61) - 1

    # 预计算前缀哈希与幂
    prefix = [0] * (n + 1)
    power = [1] * (n + 1)
    for i in range(n):
        prefix[i + 1] = (prefix[i] * base + data[i]) % mod
        power[i + 1] = (power[i] * base) % mod

    def sub_hash(start, length):
        return (prefix[start + length] - prefix[start] * power[length]) % mod

    def has_dup_of_len(length):
        """长度为 length 的子串中是否存在重复。"""
        if length == 0:
            return True
        seen = set()
        for start in range(n - length + 1):
            h = sub_hash(start, length)
            if h in seen:
                return True
            seen.add(h)
        return False

    lo, hi, ans = 1, n - 1, 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if has_dup_of_len(mid):
            ans = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ans


# --------------------------------------------------------------------------
# 语言识别（fasttext 优先，回退 langdetect）
# --------------------------------------------------------------------------
class LanguageDetector:
    def __init__(self, fasttext_model_path=None):
        self.backend = None
        self._ft = None
        self._langdetect = None

        if fasttext_model_path:
            try:
                import fasttext

                self._ft = fasttext.load_model(fasttext_model_path)
                self.backend = "fasttext"
                logger.info("语言识别后端：fasttext (%s)", fasttext_model_path)
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("fasttext 加载失败（%s），尝试 langdetect", e)

        try:
            from langdetect import DetectorFactory, detect_langs

            DetectorFactory.seed = 42  # 结果可复现
            self._langdetect = detect_langs
            self.backend = "langdetect"
            logger.info("语言识别后端：langdetect")
        except Exception as e:  # noqa: BLE001
            logger.warning("langdetect 不可用（%s），将退化为字符比例启发式", e)
            self.backend = "heuristic"

    def chinese_prob(self, text):
        """返回该文档为中文的置信度 [0, 1]。"""
        sample = text[:1000]  # 取前若干字符即可，加速
        if self.backend == "fasttext":
            labels, probs = self._ft.predict(sample.replace("\n", " "), k=5)
            for label, prob in zip(labels, probs):
                if label.replace("__label__", "") in ("zh", "zho", "cmn"):
                    return float(prob)
            return 0.0
        if self.backend == "langdetect":
            # 先看 CJK 字符占比：繁体中文 > 0.5，韩文(Hangul)/英文极低
            # 这是最直接的中文指标，不受 langdetect 置信度影响
            cjk = char_ratio(sample, _CJK_RE)
            if cjk > 0.5:
                return 1.0

            # CJK 不足 50% 时再跑 langdetect，只要有任意 zh 变体就通过
            # （不取 prob 值，避免置信度 < lang_threshold 导致误过滤）
            try:
                for item in self._langdetect(sample):
                    if item.lang.startswith("zh"):
                        return 1.0
            except Exception:  # noqa: BLE001
                pass
            return 0.0
        # 启发式：直接用 CJK 字符占比
        return char_ratio(sample, _CJK_RE)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------
def iter_jsonl(path):
    """流式读取 jsonl，逐行 yield dict。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("第 %d 行 JSON 解析失败，已跳过", line_no)


def build_lsh(args):
    """构建 MinHashLSH 索引。

    内存行为说明（回应大规模去重的内存担忧）：
      - 本流程是「单遍流式」的：逐文档读入、计算 MinHash、查询/插入，
        不会把整个语料或全部 MinHash 对象的列表常驻内存。
      - 但 LSH 索引本身必须保留所有「已保留文档」的 band 桶才能做跨文档
        去重，索引大小 ~ O(保留文档数)。百万级文档下 memory 后端会占数 GB。
      - 大规模场景请用 --dedup-backend redis，把索引落到 Redis，
        Python 进程内存不再随语料规模增长。
    """
    from datasketch import MinHashLSH

    if args.dedup_backend == "redis":
        storage_config = {
            "type": "redis",
            "redis": {"host": args.redis_host, "port": args.redis_port},
            "basename": args.redis_prefix.encode("utf-8"),
        }
        logger.info("MinHash 去重后端：redis (%s:%d)",
                    args.redis_host, args.redis_port)
        return MinHashLSH(
            threshold=args.dedup_threshold,
            num_perm=args.num_perm,
            storage_config=storage_config,
        )
    logger.info("MinHash 去重后端：memory（大规模请改用 --dedup-backend redis）")
    return MinHashLSH(threshold=args.dedup_threshold, num_perm=args.num_perm)


def clean(args):
    ensure_distinct_paths(args.input, args.output, args.report)
    ensure_parent_dir(args.output)
    ensure_parent_dir(args.report)

    if args.dedup_backend == "redis" and args.reset_dedup_store:
        reset_redis_prefix(args.redis_host, args.redis_port, args.redis_prefix)

    detector = LanguageDetector(args.fasttext_model)

    # MinHash 去重所需组件（流式：边读边建索引）
    from datasketch import MinHash

    lsh = build_lsh(args)
    exact_hashes = ExactHashStore(
        args.dedup_backend,
        args.redis_host,
        args.redis_port,
        args.redis_prefix,
    )

    stats = {
        "total": 0,
        "drop_empty": 0,
        "drop_length": 0,
        "drop_language": 0,
        "drop_punct": 0,
        "drop_digit": 0,
        "drop_repeat": 0,
        "drop_exact_dup": 0,   # SHA-256 精确去重
        "drop_dedup": 0,       # MinHash 近似去重
        "kept": 0,
    }
    # 保留语料的规模统计（O(1) 内存，只存运行聚合量）
    kept_chars = 0
    kept_min_len = None
    kept_max_len = 0

    def make_minhash(text):
        """用字符 3-gram 作为 shingle 构建 MinHash。"""
        mh = MinHash(num_perm=args.num_perm)
        shingles = {text[i : i + 3] for i in range(max(1, len(text) - 2))}
        for sh in shingles:
            mh.update(sh.encode("utf-8"))
        return mh

    t0 = time.time()
    tmp_output = args.output + ".tmp"
    out = open(tmp_output, "w", encoding="utf-8")
    try:
        for obj in iter_jsonl(args.input):
            if args.max_docs and stats["total"] >= args.max_docs:
                logger.info("已达 --max-docs=%d 上限，停止读取", args.max_docs)
                break
            stats["total"] += 1
            if stats["total"] % 50000 == 0:
                logger.info(
                    "已处理 %d 篇，保留 %d 篇，用时 %.1fs",
                    stats["total"], stats["kept"], time.time() - t0,
                )

            raw = obj.get(args.text_field, "")

            # 步骤 1：编码统一
            text = normalize_text(raw)
            if not text:
                stats["drop_empty"] += 1
                continue

            # 步骤 2：长度过滤（统一语义：过短或过长一律「整篇丢弃」，
            # 不做截断——半截文档会在句中断裂，价值低于直接丢弃）
            n = len(text)
            if n < args.min_len or n > args.max_len:
                stats["drop_length"] += 1
                continue

            # 步骤 3：语言过滤
            if detector.chinese_prob(text) < args.lang_threshold:
                stats["drop_language"] += 1
                continue

            # 步骤 4：质量过滤
            if char_ratio(text, _PUNCT_RE) > args.punct_ratio:
                stats["drop_punct"] += 1
                continue
            if char_ratio(text, _DIGIT_RE) > args.digit_ratio:
                stats["drop_digit"] += 1
                continue
            if longest_repeated_substring_len(text) > args.repeat_ratio * n:
                stats["drop_repeat"] += 1
                continue

            # 步骤 4.5：SHA-256 精确去重（成本极低，先于 MinHash 运行）
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if exact_hashes.contains(text_hash):
                stats["drop_exact_dup"] += 1
                continue

            # 步骤 5：MinHash 近似去重
            mh = make_minhash(text)
            if lsh.query(mh):  # 已存在近似重复
                stats["drop_dedup"] += 1
                continue
            lsh.insert(f"doc-{stats['kept']}", mh)
            exact_hashes.add(text_hash)

            # 通过全部过滤器，写出
            obj[args.text_field] = text
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            stats["kept"] += 1
            kept_chars += n
            kept_max_len = max(kept_max_len, n)
            kept_min_len = n if kept_min_len is None else min(kept_min_len, n)
    finally:
        out.close()
    os.replace(tmp_output, args.output)

    # 生成清洗报告
    elapsed = time.time() - t0
    total = max(1, stats["total"])
    report = {
        "input": args.input,
        "output": args.output,
        "elapsed_seconds": round(elapsed, 1),
        "params": {
            "min_len": args.min_len,
            "max_len": args.max_len,
            "lang_threshold": args.lang_threshold,
            "punct_ratio": args.punct_ratio,
            "digit_ratio": args.digit_ratio,
            "repeat_ratio": args.repeat_ratio,
            "dedup_threshold": args.dedup_threshold,
            "num_perm": args.num_perm,
            "dedup_backend": args.dedup_backend,
            "redis_prefix": args.redis_prefix if args.dedup_backend == "redis" else None,
        },
        "counts": stats,
        "ratios": {
            k: round(v / total, 4)
            for k, v in stats.items()
            if k != "total"
        },
        # 保留语料规模 —— 判断是否够 2B tokens 目标的关键指标
        "corpus": {
            "kept_chars": kept_chars,
            "avg_doc_len": round(kept_chars / max(1, stats["kept"]), 1),
            "min_doc_len": kept_min_len or 0,
            "max_doc_len": kept_max_len,
            # 估算 token 数：中文约 0.8 tokens/char（tokenizer 训练后可精确化）
            "estimated_tokens": int(kept_chars * 0.8),
        },
    }
    tmp_report = args.report + ".tmp"
    with open(tmp_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_report, args.report)

    logger.info("=" * 60)
    logger.info("清洗完成：输入 %d 篇 -> 保留 %d 篇 (%.2f%%)",
                stats["total"], stats["kept"], 100 * stats["kept"] / total)
    for step in ("drop_empty", "drop_length", "drop_language",
                 "drop_punct", "drop_digit", "drop_repeat",
                 "drop_exact_dup", "drop_dedup"):
        logger.info("  %-14s 过滤 %8d 篇 (%.2f%%)",
                    step, stats[step], 100 * stats[step] / total)
    logger.info("-" * 60)
    logger.info("保留语料规模：%.2f M 字符，约 %.1f M tokens（估算）",
                kept_chars / 1e6, kept_chars * 0.8 / 1e6)
    logger.info("文档长度  平均/最短/最长：%.0f / %d / %d 字",
                kept_chars / max(1, stats["kept"]),
                kept_min_len or 0, kept_max_len)
    logger.info("报告已写入：%s", args.report)


def build_parser():
    p = argparse.ArgumentParser(
        description="中文语料清洗流水线（Task 1）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="输入 jsonl 路径")
    p.add_argument("--output", required=True, help="清洗后 jsonl 输出路径")
    p.add_argument("--report", default="clean_report.json", help="清洗报告输出路径")
    p.add_argument("--text-field", default="text", help="jsonl 中文本字段名")

    p.add_argument("--min-len", type=int, default=50,
                   help="文档最短字数，低于则整篇丢弃")
    p.add_argument("--max-len", type=int, default=100000,
                   help="文档最长字数，超过则整篇丢弃。默认 10 万字足以容纳"
                        "所有正常长文（中文维基最长条目约 4 万字），只拦截"
                        "抓取拼接异常等真·垃圾 blob")

    p.add_argument("--fasttext-model", default=None,
                   help="fasttext 语种识别模型路径（如 lid.176.bin），不填则用 langdetect")
    p.add_argument("--lang-threshold", type=float, default=0.5,
                   help="判定为中文的最低置信度")

    p.add_argument("--punct-ratio", type=float, default=0.30,
                   help="标点符号比例上限，超过则过滤")
    p.add_argument("--digit-ratio", type=float, default=0.40,
                   help="数字字符比例上限，超过则过滤")
    p.add_argument("--repeat-ratio", type=float, default=0.30,
                   help="最长重复子串占文档长度上限，超过则过滤")

    p.add_argument("--dedup-threshold", type=float, default=0.85,
                   help="MinHash 去重 Jaccard 阈值")
    p.add_argument("--num-perm", type=int, default=128,
                   help="MinHash 排列数（精度/速度权衡）")
    p.add_argument("--dedup-backend", choices=["memory", "redis"],
                   default="memory",
                   help="MinHashLSH 存储后端；大规模语料建议用 redis")
    p.add_argument("--redis-host", default="localhost", help="Redis 主机")
    p.add_argument("--redis-port", type=int, default=6379, help="Redis 端口")
    p.add_argument("--redis-prefix", default="cleancorpus",
                   help="Redis 键前缀（多次任务隔离用）")
    p.add_argument("--reset-dedup-store", action="store_true",
                   help="运行前清理当前 redis-prefix 下的旧去重索引；仅 redis 后端有效")

    p.add_argument("--max-docs", type=int, default=None,
                   help="最多处理的输入文档数（Tiny 验证/调试用，默认全部）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        clean(args)
    except KeyboardInterrupt:
        logger.warning("被用户中断")
        sys.exit(130)
