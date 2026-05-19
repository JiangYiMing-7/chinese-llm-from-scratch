#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2: 数据混合采样。

- 从 BAAI/CCI3-HQ（HuggingFace streaming）采样 70%
- 从 Task 1 清洗后的自建语料采样 30%
- 按字符数控制总规模（默认约 60 亿字，对应约 2B tokens，按平均每字约 3 token 估算）
- 用 shuffle buffer 做近似全局打散，最后输出到 mixed_corpus.jsonl

全流程流式处理：CCI3-HQ 用 datasets streaming，自建语料逐行读，
打散用固定大小的 shuffle buffer，不会一次性把全部语料读进内存。
"""

import argparse
import json
import logging
import random
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mix_datasets")


def iter_local_jsonl(path, text_field):
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


def iter_cci3_hq(text_field, hf_token=None):
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


def shuffle_buffer(source, buffer_size, rng):
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
    for item in buf:
        yield item


def take_until_chars(source, char_budget, source_name, max_docs=None):
    """从 source 取文档，直到累计字符数达到 char_budget（或文档数达到 max_docs）。"""
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
            logger.info("[%s] 已采样 %d 篇 / %d 亿字",
                        source_name, n_docs, used / 1e8)
    logger.info("[%s] 采样完成：%d 篇，共 %.3f 亿字",
                source_name, n_docs, used / 1e8)


def mix(args):
    rng = random.Random(args.seed)

    total_chars = int(args.total_chars)
    cci_budget = int(total_chars * args.cci_ratio)
    local_budget = total_chars - cci_budget
    logger.info("目标总字符数：%.2f 亿", total_chars / 1e8)
    logger.info("  CCI3-HQ   预算：%.2f 亿字 (%.0f%%)",
                cci_budget / 1e8, 100 * args.cci_ratio)
    logger.info("  自建语料  预算：%.2f 亿字 (%.0f%%)",
                local_budget / 1e8, 100 * (1 - args.cci_ratio))

    t0 = time.time()

    # CCI 来源：默认 HF streaming；--cci-local 指定时从本地 jsonl 读取
    # （用于离线 / Tiny 验证，避免触发数据下载）
    if args.cci_local:
        logger.info("CCI 来源：本地文件 %s（离线模式）", args.cci_local)
        cci_raw = iter_local_jsonl(args.cci_local, args.text_field)
    else:
        logger.info("CCI 来源：HuggingFace streaming BAAI/CCI3-HQ")
        cci_raw = iter_cci3_hq(args.text_field, args.hf_token)

    # 两路来源各自先按预算截断
    cci_stream = take_until_chars(
        cci_raw, cci_budget, "CCI3-HQ", args.max_docs,
    )
    local_stream = take_until_chars(
        iter_local_jsonl(args.local_corpus, args.text_field),
        local_budget, "自建语料", args.max_docs,
    )

    def tagged(stream, tag):
        for text in stream:
            yield {"text": text, "source": tag}

    # 交错合并两路（轮流取，避免一路在前一路在后导致打散不均）
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

    # shuffle buffer 打散后写出
    n_written = 0
    chars_written = 0
    src_counter = {"cci3-hq": 0, "local": 0}
    with open(args.output, "w", encoding="utf-8") as out:
        for obj in shuffle_buffer(merged, args.shuffle_buffer, rng):
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1
            chars_written += len(obj["text"])
            src_counter[obj["source"]] += 1
            if n_written % 50000 == 0:
                logger.info("已写出 %d 篇 / %.2f 亿字",
                            n_written, chars_written / 1e8)

    logger.info("=" * 60)
    logger.info("混合完成：%d 篇，共 %.3f 亿字，用时 %.1fs",
                n_written, chars_written / 1e8, time.time() - t0)
    logger.info("  来源分布：CCI3-HQ %d 篇 / 自建 %d 篇",
                src_counter["cci3-hq"], src_counter["local"])
    logger.info("  实际比例：CCI3-HQ %.1f%% / 自建 %.1f%%（按篇数）",
                100 * src_counter["cci3-hq"] / max(1, n_written),
                100 * src_counter["local"] / max(1, n_written))
    logger.info("输出文件：%s", args.output)


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
                   help="每个来源最多采样的文档数（Tiny 验证/调试用）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    mix(args)
