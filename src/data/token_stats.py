#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stream a jsonl corpus and report exact tokenizer token counts.

This is intended for cloud-side checks after mixing datasets: the mixed corpus
can stay on the cloud machine, while this script scans it once and writes a
small JSON report with token counts by source.
"""

import argparse
import json
import logging
import time
from collections import defaultdict

from dedup_utils import ensure_parent_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("token_stats")


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("第 %d 行 JSON 解析失败，已跳过", line_no)


def compute(args):
    import sentencepiece as spm

    ensure_parent_dir(args.output)

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    totals = {
        "documents": 0,
        "chars": 0,
        "tokens": 0,
    }
    by_source = defaultdict(lambda: {"documents": 0, "chars": 0, "tokens": 0})
    t0 = time.time()

    for obj in iter_jsonl(args.input):
        if args.max_docs and totals["documents"] >= args.max_docs:
            logger.info("已达 --max-docs=%d 上限，停止统计", args.max_docs)
            break

        text = obj.get(args.text_field, "")
        if not isinstance(text, str) or not text:
            continue
        source = obj.get(args.source_field, "unknown")
        if not isinstance(source, str) or not source:
            source = "unknown"

        n_chars = len(text)
        n_tokens = len(sp.encode(text, out_type=int))

        totals["documents"] += 1
        totals["chars"] += n_chars
        totals["tokens"] += n_tokens

        bucket = by_source[source]
        bucket["documents"] += 1
        bucket["chars"] += n_chars
        bucket["tokens"] += n_tokens

        if totals["documents"] % args.log_interval == 0:
            logger.info("已统计 %d 篇 / %.2f M tokens / %.1fs",
                        totals["documents"], totals["tokens"] / 1e6,
                        time.time() - t0)

    def enrich(bucket):
        docs = max(1, bucket["documents"])
        chars = max(1, bucket["chars"])
        return {
            **bucket,
            "tokens_per_char": round(bucket["tokens"] / chars, 6),
            "avg_doc_chars": round(bucket["chars"] / docs, 1),
            "avg_doc_tokens": round(bucket["tokens"] / docs, 1),
            "token_ratio": round(bucket["tokens"] / max(1, totals["tokens"]), 6),
            "char_ratio": round(bucket["chars"] / max(1, totals["chars"]), 6),
        }

    report = {
        "input": args.input,
        "tokenizer": args.tokenizer,
        "elapsed_seconds": round(time.time() - t0, 1),
        "total": enrich(totals),
        "by_source": {
            source: enrich(dict(bucket))
            for source, bucket in sorted(by_source.items())
        },
    }

    tmp_output = args.output + ".tmp"
    with open(tmp_output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    import os

    os.replace(tmp_output, args.output)

    logger.info("=" * 60)
    logger.info("统计完成：%d 篇 / %.2f M chars / %.2f M tokens",
                totals["documents"], totals["chars"] / 1e6,
                totals["tokens"] / 1e6)
    for source, bucket in report["by_source"].items():
        logger.info("%s: %.2f M tokens (%.2f%%), %.4f tokens/char",
                    source, bucket["tokens"] / 1e6,
                    100 * bucket["token_ratio"], bucket["tokens_per_char"])
    logger.info("报告已写入：%s", args.output)
    return report


def build_parser():
    p = argparse.ArgumentParser(
        description="流式统计 mixed_corpus 的真实 tokenizer token 数",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="输入 jsonl 路径")
    p.add_argument("--tokenizer", required=True, help="SentencePiece .model 路径")
    p.add_argument("--output", default="results/token_stats.json",
                   help="统计报告输出路径")
    p.add_argument("--text-field", default="text", help="文本字段名")
    p.add_argument("--source-field", default="source", help="来源字段名")
    p.add_argument("--max-docs", type=int, default=None,
                   help="最多统计的文档数（调试用，默认全部）")
    p.add_argument("--log-interval", type=int, default=50000,
                   help="进度日志间隔文档数")
    return p


if __name__ == "__main__":
    compute(build_parser().parse_args())
