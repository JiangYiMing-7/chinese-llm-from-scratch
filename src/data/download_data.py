#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地 Pilot / Debug 数据下载脚本。

支持两种来源：
  1. 中文维基百科（wikimedia/wikipedia zh，HuggingFace streaming）
     - Debug：约 10K 篇文章，~15M tokens
     - Pilot：约 80K 篇文章，~50M tokens
  2. CLUECorpus2020 新闻子集（news2016zh，HuggingFace）
     - 仅在 --stage pilot 时追加，补足到 ~100MB 文本

全程 streaming，不会把整个数据集下载到本地。
输出：data/raw/{stage}_wiki.jsonl 和（Pilot 时）data/raw/{stage}_clue.jsonl
输出格式与 clean_corpus.py 完全兼容（每行 {"text": "..."}）。

用法示例：
  python src/data/download_data.py --stage debug
  python src/data/download_data.py --stage pilot
  python src/data/download_data.py --stage pilot --wiki-only  # 只下维基
"""

import argparse
import json
import logging
import os
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_data")

# 各阶段目标篇数（维基百科；中文维基平均每篇约 600-1500 字）
STAGE_CONFIG = {
    "debug": {
        "wiki_articles": 12_000,    # ~15M tokens
        "clue_articles": 0,
        "desc": "Debug（~15M tokens，用于过拟合验证）",
    },
    "pilot": {
        "wiki_articles": 80_000,    # ~50M tokens（维基贡献约 60-70%）
        "clue_articles": 30_000,    # 补充约 20-30M tokens
        "desc": "Pilot（~50M tokens，用于完整训练流程验证）",
    },
}


def stream_wikipedia_zh(max_articles, hf_token=None):
    """流式读取中文维基百科，yield 文章文本。"""
    from datasets import load_dataset

    logger.info("加载 wikimedia/wikipedia 20231101.zh（streaming）...")
    ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.zh",
        split="train",
        streaming=True,
        token=hf_token,
    )
    n = 0
    for ex in ds:
        text = (ex.get("text") or "").strip()
        if len(text) < 100:   # 跳过极短重定向页
            continue
        yield text
        n += 1
        if n >= max_articles:
            break
    logger.info("维基百科：共读取 %d 篇", n)


def stream_clue_news(max_articles, hf_token=None):
    """流式读取 CLUECorpus2020 新闻子集。"""
    from datasets import load_dataset

    logger.info("加载 CLUECorpus2020 news2016zh（streaming）...")
    try:
        ds = load_dataset(
            "clue",
            "news2016zh",
            split="train",
            streaming=True,
            token=hf_token,
        )
    except Exception as e:
        logger.warning("CLUECorpus 加载失败（%s）；跳过该来源", e)
        return
    n = 0
    for ex in ds:
        # CLUE news2016zh 字段为 content 或 title+content
        text = (ex.get("content") or ex.get("text") or "").strip()
        if len(text) < 100:
            continue
        yield text
        n += 1
        if n >= max_articles:
            break
    logger.info("CLUECorpus 新闻：共读取 %d 篇", n)


def save_jsonl(source, output_path, source_name):
    """把 source 迭代器写成 jsonl，每行 {"text": "..."}，打印进度。"""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    n, total_chars = 0, 0
    t0 = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        for text in source:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
            total_chars += len(text)
            if n % 5000 == 0:
                logger.info("[%s] %d 篇 / %.1f MB / %.1fs",
                            source_name, n, total_chars / 1e6, time.time() - t0)
    logger.info("[%s] 完成：%d 篇，%.2f MB 文本，用时 %.1fs",
                source_name, n, total_chars / 1e6, time.time() - t0)
    return n, total_chars


def download(args):
    cfg = STAGE_CONFIG[args.stage]
    logger.info("下载阶段：%s", cfg["desc"])
    out_dir = os.path.join(args.output_dir, args.stage)
    os.makedirs(out_dir, exist_ok=True)

    total_chars = 0

    # --max-articles 覆盖阶段默认值（本地快速验证用）
    wiki_limit = args.max_articles if args.max_articles else cfg["wiki_articles"]
    clue_limit = min(args.max_articles, cfg["clue_articles"]) \
        if args.max_articles else cfg["clue_articles"]

    # 1) 中文维基百科
    wiki_path = os.path.join(out_dir, "wiki.jsonl")
    if os.path.exists(wiki_path) and not args.force:
        logger.info("已存在 %s，跳过（--force 可重新下载）", wiki_path)
    else:
        _, chars = save_jsonl(
            stream_wikipedia_zh(wiki_limit, args.hf_token),
            wiki_path,
            "wiki",
        )
        total_chars += chars

    # 2) CLUECorpus 新闻（仅 Pilot，且未指定 --wiki-only）
    if clue_limit > 0 and not args.wiki_only:
        clue_path = os.path.join(out_dir, "clue_news.jsonl")
        if os.path.exists(clue_path) and not args.force:
            logger.info("已存在 %s，跳过（--force 可重新下载）", clue_path)
        else:
            _, chars = save_jsonl(
                stream_clue_news(clue_limit, args.hf_token),
                clue_path,
                "clue_news",
            )
            total_chars += chars

    logger.info("=" * 60)
    logger.info("下载完成，输出目录：%s", out_dir)
    logger.info("总文本量：%.2f MB", total_chars / 1e6)
    logger.info("")
    logger.info("下一步 —— 清洗（把 wiki.jsonl 和 clue_news.jsonl 合并或分别清洗）：")
    logger.info("  cat %s/*.jsonl > data/raw/%s_raw.jsonl", out_dir, args.stage)
    logger.info("  python src/data/clean_corpus.py \\")
    logger.info("    --input data/raw/%s_raw.jsonl \\", args.stage)
    logger.info("    --output data/processed/%s_clean.jsonl \\", args.stage)
    logger.info("    --report results/%s_clean_report.json", args.stage)


def build_parser():
    p = argparse.ArgumentParser(
        description="下载本地 Pilot/Debug 数据（中文维基 + CLUECorpus）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--stage", choices=["debug", "pilot"], default="pilot",
                   help="下载阶段：debug（~15M tokens）或 pilot（~50M tokens）")
    p.add_argument("--output-dir", default="data/raw",
                   help="输出根目录（会在其下创建 debug/ 或 pilot/ 子目录）")
    p.add_argument("--wiki-only", action="store_true",
                   help="只下载维基百科，跳过 CLUECorpus")
    p.add_argument("--force", action="store_true",
                   help="强制重新下载，覆盖已有文件")
    p.add_argument("--hf-token", default=None,
                   help="HuggingFace token（一般不需要，公开数据集无需登录）")
    p.add_argument("--max-articles", type=int, default=None,
                   help="覆盖阶段默认篇数上限（本地快速验证用，如 --max-articles 300）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    download(args)
