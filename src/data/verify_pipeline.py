#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端验证脚本：用极小数据（默认 100 条样本）串行跑完 Task 1→2→3→4。

目的：在投入 2B token 大规模处理之前，先确认整条 pipeline 的数据格式、
接口契约从输入到 DataLoader 输出都对得上，避免跑到 Task 3 才发现
Task 1 的输出格式不匹配。

本脚本**不下载任何数据**：自动生成合成中文语料（含若干脏数据），
其中混入过短、纯数字、重复文档，用来验证 Task 1 的过滤器确实生效。
CCI3-HQ 一侧也用合成的本地文件代替（通过 mix_datasets 的 --cci-local）。

最终打印：
  - 清洗前 / 清洗后文档数
  - tokenizer 的平均每字 token 数
  - DataLoader 输出的 input_ids / labels shape 与一个样本的解码结果
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import clean_corpus  # noqa: E402
import mix_datasets  # noqa: E402
import pretrain_dataset  # noqa: E402
import train_tokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_pipeline")

# 常用中文词汇池：用「真实词汇」而非随机汉字拼句子，
# 这样 langdetect/fasttext 才能稳定识别为中文（随机汉字会被误判成日文等）。
_WORD_POOL = (
    "我们 他们 今天 明天 昨天 现在 时间 工作 学习 研究 数据 模型 训练 中国 城市 "
    "国家 社会 经济 发展 技术 科学 教育 文化 历史 世界 人类 自然 环境 生活 健康 "
    "问题 方法 结果 分析 系统 网络 信息 知识 语言 文字 内容 重要 因为 所以 但是 "
    "如果 这个 那个 一个 这样 那样 可以 应该 需要 已经 正在 开始 完成 提高 增加 "
    "减少 改变 影响 关系 能力 水平 质量 效果 过程 方面 部分 整体 各种 不同 相同 "
    "继续 实现 解决 处理 计算 设计 建立 使用 通过 进行 包括 提供 支持 关注 发现 "
    "公司 团队 项目 计划 目标 任务 管理 服务 产品 市场 用户 客户 价值 成本 收入 "
    "学生 老师 学校 大学 课程 考试 成绩 朋友 家庭 父母 孩子 工程 软件 程序 算法"
).split()


def gen_sentence(rng, min_words=8, max_words=18):
    n = rng.randint(min_words, max_words)
    return "".join(rng.choice(_WORD_POOL) for _ in range(n)) + "。"


def gen_doc(rng, n_sent=None):
    # 用较长文档（语言识别在长文本上更稳定，减少合成数据的误判）
    n = n_sent or rng.randint(8, 14)
    return "\n".join(gen_sentence(rng) for _ in range(n))


def write_jsonl(path, docs):
    with open(path, "w", encoding="utf-8") as f:
        for text in docs:
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")


def synth_local_corpus(rng, num_good):
    """生成自建语料的原始输入，混入脏数据以验证每个过滤器都生效。"""
    docs = [gen_doc(rng) for _ in range(num_good)]
    dirty = 0

    # 5 篇过短文档 -> 触发长度过滤
    for _ in range(5):
        docs.append("太短了。")
        dirty += 1
    # 4 篇与首篇完全相同 -> 触发 SHA-256 精确去重（新增步骤 4.5）
    for _ in range(4):
        docs.append(docs[0])
        dirty += 1
    # 2 篇与首篇近似相同（仅末尾替换一个词，3-gram Jaccard >> 0.85）
    # -> 通过 SHA-256（不完全相同），触发 MinHash 近似去重
    for i in range(2):
        replacement = _WORD_POOL[i % len(_WORD_POOL)]
        near_dup = docs[0][: -len(replacement)] + replacement
        docs.append(near_dup)
        dirty += 1
    # 3 篇纯数字 -> 语言过滤（chinese_prob 极低）兜底
    for _ in range(3):
        docs.append("1234567890" * 12)
        dirty += 1

    rng.shuffle(docs)
    return docs, dirty


def run(args):
    workdir = os.path.abspath(args.workdir)
    if os.path.exists(workdir):
        shutil.rmtree(workdir)
    os.makedirs(workdir)
    logger.info("验证工作目录：%s", workdir)

    rng = random.Random(args.seed)

    # ---------------- 准备合成数据 ----------------
    raw_local = os.path.join(workdir, "raw_local.jsonl")
    fake_cci = os.path.join(workdir, "fake_cci.jsonl")

    local_docs, n_dirty = synth_local_corpus(rng, args.num_docs)
    write_jsonl(raw_local, local_docs)
    write_jsonl(fake_cci, [gen_doc(rng) for _ in range(args.num_docs)])
    n_raw = len(local_docs)
    logger.info("已生成合成语料：自建原始 %d 篇（含 %d 篇脏数据）+ 伪 CCI %d 篇",
                n_raw, n_dirty, args.num_docs)

    # ---------------- Task 1：清洗 ----------------
    logger.info("=" * 60)
    logger.info("Task 1：清洗自建语料")
    clean_out = os.path.join(workdir, "local_clean.jsonl")
    clean_report = os.path.join(workdir, "clean_report.json")
    clean_args = clean_corpus.build_parser().parse_args([
        "--input", raw_local,
        "--output", clean_out,
        "--report", clean_report,
        "--max-docs", str(n_raw),
    ])
    clean_corpus.clean(clean_args)
    with open(clean_report, encoding="utf-8") as f:
        report = json.load(f)
    n_clean = report["counts"]["kept"]

    # ---------------- Task 2：混合采样 ----------------
    logger.info("=" * 60)
    logger.info("Task 2：混合采样（CCI 走 --cci-local，不下载）")
    mixed = os.path.join(workdir, "mixed_corpus.jsonl")
    mix_args = mix_datasets.build_parser().parse_args([
        "--local-corpus", clean_out,
        "--cci-local", fake_cci,
        "--output", mixed,
        "--total-chars", "1e9",      # 远大于合成数据量 -> 全量纳入
        "--cci-ratio", "0.70",
        "--shuffle-buffer", "256",
        "--no-cross-dedup",          # 验证脚本不测去重：合成语料词汇池小、
                                     # 3-gram 重叠高，会随机丢弃 CCI 文档导致断言不稳定
    ])
    mix_datasets.mix(mix_args)
    n_mixed = sum(1 for _ in open(mixed, encoding="utf-8"))

    # ---------------- Task 3：训练 tokenizer ----------------
    logger.info("=" * 60)
    logger.info("Task 3：训练 SentencePiece BPE（小词表，验证用）")
    model_prefix = os.path.join(workdir, "verify_bpe")
    tok_args = train_tokenizer.build_parser().parse_args([
        "--input", mixed,
        "--model-prefix", model_prefix,
        "--vocab-size", str(args.verify_vocab_size),
        "--input-sentence-size", "0",
        "--analyze-sample-lines", "100000",
    ])
    train_tokenizer.train(tok_args)
    with open(f"{model_prefix}_stats.json", encoding="utf-8") as f:
        tok_stats = json.load(f)
    tokens_per_char = tok_stats["tokens_per_char"]

    # ---------------- Task 4：构建预训练 Dataset ----------------
    logger.info("=" * 60)
    logger.info("Task 4：构建预训练 Dataset 并加载自检")
    pt_path = os.path.join(workdir, "pretrain_tokens.pt")
    ds_args = pretrain_dataset.build_parser().parse_args([
        "--mode", "build",
        "--input", mixed,
        "--tokenizer", f"{model_prefix}.model",
        "--output", pt_path,
        "--context-length", str(args.verify_ctx),
    ])
    pretrain_dataset.build(ds_args)

    # 用 PyTorch DataLoader 取一个 batch，验证 shape 与解码
    import sentencepiece as spm
    import torch
    from torch.utils.data import DataLoader

    ds = pretrain_dataset.PretrainDataset(pt_path, args.verify_ctx)
    loader = DataLoader(ds, batch_size=min(4, max(1, len(ds))), shuffle=False)
    input_ids, labels = next(iter(loader))

    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    sample_ids = input_ids[0].tolist()
    decoded = sp.decode(sample_ids)
    shift_ok = bool((input_ids[0][1:] == labels[0][:-1]).all())

    # ---------------- 汇总报告 ----------------
    print("\n" + "=" * 64)
    print("           端到端验证报告 (Task 1 -> 2 -> 3 -> 4)")
    print("=" * 64)
    print(f"[Task 1] 清洗前文档数        : {n_raw}")
    print(f"         清洗后文档数        : {n_clean}")
    print(f"         过滤明细            : 长度={report['counts']['drop_length']} "
          f"精确去重={report['counts']['drop_exact_dup']} "
          f"近似去重={report['counts']['drop_dedup']} "
          f"数字={report['counts']['drop_digit']} "
          f"标点={report['counts']['drop_punct']} "
          f"语言={report['counts']['drop_language']}")
    print(f"[Task 2] 混合后文档数        : {n_mixed} "
          f"（自建 {n_clean} + 伪CCI {args.num_docs}）")
    print(f"[Task 3] 词表大小            : {tok_stats['vocab_size']}")
    print(f"         平均每字 token 数   : {tokens_per_char:.4f} "
          f"（每 token 约 {tok_stats['chars_per_token']:.2f} 字）")
    print(f"         注：合成语料随机性强，该值不代表真实语料"
          f"（真实预期约 1.2~1.5）")
    print(f"[Task 4] 训练样本数          : {len(ds)}  (context={args.verify_ctx})")
    print(f"         input_ids shape     : {tuple(input_ids.shape)}  "
          f"dtype={input_ids.dtype}")
    print(f"         labels    shape     : {tuple(labels.shape)}  "
          f"dtype={labels.dtype}")
    print(f"         labels 右移校验     : {'通过' if shift_ok else '失败 !!!'}")
    print(f"         样本[0] 前若干 token: {sample_ids[:16]}")
    print(f"         样本[0] 解码文本    : {decoded[:60]}")
    print("=" * 64)

    # 基本断言：pipeline 契约
    assert n_clean > 0, "Task 1 清洗后无文档"
    assert report["counts"]["drop_length"] >= 5, "长度过滤未按预期生效"
    assert report["counts"]["drop_exact_dup"] >= 1, "SHA-256 精确去重未按预期生效"
    assert report["counts"]["drop_dedup"] >= 1, "MinHash 近似去重未按预期生效"
    assert n_mixed == n_clean + args.num_docs, "Task 2 文档数与预期不符"
    assert input_ids.shape[1] == args.verify_ctx, "input_ids 序列长度错误"
    assert input_ids.shape == labels.shape, "input_ids 与 labels shape 不一致"
    assert shift_ok, "labels 不是 input_ids 右移一位"
    print(">>> 全部契约校验通过，pipeline 端到端连通 <<<\n")

    if not args.keep:
        shutil.rmtree(workdir)
        logger.info("已清理临时目录（加 --keep 可保留中间产物）")
    else:
        logger.info("中间产物已保留在：%s", workdir)


def build_parser():
    p = argparse.ArgumentParser(
        description="端到端 Tiny 验证（Task 1→2→3→4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--workdir", default=".verify_tmp",
                   help="临时工作目录")
    p.add_argument("--num-docs", type=int, default=100,
                   help="每个来源的合成文档数")
    p.add_argument("--verify-vocab-size", type=int, default=600,
                   help="验证用的小词表大小（真实训练为 20000；byte_fallback=True 时"
                        "最小值约 458，故 600 是安全下限）")
    p.add_argument("--verify-ctx", type=int, default=64,
                   help="验证用的小 context_length（真实训练为 1024）")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--keep", action="store_true",
                   help="保留临时目录与中间产物")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    try:
        run(args)
    except AssertionError as e:
        logger.error("验证失败：%s", e)
        sys.exit(1)
