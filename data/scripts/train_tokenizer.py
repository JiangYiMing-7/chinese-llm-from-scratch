#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 3: 训练 SentencePiece BPE Tokenizer。

- 在 mixed_corpus.jsonl 上训练
- vocab_size=20000, model_type=bpe, character_coverage=0.9995
- 设置 pad / unk / bos / eos 四个基础 special token（id 0-3）
- 额外预留 SFT 专用 token（id 4-7）：
    <|endoftext|>  预训练文档边界 / 与 nanoGPT 保持一致
    <|system|>     SFT 系统提示起始
    <|user|>       SFT 用户轮次起始
    <|assistant|>  SFT 助手轮次起始
- 训练完成后输出统计：
    * 平均每字 token 数（在抽样语料上估计）
    * 词表中单字 / 双字 / 多字 token 的数量分布

默认训练语料通过流式迭代器喂给 SentencePieceTrainer，不落地大文本文件。
大规模训练时也支持先把 JSONL 的 text 字段抽样落地为纯文本文件，再让
SentencePiece 直接读取该 txt；这样避免 Python generator 在训练阶段反复
跨 Python/C++ 边界喂数据，速度和稳定性更好。
"""

import argparse
import json
import logging
import os
import re
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tokenizer")


# SentencePiece 默认 max_sentence_length=4192 字节，超过的句子会被「整条丢弃」
# （日志里的 "Skipped N too long sentences"）。中文 1 字 ≈ 3 字节，故用
# 1200 字 ≈ 3600 字节作为安全切块上限，留出余量。
_MAX_SEG_CHARS = 1200
# 优先在句末标点处断句，避免把词组从中间切断
_SENT_END_RE = re.compile(r"[。！？!?；;\n]")


def _split_long_segment(seg, max_chars=_MAX_SEG_CHARS):
    """把过长文本切成 <= max_chars 的块，尽量在句末标点处断开。

    SentencePiece 训练时会丢弃超过 max_sentence_length(4192 字节) 的句子；
    中文维基百科约 77% 的文档超过该限制，若不切块会损失大量训练语料。
    """
    if len(seg) <= max_chars:
        yield seg
        return
    start = 0
    n = len(seg)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            # 在 [start, end] 内找最后一个句末标点，作为更自然的断点
            window = seg[start:end]
            matches = list(_SENT_END_RE.finditer(window))
            if matches:
                end = start + matches[-1].end()
        chunk = seg[start:end].strip()
        if chunk:
            yield chunk
        start = end


def jsonl_sentence_iterator(path, text_field, max_lines=None):
    """流式产出训练句子。

    关键点：这里只取出 jsonl 每行的 text_field 字段，喂给 SentencePiece 的
    是「纯文本句子」，绝不会把整行 JSON（含 {} "" 等）喂进训练器，因此不会
    训练出带 JSON 符号的奇怪词表。SentencePieceTrainer 原生支持
    sentence_iterator，无需先落地一个巨大的 .txt 文件（2B tokens 的纯文本
    约 18GB），流式迭代更省磁盘。

    长文档处理：先按换行切段，再把仍超长的段切成 <= _MAX_SEG_CHARS 的块，
    确保不会触发 SentencePiece 的「too long sentence」丢弃。
    """
    n = 0
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
            if not isinstance(text, str) or not text:
                continue
            # 按换行切成短句，再对超长段做二次切块；空段跳过
            for seg in text.split("\n"):
                seg = seg.strip()
                if not seg:
                    continue
                for chunk in _split_long_segment(seg):
                    yield chunk
            n += 1
            if max_lines and n >= max_lines:
                return


def prepare_train_text(args):
    """把 JSONL 的 text 字段预处理成 SentencePiece 可直接读取的纯文本文件。

    输出的每一行都是一条已切短的训练句子/片段，不包含 JSON 字段名与结构符号。
    当 args.input_sentence_size > 0 时，最多写出对应数量的句子；否则写出全部。
    """
    if not args.train_text:
        raise ValueError("--train-text is required")

    out_path = os.path.abspath(args.train_text)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    max_sentences = args.input_sentence_size if args.input_sentence_size > 0 else None
    n_sentences = 0
    n_chars = 0
    t0 = time.time()
    logger.info("准备 SentencePiece 纯文本训练样本：%s", args.train_text)
    logger.info("  source=%s, max_sentences=%s",
                args.input, max_sentences if max_sentences is not None else "all")

    with open(out_path, "w", encoding="utf-8") as out:
        for sent in jsonl_sentence_iterator(
            args.input, args.text_field, args.max_train_lines
        ):
            sent = sent.replace("\n", " ").strip()
            if not sent:
                continue
            out.write(sent + "\n")
            n_sentences += 1
            n_chars += len(sent)
            if n_sentences % 500000 == 0:
                logger.info("  已写出 %d 句 / %.3f 亿字",
                            n_sentences, n_chars / 1e8)
            if max_sentences is not None and n_sentences >= max_sentences:
                break

    logger.info("训练样本准备完成：%d 句 / %.3f 亿字，用时 %.1fs",
                n_sentences, n_chars / 1e8, time.time() - t0)
    logger.info("训练样本文件：%s", args.train_text)


def train(args):
    import sentencepiece as spm

    os.makedirs(os.path.dirname(os.path.abspath(args.model_prefix)) or ".",
                exist_ok=True)

    spm_train_kwargs = {}
    if args.train_text:
        if args.rebuild_train_text or not os.path.exists(args.train_text):
            prepare_train_text(args)
        else:
            logger.info("复用已有 SentencePiece 纯文本训练样本：%s", args.train_text)
        spm_train_kwargs["input"] = args.train_text
        # train_text 已经按 input_sentence_size 抽样/截断，这里让 SentencePiece
        # 使用文件中的全部句子，避免再次抽样带来额外内存与时间开销。
        spm_input_sentence_size = 0
    else:
        spm_train_kwargs["sentence_iterator"] = jsonl_sentence_iterator(
            args.input, args.text_field, args.max_train_lines
        )
        spm_input_sentence_size = args.input_sentence_size

    t0 = time.time()
    logger.info("开始训练 SentencePiece BPE tokenizer ...")
    logger.info("  vocab_size=%d, character_coverage=%.4f",
                args.vocab_size, args.character_coverage)

    # id 0-3：基础 special tokens（固定顺序）
    # id 4-7：SFT / 文档边界专用 token（user_defined_symbols 按顺序分配 id）
    # <|endoftext|> 对齐 nanoGPT 约定，作为预训练文档间分隔符
    #
    # byte_fallback=True（关键）：词表里加入 256 个字节 token <0x00>..<0xFF>。
    # character_coverage=0.9995 之外的 0.05% 生僻字 / 日文假名 / emoji 不再
    # 变成 <unk>，而是拆成 UTF-8 字节序列——解码时可无损还原，往返一致性
    # 达到 ~100%。这是 LLaMA / Qwen / Mistral 的标准配置。
    spm.SentencePieceTrainer.train(
        **spm_train_kwargs,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        byte_fallback=True,
        pad_id=0, pad_piece="<pad>",
        unk_id=1, unk_piece="<unk>",
        bos_id=2, bos_piece="<bos>",
        eos_id=3, eos_piece="<eos>",
        user_defined_symbols=[
            "<|endoftext|>",   # id=4  预训练文档边界
            "<|system|>",      # id=5  SFT 系统提示
            "<|user|>",        # id=6  SFT 用户轮次
            "<|assistant|>",   # id=7  SFT 助手轮次
        ],
        input_sentence_size=spm_input_sentence_size,
        shuffle_input_sentence=True,
        num_threads=args.num_threads,
        train_extremely_large_corpus=True,
    )
    logger.info("训练完成，用时 %.1fs，模型：%s.model",
                time.time() - t0, args.model_prefix)

    analyze(args)


def analyze(args):
    """输出词表与编码效率统计。"""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(f"{args.model_prefix}.model")
    vocab_size = sp.get_piece_size()

    # 1) 词表中单字 / 双字 / 多字 token 分布
    # SentencePiece 用 "▁"(U+2581) 表示词首空格，统计时去掉
    dist = {"single": 0, "double": 0, "multi": 0, "special_or_meta": 0}
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        core = piece.replace("▁", "")
        if not core or piece.startswith("<") and piece.endswith(">"):
            dist["special_or_meta"] += 1
            continue
        length = len(core)
        if length == 1:
            dist["single"] += 1
        elif length == 2:
            dist["double"] += 1
        else:
            dist["multi"] += 1

    # 2) 在抽样语料上估计平均每字 token 数 + UNK 率 + 往返一致性
    #    UNK 率是 tokenizer 最关键的质量指标：character_coverage=0.9995 时
    #    理论上约 0.05% 的字符会落到 <unk>，模型无法学习这些字符。
    #    UNK 率显著偏高（如 > 0.1%）说明 character_coverage 需要调高。
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    roundtrip_ok = 0
    roundtrip_total = 0
    unk_id = sp.unk_id()
    for i, sent in enumerate(
        jsonl_sentence_iterator(args.input, args.text_field)
    ):
        total_chars += len(sent)
        ids = sp.encode(sent, out_type=int)
        total_tokens += len(ids)
        total_unk += sum(1 for t in ids if t == unk_id)
        # 往返检查：encode->decode 应能还原原文（被 UNK 吃掉的字符除外）
        if i < 2000:
            roundtrip_total += 1
            if sp.decode(ids) == sent:
                roundtrip_ok += 1
        if i + 1 >= args.analyze_sample_lines:
            break
    tokens_per_char = total_tokens / max(1, total_chars)
    unk_rate = total_unk / max(1, total_tokens)
    roundtrip_rate = roundtrip_ok / max(1, roundtrip_total)

    # 记录所有 special token 的实际 id，便于模型代码直接引用
    special_tokens = {}
    for piece in ("<pad>", "<unk>", "<bos>", "<eos>",
                  "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"):
        tid = sp.piece_to_id(piece)
        # piece_to_id 对不在词表里的 piece 返回 unk_id，但 <unk> 自身确实是 unk_id
        if piece == "<unk>" or tid != sp.unk_id():
            special_tokens[piece] = tid
        else:
            special_tokens[piece] = None  # 该 token 不在词表中

    stats = {
        "model": f"{args.model_prefix}.model",
        "vocab_size": vocab_size,
        "special_token_ids": special_tokens,
        "piece_length_distribution": dist,
        "piece_length_ratio": {
            k: round(v / vocab_size, 4) for k, v in dist.items()
        },
        "tokens_per_char": round(tokens_per_char, 4),
        "chars_per_token": round(1 / max(1e-9, tokens_per_char), 4),
        "unk_rate": round(unk_rate, 6),
        "unk_count": total_unk,
        "roundtrip_rate": round(roundtrip_rate, 4),
        "analyze_sample": {
            "chars": total_chars, "tokens": total_tokens,
        },
    }
    report_path = f"{args.model_prefix}_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("词表大小：%d", vocab_size)
    logger.info("token 长度分布：")
    logger.info("  单字 token : %6d (%.1f%%)",
                dist["single"], 100 * dist["single"] / vocab_size)
    logger.info("  双字 token : %6d (%.1f%%)",
                dist["double"], 100 * dist["double"] / vocab_size)
    logger.info("  多字 token : %6d (%.1f%%)",
                dist["multi"], 100 * dist["multi"] / vocab_size)
    logger.info("  特殊/元    : %6d (%.1f%%)",
                dist["special_or_meta"],
                100 * dist["special_or_meta"] / vocab_size)
    logger.info("平均每字 token 数：%.4f（即每 token 约 %.2f 字）",
                tokens_per_char, 1 / max(1e-9, tokens_per_char))
    logger.info("UNK 率：%.4f%%（%d / %d tokens）%s",
                100 * unk_rate, total_unk, total_tokens,
                "" if unk_rate < 0.001 else "  ⚠ 偏高，建议调高 character_coverage")
    logger.info("encode/decode 往返一致率：%.2f%%（抽查 %d 条）",
                100 * roundtrip_rate, roundtrip_total)
    logger.info("Special token IDs：")
    for piece, tid in special_tokens.items():
        logger.info("  %-18s  id = %s", piece, tid)
    logger.info("统计已写入：%s", report_path)


def build_parser():
    p = argparse.ArgumentParser(
        description="训练 SentencePiece BPE Tokenizer（Task 3）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", default="mixed_corpus.jsonl",
                   help="训练语料 jsonl 路径")
    p.add_argument("--text-field", default="text", help="文本字段名")
    p.add_argument("--model-prefix", default="data/tokenizer/zh_bpe_20k",
                   help="输出模型前缀（生成 .model 与 .vocab）")

    p.add_argument("--vocab-size", type=int, default=20000, help="词表大小")
    p.add_argument("--character-coverage", type=float, default=0.9995,
                   help="字符覆盖率")
    p.add_argument("--input-sentence-size", type=int, default=10_000_000,
                   help="参与训练的最大句子数（0 表示全部）")
    p.add_argument("--max-train-lines", type=int, default=None,
                   help="最多读取的 jsonl 文档数（调试用，默认全部）")
    p.add_argument("--num-threads", type=int, default=min(os.cpu_count() or 4, 16),
                   help="训练线程数")
    p.add_argument("--train-text", default=None,
                   help="可选：先把 JSONL 的 text 字段落地为纯文本训练文件，再用该 txt 训练")
    p.add_argument("--rebuild-train-text", action="store_true",
                   help="当 --train-text 已存在时也重新生成")
    p.add_argument("--prepare-train-text-only", action="store_true",
                   help="只生成 --train-text，不训练 tokenizer")

    p.add_argument("--analyze-sample-lines", type=int, default=100000,
                   help="估计平均每字 token 数时抽样的句子数")
    p.add_argument("--analyze-only", action="store_true",
                   help="跳过训练，只对已有模型做统计分析")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.prepare_train_text_only:
        prepare_train_text(args)
    elif args.analyze_only:
        analyze(args)
    else:
        train(args)
