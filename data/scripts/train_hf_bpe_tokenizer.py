#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train a HuggingFace tokenizers BPE model using the Rust/Rayon trainer.

This is the multi-core alternative to SentencePiece BPE. It first writes a
plain-text training file from JSONL, then trains from that file inside the
tokenizers Rust implementation. Set RAYON_NUM_THREADS/--num-threads to control
CPU parallelism.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_tokenizer import jsonl_sentence_iterator  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_hf_bpe_tokenizer")

BASE_SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<|endoftext|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
]
BYTE_FALLBACK_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


def prepare_train_text(args) -> None:
    out_path = Path(args.train_text)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_sentences = args.input_sentence_size if args.input_sentence_size > 0 else None
    n_sentences = 0
    n_chars = 0
    t0 = time.time()
    logger.info("准备 HF BPE 纯文本训练样本：%s", out_path)
    logger.info("  source=%s, max_sentences=%s",
                args.input, max_sentences if max_sentences is not None else "all")

    with out_path.open("w", encoding="utf-8") as out:
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
    logger.info("训练样本文件：%s", out_path)


def _make_tokenizer(args):
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

    try:
        model = models.BPE(unk_token="<unk>", byte_fallback=args.byte_fallback)
    except TypeError:
        if args.byte_fallback:
            logger.warning("当前 tokenizers.models.BPE 不支持 byte_fallback，回退为 unk fallback")
        model = models.BPE(unk_token="<unk>")

    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.NFKC()

    components = []
    if hasattr(pre_tokenizers, "UnicodeScripts"):
        components.append(pre_tokenizers.UnicodeScripts())
    if hasattr(pre_tokenizers, "WhitespaceSplit"):
        components.append(pre_tokenizers.WhitespaceSplit())
    elif hasattr(pre_tokenizers, "Whitespace"):
        components.append(pre_tokenizers.Whitespace())
    if len(components) == 1:
        tokenizer.pre_tokenizer = components[0]
    elif components:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(components)

    if args.byte_fallback and hasattr(decoders, "ByteFallback"):
        tokenizer.decoder = decoders.ByteFallback()
    return tokenizer


def train(args) -> None:
    if args.num_threads > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.num_threads)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    try:
        from tokenizers import trainers
    except ImportError as exc:
        raise SystemExit(
            "ERROR: missing dependency `tokenizers`. Install in the active env with:\n"
            "  python -m pip install tokenizers"
        ) from exc

    if args.rebuild_train_text or not Path(args.train_text).exists():
        prepare_train_text(args)
    else:
        logger.info("复用已有 HF BPE 纯文本训练样本：%s", args.train_text)

    tokenizer = _make_tokenizer(args)
    special_tokens = BASE_SPECIAL_TOKENS[:]
    if args.byte_fallback:
        special_tokens.extend(BYTE_FALLBACK_TOKENS)

    trainer_kwargs = {
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "special_tokens": special_tokens,
        "show_progress": True,
    }
    if args.max_token_length:
        trainer_kwargs["max_token_length"] = args.max_token_length
    try:
        trainer = trainers.BpeTrainer(**trainer_kwargs)
    except TypeError:
        trainer_kwargs.pop("max_token_length", None)
        trainer = trainers.BpeTrainer(**trainer_kwargs)

    t0 = time.time()
    logger.info("开始训练 HF/Rust BPE tokenizer ...")
    logger.info("  vocab_size=%d, min_frequency=%d, rayon_threads=%s",
                args.vocab_size, args.min_frequency,
                os.environ.get("RAYON_NUM_THREADS", "default"))
    tokenizer.train([args.train_text], trainer=trainer)

    out_json = Path(args.model_prefix + ".json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_json))
    elapsed = time.time() - t0
    logger.info("训练完成，用时 %.1fs，模型：%s", elapsed, out_json)

    write_vocab_and_stats(tokenizer, args, elapsed)


def write_vocab_and_stats(tokenizer, args, elapsed: float) -> None:
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    id_to_token = {idx: token for token, idx in vocab.items()}
    vocab_path = Path(args.model_prefix + ".vocab.json")
    with vocab_path.open("w", encoding="utf-8") as f:
        json.dump(
            {str(i): id_to_token[i] for i in range(len(id_to_token))},
            f,
            ensure_ascii=False,
            indent=2,
        )

    special_ids = {tok: tokenizer.token_to_id(tok) for tok in BASE_SPECIAL_TOKENS}
    unk_id = tokenizer.token_to_id("<unk>")
    total_chars = 0
    total_tokens = 0
    total_unk = 0
    roundtrip_ok = 0
    roundtrip_total = 0
    for i, sent in enumerate(jsonl_sentence_iterator(args.input, args.text_field)):
        ids = tokenizer.encode(sent).ids
        total_chars += len(sent)
        total_tokens += len(ids)
        if unk_id is not None:
            total_unk += sum(1 for tid in ids if tid == unk_id)
        if i < 2000:
            roundtrip_total += 1
            try:
                if tokenizer.decode(ids) == sent:
                    roundtrip_ok += 1
            except Exception:
                pass
        if i + 1 >= args.analyze_sample_lines:
            break

    stats = {
        "model": args.model_prefix + ".json",
        "backend": "hf_tokenizers_bpe",
        "vocab_size": tokenizer.get_vocab_size(with_added_tokens=True),
        "special_token_ids": special_ids,
        "byte_fallback_requested": args.byte_fallback,
        "tokens_per_char": round(total_tokens / max(1, total_chars), 6),
        "chars_per_token": round(total_chars / max(1, total_tokens), 6),
        "unk_rate": round(total_unk / max(1, total_tokens), 8),
        "unk_count": total_unk,
        "roundtrip_rate": round(roundtrip_ok / max(1, roundtrip_total), 4),
        "analyze_sample": {
            "chars": total_chars,
            "tokens": total_tokens,
        },
        "elapsed_seconds": round(elapsed, 1),
        "num_threads": args.num_threads,
        "input_sentence_size": args.input_sentence_size,
        "train_text": args.train_text,
    }
    stats_path = Path(args.model_prefix + "_stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info("词表大小：%d", stats["vocab_size"])
    logger.info("平均每字 token 数：%.4f", stats["tokens_per_char"])
    logger.info("UNK 率：%.6f%%（%d / %d tokens）",
                100 * stats["unk_rate"], total_unk, total_tokens)
    logger.info("统计已写入：%s", stats_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train a multi-core HuggingFace tokenizers BPE tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--vocab-size", type=int, default=50000)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--input-sentence-size", type=int, default=2_000_000)
    parser.add_argument("--max-train-lines", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=64)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-token-length", type=int, default=16)
    parser.add_argument("--analyze-sample-lines", type=int, default=100000)
    parser.add_argument("--rebuild-train-text", action="store_true")
    parser.add_argument("--byte-fallback", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
