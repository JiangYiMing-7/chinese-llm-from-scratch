#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parallel builder for pretraining token .bin files.

The regular pretrain_dataset.py builder is intentionally simple and streams in a
single process. This variant splits a large JSONL file by byte ranges, encodes
the ranges in multiple worker processes, then merges temporary token shards in
corpus order and performs the fixed train/val split.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dedup_utils import ensure_distinct_paths, ensure_parent_dir  # noqa: E402
from tokenizer_io import load_tokenizer  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pretrain_dataset_parallel")


@dataclass
class PartResult:
    index: int
    path: str
    docs: int
    tokens: int
    bad_json: int
    empty_text: int
    elapsed: float


def parse_size(text: str) -> int:
    raw = str(text).strip().lower()
    units = {
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
    }
    for suffix, multiplier in sorted(units.items(), key=lambda item: -len(item[0])):
        if raw.endswith(suffix):
            return int(float(raw[: -len(suffix)]) * multiplier)
    return int(raw)


def make_ranges(path: Path, chunk_bytes: int) -> list[tuple[int, int]]:
    size = path.stat().st_size
    ranges = []
    with path.open("rb") as f:
        start = 0
        while start < size:
            target = min(size, start + chunk_bytes)
            if target < size:
                f.seek(target)
                f.readline()
                end = f.tell()
            else:
                end = size
            if end > start:
                ranges.append((start, end))
            start = end
    return ranges


def worker_encode(task: dict) -> dict:
    started = time.time()
    tokenizer = load_tokenizer(task["tokenizer"])
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    unk_id = tokenizer.unk_id()
    endoftext_id = tokenizer.piece_to_id("<|endoftext|>")
    if endoftext_id is None or endoftext_id == unk_id:
        endoftext_id = eos_id

    dtype_np = np.dtype(task["dtype"])
    out_path = Path(task["out_path"])
    docs = 0
    tokens = 0
    bad_json = 0
    empty_text = 0
    text_field = task["text_field"]

    with Path(task["input"]).open("rb") as src, out_path.open("wb") as out:
        src.seek(task["start"])
        while src.tell() < task["end"]:
            line = src.readline()
            if not line:
                break
            if src.tell() > task["end"] and task["end"] != task["file_size"]:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError:
                bad_json += 1
                continue
            text = obj.get(text_field, "")
            if not isinstance(text, str) or not text:
                empty_text += 1
                continue
            ids = tokenizer.encode(text)
            if task["add_bos"]:
                ids = [bos_id] + ids
            ids.append(endoftext_id)
            np.asarray(ids, dtype=dtype_np).tofile(out)
            docs += 1
            tokens += len(ids)

    return {
        "index": task["index"],
        "path": str(out_path),
        "docs": docs,
        "tokens": tokens,
        "bad_json": bad_json,
        "empty_text": empty_text,
        "elapsed": round(time.time() - started, 3),
    }


def copy_tokens(src_path: Path, dst, offset_tokens: int, n_tokens: int,
                itemsize: int) -> None:
    if n_tokens <= 0:
        return
    remaining = n_tokens * itemsize
    with src_path.open("rb") as src:
        src.seek(offset_tokens * itemsize)
        while remaining:
            chunk = src.read(min(16 * 1024 * 1024, remaining))
            if not chunk:
                break
            dst.write(chunk)
            remaining -= len(chunk)
    if remaining:
        raise IOError(f"short read while copying {src_path}")


def write_meta(path: Path, dtype_np: np.dtype, args, n_docs: int, n_tokens: int,
               vocab_size: int, split: str, source_tokens: int) -> None:
    meta = {
        "token_ids_bin": str(path),
        "dtype": str(dtype_np),
        "context_length": args.context_length,
        "num_documents": n_docs,
        "total_tokens": int(n_tokens),
        "tokenizer": args.tokenizer,
        "vocab_size": vocab_size,
        "split": split,
        "source_total_tokens": int(source_tokens),
        "builder": "pretrain_dataset_parallel.py",
    }
    tmp = Path(str(path) + ".meta.json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, Path(str(path) + ".meta.json"))


def merge_parts(parts: list[PartResult], args, dtype_np: np.dtype,
                vocab_size: int) -> None:
    ensure_parent_dir(args.output)
    if args.val_output:
        ensure_parent_dir(args.val_output)
    total_tokens = sum(part.tokens for part in parts)
    total_docs = sum(part.docs for part in parts)

    if args.val_output:
        val_tokens = max(args.context_length + 1, int(total_tokens * args.val_ratio))
        train_tokens = total_tokens - val_tokens
    else:
        train_tokens = total_tokens
        val_tokens = 0

    if args.target_train_tokens and train_tokens < args.target_train_tokens:
        raise ValueError(
            f"train tokens {train_tokens} < target {args.target_train_tokens}"
        )

    itemsize = dtype_np.itemsize
    train_tmp = Path(args.output + ".tmp")
    val_tmp = Path(args.val_output + ".tmp") if args.val_output else None
    written_train = 0
    written_val = 0

    with train_tmp.open("wb") as train_out:
        val_out = val_tmp.open("wb") if val_tmp else None
        try:
            remaining_train = train_tokens
            for part in parts:
                part_path = Path(part.path)
                if remaining_train > 0:
                    to_train = min(part.tokens, remaining_train)
                    copy_tokens(part_path, train_out, 0, to_train, itemsize)
                    written_train += to_train
                    remaining_train -= to_train
                    rest = part.tokens - to_train
                    if rest and val_out:
                        copy_tokens(part_path, val_out, to_train, rest, itemsize)
                        written_val += rest
                elif val_out:
                    copy_tokens(part_path, val_out, 0, part.tokens, itemsize)
                    written_val += part.tokens
        finally:
            if val_out:
                val_out.close()

    os.replace(train_tmp, args.output)
    write_meta(Path(args.output), dtype_np, args, total_docs, written_train,
               vocab_size, "train", total_tokens)
    logger.info("已保存 train token 文件：%s", args.output)
    logger.info("train tokens=%d，按 context_length=%d 可得 %d 个样本",
                written_train, args.context_length,
                max(0, (written_train - 1) // args.context_length))

    if args.val_output:
        os.replace(val_tmp, args.val_output)
        write_meta(Path(args.val_output), dtype_np, args, total_docs, written_val,
                   vocab_size, "val", total_tokens)
        logger.info("已保存 val token 文件：%s", args.val_output)
        logger.info("val tokens=%d，按 context_length=%d 可得 %d 个样本",
                    written_val, args.context_length,
                    max(0, (written_val - 1) // args.context_length))


def build(args) -> None:
    ensure_distinct_paths(args.input, args.output, args.val_output)
    if args.val_output and not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be in (0, 1) when --val-output is set")
    if args.target_train_tokens and not args.val_output:
        raise ValueError("--target-train-tokens requires --val-output")

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = tokenizer.vocab_size()
    dtype_np = np.dtype(np.int16 if vocab_size <= 32767 else np.int32)
    input_path = Path(args.input)
    chunk_bytes = parse_size(args.chunk_bytes)
    ranges = make_ranges(input_path, chunk_bytes)
    if not ranges:
        raise ValueError(f"no input ranges for {input_path}")

    tmp_dir = Path(args.tmp_dir) if args.tmp_dir else Path(
        args.output + f".parts.{int(time.time())}"
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)

    target_total_tokens = None
    if args.target_train_tokens:
        target_total_tokens = math.ceil(args.target_train_tokens / (1 - args.val_ratio))
        logger.info("目标 train tokens=%d；按 val_ratio=%.4f 至少需 total tokens=%d",
                    args.target_train_tokens, args.val_ratio, target_total_tokens)

    logger.info("并行编码：input=%s tokenizer=%s vocab=%d dtype=%s",
                args.input, args.tokenizer, vocab_size, dtype_np)
    logger.info("workers=%d chunk_bytes=%s ranges=%d tmp_dir=%s",
                args.num_workers, args.chunk_bytes, len(ranges), tmp_dir)

    started = time.time()
    file_size = input_path.stat().st_size
    tasks = [
        {
            "index": i,
            "input": str(input_path),
            "start": start,
            "end": end,
            "file_size": file_size,
            "tokenizer": args.tokenizer,
            "text_field": args.text_field,
            "dtype": str(dtype_np),
            "out_path": str(tmp_dir / f"part_{i:06d}.bin"),
            "add_bos": args.add_bos,
        }
        for i, (start, end) in enumerate(ranges)
    ]

    selected: list[PartResult] = []
    total_docs = 0
    total_tokens = 0
    total_bad_json = 0
    total_empty_text = 0
    batch_size = max(1, args.num_workers)
    ctx = mp.get_context("spawn")

    try:
        with ctx.Pool(processes=args.num_workers) as pool:
            for batch_start in range(0, len(tasks), batch_size):
                batch = tasks[batch_start: batch_start + batch_size]
                results = pool.map(worker_encode, batch)
                for payload in sorted(results, key=lambda item: item["index"]):
                    part = PartResult(**payload)
                    selected.append(part)
                    total_docs += part.docs
                    total_tokens += part.tokens
                    total_bad_json += part.bad_json
                    total_empty_text += part.empty_text
                logger.info(
                    "已完成 chunks=%d/%d docs=%d tokens=%d bad_json=%d elapsed=%.1fs",
                    min(batch_start + len(batch), len(tasks)), len(tasks),
                    total_docs, total_tokens, total_bad_json,
                    time.time() - started,
                )
                if target_total_tokens and total_tokens >= target_total_tokens:
                    logger.info("已达到目标 total tokens=%d，停止提交后续 chunks",
                                total_tokens)
                    break

        logger.info("编码完成：chunks=%d docs=%d tokens=%d bad_json=%d empty_text=%d 用时 %.1fs",
                    len(selected), total_docs, total_tokens, total_bad_json,
                    total_empty_text, time.time() - started)
        merge_parts(selected, args, dtype_np, vocab_size)
    finally:
        if not args.keep_parts and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Parallel JSONL -> pretraining token .bin builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--val-output", default=None)
    parser.add_argument("--val-ratio", type=float, default=0.005)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--chunk-bytes", default="256MB")
    parser.add_argument("--tmp-dir", default=None)
    parser.add_argument("--target-train-tokens", type=int, default=0,
                        help="Stop after enough chunks to keep at least this many train tokens")
    parser.add_argument("--keep-parts", action="store_true")
    return parser


if __name__ == "__main__":
    build(build_parser().parse_args())
