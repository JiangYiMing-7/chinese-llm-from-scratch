#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect downloaded BAAI/CCI3-HQ shards and estimate corpus scale.

This script is intentionally read-only: it samples downloaded shard files,
checks JSONL/text-field health, estimates total characters, and can optionally
tokenize the sample with a SentencePiece model to estimate total tokens.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import statistics
import time
from pathlib import Path


_CJK_RE_RANGE = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
)


def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RE_RANGE)


def cjk_ratio(text: str, limit: int = 2000) -> float:
    sample = text[:limit]
    if not sample:
        return 0.0
    return sum(1 for ch in sample if _is_cjk(ch)) / len(sample)


def percentile(sorted_values: list[int], q: float) -> int:
    if not sorted_values:
        return 0
    idx = min(len(sorted_values) - 1, max(0, int(len(sorted_values) * q)))
    return sorted_values[idx]


def load_manifest_files(cci_dir: Path, manifest: Path | None) -> list[Path]:
    if manifest and manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        picked = payload.get("picked") or []
        files = [cci_dir / item for item in picked]
    else:
        files = [Path(p) for p in glob.glob(str(cci_dir / "data" / "part_*.jsonl"))]
    return sorted(files)


def pick_text(obj: dict, fields: list[str]) -> tuple[str, str | None]:
    for field in fields:
        value = obj.get(field)
        if isinstance(value, str) and value:
            return value, field
    return "", None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect downloaded CCI3-HQ shards and estimate token budget.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cci-dir", default="/root/autodl-tmp/data/cci3hq")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="reports/cci3hq_500m/raw_inspect.json")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--sample-files", type=int, default=8)
    parser.add_argument("--max-docs-per-file", type=int, default=0,
                        help="0 means scan the selected sample files fully")
    parser.add_argument("--text-fields", default="text,content",
                        help="Comma-separated candidate text fields")
    parser.add_argument("--tokenizer", default=None,
                        help="Optional SentencePiece .model or HF tokenizers .json for true token estimate")
    parser.add_argument("--tokenizer-max-docs", type=int, default=20000)
    args = parser.parse_args()

    cci_dir = Path(args.cci_dir)
    manifest = Path(args.manifest) if args.manifest else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]

    files = load_manifest_files(cci_dir, manifest)
    existing = [p for p in files if p.exists() and p.stat().st_size > 0]
    missing = [str(p) for p in files if p not in existing]
    total_raw_bytes = sum(p.stat().st_size for p in existing)

    rng = random.Random(args.seed)
    sample_files = existing[:]
    rng.shuffle(sample_files)
    sample_files = sorted(sample_files[: min(args.sample_files, len(sample_files))])

    tokenizer = None
    if args.tokenizer:
        from tokenizer_io import load_tokenizer

        tokenizer = load_tokenizer(args.tokenizer)

    started = time.time()
    stats = {
        "docs": 0,
        "bad_json": 0,
        "empty_text": 0,
        "chars": 0,
        "text_bytes": 0,
        "cjk_ratio_sum": 0.0,
        "token_docs": 0,
        "tokens": 0,
    }
    field_counts: dict[str, int] = {}
    lens: list[int] = []
    sample_raw_bytes = 0

    for path in sample_files:
        sample_raw_bytes += path.stat().st_size
        docs_in_file = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if args.max_docs_per_file and docs_in_file >= args.max_docs_per_file:
                    break
                docs_in_file += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_json"] += 1
                    continue
                text, field = pick_text(obj, text_fields)
                if not text:
                    stats["empty_text"] += 1
                    continue
                field_counts[field or ""] = field_counts.get(field or "", 0) + 1
                n_chars = len(text)
                stats["docs"] += 1
                stats["chars"] += n_chars
                stats["text_bytes"] += len(text.encode("utf-8", errors="replace"))
                stats["cjk_ratio_sum"] += cjk_ratio(text)
                lens.append(n_chars)
                if tokenizer is not None and stats["token_docs"] < args.tokenizer_max_docs:
                    stats["tokens"] += len(tokenizer.encode(text)) + 1
                    stats["token_docs"] += 1

    scale = total_raw_bytes / sample_raw_bytes if sample_raw_bytes else 0.0
    est_chars = int(stats["chars"] * scale)
    lens_sorted = sorted(lens)
    tokens_per_char = None
    est_tokens_from_tokenizer = None
    if stats["token_docs"] and stats["chars"]:
        # The tokenizer sample is taken from the same prefix of scanned docs.
        # For CCI3-HQ scale planning this is good enough; rerun with a larger
        # tokenizer-max-docs if it is near the threshold.
        token_sample_chars = sum(lens[: stats["token_docs"]])
        tokens_per_char = stats["tokens"] / max(1, token_sample_chars)
        est_tokens_from_tokenizer = int(est_chars * tokens_per_char)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cci_dir": str(cci_dir),
        "manifest": str(manifest) if manifest else None,
        "seed": args.seed,
        "total_manifest_files": len(files),
        "existing_files": len(existing),
        "missing_files": missing,
        "total_raw_bytes": total_raw_bytes,
        "total_raw_gb_decimal": round(total_raw_bytes / 1e9, 3),
        "sample_files": [str(p) for p in sample_files],
        "sample_raw_bytes": sample_raw_bytes,
        "sample_raw_gb_decimal": round(sample_raw_bytes / 1e9, 3),
        "sample_stats": {
            **stats,
            "avg_cjk_ratio": round(
                stats["cjk_ratio_sum"] / max(1, stats["docs"]), 4
            ),
            "text_bytes_per_raw_byte": round(
                stats["text_bytes"] / max(1, sample_raw_bytes), 4
            ),
            "avg_doc_chars": round(stats["chars"] / max(1, stats["docs"]), 1),
            "doc_len_p50": percentile(lens_sorted, 0.50),
            "doc_len_p90": percentile(lens_sorted, 0.90),
            "doc_len_p99": percentile(lens_sorted, 0.99),
            "doc_len_max": max(lens_sorted) if lens_sorted else 0,
            "doc_len_mean": round(statistics.mean(lens_sorted), 1)
            if lens_sorted else 0,
        },
        "field_counts": field_counts,
        "estimates": {
            "estimated_total_chars": est_chars,
            "estimated_tokens_by_tokens_per_char": {
                str(rate): int(est_chars * rate)
                for rate in (0.55, 0.60, 0.65, 0.70, 0.75)
            },
            "tokenizer": args.tokenizer,
            "tokenizer_sample_docs": stats["token_docs"],
            "tokenizer_tokens_per_char": round(tokens_per_char, 6)
            if tokens_per_char is not None else None,
            "estimated_tokens_from_tokenizer": est_tokens_from_tokenizer,
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }

    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output)

    print(json.dumps({
        "existing_files": len(existing),
        "total_raw_gb": report["total_raw_gb_decimal"],
        "estimated_total_chars": est_chars,
        "estimated_tokens_0.60": report["estimates"]["estimated_tokens_by_tokens_per_char"]["0.6"],
        "estimated_tokens_from_tokenizer": est_tokens_from_tokenizer,
        "report": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
