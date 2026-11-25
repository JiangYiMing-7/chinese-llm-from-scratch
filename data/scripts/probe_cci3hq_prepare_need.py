#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample CCI3-HQ shards to decide whether full prepare is worth running.

The probe applies the same lightweight checks as prepare_cci3hq_corpus.py on a
sample, but it does not write a large cleaned corpus. Use it to decide between:

  1. Direct concat of downloaded shards.
  2. Full prepare with normalization/filtering.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_corpus import normalize_text  # noqa: E402


_CJK_RE = re.compile(r"[一-鿿㐀-䶿]")
_PUNCT_RE = re.compile(
    r"[　-〿＀-￯!-/:-@\[-`{-~，。！？、；：“”‘’（）《》【】—…·]"
)
_DIGIT_RE = re.compile(r"[0-9０-９]")


def char_ratio(text: str, pattern: re.Pattern[str]) -> float:
    if not text:
        return 0.0
    return len(pattern.findall(text)) / len(text)


def load_manifest_files(cci_dir: Path, manifest: Path | None) -> list[Path]:
    if manifest and manifest.exists():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        picked = payload.get("picked") or []
        return sorted(cci_dir / item for item in picked)
    return sorted(Path(p) for p in glob.glob(str(cci_dir / "data" / "part_*.jsonl")))


def pick_text(obj: dict, fields: list[str]) -> tuple[str, str | None]:
    for field in fields:
        value = obj.get(field)
        if isinstance(value, str) and value:
            return value, field
    return "", None


def maybe_add_example(examples: dict[str, list[dict]], reason: str, payload: dict,
                      limit: int) -> None:
    bucket = examples.setdefault(reason, [])
    if len(bucket) < limit:
        bucket.append(payload)


def shorten(text: str, max_chars: int) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def decision(counts: dict, args) -> dict:
    parseable = max(1, counts["parseable_text"])
    raw_chars = max(1, counts["raw_chars"])
    drop_doc_ratio = counts["dropped_total"] / parseable
    kept_char_ratio = counts["kept_chars"] / raw_chars
    char_loss_ratio = 1.0 - kept_char_ratio
    normalize_changed_ratio = counts["normalize_changed"] / parseable

    reasons = []
    if drop_doc_ratio >= args.doc_drop_threshold:
        reasons.append(
            f"sample doc drop ratio {drop_doc_ratio:.2%} >= "
            f"{args.doc_drop_threshold:.2%}"
        )
    if char_loss_ratio >= args.char_loss_threshold:
        reasons.append(
            f"sample char loss ratio {char_loss_ratio:.2%} >= "
            f"{args.char_loss_threshold:.2%}"
        )
    if normalize_changed_ratio >= args.normalize_threshold:
        reasons.append(
            f"normalize changed ratio {normalize_changed_ratio:.2%} >= "
            f"{args.normalize_threshold:.2%}"
        )

    if reasons:
        label = "run_prepare"
        summary = "Sample shows non-trivial filtering/normalization effect."
    else:
        label = "direct_concat_ok"
        summary = "Sample effect is small; direct concat is reasonable."

    return {
        "label": label,
        "summary": summary,
        "reasons": reasons,
        "thresholds": {
            "doc_drop_threshold": args.doc_drop_threshold,
            "char_loss_threshold": args.char_loss_threshold,
            "normalize_threshold": args.normalize_threshold,
        },
        "sample_ratios": {
            "drop_doc_ratio": round(drop_doc_ratio, 6),
            "kept_char_ratio": round(kept_char_ratio, 6),
            "char_loss_ratio": round(char_loss_ratio, 6),
            "normalize_changed_ratio": round(normalize_changed_ratio, 6),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe whether full CCI3-HQ prepare is necessary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cci-dir", default="/root/autodl-tmp/data/cci3hq")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="reports/cci3hq_500m/prepare_probe.json")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--sample-files", type=int, default=12)
    parser.add_argument("--max-docs", type=int, default=100000)
    parser.add_argument("--max-docs-per-file", type=int, default=0,
                        help="0 means no per-file cap; max-docs still applies")
    parser.add_argument("--text-fields", default="text,content")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=100000)
    parser.add_argument("--cjk-threshold", type=float, default=0.30)
    parser.add_argument("--punct-ratio", type=float, default=0.35)
    parser.add_argument("--digit-ratio", type=float, default=0.45)
    parser.add_argument("--dedup-backend", choices=["none", "memory"],
                        default="none",
                        help="Sample-only exact SHA-256 dedup check")
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--example-chars", type=int, default=240)
    parser.add_argument("--doc-drop-threshold", type=float, default=0.02)
    parser.add_argument("--char-loss-threshold", type=float, default=0.02)
    parser.add_argument("--normalize-threshold", type=float, default=0.10)
    args = parser.parse_args()

    cci_dir = Path(args.cci_dir)
    manifest = Path(args.manifest) if args.manifest else None
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]

    files = load_manifest_files(cci_dir, manifest)
    existing = [p for p in files if p.exists() and p.stat().st_size > 0]
    total_raw_bytes = sum(p.stat().st_size for p in existing)
    missing = [str(p) for p in files if p not in existing]
    if not existing:
        raise FileNotFoundError(f"no shard files found under {cci_dir}")

    rng = random.Random(args.seed)
    sample_files = existing[:]
    rng.shuffle(sample_files)
    sample_files = sorted(sample_files[: min(args.sample_files, len(sample_files))])
    sample_raw_bytes = sum(p.stat().st_size for p in sample_files)
    scale = total_raw_bytes / sample_raw_bytes if sample_raw_bytes else 0.0

    seen_hashes: set[str] = set()
    examples: dict[str, list[dict]] = {}
    field_counts: dict[str, int] = {}
    counts = {
        "sample_files": len(sample_files),
        "lines_total": 0,
        "bad_json": 0,
        "empty_text": 0,
        "parseable_text": 0,
        "raw_chars": 0,
        "normalized_chars": 0,
        "normalize_changed": 0,
        "drop_length": 0,
        "drop_cjk": 0,
        "drop_punct": 0,
        "drop_digit": 0,
        "drop_exact_dup": 0,
        "dropped_total": 0,
        "kept": 0,
        "kept_chars": 0,
    }

    started = time.time()
    stop = False
    for path in sample_files:
        docs_in_file = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if args.max_docs and counts["parseable_text"] >= args.max_docs:
                    stop = True
                    break
                if args.max_docs_per_file and docs_in_file >= args.max_docs_per_file:
                    break
                docs_in_file += 1
                counts["lines_total"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] += 1
                    maybe_add_example(
                        examples, "bad_json",
                        {"shard": path.name, "line": shorten(line, args.example_chars)},
                        args.example_limit,
                    )
                    continue

                raw_text, field = pick_text(obj, text_fields)
                if not raw_text:
                    counts["empty_text"] += 1
                    maybe_add_example(
                        examples, "empty_text", {"shard": path.name},
                        args.example_limit,
                    )
                    continue

                counts["parseable_text"] += 1
                field_counts[field or ""] = field_counts.get(field or "", 0) + 1
                counts["raw_chars"] += len(raw_text)

                text = normalize_text(raw_text)
                counts["normalized_chars"] += len(text)
                if text != raw_text:
                    counts["normalize_changed"] += 1

                reason = None
                if not text:
                    reason = "empty_after_normalize"
                    counts["empty_text"] += 1
                elif len(text) < args.min_len or len(text) > args.max_len:
                    reason = "drop_length"
                    counts["drop_length"] += 1
                elif char_ratio(text[:2000], _CJK_RE) < args.cjk_threshold:
                    reason = "drop_cjk"
                    counts["drop_cjk"] += 1
                elif char_ratio(text, _PUNCT_RE) > args.punct_ratio:
                    reason = "drop_punct"
                    counts["drop_punct"] += 1
                elif char_ratio(text, _DIGIT_RE) > args.digit_ratio:
                    reason = "drop_digit"
                    counts["drop_digit"] += 1
                elif args.dedup_backend == "memory":
                    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if digest in seen_hashes:
                        reason = "drop_exact_dup"
                        counts["drop_exact_dup"] += 1
                    else:
                        seen_hashes.add(digest)

                if reason:
                    counts["dropped_total"] += 1
                    maybe_add_example(
                        examples,
                        reason,
                        {
                            "shard": path.name,
                            "raw_len": len(raw_text),
                            "normalized_len": len(text),
                            "cjk_ratio": round(char_ratio(text[:2000], _CJK_RE), 4),
                            "punct_ratio": round(char_ratio(text, _PUNCT_RE), 4),
                            "digit_ratio": round(char_ratio(text, _DIGIT_RE), 4),
                            "text": shorten(text, args.example_chars),
                        },
                        args.example_limit,
                    )
                    continue

                counts["kept"] += 1
                counts["kept_chars"] += len(text)
                maybe_add_example(
                    examples,
                    "kept",
                    {"shard": path.name, "text": shorten(text, args.example_chars)},
                    args.example_limit,
                )
        if stop:
            break

    dec = decision(counts, args)
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
        "scale_to_existing_files": round(scale, 4),
        "params": {
            "sample_files": args.sample_files,
            "max_docs": args.max_docs,
            "max_docs_per_file": args.max_docs_per_file,
            "text_fields": text_fields,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "cjk_threshold": args.cjk_threshold,
            "punct_ratio": args.punct_ratio,
            "digit_ratio": args.digit_ratio,
            "dedup_backend": args.dedup_backend,
        },
        "counts": counts,
        "field_counts": field_counts,
        "ratios": {
            "bad_json_per_line": round(counts["bad_json"] / max(1, counts["lines_total"]), 6),
            "empty_text_per_line": round(counts["empty_text"] / max(1, counts["lines_total"]), 6),
            "drop_doc_ratio": dec["sample_ratios"]["drop_doc_ratio"],
            "kept_doc_ratio": round(counts["kept"] / max(1, counts["parseable_text"]), 6),
            "kept_char_ratio": dec["sample_ratios"]["kept_char_ratio"],
            "char_loss_ratio": dec["sample_ratios"]["char_loss_ratio"],
            "normalize_changed_ratio": dec["sample_ratios"]["normalize_changed_ratio"],
        },
        "estimated_full_effect": {
            "raw_chars": int(counts["raw_chars"] * scale),
            "kept_chars_after_prepare": int(counts["kept_chars"] * scale),
            "char_loss": int((counts["raw_chars"] - counts["kept_chars"]) * scale),
            "kept_tokens_at_0.60": int(counts["kept_chars"] * scale * 0.60),
            "kept_tokens_at_0.65": int(counts["kept_chars"] * scale * 0.65),
            "kept_tokens_at_0.70": int(counts["kept_chars"] * scale * 0.70),
        },
        "decision": dec,
        "examples": examples,
        "elapsed_seconds": round(time.time() - started, 1),
    }

    tmp = output.with_name(output.name + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, output)

    print(json.dumps({
        "decision": dec["label"],
        "summary": dec["summary"],
        "drop_doc_ratio": dec["sample_ratios"]["drop_doc_ratio"],
        "char_loss_ratio": dec["sample_ratios"]["char_loss_ratio"],
        "normalize_changed_ratio": dec["sample_ratios"]["normalize_changed_ratio"],
        "estimated_kept_tokens_at_0.60": report["estimated_full_effect"]["kept_tokens_at_0.60"],
        "report": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
