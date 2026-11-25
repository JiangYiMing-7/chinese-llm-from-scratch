#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare a single-corpus CCI3-HQ JSONL for 500M pretraining.

The script reads downloaded shard files, applies lightweight quality filters,
normalizes text, optionally performs exact SHA-256 deduplication, and writes a
plain JSONL corpus with a `text` field. It is designed for the single-source
CCI3-HQ plan, not the earlier mixed-corpus pipeline.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_corpus import normalize_text  # noqa: E402
from dedup_utils import ExactHashStore, ensure_distinct_paths, ensure_parent_dir  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("prepare_cci3hq_corpus")

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


def make_hash_store(args) -> ExactHashStore | None:
    if args.dedup_backend == "none":
        logger.info("SHA-256 精确去重关闭")
        return None
    return ExactHashStore(
        args.dedup_backend,
        args.redis_host,
        args.redis_port,
        args.redis_prefix,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare cleaned single-corpus CCI3-HQ JSONL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cci-dir", default="/root/autodl-tmp/data/cci3hq")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--text-fields", default="text,content")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=100000)
    parser.add_argument("--cjk-threshold", type=float, default=0.30)
    parser.add_argument("--punct-ratio", type=float, default=0.35)
    parser.add_argument("--digit-ratio", type=float, default=0.45)
    parser.add_argument("--dedup-backend", choices=["none", "memory", "redis"],
                        default="memory")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--redis-prefix", default="cci3hq500m")
    parser.add_argument("--max-docs", type=int, default=0,
                        help="0 means process all documents")
    parser.add_argument("--write-source-metadata", action="store_true")
    parser.add_argument("--progress-interval", type=int, default=100000)
    args = parser.parse_args()

    cci_dir = Path(args.cci_dir)
    manifest = Path(args.manifest) if args.manifest else None
    output = Path(args.output)
    report_path = Path(args.report)
    ensure_distinct_paths(str(output), str(report_path))
    ensure_parent_dir(str(output))
    ensure_parent_dir(str(report_path))
    text_fields = [x.strip() for x in args.text_fields.split(",") if x.strip()]

    files = load_manifest_files(cci_dir, manifest)
    existing = [p for p in files if p.exists() and p.stat().st_size > 0]
    missing = [str(p) for p in files if p not in existing]
    if not existing:
        raise FileNotFoundError(f"no shard files found under {cci_dir}")

    hash_store = make_hash_store(args)
    stats = {
        "files_total": len(files),
        "files_existing": len(existing),
        "lines_total": 0,
        "bad_json": 0,
        "drop_empty": 0,
        "drop_length": 0,
        "drop_cjk": 0,
        "drop_punct": 0,
        "drop_digit": 0,
        "drop_exact_dup": 0,
        "kept": 0,
        "kept_chars": 0,
        "kept_text_bytes": 0,
    }
    field_counts: dict[str, int] = {}
    kept_min_len = None
    kept_max_len = 0
    started = time.time()
    tmp_output = output.with_name(output.name + ".tmp")

    with tmp_output.open("w", encoding="utf-8") as out:
        for path in existing:
            logger.info("processing shard: %s", path)
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if args.max_docs and stats["kept"] >= args.max_docs:
                        break
                    stats["lines_total"] += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        stats["bad_json"] += 1
                        continue
                    raw_text, field = pick_text(obj, text_fields)
                    if not raw_text:
                        stats["drop_empty"] += 1
                        continue
                    field_counts[field or ""] = field_counts.get(field or "", 0) + 1

                    text = normalize_text(raw_text)
                    if not text:
                        stats["drop_empty"] += 1
                        continue
                    n = len(text)
                    if n < args.min_len or n > args.max_len:
                        stats["drop_length"] += 1
                        continue
                    if char_ratio(text[:2000], _CJK_RE) < args.cjk_threshold:
                        stats["drop_cjk"] += 1
                        continue
                    if char_ratio(text, _PUNCT_RE) > args.punct_ratio:
                        stats["drop_punct"] += 1
                        continue
                    if char_ratio(text, _DIGIT_RE) > args.digit_ratio:
                        stats["drop_digit"] += 1
                        continue

                    if hash_store is not None:
                        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                        if hash_store.contains(digest):
                            stats["drop_exact_dup"] += 1
                            continue
                        hash_store.add(digest)

                    row = {"text": text}
                    if args.write_source_metadata:
                        row.update({"source": "cci3-hq", "shard": path.name})
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stats["kept"] += 1
                    stats["kept_chars"] += n
                    stats["kept_text_bytes"] += len(text.encode("utf-8"))
                    kept_min_len = n if kept_min_len is None else min(kept_min_len, n)
                    kept_max_len = max(kept_max_len, n)

                    if stats["kept"] % args.progress_interval == 0:
                        logger.info(
                            "kept=%d chars=%.3fB elapsed=%.1fs",
                            stats["kept"],
                            stats["kept_chars"] / 1e9,
                            time.time() - started,
                        )
            if args.max_docs and stats["kept"] >= args.max_docs:
                break

    os.replace(tmp_output, output)

    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "cci_dir": str(cci_dir),
        "manifest": str(manifest) if manifest else None,
        "output": str(output),
        "missing_files": missing,
        "params": {
            "text_fields": text_fields,
            "min_len": args.min_len,
            "max_len": args.max_len,
            "cjk_threshold": args.cjk_threshold,
            "punct_ratio": args.punct_ratio,
            "digit_ratio": args.digit_ratio,
            "dedup_backend": args.dedup_backend,
        },
        "counts": stats,
        "field_counts": field_counts,
        "corpus": {
            "kept_chars": stats["kept_chars"],
            "kept_text_bytes": stats["kept_text_bytes"],
            "avg_doc_len": round(stats["kept_chars"] / max(1, stats["kept"]), 1),
            "min_doc_len": kept_min_len or 0,
            "max_doc_len": kept_max_len,
            "estimated_tokens_at_0.60": int(stats["kept_chars"] * 0.60),
            "estimated_tokens_at_0.65": int(stats["kept_chars"] * 0.65),
            "estimated_tokens_at_0.70": int(stats["kept_chars"] * 0.70),
        },
        "elapsed_seconds": round(time.time() - started, 1),
    }
    tmp_report = report_path.with_name(report_path.name + ".tmp")
    tmp_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_report, report_path)

    print(json.dumps({
        "kept_docs": stats["kept"],
        "kept_chars": stats["kept_chars"],
        "estimated_tokens_at_0.65": report["corpus"]["estimated_tokens_at_0.65"],
        "output": str(output),
        "report": str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
