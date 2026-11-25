#!/usr/bin/env python3
"""Validate split isolation and coverage for the 500M v2 posttrain/eval flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_SFT_SOURCES = {"cmrc2018", "coig_cqia", "belle_train_1m_cn", "cluener2020", "pclue", "lcsts", "duie"}
REQUIRED_EVAL_TASKS = {
    "summarization",
    "reading_qa",
    "search_info",
    "entity_extraction",
    "relation_extraction",
    "classification",
    "sentence_matching",
    "nli",
    "instruction_format",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
    return rows


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def values(rows: list[dict[str, Any]], key: str) -> set[str]:
    return {str(row[key]) for row in rows if row.get(key)}


def pclue_bad_sft(rows: list[dict[str, Any]]) -> list[str]:
    bad = []
    for row in rows:
        if row.get("source") != "pclue":
            continue
        split = str(row.get("source_split", ""))
        uid = str(row.get("source_uid", ""))
        if split != "train" or "pCLUE_dev" in uid or "test_public" in uid:
            bad.append(uid or str(row.get("id", "")))
    return bad[:50]


def pclue_bad_eval(rows: list[dict[str, Any]]) -> list[str]:
    bad = []
    for row in rows:
        if row.get("source") != "pclue":
            continue
        split = str(row.get("source_split", ""))
        uid = str(row.get("source_uid", ""))
        if split == "train" or "pCLUE_train" in uid:
            bad.append(uid or str(row.get("id", "")))
    return bad[:50]


def official_bad_sft(rows: list[dict[str, Any]]) -> list[str]:
    bad = []
    for row in rows:
        source = str(row.get("source", ""))
        if source in {"clue_official", "fewclue_official"}:
            bad.append(str(row.get("source_uid", row.get("id", ""))))
    return bad[:50]


def validate_json_references(rows: list[dict[str, Any]]) -> list[str]:
    bad = []
    for row in rows:
        keys = row.get("required_json_keys") or []
        if not keys:
            continue
        ok = False
        for ref in row.get("references", []):
            try:
                payload = json.loads(ref)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and all(key in payload for key in keys):
                ok = True
                break
        if not ok:
            bad.append(str(row.get("id", row.get("source_uid", ""))))
    return bad[:50]


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 500M v2 Posttrain/Eval Validation",
        "",
        f"- status: `{report['status']}`",
        f"- sft_train_examples: {report['sft_train_examples']}",
        f"- sft_val_examples: {report['sft_val_examples']}",
        f"- reference_eval_examples: {report['reference_eval_examples']}",
        "",
        "## Errors",
        "",
    ]
    lines.extend(f"- {item}" for item in report["errors"] or ["none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {item}" for item in report["warnings"] or ["none"])
    lines.extend([
        "",
        "## Coverage",
        "",
        "### SFT Train Sources",
        "",
    ])
    lines.extend(f"- {key}: {value}" for key, value in report["sft_train_source_counts"].items())
    lines.extend(["", "### Reference Eval Tasks", ""])
    lines.extend(f"- {key}: {value}" for key, value in report["reference_task_counts"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate 500M v2 SFT/eval split isolation.")
    parser.add_argument("--sft-train", type=Path, default=Path("data/sft/normal_sft_500m_v2_train.jsonl"))
    parser.add_argument("--sft-val", type=Path, default=Path("data/sft/normal_sft_500m_v2_val.jsonl"))
    parser.add_argument("--reference-eval", type=Path, default=Path("eval/data/zh_reference_tasks_500m_v2.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    train = read_jsonl(args.sft_train)
    val = read_jsonl(args.sft_val)
    reference = read_jsonl(args.reference_eval)
    sft_all = train + val

    errors: list[str] = []
    warnings: list[str] = []
    train_hashes = values(train, "text_hash")
    val_hashes = values(val, "text_hash")
    ref_hashes = values(reference, "text_hash")
    train_core = values(train, "core_task_hash")
    ref_core = values(reference, "core_task_hash")

    overlap_train_val = train_hashes & val_hashes
    overlap_train_ref = train_hashes & ref_hashes
    overlap_core = train_core & ref_core
    if overlap_train_val:
        errors.append(f"SFT train/val text_hash overlap: {len(overlap_train_val)}")
    if overlap_train_ref:
        errors.append(f"SFT train/reference eval text_hash overlap: {len(overlap_train_ref)}")
    if overlap_core:
        errors.append(f"SFT train/reference eval core_task_hash overlap: {len(overlap_core)}")

    bad = pclue_bad_sft(sft_all)
    if bad:
        errors.append(f"pCLUE non-train rows found in SFT: {bad[:5]}")
    bad = pclue_bad_eval(reference)
    if bad:
        errors.append(f"pCLUE train rows found in reference eval: {bad[:5]}")
    bad = official_bad_sft(sft_all)
    if bad:
        errors.append(f"official CLUE/FewCLUE rows found in SFT: {bad[:5]}")
    bad_json = validate_json_references(reference)
    if bad_json:
        errors.append(f"structured reference rows have invalid JSON references: {bad_json[:5]}")

    train_sources = set(count_by(train, "source"))
    missing_sources = sorted(REQUIRED_SFT_SOURCES - train_sources)
    if missing_sources:
        warnings.append(f"SFT train missing core sources: {missing_sources}")
    ref_tasks = set(count_by(reference, "task_type"))
    missing_tasks = sorted(REQUIRED_EVAL_TASKS - ref_tasks)
    if missing_tasks:
        warnings.append(f"reference eval missing task types: {missing_tasks}")

    report = {
        "status": "failed" if errors else "passed",
        "sft_train_path": str(args.sft_train),
        "sft_val_path": str(args.sft_val),
        "reference_eval_path": str(args.reference_eval),
        "sft_train_examples": len(train),
        "sft_val_examples": len(val),
        "reference_eval_examples": len(reference),
        "errors": errors,
        "warnings": warnings,
        "overlap_counts": {
            "train_val_text_hash": len(overlap_train_val),
            "train_reference_text_hash": len(overlap_train_ref),
            "train_reference_core_task_hash": len(overlap_core),
        },
        "sft_train_source_counts": count_by(train, "source"),
        "sft_train_task_counts": count_by(train, "task_type"),
        "sft_val_source_counts": count_by(val, "source"),
        "reference_source_counts": count_by(reference, "source"),
        "reference_task_counts": count_by(reference, "task_type"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "posttrain_eval_v2_validation.json"
    md_path = args.output_dir / "posttrain_eval_v2_validation.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"status": report["status"], "json": str(json_path), "markdown": str(md_path), "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
    if errors and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
