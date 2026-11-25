#!/usr/bin/env python3
"""Validate task-aware safety SFT/eval JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
    return rows


def row_text(row: dict[str, Any]) -> str:
    if "messages" in row:
        return "\n".join(str(msg.get("content", "")) for msg in row["messages"])
    return "\n".join(str(row.get(key, "")) for key in ("system", "user", "reference_answer"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def validate_sft_rows(name: str, rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not rows:
        return [f"{name}: no rows"]
    for idx, row in enumerate(rows[:]):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"{name}:{idx}: missing messages")
            continue
        roles = [msg.get("role") for msg in messages]
        if "user" not in roles or "assistant" not in roles:
            errors.append(f"{name}:{idx}: messages need user and assistant")
        if not row.get("task_type"):
            errors.append(f"{name}:{idx}: missing task_type")
        if not row.get("category"):
            errors.append(f"{name}:{idx}: missing category")
    return errors


def validate_eval_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    required = ["id", "system", "user", "expected_behavior", "required_terms", "forbidden_terms"]
    for idx, row in enumerate(rows):
        for key in required:
            if key not in row:
                errors.append(f"eval:{idx}: missing {key}")
        if row.get("expected_behavior") == "complete_task_ignore_injection" and not row.get("required_terms"):
            errors.append(f"eval:{idx}: prompt-injection case has no required_terms")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate safety SFT/eval v1 files.")
    parser.add_argument("--train", type=Path, default=Path("data/sft/safety_sft_train.jsonl"))
    parser.add_argument("--val", type=Path, default=Path("data/sft/safety_sft_val.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("eval/data/zh_safety_tasks_v1.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/reports/safety_sft_v1_validation.json"))
    args = parser.parse_args()

    train_rows = read_jsonl(args.train)
    val_rows = read_jsonl(args.val)
    eval_rows = read_jsonl(args.eval)
    errors = []
    errors.extend(validate_sft_rows("train", train_rows))
    errors.extend(validate_sft_rows("val", val_rows))
    errors.extend(validate_eval_rows(eval_rows))

    train_hashes = {sha256_text(row_text(row)) for row in train_rows}
    val_hashes = {sha256_text(row_text(row)) for row in val_rows}
    eval_hashes = {sha256_text(row_text(row)) for row in eval_rows}
    overlap = {
        "train_val_text_hash_overlap": len(train_hashes & val_hashes),
        "train_eval_text_hash_overlap": len(train_hashes & eval_hashes),
        "val_eval_text_hash_overlap": len(val_hashes & eval_hashes),
    }
    for key, value in overlap.items():
        if value:
            errors.append(f"{key}: {value}")

    report = {
        "status": "failed" if errors else "passed",
        "errors": errors,
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "eval": len(eval_rows),
        },
        "overlap": overlap,
        "train_task_counts": count_by(train_rows, "task_type"),
        "val_task_counts": count_by(val_rows, "task_type"),
        "eval_task_counts": count_by(eval_rows, "task_type"),
        "eval_attack_counts": count_by(eval_rows, "attack_type"),
        "eval_category_counts": count_by(eval_rows, "category"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
