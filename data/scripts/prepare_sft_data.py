#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


LANGUAGE_TASK_TYPES = {
    "instruction_following",
    "summary",
    "summarization",
    "extract",
    "information_extraction",
    "reading_qa",
    "search_info",
    "rewrite",
    "classification",
    "format_following",
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_row(row: dict[str, Any], source_name: str, default_task_type: str | None) -> dict[str, Any] | None:
    if "messages" in row:
        messages = row["messages"]
    elif "instruction" in row and "output" in row:
        user = str(row.get("instruction", "")).strip()
        extra = str(row.get("input", "")).strip()
        if extra:
            user = f"{user}\n{extra}"
        answer = str(row.get("output", "")).strip()
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
    elif "prompt" in row and "response" in row:
        messages = [
            {"role": "user", "content": str(row.get("prompt", "")).strip()},
            {"role": "assistant", "content": str(row.get("response", "")).strip()},
        ]
    else:
        return None

    clean_messages = []
    for message in messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role not in {"system", "user", "assistant"} or not content:
            continue
        clean_messages.append({"role": role, "content": content})
    if not any(message["role"] == "assistant" for message in clean_messages):
        return None
    if not any(message["role"] == "user" for message in clean_messages):
        return None

    task_type = row.get("task_type") or row.get("category") or default_task_type or "unknown"
    return {
        "messages": clean_messages,
        "source": row.get("source") or source_name,
        "task_type": task_type,
        "license": row.get("license"),
        "original_id": row.get("id"),
    }


def too_long_or_short(row: dict[str, Any], min_chars: int, max_chars: int) -> bool:
    text = "\n".join(message["content"] for message in row["messages"])
    return len(text) < min_chars or len(text) > max_chars


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Chinese SFT data into chat JSONL.")
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="normal_sft")
    parser.add_argument("--default-task-type", default=None)
    parser.add_argument("--language-tasks-only", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-chars", type=int, default=8)
    parser.add_argument("--max-chars", type=int, default=6000)
    args = parser.parse_args()

    rows = []
    for path in args.input:
        source_name = path.stem
        for raw in read_jsonl(path):
            row = normalize_row(raw, source_name, args.default_task_type)
            if row is None:
                continue
            if too_long_or_short(row, args.min_chars, args.max_chars):
                continue
            task_type = str(row.get("task_type", "")).lower()
            if args.language_tasks_only and task_type not in LANGUAGE_TASK_TYPES:
                continue
            rows.append(row)

    random.Random(args.seed).shuffle(rows)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("no usable SFT rows after normalization/filtering")

    val_size = int(round(len(rows) * args.val_ratio))
    if len(rows) > 1:
        val_size = max(1, min(val_size, len(rows) - 1))
    else:
        val_size = 0
    val_rows = rows[:val_size]
    train_rows = rows[val_size:] if val_size else rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / f"{args.name}_train.jsonl"
    val_path = args.output_dir / f"{args.name}_val.jsonl"
    report_path = args.output_dir / f"{args.name}_report.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows or train_rows[:1])
    task_counts = {}
    for row in rows:
        task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
    report = {
        "name": args.name,
        "inputs": [str(path) for path in args.input],
        "train_path": str(train_path),
        "val_path": str(val_path),
        "total_rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows or train_rows[:1]),
        "language_tasks_only": args.language_tasks_only,
        "task_counts": task_counts,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
