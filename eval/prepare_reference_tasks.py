#!/usr/bin/env python3
"""Prepare larger Chinese reference-answer eval sets.

The output JSONL is consumed by eval/evaluate_reference_tasks.py. This script
supports common LCSTS-style summarization files, CMRC/SQuAD-style MRC files,
and already-normalized project JSONL rows.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


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


def first(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def normalize_references(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        refs = value
    elif isinstance(value, dict):
        refs = value.get("text") or value.get("answers") or []
    else:
        refs = [value]
    normalized = []
    for ref in refs:
        if isinstance(ref, dict):
            ref = ref.get("text") or ref.get("answer") or ref.get("value")
        text = str(ref).strip()
        if text:
            normalized.append(text)
    return normalized


def load_lcsts(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(read_jsonl(path)):
        text = first(row, ["text", "content", "article", "document"])
        summary = first(row, ["summary", "summarization", "target", "title"])
        if not text or not summary:
            continue
        rows.append({
            "id": f"lcsts_{i:06d}",
            "task_type": "summarization",
            "source": "LCSTS_or_summary_jsonl",
            "user": f"请概括下面文本的主要内容，要求忠实原文、简洁：\n文本：{text}",
            "references": [str(summary).strip()],
            "metric": "rouge_l",
            "max_new_tokens": 80,
        })
        if limit and len(rows) >= limit:
            break
    return rows


def iter_cmrc_items(payload: Any):
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return
    if "data" in payload:
        for article in payload["data"]:
            for paragraph in article.get("paragraphs", []):
                context = paragraph.get("context", "")
                for qa in paragraph.get("qas", []):
                    yield {"context": context, **qa}
        return
    if isinstance(payload, dict):
        yield payload


def load_cmrc(path: Path, limit: int | None) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for i, row in enumerate(iter_cmrc_items(payload)):
        context = first(row, ["context", "paragraph", "document", "text"])
        question = first(row, ["question", "query"])
        answers = normalize_references(first(row, ["answers", "answer"]))
        if not context or not question or not answers:
            continue
        rows.append({
            "id": str(first(row, ["id", "qid"], f"cmrc_{i:06d}")),
            "task_type": "reading_qa",
            "source": "CMRC_or_mrc_json",
            "user": f"请只根据给定文本回答问题。\n文本：{context}\n问题：{question}",
            "references": answers,
            "metric": "char_f1",
            "max_new_tokens": 64,
        })
        if limit and len(rows) >= limit:
            break
    return rows


def load_generic_jsonl(path: Path, task_type: str | None, metric: str | None, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(read_jsonl(path)):
        user = first(row, ["user", "prompt", "question", "instruction"])
        context = first(row, ["context", "text", "document", "article"])
        references = normalize_references(first(row, ["references", "answers", "answer", "target", "output"]))
        if not user and context:
            user = f"请根据文本完成任务：\n文本：{context}"
        if not user or not references:
            continue
        row_task_type = row.get("task_type") or task_type or "custom_reference"
        rows.append({
            "id": str(row.get("id", f"{path.stem}_{i:06d}")),
            "task_type": row_task_type,
            "source": row.get("source") or path.stem,
            "user": str(user),
            "references": references,
            "metric": row.get("metric") or metric or "char_f1",
            "max_new_tokens": int(row.get("max_new_tokens", 96)),
        })
        if limit and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified larger reference eval JSONL.")
    parser.add_argument("--lcsts", action="append", default=[], type=Path)
    parser.add_argument("--cmrc", action="append", default=[], type=Path)
    parser.add_argument("--jsonl", action="append", default=[], type=Path)
    parser.add_argument("--jsonl-task-type", default=None)
    parser.add_argument("--jsonl-metric", default=None, choices=[None, "char_f1", "rouge_l", "em"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit-per-source", type=int, default=None)
    parser.add_argument("--max-total", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rows = []
    for path in args.lcsts:
        rows.extend(load_lcsts(path, args.limit_per_source))
    for path in args.cmrc:
        rows.extend(load_cmrc(path, args.limit_per_source))
    for path in args.jsonl:
        rows.extend(load_generic_jsonl(path, args.jsonl_task_type, args.jsonl_metric, args.limit_per_source))
    if not rows:
        raise ValueError("no reference eval rows were prepared")
    random.Random(args.seed).shuffle(rows)
    if args.max_total:
        rows = rows[:args.max_total]
    write_jsonl(args.output, rows)
    counts = {}
    for row in rows:
        counts[row["task_type"]] = counts.get(row["task_type"], 0) + 1
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "task_counts": counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
