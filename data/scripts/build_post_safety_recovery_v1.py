#!/usr/bin/env python3
"""Build small post-safety utility recovery SFT sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


RECIPES = {
    "r1": {
        "train": {
            "entity_extraction": 300,
            "relation_extraction": 300,
            "safety": 200,
        },
        "val": {
            "entity_extraction": 40,
            "relation_extraction": 40,
            "safety": 40,
        },
    },
    "r2": {
        "train": {
            "entity_extraction": 600,
            "relation_extraction": 600,
            "safety": 400,
        },
        "val": {
            "entity_extraction": 80,
            "relation_extraction": 80,
            "safety": 80,
        },
    },
    "r4": {
        "train": {
            "entity_extraction": 300,
            "relation_extraction": 300,
            "safety": 600,
            "reading": 300,
        },
        "val": {
            "entity_extraction": 50,
            "relation_extraction": 50,
            "safety": 100,
            "reading": 50,
        },
    },
    "r5_relation_heavy": {
        "train": {
            "entity_extraction": 400,
            "relation_extraction": 800,
            "safety": 500,
            "reading": 200,
        },
        "val": {
            "entity_extraction": 60,
            "relation_extraction": 120,
            "safety": 80,
            "reading": 40,
        },
    },
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def row_hash(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def task_type(row: dict[str, Any]) -> str:
    task = str(row.get("task_type", ""))
    if task == "summarization":
        return "summary"
    return task


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random, label: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"not enough rows for {label}: requested {count}, available {len(rows)}")
    pool = list(rows)
    rng.shuffle(pool)
    return pool[:count]


def tag_rows(rows: list[dict[str, Any]], component: str) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["recovery_component"] = component
        item["recovery_source_hash"] = row_hash(row)
        tagged.append(item)
    return tagged


def build_split(
    normal_rows: list[dict[str, Any]],
    safety_rows: list[dict[str, Any]],
    recipe: dict[str, int],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools = {
        "entity_extraction": [row for row in normal_rows if task_type(row) == "entity_extraction"],
        "relation_extraction": [row for row in normal_rows if task_type(row) == "relation_extraction"],
        "reading": [row for row in normal_rows if task_type(row) in {"reading_qa", "search_info"}],
        "safety": safety_rows,
    }
    selected: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    selected_hashes: set[str] = set()
    for component, count in recipe.items():
        candidates = [row for row in pools[component] if row_hash(row) not in selected_hashes]
        rows = sample_rows(candidates, count, rng, component)
        for row in rows:
            selected_hashes.add(row_hash(row))
        selected.extend(tag_rows(rows, component))
        report[component] = {
            "requested": count,
            "available": len(pools[component]),
            "selected": len(rows),
        }
    rng.shuffle(selected)
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-safety recovery SFT data.")
    parser.add_argument("--normal-train", type=Path, default=Path("data/sft/normal_sft_500m_v3_final_train.jsonl"))
    parser.add_argument("--normal-val", type=Path, default=Path("data/sft/normal_sft_500m_v3_final_val.jsonl"))
    parser.add_argument("--safety-train", type=Path, default=Path("data/sft/safety_sft_final_b_v3_train.jsonl"))
    parser.add_argument("--safety-val", type=Path, default=Path("data/sft/safety_sft_final_b_v3_val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--seed", type=int, default=20260624)
    args = parser.parse_args()

    normal_train = read_jsonl(args.normal_train)
    normal_val = read_jsonl(args.normal_val)
    safety_train = read_jsonl(args.safety_train)
    safety_val = read_jsonl(args.safety_val)
    rng = random.Random(args.seed)
    summary: dict[str, Any] = {
        "seed": args.seed,
        "inputs": {
            "normal_train": str(args.normal_train),
            "normal_val": str(args.normal_val),
            "safety_train": str(args.safety_train),
            "safety_val": str(args.safety_val),
        },
        "outputs": {},
    }

    for name, recipe in RECIPES.items():
        train_rows, train_report = build_split(normal_train, safety_train, recipe["train"], rng)
        val_rows, val_report = build_split(normal_val, safety_val, recipe["val"], rng)
        train_path = args.output_dir / f"recovery_v3_final_{name}_train.jsonl"
        val_path = args.output_dir / f"recovery_v3_final_{name}_val.jsonl"
        report_path = args.output_dir / f"recovery_v3_final_{name}_report.json"
        write_jsonl(train_path, train_rows)
        write_jsonl(val_path, val_rows)
        report = {
            "recipe": name,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
            "train_report": train_report,
            "val_report": val_report,
            "train_task_counts": dict(Counter(task_type(row) for row in train_rows)),
            "val_task_counts": dict(Counter(task_type(row) for row in val_rows)),
            "train_component_counts": dict(Counter(row.get("recovery_component") for row in train_rows)),
            "val_component_counts": dict(Counter(row.get("recovery_component") for row in val_rows)),
        }
        write_json(report_path, report)
        summary["outputs"][name] = report
        print(json.dumps({
            "recipe": name,
            "train": str(train_path),
            "val": str(val_path),
            "train_examples": len(train_rows),
            "val_examples": len(val_rows),
        }, ensure_ascii=False))
    write_json(args.output_dir / "recovery_v3_final_summary.json", summary)


if __name__ == "__main__":
    main()
