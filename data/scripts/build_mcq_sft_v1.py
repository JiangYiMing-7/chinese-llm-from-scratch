#!/usr/bin/env python3
"""Build a small MCQ-format SFT set from non-test C-Eval/CMMLU splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CHOICE_LABELS = ["A", "B", "C", "D"]
DEFAULT_SYSTEM = "你是一个中文单项选择题助手。请只输出正确选项的字母 A、B、C 或 D，不要解释。"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_bucket(key: str, modulo: int = 1000) -> int:
    return int(sha256_text(key)[:12], 16) % modulo


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def get_first(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    return text[0].upper() if text else ""


def subject_from_csv(path: Path, split: str) -> str:
    stem = path.stem
    for suffix in (f"_{split}", f"-{split}", split):
        if stem.endswith(suffix):
            return stem[: -len(suffix)].strip("_-")
    return stem


def load_split(root: Path, benchmark: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_dir = root / split
    paths = sorted(split_dir.glob("*.csv")) if split_dir.exists() else []
    for path in paths:
        subject = subject_from_csv(path, split)
        for i, raw in enumerate(read_csv_rows(path)):
            question = str(get_first(raw, ["question", "Question", "问题"], "")).strip()
            choices = {
                label: str(get_first(raw, [label, label.lower()], "")).strip()
                for label in CHOICE_LABELS
            }
            answer = normalize_answer(get_first(raw, ["answer", "Answer", "答案"], ""))
            if not question or answer not in CHOICE_LABELS or any(not choices[label] for label in CHOICE_LABELS):
                continue
            rows.append({
                "benchmark": benchmark,
                "split": split,
                "subject": subject,
                "id": str(get_first(raw, ["id", "ID"], f"{subject}_{i:05d}")),
                "question": question,
                "choices": choices,
                "answer": answer,
            })
    return rows


def render_user(example: dict[str, Any], variant: int) -> str:
    subject = example["subject"]
    question = example["question"]
    choices = example["choices"]
    option_block = (
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}"
    )
    if variant == 0:
        return (
            f"以下是关于{subject}的单项选择题，请选出正确答案。\n\n"
            f"题目：{question}\n{option_block}\n请只输出 A、B、C 或 D。"
        )
    if variant == 1:
        return (
            "请阅读题目，并从 A、B、C、D 中选择唯一正确的选项。不要解释。\n"
            f"科目：{subject}\n题目：{question}\n{option_block}\n答案："
        )
    return (
        f"单项选择题：{question}\n{option_block}\n"
        "最终答案只能是一个大写字母。"
    )


def make_chat_row(example: dict[str, Any], variant: int, row_id: str) -> dict[str, Any]:
    user = render_user(example, variant)
    text_hash = sha256_text(json.dumps({
        "system": DEFAULT_SYSTEM,
        "user": user,
        "assistant": example["answer"],
    }, ensure_ascii=False, sort_keys=True))
    return {
        "id": row_id,
        "source": f"{example['benchmark']}_{example['split']}",
        "task_type": "multiple_choice",
        "benchmark": example["benchmark"],
        "subject": example["subject"],
        "source_split": example["split"],
        "source_uid": f"{example['benchmark']}:{example['split']}:{example['subject']}:{example['id']}",
        "choice_labels": CHOICE_LABELS,
        "answer": example["answer"],
        "text_hash": text_hash,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": example["answer"]},
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))


def split_examples(examples: list[dict[str, Any]], val_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, val = [], []
    threshold = int(1000 * val_ratio)
    for ex in examples:
        key = f"{ex['benchmark']}:{ex['split']}:{ex['subject']}:{ex['id']}"
        if stable_bucket(key + ":mcq_val") < threshold:
            val.append(ex)
        else:
            train.append(ex)
    return train, val


def expand_rows(examples: list[dict[str, Any]], variants: int, prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, ex in enumerate(examples):
        for variant in range(variants):
            rows.append(make_chat_row(ex, variant, f"{prefix}_{i:06d}_v{variant}"))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MCQ-format SFT data from C-Eval/CMMLU non-test splits.")
    parser.add_argument("--ceval-dir", type=Path, default=Path("data/eval/ceval-exam"))
    parser.add_argument("--cmmlu-dir", type=Path, default=Path("data/eval/CMMLU/data"))
    parser.add_argument("--output-train", type=Path, default=Path("data/sft/mcq_sft_500m_v1_train.jsonl"))
    parser.add_argument("--output-val", type=Path, default=Path("data/sft/mcq_sft_500m_v1_val.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("data/sft/mcq_sft_500m_v1_report.json"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    if args.variants < 1:
        raise ValueError("--variants must be >= 1")
    examples: list[dict[str, Any]] = []
    for split in ["dev", "val"]:
        if (args.ceval_dir / split).exists():
            examples.extend(load_split(args.ceval_dir, "ceval", split))
    if (args.cmmlu_dir / "dev").exists():
        examples.extend(load_split(args.cmmlu_dir, "cmmlu", "dev"))
    if not examples:
        raise ValueError("no MCQ examples found; check C-Eval/CMMLU directories")

    rng = random.Random(args.seed)
    train_examples, val_examples = split_examples(examples, args.val_ratio)
    train_rows = expand_rows(train_examples, args.variants, "mcq_train")
    val_rows = expand_rows(val_examples, args.variants, "mcq_val")
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(args.output_train, train_rows)
    write_jsonl(args.output_val, val_rows)
    report = {
        "output_train": str(args.output_train),
        "output_val": str(args.output_val),
        "base_examples": len(examples),
        "train_base_examples": len(train_examples),
        "val_base_examples": len(val_examples),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "variants": args.variants,
        "val_ratio": args.val_ratio,
        "base_source_counts": count_by(examples, "benchmark"),
        "base_split_counts": count_by(examples, "split"),
        "train_source_counts": count_by(train_rows, "source"),
        "val_source_counts": count_by(val_rows, "source"),
        "notes": [
            "Uses only C-Eval dev/val and CMMLU dev.",
            "Does not use C-Eval test or CMMLU test.",
            "Rows are chat-format SFT examples whose assistant target is exactly one letter: A/B/C/D.",
        ],
    }
    write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
