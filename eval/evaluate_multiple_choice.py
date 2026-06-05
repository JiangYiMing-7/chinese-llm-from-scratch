#!/usr/bin/env python3
"""Conditional log-likelihood evaluation for Chinese multiple-choice tasks."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import (  # noqa: E402
    add_common_model_args,
    base_manifest,
    build_output_dir,
    load_model_and_tokenizer,
    read_jsonl,
    resolve_device,
    resolve_dtype,
    set_seed,
    write_csv,
    write_json,
    write_jsonl,
)


CHOICE_LABELS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate C-Eval, CMMLU, or project JSONL MCQ data."
    )
    add_common_model_args(parser)
    parser.add_argument("--suite-name", default="zh_mcq_v1")
    parser.add_argument(
        "--benchmark",
        choices=["jsonl", "ceval", "cmmlu"],
        required=True,
    )
    parser.add_argument("--data-file", type=Path, default=None, help="JSONL input for --benchmark jsonl")
    parser.add_argument("--data-dir", type=Path, default=None, help="Benchmark root for C-Eval/CMMLU")
    parser.add_argument("--split", default="test", choices=["dev", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--length-normalize", action="store_true")
    parser.add_argument("--few-shot", type=int, default=0, help="Reserved for later fixed-shot runs.")
    return parser.parse_args()


def normalize_answer(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text[0].upper()


def get_first(row: dict[str, Any], names: list[str], default: Any = "") -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def subject_from_csv(path: Path, split: str) -> str:
    stem = path.stem
    for suffix in (f"_{split}", f"-{split}", split):
        if stem.endswith(suffix):
            return stem[: -len(suffix)].strip("_-")
    return stem


def load_official_csv_benchmark(data_dir: Path, split: str) -> list[dict[str, Any]]:
    data_dir = Path(data_dir)
    candidates = []
    split_dir = data_dir / split
    if split_dir.exists():
        candidates.extend(sorted(split_dir.glob("*.csv")))
    candidates.extend(sorted(data_dir.glob(f"*_{split}.csv")))
    candidates.extend(sorted(data_dir.glob(f"*-{split}.csv")))
    seen = set()
    paths = []
    for path in candidates:
        if path not in seen:
            paths.append(path)
            seen.add(path)
    if not paths:
        raise FileNotFoundError(f"no CSV files found for split={split} under {data_dir}")

    examples: list[dict[str, Any]] = []
    for path in paths:
        subject = subject_from_csv(path, split)
        for i, row in enumerate(read_csv_rows(path)):
            choices = {
                label: str(get_first(row, [label, label.lower()], "")).strip()
                for label in CHOICE_LABELS
            }
            question = str(get_first(row, ["question", "Question", "问题"], "")).strip()
            answer = normalize_answer(get_first(row, ["answer", "Answer", "答案"], ""))
            if not question or answer not in CHOICE_LABELS:
                continue
            examples.append({
                "id": str(get_first(row, ["id", "ID"], f"{subject}_{i:05d}")),
                "subject": subject,
                "category": subject,
                "question": question,
                "choices": choices,
                "answer": answer,
            })
    return examples


def load_jsonl_benchmark(path: Path) -> list[dict[str, Any]]:
    examples = []
    for i, row in enumerate(read_jsonl(path)):
        raw_choices = row.get("choices", {})
        if isinstance(raw_choices, list):
            choices = {label: str(raw_choices[j]) for j, label in enumerate(CHOICE_LABELS)}
        else:
            choices = {label: str(raw_choices.get(label, raw_choices.get(label.lower(), ""))) for label in CHOICE_LABELS}
        answer = normalize_answer(row.get("answer"))
        if answer not in CHOICE_LABELS:
            raise ValueError(f"invalid answer in {path}: row {i}")
        examples.append({
            "id": row.get("id", f"{Path(path).stem}_{i:05d}"),
            "subject": row.get("subject", row.get("category", "custom")),
            "category": row.get("category", row.get("subject", "custom")),
            "question": row["question"],
            "choices": choices,
            "answer": answer,
            "source": row.get("source"),
        })
    return examples


def render_prompt(example: dict[str, Any], benchmark: str) -> str:
    subject = example.get("subject", "综合")
    question = example["question"]
    choices = example["choices"]
    if benchmark == "ceval":
        header = f"以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。"
    elif benchmark == "cmmlu":
        header = f"以下是关于{subject}的单项选择题，请直接给出正确答案。"
    else:
        header = "以下是一个中文单项选择题，请选出其中的正确答案。"
    return (
        f"{header}\n\n"
        f"{question}\n"
        f"A. {choices['A']}\n"
        f"B. {choices['B']}\n"
        f"C. {choices['C']}\n"
        f"D. {choices['D']}\n"
        f"答案："
    )


@torch.inference_mode()
def score_continuation(
    model,
    sp,
    prompt: str,
    continuation: str,
    device: torch.device,
    length_normalize: bool,
) -> dict[str, Any]:
    context_ids = sp.encode(prompt, out_type=int, add_bos=False)
    continuation_ids = sp.encode(continuation, out_type=int, add_bos=False)
    if not context_ids:
        context_ids = [sp.bos_id()]
    if not continuation_ids:
        raise ValueError("empty continuation after tokenization")

    max_full = model.config.max_seq_len + 1
    full = context_ids + continuation_ids
    if len(full) > max_full:
        overflow = len(full) - max_full
        if overflow >= len(context_ids):
            raise ValueError("continuation is too long for model context")
        context_ids = context_ids[overflow:]
        full = context_ids + continuation_ids
    start_label = len(context_ids) - 1

    input_ids = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    labels = torch.tensor(full[1:], dtype=torch.long, device=device)
    output = model(input_ids)
    log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
    positions = torch.arange(start_label, start_label + len(continuation_ids), device=device)
    target_ids = labels[positions]
    token_logps = log_probs[positions, target_ids]
    raw_score = float(token_logps.sum().detach().cpu())
    score = raw_score / len(continuation_ids) if length_normalize else raw_score
    return {
        "score": score,
        "raw_logprob": raw_score,
        "tokens": len(continuation_ids),
    }


@torch.inference_mode()
def evaluate_examples(
    model,
    sp,
    examples: list[dict[str, Any]],
    benchmark: str,
    device: torch.device,
    length_normalize: bool,
) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        prompt = render_prompt(example, benchmark)
        scores = {}
        raw_scores = {}
        token_counts = {}
        for label in CHOICE_LABELS:
            scored = score_continuation(model, sp, prompt, label, device, length_normalize)
            scores[label] = scored["score"]
            raw_scores[label] = scored["raw_logprob"]
            token_counts[label] = scored["tokens"]
        prediction = max(scores, key=scores.get)
        rows.append({
            "id": example["id"],
            "subject": example["subject"],
            "category": example.get("category", example["subject"]),
            "answer": example["answer"],
            "prediction": prediction,
            "correct": prediction == example["answer"],
            "prompt": prompt,
            "scores": scores,
            "raw_logprobs": raw_scores,
            "choice_token_counts": token_counts,
        })
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    subjects = defaultdict(list)
    categories = defaultdict(list)
    for row in rows:
        subjects[row["subject"]].append(row)
        categories[row["category"]].append(row)

    def summarize_group(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(group_rows),
            "correct": sum(1 for row in group_rows if row["correct"]),
            "accuracy": sum(1 for row in group_rows if row["correct"]) / max(1, len(group_rows)),
        }

    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / max(1, total),
        "subjects": [
            {"subject": subject, **summarize_group(group_rows)}
            for subject, group_rows in sorted(subjects.items())
        ],
        "categories": [
            {"category": category, **summarize_group(group_rows)}
            for category, group_rows in sorted(categories.items())
        ],
    }


def write_report(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 中文多选评测报告",
        "",
        "## 配置",
        "",
        f"- benchmark: `{manifest['benchmark']}`",
        f"- split: `{manifest['split']}`",
        f"- checkpoint: `{manifest['checkpoint']['path']}`",
        f"- dtype/device: `{manifest['dtype']}` / `{manifest['device']}`",
        f"- length normalize: `{manifest['length_normalize']}`",
        "",
        "## 总体结果",
        "",
        "| examples | correct | accuracy |",
        "|---:|---:|---:|",
        f"| {summary['examples']} | {summary['correct']} | {summary['accuracy']:.4f} |",
        "",
        "## 分学科",
        "",
        "| subject | examples | accuracy |",
        "|---|---:|---:|",
    ]
    for row in summary["subjects"]:
        lines.append(f"| {row['subject']} | {row['examples']} | {row['accuracy']:.4f} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.few_shot != 0:
        raise ValueError("few-shot runs are reserved for a later fixed-template pass; use --few-shot 0")
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is not supported reliably")
    output_dir = build_output_dir(
        args.output_root,
        args.checkpoint,
        f"{args.benchmark}_{args.split}",
        args.output_dir,
    )
    model, sp, payload = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device, dtype)

    if args.benchmark == "jsonl":
        if not args.data_file:
            raise ValueError("--data-file is required for --benchmark jsonl")
        examples = load_jsonl_benchmark(args.data_file)
    else:
        if not args.data_dir:
            raise ValueError("--data-dir is required for official benchmark adapters")
        examples = load_official_csv_benchmark(args.data_dir, args.split)
    if args.subject:
        requested = set(args.subject)
        examples = [ex for ex in examples if ex["subject"] in requested]
    if args.limit is not None:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError("no examples selected")

    start = time.perf_counter()
    rows = evaluate_examples(
        model,
        sp,
        examples,
        args.benchmark,
        device,
        args.length_normalize,
    )
    elapsed = time.perf_counter() - start
    summary = aggregate(rows)
    summary["elapsed_sec"] = elapsed
    summary["examples_per_sec"] = len(rows) / max(elapsed, 1e-9)

    flat_rows = []
    for row in rows:
        flat_rows.append({
            "id": row["id"],
            "subject": row["subject"],
            "category": row["category"],
            "answer": row["answer"],
            "prediction": row["prediction"],
            "correct": row["correct"],
            **{f"score_{label}": row["scores"][label] for label in CHOICE_LABELS},
        })

    manifest = base_manifest(
        args,
        output_dir,
        payload,
        include_sha256=args.sha256_files,
        extra={
            "suite_name": args.suite_name,
            "benchmark": args.benchmark,
            "split": args.split,
            "data_file": str(args.data_file) if args.data_file else None,
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "length_normalize": args.length_normalize,
            "scoring": "conditional log-likelihood over answer labels A/B/C/D",
            "language": "Chinese",
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_example.jsonl", rows)
    write_csv(output_dir / "per_example.csv", flat_rows)
    write_csv(output_dir / "subjects.csv", summary["subjects"])
    write_report(output_dir / "report.md", manifest, summary)
    print(f"output_dir: {output_dir}")
    print(f"accuracy: {summary['accuracy']:.4f}")


if __name__ == "__main__":
    main()
