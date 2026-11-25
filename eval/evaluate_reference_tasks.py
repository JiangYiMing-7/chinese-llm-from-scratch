#!/usr/bin/env python3
"""Evaluate larger Chinese reference-answer language tasks."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
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
from eval.evaluate_generation import cjk_ratio, duplicate_ngram_ratio, max_char_run  # noqa: E402
from eval.reference_metrics import best_reference_score, exact_match  # noqa: E402
from model.generation import generate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reference-answer Chinese language tasks.")
    add_common_model_args(parser)
    parser.add_argument("--suite-name", default="zh_reference_tasks_v1")
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--template", choices=["special", "dialogue"], default="special")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-stop-at-endoftext", action="store_true")
    return parser.parse_args()


def load_cases(path: Path, task_types: list[str], case_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    if task_types:
        requested = set(task_types)
        cases = [case for case in cases if case.get("task_type") in requested]
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case.get("id") in requested]
    if limit is not None:
        cases = cases[:limit]
    return cases


def render_case(case: dict[str, Any], template: str) -> str:
    system = case.get("system", "你是一个中文助手，请只根据用户给出的文本完成任务。")
    user = case["user"]
    if template == "special":
        return f"<|system|>{system}<|user|>{user}<|assistant|>"
    return f"系统：{system}\n用户：{user}\n助手："


@torch.inference_mode()
def run_case(case: dict[str, Any], args: argparse.Namespace, model, sp, device: torch.device) -> dict[str, Any]:
    prompt = render_case(case, args.template)
    prompt_ids = sp.encode(prompt, out_type=int, add_bos=False)
    if not prompt_ids:
        prompt_ids = [sp.bos_id()]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    endoftext_id = sp.piece_to_id("<|endoftext|>")
    stop_id = None if args.no_stop_at_endoftext or endoftext_id == sp.unk_id() else endoftext_id
    max_new_tokens = int(case.get("max_new_tokens", args.max_new_tokens))
    start = time.perf_counter()
    generated = generate(
        model,
        input_ids,
        max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=stop_id,
    )
    elapsed = time.perf_counter() - start
    completion_ids = generated[0, len(prompt_ids):].detach().cpu().tolist()
    completion = sp.decode(completion_ids)
    references = [str(item) for item in case.get("references", []) if str(item).strip()]
    metric = case.get("metric", "char_f1")
    return {
        "id": case["id"],
        "task_type": case.get("task_type", "unknown"),
        "source": case.get("source"),
        "metric": metric,
        "rendered_prompt": prompt,
        "references": references,
        "completion": completion,
        "score": best_reference_score(completion, references, metric),
        "exact_match": exact_match(completion, references),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "generation_sec": elapsed,
        "generation_tokens_per_sec": len(completion_ids) / max(elapsed, 1e-9),
        "char_count": len(completion),
        "cjk_ratio": cjk_ratio(completion),
        "duplicate_4gram_ratio": duplicate_ngram_ratio(completion),
        "max_char_run": max_char_run(completion),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = defaultdict(list)
    by_source = defaultdict(list)
    for row in rows:
        by_task[row["task_type"]].append(row)
        by_source[row.get("source") or "unknown"].append(row)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "examples": len(items),
            "mean_score": mean(item["score"] for item in items) if items else 0.0,
            "exact_match_rate": sum(bool(item["exact_match"]) for item in items) / max(1, len(items)),
            "mean_cjk_ratio": mean(item["cjk_ratio"] for item in items) if items else 0.0,
            "mean_duplicate_4gram_ratio": mean(item["duplicate_4gram_ratio"] for item in items) if items else 0.0,
            "mean_completion_tokens": mean(item["completion_tokens"] for item in items) if items else 0.0,
        }

    return {
        **summarize(rows),
        "tasks": [
            {"task_type": key, **summarize(items)}
            for key, items in sorted(by_task.items())
        ],
        "sources": [
            {"source": key, **summarize(items)}
            for key, items in sorted(by_source.items())
        ],
    }


def write_manual_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id", "task_type", "source", "prompt", "completion", "references",
        "score", "faithfulness_1_5", "relevance_1_5", "format_1_5",
        "usable", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "task_type": row["task_type"],
                "source": row.get("source"),
                "prompt": row["rendered_prompt"],
                "completion": row["completion"],
                "references": " | ".join(row["references"]),
                "score": row["score"],
            })


def write_report(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 中文参考答案语言任务评测报告",
        "",
        "本评测用于较大规模的最终对比，可接入 LCSTS/CMRC/DuReader/自建 JSONL 等参考答案数据。",
        "指标包括精确匹配、字符 F1、ROUGE-L；最终报告仍需抽样人工复核。",
        "",
        "## 总体",
        "",
        "| examples | mean score | exact match | CJK | repeat-4 | tokens |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {summary['examples']} | {summary['mean_score']:.4f} | "
        f"{summary['exact_match_rate']:.4f} | {summary['mean_cjk_ratio']:.3f} | "
        f"{summary['mean_duplicate_4gram_ratio']:.3f} | {summary['mean_completion_tokens']:.1f} |",
        "",
        "## 按任务类型",
        "",
        "| task type | examples | mean score | exact match | repeat-4 | tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task_type']} | {row['examples']} | {row['mean_score']:.4f} | "
            f"{row['exact_match_rate']:.4f} | {row['mean_duplicate_4gram_ratio']:.3f} | "
            f"{row['mean_completion_tokens']:.1f} |"
        )
    lines.extend([
        "",
        "## 按来源",
        "",
        "| source | examples | mean score | exact match |",
        "|---|---:|---:|---:|",
    ])
    for row in summary["sources"]:
        lines.append(
            f"| {row['source']} | {row['examples']} | {row['mean_score']:.4f} | "
            f"{row['exact_match_rate']:.4f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 CPU inference is not supported reliably")
    output_dir = build_output_dir(args.output_root, args.checkpoint, args.suite_name, args.output_dir)
    model, sp, payload = load_model_and_tokenizer(args.checkpoint, args.tokenizer, device, dtype)
    cases = load_cases(args.data_file, args.task_type, args.case_id, args.limit)
    if not cases:
        raise ValueError("no reference task cases selected")
    rows = []
    for case in cases:
        row = run_case(case, args, model, sp, device)
        rows.append(row)
        print(f"[{row['task_type']}/{row['id']}] score={row['score']:.4f} tokens={row['completion_tokens']}")
    summary = aggregate(rows)
    manifest = base_manifest(
        args,
        output_dir,
        payload,
        include_sha256=args.sha256_files,
        extra={
            "suite_name": args.suite_name,
            "data_file": str(args.data_file),
            "template": args.template,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "scoring_note": "reference-answer language-task comparison; combine with manual review",
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "generations.jsonl", rows)
    write_csv(output_dir / "case_scores.csv", [
        {
            "id": row["id"],
            "task_type": row["task_type"],
            "source": row.get("source"),
            "metric": row["metric"],
            "score": row["score"],
            "exact_match": row["exact_match"],
            "completion_tokens": row["completion_tokens"],
            "cjk_ratio": row["cjk_ratio"],
            "duplicate_4gram_ratio": row["duplicate_4gram_ratio"],
        }
        for row in rows
    ])
    write_manual_review(output_dir / "manual_review.csv", rows)
    write_report(output_dir / "report.md", manifest, summary)
    print(f"output_dir: {output_dir}")
    print(f"mean_score: {summary['mean_score']}")


if __name__ == "__main__":
    main()

