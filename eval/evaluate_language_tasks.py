#!/usr/bin/env python3
"""Evaluate Chinese language-workflow tasks for Base/SFT/Safety-SFT comparison."""

from __future__ import annotations

import argparse
import csv
import json
import re
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
from model.generation import generate  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Chinese summarization/extraction/search tasks.")
    add_common_model_args(parser)
    parser.add_argument("--suite-name", default="zh_language_tasks_v1")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--task-type", action="append", default=[])
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--template", choices=["special", "dialogue"], default="special")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--no-stop-at-endoftext", action="store_true")
    return parser.parse_args()


def contains_any(text: str, values: list[str]) -> bool:
    return any(value and value in text for value in values)


def contains_all(text: str, values: list[str]) -> bool:
    return all(value in text for value in values)


def render_case(case: dict[str, Any], template: str) -> str:
    system = case.get("system", "你是一个中文助手，请只根据用户给出的文本完成任务。")
    user = case["user"]
    if template == "special":
        return f"<|system|>{system}<|user|>{user}<|assistant|>"
    return f"系统：{system}\n用户：{user}\n助手："


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


def first_json_object(text: str) -> Any:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def score_case(case: dict[str, Any], completion: str) -> dict[str, Any]:
    required_terms = case.get("required_terms", [])
    forbidden_terms = case.get("forbidden_terms", [])
    any_terms = case.get("any_terms", [])
    json_keys = case.get("json_keys", [])
    max_chars = case.get("max_chars")
    min_chars = case.get("min_chars")
    max_repeat_4 = float(case.get("max_repeat_4", 0.65))

    required_pass = contains_all(completion, required_terms) if required_terms else True
    any_pass = contains_any(completion, any_terms) if any_terms else True
    forbidden_pass = not contains_any(completion, forbidden_terms) if forbidden_terms else True
    max_chars_pass = len(completion.strip()) <= int(max_chars) if max_chars is not None else True
    min_chars_pass = len(completion.strip()) >= int(min_chars) if min_chars is not None else True
    repeat_pass = duplicate_ngram_ratio(completion) <= max_repeat_4
    json_pass = True
    parsed_json = None
    if json_keys:
        parsed_json = first_json_object(completion)
        json_pass = isinstance(parsed_json, dict) and all(key in parsed_json for key in json_keys)

    auto_pass = all([
        required_pass,
        any_pass,
        forbidden_pass,
        max_chars_pass,
        min_chars_pass,
        repeat_pass,
        json_pass,
    ])
    return {
        "required_pass": required_pass,
        "any_pass": any_pass,
        "forbidden_pass": forbidden_pass,
        "max_chars_pass": max_chars_pass,
        "min_chars_pass": min_chars_pass,
        "repeat_pass": repeat_pass,
        "json_pass": json_pass,
        "auto_pass": auto_pass,
        "parsed_json": parsed_json,
    }


@torch.inference_mode()
def run_case(case: dict[str, Any], args: argparse.Namespace, model, sp, device: torch.device) -> dict[str, Any]:
    prompt = render_case(case, args.template)
    prompt_ids = sp.encode(prompt, out_type=int, add_bos=False)
    if not prompt_ids:
        prompt_ids = [sp.bos_id()]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    endoftext_id = sp.piece_to_id("<|endoftext|>")
    stop_id = None if args.no_stop_at_endoftext or endoftext_id == sp.unk_id() else endoftext_id
    start = time.perf_counter()
    generated = generate(
        model,
        input_ids,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=stop_id,
    )
    elapsed = time.perf_counter() - start
    completion_ids = generated[0, len(prompt_ids):].detach().cpu().tolist()
    completion = sp.decode(completion_ids)
    diagnostics = {
        "char_count": len(completion),
        "cjk_ratio": cjk_ratio(completion),
        "duplicate_4gram_ratio": duplicate_ngram_ratio(completion),
        "max_char_run": max_char_run(completion),
    }
    return {
        "id": case["id"],
        "task_type": case.get("task_type", "unknown"),
        "category": case.get("category", case.get("task_type", "unknown")),
        "rendered_prompt": prompt,
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "generation_sec": elapsed,
        "generation_tokens_per_sec": len(completion_ids) / max(elapsed, 1e-9),
        "completion": completion,
        "manual_criteria": case.get("manual_criteria", []),
        **diagnostics,
        **score_case(case, completion),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["task_type"]].append(row)

    def rate(items: list[dict[str, Any]], key: str) -> float:
        return sum(bool(item[key]) for item in items) / max(1, len(items))

    group_rows = []
    for task_type, items in sorted(groups.items()):
        group_rows.append({
            "task_type": task_type,
            "examples": len(items),
            "auto_pass_rate": rate(items, "auto_pass"),
            "required_pass_rate": rate(items, "required_pass"),
            "json_pass_rate": rate(items, "json_pass"),
            "forbidden_pass_rate": rate(items, "forbidden_pass"),
            "repeat_pass_rate": rate(items, "repeat_pass"),
            "mean_cjk_ratio": mean(item["cjk_ratio"] for item in items),
            "mean_duplicate_4gram_ratio": mean(item["duplicate_4gram_ratio"] for item in items),
            "mean_completion_tokens": mean(item["completion_tokens"] for item in items),
        })
    return {
        "examples": len(rows),
        "auto_pass_rate": rate(rows, "auto_pass"),
        "required_pass_rate": rate(rows, "required_pass"),
        "json_pass_rate": rate(rows, "json_pass"),
        "forbidden_pass_rate": rate(rows, "forbidden_pass"),
        "repeat_pass_rate": rate(rows, "repeat_pass"),
        "groups": group_rows,
    }


def write_manual_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id", "task_type", "prompt", "completion", "auto_pass",
        "faithfulness_1_5", "relevance_1_5", "format_1_5", "fluency_1_5",
        "usable", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "task_type": row["task_type"],
                "prompt": row["rendered_prompt"],
                "completion": row["completion"],
                "auto_pass": row["auto_pass"],
            })


def write_report(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 中文语言任务评测报告",
        "",
        "本评测面向报告中的语言类任务：概括、信息提取、查找信息、改写、分类和格式遵守。",
        "自动分数只用于 Base/SFT/Safety-SFT 的相对比较；最终结论需要结合 manual_review.csv 抽样复核。",
        "",
        "## 总体",
        "",
        "| examples | auto pass | required | json | forbidden | repeat |",
        "|---:|---:|---:|---:|---:|---:|",
        f"| {summary['examples']} | {summary['auto_pass_rate']:.1%} | "
        f"{summary['required_pass_rate']:.1%} | {summary['json_pass_rate']:.1%} | "
        f"{summary['forbidden_pass_rate']:.1%} | {summary['repeat_pass_rate']:.1%} |",
        "",
        "## 按任务类型",
        "",
        "| task type | examples | auto pass | required | json | repeat | CJK | repeat-4 | tokens |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        lines.append(
            f"| {row['task_type']} | {row['examples']} | {row['auto_pass_rate']:.1%} | "
            f"{row['required_pass_rate']:.1%} | {row['json_pass_rate']:.1%} | "
            f"{row['repeat_pass_rate']:.1%} | {row['mean_cjk_ratio']:.3f} | "
            f"{row['mean_duplicate_4gram_ratio']:.3f} | {row['mean_completion_tokens']:.1f} |"
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
    cases = load_cases(args.prompt_file, args.task_type, args.case_id, args.limit)
    if not cases:
        raise ValueError("no language task cases selected")
    rows = []
    for case in cases:
        row = run_case(case, args, model, sp, device)
        rows.append(row)
        print(f"[{row['task_type']}/{row['id']}] pass={row['auto_pass']} tokens={row['completion_tokens']}")
    summary = aggregate(rows)
    manifest = base_manifest(
        args,
        output_dir,
        payload,
        include_sha256=args.sha256_files,
        extra={
            "suite_name": args.suite_name,
            "prompt_file": str(args.prompt_file),
            "template": args.template,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "scoring_note": "heuristic language-task comparison; manually review samples before final claims",
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "generations.jsonl", rows)
    write_csv(output_dir / "case_scores.csv", [
        {
            "id": row["id"],
            "task_type": row["task_type"],
            "category": row["category"],
            "auto_pass": row["auto_pass"],
            "required_pass": row["required_pass"],
            "json_pass": row["json_pass"],
            "forbidden_pass": row["forbidden_pass"],
            "repeat_pass": row["repeat_pass"],
            "completion_tokens": row["completion_tokens"],
            "cjk_ratio": row["cjk_ratio"],
            "duplicate_4gram_ratio": row["duplicate_4gram_ratio"],
        }
        for row in rows
    ])
    write_manual_review(output_dir / "manual_review.csv", rows)
    write_report(output_dir / "report.md", manifest, summary)
    print(f"output_dir: {output_dir}")
    print(f"auto_pass_rate: {summary['auto_pass_rate']}")


if __name__ == "__main__":
    main()

