#!/usr/bin/env python3
"""Evaluate frozen Chinese generation, instruction sanity, and probe prompts."""

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
from model.generation import generate  # noqa: E402


PROFILES = {
    "greedy": {"temperature": 0.0, "top_k": None, "seed_offset": 0},
    "balanced_s1": {"temperature": 0.7, "top_k": 40, "seed_offset": 1},
    "balanced_s2": {"temperature": 0.7, "top_k": 40, "seed_offset": 2},
}
REFUSAL_TERMS = ["不能", "无法", "不可以", "抱歉", "不提供", "不建议"]
SUPPORT_TERMS = ["建议", "可以", "步骤", "首先", "其次", "例如", "总结"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Chinese generation prompts.")
    add_common_model_args(parser)
    parser.add_argument("--suite-name", default="zh_generation_v1")
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--suites", nargs="*", default=[])
    parser.add_argument("--profiles", nargs="+", choices=sorted(PROFILES), default=["greedy"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--template", choices=["auto", "plain", "dialogue", "special"], default="auto")
    parser.add_argument("--add-bos", action="store_true")
    parser.add_argument("--no-stop-at-endoftext", action="store_true")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_cases(path: Path, suites: list[str], case_ids: list[str], limit: int | None) -> list[dict[str, Any]]:
    cases = read_jsonl(path)
    if suites:
        cases = [case for case in cases if case.get("suite") in suites]
    if case_ids:
        requested = set(case_ids)
        cases = [case for case in cases if case.get("id") in requested]
    if limit is not None:
        cases = cases[:limit]
    return cases


def render_prompt(case: dict[str, Any], template: str) -> str:
    prompt_type = case.get("prompt_type", "completion")
    prompt = case.get("prompt", "")
    if prompt_type == "completion":
        return prompt
    chosen = "dialogue" if template == "auto" else template
    if chosen == "plain":
        return prompt
    if chosen == "special":
        system = case.get("system")
        if system:
            return f"<|system|>{system}<|user|>{prompt}<|assistant|>"
        return f"<|user|>{prompt}<|assistant|>"
    return f"用户：{prompt}\n助手："


def cjk_ratio(text: str) -> float:
    content = [ch for ch in text if not ch.isspace()]
    if not content:
        return 0.0
    return sum(1 for ch in content if "\u3400" <= ch <= "\u9fff") / len(content)


def duplicate_ngram_ratio(text: str, n: int = 4) -> float:
    compact = re.sub(r"\s+", "", text)
    if len(compact) < n:
        return 0.0
    ngrams = [compact[i:i + n] for i in range(len(compact) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def max_char_run(text: str) -> int:
    longest = 0
    current = 0
    previous = None
    for ch in text:
        current = current + 1 if ch == previous else 1
        previous = ch
        longest = max(longest, current)
    return longest


def contains_any(text: str, values: list[str]) -> bool:
    return any(value in text for value in values)


def evaluate_check(check: dict[str, Any], text: str, diagnostics: dict[str, Any]) -> bool:
    kind = check["type"]
    if kind == "contains":
        return check["value"] in text
    if kind == "contains_any":
        return contains_any(text, check["values"])
    if kind == "contains_all":
        return all(value in text for value in check["values"])
    if kind == "not_contains":
        return check["value"] not in text
    if kind == "max_chars":
        return len(text.strip()) <= int(check["value"])
    if kind == "min_chars":
        return len(text.strip()) >= int(check["value"])
    if kind == "min_cjk_ratio":
        return diagnostics["cjk_ratio"] >= float(check["value"])
    if kind == "line_count_at_least":
        return len([line for line in text.splitlines() if line.strip()]) >= int(check["value"])
    if kind == "valid_json":
        try:
            json.loads(text.strip())
            return True
        except json.JSONDecodeError:
            return False
    raise ValueError(f"unknown check type: {kind}")


def diagnose(case: dict[str, Any], completion: str) -> dict[str, Any]:
    diagnostics = {
        "char_count": len(completion),
        "cjk_ratio": cjk_ratio(completion),
        "duplicate_4gram_ratio": duplicate_ngram_ratio(completion),
        "max_char_run": max_char_run(completion),
        "refusal_detected": contains_any(completion, REFUSAL_TERMS),
        "helpful_signal_detected": contains_any(completion, SUPPORT_TERMS),
    }
    check_rows = []
    for check in case.get("checks", []):
        passed = evaluate_check(check, completion, diagnostics)
        check_rows.append({"definition": check, "passed": passed})
    hard_pass = all(row["passed"] for row in check_rows) if check_rows else None
    return {**diagnostics, "checks": check_rows, "auto_pass": hard_pass}


@torch.inference_mode()
def run_case(case: dict[str, Any], profile_name: str, args: argparse.Namespace, model, sp, device: torch.device) -> dict[str, Any]:
    profile = PROFILES[profile_name]
    rendered = render_prompt(case, args.template)
    prompt_ids = sp.encode(rendered, out_type=int, add_bos=args.add_bos)
    if not prompt_ids:
        prompt_ids = [sp.bos_id()]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    stop_id = None
    endoftext_id = sp.piece_to_id("<|endoftext|>")
    if not args.no_stop_at_endoftext and endoftext_id != sp.unk_id():
        stop_id = endoftext_id
    run_seed = args.seed + profile["seed_offset"]
    torch.manual_seed(run_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_seed)
    start = time.perf_counter()
    generated = generate(
        model,
        input_ids,
        args.max_new_tokens,
        temperature=profile["temperature"],
        top_k=profile["top_k"],
        eos_token_id=stop_id,
    )
    elapsed = time.perf_counter() - start
    completion_ids = generated[0, len(prompt_ids):].detach().cpu().tolist()
    completion = sp.decode(completion_ids)
    return {
        "id": case["id"],
        "suite": case.get("suite", "generation"),
        "category": case.get("category", "general"),
        "prompt_type": case.get("prompt_type", "completion"),
        "prompt": case.get("prompt", ""),
        "rendered_prompt": rendered,
        "profile": profile_name,
        "seed": run_seed,
        "temperature": profile["temperature"],
        "top_k": profile["top_k"],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "generation_sec": elapsed,
        "generation_tokens_per_sec": len(completion_ids) / max(elapsed, 1e-9),
        "manual_criteria": case.get("manual_criteria", []),
        "completion": completion,
        **diagnose(case, completion),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["category"], row["profile"])].append(row)
    group_rows = []
    for (suite, category, profile), items in sorted(groups.items()):
        scored = [item["auto_pass"] for item in items if item["auto_pass"] is not None]
        group_rows.append({
            "suite": suite,
            "category": category,
            "profile": profile,
            "examples": len(items),
            "auto_scored": len(scored),
            "auto_pass_rate": sum(bool(x) for x in scored) / len(scored) if scored else None,
            "mean_cjk_ratio": mean(item["cjk_ratio"] for item in items),
            "mean_duplicate_4gram_ratio": mean(item["duplicate_4gram_ratio"] for item in items),
            "mean_completion_tokens": mean(item["completion_tokens"] for item in items),
            "mean_tokens_per_sec": mean(item["generation_tokens_per_sec"] for item in items),
        })
    return {
        "generations": len(rows),
        "groups": group_rows,
    }


def write_manual_review(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "id", "suite", "category", "profile", "prompt", "completion",
        "fluency_1_5", "relevance_1_5", "coherence_1_5",
        "factual_or_logic_issue", "repetition_issue", "overall_1_5", "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row["profile"] != "greedy":
                continue
            writer.writerow({
                "id": row["id"],
                "suite": row["suite"],
                "category": row["category"],
                "profile": row["profile"],
                "prompt": row["rendered_prompt"],
                "completion": row["completion"],
            })


def write_report(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 中文生成与指令小测报告",
        "",
        "## 配置",
        "",
        f"- prompt file: `{manifest['prompt_file']}`",
        f"- checkpoint: `{manifest['checkpoint']['path']}`",
        f"- profiles: `{', '.join(manifest['profiles'])}`",
        "",
        "## 分组汇总",
        "",
        "| suite | category | profile | examples | auto pass | CJK | repeat-4 | tokens | tok/s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["groups"]:
        rate = row["auto_pass_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.1%}"
        lines.append(
            f"| {row['suite']} | {row['category']} | {row['profile']} | "
            f"{row['examples']} | {rate_text} | {row['mean_cjk_ratio']:.3f} | "
            f"{row['mean_duplicate_4gram_ratio']:.3f} | "
            f"{row['mean_completion_tokens']:.1f} | {row['mean_tokens_per_sec']:.2f} |"
        )
    lines.extend([
        "",
        "完整生成内容保存在 `generations.jsonl`；人工评分请填写 `manual_review.csv`。",
        "",
    ])
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
    cases = load_cases(args.prompt_file, args.suites, args.case_id, args.limit)
    if not cases:
        raise ValueError("no prompt cases selected")
    rows = []
    for case in cases:
        for profile in args.profiles:
            row = run_case(case, profile, args, model, sp, device)
            rows.append(row)
            print(f"[{row['suite']}/{row['id']}/{profile}] tokens={row['completion_tokens']}")
    summary = aggregate(rows)
    manifest = base_manifest(
        args,
        output_dir,
        payload,
        include_sha256=args.sha256_files,
        extra={
            "suite_name": args.suite_name,
            "prompt_file": str(args.prompt_file),
            "profiles": args.profiles,
            "max_new_tokens": args.max_new_tokens,
            "template": args.template,
            "language": "Chinese primary; multilingual probes are appendix only",
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "generations.jsonl", rows)
    write_csv(output_dir / "case_scores.csv", [
        {
            "id": row["id"],
            "suite": row["suite"],
            "category": row["category"],
            "profile": row["profile"],
            "auto_pass": row["auto_pass"],
            "completion_tokens": row["completion_tokens"],
            "cjk_ratio": row["cjk_ratio"],
            "duplicate_4gram_ratio": row["duplicate_4gram_ratio"],
            "max_char_run": row["max_char_run"],
            "generation_tokens_per_sec": row["generation_tokens_per_sec"],
        }
        for row in rows
    ])
    write_manual_review(output_dir / "manual_review.csv", rows)
    write_report(output_dir / "report.md", manifest, summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()

