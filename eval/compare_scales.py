#!/usr/bin/env python3
"""Aggregate Tiny/Pilot/Main evaluation outputs into scale-comparison tables."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import ensure_new_output_dir, write_csv, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation outputs across model scales.")
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Model result mapping in name=/path/to/eval_dir form. May be repeated.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_mapping(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--result must use name=/path/to/eval_dir")
    name, path = value.split("=", 1)
    return name, Path(path)


def flatten_result(model_name: str, result_dir: Path) -> list[dict[str, Any]]:
    summary_path = result_dir / "summary.json"
    manifest_path = result_dir / "manifest.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    suite = manifest.get("suite_name") or manifest.get("benchmark") or result_dir.name
    rows: list[dict[str, Any]] = []

    if "internal_validation" in summary:
        row = summary["internal_validation"]
        rows.append({
            "model": model_name,
            "suite": "internal_validation",
            "metric_group": "lm",
            "examples_or_tokens": row.get("tokens"),
            "loss": row.get("loss"),
            "ppl": row.get("ppl"),
            "token_acc": row.get("token_acc"),
            "accuracy": None,
            "bpc": None,
            "bpb": None,
            "source_dir": str(result_dir),
        })
    if "external_lm" in summary:
        row = summary["external_lm"]["summary"]
        rows.append({
            "model": model_name,
            "suite": "external_lm",
            "metric_group": "lm",
            "examples_or_tokens": row.get("tokens"),
            "loss": row.get("loss"),
            "ppl": row.get("ppl"),
            "token_acc": row.get("token_acc"),
            "accuracy": None,
            "bpc": row.get("bpc"),
            "bpb": row.get("bpb"),
            "source_dir": str(result_dir),
        })
    if "accuracy" in summary and "correct" in summary:
        rows.append({
            "model": model_name,
            "suite": suite,
            "metric_group": "mcq",
            "examples_or_tokens": summary.get("examples"),
            "loss": None,
            "ppl": None,
            "token_acc": None,
            "accuracy": summary.get("accuracy"),
            "bpc": None,
            "bpb": None,
            "source_dir": str(result_dir),
        })
    if "groups" in summary and "generations" in summary:
        for group in summary["groups"]:
            rows.append({
                "model": model_name,
                "suite": f"{group['suite']}:{group['category']}:{group['profile']}",
                "metric_group": "generation",
                "examples_or_tokens": group.get("examples"),
                "loss": None,
                "ppl": None,
                "token_acc": None,
                "accuracy": group.get("auto_pass_rate"),
                "bpc": None,
                "bpb": None,
                "mean_cjk_ratio": group.get("mean_cjk_ratio"),
                "mean_repeat_4gram": group.get("mean_duplicate_4gram_ratio"),
                "source_dir": str(result_dir),
            })
    if "groups" in summary and "auto_pass_rate" in summary:
        rows.append({
            "model": model_name,
            "suite": "language_tasks_overall",
            "metric_group": "language_tasks",
            "examples_or_tokens": summary.get("examples"),
            "loss": None,
            "ppl": None,
            "token_acc": None,
            "accuracy": summary.get("auto_pass_rate"),
            "bpc": None,
            "bpb": None,
            "source_dir": str(result_dir),
        })
        for group in summary["groups"]:
            rows.append({
                "model": model_name,
                "suite": f"language_tasks:{group['task_type']}",
                "metric_group": "language_tasks",
                "examples_or_tokens": group.get("examples"),
                "loss": None,
                "ppl": None,
                "token_acc": None,
                "accuracy": group.get("auto_pass_rate"),
                "bpc": None,
                "bpb": None,
                "mean_cjk_ratio": group.get("mean_cjk_ratio"),
                "mean_repeat_4gram": group.get("mean_duplicate_4gram_ratio"),
                "source_dir": str(result_dir),
            })
    if "tasks" in summary and "mean_score" in summary:
        rows.append({
            "model": model_name,
            "suite": "reference_tasks_overall",
            "metric_group": "reference_tasks",
            "examples_or_tokens": summary.get("examples"),
            "loss": None,
            "ppl": None,
            "token_acc": None,
            "accuracy": summary.get("mean_score"),
            "exact_match_rate": summary.get("exact_match_rate"),
            "bpc": None,
            "bpb": None,
            "source_dir": str(result_dir),
        })
        for task in summary["tasks"]:
            rows.append({
                "model": model_name,
                "suite": f"reference_tasks:{task['task_type']}",
                "metric_group": "reference_tasks",
                "examples_or_tokens": task.get("examples"),
                "loss": None,
                "ppl": None,
                "token_acc": None,
                "accuracy": task.get("mean_score"),
                "exact_match_rate": task.get("exact_match_rate"),
                "bpc": None,
                "bpb": None,
                "mean_cjk_ratio": task.get("mean_cjk_ratio"),
                "mean_repeat_4gram": task.get("mean_duplicate_4gram_ratio"),
                "source_dir": str(result_dir),
            })
    if "attack_success_rate" in summary:
        rows.append({
            "model": model_name,
            "suite": "prompt_injection_focus",
            "metric_group": "safety_focus",
            "examples_or_tokens": summary.get("examples"),
            "loss": None,
            "ppl": None,
            "token_acc": None,
            "accuracy": None,
            "attack_success_rate": summary.get("attack_success_rate"),
            "defense_success_rate": summary.get("defense_success_rate"),
            "utility_success_rate": summary.get("utility_success_rate"),
            "bpc": None,
            "bpb": None,
            "source_dir": str(result_dir),
        })
    return rows


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 小/中/大模型评测汇总",
        "",
        f"- generated_at: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`",
        "",
        "| model | suite | group | n | loss | ppl | token acc | accuracy | BPC | BPB | ASR | DSR | Utility |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None:
                return ""
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        lines.append(
            f"| {row['model']} | {row['suite']} | {row['metric_group']} | "
            f"{fmt('examples_or_tokens')} | {fmt('loss')} | {fmt('ppl')} | "
            f"{fmt('token_acc')} | {fmt('accuracy')} | {fmt('bpc')} | {fmt('bpb')} | "
            f"{fmt('attack_success_rate')} | {fmt('defense_success_rate')} | "
            f"{fmt('utility_success_rate')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = ensure_new_output_dir(args.output_dir)
    rows: list[dict[str, Any]] = []
    for item in args.result:
        model_name, result_dir = parse_mapping(item)
        rows.extend(flatten_result(model_name, result_dir))
    write_json(output_dir / "summary.json", {"rows": rows})
    write_csv(output_dir / "scale_comparison.csv", rows)
    write_report(output_dir / "report.md", rows)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
