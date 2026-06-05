#!/usr/bin/env python3
"""Create a compact Markdown/CSV index for one automated eval run root."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_row(summary_path: Path, root: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    manifest_path = summary_path.parent / "manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    rel = summary_path.parent.relative_to(root).as_posix()
    checkpoint_id = manifest.get("checkpoint_id", rel.split("/")[0])
    suite = manifest.get("suite_name") or manifest.get("benchmark") or summary_path.parent.name
    row: dict[str, Any] = {
        "checkpoint_id": checkpoint_id,
        "suite": suite,
        "result_dir": rel,
        "examples": None,
        "tokens": None,
        "loss": None,
        "ppl": None,
        "token_acc": None,
        "accuracy": None,
        "asr": None,
        "dsr": None,
        "utility": None,
    }
    if "internal_validation" in summary:
        v = summary["internal_validation"]
        row.update({
            "suite": "internal_validation",
            "tokens": v.get("tokens"),
            "loss": v.get("loss"),
            "ppl": v.get("ppl"),
            "token_acc": v.get("token_acc"),
        })
    elif "accuracy" in summary and "correct" in summary:
        row.update({
            "examples": summary.get("examples"),
            "accuracy": summary.get("accuracy"),
        })
    elif "attack_success_rate" in summary:
        row.update({
            "examples": summary.get("examples"),
            "asr": summary.get("attack_success_rate"),
            "dsr": summary.get("defense_success_rate"),
            "utility": summary.get("utility_success_rate"),
        })
    elif "generations" in summary:
        row.update({
            "examples": summary.get("generations"),
            "suite": suite,
        })
    return row


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python eval/summarize_eval_run.py RESULTS_ROOT")
    root = Path(sys.argv[1])
    rows = [extract_row(path, root) for path in sorted(root.glob("*/*/summary.json"))]
    csv_path = root / "summary_index.csv"
    fields = [
        "checkpoint_id", "suite", "examples", "tokens", "loss", "ppl",
        "token_acc", "accuracy", "asr", "dsr", "utility", "result_dir",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})

    lines = [
        "# Evaluation Run Index",
        "",
        f"- root: `{root}`",
        f"- rows: `{len(rows)}`",
        "",
        "| checkpoint | suite | n/examples | tokens | loss | ppl | token acc | acc | ASR | DSR | Utility |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint_id']} | {row['suite']} | {fmt(row['examples'])} | "
            f"{fmt(row['tokens'])} | {fmt(row['loss'])} | {fmt(row['ppl'])} | "
            f"{fmt(row['token_acc'])} | {fmt(row['accuracy'])} | {fmt(row['asr'])} | "
            f"{fmt(row['dsr'])} | {fmt(row['utility'])} |"
        )
    report = "\n".join(lines)
    (root / "summary_index.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

