#!/usr/bin/env python3
"""Audit raw posttraining/evaluation sources for the 500M v2 flow."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any


CORE_SOURCES = {
    "cmrc2018": ["cmrc2018/data/cmrc2018_train.json", "cmrc2018/data/cmrc2018_dev.json"],
    "coig_cqia": ["coig_cqia"],
    "belle_train_1m_cn": ["belle_train_1m_cn/Belle_open_source_1M.json"],
    "cluener2020": ["cluener2020/train.json", "cluener2020/dev.json"],
    "pclue": ["pclue/datasets/pCLUE_dev.json"],
    "lcsts": ["lcsts/train.jsonl", "lcsts/valid.jsonl", "lcsts/test_public.jsonl"],
    "duie": ["duie/duie_train.json.zip", "duie/duie_dev.json.zip"],
}

OFFICIAL_EVAL_SOURCES = {
    "clue": "data/eval/clue",
    "fewclue": "data/eval/fewclue",
    "ceval_exam": "data/eval/ceval-exam",
    "cmmlu": "data/eval/CMMLU/data",
}


def count_lines(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def count_zip_json_lines(path: Path) -> int:
    total = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith("/") or not name.endswith((".json", ".jsonl")):
                    continue
                with archive.open(name) as f:
                    total += sum(1 for line in f if line.strip())
    except (OSError, zipfile.BadZipFile):
        return 0
    return total


def sample_count(path: Path) -> int | None:
    if path.is_dir():
        return None
    if path.suffix == ".zip":
        return count_zip_json_lines(path)
    if path.suffix in {".jsonl", ".json"}:
        return count_lines(path)
    return None


def file_entry(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": stat.st_size,
        "sample_count_hint": sample_count(path),
    }


def audit_core_source(raw_root: Path, name: str, expected: list[str]) -> dict[str, Any]:
    files = []
    existing_expected = []
    for rel in expected:
        path = raw_root / rel
        if path.exists():
            existing_expected.append(rel)
    root = raw_root / name
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and ".git" not in path.parts and not path.name.startswith("._"):
                files.append(file_entry(path, raw_root))
    return {
        "present": bool(existing_expected) or root.exists(),
        "expected_present": existing_expected,
        "expected_missing": [rel for rel in expected if not (raw_root / rel).exists()],
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "line_or_json_record_hint": sum(
            int(item["sample_count_hint"] or 0)
            for item in files
            if item["sample_count_hint"] is not None
        ),
        "files_preview": files[:30],
    }


def audit_eval_root(project_root: Path, rel: str) -> dict[str, Any]:
    root = project_root / rel
    files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.name.startswith("._"):
                files.append(file_entry(path, project_root))
    return {
        "path": rel,
        "present": root.exists(),
        "file_count": len(files),
        "size_bytes": sum(int(item["size_bytes"]) for item in files),
        "files_preview": files[:30],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 500M v2 Raw Source Audit",
        "",
        f"- raw_root: `{report['raw_root']}`",
        f"- project_root: `{report['project_root']}`",
        "",
        "## Core SFT / Reference Sources",
        "",
        "| source | present | files | size MB | record hint | missing expected |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in report["core_sources"].items():
        size_mb = item["size_bytes"] / 1024 / 1024
        missing = ", ".join(item["expected_missing"]) if item["expected_missing"] else ""
        lines.append(
            f"| {name} | {item['present']} | {item['file_count']} | "
            f"{size_mb:.1f} | {item['line_or_json_record_hint']} | {missing} |"
        )
    lines.extend([
        "",
        "## Official / Boundary Eval Sources",
        "",
        "| source | present | files | size MB | path |",
        "|---|---:|---:|---:|---|",
    ])
    for name, item in report["official_eval_sources"].items():
        size_mb = item["size_bytes"] / 1024 / 1024
        lines.append(f"| {name} | {item['present']} | {item['file_count']} | {size_mb:.1f} | {item['path']} |")
    if report["missing_core_sources"]:
        lines.extend(["", "## Blocking Missing Core Sources", ""])
        lines.extend(f"- {name}" for name in report["missing_core_sources"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit 500M v2 raw SFT/eval sources.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/sft_sources/raw"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data/reports"))
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any core source is missing.")
    args = parser.parse_args()

    raw_root = args.raw_root
    project_root = args.project_root
    report = {
        "raw_root": str(raw_root),
        "project_root": str(project_root),
        "core_sources": {
            name: audit_core_source(raw_root, name, expected)
            for name, expected in CORE_SOURCES.items()
        },
        "official_eval_sources": {
            name: audit_eval_root(project_root, rel)
            for name, rel in OFFICIAL_EVAL_SOURCES.items()
        },
    }
    report["missing_core_sources"] = [
        name for name, item in report["core_sources"].items() if not item["present"]
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "sft_sources_v2_audit.json"
    md_path = args.output_dir / "sft_sources_v2_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "missing": report["missing_core_sources"]}, ensure_ascii=False, indent=2))
    if args.strict and report["missing_core_sources"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
