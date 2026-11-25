#!/usr/bin/env python3
"""Prepare C-Eval CSV split directories from the current Hugging Face dataset.

The original C-Eval README used to point at a single ceval-exam.zip file.
The Hugging Face dataset now exposes subject folders with parquet shards, so
this helper exports the data back to the CSV layout expected by this repo:

  data/eval/ceval-exam/{dev,val,test}/{subject}_{split}.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable


SPLITS = ("dev", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/eval/ceval-exam"))
    parser.add_argument("--hf-repo", default="ceval/ceval-exam")
    parser.add_argument(
        "--local-hf-root",
        type=Path,
        default=None,
        help="Optional local clone/download of the Hugging Face dataset repo.",
    )
    parser.add_argument("--subject", action="append", default=[])
    parser.add_argument("--split", action="append", choices=SPLITS, default=[])
    return parser.parse_args()


def need_pandas():
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to export C-Eval CSV files. Install with: "
            "python -m pip install pandas pyarrow datasets huggingface_hub"
        ) from exc
    return pd


def normalize_columns(df):
    required = ["question", "A", "B", "C", "D", "answer"]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if "id" not in df.columns:
        df.insert(0, "id", range(len(df)))
    ordered = [name for name in ["id", "question", "A", "B", "C", "D", "answer", "explanation"] if name in df.columns]
    rest = [name for name in df.columns if name not in ordered]
    return df[ordered + rest]


def write_split_csv(df, output: Path, subject: str, split: str) -> Path:
    output_dir = output / split
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{subject}_{split}.csv"
    normalize_columns(df).to_csv(path, index=False, encoding="utf-8")
    return path


def selected(items: Iterable[str], allow: set[str]) -> list[str]:
    values = list(items)
    if not allow:
        return values
    return [item for item in values if item in allow]


def export_from_local_parquet(local_root: Path, output: Path, subjects: set[str], splits: set[str]) -> int:
    pd = need_pandas()
    if not local_root.exists():
        raise FileNotFoundError(local_root)

    written = 0
    for path in sorted(local_root.rglob("*.parquet")):
        split = None
        match = re.match(r"^(dev|val|test)-", path.name)
        if match:
            split = match.group(1)
            subject = path.parent.name
        elif path.parent.name in SPLITS:
            split = path.parent.name
            subject = path.stem
            for suffix in (f"_{split}", f"-{split}", split):
                if subject.endswith(suffix):
                    subject = subject[: -len(suffix)].strip("_-")
                    break
        else:
            continue
        if splits and split not in splits:
            continue
        if subjects and subject not in subjects:
            continue
        df = pd.read_parquet(path)
        out_path = write_split_csv(df, output, subject, split)
        print(f"[write] {out_path} rows={len(df)}")
        written += 1
    if written == 0:
        raise SystemExit(f"no C-Eval parquet files found under {local_root}")
    return written


def export_from_datasets(hf_repo: str, output: Path, subjects: set[str], splits: set[str]) -> int:
    try:
        from datasets import get_dataset_config_names, load_dataset  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "datasets is required for direct C-Eval download. Install with: "
            "python -m pip install datasets pandas pyarrow"
        ) from exc

    configs = selected(get_dataset_config_names(hf_repo), subjects)
    wanted_splits = splits or set(SPLITS)
    written = 0
    for subject in configs:
        ds = load_dataset(hf_repo, name=subject)
        for split in SPLITS:
            if split not in wanted_splits or split not in ds:
                continue
            df = ds[split].to_pandas()
            out_path = write_split_csv(df, output, subject, split)
            print(f"[write] {out_path} rows={len(df)}")
            written += 1
    if written == 0:
        raise SystemExit(f"no C-Eval splits exported from {hf_repo}")
    return written


def main() -> None:
    args = parse_args()
    subjects = set(args.subject)
    splits = set(args.split)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.local_hf_root is not None:
        written = export_from_local_parquet(args.local_hf_root, args.output, subjects, splits)
    else:
        written = export_from_datasets(args.hf_repo, args.output, subjects, splits)
    print(f"[done] exported {written} C-Eval CSV files to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise
