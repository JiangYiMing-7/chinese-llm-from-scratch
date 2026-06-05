#!/usr/bin/env python3
"""Shared helpers for reproducible Chinese model evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.causal_lm import CausalLM  # noqa: E402
from model.config import ModelConfig  # noqa: E402


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def checkpoint_id(path: Path) -> str:
    path = Path(path)
    parts = path.parts
    for marker in ("runs", "run"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return f"{parts[idx + 1]}_{path.stem}"
    return path.stem


def ensure_new_output_dir(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to reuse existing evaluation output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    return output_dir


def build_output_dir(
    output_root: Path,
    checkpoint: Path,
    suite_name: str,
    explicit_output_dir: Path | None = None,
) -> Path:
    if explicit_output_dir is not None:
        return ensure_new_output_dir(explicit_output_dir)
    return ensure_new_output_dir(
        Path(output_root) / checkpoint_id(checkpoint) / f"{suite_name}_{timestamp()}"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path, include_sha256: bool = False) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(stat.st_mtime)),
    }
    if include_sha256:
        info["sha256"] = sha256_file(path)
    return info


def git_info(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            proc = subprocess.run(
                args,
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run(["git", "status", "--short"])
    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def load_sentencepiece(tokenizer_path: Path):
    try:
        import sentencepiece as spm
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise RuntimeError("sentencepiece is required for model evaluation") from exc
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def load_model_and_tokenizer(
    checkpoint: Path,
    tokenizer_path: Path,
    device: torch.device,
    dtype: torch.dtype,
):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = payload.get("config")
    if not cfg or "model" not in cfg:
        raise ValueError("checkpoint does not contain config.model")
    model = CausalLM(ModelConfig(**cfg["model"]))
    model.load_state_dict(payload["model"])
    model = model.to(device=device, dtype=dtype)
    model.eval()
    sp = load_sentencepiece(tokenizer_path)
    if sp.get_piece_size() != model.config.vocab_size:
        raise ValueError(
            f"tokenizer/model vocab mismatch: {sp.get_piece_size()} != "
            f"{model.config.vocab_size}"
        )
    return model, sp, payload


def write_json(path: Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def safe_exp(value: float) -> float:
    return math.exp(value) if value < 700 else float("inf")


def bpc(nll: float, char_count: int) -> float | None:
    if char_count <= 0:
        return None
    return nll / (char_count * math.log(2))


def bpb(nll: float, byte_count: int) -> float | None:
    if byte_count <= 0:
        return None
    return nll / (byte_count * math.log(2))


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def base_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    checkpoint_payload: dict[str, Any] | None = None,
    include_sha256: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint)
    tokenizer = Path(args.tokenizer)
    manifest: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "checkpoint": file_info(checkpoint, include_sha256=include_sha256),
        "checkpoint_id": checkpoint_id(checkpoint),
        "tokenizer": file_info(tokenizer, include_sha256=include_sha256),
        "device": str(getattr(args, "device", None)),
        "dtype": getattr(args, "dtype", None),
        "seed": getattr(args, "seed", None),
        "git": git_info(),
    }
    if checkpoint_payload is not None:
        manifest["checkpoint_step"] = checkpoint_payload.get("step")
        if "tokens_seen" in checkpoint_payload:
            manifest["checkpoint_tokens_seen"] = checkpoint_payload.get("tokens_seen")
        cfg = checkpoint_payload.get("config")
        if cfg:
            manifest["checkpoint_config"] = cfg
    if extra:
        manifest.update(extra)
    return manifest


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("results/evals"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sha256-files", action="store_true")

