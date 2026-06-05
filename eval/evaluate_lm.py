#!/usr/bin/env python3
"""Evaluate language-modeling metrics on internal and external Chinese data."""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eval.common import (  # noqa: E402
    add_common_model_args,
    base_manifest,
    bpb,
    bpc,
    build_output_dir,
    load_model_and_tokenizer,
    read_jsonl,
    resolve_device,
    resolve_dtype,
    safe_exp,
    set_seed,
    write_csv,
    write_json,
    write_jsonl,
)
from pretrain.dataset import TokenDataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate internal validation and external Chinese LM metrics."
    )
    add_common_model_args(parser)
    parser.add_argument("--suite-name", default="zh_lm_v1")
    parser.add_argument("--validation-bin", type=Path, default=None)
    parser.add_argument("--validation-block-size", type=int, default=1024)
    parser.add_argument("--validation-batch-size", type=int, default=16)
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=None,
        help="Limit validation batches. Omit for full validation.",
    )
    parser.add_argument(
        "--external-jsonl",
        action="append",
        default=[],
        type=Path,
        help="External JSONL with fields id, domain, text, optional source/license.",
    )
    parser.add_argument(
        "--external-text-dir",
        action="append",
        default=[],
        type=Path,
        help="Directory of .txt files. Parent folder name is used as domain.",
    )
    parser.add_argument("--external-limit", type=int, default=None)
    return parser.parse_args()


@torch.inference_mode()
def internal_validation_metrics(
    model,
    token_path: Path,
    block_size: int,
    batch_size: int,
    max_batches: int | None,
    device: torch.device,
) -> dict[str, Any]:
    dataset = TokenDataset(token_path, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0
    start = time.perf_counter()
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        output = model(input_ids, labels=labels)
        n_tokens = labels.numel()
        total_nll += float(output.loss.detach().float().cpu()) * n_tokens
        total_correct += int((output.logits.argmax(dim=-1) == labels).sum().item())
        total_tokens += n_tokens
        batches += 1
    loss = total_nll / max(1, total_tokens)
    elapsed = time.perf_counter() - start
    return {
        "name": "internal_validation",
        "token_file": str(token_path),
        "block_size": block_size,
        "batch_size": batch_size,
        "requested_batches": max_batches,
        "batches": batches,
        "tokens": total_tokens,
        "nll": total_nll,
        "loss": loss,
        "ppl": safe_exp(loss),
        "token_acc": total_correct / max(1, total_tokens),
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-9),
    }


def load_external_records(jsonl_paths: list[Path], text_dirs: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in jsonl_paths:
        for i, row in enumerate(read_jsonl(path)):
            text = str(row.get("text", ""))
            if not text.strip():
                continue
            records.append({
                "id": row.get("id", f"{path.stem}_{i:06d}"),
                "domain": row.get("domain", "unknown"),
                "text": text,
                "source": row.get("source"),
                "license": row.get("license"),
                "path": str(path),
            })
    for directory in text_dirs:
        directory = Path(directory)
        for path in sorted(directory.rglob("*.txt")):
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                continue
            domain = path.parent.name if path.parent != directory else "external_text"
            records.append({
                "id": path.relative_to(directory).as_posix(),
                "domain": domain,
                "text": text,
                "source": str(path),
                "license": None,
                "path": str(path),
            })
    return records


@torch.inference_mode()
def score_token_ids(model, ids: list[int], device: torch.device) -> dict[str, Any]:
    max_seq_len = model.config.max_seq_len
    if len(ids) < 2:
        return {"tokens": 0, "nll": 0.0, "correct": 0}
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    for start in range(0, len(ids) - 1, max_seq_len):
        chunk = ids[start:start + max_seq_len + 1]
        if len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        labels = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        output = model(input_ids, labels=labels)
        n_tokens = labels.numel()
        total_nll += float(output.loss.detach().float().cpu()) * n_tokens
        total_correct += int((output.logits.argmax(dim=-1) == labels).sum().item())
        total_tokens += n_tokens
    return {"tokens": total_tokens, "nll": total_nll, "correct": total_correct}


def summarize_lm_counts(counts: dict[str, Any]) -> dict[str, Any]:
    tokens = int(counts.get("tokens", 0))
    nll = float(counts.get("nll", 0.0))
    chars = int(counts.get("chars", 0))
    bytes_ = int(counts.get("bytes", 0))
    loss = nll / max(1, tokens)
    return {
        "tokens": tokens,
        "nll": nll,
        "loss": loss,
        "ppl": safe_exp(loss),
        "token_acc": counts.get("correct", 0) / max(1, tokens),
        "chars": chars,
        "bytes": bytes_,
        "bpc": bpc(nll, chars),
        "bpb": bpb(nll, bytes_),
        "documents": counts.get("documents", 0),
    }


@torch.inference_mode()
def external_lm_metrics(model, sp, records: list[dict[str, Any]], device: torch.device) -> dict[str, Any]:
    start = time.perf_counter()
    per_doc = []
    totals = {"tokens": 0, "nll": 0.0, "correct": 0, "chars": 0, "bytes": 0, "documents": 0}
    by_domain: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tokens": 0, "nll": 0.0, "correct": 0, "chars": 0, "bytes": 0, "documents": 0}
    )
    for record in records:
        text = record["text"]
        ids = sp.encode(text, out_type=int, add_bos=False)
        scored = score_token_ids(model, ids, device)
        chars = len(text)
        bytes_ = len(text.encode("utf-8"))
        counts = {
            **scored,
            "chars": chars,
            "bytes": bytes_,
            "documents": 1,
        }
        doc_summary = summarize_lm_counts(counts)
        row = {
            "id": record["id"],
            "domain": record["domain"],
            "source": record.get("source"),
            "license": record.get("license"),
            **doc_summary,
        }
        per_doc.append(row)
        for key in ("tokens", "nll", "correct", "chars", "bytes", "documents"):
            totals[key] += counts[key]
            by_domain[record["domain"]][key] += counts[key]
    elapsed = time.perf_counter() - start
    summary = summarize_lm_counts(totals)
    summary.update({
        "name": "external_lm",
        "elapsed_sec": elapsed,
        "tokens_per_sec": summary["tokens"] / max(elapsed, 1e-9),
    })
    domains = []
    for domain, counts in sorted(by_domain.items()):
        domains.append({"domain": domain, **summarize_lm_counts(counts)})
    return {"summary": summary, "domains": domains, "per_doc": per_doc}


def write_report(path: Path, manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# 中文语言建模评测报告",
        "",
        "## 配置",
        "",
        f"- checkpoint: `{manifest['checkpoint']['path']}`",
        f"- tokenizer: `{manifest['tokenizer']['path']}`",
        f"- device: `{manifest['device']}`",
        f"- dtype: `{manifest['dtype']}`",
        "",
    ]
    internal = summary.get("internal_validation")
    if internal:
        lines.extend([
            "## 内部固定 Validation",
            "",
            "| batches | tokens | loss | ppl | token acc | elapsed sec |",
            "|---:|---:|---:|---:|---:|---:|",
            f"| {internal['batches']} | {internal['tokens']} | {internal['loss']:.4f} | "
            f"{internal['ppl']:.2f} | {internal['token_acc']:.4f} | "
            f"{internal['elapsed_sec']:.1f} |",
            "",
        ])
    external = summary.get("external_lm")
    if external:
        ext = external["summary"]
        lines.extend([
            "## 外部冻结中文语料",
            "",
            "| documents | tokens | loss | ppl | BPC | BPB | token acc |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            f"| {ext['documents']} | {ext['tokens']} | {ext['loss']:.4f} | "
            f"{ext['ppl']:.2f} | {ext['bpc']:.4f} | {ext['bpb']:.4f} | "
            f"{ext['token_acc']:.4f} |",
            "",
            "### 分领域",
            "",
            "| domain | docs | tokens | loss | ppl | BPC | BPB |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in external["domains"]:
            lines.append(
                f"| {row['domain']} | {row['documents']} | {row['tokens']} | "
                f"{row['loss']:.4f} | {row['ppl']:.2f} | "
                f"{row['bpc']:.4f} | {row['bpb']:.4f} |"
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

    summary: dict[str, Any] = {}
    if args.validation_bin:
        summary["internal_validation"] = internal_validation_metrics(
            model,
            args.validation_bin,
            args.validation_block_size,
            args.validation_batch_size,
            args.validation_batches,
            device,
        )

    records = load_external_records(args.external_jsonl, args.external_text_dir)
    if args.external_limit is not None:
        records = records[:args.external_limit]
    if records:
        external = external_lm_metrics(model, sp, records, device)
        summary["external_lm"] = {
            "summary": external["summary"],
            "domains": external["domains"],
        }
        write_csv(output_dir / "external_per_doc.csv", external["per_doc"])
        write_jsonl(output_dir / "external_per_doc.jsonl", external["per_doc"])

    if not summary:
        raise ValueError("nothing to evaluate: provide --validation-bin or external data")

    manifest = base_manifest(
        args,
        output_dir,
        payload,
        include_sha256=args.sha256_files,
        extra={
            "suite_name": args.suite_name,
            "validation_bin": str(args.validation_bin) if args.validation_bin else None,
            "external_jsonl": [str(p) for p in args.external_jsonl],
            "external_text_dir": [str(p) for p in args.external_text_dir],
        },
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", manifest, summary)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
