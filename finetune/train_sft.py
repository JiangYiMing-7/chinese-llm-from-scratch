#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finetune.sft_dataset import SFTCollator, SFTDataset  # noqa: E402
from model.causal_lm import CausalLM  # noqa: E402
from model.config import ModelConfig  # noqa: E402
from pretrain.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from pretrain.optim import build_optimizer, get_cosine_lr  # noqa: E402
from utils.tokenizer_adapter import load_tokenizer  # noqa: E402
from utils.params import count_parameters  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


def device_from_name(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def resolve_precision(name: str, device: torch.device) -> str:
    precision = name.lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError("train.precision must be fp32 or bf16")
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("bf16 SFT requires CUDA")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA device does not support bf16")
    return precision


def autocast_context(precision: str, device: torch.device):
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


def safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    if value >= 700:
        return float("inf")
    return math.exp(value)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


# 按名字正则把参数分成几组，做“只训某一部分、冻结其余”的消融用（如只放开 attention 或 ffn）
FREEZE_GROUP_PATTERNS = {
    "attention": [r"^blocks\.[0-9]+\.attention\."],
    "ffn": [r"^blocks\.[0-9]+\.ffn\."],
    "norms": [r"(^final_norm\.)|(^blocks\.[0-9]+\..*norm\.)"],
    "tied_embedding_head": [r"^(token_embedding|lm_head)\."],
}


def parameter_count(params: list[torch.nn.Parameter]) -> int:
    return sum(param.numel() for param in params)


def match_parameter_names(model, patterns: list[str]) -> set[str]:
    compiled = [re.compile(pattern) for pattern in patterns]
    return {
        name
        for name, _param in model.named_parameters()
        if any(pattern.search(name) for pattern in compiled)
    }


def apply_freeze_config(model, freeze_cfg: dict[str, Any] | None) -> dict[str, Any]:
    all_named_params = list(model.named_parameters())
    all_names = {name for name, _param in all_named_params}
    if not freeze_cfg:
        for _name, param in all_named_params:
            param.requires_grad = True
        total = parameter_count([param for _name, param in all_named_params])
        return {
            "enabled": False,
            "total_params": total,
            "trainable_params": total,
            "frozen_params": 0,
            "trainable_ratio": 1.0,
        }

    mode = str(freeze_cfg.get("mode", "train_only"))
    if mode not in {"train_only", "freeze_only"}:
        raise ValueError("train.freeze.mode must be train_only or freeze_only")

    selected: set[str] = set()
    group_summaries: list[dict[str, Any]] = []
    for group in freeze_cfg.get("groups", []) or []:
        if group not in FREEZE_GROUP_PATTERNS:
            raise ValueError(f"unknown freeze group: {group}")
        matches = match_parameter_names(model, FREEZE_GROUP_PATTERNS[group])
        if not matches:
            raise ValueError(f"freeze group matched no parameters: {group}")
        selected.update(matches)
        group_summaries.append({
            "group": group,
            "patterns": FREEZE_GROUP_PATTERNS[group],
            "matched_params": len(matches),
            "matched_names": sorted(matches),
        })

    pattern_summaries: list[dict[str, Any]] = []
    for pattern in freeze_cfg.get("parameter_patterns", []) or []:
        matches = match_parameter_names(model, [str(pattern)])
        if not matches:
            raise ValueError(f"freeze pattern matched no parameters: {pattern}")
        selected.update(matches)
        pattern_summaries.append({
            "pattern": str(pattern),
            "matched_params": len(matches),
            "matched_names": sorted(matches),
        })

    unknown_names = selected - all_names
    if unknown_names:
        raise ValueError(f"internal freeze match error, unknown parameters: {sorted(unknown_names)[:5]}")
    if not selected:
        raise ValueError("train.freeze selected no parameters")

    if mode == "train_only":
        trainable_names = selected
    else:
        trainable_names = all_names - selected

    for name, param in all_named_params:
        param.requires_grad = name in trainable_names

    trainable_params = parameter_count([param for _name, param in all_named_params if param.requires_grad])
    total_params = parameter_count([param for _name, param in all_named_params])
    frozen_params = total_params - trainable_params
    if trainable_params <= 0:
        raise ValueError("freeze config left no trainable parameters")

    return {
        "enabled": True,
        "mode": mode,
        "config": freeze_cfg,
        "groups": group_summaries,
        "patterns": pattern_summaries,
        "selected_param_names": sorted(selected),
        "trainable_param_names": sorted(trainable_names),
        "frozen_param_names": sorted(all_names - trainable_names),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "trainable_ratio": trainable_params / max(1, total_params),
    }


class MetricWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", encoding="utf-8", buffering=1)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        self.file.write(json.dumps(json_ready({"event": event, "time": now(), **payload}), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.file.close()


def unique_run_dir(root: str | Path, run_name: str) -> Path:
    root = Path(root)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base = root / f"{run_name}_{timestamp}"
    candidate = base
    for i in range(1, 1000):
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{i:02d}")
    raise FileExistsError(f"could not create unique run dir under {root}")


def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # 只在被监督的 token（label != -100）上算准确率，prompt 部分不计
    mask = labels != -100
    total = int(mask.sum().item())
    if total == 0:
        return float("nan")
    preds = logits.argmax(dim=-1)
    return float(((preds == labels) & mask).sum().item() / total)


def sft_micro_step(model, batch, precision: str, device: torch.device, loss_scale: float) -> dict[str, Any]:
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch.get("attention_mask")
    step_input = input_ids[:, :-1]
    step_labels = labels[:, 1:]
    if attention_mask is not None:
        token_count = int(attention_mask[:, 1:].sum().item())
    else:
        token_count = int((input_ids[:, 1:] != 0).sum().item())
    supervised_tokens = int((step_labels != -100).sum().item())
    if supervised_tokens == 0:
        # 整个 batch 都没有 assistant token 可学，多半是数据或模板出问题了
        raise ValueError("batch has no supervised assistant tokens")
    with autocast_context(precision, device):
        output = model(step_input, labels=step_labels)
    output_total_loss = getattr(output, "total_loss", None)
    backward_loss = output_total_loss if output_total_loss is not None else output.loss
    loss = backward_loss / loss_scale
    loss.backward()
    loss_value = float(output.loss.detach().float().cpu())
    metrics = {
        "loss": loss_value,
        "total_loss": float(backward_loss.detach().float().cpu()),
        "token_acc": masked_accuracy(output.logits.detach(), step_labels),
        "supervised_tokens": supervised_tokens,
        "tokens": token_count,
        "samples": input_ids.size(0),
        "seq_len": input_ids.size(1),
    }
    moe_aux_loss = getattr(output, "moe_aux_loss", None)
    if moe_aux_loss is not None:
        metrics["moe_aux_loss"] = float(moe_aux_loss.detach().float().cpu())
    return metrics


def grad_norm_and_step(model, optimizer, grad_clip: float | None) -> float:
    if grad_clip:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).detach().cpu())
    else:
        norms = [p.grad.detach().float().norm(2) for p in model.parameters() if p.grad is not None]
        grad_norm = float(torch.norm(torch.stack(norms), 2).detach().cpu()) if norms else 0.0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return grad_norm


def aggregate_micro_metrics(rows: list[dict[str, Any]], grad_norm: float, grad_accum: int) -> dict[str, Any]:
    supervised = sum(int(row["supervised_tokens"]) for row in rows)
    tokens = sum(int(row["tokens"]) for row in rows)
    samples = sum(int(row["samples"]) for row in rows)
    weighted_loss = sum(float(row["loss"]) * int(row["supervised_tokens"]) for row in rows)
    weighted_total_loss = sum(
        float(row["total_loss"]) * int(row["supervised_tokens"]) for row in rows
    )
    weighted_acc = sum(float(row["token_acc"]) * int(row["supervised_tokens"]) for row in rows)
    loss = weighted_loss / max(1, supervised)
    metrics = {
        "loss": loss,
        "total_loss": weighted_total_loss / max(1, supervised),
        "ppl": safe_exp(loss),
        "token_acc": weighted_acc / max(1, supervised),
        "grad_norm": grad_norm,
        "supervised_tokens": supervised,
        "tokens": tokens,
        "samples": samples,
        "seq_len": max(int(row["seq_len"]) for row in rows),
        "micro_batches": len(rows),
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": samples,
        "effective_supervised_tokens_per_step": supervised,
    }
    if "moe_aux_loss" in rows[0]:
        weighted_aux_loss = sum(
            float(row["moe_aux_loss"]) * int(row["supervised_tokens"])
            for row in rows
        )
        metrics["moe_aux_loss"] = weighted_aux_loss / max(1, supervised)
    return metrics


@torch.no_grad()
def validate(model, loader, precision: str, device: torch.device, max_batches: int | None) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_supervised = 0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        step_input = input_ids[:, :-1]
        step_labels = labels[:, 1:]
        mask = step_labels != -100
        supervised = int(mask.sum().item())
        if supervised == 0:
            continue
        with autocast_context(precision, device):
            output = model(step_input, labels=step_labels)
        total_loss += float(output.loss.detach().float().cpu()) * supervised
        preds = output.logits.argmax(dim=-1)
        total_correct += int(((preds == step_labels) & mask).sum().item())
        total_supervised += supervised
        batches += 1
    model.train()
    loss = total_loss / max(1, total_supervised)
    return {
        "val_loss": loss,
        "val_ppl": safe_exp(loss),
        "val_token_acc": total_correct / max(1, total_supervised),
        "val_supervised_tokens": total_supervised,
        "val_batches": batches,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run supervised fine-tuning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    seed_everything(int(train_cfg.get("seed", 1337)))
    device = device_from_name(str(train_cfg.get("device", "auto")))
    precision = resolve_precision(str(train_cfg.get("precision", "fp32")), device)

    tokenizer = load_tokenizer(data_cfg["tokenizer_path"])
    model_cfg = ModelConfig(**cfg["model"])
    if tokenizer.get_piece_size() != model_cfg.vocab_size:
        raise ValueError(f"tokenizer/model vocab mismatch: {tokenizer.get_piece_size()} != {model_cfg.vocab_size}")
    pad_id = tokenizer.pad_id()
    if pad_id < 0:
        pad_id = tokenizer.piece_to_id("<|endoftext|>")

    model = CausalLM(model_cfg)
    init_from = train_cfg.get("init_from")
    if init_from:
        load_checkpoint(init_from, model, optimizer=None)
    elif not train_cfg.get("allow_random_init", False):
        raise ValueError("train.init_from is required unless allow_random_init=true")
    freeze_summary = apply_freeze_config(model, train_cfg.get("freeze"))
    model = model.to(device)
    optimizer = build_optimizer(model, float(train_cfg["lr"]), float(train_cfg.get("weight_decay", 0.0)))
    if train_cfg.get("resume_from"):
        load_checkpoint(train_cfg["resume_from"], model, optimizer)

    max_seq_len = int(data_cfg.get("max_seq_len", model_cfg.max_seq_len))
    default_system = data_cfg.get("default_system")
    train_dataset = SFTDataset(data_cfg["train_path"], tokenizer, max_seq_len, default_system=default_system)
    val_dataset = SFTDataset(data_cfg.get("val_path", data_cfg["train_path"]), tokenizer, max_seq_len, default_system=default_system)
    collator = SFTCollator(pad_id, max_seq_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=bool(train_cfg.get("drop_last", False)),
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg.get("eval_batch_size", train_cfg["batch_size"])),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=False,
        collate_fn=collator,
    )
    grad_accum = max(1, int(train_cfg.get("gradient_accumulation_steps", 1)))
    effective_batch_size = int(train_cfg["batch_size"]) * grad_accum

    run_dir = Path(train_cfg["run_dir"]) if train_cfg.get("run_dir") else unique_run_dir(
        train_cfg.get("run_dir_root", "results/sft_runs"),
        train_cfg.get("run_name", Path(args.config).stem),
    )
    checkpoint_dir = run_dir / "checkpoints"
    metrics_path = run_dir / "metrics.jsonl"
    write_yaml(run_dir / "config.yaml", cfg)
    write_json(run_dir / "run_info.json", {
        "run_dir": run_dir,
        "started_at": now(),
        "config_path": args.config,
        "init_from": init_from,
        "device": str(device),
        "precision": precision,
        "tokenizer_kind": getattr(tokenizer, "kind", "unknown"),
        "model_params": count_parameters(model),
        "freeze": freeze_summary,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size": effective_batch_size,
        "model": cfg["model"],
        "data": cfg["data"],
        "train": cfg["train"],
    })

    writer = MetricWriter(metrics_path)
    writer.write("run_start", {
        "run_dir": str(run_dir),
        "config_path": args.config,
        "init_from": init_from,
        "device": str(device),
        "precision": precision,
        "tokenizer_kind": getattr(tokenizer, "kind", "unknown"),
        "model_params": count_parameters(model),
        "freeze": freeze_summary,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
    })

    step = 0
    best_val_loss = float("inf")
    max_steps = int(train_cfg["max_steps"])
    start_time = time.time()
    model.train()
    try:
        train_iter = iter(train_loader)
        optimizer.zero_grad(set_to_none=True)
        while step < max_steps:
            lr = get_cosine_lr(
                step,
                int(train_cfg.get("warmup_steps", 0)),
                max_steps,
                float(train_cfg["lr"]),
                float(train_cfg.get("min_lr", train_cfg["lr"])),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            step_start = time.perf_counter()
            micro_metrics: list[dict[str, Any]] = []
            for _micro_idx in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                micro_metrics.append(sft_micro_step(model, batch, precision, device, grad_accum))
            grad_norm = grad_norm_and_step(model, optimizer, train_cfg.get("grad_clip"))
            elapsed_step = time.perf_counter() - step_start
            step += 1
            metrics = aggregate_micro_metrics(micro_metrics, grad_norm, grad_accum)
            metrics.update({
                "step": step,
                "lr": lr,
                "elapsed_sec": time.time() - start_time,
                "step_time_sec": elapsed_step,
                "supervised_tokens_per_sec": metrics["supervised_tokens"] / max(elapsed_step, 1e-9),
            })
            if step == 1 or step % int(train_cfg.get("metrics_interval", 1)) == 0:
                writer.write("train", metrics)
            if step == 1 or step % int(train_cfg.get("log_interval", 10)) == 0:
                print(
                    f"step={step} loss={metrics['loss']:.4f} "
                    f"acc={metrics['token_acc']:.4f} lr={lr:.2e} "
                    f"eff_bs={metrics['effective_batch_size']} "
                    f"sup_tok/s={metrics['supervised_tokens_per_sec']:.1f}",
                    flush=True,
                )
            if step % int(train_cfg.get("eval_interval", 100)) == 0:
                val_metrics = validate(
                    model,
                    val_loader,
                    precision,
                    device,
                    train_cfg.get("eval_batches"),
                )
                val_metrics.update({"step": step, "elapsed_sec": time.time() - start_time})
                writer.write("eval", val_metrics)
                print(
                    f"step={step} val_loss={val_metrics['val_loss']:.4f} "
                    f"val_acc={val_metrics['val_token_acc']:.4f}",
                    flush=True,
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        checkpoint_dir / "best.pt",
                        model,
                        optimizer,
                        step,
                        cfg,
                        extra_state={"best_val_loss": best_val_loss},
                    )
                    writer.write("checkpoint", {
                        "step": step,
                        "kind": "best",
                        "path": str(checkpoint_dir / "best.pt"),
                        "best_val_loss": best_val_loss,
                    })
            if step % int(train_cfg.get("save_interval", 500)) == 0:
                save_checkpoint(checkpoint_dir / "ckpt.pt", model, optimizer, step, cfg)
                writer.write("checkpoint", {"step": step, "kind": "latest", "path": str(checkpoint_dir / "ckpt.pt")})
        if not (checkpoint_dir / "best.pt").exists():
            save_checkpoint(checkpoint_dir / "best.pt", model, optimizer, step, cfg, extra_state={"best_val_loss": best_val_loss})
        save_checkpoint(checkpoint_dir / "ckpt.pt", model, optimizer, step, cfg)
        ended_at = now()
        summary = {
            "status": "completed",
            "run_dir": str(run_dir),
            "ended_at": ended_at,
            "step": step,
            "best_val_loss": best_val_loss,
            "best_checkpoint": str(checkpoint_dir / "best.pt"),
            "latest_checkpoint": str(checkpoint_dir / "ckpt.pt"),
            "metrics_path": str(metrics_path),
        }
        write_json(run_dir / "summary.json", summary)
        writer.write("run_end", summary)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
