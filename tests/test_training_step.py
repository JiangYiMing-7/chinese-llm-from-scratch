import json

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

from model.causal_lm import CausalLM
from model.config import tiny_config
from pretrain.checkpoint import load_checkpoint, save_checkpoint
from pretrain.dataset import TokenDataset
from pretrain.optim import build_optimizer
from pretrain.pretrain import pretrain, train_accumulated_step, train_one_step, validate


def test_dataset_formats_and_training_step(tmp_path):
    tokens = np.arange(256, dtype=np.int64) % 64
    npy = tmp_path / "ids.npy"
    np.save(npy, tokens)
    ds = TokenDataset(npy, block_size=16)
    assert len(ds) == 15
    assert ds[0].shape == (17,)

    pt = tmp_path / "ids.pt"
    torch.save({"token_ids": torch.from_numpy(tokens)}, pt)
    assert TokenDataset(pt, block_size=16)[0].shape == (17,)

    bin_path = tmp_path / "ids.bin"
    tokens.astype(np.int16).tofile(bin_path)
    with open(str(bin_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"dtype": "int16", "total_tokens": len(tokens)}, f)
    assert TokenDataset(bin_path, block_size=16)[0].shape == (17,)

    cfg = tiny_config()
    cfg.vocab_size = 64
    cfg.max_seq_len = 16
    model = CausalLM(cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    before = model.token_embedding.weight.detach().clone()
    batch = torch.stack([ds[0], ds[1]])
    metrics = train_one_step(model, batch, optimizer, {"grad_clip": 1.0})
    assert metrics["loss"] > 0
    assert metrics["ppl"] > 1
    assert 0 <= metrics["token_acc"] <= 1
    assert metrics["grad_norm"] >= 0
    assert metrics["tokens"] == 32
    assert not torch.allclose(before, model.token_embedding.weight)

    val_metrics = validate(model, DataLoader(ds, batch_size=2), max_batches=2)
    assert val_metrics["val_loss"] > 0
    assert val_metrics["val_ppl"] > 1
    assert 0 <= val_metrics["val_token_acc"] <= 1
    assert val_metrics["val_tokens"] == 64

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt,
        model,
        optimizer,
        step=1,
        config={"model": cfg.__dict__},
        extra_state={"tokens_seen": metrics["tokens"], "best_val_loss": 1.23},
    )
    restored = CausalLM(cfg)
    state = load_checkpoint(ckpt, restored)
    assert state["step"] == 1
    assert state["tokens_seen"] == metrics["tokens"]
    assert state["best_val_loss"] == 1.23


def test_train_accumulated_step_counts_effective_tokens(tmp_path):
    tokens = np.arange(256, dtype=np.int64) % 64
    npy = tmp_path / "ids.npy"
    np.save(npy, tokens)
    ds = TokenDataset(npy, block_size=16)

    cfg = tiny_config()
    cfg.vocab_size = 64
    cfg.max_seq_len = 16
    model = CausalLM(cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    before = model.token_embedding.weight.detach().clone()
    micro_batches = [
        torch.stack([ds[0], ds[1]]),
        torch.stack([ds[2], ds[3]]),
    ]

    metrics = train_accumulated_step(
        model,
        micro_batches,
        optimizer,
        {"grad_clip": 1.0, "track_token_acc": True},
        precision="fp32",
    )
    assert metrics["loss"] > 0
    assert metrics["ppl"] > 1
    assert metrics["tokens"] == 64
    assert metrics["samples"] == 4
    assert metrics["gradient_accumulation_steps"] == 2
    assert metrics["micro_steps"] == 2
    assert metrics["tokens_per_micro_step"] == 32
    assert metrics["effective_batch_tokens"] == 64
    assert 0 <= metrics["token_acc"] <= 1
    assert metrics["grad_norm"] >= 0
    assert not torch.allclose(before, model.token_embedding.weight)


def test_pretrain_writes_metrics_and_checkpoints(tmp_path):
    tokens = torch.arange(256, dtype=torch.long) % 64
    token_path = tmp_path / "tokens.pt"
    torch.save({"token_ids": tokens}, token_path)

    run_root = tmp_path / "runs"
    config = {
        "model": {
            "vocab_size": 64,
            "max_seq_len": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 32,
            "head_dim": 16,
            "ffn_hidden_dim": 64,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "norm_eps": 0.00001,
            "tie_embeddings": True,
            "bias": False,
        },
        "data": {
            "train_path": str(token_path),
            "val_path": None,
            "val_split": 0.2,
            "block_size": 8,
            "num_workers": 0,
        },
        "train": {
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 2,
            "warmup_steps": 1,
            "lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_interval": 1,
            "metrics_interval": 1,
            "run_dir_root": str(run_root),
            "run_name": "unit",
            "metrics_filename": "metrics.jsonl",
            "hardware_metrics_interval": 1,
            "eval_interval": 1,
            "eval_batches": 1,
            "save_interval": 2,
            "save_best": True,
            "best_checkpoint_name": "best.pt",
            "resume_from": None,
            "seed": 1337,
            "device": "cpu",
            "track_token_acc": True,
            "loss_ema_beta": 0.95,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pretrain(str(config_path))

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    checkpoint_dir = run_dir / "checkpoints"
    assert (checkpoint_dir / "ckpt.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    metrics_path = run_dir / "metrics.jsonl"
    run_info_path = run_dir / "run_info.json"
    config_snapshot_path = run_dir / "config.yaml"
    summary_path = run_dir / "summary.json"
    assert metrics_path.exists()
    assert run_info_path.exists()
    assert config_snapshot_path.exists()
    assert summary_path.exists()
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    assert run_info["run_id"].startswith("unit_")
    assert run_info["metrics_path"] == str(metrics_path)
    assert run_info["train"]["run_dir"] == str(run_dir)
    assert run_info["gradient_accumulation_steps"] == 1
    assert run_info["tokens_per_micro_step"] == 16
    assert run_info["effective_batch_tokens"] == 16
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["latest_checkpoint"] == str(checkpoint_dir / "ckpt.pt")
    assert summary["best_checkpoint"] == str(checkpoint_dir / "best.pt")
    assert summary["gradient_accumulation_steps"] == 1
    assert summary["tokens_per_micro_step"] == 16
    assert summary["effective_batch_tokens"] == 16
    rows = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [row["event"] for row in rows]
    assert "run_start" in events
    assert "train" in events
    assert "eval" in events
    assert "checkpoint" in events
    assert "run_end" in events
    train_row = next(row for row in rows if row["event"] == "train")
    assert train_row["loss"] > 0
    assert train_row["ppl"] > 1
    assert 0 <= train_row["token_acc"] <= 1
    assert train_row["hw_device_type"] == "cpu"
    assert "hw_memory_allocated_mb" in train_row
    assert "hw_gpu_utilization_percent" in train_row
    assert train_row["hw_external_sampled"] is False
    assert train_row["gradient_accumulation_steps"] == 1
    assert train_row["tokens_per_micro_step"] == 16
    assert train_row["effective_batch_tokens"] == 16
    assert train_row["micro_steps_seen"] == train_row["step"]
    eval_row = next(row for row in rows if row["event"] == "eval")
    assert eval_row["val_loss"] > 0
    assert 0 <= eval_row["val_token_acc"] <= 1


def test_pretrain_gradient_accumulation_metadata(tmp_path):
    tokens = torch.arange(256, dtype=torch.long) % 64
    token_path = tmp_path / "tokens.pt"
    torch.save({"token_ids": tokens}, token_path)

    run_root = tmp_path / "runs"
    config = {
        "model": {
            "vocab_size": 64,
            "max_seq_len": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 32,
            "head_dim": 16,
            "ffn_hidden_dim": 64,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "norm_eps": 0.00001,
            "tie_embeddings": True,
            "bias": False,
        },
        "data": {
            "train_path": str(token_path),
            "val_path": None,
            "val_split": 0.2,
            "block_size": 8,
            "num_workers": 0,
        },
        "train": {
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "max_steps": 2,
            "warmup_steps": 1,
            "lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_interval": 1,
            "metrics_interval": 1,
            "run_dir_root": str(run_root),
            "run_name": "accum",
            "metrics_filename": "metrics.jsonl",
            "hardware_metrics_interval": 1,
            "eval_interval": 1,
            "eval_batches": 1,
            "save_interval": 2,
            "save_best": True,
            "best_checkpoint_name": "best.pt",
            "resume_from": None,
            "seed": 1337,
            "device": "cpu",
            "track_token_acc": True,
            "loss_ema_beta": 0.95,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    pretrain(str(config_path))

    run_dir = next(run_root.iterdir())
    run_info = json.loads((run_dir / "run_info.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert run_info["gradient_accumulation_steps"] == 2
    assert run_info["tokens_per_micro_step"] == 16
    assert run_info["effective_batch_tokens"] == 32
    assert summary["tokens_seen"] == 64
    assert summary["micro_steps_seen"] == 4
    assert summary["gradient_accumulation_steps"] == 2
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    train_rows = [row for row in rows if row["event"] == "train"]
    assert train_rows[-1]["step"] == 2
    assert train_rows[-1]["tokens_seen"] == 64
    assert train_rows[-1]["micro_steps_seen"] == 4
    assert train_rows[-1]["tokens_per_micro_step"] == 16
    assert train_rows[-1]["effective_batch_tokens"] == 32


def test_pretrain_writes_failed_summary_for_runtime_errors(tmp_path):
    tokens = torch.arange(32, dtype=torch.long) % 16
    token_path = tmp_path / "tokens.pt"
    torch.save({"token_ids": tokens}, token_path)

    run_root = tmp_path / "runs"
    config = {
        "model": {
            "vocab_size": 16,
            "max_seq_len": 8,
            "n_layers": 1,
            "n_heads": 2,
            "d_model": 32,
            "head_dim": 16,
            "ffn_hidden_dim": 64,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "norm_eps": 0.00001,
            "tie_embeddings": True,
            "bias": False,
        },
        "data": {
            "train_path": str(token_path),
            "val_path": None,
            "val_split": 0.2,
            "block_size": 8,
            "num_workers": 0,
        },
        "train": {
            "batch_size": 100,
            "max_steps": 1,
            "warmup_steps": 1,
            "lr": 1e-3,
            "min_lr": 1e-4,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "log_interval": 1,
            "metrics_interval": 1,
            "run_dir_root": str(run_root),
            "run_name": "failed",
            "metrics_filename": "metrics.jsonl",
            "hardware_metrics_interval": 1,
            "eval_interval": 1,
            "eval_batches": 1,
            "save_interval": 1,
            "save_best": True,
            "best_checkpoint_name": "best.pt",
            "resume_from": None,
            "seed": 1337,
            "device": "cpu",
            "track_token_acc": True,
            "loss_ema_beta": 0.95,
        },
    }
    config_path = tmp_path / "bad_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="training dataset is too small"):
        pretrain(str(config_path))

    run_dirs = list(run_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["error_type"] == "ValueError"
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "run_start" in [row["event"] for row in rows]
    assert "run_error" in [row["event"] for row in rows]
