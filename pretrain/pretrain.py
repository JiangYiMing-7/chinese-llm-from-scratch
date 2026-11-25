from contextlib import nullcontext
from pathlib import Path
import json
import math
import os
import shutil
import subprocess
import time

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.causal_lm import CausalLM
from model.config import ModelConfig
from pretrain.checkpoint import load_checkpoint, save_checkpoint
from pretrain.dataset import TokenDataset
from pretrain.optim import build_optimizer, get_cosine_lr
from utils.logging import get_logger
from utils.params import count_parameters
from utils.seed import seed_everything

logger = get_logger("pretrain")


def _dist_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _init_distributed() -> dict:
    world_size = _dist_env_int("WORLD_SIZE", 1)
    rank = _dist_env_int("RANK", 0)
    local_rank = _dist_env_int("LOCAL_RANK", 0)
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return {
        "enabled": enabled,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def _barrier(distributed: dict) -> None:
    if distributed["enabled"] and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[distributed["local_rank"]])
        else:
            dist.barrier()


def _cleanup_distributed(distributed: dict) -> None:
    if distributed["enabled"] and dist.is_initialized():
        dist.destroy_process_group()


def _safe_exp(value: float) -> float:
    # perplexity = exp(loss)，loss 太大先截一下，免得 exp 溢出报错
    if value is None or not math.isfinite(value):
        return float("nan")
    if value >= 700:
        return float("inf")
    return math.exp(value)


def _json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _fmt(value: float, precision: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.{precision}f}"


class JsonlMetricWriter:
    def __init__(self, path: str | Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self.file = open(self.path, mode, encoding="utf-8", buffering=1)

    def write(self, event: str, payload: dict) -> None:
        row = {
            "event": event,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            **payload,
        }
        self.file.write(json.dumps(_json_ready(row), ensure_ascii=False) + "\n")

    def close(self) -> None:
        self.file.close()


class HardwareMonitor:
    """Collect lightweight accelerator metrics for training logs.

    PyTorch memory metrics are cheap and can be recorded every train event.
    NVIDIA utilization metrics require `nvidia-smi`, so they are sampled at a
    configurable interval and left as null on non-sampled rows.
    """

    _FIELDS = {
        "hw_device_type": None,
        "hw_device_index": None,
        "hw_device_name": None,
        "hw_memory_allocated_mb": None,
        "hw_memory_reserved_mb": None,
        "hw_memory_peak_allocated_mb": None,
        "hw_memory_total_mb": None,
        "hw_memory_allocated_ratio": None,
        "hw_memory_reserved_ratio": None,
        "hw_gpu_utilization_percent": None,
        "hw_memory_utilization_percent": None,
        "hw_memory_used_external_mb": None,
        "hw_memory_total_external_mb": None,
        "hw_external_sampled": False,
        "hw_external_source": None,
    }

    def __init__(self, device: torch.device):
        self.device = device
        self.device_type = device.type
        self.device_index = device.index
        if self.device_type == "cuda":
            self.device_index = torch.cuda.current_device() if device.index is None else device.index
        self.nvidia_smi = shutil.which("nvidia-smi")

    def static_info(self) -> dict:
        info = dict(self._FIELDS)
        info.update({
            "hw_device_type": self.device_type,
            "hw_device_index": self.device_index,
            "hw_device_name": self._device_name(),
        })
        if self.device_type == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(self.device_index or 0)
            info["hw_memory_total_mb"] = props.total_memory / 1024**2
        return info

    def sample(self, external: bool = False) -> dict:
        metrics = self.static_info()
        metrics.update(self._torch_memory())
        if external:
            external_metrics = self._nvidia_smi_metrics()
            metrics.update(external_metrics)
        return metrics

    def _device_name(self):
        if self.device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(self.device_index or 0)
        if self.device_type == "mps":
            return "Apple MPS"
        return "CPU"

    def _torch_memory(self) -> dict:
        metrics = {}
        if self.device_type == "cuda" and torch.cuda.is_available():
            idx = self.device_index or 0
            allocated = torch.cuda.memory_allocated(idx) / 1024**2
            reserved = torch.cuda.memory_reserved(idx) / 1024**2
            peak = torch.cuda.max_memory_allocated(idx) / 1024**2
            total = torch.cuda.get_device_properties(idx).total_memory / 1024**2
            metrics.update({
                "hw_memory_allocated_mb": allocated,
                "hw_memory_reserved_mb": reserved,
                "hw_memory_peak_allocated_mb": peak,
                "hw_memory_total_mb": total,
                "hw_memory_allocated_ratio": allocated / total if total else None,
                "hw_memory_reserved_ratio": reserved / total if total else None,
            })
        elif self.device_type == "mps" and hasattr(torch, "mps"):
            allocated = None
            reserved = None
            if hasattr(torch.mps, "current_allocated_memory"):
                allocated = torch.mps.current_allocated_memory() / 1024**2
            if hasattr(torch.mps, "driver_allocated_memory"):
                reserved = torch.mps.driver_allocated_memory() / 1024**2
            metrics.update({
                "hw_memory_allocated_mb": allocated,
                "hw_memory_reserved_mb": reserved,
                "hw_memory_peak_allocated_mb": None,
                "hw_memory_total_mb": None,
                "hw_memory_allocated_ratio": None,
                "hw_memory_reserved_ratio": None,
            })
        return metrics

    def _nvidia_smi_metrics(self) -> dict:
        metrics = {
            "hw_gpu_utilization_percent": None,
            "hw_memory_utilization_percent": None,
            "hw_memory_used_external_mb": None,
            "hw_memory_total_external_mb": None,
            "hw_external_sampled": False,
            "hw_external_source": None,
        }
        if self.device_type != "cuda" or not self.nvidia_smi:
            return metrics

        cmd = [
            self.nvidia_smi,
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        smi_index = self._nvidia_smi_index()
        if smi_index is not None:
            cmd.insert(1, f"--id={smi_index}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            return metrics
        if result.returncode != 0 or not result.stdout.strip():
            return metrics

        line = result.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return metrics
        try:
            gpu_util, mem_util, mem_used, mem_total = (float(p) for p in parts[:4])
        except ValueError:
            return metrics
        metrics.update({
            "hw_gpu_utilization_percent": gpu_util,
            "hw_memory_utilization_percent": mem_util,
            "hw_memory_used_external_mb": mem_used,
            "hw_memory_total_external_mb": mem_total,
            "hw_external_sampled": True,
            "hw_external_source": "nvidia-smi",
        })
        return metrics

    def _nvidia_smi_index(self):
        if self.device_index is None:
            return None
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
            if self.device_index < len(visible_ids):
                physical_id = visible_ids[self.device_index]
                if physical_id.isdigit():
                    return int(physical_id)
                return None
        return self.device_index


def _device(name: str, distributed: dict | None = None) -> torch.device:
    distributed = distributed or {"enabled": False, "local_rank": 0}
    if distributed["enabled"] and torch.cuda.is_available():
        torch.cuda.set_device(distributed["local_rank"])
        return torch.device("cuda", distributed["local_rank"])
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _unique_dir(base: Path) -> Path:
    candidate = base
    for i in range(1, 1000):
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{i:02d}")
    raise FileExistsError(f"could not create a unique run directory under {base}")


def _init_run_artifacts(train_cfg: dict, config_path: str) -> dict:
    run_name = train_cfg.get("run_name") or Path(config_path).stem
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{timestamp}"
    if train_cfg.get("run_dir"):
        run_dir = _unique_dir(Path(train_cfg["run_dir"]))
        run_id = run_dir.name
    else:
        run_root = Path(train_cfg.get("run_dir_root", "results/runs"))
        run_dir = _unique_dir(run_root / run_id)

    metrics_filename = train_cfg.get("metrics_filename")
    if not metrics_filename and train_cfg.get("metrics_path"):
        metrics_filename = Path(train_cfg["metrics_path"]).name
    metrics_filename = metrics_filename or "metrics.jsonl"

    paths = {
        "run_id": run_id,
        "run_dir": run_dir,
        "metrics_path": run_dir / metrics_filename,
        "run_info_path": run_dir / "run_info.json",
        "config_snapshot_path": run_dir / "config.yaml",
        "summary_path": run_dir / "summary.json",
    }
    train_cfg["run_id"] = run_id
    train_cfg["run_dir"] = str(run_dir)
    train_cfg["metrics_path"] = str(paths["metrics_path"])
    return paths


def _init_run_artifacts_distributed(
    train_cfg: dict, config_path: str, distributed: dict
) -> dict:
    if not distributed["enabled"]:
        return _init_run_artifacts(train_cfg, config_path)

    payload = [None]
    if distributed["is_main"]:
        paths = _init_run_artifacts(train_cfg, config_path)
        payload[0] = {
            "paths": {k: str(v) for k, v in paths.items()},
            "train_updates": {
                "run_id": train_cfg["run_id"],
                "run_dir": train_cfg["run_dir"],
                "metrics_path": train_cfg["metrics_path"],
            },
        }
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    train_cfg.update(received["train_updates"])
    paths = {
        key: (Path(value) if key.endswith("_path") or key == "run_dir" else value)
        for key, value in received["paths"].items()
    }
    return paths


def _write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(_json_ready(payload), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _write_yaml(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
    os.replace(tmp_path, path)


def _check_tokenizer(model_cfg: ModelConfig, data_cfg: dict) -> None:
    tokenizer_path = data_cfg.get("tokenizer_path") or data_cfg.get("tokenizer")
    if not tokenizer_path:
        logger.warning("data.tokenizer_path 未设置，跳过 tokenizer/vocab 一致性检查")
        return
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"tokenizer_path does not exist: {tokenizer_path}"
        )
    if tokenizer_path.suffix == ".json":
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ImportError(
                "tokenizers is required to check HuggingFace tokenizer_path; "
                "install it with `python -m pip install tokenizers`"
            ) from exc
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    else:
        try:
            import sentencepiece as spm
        except ImportError as exc:
            raise ImportError(
                "sentencepiece is required to check tokenizer_path; "
                "install data dependencies with `python -m pip install -r data/requirements.txt`"
            ) from exc
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer_path))
        tokenizer_vocab = sp.get_piece_size()
    if tokenizer_vocab != model_cfg.vocab_size:
        raise ValueError(
            "tokenizer vocab size does not match model.vocab_size: "
            f"{tokenizer_vocab} != {model_cfg.vocab_size} ({tokenizer_path})"
        )
    logger.info("tokenizer=%s vocab=%d", tokenizer_path, tokenizer_vocab)


def _check_token_file_metadata(path: str, model_cfg: ModelConfig,
                               tokenizer_path: str | None) -> None:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"token file does not exist: {path_obj}")
    metadata = {}
    if path_obj.suffix == ".bin":
        meta_path = Path(str(path_obj) + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(f"token metadata does not exist: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    elif path_obj.suffix == ".pt":
        payload = torch.load(path_obj, map_location="cpu")
        if isinstance(payload, dict):
            metadata = payload
    vocab_size = metadata.get("vocab_size")
    if vocab_size is not None and int(vocab_size) != model_cfg.vocab_size:
        raise ValueError(
            "token file vocab_size does not match model.vocab_size: "
            f"{vocab_size} != {model_cfg.vocab_size} ({path_obj})"
        )
    token_file_tokenizer = metadata.get("tokenizer")
    if tokenizer_path and token_file_tokenizer:
        if Path(token_file_tokenizer) != Path(tokenizer_path):
            logger.warning(
                "token file tokenizer differs from config tokenizer_path: %s != %s",
                token_file_tokenizer,
                tokenizer_path,
            )


def _check_training_config(model_cfg: ModelConfig, data_cfg: dict) -> None:
    block_size = int(data_cfg["block_size"])
    if block_size > model_cfg.max_seq_len:
        raise ValueError(
            f"data.block_size ({block_size}) must not exceed "
            f"model.max_seq_len ({model_cfg.max_seq_len})"
        )
    val_path = data_cfg.get("val_path")
    if val_path and Path(data_cfg["train_path"]).resolve() == Path(val_path).resolve():
        raise ValueError("data.train_path and data.val_path must be different files")


def _resolve_precision(train_cfg: dict, device: torch.device) -> tuple[str, dict]:
    precision = str(train_cfg.get("precision", "fp32")).lower()
    if precision not in {"fp32", "bf16"}:
        raise ValueError(
            f"unsupported train.precision: {precision}; expected fp32 or bf16"
        )

    runtime = {
        "precision": precision,
        "autocast_enabled": precision == "bf16",
        "autocast_dtype": "bfloat16" if precision == "bf16" else None,
        "cuda_bf16_supported": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
    }
    if device.type == "cuda":
        runtime.update({
            "cuda_bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        })

    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("train.precision=bf16 requires a CUDA device")
        if not runtime["cuda_bf16_supported"]:
            raise RuntimeError("train.precision=bf16 requires CUDA BF16 support")

    train_cfg["precision"] = precision
    return precision, runtime


def _autocast_context(precision: str, device: torch.device):
    # bf16 动态范围够大，不像 fp16 那样还得配 GradScaler，前向用 autocast 包一下就行
    if precision == "bf16":
        if device.type != "cuda":
            raise ValueError("train.precision=bf16 requires a CUDA device")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _grad_norm(parameters) -> float:
    norms = [
        p.grad.detach().float().norm(2)
        for p in parameters
        if p.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).detach().cpu())


def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / max(1, total)


def _forward_backward_micro_step(
    model,
    batch,
    train_cfg,
    precision: str,
    loss_scale: float = 1.0,
) -> dict:
    input_ids = batch[:, :-1]
    labels = batch[:, 1:]  # 右移一位，用前 t 个预测第 t+1 个 token
    with _autocast_context(precision, batch.device):
        output = model(input_ids, labels=labels)
    backward_loss = output.total_loss if output.total_loss is not None else output.loss
    loss_value = float(output.loss.detach().cpu())
    total_loss_value = float(backward_loss.detach().cpu())
    token_acc = None
    if train_cfg.get("track_token_acc", True):
        token_acc = _token_accuracy(output.logits.detach(), labels)
    # loss 先除以累积步数，几个 micro step 的梯度加起来才相当于在大 batch 上求平均
    (backward_loss / loss_scale).backward()
    metrics = {
        "loss": loss_value,
        "total_loss": total_loss_value,
        "ppl": _safe_exp(loss_value),
        "tokens": labels.numel(),
        "samples": batch.size(0),
        "seq_len": labels.size(1),
    }
    if output.moe_aux_loss is not None:
        metrics["moe_aux_loss"] = float(output.moe_aux_loss.detach().cpu())
    if token_acc is not None:
        metrics["token_acc"] = token_acc
    return metrics


def _finish_optimizer_step(model, optimizer, train_cfg) -> dict:
    grad_clip = train_cfg.get("grad_clip")
    if grad_clip:
        # 裁剪梯度范数，挡住偶发的大梯度把训练带飞
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            .detach()
            .cpu()
        )
    else:
        grad_norm = _grad_norm(model.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "grad_norm": grad_norm,
        "grad_clipped": bool(grad_clip and grad_norm > float(grad_clip)),
    }


def _weighted_micro_average(micro_metrics: list[dict], key: str) -> float:
    total_tokens = sum(m["tokens"] for m in micro_metrics)
    if not total_tokens:
        return float("nan")
    return sum(float(m[key]) * m["tokens"] for m in micro_metrics) / total_tokens


def train_one_step(model, batch, optimizer, train_cfg, precision: str | None = None) -> dict:
    precision = precision or str(train_cfg.get("precision", "fp32")).lower()
    metrics = _forward_backward_micro_step(
        model, batch, train_cfg, precision, loss_scale=1.0
    )
    metrics.update(_finish_optimizer_step(model, optimizer, train_cfg))
    metrics["gradient_accumulation_steps"] = 1
    metrics["micro_steps"] = 1
    metrics["tokens_per_micro_step"] = metrics["tokens"]
    metrics["effective_batch_tokens"] = metrics["tokens"]
    return metrics


def train_accumulated_step(
    model,
    micro_batches: list[torch.Tensor],
    optimizer,
    train_cfg,
    precision: str,
) -> dict:
    grad_accum = len(micro_batches)
    micro_metrics = []
    for micro_idx, batch in enumerate(micro_batches):
        sync_context = nullcontext()
        # DDP 下只在最后一个 micro step 同步梯度，前面几步 no_sync 省掉 all-reduce 通信
        if hasattr(model, "no_sync") and micro_idx < grad_accum - 1:
            sync_context = model.no_sync()
        with sync_context:
            micro_metrics.append(
                _forward_backward_micro_step(
                    model,
                    batch,
                    train_cfg,
                    precision,
                    loss_scale=float(grad_accum),
                )
            )

    optimizer_metrics = _finish_optimizer_step(model, optimizer, train_cfg)
    total_tokens = sum(m["tokens"] for m in micro_metrics)
    total_samples = sum(m["samples"] for m in micro_metrics)
    seq_len = micro_metrics[-1]["seq_len"] if micro_metrics else 0
    metrics = {
        "loss": _weighted_micro_average(micro_metrics, "loss"),
        "total_loss": _weighted_micro_average(micro_metrics, "total_loss"),
        "tokens": total_tokens,
        "samples": total_samples,
        "seq_len": seq_len,
        "gradient_accumulation_steps": grad_accum,
        "micro_steps": grad_accum,
        "tokens_per_micro_step": (
            total_tokens // grad_accum if grad_accum and total_tokens % grad_accum == 0
            else total_tokens / max(1, grad_accum)
        ),
        "effective_batch_tokens": total_tokens,
    }
    metrics["ppl"] = _safe_exp(metrics["loss"])
    if "moe_aux_loss" in micro_metrics[0]:
        metrics["moe_aux_loss"] = _weighted_micro_average(
            micro_metrics, "moe_aux_loss"
        )
    if "token_acc" in micro_metrics[0]:
        metrics["token_acc"] = _weighted_micro_average(micro_metrics, "token_acc")
    metrics.update(optimizer_metrics)
    return metrics


@torch.no_grad()
def validate(model, dataloader, max_batches: int, device=None,
             precision: str = "fp32") -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    batches = 0
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if device is not None:
            batch = batch.to(device)
        labels = batch[:, 1:]
        with _autocast_context(precision, batch.device):
            output = model(batch[:, :-1], labels=labels)
        tokens = labels.numel()
        total_loss += float(output.loss.detach().cpu()) * tokens
        total_correct += (output.logits.argmax(dim=-1) == labels).sum().item()
        total_tokens += tokens
        batches += 1
    model.train()
    loss = total_loss / total_tokens if total_tokens else float("nan")
    token_acc = total_correct / total_tokens if total_tokens else float("nan")
    return {
        "val_loss": loss,
        "val_ppl": _safe_exp(loss),
        "val_token_acc": token_acc,
        "val_tokens": total_tokens,
        "val_batches": batches,
    }


def pretrain(config_path: str) -> None:
    distributed = _init_distributed()
    cfg = _load_config(config_path)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    if grad_accum <= 0:
        raise ValueError("train.gradient_accumulation_steps must be positive")
    train_cfg["gradient_accumulation_steps"] = grad_accum
    seed_everything(train_cfg.get("seed", 1337) + distributed["rank"])
    device = _device(train_cfg.get("device", "auto"), distributed)
    precision, precision_runtime = _resolve_precision(train_cfg, device)

    model_cfg = ModelConfig(**cfg["model"])
    _check_training_config(model_cfg, data_cfg)
    _check_tokenizer(model_cfg, data_cfg)
    _check_token_file_metadata(
        data_cfg["train_path"], model_cfg, data_cfg.get("tokenizer_path")
    )
    if data_cfg.get("val_path"):
        _check_token_file_metadata(
            data_cfg["val_path"], model_cfg, data_cfg.get("tokenizer_path")
        )
    run_artifacts = _init_run_artifacts_distributed(train_cfg, config_path, distributed)
    run_id = run_artifacts["run_id"]
    run_dir = run_artifacts["run_dir"]
    metrics_path = run_artifacts["metrics_path"]
    checkpoint_dir = Path(train_cfg.get("checkpoint_dir") or (run_dir / "checkpoints"))
    train_cfg["checkpoint_dir"] = str(checkpoint_dir)
    started_at = _now_iso()
    if distributed["is_main"]:
        _write_yaml(run_artifacts["config_snapshot_path"], cfg)

    model = CausalLM(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])
    hardware_monitor = HardwareMonitor(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(hardware_monitor.device_index or 0)

    start_step = 0
    best_val_loss = float("inf")
    tokens_seen = 0
    if train_cfg.get("resume_from"):
        ckpt = load_checkpoint(train_cfg["resume_from"], model, optimizer)
        start_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        tokens_seen = int(ckpt.get("tokens_seen", 0))

    raw_model = model
    if distributed["enabled"]:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs.update({
                "device_ids": [distributed["local_rank"]],
                "output_device": distributed["local_rank"],
            })
        model = DDP(model, **ddp_kwargs)

    val_path = data_cfg.get("val_path")
    train_dataset = TokenDataset(
        data_cfg["train_path"],
        data_cfg["block_size"],
        split="train",
        val_split=float(data_cfg.get("val_split", 0.0)) if not val_path else 0.0,
    )
    if val_path:
        val_dataset = TokenDataset(val_path, data_cfg["block_size"])
    else:
        val_dataset = TokenDataset(
            data_cfg["train_path"],
            data_cfg["block_size"],
            split="val",
            val_split=float(data_cfg.get("val_split", 0.0)),
        )

    train_sampler = None
    if distributed["enabled"]:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=distributed["world_size"],
            rank=distributed["rank"],
            shuffle=True,
            drop_last=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=data_cfg.get("num_workers", 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 0),
        drop_last=False,
    )

    param_count = count_parameters(raw_model)
    tokens_per_micro_step = (
        train_cfg["batch_size"]
        * data_cfg["block_size"]
        * distributed["world_size"]
    )
    effective_batch_tokens = tokens_per_micro_step * grad_accum
    if not tokens_seen:
        tokens_seen = start_step * effective_batch_tokens

    if distributed["is_main"]:
        logger.info("model params: %.2f M", param_count / 1e6)
        logger.info("train samples=%d val samples=%d device=%s world_size=%d",
                    len(train_dataset), len(val_dataset), device,
                    distributed["world_size"])
        logger.info(
            "micro_batch=%d block_size=%d grad_accum=%d "
            "tokens_per_micro_step=%d effective_batch_tokens=%d",
            train_cfg["batch_size"],
            data_cfg["block_size"],
            grad_accum,
            tokens_per_micro_step,
            effective_batch_tokens,
        )
        logger.info(
            "precision=%s autocast=%s cuda_bf16_supported=%s "
            "cuda_matmul_allow_tf32=%s cudnn_allow_tf32=%s",
            precision,
            precision_runtime["autocast_enabled"],
            precision_runtime["cuda_bf16_supported"],
            precision_runtime["cuda_matmul_allow_tf32"],
            precision_runtime["cudnn_allow_tf32"],
        )
        logger.info("run_dir=%s", run_dir)

    step = start_step
    start_time = time.time()
    ema_loss = None
    ema_beta = float(train_cfg.get("loss_ema_beta", 0.95))
    metrics_interval = int(train_cfg.get("metrics_interval", 1))
    hardware_metrics_interval = int(
        train_cfg.get("hardware_metrics_interval", train_cfg.get("log_interval", 10))
    )
    latest_checkpoint = checkpoint_dir / "ckpt.pt"
    best_checkpoint = checkpoint_dir / train_cfg.get("best_checkpoint_name", "best.pt")
    hardware_start = hardware_monitor.sample(external=hardware_metrics_interval > 0)
    run_info = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": started_at,
        "config_path": config_path,
        "config_snapshot_path": str(run_artifacts["config_snapshot_path"]),
        "metrics_path": str(metrics_path),
        "summary_path": str(run_artifacts["summary_path"]),
        "latest_checkpoint": str(latest_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "device": str(device),
        "precision": precision,
        "precision_runtime": precision_runtime,
        "model_params": param_count,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seed": train_cfg.get("seed", 1337),
        "micro_batch_size": train_cfg["batch_size"],
        "gradient_accumulation_steps": grad_accum,
        "global_batch_size": train_cfg["batch_size"] * distributed["world_size"],
        "tokens_per_micro_step": tokens_per_micro_step,
        "effective_batch_tokens": effective_batch_tokens,
        "model": cfg["model"],
        "data": cfg["data"],
        "train": cfg["train"],
        "hardware": hardware_start,
    }
    run_info["distributed"] = distributed
    if distributed["is_main"]:
        _write_json(run_artifacts["run_info_path"], run_info)

    writer = JsonlMetricWriter(metrics_path, append=False) if distributed["is_main"] else None
    if writer:
        writer.write("run_start", {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "config_path": config_path,
            "device": str(device),
            **precision_runtime,
            "model_params": param_count,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "tokenizer_path": data_cfg.get("tokenizer_path"),
            "batch_size": train_cfg["batch_size"],
            "global_batch_size": train_cfg["batch_size"] * distributed["world_size"],
            "micro_batch_size": train_cfg["batch_size"],
            "gradient_accumulation_steps": grad_accum,
            "world_size": distributed["world_size"],
            "block_size": data_cfg["block_size"],
            "tokens_per_micro_step": tokens_per_micro_step,
            "effective_batch_tokens": effective_batch_tokens,
            "max_steps": train_cfg["max_steps"],
            "start_step": start_step,
            "tokens_seen": tokens_seen,
            "metrics_path": str(metrics_path),
            **hardware_start,
        })

    model.train()
    try:
        if len(train_loader) == 0:
            raise ValueError(
                "training dataset is too small for the configured batch_size/block_size"
            )
        epoch = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_iter = iter(train_loader)
        while step < train_cfg["max_steps"]:
            lr = get_cosine_lr(
                step,
                train_cfg["warmup_steps"],
                train_cfg["max_steps"],
                train_cfg["lr"],
                train_cfg["min_lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            micro_batches = []
            for _micro_idx in range(grad_accum):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    epoch += 1
                    if train_sampler is not None:
                        train_sampler.set_epoch(epoch)
                    train_iter = iter(train_loader)
                    batch = next(train_iter)
                micro_batches.append(batch.to(device))
            step_start = time.perf_counter()
            metrics = train_accumulated_step(
                model, micro_batches, optimizer, train_cfg, precision=precision
            )
            step_time = time.perf_counter() - step_start
            step += 1
            local_tokens = metrics["tokens"]
            local_samples = metrics["samples"]
            global_tokens = local_tokens * distributed["world_size"]
            global_samples = local_samples * distributed["world_size"]
            metrics["local_tokens"] = local_tokens
            metrics["local_samples"] = local_samples
            metrics["global_tokens"] = global_tokens
            metrics["global_samples"] = global_samples
            metrics["tokens"] = global_tokens
            metrics["samples"] = global_samples
            metrics["tokens_per_micro_step"] = tokens_per_micro_step
            metrics["effective_batch_tokens"] = effective_batch_tokens
            tokens_seen += global_tokens
            elapsed = time.time() - start_time
            ema_loss = metrics["loss"] if ema_loss is None else (
                ema_beta * ema_loss + (1.0 - ema_beta) * metrics["loss"]
            )
            metrics.update({
                "run_id": run_id,
                "step": step,
                "optimizer_step": step,
                "micro_steps_seen": step * grad_accum,
                "lr": lr,
                "loss_ema": ema_loss,
                "tokens_seen": tokens_seen,
                "elapsed_sec": elapsed,
                "step_time_sec": step_time,
                "tokens_per_sec": metrics["tokens"] / max(step_time, 1e-9),
                "samples_per_sec": metrics["samples"] / max(step_time, 1e-9),
            })
            sample_hardware = (
                hardware_metrics_interval > 0
                and (step == 1 or step % hardware_metrics_interval == 0)
            )
            metrics.update(hardware_monitor.sample(external=sample_hardware))
            if writer and (step == 1 or step % metrics_interval == 0):
                writer.write("train", metrics)
            if distributed["is_main"] and (
                step == 1 or step % train_cfg.get("log_interval", 10) == 0
            ):
                logger.info(
                    "step=%d micro=%d loss=%s ema=%s ppl=%s acc=%s "
                    "grad_norm=%s lr=%.2e tok/s=%.1f eff_tok=%d mem=%sMB "
                    "peak=%sMB gpu=%s%%",
                    step,
                    grad_accum,
                    _fmt(metrics["loss"]),
                    _fmt(metrics["loss_ema"]),
                    _fmt(metrics["ppl"], 2),
                    _fmt(metrics.get("token_acc", float("nan"))),
                    _fmt(metrics["grad_norm"]),
                    lr,
                    metrics["tokens_per_sec"],
                    effective_batch_tokens,
                    _fmt(metrics.get("hw_memory_allocated_mb"), 1),
                    _fmt(metrics.get("hw_memory_peak_allocated_mb"), 1),
                    _fmt(metrics.get("hw_gpu_utilization_percent"), 1),
                )
            if step % train_cfg["eval_interval"] == 0:
                val_metrics = None
                if distributed["is_main"]:
                    val_metrics = validate(
                        raw_model, val_loader, train_cfg["eval_batches"], device,
                        precision=precision,
                    )
                    val_metrics.update({
                        "run_id": run_id,
                        "step": step,
                        "optimizer_step": step,
                        "micro_steps_seen": step * grad_accum,
                        "tokens_seen": tokens_seen,
                        "elapsed_sec": time.time() - start_time,
                        "gradient_accumulation_steps": grad_accum,
                        "effective_batch_tokens": effective_batch_tokens,
                    })
                    logger.info(
                        "step=%d val_loss=%s val_ppl=%s val_acc=%s",
                        step,
                        _fmt(val_metrics["val_loss"]),
                        _fmt(val_metrics["val_ppl"], 2),
                        _fmt(val_metrics["val_token_acc"]),
                    )
                    writer.write("eval", val_metrics)
                    val_loss = val_metrics["val_loss"]
                    if train_cfg.get("save_best", True) and math.isfinite(val_loss):
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            save_checkpoint(
                                best_checkpoint, raw_model, optimizer, step, cfg,
                                extra_state={
                                    "tokens_seen": tokens_seen,
                                    "best_val_loss": best_val_loss,
                                },
                            )
                            writer.write("checkpoint", {
                                "run_id": run_id,
                                "step": step,
                                "kind": "best",
                                "path": str(best_checkpoint),
                                "best_val_loss": best_val_loss,
                                "tokens_seen": tokens_seen,
                                "gradient_accumulation_steps": grad_accum,
                                "effective_batch_tokens": effective_batch_tokens,
                            })
                _barrier(distributed)
            if distributed["is_main"] and step % train_cfg["save_interval"] == 0:
                save_checkpoint(
                    latest_checkpoint, raw_model, optimizer, step, cfg,
                    extra_state={
                        "tokens_seen": tokens_seen,
                        "best_val_loss": best_val_loss,
                    },
                )
                writer.write("checkpoint", {
                    "run_id": run_id,
                    "step": step,
                    "kind": "latest",
                    "path": str(latest_checkpoint),
                    "best_val_loss": best_val_loss,
                    "tokens_seen": tokens_seen,
                    "gradient_accumulation_steps": grad_accum,
                    "effective_batch_tokens": effective_batch_tokens,
                })
            if step % train_cfg["save_interval"] == 0:
                _barrier(distributed)

        if distributed["is_main"]:
            save_checkpoint(
                latest_checkpoint, raw_model, optimizer, step, cfg,
                extra_state={"tokens_seen": tokens_seen, "best_val_loss": best_val_loss},
            )
        _barrier(distributed)
        ended_at = _now_iso()
        summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "completed",
            "started_at": started_at,
            "ended_at": ended_at,
            "step": step,
            "tokens_seen": tokens_seen,
            "precision": precision,
            "micro_batch_size": train_cfg["batch_size"],
            "gradient_accumulation_steps": grad_accum,
            "tokens_per_micro_step": tokens_per_micro_step,
            "effective_batch_tokens": effective_batch_tokens,
            "micro_steps_seen": step * grad_accum,
            "elapsed_sec": time.time() - start_time,
            "best_val_loss": best_val_loss,
            "latest_checkpoint": str(latest_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "metrics_path": str(metrics_path),
            "config_snapshot_path": str(run_artifacts["config_snapshot_path"]),
        }
        if writer:
            _write_json(run_artifacts["summary_path"], summary)
            writer.write("run_end", summary)
    except Exception as exc:
        failed_summary = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "status": "failed",
            "started_at": started_at,
            "ended_at": _now_iso(),
            "step": step,
            "tokens_seen": tokens_seen,
            "precision": precision,
            "micro_batch_size": train_cfg["batch_size"],
            "gradient_accumulation_steps": grad_accum,
            "tokens_per_micro_step": tokens_per_micro_step,
            "effective_batch_tokens": effective_batch_tokens,
            "micro_steps_seen": step * grad_accum,
            "elapsed_sec": time.time() - start_time,
            "best_val_loss": best_val_loss,
            "latest_checkpoint": str(latest_checkpoint),
            "best_checkpoint": str(best_checkpoint),
            "metrics_path": str(metrics_path),
            "config_snapshot_path": str(run_artifacts["config_snapshot_path"]),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if writer:
            _write_json(run_artifacts["summary_path"], failed_summary)
            writer.write("run_error", failed_summary)
        raise
    finally:
        if writer:
            writer.close()
        _cleanup_distributed(distributed)
