from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from mini_llm.model.causal_lm import CausalLM
from mini_llm.model.config import ModelConfig
from mini_llm.train.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.train.dataset import TokenDataset
from mini_llm.train.optim import build_optimizer, get_cosine_lr
from mini_llm.utils.logging import get_logger
from mini_llm.utils.params import count_parameters
from mini_llm.utils.seed import seed_everything

logger = get_logger("pretrain")


def _device(name: str) -> torch.device:
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


def train_one_step(model, batch, optimizer, train_cfg) -> dict:
    input_ids = batch[:, :-1]
    labels = batch[:, 1:]
    output = model(input_ids, labels=labels)
    output.loss.backward()
    grad_clip = train_cfg.get("grad_clip")
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": float(output.loss.detach().cpu())}


@torch.no_grad()
def validate(model, dataloader, max_batches: int, device=None) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        if device is not None:
            batch = batch.to(device)
        output = model(batch[:, :-1], labels=batch[:, 1:])
        losses.append(float(output.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def pretrain(config_path: str) -> None:
    cfg = _load_config(config_path)
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    seed_everything(train_cfg.get("seed", 1337))
    device = _device(train_cfg.get("device", "auto"))

    model_cfg = ModelConfig(**cfg["model"])
    model = CausalLM(model_cfg).to(device)
    optimizer = build_optimizer(model, train_cfg["lr"], train_cfg["weight_decay"])

    start_step = 0
    if train_cfg.get("resume_from"):
        ckpt = load_checkpoint(train_cfg["resume_from"], model, optimizer)
        start_step = int(ckpt.get("step", 0))

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
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

    logger.info("model params: %.2f M", count_parameters(model) / 1e6)
    logger.info("train samples=%d val samples=%d device=%s",
                len(train_dataset), len(val_dataset), device)

    step = start_step
    model.train()
    while step < train_cfg["max_steps"]:
        for batch in train_loader:
            if step >= train_cfg["max_steps"]:
                break
            lr = get_cosine_lr(
                step,
                train_cfg["warmup_steps"],
                train_cfg["max_steps"],
                train_cfg["lr"],
                train_cfg["min_lr"],
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = batch.to(device)
            metrics = train_one_step(model, batch, optimizer, train_cfg)
            step += 1
            if step == 1 or step % train_cfg.get("log_interval", 10) == 0:
                logger.info("step=%d loss=%.4f lr=%.2e", step, metrics["loss"], lr)
            if step % train_cfg["eval_interval"] == 0:
                val_loss = validate(model, val_loader, train_cfg["eval_batches"], device)
                logger.info("step=%d val_loss=%.4f", step, val_loss)
            if step % train_cfg["save_interval"] == 0:
                save_checkpoint(Path(train_cfg["out_dir"]) / "ckpt.pt",
                                model, optimizer, step, cfg)

    save_checkpoint(Path(train_cfg["out_dir"]) / "ckpt.pt", model, optimizer, step, cfg)
