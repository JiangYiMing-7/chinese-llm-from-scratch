from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step, config, extra_state=None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写 .tmp 再 rename，万一中途被打断也不会把原来好的 checkpoint 写坏，相当于原子替换
    tmp_path = path.with_name(path.name + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config,
    }
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path, model, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt
