import math

import torch


def build_optimizer(model, lr: float, weight_decay: float) -> torch.optim.AdamW:
    # 权重衰减只加在二维以上的矩阵权重上，norm/bias 这种一维参数不衰减，gpt 系列的惯例
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:  # 冻结的参数不进优化器
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),  # beta2 用 0.95 不是默认 0.999，预训练里更稳一点
    )


def get_cosine_lr(step, warmup_steps, max_steps, lr, min_lr) -> float:
    # 先线性 warmup 到峰值 lr，再余弦退火到 min_lr
    if step < warmup_steps:
        return lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr  # 训练后期都按 min_lr 走，不再降
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)
