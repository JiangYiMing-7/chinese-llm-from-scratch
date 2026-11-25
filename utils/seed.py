import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    # 把 python/numpy/torch 三处随机源一起固定，实验才可复现
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # 多卡时每张卡的 CUDA 随机种子也要设
