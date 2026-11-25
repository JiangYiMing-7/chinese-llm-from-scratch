# import json
# from pathlib import Path

# import numpy as np
# import torch
# from torch import Tensor
# from torch.utils.data import Dataset


# def _load_token_array(path: Path):
#     suffix = path.suffix
#     if suffix == ".npy":
#         return np.load(path, mmap_mode="r")
#     if suffix == ".bin":
#         with open(str(path) + ".meta.json", encoding="utf-8") as f:
#             meta = json.load(f)
#         return np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")
#     if suffix == ".pt":
#         payload = torch.load(path, map_location="cpu")
#         if isinstance(payload, dict) and "token_ids" in payload:
#             return payload["token_ids"]
#         if torch.is_tensor(payload):
#             return payload
#         raise ValueError(f"unsupported .pt payload in {path}")
#     raise ValueError(f"unsupported token file: {path}")


# class TokenDataset(Dataset):
#     """Dataset over a continuous token id array.

#     Supports .npy, data/pretrain_dataset.py .bin+.meta.json, and the older .pt
#     payload with a token_ids tensor.
#     """

#     def __init__(self, token_file: str | Path, block_size: int,
#                  split: str = "train", val_split: float = 0.0):
#         self.token_file = Path(token_file)
#         self.block_size = block_size
#         tokens = _load_token_array(self.token_file)
#         n = len(tokens)
#         if val_split > 0:
#             split_at = max(block_size + 2, int(n * (1 - val_split)))
#             if split == "train":
#                 self.offset = 0
#                 self.length = split_at
#             elif split == "val":
#                 self.offset = split_at
#                 self.length = n - split_at
#             else:
#                 raise ValueError("split must be train or val")
#         else:
#             self.offset = 0
#             self.length = n
#         self.tokens = tokens
#         self.num_samples = max(0, (self.length - 1) // block_size)

#     def __len__(self) -> int:
#         return self.num_samples

#     def __getitem__(self, idx: int) -> Tensor:
#         if idx < 0:
#             idx += self.num_samples
#         if not 0 <= idx < self.num_samples:
#             raise IndexError(idx)
#         start = self.offset + idx
#         chunk = self.tokens[start:start + self.block_size + 1]
#         if torch.is_tensor(chunk):
#             return chunk.long()
#         return torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True))


import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


# 兼容三种 token 文件 .npy / .bin(+meta) / 旧的 .pt，前两种用 mmap，不把整个语料读进内存
def _load_token_array(path: Path):
    suffix = path.suffix
    if suffix == ".npy":
        return np.load(path, mmap_mode="r")  # mmap 模式，按需读盘
    if suffix == ".bin":
        with open(str(path) + ".meta.json", encoding="utf-8") as f:
            meta = json.load(f)  # .bin 自己不带 dtype，靠旁边的 meta.json 说明
        return np.memmap(path, dtype=np.dtype(meta["dtype"]), mode="r")
    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "token_ids" in payload:
            return payload["token_ids"]
        if torch.is_tensor(payload):
            return payload
        raise ValueError(f"unsupported .pt payload in {path}")
    raise ValueError(f"unsupported token file: {path}")


class TokenDataset(Dataset):
    """Dataset over a continuous token id array.

    Supports .npy, data/pretrain_dataset.py .bin+.meta.json, and the older .pt
    payload with a token_ids tensor.

    Each sample is a non-overlapping block of block_size + 1 tokens. The
    training loop then uses batch[:, :-1] as inputs and batch[:, 1:] as labels.
    """

    def __init__(self, token_file: str | Path, block_size: int,
                 split: str = "train", val_split: float = 0.0):
        self.token_file = Path(token_file)
        self.block_size = block_size
        tokens = _load_token_array(self.token_file)
        n = len(tokens)

        if val_split > 0:
            split_at = max(block_size + 2, int(n * (1 - val_split)))
            if split == "train":
                self.offset = 0
                self.length = split_at
            elif split == "val":
                self.offset = split_at
                self.length = n - split_at
            else:
                raise ValueError("split must be train or val")
        else:
            self.offset = 0
            self.length = n

        self.tokens = tokens
        # 非重叠切块，每个样本 block_size+1 个 token，所以样本数是 (length-1)//block_size
        self.num_samples = max(0, (self.length - 1) // block_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tensor:
        if idx < 0:
            idx += self.num_samples
        if not 0 <= idx < self.num_samples:
            raise IndexError(idx)

        # 用 idx*block_size 定位，块之间不重叠，多取 1 个 token 给训练右移做 label 用
        start = self.offset + idx * self.block_size
        chunk = self.tokens[start:start + self.block_size + 1]

        if torch.is_tensor(chunk):
            return chunk.long()
        return torch.from_numpy(np.array(chunk, dtype=np.int64, copy=True))