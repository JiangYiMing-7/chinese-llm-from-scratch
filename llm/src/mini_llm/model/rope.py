import torch
from torch import Tensor, nn


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :],
                             persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :],
                             persistent=False)

    def forward(self, seq_len: int, device=None, dtype=None) -> tuple[Tensor, Tensor]:
        if seq_len > self.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.max_seq_len}")
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        if device is not None:
            cos = cos.to(device=device)
            sin = sin.to(device=device)
        if dtype is not None:
            cos = cos.to(dtype=dtype)
            sin = sin.to(dtype=dtype)
        return cos, sin
