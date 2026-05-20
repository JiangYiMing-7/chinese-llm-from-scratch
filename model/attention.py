import math

import torch
from torch import Tensor, nn

from model.config import ModelConfig
from model.rope import RotaryEmbedding, apply_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.d_model = config.d_model
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len,
                                    config.rope_theta)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.triu(torch.ones(config.max_seq_len, config.max_seq_len,
                                     dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask, persistent=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        cos, sin = self.rope(seq_len, device=x.device, dtype=x.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask[None, None, :, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.o_proj(out)
