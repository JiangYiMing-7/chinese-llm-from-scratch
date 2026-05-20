import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
