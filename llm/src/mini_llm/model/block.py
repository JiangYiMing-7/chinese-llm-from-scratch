from torch import Tensor, nn

from mini_llm.model.attention import CausalSelfAttention
from mini_llm.model.config import ModelConfig
from mini_llm.model.ffn import SwiGLUFeedForward
from mini_llm.model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLUFeedForward(config.d_model, config.ffn_hidden_dim,
                                     config.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
