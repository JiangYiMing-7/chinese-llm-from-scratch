from torch import Tensor, nn

from model.attention import CausalSelfAttention
from model.config import ModelConfig
from model.ffn import MoEFeedForward, SwiGLUFeedForward
from model.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # norm_arch 决定归一化放哪，做 PreNorm/PostNorm 这类消融时只改它
        self.norm_arch = config.norm_arch
        self.attention = CausalSelfAttention(config)
        if config.ffn_type == "moe":
            self.ffn = MoEFeedForward(
                config.d_model,
                config.moe_expert_hidden_dim or config.ffn_hidden_dim,
                config.moe_num_experts,
                config.moe_top_k,
                config.bias,
            )
        else:
            self.ffn = SwiGLUFeedForward(
                config.d_model, config.ffn_hidden_dim, config.bias
            )

        if self.norm_arch in {"pre_norm", "post_norm", "pre_output_norm"}:
            self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
            self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        elif self.norm_arch == "branch_norm":
            self.attn_branch_norm = RMSNorm(config.d_model, config.norm_eps)
            self.ffn_branch_norm = RMSNorm(config.d_model, config.norm_eps)

        if self.norm_arch == "pre_output_norm":
            self.attn_output_norm = RMSNorm(config.d_model, config.norm_eps)
            self.ffn_output_norm = RMSNorm(config.d_model, config.norm_eps)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_arch == "pre_norm":
            # baseline，残差流保持干净，只对进分支前的输入做 norm
            x = x + self.attention(self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x

        if self.norm_arch == "post_norm":
            x = self.attn_norm(x + self.attention(x))
            x = self.ffn_norm(x + self.ffn(x))
            return x

        if self.norm_arch == "pre_output_norm":
            # 残差相加之后又套一层 norm，故意破坏“干净残差流”，用来和 pre_norm 对照
            x = self.attn_output_norm(x + self.attention(self.attn_norm(x)))
            x = self.ffn_output_norm(x + self.ffn(self.ffn_norm(x)))
            return x

        if self.norm_arch == "branch_norm":
            x = x + self.attn_branch_norm(self.attention(x))
            x = x + self.ffn_branch_norm(self.ffn(x))
            return x

        raise ValueError(f"unsupported norm_arch: {self.norm_arch}")
