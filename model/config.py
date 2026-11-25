from dataclasses import dataclass


ALLOWED_NORM_ARCHES = (
    "pre_norm",
    "post_norm",
    "pre_output_norm",
    "branch_norm",
)

ALLOWED_FFN_TYPES = (
    "dense",
    "moe",
)


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int
    n_layers: int
    n_heads: int
    d_model: int
    head_dim: int
    ffn_hidden_dim: int
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    bias: bool = False
    # 下面这俩是架构消融的开关，norm_arch 换 PreNorm/PostNorm，ffn_type 换 dense/moe
    norm_arch: str = "pre_norm"
    ffn_type: str = "dense"
    moe_num_experts: int = 4
    moe_top_k: int = 1
    moe_expert_hidden_dim: int | None = None
    moe_aux_loss_coef: float = 0.01


def validate_config(config: ModelConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    # d_model 必须正好等于 n_heads*head_dim，否则多头切分对不上
    if config.d_model != config.n_heads * config.head_dim:
        raise ValueError("d_model must equal n_heads * head_dim")
    # RoPE 按偶数奇数维配对转，head_dim 必须是偶数
    if config.head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if config.n_layers <= 0 or config.n_heads <= 0:
        raise ValueError("n_layers and n_heads must be positive")
    if config.norm_arch not in ALLOWED_NORM_ARCHES:
        allowed = ", ".join(ALLOWED_NORM_ARCHES)
        raise ValueError(f"norm_arch must be one of: {allowed}")
    if config.ffn_type not in ALLOWED_FFN_TYPES:
        allowed = ", ".join(ALLOWED_FFN_TYPES)
        raise ValueError(f"ffn_type must be one of: {allowed}")
    if config.ffn_type == "moe":
        if config.moe_num_experts <= 1:
            raise ValueError("moe_num_experts must be greater than 1 for MoE")
        if config.moe_top_k <= 0:
            raise ValueError("moe_top_k must be positive for MoE")
        if config.moe_top_k > config.moe_num_experts:
            raise ValueError("moe_top_k must not exceed moe_num_experts")
        hidden_dim = config.moe_expert_hidden_dim or config.ffn_hidden_dim
        if hidden_dim <= 0:
            raise ValueError("moe_expert_hidden_dim must be positive")
        if config.moe_aux_loss_coef < 0:
            raise ValueError("moe_aux_loss_coef must be non-negative")


def tiny_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=4096,
        max_seq_len=128,
        n_layers=2,
        n_heads=4,
        d_model=128,
        head_dim=32,
        ffn_hidden_dim=384,
    )
