from dataclasses import dataclass


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


def validate_config(config: ModelConfig) -> None:
    if config.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    if config.d_model != config.n_heads * config.head_dim:
        raise ValueError("d_model must equal n_heads * head_dim")
    if config.head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if config.n_layers <= 0 or config.n_heads <= 0:
        raise ValueError("n_layers and n_heads must be positive")


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
