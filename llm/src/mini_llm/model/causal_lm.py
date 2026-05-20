from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mini_llm.model.block import TransformerBlock
from mini_llm.model.config import ModelConfig, validate_config
from mini_llm.model.norm import RMSNorm


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None


class CausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        validate_config(config)
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor, labels: Tensor | None = None) -> CausalLMOutput:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("input length exceeds max_seq_len")
        x = self.token_embedding(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
        return CausalLMOutput(logits=logits, loss=loss)
