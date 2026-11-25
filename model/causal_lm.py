from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.block import TransformerBlock
from model.config import ModelConfig, validate_config
from model.norm import RMSNorm


@dataclass
class CausalLMOutput:
    logits: Tensor
    loss: Tensor | None = None
    total_loss: Tensor | None = None
    moe_aux_loss: Tensor | None = None


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
            # 输入 embedding 和输出 lm_head 共享权重，省参数，小模型里常这么干
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        # gpt2 那套初始化，std 0.02
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
        moe_aux_losses = [
            block.ffn.last_aux_loss
            for block in self.blocks
            if getattr(block.ffn, "last_aux_loss", None) is not None
        ]
        moe_aux_loss = (
            torch.stack(moe_aux_losses).mean() if moe_aux_losses else None
        )

        loss = None
        total_loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            # 模型里不做移位，input_ids 跟 labels 在训练那步已经错开一位对齐了，这里直接算 CE
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )
            total_loss = loss
            if moe_aux_loss is not None and self.config.moe_aux_loss_coef:
                total_loss = total_loss + self.config.moe_aux_loss_coef * moe_aux_loss
        return CausalLMOutput(
            logits=logits,
            loss=loss,
            total_loss=total_loss,
            moe_aux_loss=moe_aux_loss,
        )
