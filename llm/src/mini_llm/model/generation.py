import torch
from torch import Tensor

from mini_llm.model.causal_lm import CausalLM


@torch.no_grad()
def generate(
    model: CausalLM,
    input_ids: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    eos_token_id: int | None = None,
) -> Tensor:
    was_training = model.training
    model.eval()
    out = input_ids
    for _ in range(max_new_tokens):
        idx = out[:, -model.config.max_seq_len:]
        logits = model(idx).logits[:, -1, :]
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
        if temperature == 0:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        if eos_token_id is not None and torch.all(next_id == eos_token_id):
            break
    if was_training:
        model.train()
    return out
