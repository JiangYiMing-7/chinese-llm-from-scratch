import torch
from torch import Tensor

from model.causal_lm import CausalLM


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
        # 超过 max_seq_len 就只喂最近的窗口，没做 KV cache，纯靠每步重算（慢但简单）
        idx = out[:, -model.config.max_seq_len:]
        logits = model(idx).logits[:, -1, :]  # 只取最后一个位置的 logits 来预测下一个 token
        if top_k is not None:
            # 只保留概率最高的 top_k 个，其余置 -inf，softmax 后自然为 0
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
        if temperature == 0:
            # temperature=0 退化成贪心（直接取 argmax），避免除零
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        # 整个 batch 都出 eos 才停，没做逐样本提前停
        if eos_token_id is not None and torch.all(next_id == eos_token_id):
            break
    if was_training:
        model.train()
    return out
