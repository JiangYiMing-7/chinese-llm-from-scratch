import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        # SwiGLU，gate 过 silu 做门控乘上 up，再 down 投回 d_model
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUFeedForward(d_model, hidden_dim, bias=bias)
                for _ in range(num_experts)
            ]
        )
        self.last_aux_loss: Tensor | None = None

    def _load_balance_loss(self, router_probs: Tensor, topk_indices: Tensor) -> Tensor:
        # Switch Transformer 那个负载均衡 loss，免得 token 全挤到少数几个 expert 上
        expert_mask = F.one_hot(topk_indices, self.num_experts).float().sum(dim=1)
        expert_fraction = (expert_mask / self.top_k).mean(dim=0).detach()
        prob_fraction = router_probs.mean(dim=0)
        return self.num_experts * torch.sum(expert_fraction * prob_fraction)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, d_model = x.shape
        flat_x = x.reshape(-1, d_model)
        router_logits = self.router(flat_x)
        # 路由这步 softmax 用 fp32 算，bf16 直接算不太稳
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        topk_probs, topk_indices = torch.topk(
            router_probs, k=self.top_k, dim=-1
        )
        if self.top_k > 1:
            topk_probs = topk_probs / topk_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-9)
        self.last_aux_loss = self._load_balance_loss(router_probs, topk_indices)

        flat_out = torch.zeros_like(flat_x)
        # 按 expert 一个个把分给它的 token 取出来算，再 index_add 写回，写得直白没特意省显存
        for expert_id, expert in enumerate(self.experts):
            matches = topk_indices == expert_id
            if not matches.any():
                continue
            token_idx, route_idx = matches.nonzero(as_tuple=True)
            expert_out = expert(flat_x[token_idx])
            weights = topk_probs[token_idx, route_idx].to(dtype=expert_out.dtype)
            weighted_out = expert_out * weights.unsqueeze(-1)
            flat_out.index_add_(0, token_idx, weighted_out.to(dtype=flat_out.dtype))

        return flat_out.reshape(batch_size, seq_len, d_model)
