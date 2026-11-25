import torch

from model.attention import CausalSelfAttention
from model.config import tiny_config


def test_attention_shape_grad_and_causal_mask():
    cfg = tiny_config()
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 6, cfg.d_model, requires_grad=True)
    out = attn(x)
    assert out.shape == x.shape
    out.sum().backward()
    for module in (attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj):
        assert module.weight.grad is not None

    attn.eval()
    x1 = torch.randn(1, 6, cfg.d_model)
    x2 = x1.clone()
    x2[:, 1:] = torch.randn_like(x2[:, 1:])
    y1 = attn(x1)
    y2 = attn(x2)
    assert torch.allclose(y1[:, 0], y2[:, 0], atol=1e-5)
