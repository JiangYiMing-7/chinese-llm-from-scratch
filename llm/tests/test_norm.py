import torch

from mini_llm.model.norm import RMSNorm


def test_rmsnorm_shape_and_grad():
    norm = RMSNorm(8)
    x = torch.randn(2, 3, 8, requires_grad=True)
    y = norm(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert norm.weight.grad is not None
