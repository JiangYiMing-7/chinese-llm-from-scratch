import pytest
import torch

from mini_llm.model.rope import RotaryEmbedding, apply_rope


def test_rope_shape_and_positions():
    rope = RotaryEmbedding(head_dim=8, max_seq_len=16)
    cos, sin = rope(4)
    x = torch.ones(2, 3, 4, 8)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    assert not torch.allclose(y[:, :, 0], y[:, :, 1])


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7, max_seq_len=16)
