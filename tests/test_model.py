import torch
import pytest

from model.causal_lm import CausalLM
from model.config import ALLOWED_NORM_ARCHES, tiny_config, validate_config
from model.generation import generate
from utils.params import count_parameters


def test_causal_lm_loss_backward_and_generate():
    cfg = tiny_config()
    cfg.vocab_size = 128
    model = CausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss.ndim == 0
    out.loss.backward()
    assert model.token_embedding.weight.grad is not None

    generated = generate(model, input_ids[:, :2], max_new_tokens=4, temperature=0)
    assert generated.shape == (2, 6)
    assert int(generated.max()) < cfg.vocab_size


@pytest.mark.parametrize("norm_arch", ALLOWED_NORM_ARCHES)
def test_causal_lm_norm_arch_variants_forward_backward(norm_arch):
    cfg = tiny_config()
    cfg.vocab_size = 128
    cfg.norm_arch = norm_arch
    model = CausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss.ndim == 0
    out.loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_norm_arch_validation_and_parameter_counts():
    cfg = tiny_config()
    cfg.norm_arch = "not_a_real_arch"
    with pytest.raises(ValueError, match="norm_arch"):
        validate_config(cfg)

    cfg = tiny_config()
    cfg.ffn_type = "not_a_real_ffn"
    with pytest.raises(ValueError, match="ffn_type"):
        validate_config(cfg)

    counts = {}
    for norm_arch in ALLOWED_NORM_ARCHES:
        variant_cfg = tiny_config()
        variant_cfg.norm_arch = norm_arch
        counts[norm_arch] = count_parameters(CausalLM(variant_cfg))

    assert counts["pre_norm"] == counts["post_norm"]
    assert counts["pre_norm"] == counts["branch_norm"]
    assert counts["pre_output_norm"] == (
        counts["pre_norm"] + 2 * tiny_config().n_layers * tiny_config().d_model
    )


def test_causal_lm_moe_forward_backward_and_aux_loss():
    cfg = tiny_config()
    cfg.vocab_size = 128
    cfg.ffn_type = "moe"
    cfg.moe_num_experts = 4
    cfg.moe_top_k = 1
    cfg.moe_expert_hidden_dim = cfg.ffn_hidden_dim
    cfg.moe_aux_loss_coef = 0.01
    model = CausalLM(cfg)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
    labels = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(input_ids, labels=labels)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss.ndim == 0
    assert out.total_loss.ndim == 0
    assert out.moe_aux_loss.ndim == 0
    assert out.total_loss > out.loss

    out.total_loss.backward()
    router_grads = [
        block.ffn.router.weight.grad
        for block in model.blocks
        if hasattr(block.ffn, "router")
    ]
    assert router_grads
    assert all(grad is not None for grad in router_grads)


def test_moe_config_validation():
    cfg = tiny_config()
    cfg.ffn_type = "moe"
    cfg.moe_num_experts = 1
    with pytest.raises(ValueError, match="moe_num_experts"):
        validate_config(cfg)

    cfg = tiny_config()
    cfg.ffn_type = "moe"
    cfg.moe_top_k = 8
    cfg.moe_num_experts = 4
    with pytest.raises(ValueError, match="moe_top_k"):
        validate_config(cfg)
