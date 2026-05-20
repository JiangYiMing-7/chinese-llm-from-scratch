import torch

from model.causal_lm import CausalLM
from model.config import tiny_config
from model.generation import generate


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
