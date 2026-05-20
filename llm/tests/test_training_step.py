import json

import numpy as np
import torch

from mini_llm.model.causal_lm import CausalLM
from mini_llm.model.config import tiny_config
from mini_llm.train.checkpoint import load_checkpoint, save_checkpoint
from mini_llm.train.dataset import TokenDataset
from mini_llm.train.optim import build_optimizer
from mini_llm.train.pretrain import train_one_step


def test_dataset_formats_and_training_step(tmp_path):
    tokens = np.arange(256, dtype=np.int64) % 64
    npy = tmp_path / "ids.npy"
    np.save(npy, tokens)
    ds = TokenDataset(npy, block_size=16)
    assert len(ds) == 240
    assert ds[0].shape == (17,)

    pt = tmp_path / "ids.pt"
    torch.save({"token_ids": torch.from_numpy(tokens)}, pt)
    assert TokenDataset(pt, block_size=16)[0].shape == (17,)

    bin_path = tmp_path / "ids.bin"
    tokens.astype(np.int16).tofile(bin_path)
    with open(str(bin_path) + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"dtype": "int16", "total_tokens": len(tokens)}, f)
    assert TokenDataset(bin_path, block_size=16)[0].shape == (17,)

    cfg = tiny_config()
    cfg.vocab_size = 64
    cfg.max_seq_len = 16
    model = CausalLM(cfg)
    optimizer = build_optimizer(model, lr=1e-3, weight_decay=0.0)
    before = model.token_embedding.weight.detach().clone()
    batch = torch.stack([ds[0], ds[1]])
    metrics = train_one_step(model, batch, optimizer, {"grad_clip": 1.0})
    assert metrics["loss"] > 0
    assert not torch.allclose(before, model.token_embedding.weight)

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, optimizer, step=1, config={"model": cfg.__dict__})
    restored = CausalLM(cfg)
    state = load_checkpoint(ckpt, restored)
    assert state["step"] == 1
