#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.causal_lm import CausalLM
from model.config import ModelConfig
from model.generation import generate
from pretrain.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Sample token ids from a checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt_ids", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--eos-token-id", type=int, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config")
    if cfg is None:
        raise ValueError("checkpoint does not contain config")
    model = CausalLM(ModelConfig(**cfg["model"]))
    load_checkpoint(args.checkpoint, model)
    prompt = torch.from_numpy(np.load(args.prompt_ids).astype(np.int64))[None, :]
    out = generate(
        model,
        prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_token_id=args.eos_token_id,
    )
    print(out[0].tolist())


if __name__ == "__main__":
    main()
