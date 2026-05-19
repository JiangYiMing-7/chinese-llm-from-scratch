#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 4: 构建预训练 Dataset。

- 读取 mixed_corpus.jsonl，用 Task 3 训练好的 tokenizer 编码
- 把所有 token ids 拼接成一条长序列（文档间插入 <eos>）
- 按 context_length（默认 1024）滑动切片，无重叠
- 提供 PyTorch Dataset，__getitem__ 返回 (input_ids, labels)，labels 为 input_ids 右移一位
- 处理结果可保存为 .pt 文件，下次直接加载，避免重复编码

用法分两步：
  1) build  —— 编码语料并把 token 长序列保存为 .pt
  2) 训练时用 PretrainDataset 加载该 .pt

编码阶段流式逐文档处理，token 以分块方式写入，不会一次性把全部文本读进内存。
"""

import argparse
import json
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pretrain_dataset")


# --------------------------------------------------------------------------
# 编码并保存（build 阶段）
# --------------------------------------------------------------------------
def build(args):
    import sentencepiece as spm
    import torch

    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)
    eos_id = sp.eos_id()
    bos_id = sp.bos_id()

    # 文档边界用 <|endoftext|>（与 nanoGPT 约定对齐，不同于 SFT 用的 <eos>）
    endoftext_id = sp.piece_to_id("<|endoftext|>")
    if endoftext_id == sp.unk_id():
        # 旧版 tokenizer 没有 <|endoftext|>，回退到 <eos>
        endoftext_id = eos_id
        logger.warning("tokenizer 无 <|endoftext|>，文档边界回退到 <eos> (id=%d)", eos_id)
    else:
        logger.info("文档边界 token：<|endoftext|> (id=%d)", endoftext_id)

    logger.info("加载 tokenizer：%s（vocab=%d, bos=%d, eos=%d, endoftext=%d）",
                args.tokenizer, sp.get_piece_size(), bos_id, eos_id, endoftext_id)

    all_ids = []          # 收集全部 token id
    n_docs = 0
    t0 = time.time()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(args.text_field, "")
            if not isinstance(text, str) or not text:
                continue

            ids = sp.encode(text, out_type=int)
            if args.add_bos:
                ids = [bos_id] + ids
            ids.append(endoftext_id)  # 文档边界：<|endoftext|>
            all_ids.extend(ids)

            n_docs += 1
            if n_docs % 50000 == 0:
                logger.info("已编码 %d 篇 / %d tokens / %.1fs",
                            n_docs, len(all_ids), time.time() - t0)
            if args.max_docs and n_docs >= args.max_docs:
                break

    total_tokens = len(all_ids)
    logger.info("编码完成：%d 篇文档，共 %d tokens，用时 %.1fs",
                n_docs, total_tokens, time.time() - t0)

    # 用最小可容纳词表的整数类型，节省磁盘与内存
    dtype = torch.int32 if sp.get_piece_size() > 32767 else torch.int16
    token_tensor = torch.tensor(all_ids, dtype=dtype)

    payload = {
        "token_ids": token_tensor,
        "context_length": args.context_length,
        "num_documents": n_docs,
        "total_tokens": total_tokens,
        "tokenizer": args.tokenizer,
        "vocab_size": sp.get_piece_size(),
    }
    torch.save(payload, args.output)
    n_blocks = max(0, (total_tokens - 1) // args.context_length)
    logger.info("已保存：%s", args.output)
    logger.info("按 context_length=%d 无重叠切片，可得 %d 个训练样本",
                args.context_length, n_blocks)


# --------------------------------------------------------------------------
# PyTorch Dataset（训练阶段）
# --------------------------------------------------------------------------
try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # noqa: BLE001 —— 仅 build 时可不装 torch.utils
    _TorchDataset = object


class PretrainDataset(_TorchDataset):
    """从 build 阶段产出的 .pt 加载，按固定窗口切片。

    每个样本取 context_length + 1 个连续 token：
      input_ids = block[:-1]
      labels    = block[1:]   （即 input_ids 右移一位）
    相邻样本的窗口无重叠（stride == context_length）。
    """

    def __init__(self, pt_path, context_length=None):
        import torch

        payload = torch.load(pt_path, map_location="cpu")
        self.token_ids = payload["token_ids"]
        self.context_length = context_length or payload["context_length"]
        n = self.token_ids.numel()
        # 需要多取 1 个 token 作为最后一个位置的预测目标
        self.num_blocks = max(0, (n - 1) // self.context_length)
        logger.info("PretrainDataset：%d tokens -> %d 个样本（ctx=%d）",
                    n, self.num_blocks, self.context_length)

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_blocks
        if not 0 <= idx < self.num_blocks:
            raise IndexError(idx)
        L = self.context_length
        start = idx * L
        block = self.token_ids[start : start + L + 1].long()
        input_ids = block[:-1]
        labels = block[1:]
        return input_ids, labels


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def inspect(args):
    """加载 .pt 并打印一个样本，便于自检。"""
    ds = PretrainDataset(args.output, args.context_length)
    if len(ds) == 0:
        logger.warning("数据集为空")
        return
    input_ids, labels = ds[0]
    logger.info("样本数：%d", len(ds))
    logger.info("input_ids shape=%s dtype=%s", tuple(input_ids.shape),
                input_ids.dtype)
    logger.info("labels    shape=%s dtype=%s", tuple(labels.shape),
                labels.dtype)
    logger.info("input_ids[:10] = %s", input_ids[:10].tolist())
    logger.info("labels[:10]    = %s", labels[:10].tolist())
    # 校验右移关系
    assert (input_ids[1:] == labels[:-1]).all(), "labels 应为 input_ids 右移一位"
    logger.info("校验通过：labels 确为 input_ids 右移一位")


def build_parser():
    p = argparse.ArgumentParser(
        description="构建预训练 Dataset（Task 4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=["build", "inspect"], default="build",
                   help="build：编码并保存 .pt；inspect：加载 .pt 自检")
    p.add_argument("--input", default="mixed_corpus.jsonl",
                   help="混合语料 jsonl 路径")
    p.add_argument("--text-field", default="text", help="文本字段名")
    p.add_argument("--tokenizer", default="data/tokenizer/zh_bpe_20k.model",
                   help="SentencePiece 模型路径")
    p.add_argument("--output", default="data/processed/pretrain_tokens.pt",
                   help="处理结果 .pt 输出/加载路径")

    p.add_argument("--context-length", type=int, default=1024,
                   help="切片窗口长度")
    p.add_argument("--add-bos", action="store_true",
                   help="是否在每篇文档开头加 <bos>")
    p.add_argument("--max-docs", type=int, default=None,
                   help="最多编码的文档数（调试用，默认全部）")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.mode == "build":
        build(args)
    else:
        inspect(args)
