from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from finetune.chat_template import encode_chat, normalize_messages


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL in {path}:{line_no}") from exc
    return rows


class SFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_seq_len: int,
        default_system: str | None = "你是一个中文助手，请根据用户任务给出准确、简洁的回答。",
        min_assistant_tokens: int = 1,
    ):
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.default_system = default_system
        self.examples = []
        for row in read_jsonl(self.path):
            messages = normalize_messages(row, default_system=default_system)
            encoded = encode_chat(tokenizer, messages, max_seq_len=max_seq_len)
            if len(encoded.input_ids) < 2:
                continue
            # 没可监督 assistant token 的样本（被截断或本来就空）直接丢，不然这条 loss 是 0 白占 batch
            if encoded.assistant_tokens < min_assistant_tokens:
                continue
            self.examples.append({
                "input_ids": encoded.input_ids,
                "labels": encoded.labels,
                "source": row.get("source"),
                "task_type": row.get("task_type"),
            })
        if not self.examples:
            raise ValueError(f"no usable SFT examples loaded from {self.path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


class SFTCollator:
    def __init__(self, pad_token_id: int, max_seq_len: int):
        self.pad_token_id = int(pad_token_id)
        self.max_seq_len = int(max_seq_len)

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # 动态 padding，按这个 batch 里最长的补齐，不是一律补到 max_seq_len，省算力
        max_len = min(self.max_seq_len, max(len(item["input_ids"]) for item in batch))
        input_rows = []
        label_rows = []
        attention_rows = []
        for item in batch:
            input_ids = item["input_ids"][:max_len]
            labels = item["labels"][:max_len]
            pad = max_len - len(input_ids)
            input_rows.append(input_ids + [self.pad_token_id] * pad)
            label_rows.append(labels + [-100] * pad)  # padding 位置 label 填 -100，不参与 loss
            attention_rows.append([1] * len(input_ids) + [0] * pad)  # attention_mask 把 padding 屏蔽掉
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
        }

