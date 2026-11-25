from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ROLES = {"system", "user", "assistant"}


@dataclass
class EncodedChat:
    input_ids: list[int]
    labels: list[int]
    assistant_tokens: int


def normalize_messages(row: dict[str, Any], default_system: str | None = None) -> list[dict[str, str]]:
    if "messages" in row:
        messages = row["messages"]
    elif "instruction" in row and "output" in row:
        user = str(row["instruction"])
        if row.get("input"):
            user = f"{user}\n{row['input']}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(row["output"])},
        ]
    else:
        raise ValueError("row must contain messages or instruction/output fields")

    normalized: list[dict[str, str]] = []
    if default_system and (not messages or messages[0].get("role") != "system"):
        normalized.append({"role": "system", "content": default_system})
    for message in messages:
        role = str(message.get("role", "")).strip()
        if role not in ROLES:
            raise ValueError(f"unsupported chat role: {role}")
        normalized.append({"role": role, "content": str(message.get("content", ""))})
    return normalized


def special_id(sp, piece: str) -> int:
    idx = sp.piece_to_id(piece)
    if idx == sp.unk_id():
        raise ValueError(f"tokenizer does not contain required special token: {piece}")
    return int(idx)


def encode_chat(
    sp,
    messages: list[dict[str, str]],
    max_seq_len: int,
    train_on_assistant: bool = True,
    add_eos_after_each_message: bool = True,
) -> EncodedChat:
    if max_seq_len < 2:
        raise ValueError("max_seq_len must be >= 2")

    end_id = special_id(sp, "<|endoftext|>")
    input_ids: list[int] = []
    labels: list[int] = []
    assistant_tokens = 0

    # SFT 这里最关键，role 标记和非 assistant 的内容一律用 -100 屏蔽掉，
    # 只在 assistant 回答的 token 上算 loss，模型才学“怎么答”而不是“怎么复述问题”
    for message in messages:
        role = message["role"]
        prefix_ids = sp.encode(f"<|{role}|>", out_type=int, add_bos=False)
        content_ids = sp.encode(message["content"], out_type=int, add_bos=False)
        suffix_ids = [end_id] if add_eos_after_each_message else []
        supervise = train_on_assistant and role == "assistant"

        input_ids.extend(prefix_ids)
        labels.extend([-100] * len(prefix_ids))
        input_ids.extend(content_ids)
        if supervise:
            labels.extend(content_ids)
            assistant_tokens += len(content_ids)
        else:
            labels.extend([-100] * len(content_ids))
        input_ids.extend(suffix_ids)
        if supervise:
            labels.extend(suffix_ids)
            assistant_tokens += len(suffix_ids)
        else:
            labels.extend([-100] * len(suffix_ids))

    if len(input_ids) > max_seq_len:
        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]
        assistant_tokens = sum(1 for value in labels if value != -100)

    return EncodedChat(input_ids=input_ids, labels=labels, assistant_tokens=assistant_tokens)


def render_chat_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        parts.append(f"<|{message['role']}|>{message['content']}<|endoftext|>")
    return "".join(parts)

