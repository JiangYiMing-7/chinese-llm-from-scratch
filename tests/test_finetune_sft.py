import json

from finetune.chat_template import encode_chat, normalize_messages
from finetune.sft_dataset import SFTCollator, SFTDataset


class FakeTokenizer:
    def __init__(self):
        self.special = {
            "<|endoftext|>": 4,
            "<|system|>": 5,
            "<|user|>": 6,
            "<|assistant|>": 7,
        }

    def encode(self, text, out_type=int, add_bos=False):
        if text in self.special:
            return [self.special[text]]
        return [100 + (ord(ch) % 50) for ch in text if ch.strip()]

    def piece_to_id(self, piece):
        return self.special.get(piece, 0)

    def unk_id(self):
        return 0


def test_encode_chat_masks_non_assistant_tokens():
    sp = FakeTokenizer()
    messages = normalize_messages({
        "messages": [
            {"role": "user", "content": "请概括"},
            {"role": "assistant", "content": "一句话总结"},
        ]
    })
    encoded = encode_chat(sp, messages, max_seq_len=64)
    assistant_marker = sp.piece_to_id("<|assistant|>")
    assert assistant_marker in encoded.input_ids
    marker_pos = encoded.input_ids.index(assistant_marker)
    assert encoded.labels[marker_pos] == -100
    assert encoded.assistant_tokens > 0
    assert any(label != -100 for label in encoded.labels[marker_pos + 1:])


def test_sft_dataset_and_collator(tmp_path):
    path = tmp_path / "sft.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "从文本中提取城市"},
            {"role": "assistant", "content": "{\"城市\":\"杭州\"}"},
        ],
        "task_type": "information_extraction",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    dataset = SFTDataset(path, FakeTokenizer(), max_seq_len=64, default_system=None)
    assert len(dataset) == 1
    collated = SFTCollator(pad_token_id=0, max_seq_len=64)([dataset[0], dataset[0]])
    assert collated["input_ids"].shape[0] == 2
    assert collated["labels"].shape == collated["input_ids"].shape
    assert (collated["labels"] != -100).any()

