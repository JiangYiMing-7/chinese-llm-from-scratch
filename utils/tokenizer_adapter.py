#!/usr/bin/env python3
"""Runtime tokenizer adapter for SentencePiece .model and HF tokenizers .json."""
# 训练/评测里分词器可能是 SentencePiece(.model) 也可能是 HF tokenizers(.json)，
# 这里包一层统一接口（encode/decode/各种 special id），上层代码就不用关心底层是哪种了。

from __future__ import annotations

from pathlib import Path
from typing import Any


class SentencePieceAdapter:
    kind = "sentencepiece"

    def __init__(self, path: str | Path):
        try:
            import sentencepiece as spm
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("sentencepiece is required for .model tokenizers") from exc

        self.path = str(path)
        self.sp = spm.SentencePieceProcessor(model_file=self.path)

    def encode(
        self,
        text: str,
        out_type: type = int,
        add_bos: bool = False,
        add_eos: bool = False,
        **_: Any,
    ) -> list[int] | list[str]:
        return self.sp.encode(str(text), out_type=out_type, add_bos=add_bos, add_eos=add_eos)

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        return self.sp.decode([int(x) for x in ids if int(x) >= 0])

    def get_piece_size(self) -> int:
        return int(self.sp.get_piece_size())

    def vocab_size(self) -> int:
        return self.get_piece_size()

    def piece_to_id(self, piece: str) -> int:
        return int(self.sp.piece_to_id(piece))

    def token_to_id(self, token: str) -> int:
        return self.piece_to_id(token)

    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    def unk_id(self) -> int:
        return int(self.sp.unk_id())

    def pad_id(self) -> int:
        return int(self.sp.pad_id())


class HFTokenizerAdapter:
    kind = "hf_tokenizers_json"

    def __init__(self, path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:  # pragma: no cover - depends on runtime env
            raise RuntimeError("tokenizers is required for HF tokenizer .json files") from exc

        self.path = str(path)
        self.tokenizer = Tokenizer.from_file(self.path)
        self._unk_id = self._first_existing_id(["<unk>", "[UNK]", "<|unk|>"], default=-1)

    def _first_existing_id(self, candidates: list[str], default: int = -1) -> int:
        for token in candidates:
            idx = self.tokenizer.token_to_id(token)
            if idx is not None:
                return int(idx)
        return int(default)

    def encode(
        self,
        text: str,
        out_type: type = int,
        add_bos: bool = False,
        add_eos: bool = False,
        **_: Any,
    ) -> list[int] | list[str]:
        encoded = self.tokenizer.encode(str(text), add_special_tokens=False)
        ids: list[int] | list[str]
        if out_type is str:
            ids = list(encoded.tokens)
        else:
            ids = [int(x) for x in encoded.ids]
        if add_bos:
            bos = self.bos_id()
            if bos >= 0:
                bos_token = self.tokenizer.id_to_token(bos)
                ids = ([bos_token or str(bos)] if out_type is str else [bos]) + ids
        if add_eos:
            eos = self.eos_id()
            if eos >= 0:
                eos_token = self.tokenizer.id_to_token(eos)
                ids = ids + ([eos_token or str(eos)] if out_type is str else [eos])
        return ids

    def decode(self, ids: list[int] | tuple[int, ...]) -> str:
        clean_ids = [int(x) for x in ids if int(x) >= 0]
        return self.tokenizer.decode(clean_ids, skip_special_tokens=True)

    def get_piece_size(self) -> int:
        return int(self.tokenizer.get_vocab_size(with_added_tokens=True))

    def vocab_size(self) -> int:
        return self.get_piece_size()

    def piece_to_id(self, piece: str) -> int:
        idx = self.tokenizer.token_to_id(piece)
        return self.unk_id() if idx is None else int(idx)

    def token_to_id(self, token: str) -> int:
        return self.piece_to_id(token)

    # HF 的 .json 里特殊 token 叫法五花八门，按一串候选名依次找，找到哪个用哪个
    def bos_id(self) -> int:
        return self._first_existing_id(["<s>", "<bos>", "<|bos|>", "<|beginoftext|>", "<|endoftext|>"], default=0)

    def eos_id(self) -> int:
        return self._first_existing_id(["<|endoftext|>", "</s>", "<eos>", "<|eos|>"], default=-1)

    def unk_id(self) -> int:
        return int(self._unk_id)

    def pad_id(self) -> int:
        return self._first_existing_id(["<pad>", "[PAD]", "<|pad|>", "<|endoftext|>"], default=-1)


def load_tokenizer(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".json":
        return HFTokenizerAdapter(path)
    return SentencePieceAdapter(path)
