import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--vocab_size", type=int, default=16000)
    parser.add_argument("--model_type", default="bpe", choices=["bpe", "unigram"])
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    args = parser.parse_args()

    Path(args.model_prefix).parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[
            "<|system|>",
            "<|user|>",
            "<|assistant|>",
        ],
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

    print(f"model: {args.model_prefix}.model")
    print(f"vocab: {args.model_prefix}.vocab")


if __name__ == "__main__":
    main()

