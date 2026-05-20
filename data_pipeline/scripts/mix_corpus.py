"""
Mix local_clean.jsonl (30%) with streaming BAAI/CCI3-HQ (70%) -> mixed_corpus.jsonl

The local corpus is loaded into memory and sampled probabilistically.
CCI3-HQ is streamed from HuggingFace and never fully downloaded.

Usage:
  pip install datasets

  # Full mix (streams until CCI3-HQ exhausted or --max_docs reached)
  python scripts/mix_corpus.py \
      --local data/processed/local_clean.jsonl \
      --output data/processed/mixed_corpus.jsonl \
      --local_ratio 0.3 \
      --hf_token YOUR_TOKEN

  # Test run
  python scripts/mix_corpus.py \
      --local data/processed/local_clean.jsonl \
      --output data/processed/mixed_corpus.jsonl \
      --local_ratio 0.3 \
      --max_docs 5000 \
      --hf_token YOUR_TOKEN

  # Filter CCI3-HQ by quality score (recommended: >= 3.5)
  python scripts/mix_corpus.py ... --min_score 3.5
"""
import argparse
import json
import random
import sys
from pathlib import Path


def load_local(path: Path) -> list[dict]:
    docs = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"loaded {len(docs)} local docs from {path}")
    return docs


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", required=True, type=Path, help="local_clean.jsonl")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--local_ratio", type=float, default=0.3,
                        help="Fraction of output that comes from local corpus (default: 0.3)")
    parser.add_argument("--min_score", type=float, default=0.0,
                        help="Min CCI3-HQ quality score to keep (0=no filter, try 3.5)")
    parser.add_argument("--max_docs", type=int, default=0, help="0 = no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_token", default=None)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Run: pip install datasets")
        sys.exit(1)

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    local_docs = load_local(args.local)
    if not local_docs:
        print("ERROR: no local docs found")
        sys.exit(1)

    print(f"streaming BAAI/CCI3-HQ (min_score={args.min_score}) ...")
    ds = load_dataset(
        "BAAI/CCI3-HQ",
        split="train",
        streaming=True,
        token=args.hf_token,
        trust_remote_code=True,
    )

    written = 0
    local_count = 0
    remote_count = 0
    skipped_score = 0

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for example in ds:
            score = example.get("score", 0.0)
            if args.min_score > 0 and score < args.min_score:
                skipped_score += 1
                continue

            text = example.get("text", "").strip()
            if not text:
                continue

            # For each remote doc, probabilistically insert local docs to maintain ratio
            # local_ratio = local / (local + remote)
            # => for each remote doc, we should also emit local_ratio/(1-local_ratio) local docs
            local_per_remote = args.local_ratio / max(1 - args.local_ratio, 1e-9)
            num_local_to_emit = int(local_per_remote + (random.random() < (local_per_remote % 1)))

            for _ in range(num_local_to_emit):
                doc = random.choice(local_docs)
                out.write(json.dumps({"text": doc["text"], "source": doc.get("source", "local")},
                                     ensure_ascii=False) + "\n")
                local_count += 1
                written += 1
                if args.max_docs and written >= args.max_docs:
                    break

            if args.max_docs and written >= args.max_docs:
                break

            out.write(json.dumps({"text": text, "source": "cci3hq", "score": score},
                                  ensure_ascii=False) + "\n")
            remote_count += 1
            written += 1

            if written % 10000 == 0:
                ratio = local_count / max(written, 1)
                print(f"  {written} docs | local {local_count} ({ratio:.1%}) | remote {remote_count}", flush=True)

            if args.max_docs and written >= args.max_docs:
                break

    actual_ratio = local_count / max(written, 1)
    print(f"\ndone: {written} total docs")
    print(f"  local:  {local_count} ({actual_ratio:.1%})")
    print(f"  remote: {remote_count} ({1-actual_ratio:.1%})")
    if skipped_score:
        print(f"  skipped (low score): {skipped_score}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
