"""
Clean local JSONL corpus (wiki + news) -> data/processed/local_clean.jsonl

Reads all .jsonl files from input_dir, applies cleaning and deduplication,
outputs one clean JSONL per line: {"text": "...", "source": "..."}

Usage:
  python scripts/clean_corpus.py \
      --input_dir data/raw \
      --output data/processed/local_clean.jsonl
"""
import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from pathlib import Path


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t　]+")
REPEAT_PUNCT_RE = re.compile(r"([。！？!?，,；;：:])\1{3,}")
CHINESE_RE = re.compile(r"[一-鿿]")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CHINESE_RE.findall(text)) / max(len(text), 1)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1\1\1", text)
    return text.strip()


def clean_doc(text: str, min_chars: int, min_chinese_ratio: float) -> str | None:
    paragraphs = []
    for para in text.split("\n"):
        para = clean_text(para)
        if len(para) >= min_chars and chinese_ratio(para) >= min_chinese_ratio:
            paragraphs.append(para)
    if not paragraphs:
        return None
    return "\n".join(paragraphs)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_chars", type=int, default=30)
    parser.add_argument("--min_chinese_ratio", type=float, default=0.35)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    stats = {"input": 0, "kept": 0, "too_short": 0, "low_chinese": 0, "duplicate": 0}

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for jsonl_file in sorted(args.input_dir.glob("*.jsonl")):
            print(f"reading {jsonl_file} ...")
            with jsonl_file.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    stats["input"] += 1
                    text = record.get("text", "")
                    source = record.get("source", "unknown")

                    cleaned = clean_doc(text, args.min_chars, args.min_chinese_ratio)
                    if cleaned is None:
                        stats["too_short"] += 1
                        continue

                    if chinese_ratio(cleaned) < args.min_chinese_ratio:
                        stats["low_chinese"] += 1
                        continue

                    digest = hashlib.md5(cleaned.encode("utf-8")).hexdigest()
                    if digest in seen:
                        stats["duplicate"] += 1
                        continue
                    seen.add(digest)

                    out.write(json.dumps({"text": cleaned, "source": source}, ensure_ascii=False) + "\n")
                    stats["kept"] += 1

                    if stats["kept"] % 50000 == 0:
                        print(f"  kept {stats['kept']} docs so far", flush=True)

    print("\nclean_corpus stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
