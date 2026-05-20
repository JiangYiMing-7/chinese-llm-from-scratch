"""
Download Chinese Wikipedia and CLUECorpus2020 news to data/raw/ as JSONL.

Usage:
  pip install datasets

  # Full download
  python scripts/download_data.py --output_dir data/raw

  # Test run (first 1000 docs each)
  python scripts/download_data.py --output_dir data/raw --max_docs 1000
"""
import argparse
import json
import sys
from pathlib import Path


SOURCES = {
    "wiki": {
        "hf_path": "wikimedia/wikipedia",
        "hf_name": "20231101.zh",
        "split": "train",
        "text_field": "text",
        "output_file": "wiki_raw.jsonl",
    },
    "news": {
        "hf_path": "liwu/MNBVC",
        "hf_name": "news2016zh_cn",
        "split": "train",
        "text_field": "text",
        "output_file": "news_raw.jsonl",
    },
}


def download_source(name: str, cfg: dict, output_dir: Path, max_docs: int, hf_token: str | None) -> None:
    from datasets import load_dataset

    out_path = output_dir / cfg["output_file"]
    print(f"\n[{name}] streaming {cfg['hf_path']} / {cfg['hf_name']} -> {out_path}")

    ds = load_dataset(
        cfg["hf_path"],
        cfg["hf_name"],
        split=cfg["split"],
        streaming=True,
        token=hf_token,
        trust_remote_code=True,
    )

    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for example in ds:
            text = example.get(cfg["text_field"], "")
            if not text or not text.strip():
                continue
            record = {"text": text.strip(), "source": name}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written % 10000 == 0:
                print(f"  [{name}] {written} docs", flush=True)
            if max_docs and written >= max_docs:
                print(f"  [{name}] reached --max_docs {max_docs}, stopping")
                break

    print(f"  [{name}] done: {written} docs -> {out_path}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--sources", nargs="+", default=list(SOURCES.keys()),
                        choices=list(SOURCES.keys()), help="Which sources to download")
    parser.add_argument("--max_docs", type=int, default=0, help="0 = no limit (full download)")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token")
    args = parser.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        print("Run: pip install datasets")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in args.sources:
        download_source(name, SOURCES[name], args.output_dir, args.max_docs, args.hf_token)

    print("\nall done.")


if __name__ == "__main__":
    main()
