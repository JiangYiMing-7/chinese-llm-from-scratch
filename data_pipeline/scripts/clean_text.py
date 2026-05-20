import argparse
import hashlib
import html
import re
import unicodedata
from pathlib import Path


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
SPACE_RE = re.compile(r"[ \t\u3000]+")
REPEAT_PUNCT_RE = re.compile(r"([。！？!?，,；;：:])\1{3,}")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(CHINESE_RE.findall(text)) / max(len(text), 1)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = CONTROL_RE.sub("", text)
    text = SPACE_RE.sub(" ", text)
    text = REPEAT_PUNCT_RE.sub(r"\1\1\1", text)
    return text.strip()


def iter_text_files(input_dir: Path):
    for path in sorted(input_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min_chars", type=int, default=30)
    parser.add_argument("--min_chinese_ratio", type=float, default=0.35)
    parser.add_argument("--max_lines", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    stats = {
        "input_lines": 0,
        "kept_lines": 0,
        "too_short": 0,
        "low_chinese": 0,
        "duplicates": 0,
    }

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for path in iter_text_files(args.input_dir):
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    stats["input_lines"] += 1
                    text = normalize_text(line)
                    if len(text) < args.min_chars:
                        stats["too_short"] += 1
                        continue
                    if chinese_ratio(text) < args.min_chinese_ratio:
                        stats["low_chinese"] += 1
                        continue

                    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
                    if digest in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(digest)

                    out.write(text + "\n")
                    stats["kept_lines"] += 1

                    if args.max_lines and stats["kept_lines"] >= args.max_lines:
                        break
            if args.max_lines and stats["kept_lines"] >= args.max_lines:
                break

    print("clean_text stats")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

