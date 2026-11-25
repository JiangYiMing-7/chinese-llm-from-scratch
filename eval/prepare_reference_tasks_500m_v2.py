#!/usr/bin/env python3
"""Prepare 500M v2 reference-answer language evaluation tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V11_PATH = PROJECT_ROOT / "data" / "scripts" / "convert_normal_sft_sources_v11.py"
spec = importlib.util.spec_from_file_location("_normal_sft_v11", V11_PATH)
v11 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(v11)


DEFAULT_TARGETS = {
    "summarization": 300,
    "reading_qa": 350,
    "search_info": 250,
    "entity_extraction": 300,
    "relation_extraction": 300,
    "classification": 300,
    "sentence_matching": 150,
    "nli": 150,
    "instruction_format": 200,
}

CLUENER_LABEL_ZH = {
    "address": "地址",
    "book": "书名",
    "company": "公司",
    "game": "游戏",
    "government": "政府",
    "movie": "电影",
    "name": "姓名",
    "organization": "组织机构",
    "position": "职位",
    "scene": "景点",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def convert_cluener_entities(entities: list[dict[str, Any]], schema_version: str) -> list[dict[str, Any]]:
    if schema_version != "v3_final":
        return entities
    converted = []
    for entity in entities:
        item = dict(entity)
        item["类型"] = CLUENER_LABEL_ZH.get(str(item.get("类型", "")), str(item.get("类型", "")))
        converted.append(item)
    return converted


def stable_bucket(key: str, modulo: int = 1000) -> int:
    return int(sha256_text(key)[:12], 16) % modulo


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", "", str(text))


def core_task_hash(user: str) -> str:
    text = normalize_space(user)
    text = re.sub(r"[A-D][.．、]\s*[^。\n]+", "", text)
    return sha256_text(text[:2000])


def enrich(row: dict[str, Any], source_split: str, source_uid: str) -> dict[str, Any]:
    text = json.dumps(row, ensure_ascii=False, sort_keys=True)
    row = dict(row)
    row["source_split"] = source_split
    row["source_uid"] = str(source_uid)
    row["text_hash"] = sha256_text(text)
    row["core_task_hash"] = core_task_hash(str(row.get("user", "")))
    return row


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    yield from v11.read_jsonl(path)


def first(row: dict[str, Any], names: list[str], default: Any = None) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return default


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add(rows: list[dict[str, Any]], row: dict[str, Any] | None, split: str, uid: str) -> None:
    if row is None:
        return
    if not str(row.get("user", "")).strip() or not row.get("references"):
        return
    rows.append(enrich(row, split, uid))


def load_lcsts(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, name in [("valid", "valid.jsonl"), ("test_public", "test_public.jsonl")]:
        path = raw_root / "lcsts" / name
        if not path.exists():
            continue
        for line_no, raw in read_jsonl(path):
            text = str(first(raw, ["text", "content", "article"], "")).strip()
            summary = str(first(raw, ["summary", "target", "title"], "")).strip()
            if not text or not summary:
                continue
            add(rows, {
                "id": f"lcsts_{split}_{line_no:06d}",
                "task_type": "summarization",
                "source": "lcsts",
                "user": f"请概括下面文本的主要内容，要求忠实原文、简洁：\n文本：{text}",
                "references": [summary],
                "metric": "rouge_l",
                "max_new_tokens": 80,
            }, split, f"lcsts:{split}:{line_no}")
    return rows


def load_cmrc(raw_root: Path) -> list[dict[str, Any]]:
    path = raw_root / "cmrc2018" / "data" / "cmrc2018_dev.json"
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(v11.cmrc_items(path)):
        context = str(item.get("context_text", "")).strip()
        title = str(item.get("title", "")).strip()
        if not context:
            continue
        for qa_idx, qa in enumerate(item.get("qas", [])):
            question = str(qa.get("query_text", "")).strip()
            answers = [str(x).strip() for x in qa.get("answers", []) if str(x).strip()]
            if not question or not answers:
                continue
            qid = qa.get("query_id", f"{item_idx}_{qa_idx}")
            add(rows, {
                "id": f"cmrc2018_dev_{qid}",
                "task_type": "reading_qa",
                "source": "cmrc2018",
                "user": f"请只根据给定文本回答问题。\n标题：{title}\n文本：{context}\n问题：{question}",
                "references": answers,
                "metric": "char_f1",
                "max_new_tokens": 64,
            }, "dev", f"cmrc2018:dev:{qid}")
            add(rows, {
                "id": f"cmrc2018_search_dev_{qid}",
                "task_type": "search_info",
                "source": "cmrc2018",
                "user": f"请根据文本找出问题答案，并尽量只输出答案短语。\n标题：{title}\n文本：{context}\n问题：{question}",
                "references": answers,
                "metric": "char_f1",
                "max_new_tokens": 64,
            }, "dev", f"cmrc2018:dev:{qid}:search")
    return rows


def load_cluener(raw_root: Path, schema_version: str = "v2") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["dev", "test"]:
        path = raw_root / "cluener2020" / f"{split}.json"
        if not path.exists():
            continue
        for line_no, raw in read_jsonl(path):
            text = str(raw.get("text", "")).strip()
            entities = v11.cluener_entities(raw)
            if not text or not entities:
                continue
            entities = convert_cluener_entities(entities, schema_version)
            if schema_version == "v3_final":
                ref = json.dumps({"实体": entities}, ensure_ascii=False)
                user = "从文本中抽取命名实体，输出 JSON，字段为 实体；每个实体包含 类型 和 文本。\n文本：" + text
                required_json_keys = ["实体"]
            else:
                ref = json.dumps({"entities": entities}, ensure_ascii=False)
                user = "从文本中抽取命名实体，输出 JSON，字段为 entities；每个实体包含 类型 和 文本。\n文本：" + text
                required_json_keys = ["entities"]
            add(rows, {
                "id": f"cluener2020_{split}_{raw.get('id', line_no)}",
                "task_type": "entity_extraction",
                "source": "cluener2020",
                "user": user,
                "references": [ref],
                "metric": "entity_f1",
                "required_json_keys": required_json_keys,
                "max_new_tokens": 128,
            }, split, f"cluener2020:{split}:{raw.get('id', line_no)}")
    return rows


def load_duie(raw_root: Path, schema_version: str = "v2") -> list[dict[str, Any]]:
    root = raw_root / "duie"
    paths = [
        root / "duie_dev.json.zip",
        root / "duie_dev.json",
        root / "dev_data.json",
        root / "dev.json",
    ]
    rows: list[dict[str, Any]] = []
    for path in [p for p in paths if p.exists()]:
        for line_no, raw in v11.read_json_records(path):
            text = str(raw.get("text", "")).strip()
            spo_list = raw.get("spo_list") or raw.get("spo") or []
            triples = []
            if not text or not isinstance(spo_list, list):
                continue
            for spo in spo_list:
                if not isinstance(spo, dict):
                    continue
                subject = str(spo.get("subject", "")).strip()
                predicate = str(spo.get("predicate", "")).strip()
                obj = v11.duie_object_value(spo.get("object", ""))
                if subject and predicate and obj:
                    triples.append({"主语": subject, "关系": predicate, "宾语": obj})
            if not triples:
                continue
            if schema_version == "v3_final":
                ref = json.dumps({"三元组": triples}, ensure_ascii=False)
                user = "从文本中抽取关系三元组，输出 JSON，字段为 三元组；每个三元组包含 主语、关系、宾语。\n文本：" + text
                required_json_keys = ["三元组"]
            else:
                ref = json.dumps({"spo_list": triples}, ensure_ascii=False)
                user = "从文本中抽取关系三元组，输出 JSON，字段为 spo_list；每个三元组包含 主语、关系、宾语。\n文本：" + text
                required_json_keys = ["spo_list"]
            add(rows, {
                "id": f"duie_dev_{raw.get('id', line_no)}",
                "task_type": "relation_extraction",
                "source": "duie",
                "user": user,
                "references": [ref],
                "metric": "triple_f1",
                "required_json_keys": required_json_keys,
                "max_new_tokens": 160,
            }, "dev", f"duie:dev:{raw.get('id', line_no)}")
    return rows


def pclue_task(raw: dict[str, Any], user: str) -> str:
    raw_type = str(raw.get("type", "")).lower()
    if raw_type in {"ocnli", "cmnli"}:
        return "nli"
    if raw_type in {"afqmc", "bustm", "cluewsc2020"}:
        return "sentence_matching"
    if raw_type in {"c3", "cmrc2018", "dureader"}:
        return "reading_qa"
    if raw.get("answer_choices") or "选项" in user:
        return "classification"
    return "classification"


def load_pclue(raw_root: Path) -> list[dict[str, Any]]:
    root = raw_root / "pclue" / "datasets"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    paths = [root / "pCLUE_dev.json", root / "pCLUE_test_public_1.json", root / "pCLUE_test_public_2.json"]
    for path in [p for p in paths if p.exists()]:
        split = path.stem.replace("pCLUE_", "")
        for line_no, raw in read_jsonl(path):
            user = str(raw.get("input", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not user or not target:
                continue
            task_type = pclue_task(raw, user)
            metric = "char_f1" if task_type == "reading_qa" else "em"
            add(rows, {
                "id": f"pclue_{split}_{raw.get('id', line_no)}",
                "task_type": task_type,
                "source": "pclue",
                "user": user,
                "references": [target],
                "metric": metric,
                "max_new_tokens": 96 if task_type == "reading_qa" else 16,
            }, split, f"pclue:{split}:{raw.get('id', line_no)}")
    return rows


def read_csv_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def load_official_nlu(root: Path, source_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("._") or ".git" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        split = "unknown"
        lowered = rel.lower()
        for candidate in ["dev", "validation", "test", "test_public"]:
            if candidate in lowered:
                split = candidate
                break
        if split not in {"dev", "validation", "test", "test_public"}:
            continue
        records: list[dict[str, Any]] = []
        if path.suffix in {".json", ".jsonl"}:
            try:
                if path.suffix == ".jsonl":
                    records = [row for _, row in read_jsonl(path)]
                else:
                    payload = read_json(path)
                    records = payload if isinstance(payload, list) else payload.get("data", []) if isinstance(payload, dict) else []
            except (OSError, json.JSONDecodeError):
                continue
        elif path.suffix == ".csv":
            records = list(read_csv_rows(path))
        for i, raw in enumerate(records):
            if not isinstance(raw, dict):
                continue
            label = first(raw, ["label", "answer", "target", "答案"])
            if label in (None, ""):
                continue
            text_a = first(raw, ["sentence1", "sentence", "text", "content", "question", "query", "premise"], "")
            text_b = first(raw, ["sentence2", "hypothesis", "keyword", "option", "choices"], "")
            if not text_a:
                continue
            task_type = "classification"
            if any(name in lowered for name in ["afqmc", "bustm", "match"]):
                task_type = "sentence_matching"
            if any(name in lowered for name in ["nli", "ocnli", "cmnli"]):
                task_type = "nli"
            user = f"请完成中文NLU判断任务，只输出标签。\n文本一：{text_a}"
            if text_b:
                user += f"\n文本二：{text_b}"
            add(rows, {
                "id": f"{source_name}_{rel.replace('/', '_')}_{i:06d}",
                "task_type": task_type,
                "source": source_name,
                "user": user,
                "references": [str(label)],
                "metric": "em",
                "max_new_tokens": 16,
            }, split, f"{source_name}:{rel}:{i}")
    return rows


def load_instruction_heldout(raw_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_name, root in [
        ("coig_cqia", raw_root / "coig_cqia"),
        ("belle_train_1m_cn", raw_root / "belle_train_1m_cn"),
    ]:
        if not root.exists():
            continue
        files = sorted(root.rglob("*.jsonl")) if source_name == "coig_cqia" else [root / "Belle_open_source_1M.json"]
        for path in [p for p in files if p.exists() and v11.is_usable_jsonl(p)]:
            if path.name in getattr(v11, "COIG_EXCLUDE_NAMES", set()):
                continue
            rel = path.relative_to(root).as_posix()
            for line_no, raw in read_jsonl(path):
                uid = f"{source_name}:{rel}:{line_no}"
                if stable_bucket(uid) < 950:
                    continue
                row = v11.normalize_instruction_row(
                    raw,
                    source=source_name,
                    license_note="held-out instruction source",
                    original_id=uid,
                )
                if row is None:
                    continue
                user = ""
                answer = ""
                for msg in row["messages"]:
                    if msg["role"] == "user":
                        user = msg["content"]
                    elif msg["role"] == "assistant":
                        answer = msg["content"]
                add(rows, {
                    "id": f"{source_name}_heldout_{line_no:06d}",
                    "task_type": "instruction_format",
                    "source": source_name,
                    "user": user,
                    "references": [answer],
                    "metric": "char_f1",
                    "max_new_tokens": 128,
                }, "hash_heldout", uid)
    return rows


def sample_targets(rows: list[dict[str, Any]], max_total: int, rng: random.Random) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(str(row["task_type"]), []).append(row)
    for items in by_task.values():
        rng.shuffle(items)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task, target in DEFAULT_TARGETS.items():
        quota = max(1, round(max_total * target / sum(DEFAULT_TARGETS.values())))
        for row in by_task.get(task, [])[:quota]:
            selected.append(row)
            seen.add(row["text_hash"])
    if len(selected) < max_total:
        leftovers = [row for row in rows if row["text_hash"] not in seen]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_total - len(selected)])
    rng.shuffle(selected)
    return selected[:max_total]


def load_exclusion_hashes(paths: list[Path]) -> tuple[set[str], set[str]]:
    text_hashes: set[str] = set()
    core_hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        for _line_no, row in read_jsonl(path):
            if row.get("text_hash"):
                text_hashes.add(str(row["text_hash"]))
            if row.get("core_task_hash"):
                core_hashes.add(str(row["core_task_hash"]))
    return text_hashes, core_hashes


def filter_sft_overlaps(rows: list[dict[str, Any]], sft_paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    text_hashes, core_hashes = load_exclusion_hashes(sft_paths)
    kept: list[dict[str, Any]] = []
    dropped_text = 0
    dropped_core = 0
    seen_text: set[str] = set()
    dropped_duplicate = 0
    for row in rows:
        text_hash = str(row.get("text_hash", ""))
        core_hash = str(row.get("core_task_hash", ""))
        if text_hash in seen_text:
            dropped_duplicate += 1
            continue
        if text_hash and text_hash in text_hashes:
            dropped_text += 1
            continue
        if core_hash and core_hash in core_hashes:
            dropped_core += 1
            continue
        seen_text.add(text_hash)
        kept.append(row)
    return kept, {
        "dropped_duplicate_text_hash": dropped_duplicate,
        "dropped_sft_text_hash_overlap": dropped_text,
        "dropped_sft_core_task_hash_overlap": dropped_core,
    }


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 500M v2 reference eval JSONL.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/sft_sources/raw"))
    parser.add_argument("--clue-root", type=Path, default=Path("data/eval/clue"))
    parser.add_argument("--fewclue-root", type=Path, default=Path("data/eval/fewclue"))
    parser.add_argument("--sft-report", type=Path, default=Path("data/sft/normal_sft_500m_v2_report.json"))
    parser.add_argument("--sft-train", type=Path, default=Path("data/sft/normal_sft_500m_v2_train.jsonl"))
    parser.add_argument("--sft-val", type=Path, default=Path("data/sft/normal_sft_500m_v2_val.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/data/zh_reference_tasks_500m_v2.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("eval/data/zh_reference_tasks_500m_v2_report.json"))
    parser.add_argument("--target-examples", type=int, default=2000)
    parser.add_argument("--schema-version", choices=["v2", "v3_final"], default="v2")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    groups = {
        "lcsts": load_lcsts(args.raw_root),
        "cmrc2018": load_cmrc(args.raw_root),
        "cluener2020": load_cluener(args.raw_root, args.schema_version),
        "duie": load_duie(args.raw_root, args.schema_version),
        "pclue": load_pclue(args.raw_root),
        "clue_official": load_official_nlu(args.clue_root, "clue_official"),
        "fewclue_official": load_official_nlu(args.fewclue_root, "fewclue_official"),
        "instruction_heldout": load_instruction_heldout(args.raw_root),
    }
    all_rows = [row for rows in groups.values() for row in rows]
    if not all_rows:
        raise ValueError("no reference eval rows were prepared")
    all_rows, filter_report = filter_sft_overlaps(all_rows, [args.sft_train, args.sft_val])
    if not all_rows:
        raise ValueError("all reference eval rows were removed by SFT overlap filtering")
    selected = sample_targets(all_rows, args.target_examples, rng)
    write_jsonl(args.output, selected)
    report = {
        "output": str(args.output),
        "target_examples": args.target_examples,
        "examples": len(selected),
        "schema_version": args.schema_version,
        "source_candidates": {
            name: {"rows": len(rows), "task_counts": count_by(rows, "task_type")}
            for name, rows in groups.items()
        },
        "selected_source_counts": count_by(selected, "source"),
        "selected_task_counts": count_by(selected, "task_type"),
        "filter_report": filter_report,
        "seed": args.seed,
        "notes": [
            "pCLUE eval rows are loaded only from pCLUE_dev/test_public files.",
            "Official CLUE/FewCLUE rows are used only from dev/validation/test-like paths.",
            "Structured extraction currently uses char_f1 plus required_json_keys metadata.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(args.report), "examples": len(selected), "task_counts": report["selected_task_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
