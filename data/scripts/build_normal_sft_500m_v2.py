#!/usr/bin/env python3
"""Build split-aware normal SFT data for the 500M HF50K v2 flow."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
V11_PATH = SCRIPT_DIR / "convert_normal_sft_sources_v11.py"
spec = importlib.util.spec_from_file_location("_normal_sft_v11", V11_PATH)
v11 = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(v11)


TASK_BUCKETS = {
    "summary": {"summary", "summarization", "rewrite"},
    "reading": {"reading_qa", "search_info"},
    "extraction": {"information_extraction", "entity_extraction", "relation_extraction"},
    "nlu": {"classification", "sentence_matching", "nli"},
    "instruction": {"instruction_following", "format_following", "custom_reference"},
}
DEFAULT_BUCKET_WEIGHTS = {
    "summary": 12,
    "reading": 16,
    "extraction": 16,
    "nlu": 14,
    "instruction": 12,
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

V3_FINAL_BASE_TOTAL = 70000
V3_FINAL_QUOTA_SPECS = [
    {"source": "cluener2020", "task_type": "entity_extraction", "count": 5000},
    {"source": "duie", "task_type": "relation_extraction", "count": 11000},
    {"task_type": "information_extraction", "count": 2000},
    {"bucket": "reading", "count": 15000},
    {"bucket": "nlu", "count": 13500},
    {"bucket": "summary", "count": 11000},
    {"bucket": "instruction", "count": 12500},
]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_bucket(key: str, modulo: int = 1000) -> int:
    return int(sha256_text(key)[:12], 16) % modulo


def row_text(row: dict[str, Any]) -> str:
    if "messages" in row:
        return "\n".join(str(msg.get("content", "")) for msg in row["messages"])
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def user_text(row: dict[str, Any]) -> str:
    for msg in row.get("messages", []):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return row_text(row)


def assistant_text(row: dict[str, Any]) -> str:
    for msg in reversed(row.get("messages", [])):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    return ""


def core_task_hash(row: dict[str, Any]) -> str:
    text = re.sub(r"\s+", "", user_text(row))
    text = re.sub(r"[A-D][.．、]\s*[^。\n]+", "", text)
    return sha256_text(text[:2000])


def enrich(row: dict[str, Any], source_split: str, source_uid: str) -> dict[str, Any] | None:
    text = row_text(row)
    if len(text.strip()) < 4:
        return None
    user = user_text(row)
    assistant = assistant_text(row)
    if not user.strip() or not assistant.strip():
        return None
    out = dict(row)
    out["source_split"] = source_split
    out["source_uid"] = str(source_uid)
    out["text_hash"] = sha256_text(text)
    out["core_task_hash"] = core_task_hash(row)
    return out


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    yield from v11.read_jsonl(path)


def add_if_usable(rows: list[dict[str, Any]], row: dict[str, Any] | None, split: str, uid: str, max_chars: int) -> None:
    if row is None:
        return
    enriched = enrich(row, split, uid)
    if enriched is None:
        return
    if len(row_text(enriched)) > max_chars:
        return
    rows.append(enriched)


def load_coig(raw_root: Path, max_chars: int) -> list[dict[str, Any]]:
    root = raw_root / "coig_cqia"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in v11.COIG_EXCLUDE_NAMES or not v11.is_usable_jsonl(path):
            continue
        rel = path.relative_to(root).as_posix()
        for line_no, raw in read_jsonl(path):
            row = v11.normalize_instruction_row(
                raw,
                source="coig_cqia",
                license_note="Apache-2.0 per ModelScope page",
                original_id=f"{rel}:{line_no}",
            )
            uid = f"coig_cqia:{rel}:{line_no}"
            bucket = stable_bucket(uid)
            split = "hash_train" if bucket < 900 else "hash_val" if bucket < 950 else "hash_heldout"
            add_if_usable(rows, row, split, uid, max_chars)
    return rows


def load_belle(raw_root: Path, max_chars: int) -> list[dict[str, Any]]:
    path = raw_root / "belle_train_1m_cn" / "Belle_open_source_1M.json"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, raw in read_jsonl(path):
        row = v11.normalize_instruction_row(
            raw,
            source="belle_train_1m_cn",
            license_note="research-only restriction on dataset card",
            original_id=str(line_no),
        )
        uid = f"belle_train_1m_cn:{line_no}"
        bucket = stable_bucket(uid)
        split = "hash_train" if bucket < 900 else "hash_val" if bucket < 950 else "hash_heldout"
        add_if_usable(rows, row, split, uid, max_chars)
    return rows


def load_cmrc(raw_root: Path, max_chars: int) -> list[dict[str, Any]]:
    path = raw_root / "cmrc2018" / "data" / "cmrc2018_train.json"
    rows: list[dict[str, Any]] = []
    for item_idx, item in enumerate(v11.cmrc_items(path)):
        context = str(item.get("context_text", "")).strip()
        title = str(item.get("title", "")).strip()
        if not context:
            continue
        for qa_idx, qa in enumerate(item.get("qas", [])):
            question = str(qa.get("query_text", "")).strip()
            answers = qa.get("answers") or []
            answer = str(answers[0]).strip() if answers else ""
            if not question or not answer:
                continue
            uid = f"cmrc2018:train:{qa.get('query_id', f'{item_idx}_{qa_idx}')}"
            qa_row = v11.chat_row(
                f"只根据给定文本回答问题。\n标题：{title}\n文本：{context}\n问题：{question}",
                answer,
                source="cmrc2018",
                task_type="reading_qa",
                license_note="CC-BY-SA-4.0",
                original_id=qa.get("query_id"),
            )
            add_if_usable(rows, qa_row, "train", uid, max_chars)
            evidence_row = v11.chat_row(
                f"根据文本回答，并写出依据。\n标题：{title}\n文本：{context}\n问题：{question}",
                f"答案：{answer}\n依据：文本中提到“{answer}”。",
                source="cmrc2018",
                task_type="search_info",
                license_note="CC-BY-SA-4.0",
                original_id=qa.get("query_id"),
            )
            add_if_usable(rows, evidence_row, "train", uid + ":evidence", max_chars)
    return rows


def convert_cluener_entities(entities: list[dict[str, Any]], schema_version: str) -> list[dict[str, Any]]:
    if schema_version != "v3_final":
        return entities
    converted = []
    for entity in entities:
        item = dict(entity)
        item["类型"] = CLUENER_LABEL_ZH.get(str(item.get("类型", "")), str(item.get("类型", "")))
        converted.append(item)
    return converted


def load_cluener(raw_root: Path, max_chars: int, schema_version: str = "v2") -> list[dict[str, Any]]:
    path = raw_root / "cluener2020" / "train.json"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, raw in read_jsonl(path):
        text = str(raw.get("text", "")).strip()
        entities = v11.cluener_entities(raw)
        if not text or not entities:
            continue
        entities = convert_cluener_entities(entities, schema_version)
        if schema_version == "v3_final":
            user = "从文本中抽取命名实体，输出 JSON，字段为 实体；每个实体包含 类型 和 文本。\n文本：" + text
            assistant = json.dumps({"实体": entities}, ensure_ascii=False)
        else:
            user = "从文本中抽取命名实体，输出 JSON，字段为 entities；每个实体包含 类型 和 文本。\n文本：" + text
            assistant = json.dumps({"entities": entities}, ensure_ascii=False)
        row = v11.chat_row(
            user,
            assistant,
            source="cluener2020",
            task_type="entity_extraction",
            license_note="check CLUENER2020 dataset page before reporting",
            original_id=raw.get("id", line_no),
        )
        add_if_usable(rows, row, "train", f"cluener2020:train:{raw.get('id', line_no)}", max_chars)
    return rows


def classify_pclue_task(raw: dict[str, Any], user: str) -> str:
    raw_type = str(raw.get("type", "")).lower()
    if raw_type in {"ocnli", "cmnli"}:
        return "nli"
    if raw_type in {"afqmc", "bustm", "cluewsc2020"}:
        return "sentence_matching"
    if raw.get("answer_choices") or "选项" in user:
        return "classification"
    if raw_type in {"c3", "cmrc2018", "dureader"}:
        return "reading_qa"
    return v11.normalize_task_type(raw.get("type"), user)


def load_pclue(raw_root: Path, max_chars: int) -> list[dict[str, Any]]:
    root = raw_root / "pclue" / "datasets"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.glob("pCLUE_train_*.json")):
        for line_no, raw in read_jsonl(path):
            user = str(raw.get("input", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not user or not target:
                continue
            row = v11.chat_row(
                user,
                target,
                source="pclue",
                task_type=classify_pclue_task(raw, user),
                license_note="check pCLUE repository license before reporting",
                original_id=raw.get("id", f"{path.name}:{line_no}"),
            )
            uid = f"pclue:{path.name}:{raw.get('id', line_no)}"
            add_if_usable(rows, row, "train", uid, max_chars)
    return rows


def load_lcsts(raw_root: Path, max_chars: int) -> list[dict[str, Any]]:
    path = raw_root / "lcsts" / "train.jsonl"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, raw in read_jsonl(path):
        text = str(raw.get("text") or raw.get("content") or raw.get("article") or "").strip()
        summary = str(raw.get("summary") or raw.get("title") or raw.get("target") or "").strip()
        if not text or not summary:
            continue
        row = v11.chat_row(
            f"请用一句话概括下面文本，不要添加原文没有的信息。\n文本：{text}",
            summary,
            source="lcsts",
            task_type="summary",
            license_note="check LCSTS dataset card/source before reporting",
            original_id=raw.get("id", line_no),
        )
        add_if_usable(rows, row, "train", f"lcsts:train:{raw.get('id', line_no)}", max_chars)
    return rows


def load_duie(raw_root: Path, max_chars: int, schema_version: str = "v2") -> list[dict[str, Any]]:
    root = raw_root / "duie"
    rows: list[dict[str, Any]] = []
    paths = [
        root / "duie_train.json.zip",
        root / "duie_train.json",
        root / "train_data.json",
        root / "train.json",
    ]
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
                user = "从文本中抽取关系三元组，输出 JSON，字段为 三元组；每个三元组包含 主语、关系、宾语。\n文本：" + text
                assistant = json.dumps({"三元组": triples}, ensure_ascii=False)
            else:
                user = "从文本中抽取关系三元组，输出 JSON，字段为 spo_list；每个三元组包含 主语、关系、宾语。\n文本：" + text
                assistant = json.dumps({"spo_list": triples}, ensure_ascii=False)
            row = v11.chat_row(
                user,
                assistant,
                source="duie",
                task_type="relation_extraction",
                license_note="check DuIE/DuIE2.0 source license before reporting",
                original_id=raw.get("id", f"{path.name}:{line_no}"),
            )
            uid = f"duie:train:{raw.get('id', line_no)}"
            add_if_usable(rows, row, "train", uid, max_chars)
    return rows


def task_bucket(row: dict[str, Any]) -> str:
    task = str(row.get("task_type", "instruction_following"))
    for bucket, names in TASK_BUCKETS.items():
        if task in names:
            return bucket
    return "instruction"


def split_train_val_candidates(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, val = [], []
    for row in rows:
        source_split = str(row.get("source_split", "train"))
        if source_split == "hash_heldout":
            continue
        key = row["source_uid"]
        if source_split == "hash_val" or stable_bucket(key + ":sft_val") >= 950:
            val.append(row)
        else:
            train.append(row)
    return train, val


def sample_by_bucket(rows: list[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = {key: [] for key in DEFAULT_BUCKET_WEIGHTS}
    for row in rows:
        by_bucket[task_bucket(row)].append(row)
    for items in by_bucket.values():
        rng.shuffle(items)
    total_weight = sum(DEFAULT_BUCKET_WEIGHTS.values())
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    for bucket, weight in DEFAULT_BUCKET_WEIGHTS.items():
        quota = round(target * weight / total_weight)
        for row in by_bucket[bucket][:quota]:
            if row["text_hash"] in selected_hashes:
                continue
            selected.append(row)
            selected_hashes.add(row["text_hash"])
    if len(selected) < target:
        leftovers = [row for row in rows if row["text_hash"] not in selected_hashes]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: target - len(selected)])
    rng.shuffle(selected)
    return selected[:target]


def quota_matches(row: dict[str, Any], spec: dict[str, Any]) -> bool:
    source = spec.get("source")
    task_type = spec.get("task_type")
    bucket = spec.get("bucket")
    if source is not None and row.get("source") != source:
        return False
    if task_type is not None and row.get("task_type") != task_type:
        return False
    if bucket is not None and task_bucket(row) != bucket:
        return False
    return True


def scaled_quota_specs(specs: list[dict[str, Any]], target: int, base_total: int | None = None) -> list[dict[str, Any]]:
    scaled: list[dict[str, Any]] = []
    if base_total is None:
        base_total = sum(int(spec["count"]) for spec in specs)
    if base_total <= 0:
        raise ValueError("quota specs must contain a positive total count")
    running = 0
    for spec in specs:
        item = {key: value for key, value in spec.items() if key != "count"}
        count = round(target * int(spec["count"]) / base_total)
        item["count"] = count
        scaled.append(item)
        running += count
    if scaled and running != target:
        scaled[-1]["count"] = max(0, int(scaled[-1]["count"]) + target - running)
    return scaled


def load_quota_specs(
    path: Path | None,
    preset: str | None,
    target: int,
    *,
    scale_json: bool = False,
) -> list[dict[str, Any]] | None:
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        specs = payload.get("quotas", payload) if isinstance(payload, dict) else payload
        if not isinstance(specs, list):
            raise ValueError("quota JSON must be a list or contain a 'quotas' list")
        specs = [dict(item) for item in specs]
        if scale_json:
            return scaled_quota_specs(specs, target)
        return specs
    if preset == "v3_final":
        return scaled_quota_specs(V3_FINAL_QUOTA_SPECS, target, V3_FINAL_BASE_TOTAL)
    return None


def sample_by_quota_specs(
    rows: list[dict[str, Any]],
    target: int,
    rng: random.Random,
    specs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    rng.shuffle(shuffled)
    by_source_task: dict[tuple[str, str], list[dict[str, Any]]] = {}
    by_task: dict[str, list[dict[str, Any]]] = {}
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in shuffled:
        source = str(row.get("source", ""))
        task = str(row.get("task_type", ""))
        bucket = task_bucket(row)
        by_source_task.setdefault((source, task), []).append(row)
        by_task.setdefault(task, []).append(row)
        by_bucket.setdefault(bucket, []).append(row)
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    report: list[dict[str, Any]] = []
    for spec in specs:
        count = int(spec.get("count", 0))
        source = spec.get("source")
        task_type = spec.get("task_type")
        bucket = spec.get("bucket")
        if source is not None and task_type is not None:
            pool = by_source_task.get((str(source), str(task_type)), [])
        elif task_type is not None:
            pool = by_task.get(str(task_type), [])
        elif bucket is not None:
            pool = by_bucket.get(str(bucket), [])
        else:
            pool = shuffled
        matched = [row for row in pool if row["text_hash"] not in selected_hashes and quota_matches(row, spec)]
        take = matched[:count]
        selected.extend(take)
        selected_hashes.update(str(row["text_hash"]) for row in take)
        report.append({
            **{key: value for key, value in spec.items() if key != "count"},
            "requested": count,
            "available": len(matched),
            "selected": len(take),
        })
    if len(selected) < target:
        leftovers = [row for row in shuffled if row["text_hash"] not in selected_hashes]
        selected.extend(leftovers[: target - len(selected)])
    rng.shuffle(selected)
    return selected[:target], report


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_text: set[str] = set()
    seen_uid: set[str] = set()
    for row in rows:
        text_hash = str(row.get("text_hash", ""))
        uid = str(row.get("source_uid", ""))
        if text_hash in seen_text or uid in seen_uid:
            continue
        seen_text.add(text_hash)
        seen_uid.add(uid)
        deduped.append(row)
    return deduped


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 500M v2 normal SFT train/val JSONL.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/sft_sources/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--target-train-examples", type=int, default=70000)
    parser.add_argument("--target-val-examples", type=int, default=3500)
    parser.add_argument("--max-total-chars", type=int, default=6000)
    parser.add_argument("--output-prefix", default="normal_sft_500m_v2")
    parser.add_argument("--schema-version", choices=["v2", "v3_final"], default="v2")
    parser.add_argument("--preset-quotas", choices=["v3_final"], default=None)
    parser.add_argument("--quota-json", type=Path, default=None)
    parser.add_argument(
        "--scale-quota-json",
        action="store_true",
        help="Scale quota JSON counts to the requested train/val target instead of using absolute counts.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source_loaders = {
        "cmrc2018": load_cmrc,
        "coig_cqia": load_coig,
        "belle_train_1m_cn": load_belle,
        "cluener2020": lambda raw_root, max_chars: load_cluener(raw_root, max_chars, args.schema_version),
        "pclue": load_pclue,
        "lcsts": load_lcsts,
        "duie": lambda raw_root, max_chars: load_duie(raw_root, max_chars, args.schema_version),
    }
    all_rows: list[dict[str, Any]] = []
    source_report: dict[str, Any] = {}
    for name, loader in source_loaders.items():
        rows = loader(args.raw_root, args.max_total_chars)
        all_rows.extend(rows)
        source_report[name] = {
            "candidate_rows": len(rows),
            "task_counts": count_by(rows, "task_type"),
            "split_counts": count_by(rows, "source_split"),
        }
    if not all_rows:
        raise ValueError(f"no SFT candidates loaded from {args.raw_root}")

    all_rows = dedupe_rows(all_rows)
    train_candidates, val_candidates = split_train_val_candidates(all_rows)
    train_candidates = dedupe_rows(train_candidates)
    train_hashes = {str(row["text_hash"]) for row in train_candidates}
    val_candidates = [
        row for row in dedupe_rows(val_candidates)
        if str(row["text_hash"]) not in train_hashes
    ]
    train_quota_specs = load_quota_specs(
        args.quota_json,
        args.preset_quotas,
        args.target_train_examples,
        scale_json=args.scale_quota_json,
    )
    val_quota_specs = load_quota_specs(
        args.quota_json,
        args.preset_quotas,
        args.target_val_examples,
        scale_json=args.scale_quota_json,
    )
    if train_quota_specs is None:
        train_rows = sample_by_bucket(train_candidates, args.target_train_examples, rng)
        train_quota_report = None
    else:
        train_rows, train_quota_report = sample_by_quota_specs(
            train_candidates,
            args.target_train_examples,
            rng,
            train_quota_specs,
        )
    selected_train_hashes = {str(row["text_hash"]) for row in train_rows}
    val_candidates = [
        row for row in val_candidates
        if str(row["text_hash"]) not in selected_train_hashes
    ]
    if val_quota_specs is None:
        val_rows = sample_by_bucket(val_candidates, args.target_val_examples, rng)
        val_quota_report = None
    else:
        val_rows, val_quota_report = sample_by_quota_specs(
            val_candidates,
            args.target_val_examples,
            rng,
            val_quota_specs,
        )

    train_path = args.output_dir / f"{args.output_prefix}_train.jsonl"
    val_path = args.output_dir / f"{args.output_prefix}_val.jsonl"
    report_path = args.output_dir / f"{args.output_prefix}_report.json"
    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    report = {
        "raw_root": str(args.raw_root),
        "train_path": str(train_path),
        "val_path": str(val_path),
        "target_train_examples": args.target_train_examples,
        "target_val_examples": args.target_val_examples,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
        "schema_version": args.schema_version,
        "preset_quotas": args.preset_quotas,
        "quota_json": str(args.quota_json) if args.quota_json else None,
        "scale_quota_json": args.scale_quota_json,
        "train_quota_report": train_quota_report,
        "val_quota_report": val_quota_report,
        "source_candidates": source_report,
        "train_source_counts": count_by(train_rows, "source"),
        "train_task_counts": count_by(train_rows, "task_type"),
        "train_bucket_counts": count_by([{**row, "bucket": task_bucket(row)} for row in train_rows], "bucket"),
        "val_source_counts": count_by(val_rows, "source"),
        "val_task_counts": count_by(val_rows, "task_type"),
        "seed": args.seed,
        "notes": [
            "pCLUE SFT candidates are loaded only from pCLUE_train_*.json.",
            "pCLUE dev/test_public and official CLUE/FewCLUE are intentionally excluded from SFT.",
            "COIG/BELLE use stable hash split into train/val/heldout.",
        ],
    }
    write_report(report_path, report)
    print(json.dumps({"train": str(train_path), "val": str(val_path), "report": str(report_path), "train_examples": len(train_rows), "val_examples": len(val_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
