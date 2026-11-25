#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path
from typing import Any


TASK_PATTERNS = [
    ("summary", r"摘要|总结|概括|标题|要点|一句话|简述"),
    ("information_extraction", r"提取|抽取|找出|字段|实体|JSON|json|关键词"),
    ("reading_qa", r"根据.*回答|只根据|阅读.*回答|文本.*问题|回答问题"),
    ("rewrite", r"改写|润色|压缩|扩写|翻译|书面语|通俗"),
    ("classification", r"判断|分类|情感|正面|负面|选择|是否|类别"),
    ("format_following", r"只输出|格式|列表|表格|每行|不要解释|不超过"),
]

LFS_PREFIX = "version https://git-lfs.github.com/spec"
COIG_EXCLUDE_NAMES = {"100poison.jsonl"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError:
                continue


def read_jsonl_from_zip(path: Path):
    with zipfile.ZipFile(path) as archive:
        names = [
            name for name in archive.namelist()
            if not name.endswith("/") and name.endswith((".json", ".jsonl"))
        ]
        for name in sorted(names):
            with archive.open(name) as f:
                for line_no, raw_line in enumerate(f, start=1):
                    line = raw_line.decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    try:
                        yield f"{path.name}:{name}:{line_no}", json.loads(line)
                    except json.JSONDecodeError:
                        continue


def read_json_records(path: Path):
    if path.suffix == ".zip":
        yield from read_jsonl_from_zip(path)
    else:
        yield from read_jsonl(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def first_line(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", errors="ignore") as f:
            return f.readline().strip()
    except OSError:
        return ""


def is_usable_jsonl(path: Path) -> bool:
    if path.name.startswith("._") or ".git" in path.parts:
        return False
    return not first_line(path).startswith(LFS_PREFIX)


def infer_task_type(text: str) -> str:
    snippet = text[:800]
    for task_type, pattern in TASK_PATTERNS:
        if re.search(pattern, snippet, flags=re.I):
            return task_type
    return "instruction_following"


def normalize_task_type(value: Any, user_text: str) -> str:
    if isinstance(value, str) and value:
        lowered = value.lower()
        aliases = {
            "extract": "information_extraction",
            "summarization": "summary",
            "qa": "reading_qa",
            "question_answering": "reading_qa",
            "mrc": "search_info",
            "classify": "classification",
            "nli": "classification",
            "generate": "summary",
        }
        return aliases.get(lowered, lowered)
    return infer_task_type(user_text)


def chat_row(
    user: str,
    assistant: str,
    *,
    source: str,
    task_type: str,
    license_note: str,
    original_id: Any,
) -> dict[str, Any] | None:
    user = str(user).strip()
    assistant = str(assistant).strip()
    if not user or not assistant:
        return None
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "source": source,
        "task_type": task_type,
        "license": license_note,
        "original_id": original_id,
    }


def normalize_instruction_row(
    row: dict[str, Any],
    *,
    source: str,
    license_note: str,
    original_id: str,
) -> dict[str, Any] | None:
    instruction = str(row.get("instruction", "")).strip()
    answer = str(row.get("output", "")).strip()
    extra = str(row.get("input", "")).strip()
    if not instruction or not answer:
        return None
    user = instruction if not extra else f"{instruction}\n{extra}"
    return chat_row(
        user,
        answer,
        source=source,
        task_type=normalize_task_type(row.get("task_type"), user),
        license_note=license_note,
        original_id=original_id,
    )


def convert_coig(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    root = raw_root / "coig_cqia"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*.jsonl")):
        if path.name in COIG_EXCLUDE_NAMES or not is_usable_jsonl(path):
            continue
        rel = path.relative_to(root)
        for line_no, raw in read_jsonl(path):
            row = normalize_instruction_row(
                raw,
                source="coig_cqia",
                license_note="Apache-2.0 per ModelScope page",
                original_id=f"{rel}:{line_no}",
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def convert_belle(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    path = raw_root / "belle_train_1m_cn" / "Belle_open_source_1M.json"
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, raw in read_jsonl(path):
        row = normalize_instruction_row(
            raw,
            source="belle_train_1m_cn",
            license_note="research-only restriction on dataset card",
            original_id=str(line_no),
        )
        if row is not None:
            rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def cmrc_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, dict) and "data" in data:
        # SQuAD-style fallback.
        items = []
        for article in data.get("data", []):
            for para in article.get("paragraphs", []):
                context = para.get("context", "")
                for qa in para.get("qas", []):
                    items.append({
                        "context_text": context,
                        "title": article.get("title", ""),
                        "qas": [{
                            "query_id": qa.get("id"),
                            "query_text": qa.get("question"),
                            "answers": [a.get("text", "") for a in qa.get("answers", [])],
                        }],
                    })
        return items
    return data if isinstance(data, list) else []


def convert_cmrc_reading(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    path = raw_root / "cmrc2018" / "data" / "cmrc2018_train.json"
    rows: list[dict[str, Any]] = []
    for item in cmrc_items(path):
        context = str(item.get("context_text", "")).strip()
        title = str(item.get("title", "")).strip()
        if not context:
            continue
        for qa in item.get("qas", []):
            question = str(qa.get("query_text", "")).strip()
            answers = qa.get("answers") or []
            answer = str(answers[0]).strip() if answers else ""
            row = chat_row(
                f"只根据给定文本回答问题。\n标题：{title}\n文本：{context}\n问题：{question}",
                answer,
                source="cmrc2018",
                task_type="reading_qa",
                license_note="CC-BY-SA-4.0",
                original_id=qa.get("query_id"),
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def convert_cmrc_evidence(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    path = raw_root / "cmrc2018" / "data" / "cmrc2018_train.json"
    rows: list[dict[str, Any]] = []
    for item in cmrc_items(path):
        context = str(item.get("context_text", "")).strip()
        title = str(item.get("title", "")).strip()
        if not context:
            continue
        for qa in item.get("qas", []):
            question = str(qa.get("query_text", "")).strip()
            answers = qa.get("answers") or []
            answer = str(answers[0]).strip() if answers else ""
            if not question or not answer:
                continue
            user = f"根据文本回答，并写出依据。\n标题：{title}\n文本：{context}\n问题：{question}"
            assistant = f"答案：{answer}\n依据：文本中提到“{answer}”。"
            row = chat_row(
                user,
                assistant,
                source="cmrc2018_evidence",
                task_type="search_info",
                license_note="CC-BY-SA-4.0",
                original_id=qa.get("query_id"),
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def find_first(root: Path, names: tuple[str, ...]) -> Path | None:
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
    for path in sorted(root.rglob("*")):
        if path.name in names and path.is_file():
            return path
    return None


def find_many(root: Path, names: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name in names and not path.name.startswith("._"):
            found.append(path)
    return found


def cluener_entities(raw: dict[str, Any]) -> list[dict[str, str]]:
    text = str(raw.get("text", ""))
    label = raw.get("label") or {}
    entities: list[dict[str, str]] = []
    if not isinstance(label, dict):
        return entities
    for entity_type, by_text in sorted(label.items()):
        if not isinstance(by_text, dict):
            continue
        for entity_text, spans in sorted(by_text.items()):
            if not spans:
                entities.append({"类型": str(entity_type), "文本": str(entity_text)})
                continue
            for _span in spans:
                entities.append({"类型": str(entity_type), "文本": str(entity_text)})
    deduped = []
    seen = set()
    for entity in entities:
        key = (entity["类型"], entity["文本"])
        if key not in seen and entity["文本"] in text:
            deduped.append(entity)
            seen.add(key)
    return deduped


def convert_cluener(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    root = raw_root / "cluener2020"
    path = find_first(root, ("train.json", "train.jsonl", "dev.json", "dev.jsonl"))
    rows: list[dict[str, Any]] = []
    if path is None:
        return rows
    for line_no, raw in read_jsonl(path):
        text = str(raw.get("text", "")).strip()
        entities = cluener_entities(raw)
        if not text or not entities:
            continue
        user = (
            "从文本中抽取命名实体，输出 JSON，字段为 entities；"
            "每个实体包含 类型 和 文本。\n文本：" + text
        )
        assistant = json.dumps({"entities": entities}, ensure_ascii=False)
        row = chat_row(
            user,
            assistant,
            source="cluener2020",
            task_type="information_extraction",
            license_note="check CLUENER2020 dataset page before reporting",
            original_id=raw.get("id", line_no),
        )
        if row is not None:
            rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def convert_pclue(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    root = raw_root / "pclue"
    if not root.exists():
        root = raw_root / "pCLUE"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows

    paths = [
        path for path in sorted(root.rglob("*.json"))
        if path.name == "pCLUE_dev.json"
        or re.fullmatch(r"pCLUE_train_\d+\.json", path.name)
        or path.name in {"train.json", "train.jsonl", "dev.json", "dev.jsonl"}
    ]
    for path in paths:
        if ".git" in path.parts or path.name.startswith("._"):
            continue
        for line_no, raw in read_jsonl(path):
            user = str(raw.get("input", "")).strip()
            target = str(raw.get("target", "")).strip()
            if not user or not target:
                continue
            raw_type = normalize_task_type(raw.get("type"), user)
            if raw_type in {"classification", "search_info", "reading_qa", "summary"}:
                task_type = raw_type
            elif "选项" in user or raw.get("answer_choices"):
                task_type = "classification"
            else:
                task_type = infer_task_type(user)
            row = chat_row(
                user,
                target,
                source="pclue",
                task_type=task_type,
                license_note="check pCLUE repository license before reporting",
                original_id=raw.get("id", f"{path.name}:{line_no}"),
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def convert_lcsts(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    root = raw_root / "lcsts"
    if not root.exists():
        root = raw_root / "LCSTS"
    paths = find_many(root, ("train.jsonl", "valid.jsonl", "test_public.jsonl", "LCSTS.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, raw in read_jsonl(path):
            text = str(raw.get("text") or raw.get("content") or raw.get("article") or "").strip()
            summary = str(raw.get("summary") or raw.get("title") or raw.get("target") or "").strip()
            if not text or not summary:
                continue
            label = raw.get("human_label")
            try:
                if label is not None and int(label) < 3:
                    continue
            except (TypeError, ValueError):
                pass
            user = f"请用一句话概括下面文本，不要添加原文没有的信息。\n文本：{text}"
            row = chat_row(
                user,
                summary,
                source="lcsts",
                task_type="summary",
                license_note="check LCSTS dataset card/source before reporting",
                original_id=raw.get("id", f"{path.name}:{line_no}"),
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def duie_object_value(value: Any) -> str:
    if isinstance(value, dict):
        if "@value" in value:
            return str(value.get("@value", "")).strip()
        parts = [str(v).strip() for v in value.values() if str(v).strip()]
        return "；".join(parts)
    return str(value).strip()


def convert_duie(raw_root: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    root = raw_root / "duie"
    if not root.exists():
        root = raw_root / "DuIE2.0"
    paths = find_many(root, (
        "train_data.json",
        "dev_data.json",
        "train.json",
        "dev.json",
        "duie_train.json",
        "duie_dev.json",
        "duie_train.json.zip",
        "duie_dev.json.zip",
    ))
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_no, raw in read_json_records(path):
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
                obj = duie_object_value(spo.get("object", ""))
                if subject and predicate and obj:
                    triples.append({"主语": subject, "关系": predicate, "宾语": obj})
            if not triples:
                continue
            user = (
                "从文本中抽取关系三元组，输出 JSON，字段为 spo_list；"
                "每个三元组包含 主语、关系、宾语。\n文本：" + text
            )
            assistant = json.dumps({"spo_list": triples}, ensure_ascii=False)
            row = chat_row(
                user,
                assistant,
                source="duie",
                task_type="information_extraction",
                license_note="check DuIE/DuIE2.0 source license before reporting",
                original_id=raw.get("id", f"{path.name}:{line_no}"),
            )
            if row is not None:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def make_format_templates(limit: int, rng: random.Random) -> list[dict[str, Any]]:
    items = [
        ("杭州发布公共交通调整通知，新增两条夜间公交线路。", {"city": "杭州", "policy": "新增两条夜间公交线路"}),
        ("成都试点社区低碳出行积分，居民步行或骑行可获得积分。", {"city": "成都", "policy": "低碳出行积分"}),
        ("学校通知6月20日前提交报名表，提交方式为官网在线填写。", {"截止日期": "6月20日", "报名方式": "官网在线填写"}),
        ("会议决定先确认需求，再评估周期，最后提交排期。", ["确认需求", "评估周期", "提交排期"]),
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(max(1, limit // len(items))):
        for text, target in items:
            if isinstance(target, dict):
                user = f"请根据文本输出 JSON，不要添加解释。\n文本：{text}"
                assistant = json.dumps(target, ensure_ascii=False)
            else:
                user = f"请把文本整理成三条短项目符号，每条不超过20字。\n文本：{text}"
                assistant = "\n".join(f"- {x}" for x in target)
            row = chat_row(
                user,
                assistant,
                source="synthetic_format_v1",
                task_type="format_following",
                license_note="project-local synthetic templates",
                original_id=idx,
            )
            if row is not None:
                rows.append(row)
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    rng.shuffle(rows)
    return rows[:limit]


def count_task_types(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        task_type = str(row.get("task_type", "unknown"))
        counts[task_type] = counts.get(task_type, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Chinese SFT v1.1 sources to chat JSONL.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/sft_sources/raw"))
    parser.add_argument("--converted-dir", type=Path, default=Path("data/sft_sources/converted"))
    parser.add_argument("--candidate-path", type=Path, default=Path("data/sft/normal_sft_v11_candidates.jsonl"))
    parser.add_argument("--report-path", type=Path, default=Path("data/sft/normal_sft_v11_candidates_report.json"))
    parser.add_argument("--coig-limit", type=int, default=10000)
    parser.add_argument("--cmrc-limit", type=int, default=8000)
    parser.add_argument("--cmrc-evidence-limit", type=int, default=3000)
    parser.add_argument("--belle-limit", type=int, default=3000)
    parser.add_argument("--cluener-limit", type=int, default=5000)
    parser.add_argument("--pclue-limit", type=int, default=8000)
    parser.add_argument("--lcsts-limit", type=int, default=8000)
    parser.add_argument("--duie-limit", type=int, default=5000)
    parser.add_argument("--format-template-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def add_source(
    all_rows: list[dict[str, Any]],
    report_sources: dict[str, Any],
    name: str,
    rows: list[dict[str, Any]],
    converted_dir: Path,
) -> None:
    write_jsonl(converted_dir / f"{name}.jsonl", rows)
    all_rows.extend(rows)
    report_sources[name] = {"rows": len(rows), "task_counts": count_task_types(rows)}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.converted_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {}
    add_source(all_rows, sources, "coig_cqia_normal_sft_v11", convert_coig(args.raw_root, args.coig_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "cmrc2018_reading_qa_v11", convert_cmrc_reading(args.raw_root, args.cmrc_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "cmrc2018_evidence_v11", convert_cmrc_evidence(args.raw_root, args.cmrc_evidence_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "belle_filtered_normal_sft_v11", convert_belle(args.raw_root, args.belle_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "cluener2020_extract_v11", convert_cluener(args.raw_root, args.cluener_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "pclue_normal_sft_v11", convert_pclue(args.raw_root, args.pclue_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "lcsts_summary_v11", convert_lcsts(args.raw_root, args.lcsts_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "duie_relation_extract_v11", convert_duie(args.raw_root, args.duie_limit, rng), args.converted_dir)
    add_source(all_rows, sources, "synthetic_format_v11", make_format_templates(args.format_template_limit, rng), args.converted_dir)

    rng.shuffle(all_rows)
    write_jsonl(args.candidate_path, all_rows)
    report = {
        "raw_root": str(args.raw_root),
        "converted_dir": str(args.converted_dir),
        "candidate_path": str(args.candidate_path),
        "sources": sources,
        "total_rows": len(all_rows),
        "task_counts": count_task_types(all_rows),
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
