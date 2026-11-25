from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text)).strip()


def strip_answer_prefix(text: str) -> str:
    return re.sub(r"^\s*答案\s*[:：]\s*", "", str(text), count=1)


def char_tokens(text: str) -> list[str]:
    return list(normalize_text(strip_answer_prefix(text)))


def exact_match(prediction: str, references: list[str]) -> bool:
    pred = normalize_text(prediction)
    return any(pred == normalize_text(ref) for ref in references)


def char_f1(prediction: str, reference: str) -> float:
    pred_tokens = char_tokens(prediction)
    ref_tokens = char_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = char_tokens(prediction)
    ref_tokens = char_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = lcs_len(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def first_json_object(text: str) -> Any | None:
    stripped = str(text).strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def prf1(pred_items: set[tuple[str, ...]], ref_items: set[tuple[str, ...]]) -> float:
    if not pred_items and not ref_items:
        return 1.0
    if not pred_items or not ref_items:
        return 0.0
    overlap = len(pred_items & ref_items)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_items)
    recall = overlap / len(ref_items)
    return 2 * precision * recall / (precision + recall)


def norm_value(value: Any) -> str:
    return normalize_text(json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value))


def json_flat_items(value: Any, prefix: str = "") -> set[tuple[str, str]]:
    items: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            items.update(json_flat_items(child, child_prefix))
    elif isinstance(value, list):
        for child in value:
            items.update(json_flat_items(child, prefix))
    else:
        items.add((prefix, norm_value(value)))
    return items


def json_f1(prediction: str, reference: str) -> float:
    pred = first_json_object(prediction)
    ref = first_json_object(reference)
    if pred is None or ref is None:
        return 0.0
    return prf1(json_flat_items(pred), json_flat_items(ref))


def entity_items(value: Any) -> set[tuple[str, str]]:
    payload = value if isinstance(value, dict) else {}
    entities = payload.get("entities") or payload.get("实体") or payload.get("entity_list") or []
    items: set[tuple[str, str]] = set()
    if not isinstance(entities, list):
        return items
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        kind = entity.get("类型") or entity.get("type") or entity.get("label") or entity.get("entity_type") or ""
        text = entity.get("文本") or entity.get("text") or entity.get("entity") or entity.get("word") or ""
        if text:
            items.add((norm_value(kind), norm_value(text)))
    return items


def entity_f1(prediction: str, reference: str) -> float:
    pred = first_json_object(prediction)
    ref = first_json_object(reference)
    if pred is None or ref is None:
        return 0.0
    return prf1(entity_items(pred), entity_items(ref))


def triple_items(value: Any) -> set[tuple[str, str, str]]:
    payload = value if isinstance(value, dict) else {}
    triples = payload.get("spo_list") or payload.get("triples") or payload.get("三元组") or []
    items: set[tuple[str, str, str]] = set()
    if not isinstance(triples, list):
        return items
    for triple in triples:
        if not isinstance(triple, dict):
            continue
        subject = triple.get("主语") or triple.get("subject") or triple.get("s") or ""
        predicate = triple.get("关系") or triple.get("predicate") or triple.get("p") or ""
        obj = triple.get("宾语") or triple.get("object") or triple.get("o") or ""
        if subject and predicate and obj:
            items.add((norm_value(subject), norm_value(predicate), norm_value(obj)))
    return items


def triple_f1(prediction: str, reference: str) -> float:
    pred = first_json_object(prediction)
    ref = first_json_object(reference)
    if pred is None or ref is None:
        return 0.0
    return prf1(triple_items(pred), triple_items(ref))


def best_reference_score(prediction: str, references: list[str], metric: str) -> float:
    if not references:
        return 0.0
    if metric == "em":
        return 1.0 if exact_match(prediction, references) else 0.0
    if metric == "char_f1":
        return max(char_f1(prediction, ref) for ref in references)
    if metric == "rouge_l":
        return max(rouge_l_f1(prediction, ref) for ref in references)
    if metric == "json_f1":
        return max(json_f1(prediction, ref) for ref in references)
    if metric == "entity_f1":
        return max(entity_f1(prediction, ref) for ref in references)
    if metric == "triple_f1":
        return max(triple_f1(prediction, ref) for ref in references)
    raise ValueError(f"unknown reference metric: {metric}")
