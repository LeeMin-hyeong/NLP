from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import re
from typing import Any

from pydantic import ValidationError
import yaml

from app.schemas import SpecOutput


class ModelOutputError(RuntimeError):
    """Raised when model output cannot be parsed or validated."""


@dataclass
class ExtractionResult:
    spec: SpecOutput | None
    json_parse_ok: bool
    pydantic_validation_ok: bool
    error: str | None = None


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _extract_fenced_blocks(text: str) -> list[str]:
    pattern = re.compile(r"```(?:json|yaml|yml)?\s*\n(.*?)```", flags=re.IGNORECASE | re.DOTALL)
    return [match.group(1).strip() for match in pattern.finditer(text) if match.group(1).strip()]


def _extract_unclosed_fence_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    start_pattern = re.compile(r"```(?:json|yaml|yml)?\s*\n", flags=re.IGNORECASE)
    for match in start_pattern.finditer(text):
        start = match.end()
        remainder = text[start:].strip()
        if remainder:
            blocks.append(remainder)
    return blocks


def _extract_yaml_mapping_segment(text: str) -> str:
    lines = text.splitlines()
    key_line_pattern = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_\-]*\s*:")

    start_index = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if key_line_pattern.match(line):
            start_index = idx
            break

    if start_index is None:
        return ""

    candidate = "\n".join(lines[start_index:]).strip()
    return candidate


def _strip_json_comments(text: str) -> str:
    no_line_comments = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    no_block_comments = re.sub(r"/\*.*?\*/", "", no_line_comments, flags=re.DOTALL)
    return no_block_comments


def _basic_json_repair(text: str) -> str:
    repaired = text
    repaired = repaired.replace("“", '"').replace("”", '"')
    repaired = repaired.replace("’", "'")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _quote_unquoted_keys(text: str) -> str:
    # Converts {key: ...} or ,key: ... to {"key": ...} / ,"key": ...
    pattern = re.compile(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)(\s*:)")
    return pattern.sub(r'\1"\2"\3', text)


def _single_to_double_quoted_strings(text: str) -> str:
    # Converts single-quoted strings to JSON-compatible double-quoted strings.
    string_pattern = re.compile(r"'([^'\\]*(?:\\.[^'\\]*)*)'")

    def repl(match: re.Match[str]) -> str:
        content = match.group(1)
        content = content.replace('"', '\\"')
        return f'"{content}"'

    return string_pattern.sub(repl, text)


def _python_literals_to_json_literals(text: str) -> str:
    converted = re.sub(r"\bTrue\b", "true", text)
    converted = re.sub(r"\bFalse\b", "false", converted)
    converted = re.sub(r"\bNone\b", "null", converted)
    return converted


def _extract_json_segment(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ModelOutputError("Could not find a valid JSON object boundary")
    return text[start : end + 1]


def _extract_balanced_json_objects(text: str) -> list[str]:
    candidates: list[str] = []

    depth = 0
    start_idx: int | None = None
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start_idx = idx
            depth += 1
            continue

        if char == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(text[start_idx : idx + 1])
                start_idx = None

    return candidates


def _iter_json_candidates(raw_text: str) -> list[str]:
    stripped = raw_text.strip()
    fence_stripped = _strip_code_fence(raw_text)

    candidates: list[str] = []
    yaml_segment = _extract_yaml_mapping_segment(raw_text)
    for candidate in (stripped, fence_stripped, yaml_segment):
        if candidate:
            candidates.append(candidate)

    for candidate in _extract_fenced_blocks(raw_text):
        candidates.append(candidate)
    for candidate in _extract_unclosed_fence_blocks(raw_text):
        candidates.append(candidate)

    try:
        segment = _extract_json_segment(raw_text)
        if segment:
            candidates.append(segment)
    except ModelOutputError:
        pass

    for candidate in _extract_balanced_json_objects(raw_text):
        if candidate:
            candidates.append(candidate)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)

    return deduped


def _candidate_variants(candidate: str) -> list[str]:
    base = _strip_json_comments(_basic_json_repair(candidate))
    lines = base.splitlines()
    if lines and lines[0].strip().lower().startswith("output:"):
        base = "\n".join(lines[1:]).strip()

    variants = [
        candidate,
        _basic_json_repair(candidate),
        base,
        _quote_unquoted_keys(base),
        _single_to_double_quoted_strings(base),
        _quote_unquoted_keys(_single_to_double_quoted_strings(base)),
        _python_literals_to_json_literals(_quote_unquoted_keys(base)),
        _python_literals_to_json_literals(_quote_unquoted_keys(_single_to_double_quoted_strings(base))),
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        key = variant.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    return deduped


def _coerce_mapping(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list) and len(payload) == 1 and isinstance(payload[0], dict):
        return payload[0]
    raise ModelOutputError("Parsed payload root is not a JSON object")


def _load_with_json(text: str) -> dict[str, Any]:
    return _coerce_mapping(json.loads(text))


def _load_with_yaml(text: str) -> dict[str, Any]:
    return _coerce_mapping(yaml.safe_load(text))


def _load_with_ast(text: str) -> dict[str, Any]:
    return _coerce_mapping(ast.literal_eval(text))


def _shorten(text: str, max_len: int = 140) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def parse_json_payload(raw_text: str) -> dict[str, Any]:
    candidates = _iter_json_candidates(raw_text)
    errors: list[str] = []

    loaders = [
        ("json", _load_with_json),
        ("yaml", _load_with_yaml),
        ("ast", _load_with_ast),
    ]

    for c_idx, candidate in enumerate(candidates, start=1):
        for v_idx, variant in enumerate(_candidate_variants(candidate), start=1):
            for loader_name, loader in loaders:
                try:
                    payload = loader(variant)
                    return payload
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"c{c_idx}/v{v_idx}/{loader_name}: {exc}")

    if not candidates:
        raise ModelOutputError("JSON parse failed after repair attempts: no JSON object candidate found")

    preview = _shorten(candidates[0])
    raise ModelOutputError(
        (
            "JSON parse failed after repair attempts: "
            f"candidates={len(candidates)}, first_candidate_preview={preview} | "
            + " | ".join(errors[-8:])
        )
    )


_EVIDENCE_ID_PATTERN = re.compile(r"\bU\d+\b", flags=re.IGNORECASE)
_EVIDENCE_KEYS = {
    "evidence",
    "evidence_id",
    "evidence_ids",
    "evidence_utterance_id",
    "evidence_utterance_ids",
    "utterance_id",
    "utterance_ids",
}
_TEXT_KEYS = {
    "text",
    "requirement",
    "question",
    "topic",
    "summary",
    "description",
    "content",
    "value",
}
_RESERVED_KEYS = {"id", "priority", *_EVIDENCE_KEYS}
_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "in",
    "on",
    "and",
    "or",
    "is",
    "are",
    "be",
    "this",
    "that",
    "it",
    "we",
    "our",
    "with",
    "by",
    "from",
}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        parts = [item for item in parts if item]
        return "; ".join(parts)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            item_text = _coerce_text(item)
            if not item_text:
                continue
            parts.append(f"{str(key).replace('_', ' ')}: {item_text}")
        return "; ".join(parts).strip()
    return str(value).strip()


def _extract_utterance_ids(value: Any) -> list[str]:
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            for token in _EVIDENCE_ID_PATTERN.findall(node):
                normalized = token.upper()
                if normalized not in found:
                    found.append(normalized)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if isinstance(node, dict):
            for item in node.values():
                walk(item)

    walk(value)
    return found


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {token for token in tokens if token and token not in _STOPWORDS and len(token) > 1}


def _infer_evidence_from_utterances(text: str, utterances: list[dict[str, str]]) -> list[str]:
    if not utterances:
        return []

    text_tokens = _tokenize(text)
    if not text_tokens:
        first_id = str(utterances[0].get("id", "")).strip()
        return [first_id] if first_id else []

    best_id = ""
    best_score = -1.0

    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip()
        utterance_text = str(utterance.get("text", "")).strip()
        if not utterance_id or not utterance_text:
            continue

        utterance_tokens = _tokenize(utterance_text)
        if not utterance_tokens:
            continue

        overlap = len(text_tokens & utterance_tokens)
        score = overlap / max(len(text_tokens), 1)
        if score > best_score:
            best_score = score
            best_id = utterance_id

    if best_id:
        return [best_id]

    fallback_id = str(utterances[0].get("id", "")).strip()
    return [fallback_id] if fallback_id else []


def _normalize_evidence(evidence: list[str], utterances: list[dict[str, str]]) -> list[str]:
    valid_ids = {str(item.get("id", "")).strip() for item in utterances}
    normalized: list[str] = []
    for token in evidence:
        cleaned = str(token).strip().upper()
        if not cleaned:
            continue
        if cleaned in valid_ids and cleaned not in normalized:
            normalized.append(cleaned)
    return normalized


def _extract_item_evidence(item: dict[str, Any], utterances: list[dict[str, str]], text: str) -> list[str]:
    evidence: list[str] = []

    for key, value in item.items():
        key_normalized = str(key).strip().lower()
        if key_normalized in _EVIDENCE_KEYS or "evidence" in key_normalized or "utterance" in key_normalized:
            evidence.extend(_extract_utterance_ids(value))

    if not evidence:
        evidence.extend(_extract_utterance_ids(item))

    normalized = _normalize_evidence(evidence, utterances)
    if normalized:
        return normalized

    return _infer_evidence_from_utterances(text=text, utterances=utterances)


def _expand_mapping_to_text_pairs(item: dict[str, Any]) -> list[str]:
    pairs: list[str] = []

    for key, value in item.items():
        key_norm = str(key).strip().lower()
        if key_norm in _TEXT_KEYS:
            text = _coerce_text(value)
            if text:
                pairs.append(text)
            continue

    if pairs:
        return pairs

    for key, value in item.items():
        key_norm = str(key).strip().lower()
        if key_norm in _RESERVED_KEYS:
            continue
        if "evidence" in key_norm or "utterance" in key_norm:
            continue

        if key_norm in {"requirements", "items"} and isinstance(value, list):
            for entry in value:
                entry_text = _coerce_text(entry)
                if entry_text:
                    pairs.append(entry_text)
            continue

        if isinstance(value, dict):
            produced_here = False
            nested_text = _coerce_text(value.get("text") or value.get("requirement") or value.get("description"))
            if nested_text:
                pairs.append(nested_text)
                produced_here = True
            nested_requirements = value.get("requirements")
            if isinstance(nested_requirements, list):
                for entry in nested_requirements:
                    entry_text = _coerce_text(entry)
                    if entry_text:
                        pairs.append(entry_text)
                        produced_here = True
            if produced_here:
                continue

        value_text = _coerce_text(value)
        if not value_text:
            continue

        # For key-value style outputs, value is usually the real requirement sentence.
        pairs.append(value_text)

    return pairs


def _normalize_priority(priority: Any) -> str | None:
    cleaned = str(priority).strip().lower()
    if cleaned in {"high", "medium", "low"}:
        return cleaned
    return None


def _normalize_question_decision(value: Any) -> str:
    normalized = _coerce_text(value).lower().replace("-", " ").replace("_", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()

    if normalized in {"needs follow up", "need follow up", "follow up needed", "ask again", "unresolved"}:
        return "needs_follow_up"
    if normalized in {
        "already asked",
        "asked",
        "already addressed",
        "already answered",
        "no follow up needed",
    }:
        return "already_asked"
    if normalized in {"resolved", "answered", "closed"}:
        return "resolved"
    return "needs_follow_up"


def _normalize_project_summary(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or "Generated requirements draft"

    if isinstance(value, dict):
        for preferred_key in ("project_summary", "summary", "text", "description", "content"):
            if preferred_key in value:
                text = _coerce_text(value.get(preferred_key))
                if text:
                    return text
        text = _coerce_text(value)
        return text or "Generated requirements draft"

    if isinstance(value, list):
        text = _coerce_text(value)
        return text or "Generated requirements draft"

    text = _coerce_text(value)
    return text or "Generated requirements draft"


def _coerce_items_to_list(items: Any) -> list[Any]:
    if items is None:
        return []
    if isinstance(items, list):
        return items
    if isinstance(items, dict):
        # Some models return map-style buckets instead of arrays.
        values = list(items.values())
        return values if values else [items]
    return [items]


def _ensure_requirement_ids(items: Any, prefix: str, utterances: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    items_list = _coerce_items_to_list(items)
    if not items_list:
        return normalized_items

    next_index = 1

    for item in items_list:
        if isinstance(item, str):
            text = _coerce_text(item)
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append(
                {
                    "id": f"{prefix}{next_index}",
                    "text": text,
                    "priority": None,
                    "evidence": evidence,
                }
            )
            next_index += 1
            continue

        if not isinstance(item, dict):
            text = _coerce_text(item)
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append(
                {
                    "id": f"{prefix}{next_index}",
                    "text": text,
                    "priority": None,
                    "evidence": evidence,
                }
            )
            next_index += 1
            continue

        item_dict = dict(item)
        item_id = str(item_dict.get("id", "")).strip()
        priority = _normalize_priority(item_dict.get("priority"))

        text_candidates = _expand_mapping_to_text_pairs(item_dict)
        if not text_candidates:
            fallback_text = _coerce_text(item_dict)
            if fallback_text:
                text_candidates = [fallback_text]

        if not text_candidates:
            continue

        for candidate_index, text in enumerate(text_candidates, start=1):
            evidence = _extract_item_evidence(item_dict, utterances=utterances, text=text)
            normalized_items.append(
                {
                    "id": item_id if item_id and candidate_index == 1 else f"{prefix}{next_index}",
                    "text": text,
                    "priority": priority,
                    "evidence": evidence,
                }
            )
            next_index += 1

    return normalized_items


def _ensure_note_items(items: Any, utterances: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    items_list = _coerce_items_to_list(items)
    if not items_list:
        return normalized_items

    for item in items_list:
        if isinstance(item, str):
            text = _coerce_text(item)
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append({"text": text, "evidence": evidence})
            continue

        if not isinstance(item, dict):
            text = _coerce_text(item)
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append({"text": text, "evidence": evidence})
            continue

        item_dict = dict(item)
        text_candidates = _expand_mapping_to_text_pairs(item_dict)
        if not text_candidates:
            fallback_text = _coerce_text(item_dict)
            if fallback_text:
                text_candidates = [fallback_text]

        for text in text_candidates:
            evidence = _extract_item_evidence(item_dict, utterances=utterances, text=text)
            normalized_items.append({"text": text, "evidence": evidence})

    return normalized_items


def _ensure_question_decisions(items: Any, utterances: list[dict[str, str]]) -> list[dict[str, Any]]:
    normalized_items: list[dict[str, Any]] = []
    items_list = _coerce_items_to_list(items)
    if not items_list:
        return normalized_items

    for item in items_list:
        if isinstance(item, str):
            text = _coerce_text(item)
            if not text:
                continue
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append(
                {
                    "text": text,
                    "decision": "needs_follow_up",
                    "suggested_follow_up": _coerce_text(item),
                    "evidence": evidence,
                }
            )
            continue

        if not isinstance(item, dict):
            text = _coerce_text(item)
            if not text:
                continue
            evidence = _infer_evidence_from_utterances(text=text, utterances=utterances)
            normalized_items.append(
                {
                    "text": text,
                    "decision": "needs_follow_up",
                    "suggested_follow_up": None,
                    "evidence": evidence,
                }
            )
            continue

        item_dict = dict(item)
        text = _coerce_text(
            item_dict.get("text")
            or item_dict.get("question")
            or item_dict.get("topic")
            or item_dict.get("item")
            or item_dict.get("ambiguity")
        )
        if not text:
            expanded = _expand_mapping_to_text_pairs(item_dict)
            if expanded:
                text = expanded[0]
        if not text:
            text = _coerce_text(item_dict)
        if not text:
            continue

        decision = _normalize_question_decision(
            item_dict.get("decision")
            or item_dict.get("status")
            or item_dict.get("follow_up_decision")
        )

        suggested_follow_up = _coerce_text(
            item_dict.get("suggested_follow_up")
            or item_dict.get("follow_up_question")
            or item_dict.get("question_to_ask")
        )
        if decision != "needs_follow_up":
            suggested_follow_up = None

        evidence = _extract_item_evidence(item_dict, utterances=utterances, text=text)
        normalized_items.append(
            {
                "text": text,
                "decision": decision,
                "suggested_follow_up": suggested_follow_up or None,
                "evidence": evidence,
            }
        )

    return normalized_items


def _normalize_payload(payload: dict[str, Any], utterances: list[dict[str, str]]) -> dict[str, Any]:
    normalized = dict(payload)

    normalized["project_summary"] = _normalize_project_summary(normalized.get("project_summary"))
    normalized["functional_requirements"] = _ensure_requirement_ids(
        normalized.get("functional_requirements", []),
        prefix="FR",
        utterances=utterances,
    )
    normalized["non_functional_requirements"] = _ensure_requirement_ids(
        normalized.get("non_functional_requirements", []),
        prefix="NFR",
        utterances=utterances,
    )

    normalized["constraints"] = _ensure_note_items(normalized.get("constraints", []), utterances=utterances)
    normalized["assumptions"] = _ensure_note_items(normalized.get("assumptions", []), utterances=utterances)
    normalized["open_questions"] = _ensure_note_items(normalized.get("open_questions", []), utterances=utterances)
    normalized["question_decisions"] = _ensure_question_decisions(
        normalized.get("question_decisions", []),
        utterances=utterances,
    )

    follow_up = normalized.get("follow_up_questions", [])
    if not isinstance(follow_up, list):
        follow_up = [str(follow_up)]
    follow_up_normalized: list[str] = []
    for item in follow_up:
        if isinstance(item, dict):
            for value in item.values():
                text = _coerce_text(value)
                if text:
                    follow_up_normalized.append(text)
        else:
            text = _coerce_text(item)
            if text:
                follow_up_normalized.append(text)

    if not follow_up_normalized:
        for decision_item in normalized["question_decisions"]:
            if decision_item.get("decision") != "needs_follow_up":
                continue
            suggested = _coerce_text(decision_item.get("suggested_follow_up"))
            if suggested:
                follow_up_normalized.append(suggested)
    normalized["follow_up_questions"] = follow_up_normalized

    normalized["utterances"] = utterances
    return normalized


def extract_spec_output(raw_text: str, utterances: list[dict[str, str]]) -> ExtractionResult:
    try:
        payload = parse_json_payload(raw_text=raw_text)
    except Exception as exc:  # noqa: BLE001
        return ExtractionResult(
            spec=None,
            json_parse_ok=False,
            pydantic_validation_ok=False,
            error=f"JSON parsing failed: {exc}",
        )

    normalized_payload = _normalize_payload(payload=payload, utterances=utterances)

    try:
        spec = SpecOutput.model_validate(normalized_payload)
        return ExtractionResult(
            spec=spec,
            json_parse_ok=True,
            pydantic_validation_ok=True,
            error=None,
        )
    except ValidationError as exc:
        return ExtractionResult(
            spec=None,
            json_parse_ok=True,
            pydantic_validation_ok=False,
            error=f"Schema validation failed: {exc}",
        )
