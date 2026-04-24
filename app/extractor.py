from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from app.schemas import (
    CandidateItem,
    ClassifiedCandidateItem,
    ConversationUnit,
    ConstraintItem,
    NoteItem,
    QuestionItem,
    RequirementItem,
    RewrittenItem,
    SpecOutput,
    Stage1CandidatesOutput,
    Stage2ClassifiedOutput,
    Stage3RewrittenOutput,
    Stage4OpenQuestionsOutput,
    Stage4FollowUpOutput,
    Stage5FollowUpOutput,
    Stage5SummaryOutput,
    Stage6SummaryOutput,
)
from app.utils import model_dump_compat, model_validate_compat, normalize_text


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractionMeta:
    json_parse_ok: bool = False
    pydantic_validation_ok: bool = False
    used_repair: bool = False
    repaired_output: str | None = None
    parse_error: str | None = None
    validation_error: str | None = None
    semantic_warnings: list[str] = field(default_factory=list)


STAGE1_ALLOWED_KINDS = {
    "possible_requirement",
    "possible_quality_expectation",
    "possible_constraint",
    "possible_ambiguity",
    "possible_future_scope",
    "possible_followup_trigger",
}

STAGE2_ALLOWED_TYPES = {
    "functional_requirement",
    "non_functional_requirement",
    "constraint",
    "open_question",
    "follow_up_trigger",
    "note",
    "discard",
}

REWRITE_ALLOWED_TYPES = {"functional_requirement", "non_functional_requirement", "constraint"}

FUTURE_SCOPE_HINTS = (
    "later",
    "future",
    "phase 2",
    "phase two",
    "not now",
    "eventually",
    "someday",
    "maybe",
    "might",
    "optional",
)

VAGUE_HINTS = (
    "clean",
    "modern",
    "simple",
    "easy",
    "intuitive",
    "user friendly",
)

HARD_BOUNDARY_HINTS = (
    "not in version one",
    "not in v1",
    "not part of the first release",
    "outside initial release",
    "initial release only",
    "web only",
    "no app",
    "within 4 weeks",
    "within four weeks",
    "limited budget",
    "budget is limited",
    "only staff",
    "only admin",
)

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "for",
    "in",
    "on",
    "with",
    "be",
    "is",
    "are",
    "it",
    "this",
    "that",
    "should",
    "shall",
    "must",
    "can",
    "could",
    "will",
    "would",
    "from",
    "by",
    "as",
    "at",
}

QUALITY_HINTS = (
    "fast",
    "faster",
    "quick",
    "quickly",
    "performance",
    "responsive",
    "responsiveness",
    "mobile",
    "mobile first",
    "mobile-first",
    "usability",
    "easy to use",
    "user friendly",
    "accessible",
    "accessibility",
    "secure",
    "security",
    "reliable",
    "reliability",
    "availability",
    "uptime",
    "load quickly",
    "load fast",
    "clean",
    "modern",
    "style",
    "design",
)

VAGUE_QUALITY_HINTS = (
    "clean",
    "modern",
    "easy to use",
    "user friendly",
    "intuitive",
    "simple",
    "style",
    "design should feel",
)

QUALITY_MEASUREMENT_HINTS = (
    "ms",
    "millisecond",
    "seconds",
    "sec",
    "latency",
    "uptime",
    "wcag",
    "%",
    "p95",
    "p99",
    "lcp",
    "cls",
    "tti",
)

CAPABILITY_HINTS = (
    "allow",
    "allows",
    "able to",
    "can ",
    "reserve",
    "book",
    "create",
    "update",
    "delete",
    "edit",
    "manage",
    "view",
    "search",
    "filter",
    "export",
    "import",
    "login",
    "log in",
    "sign up",
    "register",
    "pay",
    "checkout",
)

STAGE1_PLACEHOLDER_TEXT_HINTS = (
    "short candidate description",
    "explicit boundary or release limitation",
    "brief explanation",
    "sample candidate",
    "example candidate",
    "placeholder",
)


def _strip_code_fence(text: str) -> str:
    fenced = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def lightweight_repair_json(raw_text: str) -> str:
    repaired = raw_text.strip()
    repaired = _strip_code_fence(repaired)
    repaired = repaired.replace("\u201c", '"').replace("\u201d", '"')
    repaired = repaired.replace("\u2018", "'").replace("\u2019", "'")
    repaired = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", repaired)

    start_idx = repaired.find("{")
    end_idx = repaired.rfind("}")
    if start_idx >= 0 and end_idx > start_idx:
        repaired = repaired[start_idx : end_idx + 1]

    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    brace_diff = repaired.count("{") - repaired.count("}")
    bracket_diff = repaired.count("[") - repaired.count("]")
    if brace_diff > 0:
        repaired = repaired + ("}" * brace_diff)
    if bracket_diff > 0:
        repaired = repaired + ("]" * bracket_diff)
    return repaired.strip()


def _build_candidates(raw_output: str) -> list[str]:
    candidates: list[str] = []
    raw = raw_output.strip()
    if raw:
        candidates.append(raw)
    no_fence = _strip_code_fence(raw_output)
    if no_fence and no_fence not in candidates:
        candidates.append(no_fence)
    extracted = _extract_first_json_object(raw_output)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    return candidates


def parse_json_object_safe(
    raw_output: str,
    *,
    allow_repair: bool = True,
) -> tuple[dict[str, Any] | None, ExtractionMeta]:
    meta = ExtractionMeta()
    candidates = _build_candidates(raw_output)
    errors: list[str] = []

    for idx, candidate in enumerate(candidates, start=1):
        try:
            parsed = json.loads(candidate)
            if not isinstance(parsed, dict):
                errors.append(f"c{idx}/raw: top-level JSON must be an object")
                continue
            meta.json_parse_ok = True
            return parsed, meta
        except Exception as exc:
            errors.append(f"c{idx}/raw: {exc}")

    if allow_repair:
        for idx, candidate in enumerate(candidates, start=1):
            repaired = lightweight_repair_json(candidate)
            try:
                parsed = json.loads(repaired)
                if not isinstance(parsed, dict):
                    errors.append(f"c{idx}/repair: top-level JSON must be an object")
                    continue
                meta.json_parse_ok = True
                meta.used_repair = True
                meta.repaired_output = repaired
                return parsed, meta
            except Exception as exc:
                errors.append(f"c{idx}/repair: {exc}")

    if not errors:
        errors.append("no JSON candidate found")
    if allow_repair:
        meta.parse_error = "JSON parse failed after one repair pass: " + " | ".join(errors)
    else:
        meta.parse_error = "JSON parse failed with repair disabled: " + " | ".join(errors)
    return None, meta


def _normalize_source_units(value: Any, valid_unit_ids: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        sid = str(item).strip()
        if sid and sid in valid_unit_ids:
            normalized.append(sid)
    return normalized


def _infer_source_units_from_text(
    text: str,
    conversation_units: Iterable[ConversationUnit],
) -> list[str]:
    text_tokens = _tokens(text)
    if not text_tokens:
        return []

    scored: list[tuple[float, str]] = []
    for unit in conversation_units:
        unit_tokens = _tokens(unit.text)
        if not unit_tokens:
            continue
        overlap = len(text_tokens & unit_tokens) / max(1, len(text_tokens))
        if overlap > 0:
            scored.append((overlap, unit.id))

    if not scored:
        return []

    scored.sort(reverse=True)
    best_score = scored[0][0]
    return [unit_id for score, unit_id in scored if score >= best_score and score >= 0.08]


def _first_list_field(payload: dict[str, Any], field_names: Iterable[str]) -> tuple[str, list[Any] | None]:
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, list):
            return field_name, value
    return "", None


def _split_question_text_blob(value: str) -> list[str]:
    text = value.strip()
    if not text:
        return []
    # Allow newline or bullet-style question blobs from weaker instruction models.
    chunks = re.split(r"(?:\r?\n)+", text)
    out: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", chunk).strip()
        if cleaned:
            out.append(cleaned)
    return out


def _first_question_collection(
    payload: dict[str, Any],
    field_names: Iterable[str],
) -> tuple[str, list[Any] | None]:
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, list):
            return field_name, value
        if isinstance(value, str):
            questions = _split_question_text_blob(value)
            if questions:
                return field_name, questions
    return "", None


def _coerce_question_items(
    *,
    raw_items: list[Any],
    conversation_units: Iterable[ConversationUnit],
    stage_label: str,
) -> list[QuestionItem]:
    units = list(conversation_units)
    valid_unit_ids = {u.id for u in units}
    deduped: list[QuestionItem] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    dropped_ungrounded = 0
    non_empty_question_count = 0

    for raw in raw_items:
        if isinstance(raw, str):
            text = raw.strip()
            source_units = _infer_source_units_from_text(text, units)
        elif isinstance(raw, dict):
            text = str(raw.get("text", "") or raw.get("question", "")).strip()
            source_units = _normalize_source_units(raw.get("source_units"), valid_unit_ids)
            if not source_units:
                source_units = _normalize_source_units(raw.get("evidence"), valid_unit_ids)
            if not source_units:
                source_units = _infer_source_units_from_text(text, units)
        else:
            continue

        if not text:
            continue
        non_empty_question_count += 1
        if not text.endswith("?"):
            text = text.rstrip(".") + "?"
        if not source_units:
            dropped_ungrounded += 1
            continue

        key = (normalize_text(text), tuple(sorted(set(source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(QuestionItem(text=text, source_units=source_units))

    if non_empty_question_count > 0 and not deduped:
        raise ExtractionError(
            f"{stage_label} produced only ungrounded questions with missing valid source_units."
        )

    return deduped


def validate_stage_1_candidates(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> Stage1CandidatesOutput:
    raw_items = payload.get("candidates")
    if not isinstance(raw_items, list):
        raise ExtractionError("Stage 1 output must include list field `candidates`.")

    valid_unit_ids = {u.id for u in conversation_units}
    cleaned: list[CandidateItem] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("id", "")).strip() or f"C{idx}"
        text = str(raw.get("text", "")).strip()
        if _is_stage1_placeholder_text(text):
            continue
        kind = _normalize_stage1_kind(str(raw.get("kind", "")).strip(), text)
        source_units = _normalize_source_units(raw.get("source_units"), valid_unit_ids)

        if not text:
            continue
        if kind not in STAGE1_ALLOWED_KINDS:
            raise ExtractionError(f"Stage 1 candidate `{cid}` has invalid kind: {kind}")
        if not source_units:
            raise ExtractionError(f"Stage 1 candidate `{cid}` has no valid source_units.")

        dedupe_key = (kind, normalize_text(text), tuple(sorted(set(source_units))))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        cleaned.append(
            CandidateItem(
                id=cid,
                kind=kind,
                text=text,
                source_units=source_units,
            )
        )

    if not cleaned:
        raise ExtractionError(
            "Stage 1 produced no usable candidates (placeholder/example echo or invalid content)."
        )

    return Stage1CandidatesOutput(candidates=cleaned)


def validate_stage_2_classification(
    payload: dict[str, Any],
    stage1_output: Stage1CandidatesOutput,
    conversation_units: Iterable[ConversationUnit],
) -> Stage2ClassifiedOutput:
    raw_items = payload.get("classified_candidates")
    if not isinstance(raw_items, list):
        raise ExtractionError("Stage 2 output must include list field `classified_candidates`.")

    valid_unit_ids = {u.id for u in conversation_units}
    by_candidate_id = {c.id: c for c in stage1_output.candidates}
    classified: dict[str, ClassifiedCandidateItem] = {}

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        cid = str(raw.get("id", "")).strip()
        if not cid:
            continue
        if cid not in by_candidate_id:
            raise ExtractionError(f"Stage 2 referenced unknown candidate id: {cid}")

        final_type = str(raw.get("final_type", "")).strip()
        if final_type not in STAGE2_ALLOWED_TYPES:
            raise ExtractionError(
                f"Stage 2 candidate `{cid}` has invalid final_type: {final_type}"
            )

        reason = str(raw.get("reason", "")).strip() or "No explanation provided."
        candidate = by_candidate_id[cid]
        normalized_type, normalized_reason = _sanitize_stage2_final_type(
            candidate_kind=candidate.kind,
            text=candidate.text,
            final_type=final_type,
        )
        final_type = normalized_type
        if normalized_reason:
            reason = normalized_reason
        source_units = _normalize_source_units(raw.get("source_units"), valid_unit_ids)
        if not source_units:
            source_units = candidate.source_units

        if not source_units:
            raise ExtractionError(f"Stage 2 candidate `{cid}` has no valid source_units.")

        classified[cid] = ClassifiedCandidateItem(
            id=cid,
            final_type=final_type,
            reason=reason,
            source_units=source_units,
        )

    # Ensure every candidate has a terminal label.
    for candidate in stage1_output.candidates:
        if candidate.id in classified:
            continue
        classified[candidate.id] = ClassifiedCandidateItem(
            id=candidate.id,
            final_type="discard",
            reason="No classification returned for candidate.",
            source_units=candidate.source_units,
        )

    return Stage2ClassifiedOutput(classified_candidates=list(classified.values()))


def enrich_classified_candidates(
    classified_output: Stage2ClassifiedOutput,
    stage1_output: Stage1CandidatesOutput,
) -> list[dict[str, Any]]:
    by_candidate_id = {c.id: c for c in stage1_output.candidates}
    enriched: list[dict[str, Any]] = []
    for item in classified_output.classified_candidates:
        candidate = by_candidate_id.get(item.id)
        candidate_text = candidate.text if candidate else ""
        candidate_kind = candidate.kind if candidate else ""
        enriched.append(
            {
                "id": item.id,
                "final_type": item.final_type,
                "reason": item.reason,
                "source_units": item.source_units,
                "text": candidate_text,
                "candidate_kind": candidate_kind,
            }
        )
    return enriched


def validate_stage_3_rewriting(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
    authorized_rewrite_candidates: list[dict[str, Any]] | None = None,
) -> Stage3RewrittenOutput:
    raw_items = payload.get("rewritten_items")
    if not isinstance(raw_items, list):
        raise ExtractionError("Stage 3 output must include list field `rewritten_items`.")

    valid_unit_ids = {u.id for u in conversation_units}
    rewritten: list[RewrittenItem] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    dropped_ungrounded = 0
    for idx, raw in enumerate(raw_items, start=1):
        if not isinstance(raw, dict):
            continue
        rid = str(raw.get("id", "")).strip() or f"R{idx}"
        rtype = str(raw.get("type", "")).strip()
        text = str(raw.get("text", "")).strip()
        source_units = _normalize_source_units(raw.get("source_units"), valid_unit_ids)

        if rtype not in REWRITE_ALLOWED_TYPES:
            raise ExtractionError(f"Stage 3 item `{rid}` has invalid type: {rtype}")
        if not text:
            raise ExtractionError(f"Stage 3 item `{rid}` has empty text.")
        if not source_units:
            raise ExtractionError(f"Stage 3 item `{rid}` has no valid source_units.")
        if authorized_rewrite_candidates is not None:
            authorized_type = _resolve_rewrite_authority_type(
                rewrite_text=text,
                rewrite_sources=source_units,
                authorized_rewrite_candidates=authorized_rewrite_candidates,
            )
            if authorized_type is None:
                dropped_ungrounded += 1
                continue
            rtype = authorized_type

        key = (rtype, normalize_text(text), tuple(sorted(set(source_units))))
        if key in seen:
            continue
        seen.add(key)
        rewritten.append(
            RewrittenItem(
                id=rid,
                type=rtype,
                text=text,
                source_units=source_units,
            )
        )

    if authorized_rewrite_candidates is not None and dropped_ungrounded > 0 and not rewritten:
        raise ExtractionError("Stage 3 produced only ungrounded rewritten items.")

    return Stage3RewrittenOutput(rewritten_items=rewritten)


def validate_stage_4_open_questions(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> Stage4OpenQuestionsOutput:
    _, raw_items = _first_list_field(
        payload,
        ("open_questions", "questions", "clarification_questions", "unresolved_questions"),
    )
    if raw_items is None:
        raise ExtractionError("Stage 4 output must include list field `open_questions`.")

    return Stage4OpenQuestionsOutput(
        open_questions=_coerce_question_items(
            raw_items=raw_items,
            conversation_units=conversation_units,
            stage_label="Stage 4 open",
        )
    )


def validate_stage_5_followups(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> Stage5FollowUpOutput:
    field_name, raw_items = _first_question_collection(
        payload,
        (
            "follow_up_questions",
            "followup_questions",
            "followups",
            "followup",
            "follow_up",
            "questions",
            "clarification_questions",
            "developer_questions",
            "open_questions",
        ),
    )
    if raw_items is None:
        raise ExtractionError("Stage 5 output must include list field `follow_up_questions`.")

    if field_name == "open_questions":
        # Some weaker instruction models mirror the previous stage key in Stage 5.
        # Treat those as developer follow-up questions rather than failing the run.
        raw_items = [
            {
                "text": (
                    str(item).strip()
                    if isinstance(item, str)
                    else str(item.get("text", "") or item.get("question", "")).strip()
                ),
                "source_units": item.get("source_units", []) if isinstance(item, dict) else [],
            }
            for item in raw_items
        ]

    return Stage5FollowUpOutput(
        follow_up_questions=_coerce_question_items(
            raw_items=raw_items,
            conversation_units=conversation_units,
            stage_label="Stage 5 follow-up",
        )
    )


def validate_stage_6_summary(payload: dict[str, Any]) -> Stage6SummaryOutput:
    if "project_summary" not in payload:
        raise ExtractionError("Stage 6 output must include `project_summary`.")
    summary = str(payload.get("project_summary", "")).strip()
    if not summary:
        raise ExtractionError("Stage 6 `project_summary` must be non-empty.")
    return Stage6SummaryOutput(project_summary=summary)


# Backward-compatible aliases for old stage numbering.
def validate_stage_4_followups(
    payload: dict[str, Any],
    conversation_units: Iterable[ConversationUnit],
) -> Stage4FollowUpOutput:
    out = validate_stage_5_followups(payload, conversation_units)
    return Stage4FollowUpOutput(follow_up_questions=out.follow_up_questions)


def validate_stage_5_summary(payload: dict[str, Any]) -> Stage5SummaryOutput:
    out = validate_stage_6_summary(payload)
    return Stage5SummaryOutput(project_summary=out.project_summary)


def _tokens(text: str) -> set[str]:
    parts = normalize_text(text).split()
    return {token for token in parts if token and token not in STOPWORDS}


def _is_quality_oriented_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in QUALITY_HINTS)


def _is_vague_quality_text(text: str) -> bool:
    text_lc = normalize_text(text)
    if not _is_quality_oriented_text(text_lc):
        return False
    has_vague_hint = any(hint in text_lc for hint in VAGUE_QUALITY_HINTS)
    has_measurement_hint = any(hint in text_lc for hint in QUALITY_MEASUREMENT_HINTS)
    has_number = bool(re.search(r"\b\d+(\.\d+)?\b", text_lc))
    return has_vague_hint and not (has_measurement_hint or has_number)


def _looks_like_capability_statement(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in CAPABILITY_HINTS)


def _normalize_stage1_kind(kind: str, text: str) -> str:
    if kind == "possible_requirement":
        if _is_quality_oriented_text(text) and not _looks_like_capability_statement(text):
            return "possible_quality_expectation"
    return kind


def _sanitize_stage2_final_type(
    *,
    candidate_kind: str,
    text: str,
    final_type: str,
) -> tuple[str, str | None]:
    quality_oriented = _is_quality_oriented_text(text)
    capability_like = _looks_like_capability_statement(text)
    vague_quality = _is_vague_quality_text(text)

    if final_type == "functional_requirement":
        if candidate_kind == "possible_quality_expectation" and not capability_like:
            if vague_quality:
                return "open_question", "Quality expectation is vague and needs clarification."
            return "non_functional_requirement", "Quality-oriented statement should be treated as non-functional."
        if quality_oriented and not capability_like:
            if vague_quality:
                return "open_question", "Vague quality/style statement should be clarified before finalizing."
            return "non_functional_requirement", "Quality-oriented statement without concrete capability is non-functional."

    if final_type == "non_functional_requirement" and vague_quality:
        return "open_question", "Vague quality/style statement should remain an open question."

    return final_type, None


def _is_stage1_placeholder_text(text: str) -> bool:
    text_lc = normalize_text(text)
    if not text_lc:
        return True
    if any(hint in text_lc for hint in STAGE1_PLACEHOLDER_TEXT_HINTS):
        return True
    if text_lc.startswith(("short ", "example ", "sample ", "placeholder ")):
        return True
    return False


def _looks_like_constraint_text(text: str) -> bool:
    text_lc = normalize_text(text)
    constraint_hints = (
        "not part of the first release",
        "not in version one",
        "not in v1",
        "outside initial release",
        "initial release only",
        "web only",
        "no app",
        "within",
        "deadline",
        "budget",
        "only staff",
        "only admin",
        "must launch",
    )
    return any(hint in text_lc for hint in constraint_hints)


def _looks_like_future_scope_text(text: str) -> bool:
    text_lc = normalize_text(text)
    return any(hint in text_lc for hint in FUTURE_SCOPE_HINTS)


def build_stage_1_fallback_candidates(
    conversation_units: Iterable[ConversationUnit],
) -> Stage1CandidatesOutput:
    units = list(conversation_units)
    candidates: list[CandidateItem] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()

    def _append_candidate(kind: str, text: str, source_units: list[str]) -> None:
        normalized_text = text.strip().rstrip(".")
        if not normalized_text:
            return
        key = (kind, normalize_text(normalized_text), tuple(sorted(set(source_units))))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            CandidateItem(
                id=f"C{len(candidates) + 1}",
                kind=kind,
                text=normalized_text,
                source_units=source_units,
            )
        )

    for unit in units:
        text = unit.text.strip()
        if not text or _is_stage1_placeholder_text(text):
            continue
        source_units = [unit.id]
        quality_like = _is_quality_oriented_text(text)
        capability_like = _looks_like_capability_statement(text) or any(
            phrase in normalize_text(text) for phrase in ("need ", "needs ", "must be able to")
        )
        future_scope_like = _looks_like_future_scope_text(text)
        constraint_like = _looks_like_constraint_text(text)
        ambiguity_like = ("?" in text) or _is_vague_quality_text(text) or any(
            hint in normalize_text(text) for hint in VAGUE_HINTS
        )

        if quality_like:
            _append_candidate("possible_quality_expectation", text, source_units)
        if capability_like and not quality_like:
            _append_candidate("possible_requirement", text, source_units)
        if constraint_like:
            _append_candidate("possible_constraint", text, source_units)
        if future_scope_like:
            _append_candidate("possible_future_scope", text, source_units)
        if ambiguity_like:
            _append_candidate("possible_ambiguity", text, source_units)
            _append_candidate("possible_followup_trigger", text, source_units)

        if not any(
            kind in {
                "possible_requirement",
                "possible_quality_expectation",
                "possible_constraint",
                "possible_future_scope",
                "possible_ambiguity",
                "possible_followup_trigger",
            }
            for kind, ntext, src in seen
            if ntext == normalize_text(text) and src == tuple(source_units)
        ):
            _append_candidate("possible_requirement", text, source_units)

    if not candidates:
        raise ExtractionError("Stage 1 fallback could not derive any candidates from conversation.")

    return Stage1CandidatesOutput(candidates=candidates)


def _rewrite_text_alignment_score(rewrite_text: str, candidate_text: str) -> float:
    rewrite_tokens = _tokens(rewrite_text)
    candidate_tokens = _tokens(candidate_text)
    if not rewrite_tokens or not candidate_tokens:
        return 0.0
    overlap = len(rewrite_tokens & candidate_tokens)
    denom = max(1, min(len(rewrite_tokens), len(candidate_tokens)))
    return overlap / denom


def _resolve_rewrite_authority_type(
    *,
    rewrite_text: str,
    rewrite_sources: list[str],
    authorized_rewrite_candidates: list[dict[str, Any]],
) -> str | None:
    rewrite_sources_set = set(rewrite_sources)
    best_type: str | None = None
    best_score = 0.0
    for candidate in authorized_rewrite_candidates:
        candidate_type = str(candidate.get("final_type", "")).strip()
        candidate_text = str(candidate.get("text", "")).strip()
        candidate_sources = [
            str(x).strip() for x in candidate.get("source_units", []) if str(x).strip()
        ]
        if not rewrite_sources_set.intersection(candidate_sources):
            continue
        score = _rewrite_text_alignment_score(rewrite_text, candidate_text)
        if score > best_score:
            best_score = score
            best_type = candidate_type
    if best_score >= 0.2:
        return best_type
    return None


def coerce_rewrite_type_for_quality(text: str, rewrite_type: str) -> tuple[str, str | None]:
    if rewrite_type == "functional_requirement":
        if _is_quality_oriented_text(text) and not _looks_like_capability_statement(text):
            if _is_vague_quality_text(text):
                return "open_question", f"Please clarify measurable quality expectations for: {text}"
            return "non_functional_requirement", None
    if rewrite_type == "non_functional_requirement":
        if _is_vague_quality_text(text):
            return "open_question", f"Please clarify measurable quality expectations for: {text}"
    return rewrite_type, None


def _dedupe_question_items(items: list[QuestionItem]) -> list[QuestionItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[QuestionItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_note_items(items: list[NoteItem]) -> list[NoteItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[NoteItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _dedupe_constraint_items(items: list[ConstraintItem]) -> list[ConstraintItem]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    deduped: list[ConstraintItem] = []
    for item in items:
        key = (normalize_text(item.text), tuple(sorted(set(item.source_units))))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def semantic_verify(
    spec_output: SpecOutput,
    conversation_units: Iterable[ConversationUnit],
) -> tuple[SpecOutput, list[str]]:
    unit_map = {unit.id: unit.text for unit in conversation_units}
    warnings: list[str] = []

    verified_fr: list[RequirementItem] = []
    verified_nfr: list[RequirementItem] = []
    verified_constraints: list[ConstraintItem] = []
    open_questions: list[QuestionItem] = list(spec_output.open_questions)
    notes: list[NoteItem] = list(spec_output.notes)

    def _overlap_score(text: str, source_ids: list[str]) -> float:
        source_text = " ".join(unit_map[sid] for sid in source_ids if sid in unit_map)
        requirement_tokens = _tokens(text)
        source_tokens = _tokens(source_text)
        if not requirement_tokens:
            return 0.0
        return len(requirement_tokens & source_tokens) / len(requirement_tokens)

    def _evaluate_item(
        text: str,
        source_units: list[str],
        category: str,
    ) -> tuple[str, list[str], float, str]:
        reasons: list[str] = []
        valid_sources = [sid for sid in source_units if sid in unit_map]
        if not source_units:
            reasons.append("missing_source_units")
        if source_units and not valid_sources:
            reasons.append("source_units_not_found")

        overlap = _overlap_score(text, valid_sources)
        requirement_tokens = _tokens(text)
        overlap_threshold = 0.2 if category == "constraints" else 0.12
        if requirement_tokens and len(requirement_tokens) >= 4 and overlap < overlap_threshold:
            reasons.append("low_semantic_overlap")

        evidence_reasons = {
            "missing_source_units",
            "source_units_not_found",
            "low_semantic_overlap",
        }
        if category == "constraints" and any(reason in reasons for reason in evidence_reasons):
            inferred_sources = _infer_source_units_from_text(text, conversation_units)
            inferred_overlap = _overlap_score(text, inferred_sources)
            if (
                inferred_sources
                and inferred_overlap >= 0.2
                and inferred_overlap >= (overlap + 0.08)
            ):
                valid_sources = inferred_sources
                overlap = inferred_overlap
                reasons = [r for r in reasons if r not in evidence_reasons]
                reasons.append("source_units_repaired_by_inference")

        source_text = " ".join(unit_map[sid] for sid in valid_sources)
        text_lc = text.lower()
        source_lc = source_text.lower()
        has_future_scope = any(h in text_lc or h in source_lc for h in FUTURE_SCOPE_HINTS)
        has_vague_wording = any(h in text_lc for h in VAGUE_HINTS)
        has_hard_boundary = any(h in text_lc or h in source_lc for h in HARD_BOUNDARY_HINTS)

        action = "keep"
        if category == "constraints":
            reasons_requiring_downgrade = [
                reason for reason in reasons if reason != "source_units_repaired_by_inference"
            ]
            if has_future_scope and not has_hard_boundary:
                action = "downgrade_note"
                reasons.append("future_scope_without_explicit_boundary")
            elif any(reason in reasons for reason in evidence_reasons):
                action = "downgrade_question"
            elif reasons_requiring_downgrade and not has_hard_boundary:
                action = "downgrade_question"
            elif has_vague_wording:
                reasons.append("vague_constraint")
        else:
            if has_future_scope:
                action = "downgrade_note"
                reasons.append("future_scope")
            elif reasons:
                action = "downgrade_question"
            elif has_vague_wording and category == "functional_requirements":
                action = "downgrade_question"
                reasons.append("vague_functional_requirement")
            elif has_vague_wording and category == "non_functional_requirements":
                reasons.append("vague_non_functional_requirement")

        warning = ""
        if reasons:
            warning = f"downgraded/flagged ({', '.join(dict.fromkeys(reasons))})"
        return action, valid_sources or source_units, overlap, warning

    for item in spec_output.functional_requirements:
        action, sources, overlap, warning = _evaluate_item(
            item.text, item.source_units, "functional_requirements"
        )
        if action == "keep":
            verified_fr.append(
                RequirementItem(
                    id=item.id,
                    text=item.text,
                    source_units=sources,
                )
            )
            if warning:
                warnings.append(f"functional_requirements:{item.id} {warning}, overlap={overlap:.2f}")
            continue

        if warning:
            warnings.append(f"functional_requirements:{item.id} {warning}, overlap={overlap:.2f}")

        if action == "downgrade_note":
            notes.append(
                NoteItem(
                    text=f"Future-scope candidate requirement: {item.text}",
                    source_units=sources,
                )
            )
        else:
            open_questions.append(
                QuestionItem(
                    text=f"Please confirm requirement scope and intent: {item.text}",
                    source_units=sources,
                )
            )

    for item in spec_output.non_functional_requirements:
        action, sources, overlap, warning = _evaluate_item(
            item.text, item.source_units, "non_functional_requirements"
        )
        if action == "keep":
            verified_nfr.append(
                RequirementItem(
                    id=item.id,
                    text=item.text,
                    source_units=sources,
                )
            )
            if warning:
                warnings.append(
                    f"non_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}"
                )
                open_questions.append(
                    QuestionItem(
                        text=f"Please provide measurable acceptance criteria for: {item.text}",
                        source_units=sources,
                    )
                )
            continue

        if warning:
            warnings.append(
                f"non_functional_requirements:{item.id} {warning}, overlap={overlap:.2f}"
            )

        if action == "downgrade_note":
            notes.append(
                NoteItem(
                    text=f"Future-scope quality expectation: {item.text}",
                    source_units=sources,
                )
            )
        else:
            open_questions.append(
                QuestionItem(
                    text=f"Please confirm non-functional requirement details: {item.text}",
                    source_units=sources,
                )
            )

    for item in spec_output.constraints:
        action, sources, overlap, warning = _evaluate_item(item.text, item.source_units, "constraints")
        if action == "keep":
            verified_constraints.append(
                ConstraintItem(
                    id=item.id,
                    text=item.text,
                    source_units=sources,
                )
            )
            if warning:
                warnings.append(f"constraints:{item.id} {warning}, overlap={overlap:.2f}")
                if "vague_constraint" in warning:
                    open_questions.append(
                        QuestionItem(
                            text=f"Please specify concrete boundary details for: {item.text}",
                            source_units=sources,
                        )
                    )
            continue

        if warning:
            warnings.append(f"constraints:{item.id} {warning}, overlap={overlap:.2f}")

        if action == "downgrade_note":
            notes.append(
                NoteItem(
                    text=f"Future-scope boundary candidate: {item.text}",
                    source_units=sources,
                )
            )
        else:
            open_questions.append(
                QuestionItem(
                    text=f"Please clarify whether this is a hard project constraint: {item.text}",
                    source_units=sources,
                )
            )

    payload = model_dump_compat(spec_output)
    payload["functional_requirements"] = [model_dump_compat(item) for item in verified_fr]
    payload["non_functional_requirements"] = [model_dump_compat(item) for item in verified_nfr]
    payload["constraints"] = [
        model_dump_compat(item) for item in _dedupe_constraint_items(verified_constraints)
    ]
    payload["open_questions"] = [
        model_dump_compat(item) for item in _dedupe_question_items(open_questions)
    ]
    payload["notes"] = [model_dump_compat(item) for item in _dedupe_note_items(notes)]
    payload["verification_warnings"] = sorted(set(warnings))
    verified_spec = model_validate_compat(SpecOutput, payload)
    return verified_spec, sorted(set(warnings))


# Backward-compatible single-shot helpers.
def extract_spec_output_safe(
    raw_output: str,
    conversation_units: Iterable[ConversationUnit],
    *,
    allow_repair: bool = True,
) -> tuple[SpecOutput | None, ExtractionMeta]:
    payload, meta = parse_json_object_safe(raw_output, allow_repair=allow_repair)
    if payload is None:
        return None, meta
    try:
        required_fields = (
            "project_summary",
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
            "open_questions",
            "follow_up_questions",
            "notes",
        )
        missing = [key for key in required_fields if key not in payload]
        if missing:
            raise ExtractionError(f"Missing required top-level fields: {', '.join(missing)}")
        for list_key in (
            "functional_requirements",
            "non_functional_requirements",
            "constraints",
            "open_questions",
            "follow_up_questions",
            "notes",
        ):
            if not isinstance(payload.get(list_key), list):
                raise ExtractionError(f"Field `{list_key}` must be a list.")
        if not isinstance(payload.get("project_summary"), str) or not str(
            payload.get("project_summary", "")
        ).strip():
            raise ExtractionError("Field `project_summary` must be a non-empty string.")

        normalized = {
            "project_summary": str(payload.get("project_summary", "")).strip()
            or "The conversation describes a software project requiring clarification.",
            "functional_requirements": payload.get("functional_requirements", []),
            "non_functional_requirements": payload.get("non_functional_requirements", []),
            "constraints": payload.get("constraints", []),
            "open_questions": payload.get("open_questions", []),
            "follow_up_questions": payload.get("follow_up_questions", []),
            "notes": payload.get("notes", []),
            "conversation_units": [model_dump_compat(u) for u in conversation_units],
            "verification_warnings": payload.get("verification_warnings", []),
        }
        spec = model_validate_compat(SpecOutput, normalized)
        meta.pydantic_validation_ok = True
        return spec, meta
    except Exception as exc:
        meta.validation_error = f"Schema validation failed: {exc}"
        return None, meta


def extract_spec_output(
    raw_output: str,
    conversation_units: Iterable[ConversationUnit],
) -> tuple[SpecOutput, ExtractionMeta]:
    spec, meta = extract_spec_output_safe(raw_output, conversation_units)
    if spec is None:
        if meta.validation_error:
            raise ExtractionError(meta.validation_error)
        if meta.parse_error:
            raise ExtractionError(meta.parse_error)
        raise ExtractionError("Unknown extraction failure.")
    return spec, meta
