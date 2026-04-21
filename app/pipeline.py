from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from app.extractor import ExtractionResult, extract_spec_output, parse_json_payload
from app.formatter import spec_to_markdown
from app.model_runner import BaseModelRunner, GenerationResult, MockModelRunner, create_classification_runner
from app.parser import parse_conversation_file, parse_conversation_text
from app.schemas import QuestionDecisionItem, NoteItem, RequirementItem, RunMetadata, SpecOutput
from app.utils import ensure_directory, normalize_text

QUALITY_AMBIGUOUS_HINTS = (
    "simple",
    "easy to use",
    "clean",
    "fast",
    "modern",
    "user-friendly",
)
UNCERTAINTY_HINTS = (
    "maybe",
    "later",
    "nice to have",
    "not decided",
    "not decided yet",
    "for now",
)
FUNCTIONAL_HINTS = (
    "need",
    "needs",
    "must",
    "should",
    "allow",
    "can",
    "able to",
    "create",
    "update",
    "edit",
    "delete",
    "book",
    "reserve",
    "schedule",
    "manage",
    "view",
    "track",
    "notify",
    "login",
    "sign in",
    "post",
    "publish",
    "request",
    "approve",
)
CONSTRAINT_HINTS = (
    "budget",
    "deadline",
    "timeline",
    "must use",
    "cannot",
    "can't",
    "within",
    "only",
    "web only",
    "on-prem",
    "on prem",
    "school servers",
    "three month",
    "3 month",
    "not ios",
    "not android",
)
STOPWORDS = {
    "a",
    "an",
    "the",
    "to",
    "for",
    "of",
    "in",
    "on",
    "with",
    "and",
    "or",
    "is",
    "are",
    "be",
    "by",
    "from",
    "it",
    "that",
    "this",
    "we",
    "our",
    "should",
    "must",
    "can",
    "will",
}
PRIORITY_RANK = {"high": 3, "medium": 2, "low": 1}
PM_SPEAKER_ALIASES = {
    "pm",
    "developer",
    "dev",
    "engineer",
    "team lead",
    "assistant",
    "analyst",
    "개발자",
}
FOLLOW_UP_PROMPT_HINTS = (
    "deadline",
    "specific",
    "clarify",
    "confirm",
    "by when",
    "when",
    "target",
    "give us",
    "need to know",
    "decide",
    "finalize",
)
SUPPORT_ONLY_HINTS = (
    "you should give us",
    "can you",
    "could you",
    "please clarify",
    "please confirm",
    "specific deadline",
    "acceptance criteria",
    "scope",
    "timeline",
    "target date",
    "요구사항",
    "구체적으로",
    "확인",
    "정리",
)
ANSWER_HINTS = (
    "it's okay",
    "its okay",
    "ok for",
    "okay for",
    "can be",
    "works for",
    "가능",
    "괜찮",
    "까지",
)
DATE_TEXT_PATTERN = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    flags=re.IGNORECASE,
)
HYBRID_CLASS_LABELS = [
    "functional requirement",
    "non-functional requirement",
    "constraint",
    "assumption",
    "open question",
]
QUESTION_TOPIC_STOPWORDS = STOPWORDS | {
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "can",
    "could",
    "should",
    "would",
    "please",
    "clarify",
    "confirm",
    "define",
    "measurement",
    "measure",
    "measurable",
    "criteria",
    "target",
    "question",
    "questions",
    "follow",
    "up",
    "allowed",
    "allow",
    "need",
    "needs",
    "required",
    "requirement",
}


def _is_pm_speaker(speaker: str | None) -> bool:
    if speaker is None:
        return False
    return str(speaker).strip().lower() in PM_SPEAKER_ALIASES


def _is_constraint_like(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in CONSTRAINT_HINTS):
        return True
    if lowered.startswith("only ") and (
        " can " in lowered
        or " should " in lowered
        or " allowed " in lowered
        or "permit" in lowered
    ):
        return True
    if "must use" in lowered or "cannot" in lowered or "can't" in lowered:
        return True
    return False


def _normalize_utterance_index(utterance_id: str) -> int:
    match = re.match(r"^U(\d+)$", str(utterance_id).strip())
    if not match:
        return 10**9
    return int(match.group(1))


def _is_question_like_text(text: str) -> bool:
    lowered = text.lower()
    if "?" in text:
        return True
    return any(token in lowered for token in FOLLOW_UP_PROMPT_HINTS)


def _is_support_only_unknown_text(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return False
    if _is_question_like_text(text):
        return True
    if any(token in lowered for token in SUPPORT_ONLY_HINTS):
        has_requester_style = any(
            token in lowered
            for token in ("we need", "we want", "our users", "our team", "our school", "우리", "필요", "원해")
        )
        if not has_requester_style:
            return True
    return False


def _is_support_only_utterance(utterance: dict[str, str] | None) -> bool:
    if not utterance:
        return False
    speaker = str(utterance.get("speaker", "")).strip()
    if _is_pm_speaker(speaker):
        return True
    if speaker.lower() == "unknown":
        return _is_support_only_unknown_text(str(utterance.get("text", "")))
    return False


def _has_requester_evidence(
    evidence_ids: list[str],
    utterance_map: dict[str, dict[str, str]],
) -> bool:
    for evidence_id in evidence_ids:
        utterance = utterance_map.get(evidence_id)
        if not utterance:
            continue
        if not _is_support_only_utterance(utterance):
            return True
    return False


def _looks_answer_like_text(text: str) -> bool:
    lowered = text.lower()
    if DATE_TEXT_PATTERN.search(text):
        return True
    if re.search(r"\b\d{1,2}(?:st|nd|rd|th)\b", lowered):
        return True
    if re.search(r"\b\d{1,4}\b", text):
        return True
    return any(token in lowered for token in ANSWER_HINTS)


def _topic_follow_up_status(
    source_id: str,
    topic_text: str,
    utterances: list[dict[str, str]],
) -> str:
    source_idx = _normalize_utterance_index(source_id)
    topic_tokens = _text_token_set(topic_text)

    asked_idx: int | None = None
    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip()
        idx = _normalize_utterance_index(utterance_id)
        if idx <= source_idx:
            continue

        text = str(utterance.get("text", "")).strip()
        if not text:
            continue
        lowered = text.lower()
        tokens = _text_token_set(text)
        overlap = len(topic_tokens & tokens) if topic_tokens else 0

        if asked_idx is None:
            if overlap >= 1 and _is_question_like_text(text):
                asked_idx = idx
                continue
            if any(token in lowered for token in FOLLOW_UP_PROMPT_HINTS) and overlap >= 1:
                asked_idx = idx
                continue
            continue

        if _is_question_like_text(text):
            continue

        if overlap >= 1:
            return "resolved"
        if idx - asked_idx <= 2 and _looks_answer_like_text(text):
            return "resolved"

    if asked_idx is not None:
        return "already_asked"
    return "needs_follow_up"


def _topic_token_set(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = [token for token in normalized.split() if token and token not in QUESTION_TOPIC_STOPWORDS and len(token) > 1]
    return set(tokens)


def _topic_similarity(left: str, right: str) -> float:
    left_tokens = _topic_token_set(left)
    right_tokens = _topic_token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    inter = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    if union == 0:
        return 0.0
    jaccard = inter / union
    containment = max(inter / len(left_tokens), inter / len(right_tokens))
    return max(jaccard, containment)


def _pm_question_status(
    source_id: str,
    topic_text: str,
    utterances: list[dict[str, str]],
) -> str:
    source_idx = _normalize_utterance_index(source_id)
    topic_tokens = _text_token_set(topic_text)

    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip()
        idx = _normalize_utterance_index(utterance_id)
        if idx <= source_idx:
            continue
        if _is_support_only_utterance(utterance):
            continue

        text = str(utterance.get("text", "")).strip()
        if not text or _is_question_like_text(text):
            continue

        answer_tokens = _text_token_set(text)
        if topic_tokens and len(topic_tokens & answer_tokens) >= 1:
            return "resolved"
        if idx - source_idx <= 2 and _looks_answer_like_text(text):
            return "resolved"

    return "already_asked"


@dataclass
class PipelineDiagnostics:
    json_parse_ok: bool = True
    pydantic_validation_ok: bool = True
    first_pass_json_parse_ok: bool = True
    first_pass_pydantic_validation_ok: bool = True
    retries_used: int = 0
    chunks_used: int = 0
    used_final_refinement: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    attempt_logs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    spec: SpecOutput
    generation: GenerationResult
    json_path: Path
    markdown_path: Path
    diagnostics: PipelineDiagnostics


@dataclass
class ModelCallOutcome:
    spec: SpecOutput | None
    generation: GenerationResult
    extraction: ExtractionResult
    first_extraction: ExtractionResult
    retries_used: int
    attempt_logs: list[dict[str, Any]]


@dataclass
class PayloadCallOutcome:
    payload: dict[str, Any] | None
    generation: GenerationResult
    error: str | None
    retries_used: int
    attempt_logs: list[dict[str, Any]]


class PipelineExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        diagnostics: PipelineDiagnostics,
        generation: GenerationResult | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics
        self.generation = generation


def _sum_optional_int(values: list[int | None]) -> int | None:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return int(sum(valid))


def _progress_enabled(prompt_config: dict) -> bool:
    return bool(prompt_config.get("progress_logging", True))


def _progress_log(prompt_config: dict, message: str) -> None:
    if not _progress_enabled(prompt_config):
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [pipeline] {message}", flush=True)


def _shorten_for_log(text: str, max_len: int = 240) -> str:
    one_line = re.sub(r"\s+", " ", text).strip()
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 3] + "..."


def _attempt_log_entry(
    stage_label: str,
    attempt_index: int,
    repair_mode: bool,
    generation: GenerationResult,
    extraction: ExtractionResult,
) -> dict[str, Any]:
    return {
        "stage": stage_label,
        "attempt_index": attempt_index,
        "repair_mode": repair_mode,
        "model_name": generation.model_name,
        "latency_sec": generation.latency_sec,
        "prompt_tokens": generation.prompt_tokens,
        "completion_tokens": generation.completion_tokens,
        "json_parse_ok": extraction.json_parse_ok,
        "pydantic_validation_ok": extraction.pydantic_validation_ok,
        "extraction_error": extraction.error,
        "raw_output_chars": len(generation.raw_text),
        "raw_output_preview": _shorten_for_log(generation.raw_text, max_len=400),
        "raw_output": generation.raw_text,
    }


def _merge_generation_results(results: list[GenerationResult]) -> GenerationResult:
    if not results:
        return GenerationResult(model_name="unknown", raw_text="", latency_sec=0.0)

    return GenerationResult(
        model_name=results[-1].model_name,
        raw_text=results[-1].raw_text,
        latency_sec=sum(item.latency_sec for item in results),
        prompt_tokens=_sum_optional_int([item.prompt_tokens for item in results]),
        completion_tokens=_sum_optional_int([item.completion_tokens for item in results]),
    )


def _sort_utterance_ids(values: set[str]) -> list[str]:
    def sort_key(token: str) -> tuple[int, str]:
        match = re.match(r"^U(\d+)$", token)
        if match:
            return int(match.group(1)), token
        return 10**9, token

    return sorted(values, key=sort_key)


def _normalize_priority(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if cleaned in PRIORITY_RANK:
        return cleaned
    return None


def _priority_max(left: str | None, right: str | None) -> str | None:
    left_norm = _normalize_priority(left)
    right_norm = _normalize_priority(right)
    if left_norm is None:
        return right_norm
    if right_norm is None:
        return left_norm
    if PRIORITY_RANK[right_norm] > PRIORITY_RANK[left_norm]:
        return right_norm
    return left_norm


def _estimate_utterance_tokens(utterance: dict[str, str]) -> int:
    # rough approximation: text chars/4 + small fixed overhead for role/id markers
    text = f"{utterance.get('speaker', '')}: {utterance.get('text', '')}".strip()
    return max(8, int(len(text) / 4) + 6)


def _chunk_utterances(
    utterances: list[dict[str, str]],
    token_budget: int,
    max_chunk_utterances: int,
) -> list[list[dict[str, str]]]:
    if not utterances:
        return []

    chunks: list[list[dict[str, str]]] = []
    current_chunk: list[dict[str, str]] = []
    current_tokens = 0

    for utterance in utterances:
        estimated = _estimate_utterance_tokens(utterance)
        projected = current_tokens + estimated

        should_split = (
            current_chunk
            and (projected > token_budget or len(current_chunk) >= max_chunk_utterances)
        )
        if should_split:
            chunks.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        current_chunk.append(utterance)
        current_tokens += estimated

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def _text_token_set(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = [token for token in normalized.split() if token and token not in STOPWORDS and len(token) > 1]
    return set(tokens)


def _evidence_overlap_ratio(text: str, evidence_ids: list[str], utterance_map: dict[str, dict[str, str]]) -> float:
    item_tokens = _text_token_set(text)
    if not item_tokens:
        return 0.0

    evidence_tokens: set[str] = set()
    for evidence_id in evidence_ids:
        utterance = utterance_map.get(evidence_id)
        if not utterance:
            continue
        evidence_tokens.update(_text_token_set(utterance.get("text", "")))

    if not evidence_tokens:
        return 0.0

    return len(item_tokens & evidence_tokens) / len(item_tokens)


def _contains_phrase(text: str, phrases: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _looks_quantified(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"\d", lowered):
        return True
    quant_tokens = ("ms", "millisecond", "second", "minute", "hour", "sla", "%", "percent", "under")
    return any(token in lowered for token in quant_tokens)


def _default_question_from_note(note: str) -> str:
    stem = note.strip().rstrip(".")
    if not stem:
        return "What details are still unresolved?"
    if stem.endswith("?"):
        return stem
    return f"Can you clarify: {stem}?"


def _has_client_answer_after(
    question_id: str,
    question_text: str,
    utterances: list[dict[str, str]],
) -> bool:
    q_idx = _normalize_utterance_index(question_id)
    question_tokens = _text_token_set(question_text)
    if not question_tokens:
        question_tokens = _text_token_set(" ".join(question_text.lower().split()[:6]))

    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip()
        if _normalize_utterance_index(utterance_id) <= q_idx:
            continue

        speaker = str(utterance.get("speaker", "")).strip()
        if _is_pm_speaker(speaker):
            continue

        text = str(utterance.get("text", "")).strip()
        if not text or "?" in text:
            continue

        answer_tokens = _text_token_set(text)
        if not answer_tokens:
            continue
        if not question_tokens:
            return True
        if len(question_tokens & answer_tokens) >= 1:
            return True
    return False


def _build_question_decisions_from_notes(spec: SpecOutput) -> list[QuestionDecisionItem]:
    decisions: list[QuestionDecisionItem] = []
    utterances = [item.model_dump() for item in spec.utterances]
    utterance_map = {str(item.get("id", "")): item for item in utterances}

    for note in spec.open_questions:
        evidence = list(note.evidence)
        if not evidence:
            continue

        first_id = evidence[0]
        source_utterance = utterance_map.get(first_id, {})
        source_speaker = str(source_utterance.get("speaker", "")).strip()
        source_text = str(source_utterance.get("text", "")).strip()

        if _is_pm_speaker(source_speaker) and "?" in source_text:
            if _has_client_answer_after(first_id, note.text, utterances):
                decision = "resolved"
                suggested = None
            else:
                decision = "already_asked"
                suggested = None
        else:
            decision = "needs_follow_up"
            suggested = _default_question_from_note(note.text)

        decisions.append(
            QuestionDecisionItem(
                text=note.text,
                decision=decision,
                suggested_follow_up=suggested,
                evidence=evidence,
            )
        )

    for note in spec.assumptions:
        evidence = list(note.evidence)
        if not evidence:
            continue
        status = _topic_follow_up_status(
            source_id=evidence[0],
            topic_text=note.text,
            utterances=utterances,
        )
        suggested = _default_question_from_note(note.text) if status == "needs_follow_up" else None
        decisions.append(
            QuestionDecisionItem(
                text=note.text,
                decision=status,
                suggested_follow_up=suggested,
                evidence=evidence,
            )
        )

    return _merge_question_decisions(decisions)


def _merge_requirement_items(items: list[RequirementItem], prefix: str) -> list[RequirementItem]:
    merged: dict[str, dict[str, Any]] = {}

    for item in items:
        key = normalize_text(item.text)
        if not key:
            continue

        if key not in merged:
            merged[key] = {
                "text": item.text.strip(),
                "priority": _normalize_priority(item.priority),
                "evidence": set(item.evidence),
            }
            continue

        merged[key]["priority"] = _priority_max(merged[key]["priority"], item.priority)
        merged[key]["evidence"].update(item.evidence)

    output: list[RequirementItem] = []
    for idx, payload in enumerate(merged.values(), start=1):
        evidence_ids = _sort_utterance_ids(set(payload["evidence"]))
        if not evidence_ids:
            continue
        output.append(
            RequirementItem(
                id=f"{prefix}{idx}",
                text=payload["text"],
                priority=payload["priority"],
                evidence=evidence_ids,
            )
        )

    return output


def _merge_note_items(items: list[NoteItem]) -> list[NoteItem]:
    merged: dict[str, dict[str, Any]] = {}

    for item in items:
        key = normalize_text(item.text)
        if not key:
            continue

        if key not in merged:
            merged[key] = {
                "text": item.text.strip(),
                "evidence": set(item.evidence),
            }
            continue

        merged[key]["evidence"].update(item.evidence)

    output: list[NoteItem] = []
    for payload in merged.values():
        evidence_ids = _sort_utterance_ids(set(payload["evidence"]))
        if not evidence_ids:
            continue
        output.append(NoteItem(text=payload["text"], evidence=evidence_ids))

    return output


def _merge_question_decisions(items: list[QuestionDecisionItem]) -> list[QuestionDecisionItem]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for item in items:
        key = (normalize_text(item.text), item.decision)
        if not key[0]:
            continue

        if key not in merged:
            merged[key] = {
                "text": item.text.strip(),
                "decision": item.decision,
                "suggested_follow_up": (item.suggested_follow_up or "").strip() or None,
                "evidence": set(item.evidence),
            }
            continue

        merged[key]["evidence"].update(item.evidence)
        if not merged[key]["suggested_follow_up"] and item.suggested_follow_up:
            merged[key]["suggested_follow_up"] = item.suggested_follow_up.strip()

    output: list[QuestionDecisionItem] = []
    for payload in merged.values():
        evidence_ids = _sort_utterance_ids(set(payload["evidence"]))
        if not evidence_ids:
            continue
        output.append(
            QuestionDecisionItem(
                text=payload["text"],
                decision=payload["decision"],
                suggested_follow_up=payload["suggested_follow_up"],
                evidence=evidence_ids,
            )
        )
    return output


def _merge_specs(specs: list[SpecOutput], utterances: list[dict[str, str]]) -> SpecOutput:
    merged_functional: list[RequirementItem] = []
    merged_non_functional: list[RequirementItem] = []
    merged_constraints: list[NoteItem] = []
    merged_assumptions: list[NoteItem] = []
    merged_open_questions: list[NoteItem] = []
    merged_follow_up: list[str] = []
    merged_question_decisions: list[QuestionDecisionItem] = []

    for spec in specs:
        merged_functional.extend(spec.functional_requirements)
        merged_non_functional.extend(spec.non_functional_requirements)
        merged_constraints.extend(spec.constraints)
        merged_assumptions.extend(spec.assumptions)
        merged_open_questions.extend(spec.open_questions)
        merged_follow_up.extend(spec.follow_up_questions)
        merged_question_decisions.extend(spec.question_decisions)

    normalized_follow_up: list[str] = []
    seen_follow_up: set[str] = set()
    for question in merged_follow_up:
        cleaned = question.strip()
        key = normalize_text(cleaned)
        if not cleaned or not key or key in seen_follow_up:
            continue
        seen_follow_up.add(key)
        normalized_follow_up.append(cleaned)

    return SpecOutput(
        project_summary="Merged chunk-level requirements draft.",
        functional_requirements=_merge_requirement_items(merged_functional, prefix="FR"),
        non_functional_requirements=_merge_requirement_items(merged_non_functional, prefix="NFR"),
        constraints=_merge_note_items(merged_constraints),
        assumptions=_merge_note_items(merged_assumptions),
        open_questions=_merge_note_items(merged_open_questions),
        follow_up_questions=normalized_follow_up,
        question_decisions=_merge_question_decisions(merged_question_decisions),
        utterances=utterances,
    )


def _apply_category_sanity_guards(spec: SpecOutput) -> SpecOutput:
    moved_constraints: list[NoteItem] = []
    moved_assumptions: list[NoteItem] = []
    moved_open_questions: list[NoteItem] = []

    kept_functional: list[RequirementItem] = []
    kept_non_functional: list[RequirementItem] = []

    def route_requirement(item: RequirementItem, keep_bucket: list[RequirementItem]) -> None:
        text = item.text.strip()
        lowered = text.lower()

        if "?" in text:
            moved_open_questions.append(NoteItem(text=text, evidence=item.evidence))
            return

        if any(token in lowered for token in UNCERTAINTY_HINTS):
            moved_assumptions.append(NoteItem(text=text, evidence=item.evidence))
            return

        if _is_constraint_like(text):
            moved_constraints.append(NoteItem(text=text, evidence=item.evidence))
            return

        keep_bucket.append(item)

    for item in spec.functional_requirements:
        route_requirement(item, kept_functional)
    for item in spec.non_functional_requirements:
        route_requirement(item, kept_non_functional)

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=kept_functional,
        non_functional_requirements=kept_non_functional,
        constraints=_merge_note_items(list(spec.constraints) + moved_constraints),
        assumptions=_merge_note_items(list(spec.assumptions) + moved_assumptions),
        open_questions=_merge_note_items(list(spec.open_questions) + moved_open_questions),
        follow_up_questions=spec.follow_up_questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _reclassify_ambiguous_items(spec: SpecOutput, ambiguous_phrases: list[str]) -> SpecOutput:
    quality_phrases = tuple(set([*QUALITY_AMBIGUOUS_HINTS, *[item.lower() for item in ambiguous_phrases]]))

    kept_functional: list[RequirementItem] = []
    promoted_to_nfr: list[RequirementItem] = []
    promoted_to_assumption: list[NoteItem] = []
    generated_open_questions: list[NoteItem] = []

    for item in spec.functional_requirements:
        text = item.text.strip()
        lower_text = text.lower()

        if _contains_phrase(lower_text, UNCERTAINTY_HINTS):
            promoted_to_assumption.append(NoteItem(text=text, evidence=item.evidence))
            continue

        if _contains_phrase(lower_text, quality_phrases):
            promoted_to_nfr.append(
                RequirementItem(
                    id=item.id,
                    text=text,
                    priority=item.priority,
                    evidence=item.evidence,
                )
            )
            if not _looks_quantified(lower_text):
                generated_open_questions.append(
                    NoteItem(
                        text=f"What measurable criteria define: {text}?",
                        evidence=item.evidence,
                    )
                )
            continue

        kept_functional.append(item)

    updated_nfr = list(spec.non_functional_requirements) + promoted_to_nfr
    for item in spec.non_functional_requirements:
        if _contains_phrase(item.text.lower(), quality_phrases) and not _looks_quantified(item.text):
            generated_open_questions.append(
                NoteItem(
                    text=f"How should this quality requirement be measured: {item.text}?",
                    evidence=item.evidence,
                )
            )

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=kept_functional,
        non_functional_requirements=updated_nfr,
        constraints=spec.constraints,
        assumptions=list(spec.assumptions) + promoted_to_assumption,
        open_questions=list(spec.open_questions) + generated_open_questions,
        follow_up_questions=spec.follow_up_questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _apply_hallucination_guard(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    min_overlap = float(prompt_config.get("min_evidence_overlap", 0.08))
    utterance_map = {item.id: item.model_dump() for item in spec.utterances}

    kept_functional: list[RequirementItem] = []
    kept_nfr: list[RequirementItem] = []
    moved_to_assumptions: list[NoteItem] = []

    def split_by_support(items: list[RequirementItem], bucket: list[RequirementItem]) -> None:
        for item in items:
            overlap_ratio = _evidence_overlap_ratio(item.text, item.evidence, utterance_map)
            if overlap_ratio >= min_overlap:
                bucket.append(item)
                continue

            moved_to_assumptions.append(
                NoteItem(
                    text=f"Uncertain requirement candidate (needs confirmation): {item.text}",
                    evidence=item.evidence,
                )
            )

    split_by_support(spec.functional_requirements, kept_functional)
    split_by_support(spec.non_functional_requirements, kept_nfr)

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=kept_functional,
        non_functional_requirements=kept_nfr,
        constraints=spec.constraints,
        assumptions=list(spec.assumptions) + moved_to_assumptions,
        open_questions=spec.open_questions,
        follow_up_questions=spec.follow_up_questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _clean_evidence(spec: SpecOutput) -> SpecOutput:
    valid_ids = {item.id for item in spec.utterances}

    cleaned_fr: list[RequirementItem] = []
    for item in spec.functional_requirements:
        evidence = [evidence_id for evidence_id in item.evidence if evidence_id in valid_ids]
        if not evidence:
            continue
        cleaned_fr.append(
            RequirementItem(id=item.id, text=item.text, priority=item.priority, evidence=_sort_utterance_ids(set(evidence)))
        )

    cleaned_nfr: list[RequirementItem] = []
    for item in spec.non_functional_requirements:
        evidence = [evidence_id for evidence_id in item.evidence if evidence_id in valid_ids]
        if not evidence:
            continue
        cleaned_nfr.append(
            RequirementItem(id=item.id, text=item.text, priority=item.priority, evidence=_sort_utterance_ids(set(evidence)))
        )

    def clean_notes(items: list[NoteItem]) -> list[NoteItem]:
        output: list[NoteItem] = []
        for item in items:
            evidence = [evidence_id for evidence_id in item.evidence if evidence_id in valid_ids]
            if not evidence:
                continue
            output.append(NoteItem(text=item.text, evidence=_sort_utterance_ids(set(evidence))))
        return output

    cleaned_question_decisions: list[QuestionDecisionItem] = []
    for item in spec.question_decisions:
        evidence = [evidence_id for evidence_id in item.evidence if evidence_id in valid_ids]
        if not evidence:
            continue
        cleaned_question_decisions.append(
            QuestionDecisionItem(
                text=item.text,
                decision=item.decision,
                suggested_follow_up=item.suggested_follow_up,
                evidence=_sort_utterance_ids(set(evidence)),
            )
        )

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=cleaned_fr,
        non_functional_requirements=cleaned_nfr,
        constraints=clean_notes(spec.constraints),
        assumptions=clean_notes(spec.assumptions),
        open_questions=clean_notes(spec.open_questions),
        follow_up_questions=spec.follow_up_questions,
        question_decisions=cleaned_question_decisions,
        utterances=spec.utterances,
    )


def _enforce_requester_source_only(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    if not bool(prompt_config.get("enforce_requester_source_only", True)):
        return spec

    utterance_dicts = [item.model_dump() for item in spec.utterances]
    utterance_map = {str(item.get("id", "")): item for item in utterance_dicts}

    kept_fr: list[RequirementItem] = []
    for item in spec.functional_requirements:
        if _has_requester_evidence(item.evidence, utterance_map):
            kept_fr.append(item)

    kept_nfr: list[RequirementItem] = []
    for item in spec.non_functional_requirements:
        if _has_requester_evidence(item.evidence, utterance_map):
            kept_nfr.append(item)

    kept_constraints: list[NoteItem] = []
    for item in spec.constraints:
        if _has_requester_evidence(item.evidence, utterance_map):
            kept_constraints.append(item)

    kept_assumptions: list[NoteItem] = []
    for item in spec.assumptions:
        if _has_requester_evidence(item.evidence, utterance_map):
            kept_assumptions.append(item)

    kept_open_questions: list[NoteItem] = []
    removed_support_open_questions: list[NoteItem] = []
    for item in spec.open_questions:
        if _has_requester_evidence(item.evidence, utterance_map):
            kept_open_questions.append(item)
        else:
            removed_support_open_questions.append(item)

    decisions = list(spec.question_decisions)
    for note in removed_support_open_questions:
        if not note.evidence:
            continue
        status = _pm_question_status(
            source_id=note.evidence[0],
            topic_text=note.text,
            utterances=utterance_dicts,
        )
        decisions.append(
            QuestionDecisionItem(
                text=note.text,
                decision=status,
                suggested_follow_up=None,
                evidence=note.evidence,
            )
        )

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=kept_fr,
        non_functional_requirements=kept_nfr,
        constraints=kept_constraints,
        assumptions=kept_assumptions,
        open_questions=kept_open_questions,
        follow_up_questions=spec.follow_up_questions,
        question_decisions=_merge_question_decisions(decisions),
        utterances=spec.utterances,
    )


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _decision_priority(decision: str) -> int:
    # Prefer closed state when multiple decision variants describe the same topic.
    order = {"resolved": 3, "already_asked": 2, "needs_follow_up": 1}
    return order.get(decision, 0)


def _consolidate_question_decisions(
    items: list[QuestionDecisionItem],
    similarity_threshold: float,
) -> list[QuestionDecisionItem]:
    if not items:
        return []

    groups: list[dict[str, Any]] = []
    for item in items:
        placed = False
        for group in groups:
            if _topic_similarity(item.text, group["topic_text"]) >= similarity_threshold:
                group["items"].append(item)
                if len(item.text) > len(group["topic_text"]):
                    group["topic_text"] = item.text
                placed = True
                break
        if not placed:
            groups.append({"topic_text": item.text, "items": [item]})

    consolidated: list[QuestionDecisionItem] = []
    for group in groups:
        candidates: list[QuestionDecisionItem] = list(group["items"])
        if not candidates:
            continue
        best = max(candidates, key=lambda item: _decision_priority(item.decision))
        evidence_ids: set[str] = set()
        for item in candidates:
            evidence_ids.update(item.evidence)

        suggested: str | None = None
        if best.decision == "needs_follow_up":
            for item in candidates:
                candidate_suggested = (item.suggested_follow_up or "").strip()
                if candidate_suggested:
                    suggested = candidate_suggested
                    break

        consolidated.append(
            QuestionDecisionItem(
                text=group["topic_text"].strip(),
                decision=best.decision,
                suggested_follow_up=suggested,
                evidence=_sort_utterance_ids(evidence_ids),
            )
        )

    consolidated.sort(
        key=lambda item: (
            0 if item.decision == "needs_follow_up" else 1 if item.decision == "already_asked" else 2,
            _normalize_utterance_index(item.evidence[0]) if item.evidence else 10**9,
            item.text.lower(),
        )
    )
    return consolidated


def _follow_up_matches_unresolved(
    question: str,
    unresolved_topics: list[QuestionDecisionItem],
    similarity_threshold: float,
) -> bool:
    normalized_question = normalize_text(question)
    if not normalized_question:
        return False

    for item in unresolved_topics:
        suggested = (item.suggested_follow_up or "").strip()
        if suggested and normalize_text(suggested) == normalized_question:
            return True
        if _topic_similarity(question, item.text) >= similarity_threshold:
            return True
        if suggested and _topic_similarity(question, suggested) >= similarity_threshold:
            return True
    return False


def _trim_and_consolidate_questions(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    similarity_threshold = float(prompt_config.get("question_topic_similarity_threshold", 0.72))
    max_open_questions = _optional_positive_int(prompt_config.get("max_open_questions"))
    max_question_decisions = _optional_positive_int(prompt_config.get("max_question_decisions"))
    max_follow_up_questions = _optional_positive_int(prompt_config.get("max_follow_up_questions"))

    consolidated_decisions = _consolidate_question_decisions(
        items=list(spec.question_decisions),
        similarity_threshold=similarity_threshold,
    )
    if max_question_decisions is not None and len(consolidated_decisions) > max_question_decisions:
        consolidated_decisions = consolidated_decisions[:max_question_decisions]

    unresolved_topics = [item for item in consolidated_decisions if item.decision == "needs_follow_up"]
    closed_topics = [item for item in consolidated_decisions if item.decision != "needs_follow_up"]

    kept_open_questions: list[NoteItem] = []
    for item in spec.open_questions:
        linked_decision = None
        for decision in consolidated_decisions:
            if _topic_similarity(item.text, decision.text) >= similarity_threshold:
                linked_decision = decision
                break
        if linked_decision is None or linked_decision.decision == "needs_follow_up":
            kept_open_questions.append(item)

    kept_open_questions = _merge_note_items(kept_open_questions)
    if max_open_questions is not None and len(kept_open_questions) > max_open_questions:
        kept_open_questions = kept_open_questions[:max_open_questions]

    follow_up_candidates: list[str] = []
    seen_follow_up: set[str] = set()
    for item in unresolved_topics:
        candidate = (item.suggested_follow_up or "").strip()
        if not candidate:
            candidate = _default_question_from_note(item.text)
        key = normalize_text(candidate)
        if not key or key in seen_follow_up:
            continue
        seen_follow_up.add(key)
        follow_up_candidates.append(candidate)

    if not follow_up_candidates:
        for question in spec.follow_up_questions:
            cleaned = question.strip()
            if not _follow_up_matches_unresolved(
                question=cleaned,
                unresolved_topics=unresolved_topics,
                similarity_threshold=similarity_threshold,
            ):
                continue

            key = normalize_text(cleaned)
            if not key or key in seen_follow_up:
                continue
            seen_follow_up.add(key)
            follow_up_candidates.append(cleaned)

    if max_follow_up_questions is not None and len(follow_up_candidates) > max_follow_up_questions:
        follow_up_candidates = follow_up_candidates[:max_follow_up_questions]

    ordered_decisions = unresolved_topics + closed_topics
    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=spec.functional_requirements,
        non_functional_requirements=spec.non_functional_requirements,
        constraints=spec.constraints,
        assumptions=spec.assumptions,
        open_questions=kept_open_questions,
        follow_up_questions=follow_up_candidates,
        question_decisions=ordered_decisions,
        utterances=spec.utterances,
    )


def _enforce_assumption_follow_up_policy(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    if not spec.assumptions:
        return spec

    similarity_threshold = float(prompt_config.get("question_topic_similarity_threshold", 0.72))
    utterance_dicts = [item.model_dump() for item in spec.utterances]

    remaining_decisions = list(spec.question_decisions)
    enforced_decisions: list[QuestionDecisionItem] = []
    mandatory_follow_ups: list[str] = []

    for assumption in spec.assumptions:
        if not assumption.evidence:
            continue

        matched: list[QuestionDecisionItem] = []
        unmatched: list[QuestionDecisionItem] = []
        for item in remaining_decisions:
            if _topic_similarity(item.text, assumption.text) >= similarity_threshold:
                matched.append(item)
            else:
                unmatched.append(item)
        remaining_decisions = unmatched

        status = _topic_follow_up_status(
            source_id=assumption.evidence[0],
            topic_text=assumption.text,
            utterances=utterance_dicts,
        )

        if status == "resolved":
            decision = "resolved"
            suggested = None
        else:
            decision = "needs_follow_up"
            suggested = None
            for item in matched:
                candidate = (item.suggested_follow_up or "").strip()
                if candidate:
                    suggested = candidate
                    break
            if not suggested:
                suggested = _default_question_from_note(assumption.text)
            mandatory_follow_ups.append(suggested)

        evidence_ids: set[str] = set(assumption.evidence)
        for item in matched:
            evidence_ids.update(item.evidence)

        enforced_decisions.append(
            QuestionDecisionItem(
                text=assumption.text,
                decision=decision,
                suggested_follow_up=suggested,
                evidence=_sort_utterance_ids(evidence_ids),
            )
        )

    combined_decisions = _merge_question_decisions(remaining_decisions + enforced_decisions)

    follow_up_questions = list(spec.follow_up_questions)
    seen_follow_up = {normalize_text(item) for item in follow_up_questions if normalize_text(item)}
    for question in mandatory_follow_ups:
        key = normalize_text(question)
        if not key or key in seen_follow_up:
            continue
        seen_follow_up.add(key)
        follow_up_questions.append(question)

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=spec.functional_requirements,
        non_functional_requirements=spec.non_functional_requirements,
        constraints=spec.constraints,
        assumptions=spec.assumptions,
        open_questions=spec.open_questions,
        follow_up_questions=follow_up_questions,
        question_decisions=combined_decisions,
        utterances=spec.utterances,
    )


def _align_follow_up_with_question_decisions(spec: SpecOutput) -> SpecOutput:
    if not spec.question_decisions:
        return spec

    questions: list[str] = []
    seen: set[str] = set()

    for item in spec.question_decisions:
        if item.decision != "needs_follow_up":
            continue

        candidate = (item.suggested_follow_up or "").strip()
        if not candidate:
            candidate = _default_question_from_note(item.text)
        key = normalize_text(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        questions.append(candidate)

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=spec.functional_requirements,
        non_functional_requirements=spec.non_functional_requirements,
        constraints=spec.constraints,
        assumptions=spec.assumptions,
        open_questions=spec.open_questions,
        follow_up_questions=questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _ensure_follow_up_questions(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    questions = list(spec.follow_up_questions)
    seen = {normalize_text(item) for item in questions if normalize_text(item)}

    if not bool(prompt_config.get("auto_follow_up_fallback", False)):
        return SpecOutput(
            project_summary=spec.project_summary,
            functional_requirements=spec.functional_requirements,
            non_functional_requirements=spec.non_functional_requirements,
            constraints=spec.constraints,
            assumptions=spec.assumptions,
            open_questions=spec.open_questions,
            follow_up_questions=questions,
            question_decisions=spec.question_decisions,
            utterances=spec.utterances,
        )

    if not questions:
        for open_item in spec.open_questions[:6]:
            candidate = _default_question_from_note(open_item.text)
            key = normalize_text(candidate)
            if key and key not in seen:
                seen.add(key)
                questions.append(candidate)

    if not questions and spec.assumptions:
        default_assumption_q = "Which assumptions must be confirmed before implementation starts?"
        questions.append(default_assumption_q)
        seen.add(normalize_text(default_assumption_q))

    if not questions:
        questions.append("What acceptance criteria define success for the first release?")

    return SpecOutput(
        project_summary=spec.project_summary,
        functional_requirements=spec.functional_requirements,
        non_functional_requirements=spec.non_functional_requirements,
        constraints=spec.constraints,
        assumptions=spec.assumptions,
        open_questions=spec.open_questions,
        follow_up_questions=questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _ensure_project_summary(spec: SpecOutput) -> SpecOutput:
    summary = spec.project_summary.strip()
    if summary and "merged chunk-level" not in summary.lower():
        return spec

    speakers = sorted({item.speaker for item in spec.utterances})
    summary = (
        f"Requirements draft from {len(spec.utterances)} utterances"
        f" ({', '.join(speakers)}) with {len(spec.functional_requirements)} functional requirements, "
        f"{len(spec.non_functional_requirements)} non-functional requirements, "
        f"{len(spec.constraints)} constraints, and {len(spec.open_questions)} open questions."
    )

    return SpecOutput(
        project_summary=summary,
        functional_requirements=spec.functional_requirements,
        non_functional_requirements=spec.non_functional_requirements,
        constraints=spec.constraints,
        assumptions=spec.assumptions,
        open_questions=spec.open_questions,
        follow_up_questions=spec.follow_up_questions,
        question_decisions=spec.question_decisions,
        utterances=spec.utterances,
    )


def _count_extracted_items(spec: SpecOutput) -> int:
    return (
        len(spec.functional_requirements)
        + len(spec.non_functional_requirements)
        + len(spec.constraints)
        + len(spec.assumptions)
        + len(spec.open_questions)
    )


def _normalize_category_label(label: str) -> str:
    cleaned = normalize_text(label)
    if not cleaned:
        return "none"

    alias_map = {
        "functional": "functional",
        "functional requirement": "functional",
        "fr": "functional",
        "non functional": "non_functional",
        "nonfunctional": "non_functional",
        "non functional requirement": "non_functional",
        "nfr": "non_functional",
        "constraint": "constraint",
        "assumption": "assumption",
        "open question": "open_question",
        "question": "open_question",
        "none": "none",
        "no": "none",
        "na": "none",
        "n a": "none",
        "null": "none",
    }
    if cleaned in alias_map:
        return alias_map[cleaned]

    if "non functional" in cleaned or "nonfunctional" in cleaned:
        return "non_functional"
    if "open question" in cleaned:
        return "open_question"
    if "constraint" in cleaned:
        return "constraint"
    if "assumption" in cleaned:
        return "assumption"
    if "functional" in cleaned:
        return "functional"
    if cleaned in {"no", "none"}:
        return "none"

    return "none"


def _infer_category_from_text(text: str) -> str:
    lowered = text.lower().strip()
    if not lowered:
        return "none"
    if "?" in lowered:
        return "open_question"
    if any(token in lowered for token in UNCERTAINTY_HINTS):
        return "assumption"
    if any(token in lowered for token in CONSTRAINT_HINTS):
        return "constraint"
    if any(token in lowered for token in QUALITY_AMBIGUOUS_HINTS):
        return "non_functional"
    if any(token in lowered for token in FUNCTIONAL_HINTS):
        return "functional"
    return "none"


def _build_summary_from_utterances(utterance_dicts: list[dict[str, str]], spec: SpecOutput) -> str:
    speakers = sorted({item.get("speaker", "Unknown") for item in utterance_dicts})
    return (
        f"Requirements draft from {len(utterance_dicts)} utterances"
        f" ({', '.join(speakers)}) with {len(spec.functional_requirements)} functional requirements, "
        f"{len(spec.non_functional_requirements)} non-functional requirements, "
        f"{len(spec.constraints)} constraints, and {len(spec.open_questions)} open questions."
    )


def _rescue_with_utterance_classification(
    utterance_dicts: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
    diagnostics: PipelineDiagnostics,
) -> tuple[SpecOutput, list[GenerationResult]]:
    _progress_log(prompt_config, "Empty extraction detected. Starting utterance-level rescue pass.")
    rescue_generations: list[GenerationResult] = []

    fr_items: list[RequirementItem] = []
    nfr_items: list[RequirementItem] = []
    constraints: list[NoteItem] = []
    assumptions: list[NoteItem] = []
    open_questions: list[NoteItem] = []
    follow_up_questions: list[str] = []
    question_decisions: list[QuestionDecisionItem] = []

    for idx, utterance in enumerate(utterance_dicts, start=1):
        generation = None
        rescue_prompt_config = dict(prompt_config)
        rescue_prompt_config["stage"] = "utterance_rescue"
        rescue_prompt_config["preferred_output_format"] = "yaml"
        rescue_prompt_config["llm_retry_on_invalid"] = 0

        try:
            generation = runner.generate_with_metadata(
                utterances=[utterance],
                prompt_config=rescue_prompt_config,
            )
            rescue_generations.append(generation)

            payload = parse_json_payload(generation.raw_text)
            category = _normalize_category_label(str(payload.get("category", "none")))
            text = str(payload.get("text", "")).strip()
            priority = _normalize_priority(payload.get("priority"))
            follow_up = str(payload.get("follow_up_question", "")).strip()
            parse_ok = True

            diagnostics.attempt_logs.append(
                {
                    "stage": "utterance_rescue",
                    "utterance_id": utterance.get("id"),
                    "attempt_index": idx,
                    "model_name": generation.model_name,
                    "latency_sec": generation.latency_sec,
                    "json_parse_ok": parse_ok,
                    "pydantic_validation_ok": True,
                    "raw_output": generation.raw_text,
                    "raw_output_preview": _shorten_for_log(generation.raw_text, 300),
                }
            )

            evidence = [utterance["id"]]
            fallback_text = utterance.get("text", "").strip()
            if category == "none":
                category = _normalize_category_label(generation.raw_text)
            if category == "none":
                category = _infer_category_from_text(fallback_text)
            inferred_category = _infer_category_from_text(fallback_text)
            if category == "non_functional" and inferred_category == "functional":
                category = "functional"
            elif category == "functional" and inferred_category == "non_functional":
                category = "non_functional"
            if _is_support_only_utterance(utterance) and category in {
                "functional",
                "non_functional",
                "constraint",
                "assumption",
            }:
                category = "open_question" if _is_question_like_text(fallback_text) else "none"

            if category == "functional":
                fr_items.append(
                    RequirementItem(
                        id=f"FR{len(fr_items) + 1}",
                        text=text or fallback_text,
                        priority=priority or "medium",
                        evidence=evidence,
                    )
                )
            elif category == "non_functional":
                nfr_items.append(
                    RequirementItem(
                        id=f"NFR{len(nfr_items) + 1}",
                        text=text or fallback_text,
                        priority=priority or "medium",
                        evidence=evidence,
                    )
                )
            elif category == "constraint":
                constraints.append(NoteItem(text=text or fallback_text, evidence=evidence))
            elif category == "assumption":
                assumptions.append(NoteItem(text=text or fallback_text, evidence=evidence))
            elif category == "open_question":
                open_questions.append(NoteItem(text=text or fallback_text, evidence=evidence))
            elif category == "none":
                if "?" in fallback_text:
                    open_questions.append(NoteItem(text=fallback_text, evidence=evidence))

            if category in {"assumption", "open_question"}:
                source_text = text or fallback_text
                pm_like = str(utterance.get("speaker", "")).strip().lower() in PM_SPEAKER_ALIASES
                if category == "open_question" and pm_like and "?" in fallback_text:
                    question_decisions.append(
                        QuestionDecisionItem(
                            text=source_text,
                            decision="already_asked",
                            suggested_follow_up=None,
                            evidence=evidence,
                        )
                    )
                else:
                    question_decisions.append(
                        QuestionDecisionItem(
                            text=source_text,
                            decision="needs_follow_up",
                            suggested_follow_up=follow_up or _default_question_from_note(source_text),
                            evidence=evidence,
                        )
                    )

            if follow_up:
                follow_up_questions.append(follow_up)

        except Exception as exc:  # noqa: BLE001
            # Small models sometimes return a single category token (e.g., \"non_functional\" or \"no\").
            raw_text = generation.raw_text if generation is not None else ""
            fallback_text = utterance.get("text", "").strip()
            category = _normalize_category_label(raw_text)
            if category == "none":
                category = _infer_category_from_text(fallback_text)
            inferred_category = _infer_category_from_text(fallback_text)
            if category == "non_functional" and inferred_category == "functional":
                category = "functional"
            elif category == "functional" and inferred_category == "non_functional":
                category = "non_functional"
            if _is_support_only_utterance(utterance) and category in {
                "functional",
                "non_functional",
                "constraint",
                "assumption",
            }:
                category = "open_question" if _is_question_like_text(fallback_text) else "none"

            diagnostics.attempt_logs.append(
                {
                    "stage": "utterance_rescue",
                    "utterance_id": utterance.get("id"),
                    "attempt_index": idx,
                    "model_name": generation.model_name if generation is not None else runner.runner_name,
                    "latency_sec": generation.latency_sec if generation is not None else None,
                    "json_parse_ok": False,
                    "pydantic_validation_ok": False,
                    "extraction_error": str(exc),
                    "raw_output": raw_text,
                    "raw_output_preview": _shorten_for_log(raw_text, 300),
                }
            )

            evidence = [utterance["id"]]
            if category == "functional":
                fr_items.append(
                    RequirementItem(
                        id=f"FR{len(fr_items) + 1}",
                        text=fallback_text,
                        priority="medium",
                        evidence=evidence,
                    )
                )
            elif category == "non_functional":
                nfr_items.append(
                    RequirementItem(
                        id=f"NFR{len(nfr_items) + 1}",
                        text=fallback_text,
                        priority="medium",
                        evidence=evidence,
                    )
                )
            elif category == "constraint":
                constraints.append(NoteItem(text=fallback_text, evidence=evidence))
            elif category == "assumption":
                assumptions.append(NoteItem(text=fallback_text, evidence=evidence))
            elif category == "open_question":
                open_questions.append(NoteItem(text=fallback_text, evidence=evidence))
            else:
                diagnostics.warnings.append(
                    f"Utterance rescue failed for {utterance.get('id', 'unknown')}: {exc}"
                )

            if category in {"assumption", "open_question"}:
                pm_like = str(utterance.get("speaker", "")).strip().lower() in PM_SPEAKER_ALIASES
                if category == "open_question" and pm_like and "?" in fallback_text:
                    question_decisions.append(
                        QuestionDecisionItem(
                            text=fallback_text,
                            decision="already_asked",
                            suggested_follow_up=None,
                            evidence=evidence,
                        )
                    )
                else:
                    question_decisions.append(
                        QuestionDecisionItem(
                            text=fallback_text,
                            decision="needs_follow_up",
                            suggested_follow_up=_default_question_from_note(fallback_text),
                            evidence=evidence,
                        )
                    )

    rescue_spec = SpecOutput(
        project_summary="Rescue-pass requirements draft.",
        functional_requirements=_merge_requirement_items(fr_items, prefix="FR"),
        non_functional_requirements=_merge_requirement_items(nfr_items, prefix="NFR"),
        constraints=_merge_note_items(constraints),
        assumptions=_merge_note_items(assumptions),
        open_questions=_merge_note_items(open_questions),
        follow_up_questions=follow_up_questions,
        question_decisions=_merge_question_decisions(question_decisions),
        utterances=utterance_dicts,
    )
    rescue_spec = _postprocess_spec(rescue_spec, prompt_config=prompt_config)
    rescue_spec = SpecOutput(
        project_summary=_build_summary_from_utterances(utterance_dicts, rescue_spec),
        functional_requirements=rescue_spec.functional_requirements,
        non_functional_requirements=rescue_spec.non_functional_requirements,
        constraints=rescue_spec.constraints,
        assumptions=rescue_spec.assumptions,
        open_questions=rescue_spec.open_questions,
        follow_up_questions=rescue_spec.follow_up_questions,
        question_decisions=rescue_spec.question_decisions,
        utterances=rescue_spec.utterances,
    )
    return rescue_spec, rescue_generations


def _postprocess_spec(spec: SpecOutput, prompt_config: dict) -> SpecOutput:
    if bool(prompt_config.get("llm_centric_mode", False)):
        processed = _clean_evidence(spec)
        merged = _merge_specs([processed], utterances=[item.model_dump() for item in processed.utterances])
        if bool(prompt_config.get("llm_centric_align_follow_up", False)):
            merged = _align_follow_up_with_question_decisions(merged)
        if bool(prompt_config.get("llm_centric_trim_questions", False)):
            merged = _trim_and_consolidate_questions(merged, prompt_config=prompt_config)
        if bool(prompt_config.get("llm_centric_ensure_summary", True)):
            merged = _ensure_project_summary(merged)
        return merged

    ambiguous_phrases = [str(item).lower() for item in prompt_config.get("ambiguous_phrases", [])]

    processed = _clean_evidence(spec)
    if bool(prompt_config.get("enable_sanity_postprocess", True)):
        processed = _apply_category_sanity_guards(processed)
    if bool(prompt_config.get("enable_rule_postprocess", False)):
        processed = _reclassify_ambiguous_items(processed, ambiguous_phrases=ambiguous_phrases)
        processed = _apply_hallucination_guard(processed, prompt_config=prompt_config)
    processed = _enforce_requester_source_only(processed, prompt_config=prompt_config)

    if not processed.question_decisions:
        processed = SpecOutput(
            project_summary=processed.project_summary,
            functional_requirements=processed.functional_requirements,
            non_functional_requirements=processed.non_functional_requirements,
            constraints=processed.constraints,
            assumptions=processed.assumptions,
            open_questions=processed.open_questions,
            follow_up_questions=processed.follow_up_questions,
            question_decisions=_build_question_decisions_from_notes(processed),
            utterances=processed.utterances,
        )
    processed = _enforce_assumption_follow_up_policy(processed, prompt_config=prompt_config)

    merged = _merge_specs([processed], utterances=[item.model_dump() for item in processed.utterances])
    if bool(prompt_config.get("enforce_llm_question_decisions", True)):
        merged = _align_follow_up_with_question_decisions(merged)
    merged = _ensure_follow_up_questions(merged, prompt_config=prompt_config)
    merged = _trim_and_consolidate_questions(merged, prompt_config=prompt_config)
    merged = _ensure_project_summary(merged)
    return merged


def _extract_evidence_ids_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.upper() for token in re.findall(r"\bU\d+\b", value, flags=re.IGNORECASE)]
    if isinstance(value, list):
        found: list[str] = []
        for item in value:
            found.extend(_extract_evidence_ids_from_value(item))
        dedup: list[str] = []
        for token in found:
            if token not in dedup:
                dedup.append(token)
        return dedup
    if isinstance(value, dict):
        found: list[str] = []
        for item in value.values():
            found.extend(_extract_evidence_ids_from_value(item))
        dedup: list[str] = []
        for token in found:
            if token not in dedup:
                dedup.append(token)
        return dedup
    return []


def _infer_best_evidence_id(text: str, utterances: list[dict[str, str]]) -> str | None:
    text_tokens = _text_token_set(text)
    if not text_tokens and utterances:
        return str(utterances[0].get("id", "")).strip() or None

    best_id: str | None = None
    best_overlap = -1
    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip()
        if not utterance_id:
            continue
        utterance_tokens = _text_token_set(str(utterance.get("text", "")))
        overlap = len(text_tokens & utterance_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = utterance_id
    return best_id


def _normalize_role_label(value: Any) -> str:
    lowered = normalize_text(str(value))
    if lowered in {"requester", "client", "customer", "user", "stakeholder", "owner"}:
        return "Client"
    if lowered in {"developer", "pm", "project manager", "engineer", "team lead", "analyst"}:
        return "PM"
    return "Unknown"


def _role_map_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    roles = payload.get("roles", [])
    mapping: dict[str, str] = {}

    if isinstance(roles, dict):
        for key, value in roles.items():
            utterance_id = str(key).strip().upper()
            if re.match(r"^U\d+$", utterance_id):
                mapping[utterance_id] = _normalize_role_label(value)
        return mapping

    if not isinstance(roles, list):
        return mapping

    for item in roles:
        if not isinstance(item, dict):
            continue
        utterance_id = str(item.get("id", "")).strip().upper()
        if not re.match(r"^U\d+$", utterance_id):
            continue
        mapping[utterance_id] = _normalize_role_label(item.get("role", "unknown"))
    return mapping


def _apply_role_inference(
    utterances: list[dict[str, str]],
    role_map: dict[str, str],
) -> list[dict[str, str]]:
    if not role_map:
        return utterances
    output: list[dict[str, str]] = []
    for utterance in utterances:
        utterance_id = str(utterance.get("id", "")).strip().upper()
        mapped = role_map.get(utterance_id)
        if mapped:
            output.append({**utterance, "speaker": mapped})
        else:
            output.append(utterance)
    return output


def _normalize_candidate_items(
    payload: dict[str, Any],
    utterances: list[dict[str, str]],
) -> list[dict[str, Any]]:
    raw_candidates = payload.get("candidates", [])
    if isinstance(raw_candidates, dict):
        raw_candidates = list(raw_candidates.values())
    if not isinstance(raw_candidates, list):
        raw_candidates = [raw_candidates]

    valid_ids = {str(item.get("id", "")).strip() for item in utterances}
    utterance_map = {str(item.get("id", "")).strip(): item for item in utterances}

    normalized: list[dict[str, Any]] = []
    next_index = 1
    for item in raw_candidates:
        if isinstance(item, str):
            text = item.strip()
            evidence = []
            ambiguity = "clear"
        elif isinstance(item, dict):
            text = str(item.get("text") or item.get("candidate") or item.get("requirement") or "").strip()
            evidence = _extract_evidence_ids_from_value(item.get("evidence"))
            if not evidence:
                evidence = _extract_evidence_ids_from_value(item)
            ambiguity = str(item.get("ambiguity", "clear")).strip().lower()
        else:
            text = str(item).strip()
            evidence = []
            ambiguity = "clear"

        if not text:
            continue

        evidence = [token for token in evidence if token in valid_ids]
        if not evidence:
            inferred = _infer_best_evidence_id(text=text, utterances=utterances)
            if inferred:
                evidence = [inferred]
        if not evidence:
            continue

        # Keep requester-side sources as primary candidate evidence.
        requester_evidence = [
            token
            for token in evidence
            if not _is_support_only_utterance(utterance_map.get(token))
        ]
        if not requester_evidence:
            continue
        evidence = requester_evidence

        normalized.append(
            {
                "id": f"C{next_index}",
                "text": text,
                "evidence": _sort_utterance_ids(set(evidence)),
                "ambiguity": "ambiguous" if "ambig" in ambiguity or "uncertain" in ambiguity else "clear",
            }
        )
        next_index += 1
    return normalized


def _map_classifier_label_to_category(label: str, text: str) -> str:
    normalized = _normalize_category_label(label)
    if normalized in {"functional", "non_functional", "constraint", "assumption", "open_question"}:
        return normalized
    inferred = _infer_category_from_text(text)
    if inferred in {"functional", "non_functional", "constraint", "assumption", "open_question"}:
        return inferred
    return "open_question"


def _build_hybrid_candidate_payload(
    utterances: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    classified: list[dict[str, Any]],
) -> dict[str, Any]:
    draft: dict[str, Any] = {
        "project_summary": "Candidate-based draft before final LLM rendering.",
        "functional_requirements": [],
        "non_functional_requirements": [],
        "constraints": [],
        "assumptions": [],
        "open_questions": [],
        "follow_up_questions": [],
        "question_decisions": [],
        "classified_candidates": classified,
    }

    fr_index = 1
    nfr_index = 1
    for item in classified:
        text = str(item.get("text", "")).strip()
        evidence = list(item.get("evidence", []))
        category = str(item.get("category", "")).strip()
        if not text or not evidence:
            continue

        if category == "functional":
            draft["functional_requirements"].append(
                {"id": f"FR{fr_index}", "text": text, "priority": "medium", "evidence": evidence}
            )
            fr_index += 1
        elif category == "non_functional":
            draft["non_functional_requirements"].append(
                {"id": f"NFR{nfr_index}", "text": text, "priority": "medium", "evidence": evidence}
            )
            nfr_index += 1
        elif category == "constraint":
            draft["constraints"].append({"text": text, "evidence": evidence})
        elif category == "assumption":
            draft["assumptions"].append({"text": text, "evidence": evidence})
            draft["question_decisions"].append(
                {
                    "text": text,
                    "decision": "needs_follow_up",
                    "suggested_follow_up": _default_question_from_note(text),
                    "evidence": evidence,
                }
            )
        elif category == "open_question":
            draft["open_questions"].append({"text": text, "evidence": evidence})
            draft["question_decisions"].append(
                {
                    "text": text,
                    "decision": "needs_follow_up",
                    "suggested_follow_up": _default_question_from_note(text),
                    "evidence": evidence,
                }
            )

    if not draft["follow_up_questions"]:
        for decision in draft["question_decisions"]:
            if decision.get("decision") != "needs_follow_up":
                continue
            question = str(decision.get("suggested_follow_up", "")).strip()
            if question:
                draft["follow_up_questions"].append(question)

    if not draft["project_summary"]:
        draft["project_summary"] = _build_summary_from_utterances(
            utterance_dicts=utterances,
            spec=SpecOutput(
                project_summary="temp",
                functional_requirements=[],
                non_functional_requirements=[],
                constraints=[],
                assumptions=[],
                open_questions=[],
                follow_up_questions=[],
                question_decisions=[],
                utterances=utterances,
            ),
        )

    return draft


def _call_model_payload(
    utterances: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
    stage_label: str,
) -> PayloadCallOutcome:
    attempts: list[dict[str, Any]] = []
    retries_left = int(prompt_config.get("llm_retry_on_invalid", 1))
    retries_used = 0
    generation = GenerationResult(model_name=runner.runner_name, raw_text="", latency_sec=0.0)
    payload: dict[str, Any] | None = None
    last_error: str | None = None

    while True:
        repair_mode = retries_used > 0
        run_config = dict(prompt_config)
        if repair_mode:
            run_config["repair_mode"] = True
            if retries_used >= 2:
                run_config["preferred_output_format"] = "yaml"

        try:
            step_generation = runner.generate_with_metadata(utterances=utterances, prompt_config=run_config)
            generation = GenerationResult(
                model_name=step_generation.model_name,
                raw_text=step_generation.raw_text,
                latency_sec=generation.latency_sec + step_generation.latency_sec,
                prompt_tokens=_sum_optional_int([generation.prompt_tokens, step_generation.prompt_tokens]),
                completion_tokens=_sum_optional_int([generation.completion_tokens, step_generation.completion_tokens]),
            )
            payload = parse_json_payload(step_generation.raw_text)
            attempts.append(
                {
                    "stage": stage_label,
                    "attempt_index": retries_used + 1,
                    "repair_mode": repair_mode,
                    "model_name": step_generation.model_name,
                    "latency_sec": step_generation.latency_sec,
                    "json_parse_ok": True,
                    "raw_output": step_generation.raw_text,
                    "raw_output_preview": _shorten_for_log(step_generation.raw_text, 300),
                }
            )
            return PayloadCallOutcome(
                payload=payload,
                generation=generation,
                error=None,
                retries_used=retries_used,
                attempt_logs=attempts,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            attempts.append(
                {
                    "stage": stage_label,
                    "attempt_index": retries_used + 1,
                    "repair_mode": repair_mode,
                    "model_name": generation.model_name,
                    "latency_sec": generation.latency_sec,
                    "json_parse_ok": False,
                    "extraction_error": last_error,
                    "raw_output": generation.raw_text,
                    "raw_output_preview": _shorten_for_log(generation.raw_text, 300),
                }
            )
            if retries_left <= 0 or isinstance(runner, MockModelRunner):
                break
            retries_left -= 1
            retries_used += 1

    return PayloadCallOutcome(
        payload=None,
        generation=generation,
        error=last_error or "payload parse failed",
        retries_used=retries_used,
        attempt_logs=attempts,
    )


def _generate_spec_hybrid(
    utterance_dicts: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
) -> tuple[SpecOutput, GenerationResult, PipelineDiagnostics]:
    diagnostics = PipelineDiagnostics()
    _progress_log(prompt_config, f"Hybrid pipeline start: {len(utterance_dicts)} utterances received.")
    generations: list[GenerationResult] = []

    # 1) LLM role inference
    role_prompt = dict(prompt_config)
    role_prompt["stage"] = "hybrid_role_inference"
    role_outcome = _call_model_payload(
        utterances=utterance_dicts,
        runner=runner,
        prompt_config=role_prompt,
        stage_label="hybrid role inference",
    )
    diagnostics.retries_used += role_outcome.retries_used
    diagnostics.attempt_logs.extend(role_outcome.attempt_logs)
    generations.append(role_outcome.generation)

    role_map = _role_map_from_payload(role_outcome.payload or {})
    role_adjusted_utterances = _apply_role_inference(utterances=utterance_dicts, role_map=role_map)
    _progress_log(
        prompt_config,
        f"Hybrid role inference complete: mapped {len(role_map)} utterance role(s).",
    )

    # 2) LLM candidate extraction
    candidate_prompt = dict(prompt_config)
    candidate_prompt["stage"] = "hybrid_candidate_extraction"
    candidate_outcome = _call_model_payload(
        utterances=role_adjusted_utterances,
        runner=runner,
        prompt_config=candidate_prompt,
        stage_label="hybrid candidate extraction",
    )
    diagnostics.retries_used += candidate_outcome.retries_used
    diagnostics.attempt_logs.extend(candidate_outcome.attempt_logs)
    generations.append(candidate_outcome.generation)

    if candidate_outcome.payload is None:
        diagnostics.errors.append(
            f"Hybrid candidate extraction failed: {candidate_outcome.error}"
        )
        raise PipelineExecutionError(
            f"Hybrid candidate extraction failed: {candidate_outcome.error}",
            diagnostics=diagnostics,
            generation=_merge_generation_results(generations),
        )

    candidates = _normalize_candidate_items(
        payload=candidate_outcome.payload,
        utterances=role_adjusted_utterances,
    )
    _progress_log(prompt_config, f"Hybrid candidate extraction produced {len(candidates)} candidate(s).")
    if not candidates and bool(prompt_config.get("hybrid_enable_empty_rescue", True)):
        _progress_log(prompt_config, "Hybrid candidate list is empty; starting utterance-level rescue.")
        rescue_spec, rescue_generations = _rescue_with_utterance_classification(
            utterance_dicts=role_adjusted_utterances,
            runner=runner,
            prompt_config=prompt_config,
            diagnostics=diagnostics,
        )
        generations.extend(rescue_generations)
        if _count_extracted_items(rescue_spec) > 0 or rescue_spec.follow_up_questions:
            diagnostics.warnings.append("Hybrid candidate extraction returned empty; recovered via utterance-level rescue.")
            generation = _merge_generation_results(generations)
            _progress_log(prompt_config, f"Hybrid rescue complete (total_latency={generation.latency_sec:.2f}s).")
            return rescue_spec, generation, diagnostics
        diagnostics.errors.append("Hybrid candidate extraction returned empty and rescue produced no items.")
        raise PipelineExecutionError(
            "Hybrid candidate extraction returned empty and rescue produced no items.",
            diagnostics=diagnostics,
            generation=_merge_generation_results(generations),
        )

    # 3) Classification model
    classifier = create_classification_runner(
        prompt_config=prompt_config,
        use_mock=bool(prompt_config.get("classification_use_mock", False) or isinstance(runner, MockModelRunner)),
    )
    classified: list[dict[str, Any]] = []
    for candidate in candidates:
        text = str(candidate.get("text", "")).strip()
        if not text:
            continue
        result = classifier.classify(text=text, labels=HYBRID_CLASS_LABELS)
        category = _map_classifier_label_to_category(result.label, text=text)
        classified.append(
            {
                **candidate,
                "category": category,
                "classification_label": result.label,
                "classification_score": round(float(result.score), 6),
            }
        )
    _progress_log(prompt_config, f"Hybrid classification complete: {len(classified)} classified candidate(s).")

    candidate_payload = _build_hybrid_candidate_payload(
        utterances=role_adjusted_utterances,
        candidates=candidates,
        classified=classified,
    )

    # 4) LLM final spec rendering
    render_prompt = dict(prompt_config)
    render_prompt["stage"] = "hybrid_spec_render"
    render_prompt["candidate_payload"] = candidate_payload
    render_outcome = _call_model_extract(
        utterances=role_adjusted_utterances,
        runner=runner,
        prompt_config=render_prompt,
        stage_label="hybrid spec render",
    )
    diagnostics.retries_used += render_outcome.retries_used
    diagnostics.attempt_logs.extend(render_outcome.attempt_logs)
    generations.append(render_outcome.generation)

    diagnostics.json_parse_ok = diagnostics.json_parse_ok and render_outcome.extraction.json_parse_ok
    diagnostics.pydantic_validation_ok = diagnostics.pydantic_validation_ok and render_outcome.extraction.pydantic_validation_ok
    diagnostics.first_pass_json_parse_ok = (
        diagnostics.first_pass_json_parse_ok and render_outcome.first_extraction.json_parse_ok
    )
    diagnostics.first_pass_pydantic_validation_ok = (
        diagnostics.first_pass_pydantic_validation_ok and render_outcome.first_extraction.pydantic_validation_ok
    )

    if render_outcome.spec is None:
        diagnostics.errors.append(
            f"Hybrid spec rendering failed: {render_outcome.extraction.error}"
        )
        raise PipelineExecutionError(
            f"Hybrid spec rendering failed: {render_outcome.extraction.error}",
            diagnostics=diagnostics,
            generation=_merge_generation_results(generations),
        )

    merged = _merge_specs([render_outcome.spec], utterances=role_adjusted_utterances)
    merged = _postprocess_spec(merged, prompt_config=prompt_config)
    if _count_extracted_items(merged) == 0 and not merged.follow_up_questions and bool(
        prompt_config.get("hybrid_enable_empty_rescue", True)
    ):
        _progress_log(prompt_config, "Hybrid rendered empty spec; starting utterance-level rescue.")
        rescue_spec, rescue_generations = _rescue_with_utterance_classification(
            utterance_dicts=role_adjusted_utterances,
            runner=runner,
            prompt_config=prompt_config,
            diagnostics=diagnostics,
        )
        generations.extend(rescue_generations)
        if _count_extracted_items(rescue_spec) > 0 or rescue_spec.follow_up_questions:
            diagnostics.warnings.append("Hybrid render returned empty; recovered via utterance-level rescue.")
            generation = _merge_generation_results(generations)
            _progress_log(prompt_config, f"Hybrid rescue complete (total_latency={generation.latency_sec:.2f}s).")
            return rescue_spec, generation, diagnostics
        diagnostics.errors.append("Hybrid render returned empty and rescue produced no items.")
        raise PipelineExecutionError(
            "Hybrid render returned empty and rescue produced no items.",
            diagnostics=diagnostics,
            generation=_merge_generation_results(generations),
        )

    generation = _merge_generation_results(generations)
    _progress_log(prompt_config, f"Hybrid pipeline complete (total_latency={generation.latency_sec:.2f}s).")
    return merged, generation, diagnostics


def _call_model_extract(
    utterances: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
    stage_label: str,
) -> ModelCallOutcome:
    _progress_log(prompt_config, f"{stage_label}: model call started with {len(utterances)} utterances.")
    attempts: list[dict[str, Any]] = []

    try:
        generation = runner.generate_with_metadata(utterances=utterances, prompt_config=prompt_config)
        extraction = extract_spec_output(raw_text=generation.raw_text, utterances=utterances)
    except Exception as exc:  # noqa: BLE001
        generation = GenerationResult(
            model_name=runner.runner_name,
            raw_text="",
            latency_sec=0.0,
            prompt_tokens=None,
            completion_tokens=None,
        )
        extraction = ExtractionResult(
            spec=None,
            json_parse_ok=False,
            pydantic_validation_ok=False,
            error=f"Generation failed: {exc}",
        )

    first_extraction = extraction
    attempts.append(
        _attempt_log_entry(
            stage_label=stage_label,
            attempt_index=1,
            repair_mode=False,
            generation=generation,
            extraction=extraction,
        )
    )
    if extraction.spec is not None:
        _progress_log(prompt_config, f"{stage_label}: first pass succeeded.")

    retries_left = int(prompt_config.get("llm_retry_on_invalid", 1))
    retries_used = 0

    while extraction.spec is None and retries_left > 0 and not isinstance(runner, MockModelRunner):
        retries_left -= 1
        retries_used += 1

        _progress_log(
            prompt_config,
            f"{stage_label}: invalid output detected; retry {retries_used} with repair prompt.",
        )
        retry_prompt_config = dict(prompt_config)
        retry_prompt_config["repair_mode"] = True
        if retries_used >= 2:
            retry_prompt_config["preferred_output_format"] = "yaml"
            _progress_log(
                prompt_config,
                f"{stage_label}: switching retry {retries_used} output format to YAML for parse stability.",
            )
        try:
            retry_generation = runner.generate_with_metadata(
                utterances=utterances,
                prompt_config=retry_prompt_config,
            )
            generation = GenerationResult(
                model_name=generation.model_name,
                raw_text=retry_generation.raw_text,
                latency_sec=generation.latency_sec + retry_generation.latency_sec,
                prompt_tokens=_sum_optional_int([generation.prompt_tokens, retry_generation.prompt_tokens]),
                completion_tokens=_sum_optional_int([generation.completion_tokens, retry_generation.completion_tokens]),
            )
            extraction = extract_spec_output(raw_text=generation.raw_text, utterances=utterances)
        except Exception as exc:  # noqa: BLE001
            extraction = ExtractionResult(
                spec=None,
                json_parse_ok=False,
                pydantic_validation_ok=False,
                error=f"Generation failed during retry {retries_used}: {exc}",
            )

        attempts.append(
            _attempt_log_entry(
                stage_label=stage_label,
                attempt_index=retries_used + 1,
                repair_mode=True,
                generation=generation,
                extraction=extraction,
            )
        )
        if extraction.spec is not None:
            _progress_log(prompt_config, f"{stage_label}: retry {retries_used} succeeded.")

    if extraction.spec is None:
        _progress_log(
            prompt_config,
            f"{stage_label}: model output remained invalid after {retries_used} retries.",
        )

    return ModelCallOutcome(
        spec=extraction.spec,
        generation=generation,
        extraction=extraction,
        first_extraction=first_extraction,
        retries_used=retries_used,
        attempt_logs=attempts,
    )


def _maybe_refine_with_consolidation(
    merged_spec: SpecOutput,
    utterance_dicts: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
    diagnostics: PipelineDiagnostics,
) -> tuple[SpecOutput, list[GenerationResult]]:
    if isinstance(runner, MockModelRunner):
        return merged_spec, []

    if not bool(prompt_config.get("enable_final_refinement", True)):
        return merged_spec, []

    refinement_config = dict(prompt_config)
    refinement_config["stage"] = "consolidate"
    refinement_config["candidate_payload"] = merged_spec.model_dump(
        exclude={"utterances", "run_metadata"},
        exclude_none=True,
    )

    outcome = _call_model_extract(
        utterances=utterance_dicts,
        runner=runner,
        prompt_config=refinement_config,
        stage_label="final consolidation",
    )

    diagnostics.retries_used += outcome.retries_used
    diagnostics.attempt_logs.extend(outcome.attempt_logs)
    diagnostics.first_pass_json_parse_ok = diagnostics.first_pass_json_parse_ok and outcome.first_extraction.json_parse_ok
    diagnostics.first_pass_pydantic_validation_ok = (
        diagnostics.first_pass_pydantic_validation_ok and outcome.first_extraction.pydantic_validation_ok
    )
    diagnostics.json_parse_ok = diagnostics.json_parse_ok and outcome.extraction.json_parse_ok
    diagnostics.pydantic_validation_ok = diagnostics.pydantic_validation_ok and outcome.extraction.pydantic_validation_ok

    if outcome.spec is None:
        diagnostics.warnings.append(f"Final consolidation step failed: {outcome.extraction.error}")
        _progress_log(prompt_config, "Final consolidation failed; using merged chunk output.")
        return merged_spec, [outcome.generation]

    diagnostics.used_final_refinement = True
    _progress_log(prompt_config, "Final consolidation succeeded.")
    refined = _merge_specs(
        [merged_spec, outcome.spec],
        utterances=utterance_dicts,
    )
    refined = _postprocess_spec(refined, prompt_config=prompt_config)
    return refined, [outcome.generation]


def generate_spec_from_utterances(
    utterance_dicts: list[dict[str, str]],
    runner: BaseModelRunner,
    prompt_config: dict,
) -> tuple[SpecOutput, GenerationResult, PipelineDiagnostics]:
    if bool(prompt_config.get("hybrid_flow_mode", False)):
        return _generate_spec_hybrid(
            utterance_dicts=utterance_dicts,
            runner=runner,
            prompt_config=prompt_config,
        )

    diagnostics = PipelineDiagnostics()
    _progress_log(prompt_config, f"Pipeline start: {len(utterance_dicts)} utterances received.")

    base_budget = int(prompt_config.get("chunk_token_budget", 0))
    if base_budget <= 0:
        runner_budget = runner.get_input_token_budget(prompt_config=prompt_config)
        chunk_ratio = float(prompt_config.get("chunk_budget_ratio", 0.55))
        base_budget = max(96, int(runner_budget * chunk_ratio))
    _progress_log(prompt_config, f"Chunk token budget set to {base_budget}.")

    max_chunk_utterances = int(prompt_config.get("max_chunk_utterances", 10))
    enable_chunking = bool(prompt_config.get("enable_chunking", True))

    if enable_chunking:
        chunks = _chunk_utterances(
            utterances=utterance_dicts,
            token_budget=base_budget,
            max_chunk_utterances=max_chunk_utterances,
        )
    else:
        chunks = [utterance_dicts]

    diagnostics.chunks_used = len(chunks)
    _progress_log(prompt_config, f"Chunking completed: {len(chunks)} chunk(s).")

    chunk_specs: list[SpecOutput] = []
    generation_results: list[GenerationResult] = []

    for chunk_index, chunk in enumerate(chunks):
        chunk_prompt_config = dict(prompt_config)
        chunk_prompt_config["stage"] = "extract_chunk"
        chunk_prompt_config["chunk_index"] = chunk_index
        chunk_prompt_config["chunk_total"] = len(chunks)

        _progress_log(
            chunk_prompt_config,
            f"Processing chunk {chunk_index + 1}/{len(chunks)} ({len(chunk)} utterances).",
        )
        outcome = _call_model_extract(
            utterances=chunk,
            runner=runner,
            prompt_config=chunk_prompt_config,
            stage_label=f"chunk {chunk_index + 1}/{len(chunks)}",
        )

        generation_results.append(outcome.generation)
        diagnostics.retries_used += outcome.retries_used
        diagnostics.attempt_logs.extend(outcome.attempt_logs)

        diagnostics.first_pass_json_parse_ok = diagnostics.first_pass_json_parse_ok and outcome.first_extraction.json_parse_ok
        diagnostics.first_pass_pydantic_validation_ok = (
            diagnostics.first_pass_pydantic_validation_ok and outcome.first_extraction.pydantic_validation_ok
        )
        diagnostics.json_parse_ok = diagnostics.json_parse_ok and outcome.extraction.json_parse_ok
        diagnostics.pydantic_validation_ok = diagnostics.pydantic_validation_ok and outcome.extraction.pydantic_validation_ok

        if outcome.spec is None:
            diagnostics.errors.append(
                f"Chunk {chunk_index + 1}/{len(chunks)} failed: {outcome.extraction.error}"
            )
            _progress_log(
                chunk_prompt_config,
                f"Chunk {chunk_index + 1}/{len(chunks)} failed validation.",
            )
            continue

        chunk_specs.append(outcome.spec)
        _progress_log(
            chunk_prompt_config,
            f"Chunk {chunk_index + 1}/{len(chunks)} accepted.",
        )

    if not chunk_specs:
        if bool(prompt_config.get("enable_failure_rescue_on_total_failure", True)) and not isinstance(
            runner, MockModelRunner
        ):
            _progress_log(prompt_config, "All chunks failed; attempting utterance-level rescue.")
            rescue_spec, rescue_generations = _rescue_with_utterance_classification(
                utterance_dicts=utterance_dicts,
                runner=runner,
                prompt_config=prompt_config,
                diagnostics=diagnostics,
            )
            generation_results.extend(rescue_generations)
            if _count_extracted_items(rescue_spec) > 0 or rescue_spec.follow_up_questions:
                diagnostics.warnings.append("All chunk extractions failed; recovered via utterance-level rescue.")
                aggregate_generation = _merge_generation_results(generation_results)
                _progress_log(prompt_config, "Utterance-level rescue succeeded after total chunk failure.")
                return rescue_spec, aggregate_generation, diagnostics

        aggregate_generation = _merge_generation_results(generation_results)
        reason = diagnostics.errors[-1] if diagnostics.errors else "No valid chunk output was produced"
        _progress_log(prompt_config, "All chunks failed; aborting pipeline.")
        raise PipelineExecutionError(
            (
                "Failed to generate a valid spec output. "
                f"Details: {reason}."
            ),
            diagnostics=diagnostics,
            generation=aggregate_generation,
        )

    _progress_log(prompt_config, "Merging chunk-level specs.")
    merged_spec = _merge_specs(chunk_specs, utterances=utterance_dicts)
    merged_spec = _postprocess_spec(merged_spec, prompt_config=prompt_config)

    refined_spec, refinement_generations = _maybe_refine_with_consolidation(
        merged_spec=merged_spec,
        utterance_dicts=utterance_dicts,
        runner=runner,
        prompt_config=prompt_config,
        diagnostics=diagnostics,
    )
    generation_results.extend(refinement_generations)

    if (
        _count_extracted_items(refined_spec) == 0
        and not isinstance(runner, MockModelRunner)
        and bool(prompt_config.get("enable_empty_rescue", True))
    ):
        rescue_spec, rescue_generations = _rescue_with_utterance_classification(
            utterance_dicts=utterance_dicts,
            runner=runner,
            prompt_config=prompt_config,
            diagnostics=diagnostics,
        )
        generation_results.extend(rescue_generations)
        if _count_extracted_items(rescue_spec) > 0:
            _progress_log(prompt_config, "Utterance-level rescue produced non-empty extraction.")
            refined_spec = rescue_spec
        else:
            diagnostics.warnings.append("Utterance-level rescue did not recover any requirement items.")

    aggregate_generation = _merge_generation_results(generation_results)
    _progress_log(
        prompt_config,
        f"Pipeline complete (total_latency={aggregate_generation.latency_sec:.2f}s).",
    )
    return refined_spec, aggregate_generation, diagnostics


def _run_with_utterances(
    utterance_dicts: list[dict[str, str]],
    output_dir: str | Path,
    runner: BaseModelRunner,
    prompt_config: dict,
) -> PipelineResult:
    run_start = perf_counter()
    destination = ensure_directory(output_dir)
    _progress_log(prompt_config, f"Output directory: {destination}")
    for stale_log_name in ("error.log", "warning.log", "retry.log", "error_details.json"):
        stale_path = destination / stale_log_name
        if stale_path.exists():
            stale_path.unlink()

    try:
        spec, generation, diagnostics = generate_spec_from_utterances(
            utterance_dicts=utterance_dicts,
            runner=runner,
            prompt_config=prompt_config,
        )
    except PipelineExecutionError as exc:
        error_path = destination / "error.log"
        error_payload = {
            "reason": str(exc),
            "chunks_used": exc.diagnostics.chunks_used,
            "retries_used": exc.diagnostics.retries_used,
            "json_parse_ok": exc.diagnostics.json_parse_ok,
            "pydantic_validation_ok": exc.diagnostics.pydantic_validation_ok,
            "first_pass_json_parse_ok": exc.diagnostics.first_pass_json_parse_ok,
            "first_pass_pydantic_validation_ok": exc.diagnostics.first_pass_pydantic_validation_ok,
            "warnings": exc.diagnostics.warnings,
            "errors": exc.diagnostics.errors,
            "attempt_count": len(exc.diagnostics.attempt_logs),
            "attempt_logs": exc.diagnostics.attempt_logs,
            "aggregate_generation": (
                {
                    "model_name": exc.generation.model_name,
                    "latency_sec": exc.generation.latency_sec,
                    "prompt_tokens": exc.generation.prompt_tokens,
                    "completion_tokens": exc.generation.completion_tokens,
                    "raw_output_preview": _shorten_for_log(exc.generation.raw_text, max_len=500),
                }
                if exc.generation is not None
                else None
            ),
        }
        error_path.write_text(
            (
                "Conversation-to-Spec generation failed\n"
                f"Reason: {exc}\n\n"
                "Diagnostics (JSON):\n"
                f"{json.dumps(error_payload, ensure_ascii=False, indent=2)}\n"
            ),
            encoding="utf-8",
        )
        error_detail_path = destination / "error_details.json"
        error_detail_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"Failed to generate a valid spec output. Details recorded at {error_path}: {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        error_path = destination / "error.log"
        error_path.write_text(
            (
                "Conversation-to-Spec generation failed\n"
                f"Reason: {exc}\n"
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Failed to generate a valid spec output. Details recorded at {error_path}: {exc}"
        ) from exc

    if diagnostics.warnings:
        warning_path = destination / "warning.log"
        warning_path.write_text("\n".join(diagnostics.warnings) + "\n", encoding="utf-8")

    if diagnostics.retries_used > 0:
        retry_path = destination / "retry.log"
        retry_path.write_text(
            (
                "LLM retry succeeded after invalid output.\n"
                f"Model: {generation.model_name}\n"
                f"Retry count used: {diagnostics.retries_used}\n"
                f"Chunk count: {diagnostics.chunks_used}\n"
            ),
            encoding="utf-8",
        )

    json_path = destination / "spec.json"
    markdown_path = destination / "spec.md"

    total_execution_time_sec = perf_counter() - run_start
    spec = spec.model_copy(
        update={
            "run_metadata": RunMetadata(
                execution_time_sec=round(total_execution_time_sec, 6),
                model_latency_sec=round(generation.latency_sec, 6),
                model_name=generation.model_name,
                prompt_tokens=generation.prompt_tokens,
                completion_tokens=generation.completion_tokens,
                generated_at=datetime.now().isoformat(timespec="seconds"),
            )
        }
    )

    json_path.write_text(
        json.dumps(spec.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(spec_to_markdown(spec), encoding="utf-8")
    _progress_log(
        prompt_config,
        f"Saved outputs -> JSON: {json_path} | Markdown: {markdown_path}",
    )

    return PipelineResult(
        spec=spec,
        generation=generation,
        json_path=json_path,
        markdown_path=markdown_path,
        diagnostics=diagnostics,
    )


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path,
    runner: BaseModelRunner,
    prompt_config: dict,
) -> PipelineResult:
    utterances = parse_conversation_file(input_path)
    utterance_dicts = [utterance.model_dump() for utterance in utterances]
    return _run_with_utterances(
        utterance_dicts=utterance_dicts,
        output_dir=output_dir,
        runner=runner,
        prompt_config=prompt_config,
    )


def run_pipeline_from_text(
    conversation_text: str,
    output_dir: str | Path,
    runner: BaseModelRunner,
    prompt_config: dict,
) -> PipelineResult:
    utterances = parse_conversation_text(conversation_text)
    utterance_dicts = [utterance.model_dump() for utterance in utterances]
    return _run_with_utterances(
        utterance_dicts=utterance_dicts,
        output_dir=output_dir,
        runner=runner,
        prompt_config=prompt_config,
    )
