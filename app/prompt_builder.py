from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.schemas import ConversationUnit
from app.utils import load_yaml_file


PROMPT_CONFIG_PATH = Path("configs/prompts.yaml")


def load_prompt_config(path: Path = PROMPT_CONFIG_PATH) -> dict:
    return load_yaml_file(path)


def _unit_block(conversation_units: Iterable[ConversationUnit]) -> str:
    return "\n".join(f"{u.id} | {u.text}" for u in conversation_units)


def _dump_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _stage_text(prompt_config: dict, key: str, default_text: str) -> str:
    stage_prompts = prompt_config.get("stage_prompts", {})
    if isinstance(stage_prompts, dict):
        text = str(stage_prompts.get(key, "")).strip()
        if text:
            return text
    return default_text


def build_stage_1_candidate_extraction_prompt(
    conversation_units: Iterable[ConversationUnit], prompt_config: dict
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_1_candidate_extraction",
        (
            "Extract candidate ideas from conversation units with high recall. "
            "Do not finalize requirement categories yet, and include possible constraints."
        ),
    )
    schema_hint = {
        "candidates": [
            {
                "id": "C1",
                "kind": "possible_requirement",
                "text": "short candidate description",
                "source_units": ["U1"],
            },
            {
                "id": "C2",
                "kind": "possible_constraint",
                "text": "explicit boundary or release limitation",
                "source_units": ["U2"],
            }
        ]
    }
    return (
        "CHAIN_STAGE:1_CANDIDATE_EXTRACTION\n"
        "You are a requirements analysis assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Conversation may be unlabeled.\n"
        "- Be permissive and recall-oriented.\n"
        "- Do not invent unsupported content.\n"
        "- Return JSON only.\n"
        "- Every candidate must include source_units.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Output JSON only."
    )


def build_stage_2_candidate_classification_prompt(
    conversation_units: Iterable[ConversationUnit],
    candidates: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_2_candidate_classification",
        (
            "Classify candidates into functional/non-functional/constraint/open question/"
            "follow-up trigger/note/discard with semantic judgment."
        ),
    )
    schema_hint = {
        "classified_candidates": [
            {
                "id": "C1",
                "final_type": "functional_requirement",
                "reason": "brief explanation",
                "source_units": ["U1"],
            }
        ]
    }
    return (
        "CHAIN_STAGE:2_CANDIDATE_CLASSIFICATION\n"
        "You are a software requirements classification assistant.\n"
        f"{instructions}\n\n"
        "Allowed final_type values:\n"
        "- functional_requirement\n"
        "- non_functional_requirement\n"
        "- constraint\n"
        "- open_question\n"
        "- follow_up_trigger\n"
        "- note\n"
        "- discard\n\n"
        "Rules:\n"
        "- Do not force vague/future items into finalized requirements.\n"
        "- Use constraint for explicit boundaries, exclusions, and implementation limits.\n"
        "- Preserve source_units.\n"
        "- Return JSON only.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Candidates JSON:\n"
        f"{_dump_json({'candidates': candidates})}\n\n"
        "Output JSON only."
    )


def build_stage_3_requirement_rewriting_prompt(
    conversation_units: Iterable[ConversationUnit],
    classified_candidates: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_3_requirement_rewriting",
        (
            "Rewrite functional, non-functional, and constraint items into clear "
            "specification-style English without hallucination."
        ),
    )
    schema_hint = {
        "rewritten_items": [
            {
                "id": "R1",
                "type": "functional_requirement",
                "text": "The system shall ...",
                "source_units": ["U1"],
            },
            {
                "id": "R2",
                "type": "constraint",
                "text": "Online payment shall not be included in the initial release.",
                "source_units": ["U2"],
            }
        ]
    }
    return (
        "CHAIN_STAGE:3_REQUIREMENT_REWRITING\n"
        "You are a software specification drafting assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Rewrite only items classified as functional_requirement, non_functional_requirement, or constraint.\n"
        "- Preserve source_units exactly.\n"
        "- Keep wording concise and implementation-usable.\n"
        "- Do not invent unsupported details.\n"
        "- Return JSON only.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Classified candidates JSON:\n"
        f"{_dump_json({'classified_candidates': classified_candidates})}\n\n"
        "Output JSON only."
    )


def build_stage_4_open_question_generation_prompt(
    conversation_units: Iterable[ConversationUnit],
    classified_candidates: list[dict[str, Any]],
    rewritten_items: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_4_open_question_generation",
        (
            "Generate unresolved open questions grounded in ambiguities, unclear constraints, "
            "and missing requirement details."
        ),
    )
    schema_hint = {
        "open_questions": [
            {
                "text": "Could you clarify what 'clean and modern' means with concrete acceptance criteria?",
                "source_units": ["U3"],
            }
        ]
    }
    return (
        "CHAIN_STAGE:4_OPEN_QUESTION_GENERATION\n"
        "You are an open-question generation assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Generate only unresolved open questions.\n"
        "- Each question must be specific and end with '?'.\n"
        "- Tie each question to source_units.\n"
        "- Return JSON only.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Classified candidates JSON:\n"
        f"{_dump_json({'classified_candidates': classified_candidates})}\n\n"
        "Rewritten items JSON:\n"
        f"{_dump_json({'rewritten_items': rewritten_items})}\n\n"
        "Output JSON only."
    )


def build_stage_5_followup_generation_prompt(
    conversation_units: Iterable[ConversationUnit],
    classified_candidates: list[dict[str, Any]],
    rewritten_items: list[dict[str, Any]],
    open_questions: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_5_followup_generation",
        (
            "Generate actionable developer-facing follow-up questions that move the project "
            "toward implementation decisions."
        ),
    )
    schema_hint = {
        "follow_up_questions": [
            {
                "text": "What measurable acceptance criteria should define the first release?",
                "source_units": ["U2"],
            }
        ]
    }
    return (
        "CHAIN_STAGE:5_FOLLOWUP_GENERATION\n"
        "You are a developer-facing follow-up assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Questions must be actionable and specific.\n"
        "- Avoid generic filler questions.\n"
        "- Tie each question to source_units.\n"
        "- Return JSON only.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Classified candidates JSON:\n"
        f"{_dump_json({'classified_candidates': classified_candidates})}\n\n"
        "Rewritten items JSON:\n"
        f"{_dump_json({'rewritten_items': rewritten_items})}\n\n"
        "Open questions JSON:\n"
        f"{_dump_json({'open_questions': open_questions})}\n\n"
        "Output JSON only."
    )


def build_stage_6_summary_prompt(
    conversation_units: Iterable[ConversationUnit],
    rewritten_items: list[dict[str, Any]],
    open_questions: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    instructions = _stage_text(
        prompt_config,
        "stage_6_project_summary",
        (
            "Write a faithful 2-4 sentence project summary from the extracted requirements "
            "and unresolved points."
        ),
    )
    schema_hint = {"project_summary": "2-4 sentence summary"}
    return (
        "CHAIN_STAGE:6_PROJECT_SUMMARY\n"
        "You are a software project summarization assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Keep summary to 2-4 sentences.\n"
        "- Do not invent unsupported scope.\n"
        "- Mention future/optional scope only as future or uncertain.\n"
        "- Return JSON only.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Rewritten items JSON:\n"
        f"{_dump_json({'rewritten_items': rewritten_items})}\n\n"
        "Open questions JSON:\n"
        f"{_dump_json({'open_questions': open_questions})}\n\n"
        "Notes JSON:\n"
        f"{_dump_json({'notes': notes})}\n\n"
        "Output JSON only."
    )


def build_single_shot_spec_prompt(
    conversation_units: Iterable[ConversationUnit], prompt_config: dict
) -> str:
    instructions = _stage_text(
        prompt_config,
        "single_shot_spec_generation",
        (
            "Convert the conversation into a complete structured software specification draft. "
            "Classify functional requirements, non-functional requirements, constraints, "
            "open questions, follow-up questions, and notes in one pass."
        ),
    )
    schema_hint = {
        "project_summary": "2-4 sentence summary",
        "functional_requirements": [
            {
                "id": "FR1",
                "text": "The system shall ...",
                "source_units": ["U1"],
            }
        ],
        "non_functional_requirements": [
            {
                "id": "NFR1",
                "text": "The system should ...",
                "source_units": ["U2"],
            }
        ],
        "constraints": [
            {
                "id": "CON1",
                "text": "Initial-release boundary or implementation limitation.",
                "source_units": ["U3"],
            }
        ],
        "open_questions": [
            {
                "text": "Specific unresolved ambiguity?",
                "source_units": ["U4"],
            }
        ],
        "follow_up_questions": [
            {
                "text": "Specific developer question for the client?",
                "source_units": ["U4"],
            }
        ],
        "notes": [
            {
                "text": "Future or contextual note that is not a current hard requirement.",
                "source_units": ["U5"],
            }
        ],
        "verification_warnings": [],
    }
    return (
        "SINGLE_SHOT_SPEC_GENERATION\n"
        "You are a requirements analysis assistant.\n"
        f"{instructions}\n\n"
        "Rules:\n"
        "- Conversation may be unlabeled.\n"
        "- Do not invent unsupported content.\n"
        "- Do not convert vague or future-scope statements into hard requirements.\n"
        "- Use non_functional_requirements only for quality attributes such as performance, usability, security, reliability, accessibility, or responsiveness.\n"
        "- Every item except project_summary must include source_units grounded in the provided conversation unit IDs.\n"
        "- Return JSON only. Do not include markdown fences or explanation.\n\n"
        f"Required JSON schema:\n{_dump_json(schema_hint)}\n\n"
        "Conversation units:\n"
        f"{_unit_block(conversation_units)}\n\n"
        "Output JSON only."
    )


# Backward-compatible prompt helpers for old stage numbering.
def build_stage_4_followup_generation_prompt(
    conversation_units: Iterable[ConversationUnit],
    classified_candidates: list[dict[str, Any]],
    rewritten_items: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    return build_stage_5_followup_generation_prompt(
        conversation_units,
        classified_candidates,
        rewritten_items,
        [],
        prompt_config,
    )


def build_stage_5_summary_prompt(
    conversation_units: Iterable[ConversationUnit],
    rewritten_items: list[dict[str, Any]],
    open_questions: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    prompt_config: dict,
) -> str:
    return build_stage_6_summary_prompt(
        conversation_units,
        rewritten_items,
        open_questions,
        notes,
        prompt_config,
    )


def build_stage_retry_prompt(
    *,
    stage_name: str,
    error_message: str,
    previous_output: str,
    required_schema: dict[str, Any],
    original_context: str,
) -> str:
    trimmed_prev = previous_output.strip()
    if len(trimmed_prev) > 5000:
        trimmed_prev = trimmed_prev[:5000] + "\n...[truncated]"

    trimmed_context = original_context.strip()
    if len(trimmed_context) > 12000:
        trimmed_context = trimmed_context[:12000] + "\n...[truncated]"

    return (
        f"CHAIN_STAGE:{stage_name}_RETRY\n"
        "You are correcting a previously invalid structured output.\n"
        "Return corrected JSON only.\n"
        "Do not include markdown fences.\n"
        "Do not include explanation outside JSON.\n\n"
        f"Validation error:\n{error_message}\n\n"
        "Previous invalid output:\n"
        f"{trimmed_prev}\n\n"
        "Required schema:\n"
        f"{_dump_json(required_schema)}\n\n"
        "Original stage context:\n"
        f"{trimmed_context}\n\n"
        "Output corrected JSON only."
    )


# Backward-compatible prompt helpers kept for existing external callers.
def build_extraction_prompt(
    conversation_units: Iterable[ConversationUnit], prompt_config: dict
) -> str:
    return build_stage_1_candidate_extraction_prompt(conversation_units, prompt_config)


def build_retry_prompt(
    *,
    previous_output: str,
    error_message: str,
    conversation_units: Iterable[ConversationUnit],
    prompt_config: dict,
) -> str:
    stage_prompt = build_stage_1_candidate_extraction_prompt(conversation_units, prompt_config)
    schema_hint = {
        "candidates": [
            {
                "id": "C1",
                "kind": "possible_requirement",
                "text": "short candidate description",
                "source_units": ["U1"],
            }
        ]
    }
    return build_stage_retry_prompt(
        stage_name="1_CANDIDATE_EXTRACTION",
        error_message=error_message,
        previous_output=previous_output,
        required_schema=schema_hint,
        original_context=stage_prompt,
    )
