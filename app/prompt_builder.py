from __future__ import annotations

import json
import yaml

DEFAULT_AMBIGUOUS_PHRASES = [
    "simple",
    "easy to use",
    "clean",
    "fast",
    "modern",
    "maybe",
    "later",
    "nice to have",
    "user-friendly",
]

SCHEMA_HINT_CORE = {
    "project_summary": "<non-empty string>",
    "functional_requirements": [
        {
            "id": "FR1",
            "text": "<actionable requirement text>",
            "priority": "high|medium|low|null",
            "evidence": ["U1"],
        }
    ],
    "non_functional_requirements": [
        {
            "id": "NFR1",
            "text": "<quality requirement text>",
            "priority": "high|medium|low|null",
            "evidence": ["U2"],
        }
    ],
    "constraints": [{"text": "<constraint>", "evidence": ["U3"]}],
    "assumptions": [{"text": "<assumption>", "evidence": ["U4"]}],
    "open_questions": [{"text": "<open question>", "evidence": ["U5"]}],
    "follow_up_questions": ["<clarifying question>"],
    "question_decisions": [
        {
            "text": "<ambiguity/question topic>",
            "decision": "needs_follow_up|already_asked|resolved",
            "suggested_follow_up": "<required only when decision=needs_follow_up>",
            "evidence": ["U6"],
        }
    ],
}


def _render_utterances(utterances: list[dict]) -> str:
    return "\n".join(
        f"{utterance['id']} | {utterance['speaker']}: {utterance['text']}" for utterance in utterances
    )


def _render_strict_output_contract(output_format: str) -> str:
    if output_format == "yaml":
        return (
            "STRICT OUTPUT CONTRACT:\n"
            "1) Output must be exactly one YAML mapping object.\n"
            "2) First non-empty line must start with: project_summary:\n"
            "3) Do not output markdown fences (```), explanations, steps, or commentary.\n"
            "4) Allowed top-level keys only:\n"
            "   - project_summary\n"
            "   - functional_requirements\n"
            "   - non_functional_requirements\n"
            "   - constraints\n"
            "   - assumptions\n"
            "   - open_questions\n"
            "   - follow_up_questions\n"
            "   - question_decisions\n"
            "5) project_summary must be a plain string (not object/list).\n"
            "6) If uncertain, keep arrays empty instead of adding prose.\n"
        )

    return (
        "STRICT OUTPUT CONTRACT:\n"
        "1) Output must be exactly one JSON object.\n"
        "2) First character must be '{' and last character must be '}'.\n"
        "3) Do not output markdown fences (```), explanations, steps, or commentary.\n"
        "4) Allowed top-level keys only:\n"
        "   - project_summary\n"
        "   - functional_requirements\n"
        "   - non_functional_requirements\n"
        "   - constraints\n"
        "   - assumptions\n"
        "   - open_questions\n"
        "   - follow_up_questions\n"
        "   - question_decisions\n"
        "5) project_summary must be a plain string (not object/list).\n"
        "6) If uncertain, keep arrays empty instead of adding prose.\n"
    )


def _preferred_output_format(prompt_config: dict) -> str:
    output_format = str(prompt_config.get("preferred_output_format", "json")).strip().lower()
    if output_format not in {"json", "yaml"}:
        return "json"
    return output_format


def _use_compact_prompt(prompt_config: dict) -> bool:
    profile = str(prompt_config.get("prompt_profile", "")).strip().lower()
    if profile in {"compact", "small"}:
        return True

    max_input_tokens = prompt_config.get("max_input_tokens")
    if isinstance(max_input_tokens, int) and max_input_tokens > 0 and max_input_tokens <= 1024:
        return True
    return False


def _schema_hint_text(output_format: str, compact: bool = False) -> str:
    if compact:
        return (
            "Required keys: project_summary, functional_requirements, non_functional_requirements, "
            "constraints, assumptions, open_questions, follow_up_questions, question_decisions"
        )

    if output_format == "yaml":
        return yaml.safe_dump(SCHEMA_HINT_CORE, allow_unicode=True, sort_keys=False)

    return json.dumps(SCHEMA_HINT_CORE, ensure_ascii=False, indent=2)


def _build_common_rules(prompt_config: dict, output_format: str) -> list[str]:
    if output_format == "yaml":
        format_rule = "Return exactly one YAML mapping object. No markdown fences, no prose."
    else:
        format_rule = "Return exactly one JSON object. No markdown, no prose."

    rules = [
        format_rule,
        "Treat input as a dialogue between requester side (Client/Customer) and development side (PM/Developer).",
        "Only requester-side utterances can become functional_requirements, non_functional_requirements, constraints, assumptions, or open_questions.",
        "PM/Developer utterances are support-only: use them for clarification context and question_decisions, not as specification sources.",
        "Never invent unsupported requirements.",
        "Never output reasoning, analysis, chain-of-thought, or step-by-step explanation.",
        "Every item in functional_requirements, non_functional_requirements, constraints, assumptions, and open_questions must have at least one evidence utterance ID.",
        "Evidence IDs must reference existing utterances only (e.g., U1, U2).",
        "If information is vague, uncertain, or missing, move it to assumptions/open_questions instead of hard requirements.",
        "Ambiguous quality words (simple, easy, clean, fast, modern, user-friendly) should stay non-functional or open questions unless quantified.",
        "For every ambiguity/open question, add a question_decisions item with decision in {needs_follow_up, already_asked, resolved}.",
        "Use already_asked or resolved when PM/Developer already asked and the client already answered sufficiently.",
        "Include follow_up_questions only for question_decisions with decision=needs_follow_up.",
        "Avoid duplicates; keep each item atomic and actionable.",
        "Do not include an `utterances` key in output.",
        "Do not include any keys outside the schema.",
        "Do not copy schema placeholders (e.g., <...>) into output.",
    ]

    if bool(prompt_config.get("repair_mode", False)):
        rules.extend(
            [
                "Previous output was invalid. Fix and return one complete machine-parseable object.",
                "Do not repeat key fragments, partial arrays, or broken quotes.",
                "Do not prepend words like 'Output:', 'Here is', or any explanation before the object.",
            ]
        )
    return rules


def _build_extract_prompt(utterances: list[dict], prompt_config: dict) -> str:
    role_instruction = prompt_config.get(
        "role_instruction",
        (
            "You are a requirements analysis assistant for junior PMs and student team leads. "
            "Convert client conversations into a structured software requirements draft. "
            "주어진 입력은 개발 주체와 의뢰 주체의 대화이다. 의뢰 주체의 발화를 명세화하려고 한다."
        ),
    )
    ambiguous_phrases = prompt_config.get("ambiguous_phrases", DEFAULT_AMBIGUOUS_PHRASES)
    output_format = _preferred_output_format(prompt_config)

    chunk_index = prompt_config.get("chunk_index")
    chunk_total = prompt_config.get("chunk_total")
    chunk_tag = ""
    if isinstance(chunk_index, int) and isinstance(chunk_total, int):
        chunk_tag = f"\nChunk context: This is chunk {chunk_index + 1} of {chunk_total}."

    rules = _build_common_rules(prompt_config, output_format=output_format)
    rule_block = "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))
    output_contract = _render_strict_output_contract(output_format=output_format)

    if _use_compact_prompt(prompt_config):
        return (
            "You are a software requirements extractor.\n"
            "Return exactly one JSON object.\n"
            "No markdown, no explanation.\n"
            "This input is a dialogue between developer side and requester side.\n"
            "Only requester-side utterances can become FR/NFR/constraints/assumptions/open_questions.\n"
            "Treat PM/Developer utterances as support-only for clarification and question_decisions.\n"
            "Required keys:\n"
            "project_summary, functional_requirements, non_functional_requirements, constraints, "
            "assumptions, open_questions, follow_up_questions, question_decisions.\n"
            "Each requirement/note item must include evidence like [\"U1\"].\n"
            "question_decisions decision must be one of: needs_follow_up, already_asked, resolved.\n"
            "If PM/Developer already asked and client answered, use already_asked or resolved.\n"
            "Only add follow_up_questions for needs_follow_up decisions.\n"
            "Utterances:\n"
            f"{_render_utterances(utterances)}\n"
            "Output JSON only."
        )

    schema_text = _schema_hint_text(
        output_format=output_format,
        compact=bool(prompt_config.get("repair_mode", False)),
    )

    return_keyword = "YAML mapping" if output_format == "yaml" else "JSON object"

    prompt = (
        f"{role_instruction}\n\n"
        "Task:\n"
        "Extract structured requirements from the conversation chunk.\n"
        f"{chunk_tag}\n\n"
        f"{output_contract}\n"
        "Ambiguous phrases to treat cautiously:\n"
        f"- {', '.join(ambiguous_phrases)}\n\n"
        "Rules:\n"
        f"{rule_block}\n\n"
        "Output schema:\n"
        f"{schema_text}\n\n"
        "Conversation utterances:\n"
        f"{_render_utterances(utterances)}\n\n"
        f"Return only the {return_keyword} now. Nothing before it and nothing after it."
    )
    return prompt


def _build_consolidation_prompt(utterances: list[dict], prompt_config: dict) -> str:
    candidate_payload = prompt_config.get("candidate_payload", {})
    ambiguous_phrases = prompt_config.get("ambiguous_phrases", DEFAULT_AMBIGUOUS_PHRASES)
    output_format = _preferred_output_format(prompt_config)

    utterance_limit = int(prompt_config.get("consolidation_utterance_limit", 30))
    selected_utterances = utterances[:utterance_limit]

    rules = [
        "Use only candidate items and provided evidence context.",
        "Keep requester-side utterances as specification sources.",
        "Treat PM/Developer utterances as support-only and do not place them directly into FR/NFR/constraints/assumptions/open_questions.",
        "Do not add new requirements without evidence support.",
        "Deduplicate and improve wording, but preserve meaning.",
        "Ensure every extracted item has valid evidence IDs.",
        "Move uncertain items to assumptions or open_questions.",
        "For every ambiguity/open question, include question_decisions with decision in {needs_follow_up, already_asked, resolved}.",
        "If PM/Developer already asked and the client answered, use already_asked or resolved and do not include a follow_up_questions entry.",
        "Create targeted follow_up_questions for unresolved ambiguity.",
        "Do not include an `utterances` key in output.",
    ]

    if output_format == "yaml":
        rules.insert(0, "Return exactly one YAML mapping object.")
    else:
        rules.insert(0, "Return exactly one JSON object with the schema keys shown below.")

    if bool(prompt_config.get("repair_mode", False)):
        rules.append("Previous output was invalid; return a fully parseable output now.")

    schema_text = _schema_hint_text(
        output_format=output_format,
        compact=bool(prompt_config.get("repair_mode", False)),
    )
    output_contract = _render_strict_output_contract(output_format=output_format)

    if _use_compact_prompt(prompt_config):
        return (
            "Consolidate extracted requirement candidates.\n"
            "Return exactly one JSON object with keys:\n"
            "project_summary, functional_requirements, non_functional_requirements, constraints, "
            "assumptions, open_questions, follow_up_questions, question_decisions.\n"
            "No markdown, no explanation.\n"
            "Only requester-side utterances can be final spec sources.\n"
            "PM/Developer utterances are support-only for question decisions.\n"
            "Do not add unsupported items.\n"
            "Keep evidence IDs valid (U#).\n"
            "Only include follow_up_questions for needs_follow_up decisions.\n"
            "Candidate payload:\n"
            f"{json.dumps(candidate_payload, ensure_ascii=False)}\n"
            "Utterances:\n"
            f"{_render_utterances(selected_utterances)}\n"
            "Output JSON only."
        )

    return_keyword = "YAML mapping" if output_format == "yaml" else "JSON object"

    candidate_text = json.dumps(candidate_payload, ensure_ascii=False, indent=2)
    if output_format == "yaml":
        candidate_text = yaml.safe_dump(candidate_payload, allow_unicode=True, sort_keys=False)

    prompt = (
        "You are consolidating chunk-level software requirement extraction results.\n\n"
        f"{output_contract}\n"
        "Ambiguous phrases to treat cautiously:\n"
        f"- {', '.join(ambiguous_phrases)}\n\n"
        "Rules:\n"
        + "\n".join(f"{idx}. {rule}" for idx, rule in enumerate(rules, start=1))
        + "\n\n"
        "Output schema:\n"
        f"{schema_text}\n\n"
        "Candidate extracted payload:\n"
        f"{candidate_text}\n\n"
        "Evidence context utterances:\n"
        f"{_render_utterances(selected_utterances)}\n\n"
        f"Return only the {return_keyword} now. Nothing before it and nothing after it."
    )
    return prompt


def _build_utterance_rescue_prompt(utterances: list[dict], prompt_config: dict) -> str:
    output_format = _preferred_output_format(prompt_config)
    if output_format == "json":
        output_format = "yaml"

    utterance = utterances[0] if utterances else {"id": "U0", "speaker": "Unknown", "text": ""}
    utterance_text = utterance.get("text", "").strip()
    utterance_id = utterance.get("id", "U0")
    speaker = utterance.get("speaker", "Unknown")

    schema = {
        "category": "functional|non_functional|constraint|assumption|open_question|none",
        "text": "<extracted item text or empty>",
        "priority": "high|medium|low|null",
        "follow_up_question": "<optional>",
    }
    schema_text = yaml.safe_dump(schema, allow_unicode=True, sort_keys=False)

    return (
        "You are a strict requirement signal classifier.\n\n"
        "Given ONE utterance, decide whether it conveys a requirement-related item.\n"
        "Return one YAML mapping only.\n\n"
        "Rules:\n"
        "1. Choose category from: functional, non_functional, constraint, assumption, open_question, none.\n"
        "2. If category is none, set text to empty string.\n"
        "3. Keep text concise, grounded in the utterance, and do not invent details.\n"
        "4. If the utterance is a question or reveals missing detail, prefer open_question.\n"
        "5. For vague quality words (simple/fast/modern/etc.), prefer non_functional or open_question.\n"
        "6. If speaker is PM/Developer, do not output functional/non_functional/constraint/assumption.\n"
        "7. For PM/Developer utterances, prefer open_question or none.\n\n"
        "Output schema (YAML):\n"
        f"{schema_text}\n"
        "Utterance:\n"
        f"- id: {utterance_id}\n"
        f"  speaker: {speaker}\n"
        f"  text: {utterance_text}\n\n"
        "Return only the YAML mapping."
    )


def _build_hybrid_role_inference_prompt(utterances: list[dict], prompt_config: dict) -> str:
    return (
        "You classify each utterance speaker role in a client-software conversation.\n"
        "Return exactly one JSON object with key `roles`.\n"
        "Schema:\n"
        "{\n"
        '  "roles": [\n'
        '    {"id": "U1", "role": "requester|developer|unknown"}\n'
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- requester: client/customer/owner side expressing needs or answers.\n"
        "- developer: PM/engineer side asking clarifications/planning/constraints.\n"
        "- unknown only if impossible.\n"
        "- Do not include any keys other than `roles`.\n"
        "- JSON only.\n\n"
        "Utterances:\n"
        f"{_render_utterances(utterances)}\n"
    )


def _build_hybrid_candidate_extraction_prompt(utterances: list[dict], prompt_config: dict) -> str:
    return (
        "Extract requirement candidate items from requester-side utterances only.\n"
        "Return exactly one JSON object with keys `project_intent` and `candidates`.\n"
        "Schema:\n"
        "{\n"
        '  "project_intent": "<short intent>",\n'
        '  "candidates": [\n'
        "    {\n"
        '      "id": "C1",\n'
        '      "text": "<candidate requirement statement>",\n'
        '      "evidence": ["U1"],\n'
        '      "ambiguity": "clear|ambiguous"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "- Exclude developer-side planning/questions as spec candidates.\n"
        "- Keep each candidate atomic and actionable.\n"
        "- Evidence must reference existing U# IDs.\n"
        "- No markdown, no explanation.\n\n"
        "Utterances:\n"
        f"{_render_utterances(utterances)}\n"
    )


def _build_hybrid_spec_render_prompt(utterances: list[dict], prompt_config: dict) -> str:
    candidate_payload = prompt_config.get("candidate_payload", {})
    return (
        "You are writing a software requirements draft from classified candidates.\n"
        "Return exactly one JSON object using this schema keys only:\n"
        "project_summary, functional_requirements, non_functional_requirements, constraints, assumptions, "
        "open_questions, follow_up_questions, question_decisions.\n"
        "Rules:\n"
        "- Use candidate payload as primary source, grounded by evidence IDs.\n"
        "- Write spec-style requirement statements; do not copy conversational phrasing verbatim unless necessary.\n"
        "- If an assumption/open issue is unresolved, create a concrete follow-up question.\n"
        "- question_decisions must use decision in {needs_follow_up, already_asked, resolved}.\n"
        "- Only include follow_up_questions for needs_follow_up.\n"
        "- JSON only.\n\n"
        "Classified candidate payload:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "Utterances:\n"
        f"{_render_utterances(utterances)}\n"
    )


def build_prompt(utterances: list[dict], prompt_config: dict) -> str:
    stage = str(prompt_config.get("stage", "extract_chunk"))

    if stage == "consolidate":
        return _build_consolidation_prompt(utterances=utterances, prompt_config=prompt_config)
    if stage == "utterance_rescue":
        return _build_utterance_rescue_prompt(utterances=utterances, prompt_config=prompt_config)
    if stage == "hybrid_role_inference":
        return _build_hybrid_role_inference_prompt(utterances=utterances, prompt_config=prompt_config)
    if stage == "hybrid_candidate_extraction":
        return _build_hybrid_candidate_extraction_prompt(utterances=utterances, prompt_config=prompt_config)
    if stage == "hybrid_spec_render":
        return _build_hybrid_spec_render_prompt(utterances=utterances, prompt_config=prompt_config)

    return _build_extract_prompt(utterances=utterances, prompt_config=prompt_config)
