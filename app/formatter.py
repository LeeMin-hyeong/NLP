from __future__ import annotations

from app.schemas import NoteItem, QuestionDecisionItem, RequirementItem, SpecOutput


def _format_requirement_list(items: list[RequirementItem]) -> str:
    if not items:
        return "- None"

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. **{item.id}** {item.text}")
        if item.priority:
            lines.append(f"   - Priority: {item.priority}")
        lines.append(f"   - Evidence: {', '.join(item.evidence)}")
    return "\n".join(lines)


def _format_note_list(items: list[NoteItem]) -> str:
    if not items:
        return "- None"

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. {item.text}")
        lines.append(f"   - Evidence: {', '.join(item.evidence)}")
    return "\n".join(lines)


def _format_follow_up_questions(questions: list[str]) -> str:
    if not questions:
        return "- None"
    return "\n".join(f"{index}. {question}" for index, question in enumerate(questions, start=1))


def _format_question_decisions(items: list[QuestionDecisionItem]) -> str:
    if not items:
        return "- None"

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. [{item.decision}] {item.text}")
        if item.suggested_follow_up:
            lines.append(f"   - Suggested Follow-up: {item.suggested_follow_up}")
        lines.append(f"   - Evidence: {', '.join(item.evidence)}")
    return "\n".join(lines)


def spec_to_markdown(spec: SpecOutput) -> str:
    sections = [
        "# Conversation-to-Spec Draft",
        "",
        "## Project Summary",
        spec.project_summary,
        "",
        "## Functional Requirements",
        _format_requirement_list(spec.functional_requirements),
        "",
        "## Non-functional Requirements",
        _format_requirement_list(spec.non_functional_requirements),
        "",
        "## Constraints",
        _format_note_list(spec.constraints),
        "",
        "## Assumptions",
        _format_note_list(spec.assumptions),
        "",
        "## Open Questions",
        _format_note_list(spec.open_questions),
        "",
        "## Follow-up Questions",
        _format_follow_up_questions(spec.follow_up_questions),
        "",
        "## Question Decisions",
        _format_question_decisions(spec.question_decisions),
        "",
    ]
    return "\n".join(sections)
