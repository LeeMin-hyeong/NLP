from __future__ import annotations

from pathlib import Path
import re

from app.schemas import Utterance

SPEAKER_PREFIX_PATTERN = re.compile(r"^\s*([^:\n]{1,40})\s*:\s*(.+?)\s*$")
QUESTION_START_HINTS = (
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "can ",
    "could ",
    "do ",
    "does ",
    "is ",
    "are ",
    "would ",
    "should ",
    "deadline",
    "clarify",
    "확인",
    "질문",
    "언제",
    "어떻게",
    "무엇",
)
PM_HINTS = (
    "acceptance criteria",
    "technical",
    "api",
    "database",
    "clarify",
    "confirm",
    "priority",
    "scope",
    "timeline",
    "release",
    "deadline",
    "requirements",
    "구체적으로",
    "정리",
    "확인",
    "우선순위",
    "범위",
)
PM_GUIDANCE_HINTS = (
    "you should give us",
    "can you",
    "could you",
    "please",
    "specific deadline",
    "acceptance criteria",
    "clarify",
    "confirm",
    "scope",
    "timeline",
    "target date",
    "마감",
    "구체",
    "정리",
    "확인",
)
CLIENT_HINTS = (
    "we need",
    "we want",
    "we would like",
    "our ",
    "our team",
    "our users",
    "for our",
    "must",
    "should",
    "important",
    "pain point",
    "고객",
    "사용자",
    "우리",
    "필요",
    "원해",
    "원합니다",
    "원해요",
    "중요",
)
SHORT_ANSWER_HINTS = (
    "yes",
    "no",
    "maybe",
    "around",
    "about",
    "next",
    "probably",
    "not sure",
    "for now",
    "later",
    "네",
    "아니요",
    "아마",
    "나중",
    "일단",
)
CLIENT_SPEAKER_ALIASES = {
    "client",
    "customer",
    "user",
    "stakeholder",
    "requester",
    "고객",
    "클라이언트",
    "사용자",
    "유저",
}
PM_SPEAKER_ALIASES = {
    "pm",
    "project manager",
    "developer",
    "dev",
    "engineer",
    "team lead",
    "assistant",
    "analyst",
    "기획자",
    "개발자",
    "팀장",
}


def _normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", label).strip()


def _canonicalize_speaker_label(label: str) -> str:
    cleaned = _normalize_label(label)
    lowered = cleaned.lower()
    if lowered in CLIENT_SPEAKER_ALIASES:
        return "Client"
    if lowered in PM_SPEAKER_ALIASES:
        return "PM"
    return cleaned


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _count_matches(text: str, hints: tuple[str, ...]) -> int:
    return sum(1 for hint in hints if hint in text)


def _next_known_speaker(entries: list[tuple[str | None, str]], start_idx: int) -> str | None:
    for idx in range(start_idx, len(entries)):
        speaker = entries[idx][0]
        if speaker is not None:
            return speaker
    return None


def _infer_speaker(text: str, previous_speaker: str | None, next_speaker: str | None) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "Unknown"

    client_score = _count_matches(lowered, CLIENT_HINTS)
    pm_score = _count_matches(lowered, PM_HINTS)
    if _contains_any(lowered, PM_GUIDANCE_HINTS):
        pm_score += 1
    if "give us" in lowered and _contains_any(lowered, ("deadline", "timeline", "scope", "clarify", "confirm")):
        pm_score += 2

    if "?" in text:
        pm_score += 2
    if lowered.startswith(QUESTION_START_HINTS):
        pm_score += 1

    looks_short_answer = len(lowered.split()) <= 7 and _contains_any(lowered, SHORT_ANSWER_HINTS)

    if client_score > pm_score:
        return "Client"
    if pm_score > client_score:
        return "PM"

    if previous_speaker == "PM" and ("?" not in text):
        if looks_short_answer or len(lowered.split()) <= 8:
            return "Client"

    if previous_speaker == "Client" and looks_short_answer and "?" not in text:
        return "Client"

    if previous_speaker == "Client" and "?" in text:
        return "PM"

    if previous_speaker == "Client" and _contains_any(
        lowered,
        ("you should give us", "can you", "could you", "please"),
    ):
        return "PM"

    if previous_speaker == "Client" and _contains_any(
        lowered,
        ("we need", "we want", "our ", "must", "필요", "원해", "우리"),
    ):
        return "Client"

    if previous_speaker == "PM" and _contains_any(lowered, ("api", "database", "scope", "timeline", "요구사항")):
        return "PM"

    if _contains_any(lowered, ("you should give us", "can you", "could you", "please")) and _contains_any(
        lowered,
        ("deadline", "timeline", "scope", "target", "clarify", "confirm", "마감", "확인"),
    ):
        return "PM"

    if previous_speaker in {"Client", "PM"} and next_speaker in {"Client", "PM"}:
        if previous_speaker != next_speaker:
            return next_speaker
        return previous_speaker

    return "Unknown"


def parse_conversation_text(text: str) -> list[Utterance]:
    raw_entries: list[tuple[str | None, str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        speaker_match = SPEAKER_PREFIX_PATTERN.match(line)
        if speaker_match and not speaker_match.group(2).lstrip().startswith("//"):
            raw_entries.append(
                (
                    _canonicalize_speaker_label(speaker_match.group(1)),
                    speaker_match.group(2).strip(),
                )
            )
        else:
            raw_entries.append((None, line))

    entries: list[tuple[str, str]] = []
    for idx, (speaker, content) in enumerate(raw_entries):
        if speaker is not None:
            entries.append((speaker, content))
            continue

        previous_speaker = entries[-1][0] if entries else None
        next_speaker = _next_known_speaker(raw_entries, idx + 1)
        inferred = _infer_speaker(content, previous_speaker=previous_speaker, next_speaker=next_speaker)
        entries.append((inferred, content))

    utterances = [
        Utterance(id=f"U{index}", speaker=speaker, text=content)
        for index, (speaker, content) in enumerate(entries, start=1)
    ]

    if not utterances:
        raise ValueError("No utterances could be parsed from the input conversation")

    return utterances


def parse_conversation_file(path: str | Path) -> list[Utterance]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Conversation file not found: {input_path}")
    return parse_conversation_text(input_path.read_text(encoding="utf-8"))
