from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

UTTERANCE_ID_PATTERN = re.compile(r"^U\d+$")


class Utterance(BaseModel):
    id: str
    speaker: str
    text: str

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        value = value.strip()
        if not UTTERANCE_ID_PATTERN.match(value):
            raise ValueError("Utterance id must follow pattern U<number> (e.g., U1)")
        return value

    @field_validator("speaker", "text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value


class RequirementItem(BaseModel):
    id: str
    text: str
    priority: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)

    @field_validator("id", "text")
    @classmethod
    def validate_non_empty_field(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Field cannot be empty")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: list[str]) -> list[str]:
        normalized = []
        for item in value:
            token = str(item).strip()
            if not token:
                continue
            if not UTTERANCE_ID_PATTERN.match(token):
                raise ValueError(f"Invalid evidence utterance id: {token}")
            normalized.append(token)

        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("Evidence must include at least one utterance id")
        return deduped


class NoteItem(BaseModel):
    text: str
    evidence: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: list[str]) -> list[str]:
        normalized = []
        for item in value:
            token = str(item).strip()
            if not token:
                continue
            if not UTTERANCE_ID_PATTERN.match(token):
                raise ValueError(f"Invalid evidence utterance id: {token}")
            normalized.append(token)

        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("Evidence must include at least one utterance id")
        return deduped


class RunMetadata(BaseModel):
    execution_time_sec: float = Field(ge=0.0)
    model_latency_sec: Optional[float] = Field(default=None, ge=0.0)
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = Field(default=None, ge=0)
    completion_tokens: Optional[int] = Field(default=None, ge=0)
    generated_at: Optional[str] = None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class QuestionDecisionItem(BaseModel):
    text: str
    decision: Literal["needs_follow_up", "already_asked", "resolved"]
    suggested_follow_up: Optional[str] = None
    evidence: list[str] = Field(default_factory=list)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value

    @field_validator("suggested_follow_up")
    @classmethod
    def validate_suggested_follow_up(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: list[str]) -> list[str]:
        normalized = []
        for item in value:
            token = str(item).strip()
            if not token:
                continue
            if not UTTERANCE_ID_PATTERN.match(token):
                raise ValueError(f"Invalid evidence utterance id: {token}")
            normalized.append(token)

        deduped = list(dict.fromkeys(normalized))
        if not deduped:
            raise ValueError("Evidence must include at least one utterance id")
        return deduped


class SpecOutput(BaseModel):
    project_summary: str
    functional_requirements: list[RequirementItem] = Field(default_factory=list)
    non_functional_requirements: list[RequirementItem] = Field(default_factory=list)
    constraints: list[NoteItem] = Field(default_factory=list)
    assumptions: list[NoteItem] = Field(default_factory=list)
    open_questions: list[NoteItem] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    question_decisions: list[QuestionDecisionItem] = Field(default_factory=list)
    utterances: list[Utterance] = Field(default_factory=list)
    run_metadata: Optional[RunMetadata] = None

    @field_validator("project_summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("project_summary cannot be empty")
        return value

    @field_validator("follow_up_questions")
    @classmethod
    def validate_follow_up_questions(cls, value: list[str]) -> list[str]:
        cleaned = []
        for question in value:
            text = str(question).strip()
            if text:
                cleaned.append(text)
        return cleaned
