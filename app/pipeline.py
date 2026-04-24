from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from app.extractor import (
    build_stage_1_fallback_candidates,
    coerce_rewrite_type_for_quality,
    ExtractionError,
    ExtractionMeta,
    enrich_classified_candidates,
    extract_spec_output_safe,
    parse_json_object_safe,
    semantic_verify,
    validate_stage_1_candidates,
    validate_stage_2_classification,
    validate_stage_3_rewriting,
    validate_stage_4_open_questions,
    validate_stage_5_followups,
    validate_stage_6_summary,
)
from app.formatter import format_spec_markdown
from app.model_runner import BaseModelRunner
from app.parser import load_conversation_text
from app.progress import (
    NullProgressReporter,
    STAGE_0,
    STAGE_FINAL,
    pipeline_step_index,
    pipeline_total_steps,
)
from app.prompt_builder import (
    build_single_shot_spec_prompt,
    build_stage_1_candidate_extraction_prompt,
    build_stage_2_candidate_classification_prompt,
    build_stage_3_requirement_rewriting_prompt,
    build_stage_4_open_question_generation_prompt,
    build_stage_5_followup_generation_prompt,
    build_stage_6_summary_prompt,
    build_stage_retry_prompt,
)
from app.schemas import (
    ConversationUnit,
    ConstraintItem,
    NoteItem,
    QuestionItem,
    RequirementItem,
    SpecOutput,
    Stage4OpenQuestionsOutput,
    Stage5FollowUpOutput,
    Stage6SummaryOutput,
)
from app.segmenter import segment_conversation
from app.utils import ensure_dir, model_dump_compat, write_json_file, write_text_file


PIPELINE_STATUS_SUCCESS = "success"
PIPELINE_STATUS_REPAIRED_SUCCESS = "repaired_success"
PIPELINE_STATUS_RETRY_SUCCESS = "retry_success"
PIPELINE_STATUS_SEMANTIC_WARNING = "semantic_warning"
PIPELINE_STATUS_FAILED_INVALID_OUTPUT = "failed_invalid_output"

STAGE_1 = "stage_1_candidate_extraction"
STAGE_2 = "stage_2_candidate_classification"
STAGE_3 = "stage_3_requirement_rewriting"
STAGE_4 = "stage_4_open_question_generation"
STAGE_5 = "stage_5_followup_generation"
STAGE_6 = "stage_6_project_summary"

T = TypeVar("T")


ROBUSTNESS_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    "FullChain": {
        "repair_enabled": True,
        "max_retries": 2,
        "stage_fallback_enabled": True,
        "semantic_verify_enabled": True,
    },
    "NoRetry": {
        "repair_enabled": True,
        "max_retries": 0,
        "stage_fallback_enabled": True,
        "semantic_verify_enabled": True,
    },
    "NoSemanticVerify": {
        "repair_enabled": True,
        "max_retries": 2,
        "stage_fallback_enabled": True,
        "semantic_verify_enabled": False,
    },
    "StrictRaw": {
        "repair_enabled": False,
        "max_retries": 0,
        "stage_fallback_enabled": False,
        "semantic_verify_enabled": False,
    },
}


class StageRunError(ExtractionError):
    def __init__(
        self,
        message: str,
        *,
        stage_name: str,
        attempt_logs: list[dict[str, Any]],
        last_raw_output: str,
        retry_count: int,
        latency_sec: float,
        prompt_tokens: Optional[int],
        completion_tokens: Optional[int],
        used_repair: bool,
    ) -> None:
        super().__init__(message)
        self.stage_name = stage_name
        self.attempt_logs = attempt_logs
        self.last_raw_output = last_raw_output
        self.retry_count = retry_count
        self.latency_sec = latency_sec
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.used_repair = used_repair


@dataclass
class PipelineRunResult:
    spec: SpecOutput
    raw_output: str
    extraction_meta: ExtractionMeta
    latency_sec: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    success: bool
    status: str
    retry_count: int
    error_message: Optional[str] = None
    attempt_logs: list[dict[str, Any]] = field(default_factory=list)
    stage_retry_counts: dict[str, int] = field(default_factory=dict)
    stage_failure: Optional[str] = None
    stage_stats: dict[str, Any] = field(default_factory=dict)
    pipeline_mode: str = "chain"
    robustness_profile: Optional[str] = None
    output_json_path: Optional[Path] = None
    output_md_path: Optional[Path] = None
    debug_dir: Optional[Path] = None

    @property
    def json_parse_ok(self) -> bool:
        return self.extraction_meta.json_parse_ok

    @property
    def pydantic_validation_ok(self) -> bool:
        return self.extraction_meta.pydantic_validation_ok

    @property
    def semantic_warnings(self) -> list[str]:
        return list(self.extraction_meta.semantic_warnings)


class ConversationToSpecPipeline:
    def __init__(
        self,
        runner: BaseModelRunner,
        prompt_config: dict,
        generation_config: Optional[dict] = None,
        pipeline_mode: str = "chain",
        robustness_profile: Optional[str] = None,
        robustness_config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.runner = runner
        self.prompt_config = prompt_config
        self.generation_config = generation_config or {}
        if pipeline_mode not in {"chain", "single_shot"}:
            raise ValueError("pipeline_mode must be 'chain' or 'single_shot'.")
        self.pipeline_mode = pipeline_mode
        self.robustness_profile = robustness_profile
        profile_config = ROBUSTNESS_PROFILE_CONFIGS.get(robustness_profile or "", {})
        self.robustness_config = {**profile_config, **(robustness_config or {})}
        self.progress_reporter = NullProgressReporter()

    def _max_retries(self) -> int:
        if "max_retries" in self.robustness_config:
            return int(self.robustness_config["max_retries"])
        return int(self.generation_config.get("max_retries", 2))

    def _repair_enabled(self) -> bool:
        return bool(self.robustness_config.get("repair_enabled", True))

    def _stage_fallback_enabled(self) -> bool:
        return bool(self.robustness_config.get("stage_fallback_enabled", True))

    def _semantic_verify_enabled(self) -> bool:
        return bool(self.robustness_config.get("semantic_verify_enabled", True))

    @staticmethod
    def _slug(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-") or "stage"

    @staticmethod
    def _append_tokens(total: Optional[int], value: Any) -> Optional[int]:
        if total is None:
            return None
        if isinstance(value, int):
            return total + value
        return None

    @staticmethod
    def _dedupe_question_items(items: list[QuestionItem]) -> list[QuestionItem]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[QuestionItem] = []
        for item in items:
            key = (item.text.strip().lower(), tuple(sorted(set(item.source_units))))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _dedupe_note_items(items: list[NoteItem]) -> list[NoteItem]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        deduped: list[NoteItem] = []
        for item in items:
            key = (item.text.strip().lower(), tuple(sorted(set(item.source_units))))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    @staticmethod
    def _build_fallback_spec(conversation_units: list[Any], error_message: str) -> SpecOutput:
        return SpecOutput(
            project_summary=(
                "The chain pipeline could not produce a valid specification. "
                "Please inspect stage debug artifacts and retry with a more stable model."
            ),
            functional_requirements=[],
            non_functional_requirements=[],
            constraints=[],
            open_questions=[],
            follow_up_questions=[],
            notes=[],
            conversation_units=conversation_units,
            verification_warnings=[error_message],
        )

    @staticmethod
    def _short_progress_error(error_message: str, max_len: int = 96) -> str:
        collapsed = re.sub(r"\s+", " ", str(error_message or "").strip())
        if len(collapsed) <= max_len:
            return collapsed
        return collapsed[: max_len - 3] + "..."

    def _run_stage_json(
        self,
        *,
        stage_name: str,
        initial_prompt: str,
        required_schema: dict[str, Any],
        validator: Callable[[dict[str, Any]], T],
        max_retries: int,
    ) -> tuple[T, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        attempt_logs: list[dict[str, Any]] = []
        last_error = "Unknown stage failure."
        last_raw_output = ""
        total_latency = 0.0
        total_prompt_tokens: Optional[int] = 0
        total_completion_tokens: Optional[int] = 0
        used_repair = False
        step_index = pipeline_step_index(stage_name)
        total_steps = pipeline_total_steps()

        for attempt_index in range(max_retries + 1):
            prompt_type = "initial" if attempt_index == 0 else "retry"
            if attempt_index == 0:
                prompt = initial_prompt
            else:
                prompt = build_stage_retry_prompt(
                    stage_name=stage_name,
                    error_message=last_error,
                    previous_output=last_raw_output,
                    required_schema=required_schema,
                    original_context=initial_prompt,
                )

            attempt_handle = self.progress_reporter.stage_attempt_started(
                stage_key=stage_name,
                step_index=step_index,
                total_steps=total_steps,
                attempt_index=attempt_index + 1,
                max_attempts=max_retries + 1,
            )
            try:
                raw_output = self.runner.generate(prompt, self.generation_config)
                last_raw_output = raw_output
                info = self.runner.last_generation_info or {}
                total_latency += float(info.get("latency_sec", 0.0))
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, info.get("prompt_tokens"))
                total_completion_tokens = self._append_tokens(
                    total_completion_tokens, info.get("completion_tokens")
                )

                payload, parse_meta = parse_json_object_safe(
                    raw_output,
                    allow_repair=self._repair_enabled(),
                )
                used_repair = used_repair or parse_meta.used_repair
                error_message = parse_meta.parse_error
                pydantic_ok = False
                if payload is not None:
                    try:
                        validated = validator(payload)
                        pydantic_ok = True
                        attempt_logs.append(
                            {
                                "stage": stage_name,
                                "attempt_index": attempt_index + 1,
                                "prompt_type": prompt_type,
                                "raw_output": raw_output,
                                "repaired_output": parse_meta.repaired_output,
                                "json_parse_ok": True,
                                "pydantic_validation_ok": True,
                                "used_repair": parse_meta.used_repair,
                                "error_message": None,
                            }
                        )
                        attempt_handle.finish("completed")
                        return (
                            validated,
                            raw_output,
                            attempt_logs,
                            attempt_index,
                            total_latency,
                            total_prompt_tokens,
                            total_completion_tokens,
                            used_repair,
                        )
                    except Exception as exc:
                        error_message = f"Schema validation failed: {exc}"

                last_error = error_message or "Invalid stage output."
                attempt_logs.append(
                    {
                        "stage": stage_name,
                        "attempt_index": attempt_index + 1,
                        "prompt_type": prompt_type,
                        "raw_output": raw_output,
                        "repaired_output": parse_meta.repaired_output,
                        "json_parse_ok": bool(payload is not None),
                        "pydantic_validation_ok": pydantic_ok,
                        "used_repair": parse_meta.used_repair,
                        "error_message": last_error,
                    }
                )
                if attempt_index < max_retries:
                    attempt_handle.finish(
                        f"needs retry: {self._short_progress_error(last_error)}"
                    )
                else:
                    attempt_handle.finish(
                        f"failed: {self._short_progress_error(last_error)}"
                    )
            except Exception as exc:
                attempt_handle.finish(f"failed: {self._short_progress_error(str(exc))}")
                raise

        raise StageRunError(
            f"{stage_name} failed after retries: {last_error}",
            stage_name=stage_name,
            attempt_logs=attempt_logs,
            last_raw_output=last_raw_output,
            retry_count=max_retries,
            latency_sec=total_latency,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            used_repair=used_repair,
        )

    def run_stage_1_candidate_extraction(
        self,
        conversation_units: list[Any],
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_1_candidate_extraction_prompt(conversation_units, self.prompt_config)
        schema = {
            "candidates": [
                {
                    "id": "C1",
                    "kind": "possible_requirement",
                    "text": "short candidate description",
                    "source_units": ["U1"],
                }
            ]
        }
        return self._run_stage_json(
            stage_name=STAGE_1,
            initial_prompt=prompt,
            required_schema=schema,
            validator=lambda payload: validate_stage_1_candidates(payload, conversation_units),
            max_retries=self._max_retries(),
        )

    def run_stage_2_candidate_classification(
        self,
        conversation_units: list[Any],
        stage1_output: Any,
    ) -> tuple[Any, list[dict[str, Any]], str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_2_candidate_classification_prompt(
            conversation_units,
            [model_dump_compat(item) for item in stage1_output.candidates],
            self.prompt_config,
        )
        schema = {
            "classified_candidates": [
                {
                    "id": "C1",
                    "final_type": "functional_requirement",
                    "reason": "brief explanation",
                    "source_units": ["U1"],
                }
            ]
        }
        result = self._run_stage_json(
            stage_name=STAGE_2,
            initial_prompt=prompt,
            required_schema=schema,
            validator=lambda payload: validate_stage_2_classification(
                payload, stage1_output, conversation_units
            ),
            max_retries=self._max_retries(),
        )
        stage2_output = result[0]
        enriched = enrich_classified_candidates(stage2_output, stage1_output)
        return (
            stage2_output,
            enriched,
            result[1],
            result[2],
            result[3],
            result[4],
            result[5],
            result[6],
            result[7],
        )

    def run_stage_3_requirement_rewriting(
        self,
        conversation_units: list[Any],
        enriched_classified_candidates: list[dict[str, Any]],
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_3_requirement_rewriting_prompt(
            conversation_units,
            enriched_classified_candidates,
            self.prompt_config,
        )
        schema = {
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
        return self._run_stage_json(
            stage_name=STAGE_3,
            initial_prompt=prompt,
            required_schema=schema,
            validator=lambda payload: validate_stage_3_rewriting(
                payload,
                conversation_units,
                authorized_rewrite_candidates=enriched_classified_candidates,
            ),
            max_retries=self._max_retries(),
        )

    def run_stage_4_open_question_generation(
        self,
        conversation_units: list[Any],
        enriched_classified_candidates: list[dict[str, Any]],
        rewritten_output: Any,
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_4_open_question_generation_prompt(
            conversation_units,
            enriched_classified_candidates,
            [model_dump_compat(item) for item in rewritten_output.rewritten_items],
            self.prompt_config,
        )
        schema = {
            "open_questions": [
                {
                    "text": "Which payment methods should be considered in a later release?",
                    "source_units": ["U2"],
                }
            ]
        }
        return self._run_stage_json(
            stage_name=STAGE_4,
            initial_prompt=prompt,
            required_schema=schema,
            validator=lambda payload: validate_stage_4_open_questions(payload, conversation_units),
            max_retries=self._max_retries(),
        )

    def run_stage_5_followup_generation(
        self,
        conversation_units: list[Any],
        enriched_classified_candidates: list[dict[str, Any]],
        rewritten_output: Any,
        open_questions: list[QuestionItem],
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_5_followup_generation_prompt(
            conversation_units,
            enriched_classified_candidates,
            [model_dump_compat(item) for item in rewritten_output.rewritten_items],
            [model_dump_compat(item) for item in open_questions],
            self.prompt_config,
        )
        schema = {
            "follow_up_questions": [
                {
                    "text": "What measurable acceptance criteria should be used?",
                    "source_units": ["U2"],
                }
            ]
        }
        return self._run_stage_json(
            stage_name=STAGE_5,
            initial_prompt=prompt,
            required_schema=schema,
            validator=lambda payload: validate_stage_5_followups(payload, conversation_units),
            max_retries=self._max_retries(),
        )

    def run_stage_4_followup_generation(
        self,
        conversation_units: list[Any],
        enriched_classified_candidates: list[dict[str, Any]],
        rewritten_output: Any,
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        """Backward-compatible wrapper for callers that used the old Stage 4 name."""
        open_questions, _, _ = self._build_open_questions_and_notes(enriched_classified_candidates)
        return self.run_stage_5_followup_generation(
            conversation_units,
            enriched_classified_candidates,
            rewritten_output,
            open_questions,
        )

    def run_stage_6_project_summary(
        self,
        conversation_units: list[Any],
        rewritten_output: Any,
        open_questions: list[QuestionItem],
        notes: list[NoteItem],
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        prompt = build_stage_6_summary_prompt(
            conversation_units,
            [model_dump_compat(item) for item in rewritten_output.rewritten_items],
            [model_dump_compat(item) for item in open_questions],
            [model_dump_compat(item) for item in notes],
            self.prompt_config,
        )
        schema = {"project_summary": "2-4 sentence summary"}
        return self._run_stage_json(
            stage_name=STAGE_6,
            initial_prompt=prompt,
            required_schema=schema,
            validator=validate_stage_6_summary,
            max_retries=self._max_retries(),
        )

    def run_stage_5_project_summary(
        self,
        conversation_units: list[Any],
        rewritten_output: Any,
        open_questions: list[QuestionItem],
        notes: list[NoteItem],
    ) -> tuple[Any, str, list[dict[str, Any]], int, float, Optional[int], Optional[int], bool]:
        """Backward-compatible wrapper for callers that used the old Stage 5 name."""
        return self.run_stage_6_project_summary(
            conversation_units,
            rewritten_output,
            open_questions,
            notes,
        )

    def _build_open_questions_and_notes(
        self,
        enriched_classified_candidates: list[dict[str, Any]],
    ) -> tuple[list[QuestionItem], list[NoteItem], int]:
        open_questions: list[QuestionItem] = []
        notes: list[NoteItem] = []
        discard_count = 0
        for item in enriched_classified_candidates:
            final_type = str(item.get("final_type", "")).strip()
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            text = str(item.get("text", "")).strip()
            reason = str(item.get("reason", "")).strip()

            if final_type == "open_question":
                oq_text = text.rstrip(".")
                if oq_text and not oq_text.endswith("?"):
                    oq_text = oq_text + "?"
                if not oq_text:
                    oq_text = (reason.rstrip(".") + "?").strip()
                open_questions.append(QuestionItem(text=oq_text, source_units=source_units))
            elif final_type == "note":
                note_text = text.rstrip(".")
                if not note_text:
                    note_text = reason.rstrip(".")
                if note_text:
                    notes.append(NoteItem(text=note_text, source_units=source_units))
            elif final_type == "discard":
                discard_count += 1

        return (
            self._dedupe_question_items(open_questions),
            self._dedupe_note_items(notes),
            discard_count,
        )

    def _build_fallback_open_questions(
        self,
        enriched_classified_candidates: list[dict[str, Any]],
    ) -> Stage4OpenQuestionsOutput:
        open_questions, _, _ = self._build_open_questions_and_notes(enriched_classified_candidates)
        return Stage4OpenQuestionsOutput(open_questions=open_questions)

    def _build_fallback_followups(
        self,
        *,
        open_questions: list[QuestionItem],
        notes: list[NoteItem],
        enriched_classified_candidates: list[dict[str, Any]],
    ) -> Stage5FollowUpOutput:
        questions: list[QuestionItem] = []
        for item in open_questions:
            text = item.text.strip().rstrip("?")
            if text:
                questions.append(
                    QuestionItem(
                        text=f"What decision should we make to resolve: {text}?",
                        source_units=item.source_units,
                    )
                )

        for note in notes:
            lowered = note.text.lower()
            if any(hint in lowered for hint in ("later", "future", "phase 2", "eventually", "maybe")):
                questions.append(
                    QuestionItem(
                        text="Should this future-scope item be included in the first release plan?",
                        source_units=note.source_units,
                    )
                )

        for item in enriched_classified_candidates:
            if str(item.get("final_type", "")).strip() != "constraint":
                continue
            source_units = [str(x).strip() for x in item.get("source_units", []) if str(x).strip()]
            if source_units:
                questions.append(
                    QuestionItem(
                        text="Can you confirm this boundary as a hard constraint for the initial release?",
                        source_units=source_units,
                    )
                )

        return Stage5FollowUpOutput(
            follow_up_questions=self._dedupe_question_items(questions)
        )

    @staticmethod
    def _build_fallback_summary(
        rewritten_output: Any,
        open_questions: list[QuestionItem],
        notes: list[NoteItem],
    ) -> Stage6SummaryOutput:
        feature_count = sum(
            1
            for item in rewritten_output.rewritten_items
            if item.type in {"functional_requirement", "non_functional_requirement"}
        )
        constraint_count = sum(
            1 for item in rewritten_output.rewritten_items if item.type == "constraint"
        )
        return Stage6SummaryOutput(
            project_summary=(
                "The conversation describes a software project with "
                f"{feature_count} drafted requirements, {constraint_count} constraints, "
                f"{len(open_questions)} unresolved questions, and {len(notes)} notes. "
                "Review the generated items and confirm ambiguous scope before implementation."
            )
        )

    def assemble_final_spec(
        self,
        *,
        conversation_units: list[Any],
        summary_output: Any,
        rewritten_output: Any,
        open_questions: list[QuestionItem],
        follow_up_output: Any,
        notes: list[NoteItem],
    ) -> tuple[SpecOutput, list[str]]:
        functional_requirements: list[RequirementItem] = []
        non_functional_requirements: list[RequirementItem] = []
        constraints: list[ConstraintItem] = []
        fr_idx = 1
        nfr_idx = 1
        con_idx = 1
        adjusted_open_questions: list[QuestionItem] = []
        for item in rewritten_output.rewritten_items:
            adjusted_type, clarification_question = coerce_rewrite_type_for_quality(item.text, item.type)
            if clarification_question:
                adjusted_open_questions.append(
                    QuestionItem(
                        text=clarification_question.rstrip("?") + "?",
                        source_units=item.source_units,
                    )
                )

            if adjusted_type == "functional_requirement":
                functional_requirements.append(
                    RequirementItem(
                        id=f"FR{fr_idx}",
                        text=item.text,
                        source_units=item.source_units,
                    )
                )
                fr_idx += 1
            elif adjusted_type == "non_functional_requirement":
                non_functional_requirements.append(
                    RequirementItem(
                        id=f"NFR{nfr_idx}",
                        text=item.text,
                        source_units=item.source_units,
                    )
                )
                nfr_idx += 1
            elif adjusted_type == "constraint":
                constraints.append(
                    ConstraintItem(
                        id=f"CON{con_idx}",
                        text=item.text,
                        source_units=item.source_units,
                    )
                )
                con_idx += 1

        follow_ups = self._dedupe_question_items(list(follow_up_output.follow_up_questions))
        spec = SpecOutput(
            project_summary=summary_output.project_summary,
            functional_requirements=functional_requirements,
            non_functional_requirements=non_functional_requirements,
            constraints=constraints,
            open_questions=self._dedupe_question_items(open_questions + adjusted_open_questions),
            follow_up_questions=follow_ups,
            notes=self._dedupe_note_items(notes),
            conversation_units=conversation_units,
            verification_warnings=[],
        )
        if not self._semantic_verify_enabled():
            return spec, []
        verified_spec, warnings = semantic_verify(spec, conversation_units)
        return verified_spec, warnings

    def _aggregate_attempts(self, stage_attempt_logs: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        for attempts in stage_attempt_logs:
            merged.extend(attempts)
        return merged

    @staticmethod
    def _write_debug_artifacts(debug_dir: Path, run: PipelineRunResult) -> None:
        if debug_dir.exists():
            for child in debug_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        ensure_dir(debug_dir)
        for attempt in run.attempt_logs:
            stage = str(attempt.get("stage", "stage"))
            idx = int(attempt.get("attempt_index", 0))
            stage_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", stage).strip("._-") or "stage"
            prefix = f"{stage_slug}_attempt_{idx:02d}"
            write_text_file(debug_dir / f"{prefix}_raw.txt", str(attempt.get("raw_output", "")))
            repaired = attempt.get("repaired_output")
            if repaired:
                write_text_file(debug_dir / f"{prefix}_repaired.txt", str(repaired))
            error_message = str(attempt.get("error_message") or "").strip()
            if error_message:
                write_text_file(debug_dir / f"{prefix}_error.txt", error_message)

        summary = {
            "status": run.status,
            "success": run.success,
            "retry_count": run.retry_count,
            "json_parse_ok": run.json_parse_ok,
            "pydantic_validation_ok": run.pydantic_validation_ok,
            "used_repair": run.extraction_meta.used_repair,
            "semantic_warnings": run.semantic_warnings,
            "error_message": run.error_message,
            "latency_sec": run.latency_sec,
            "prompt_tokens": run.prompt_tokens,
            "completion_tokens": run.completion_tokens,
            "attempt_count": len(run.attempt_logs),
            "stage_retry_counts": run.stage_retry_counts,
            "stage_failure": run.stage_failure,
            "stage_stats": run.stage_stats,
            "pipeline_mode": run.pipeline_mode,
            "robustness_profile": run.robustness_profile,
        }
        write_json_file(debug_dir / "summary.json", summary)

    def _run_single_shot_text(
        self,
        conversation_text: str,
        output_dir: Optional[Path] = None,
        output_basename: str = "spec",
        save_markdown: bool = True,
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        started = time.perf_counter()
        previous_reporter = self.progress_reporter
        self.progress_reporter = progress_reporter or NullProgressReporter()
        try:
            total_steps = 2
            self.progress_reporter.pipeline_started(total_steps=total_steps)
            try:
                self.progress_reporter.stage_started(
                    stage_key=STAGE_0,
                    step_index=1,
                    total_steps=total_steps,
                )
                conversation_units = segment_conversation(conversation_text)
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_0,
                    step_index=1,
                    total_steps=total_steps,
                    result_text=f"completed ({len(conversation_units)} units)",
                )
            except Exception as exc:
                conversation_units = [
                    ConversationUnit(id="U1", text=conversation_text.strip() or "(empty conversation)")
                ]
                error_message = f"Conversation segmentation failed: {exc}"
                spec = self._build_fallback_spec(conversation_units, error_message)
                meta = ExtractionMeta(
                    json_parse_ok=False,
                    pydantic_validation_ok=False,
                    used_repair=False,
                    validation_error=error_message,
                )
                run = PipelineRunResult(
                    spec=spec,
                    raw_output="",
                    extraction_meta=meta,
                    latency_sec=time.perf_counter() - started,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    status=PIPELINE_STATUS_FAILED_INVALID_OUTPUT,
                    retry_count=0,
                    error_message=error_message,
                    attempt_logs=[],
                    stage_retry_counts={},
                    stage_failure="stage_0_segmentation",
                    stage_stats={"pipeline_mode": "single_shot"},
                    pipeline_mode="single_shot",
                    robustness_profile=self.robustness_profile,
                )
                if output_dir is not None:
                    ensure_dir(output_dir)
                    output_json_path = output_dir / f"{output_basename}.json"
                    output_md_path = output_dir / f"{output_basename}.md"
                    write_json_file(output_json_path, model_dump_compat(run.spec))
                    if save_markdown:
                        write_text_file(output_md_path, format_spec_markdown(run.spec))
                    write_text_file(output_dir / "error.log", error_message)
                    debug_dir = output_dir / "debug" / output_basename
                    self._write_debug_artifacts(debug_dir, run)
                    run.output_json_path = output_json_path
                    run.output_md_path = output_md_path
                    run.debug_dir = debug_dir
                self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
                return run

            self.progress_reporter.stage_started(
                stage_key="single_shot_spec_generation",
                step_index=2,
                total_steps=total_steps,
            )
            prompt = build_single_shot_spec_prompt(conversation_units, self.prompt_config)
            raw_output = self.runner.generate(prompt, self.generation_config)
            info = self.runner.last_generation_info or {}
            latency_sec = float(info.get("latency_sec", 0.0)) or (time.perf_counter() - started)
            prompt_tokens = info.get("prompt_tokens")
            completion_tokens = info.get("completion_tokens")
            spec, meta = extract_spec_output_safe(
                raw_output,
                conversation_units,
                allow_repair=self._repair_enabled(),
            )
            semantic_warnings: list[str] = []
            stage_failure: Optional[str] = None
            error_message: Optional[str] = None
            success = spec is not None
            if spec is None:
                error_message = meta.validation_error or meta.parse_error or "Invalid single-shot output."
                stage_failure = "single_shot_spec_generation"
                spec = self._build_fallback_spec(conversation_units, error_message)
                status = PIPELINE_STATUS_FAILED_INVALID_OUTPUT
                self.progress_reporter.stage_finished(
                    stage_key="single_shot_spec_generation",
                    step_index=2,
                    total_steps=total_steps,
                    result_text=f"failed: {self._short_progress_error(error_message)}",
                )
            else:
                if self._semantic_verify_enabled():
                    spec, semantic_warnings = semantic_verify(spec, conversation_units)
                meta.semantic_warnings = semantic_warnings
                if semantic_warnings:
                    status = PIPELINE_STATUS_SEMANTIC_WARNING
                elif meta.used_repair:
                    status = PIPELINE_STATUS_REPAIRED_SUCCESS
                else:
                    status = PIPELINE_STATUS_SUCCESS
                self.progress_reporter.stage_finished(
                    stage_key="single_shot_spec_generation",
                    step_index=2,
                    total_steps=total_steps,
                    result_text="completed",
                )

            stage_stats = {
                "pipeline_mode": "single_shot",
                "repair_enabled": self._repair_enabled(),
                "semantic_verify_enabled": self._semantic_verify_enabled(),
                "stage_fallback_enabled": self._stage_fallback_enabled(),
            }
            attempt_logs = [
                {
                    "stage": "single_shot_spec_generation",
                    "attempt_index": 1,
                    "prompt_type": "initial",
                    "raw_output": raw_output,
                    "repaired_output": meta.repaired_output,
                    "json_parse_ok": meta.json_parse_ok,
                    "pydantic_validation_ok": meta.pydantic_validation_ok,
                    "used_repair": meta.used_repair,
                    "error_message": error_message,
                }
            ]
            run = PipelineRunResult(
                spec=spec,
                raw_output=raw_output,
                extraction_meta=meta,
                latency_sec=latency_sec,
                prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                success=success,
                status=status,
                retry_count=0,
                error_message=error_message,
                attempt_logs=attempt_logs,
                stage_retry_counts={},
                stage_failure=stage_failure,
                stage_stats=stage_stats,
                pipeline_mode="single_shot",
                robustness_profile=self.robustness_profile,
            )

            output_json_path: Optional[Path] = None
            output_md_path: Optional[Path] = None
            debug_dir: Optional[Path] = None
            if output_dir is not None:
                ensure_dir(output_dir)
                output_json_path = output_dir / f"{output_basename}.json"
                output_md_path = output_dir / f"{output_basename}.md"
                write_json_file(output_json_path, model_dump_compat(spec))
                if save_markdown:
                    write_text_file(output_md_path, format_spec_markdown(spec))
                error_log_path = output_dir / "error.log"
                if not success:
                    write_text_file(error_log_path, error_message or "Invalid structured output.")
                elif error_log_path.exists():
                    error_log_path.unlink()
                debug_dir = output_dir / "debug" / output_basename
                self._write_debug_artifacts(debug_dir, run)
            run.output_json_path = output_json_path
            run.output_md_path = output_md_path
            run.debug_dir = debug_dir
            self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
            return run
        finally:
            self.progress_reporter = previous_reporter

    def run_text(
        self,
        conversation_text: str,
        output_dir: Optional[Path] = None,
        output_basename: str = "spec",
        save_markdown: bool = True,
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        if self.pipeline_mode == "single_shot":
            return self._run_single_shot_text(
                conversation_text=conversation_text,
                output_dir=output_dir,
                output_basename=output_basename,
                save_markdown=save_markdown,
                progress_reporter=progress_reporter,
            )
        return self._run_chain_text(
            conversation_text=conversation_text,
            output_dir=output_dir,
            output_basename=output_basename,
            save_markdown=save_markdown,
            progress_reporter=progress_reporter,
        )

    def _run_chain_text(
        self,
        conversation_text: str,
        output_dir: Optional[Path] = None,
        output_basename: str = "spec",
        save_markdown: bool = True,
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        started = time.perf_counter()
        previous_reporter = self.progress_reporter
        self.progress_reporter = progress_reporter or NullProgressReporter()
        try:
            total_steps = pipeline_total_steps()
            self.progress_reporter.pipeline_started(total_steps=total_steps)
            total_latency = 0.0
            total_prompt_tokens: Optional[int] = 0
            total_completion_tokens: Optional[int] = 0
            stage_retry_counts: dict[str, int] = {}
            stage_attempt_logs: list[list[dict[str, Any]]] = []
            stage_outputs_raw: list[tuple[str, str]] = []
            stage_stats: dict[str, Any] = {
                "pipeline_mode": "chain",
                "repair_enabled": self._repair_enabled(),
                "semantic_verify_enabled": self._semantic_verify_enabled(),
                "stage_fallback_enabled": self._stage_fallback_enabled(),
            }
            used_repair_any = False

            try:
                self.progress_reporter.stage_started(
                    stage_key=STAGE_0,
                    step_index=pipeline_step_index(STAGE_0),
                    total_steps=total_steps,
                )
                conversation_units = segment_conversation(conversation_text)
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_0,
                    step_index=pipeline_step_index(STAGE_0),
                    total_steps=total_steps,
                    result_text=f"completed ({len(conversation_units)} units)",
                )
            except Exception as exc:
                conversation_units = [
                    ConversationUnit(id="U1", text=conversation_text.strip() or "(empty conversation)")
                ]
                error_message = f"Conversation segmentation failed: {exc}"
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_0,
                    step_index=pipeline_step_index(STAGE_0),
                    total_steps=total_steps,
                    result_text=f"failed: {self._short_progress_error(error_message)}",
                )
                spec = self._build_fallback_spec(conversation_units, error_message)
                meta = ExtractionMeta(
                    json_parse_ok=False,
                    pydantic_validation_ok=False,
                    used_repair=False,
                    validation_error=error_message,
                )
                run = PipelineRunResult(
                    spec=spec,
                    raw_output="",
                    extraction_meta=meta,
                    latency_sec=0.0,
                    prompt_tokens=None,
                    completion_tokens=None,
                    success=False,
                    status=PIPELINE_STATUS_FAILED_INVALID_OUTPUT,
                    retry_count=0,
                    error_message=error_message,
                    attempt_logs=[],
                    stage_retry_counts={},
                    stage_failure="stage_0_segmentation",
                    stage_stats={
                        "pipeline_mode": "chain",
                        "repair_enabled": self._repair_enabled(),
                        "semantic_verify_enabled": self._semantic_verify_enabled(),
                        "stage_fallback_enabled": self._stage_fallback_enabled(),
                    },
                    pipeline_mode="chain",
                    robustness_profile=self.robustness_profile,
                )
                if output_dir is not None:
                    ensure_dir(output_dir)
                    output_json_path = output_dir / f"{output_basename}.json"
                    output_md_path = output_dir / f"{output_basename}.md"
                    write_json_file(output_json_path, model_dump_compat(run.spec))
                    if save_markdown:
                        write_text_file(output_md_path, format_spec_markdown(run.spec))
                    write_text_file(output_dir / "error.log", error_message)
                    debug_dir = output_dir / "debug" / output_basename
                    self._write_debug_artifacts(debug_dir, run)
                    run.output_json_path = output_json_path
                    run.output_md_path = output_md_path
                    run.debug_dir = debug_dir
                self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=0.0)
                return run

            stage_failure: Optional[str] = None
            error_message: Optional[str] = None
            final_spec: Optional[SpecOutput] = None
            semantic_warnings: list[str] = []
            soft_warnings: list[str] = []
            retry_count_total = 0
            status = PIPELINE_STATUS_SUCCESS

            try:
                try:
                    self.progress_reporter.stage_started(
                        stage_key=STAGE_1,
                        step_index=pipeline_step_index(STAGE_1),
                        total_steps=total_steps,
                    )
                    stage1_output, raw1, logs1, retry1, lat1, pt1, ct1, repaired1 = self.run_stage_1_candidate_extraction(
                        conversation_units
                    )
                except StageRunError as exc:
                    if not self._stage_fallback_enabled():
                        raise
                    stage1_output = build_stage_1_fallback_candidates(conversation_units)
                    raw1 = exc.last_raw_output
                    logs1 = exc.attempt_logs
                    retry1 = exc.retry_count
                    lat1 = exc.latency_sec
                    pt1 = exc.prompt_tokens
                    ct1 = exc.completion_tokens
                    repaired1 = exc.used_repair
                    soft_warnings.append(f"{STAGE_1} used deterministic fallback: {exc}")
                    stage_stats["stage_1_fallback_used"] = True
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_1,
                        step_index=pipeline_step_index(STAGE_1),
                        total_steps=total_steps,
                        result_text="used deterministic fallback",
                    )
                stage_retry_counts[STAGE_1] = retry1
                retry_count_total += retry1
                total_latency += lat1
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt1)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct1)
                stage_attempt_logs.append(logs1)
                stage_outputs_raw.append((STAGE_1, raw1))
                used_repair_any = used_repair_any or repaired1
                stage_stats["stage_1_candidate_count"] = len(stage1_output.candidates)
                if not stage_stats.get("stage_1_fallback_used"):
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_1,
                        step_index=pipeline_step_index(STAGE_1),
                        total_steps=total_steps,
                        result_text=f"completed ({len(stage1_output.candidates)} candidates)",
                    )

                self.progress_reporter.stage_started(
                    stage_key=STAGE_2,
                    step_index=pipeline_step_index(STAGE_2),
                    total_steps=total_steps,
                )
                (
                    stage2_output,
                    enriched_stage2,
                    raw2,
                    logs2,
                    retry2,
                    lat2,
                    pt2,
                    ct2,
                    repaired2,
                ) = self.run_stage_2_candidate_classification(conversation_units, stage1_output)
                stage_retry_counts[STAGE_2] = retry2
                retry_count_total += retry2
                total_latency += lat2
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt2)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct2)
                stage_attempt_logs.append(logs2)
                stage_outputs_raw.append((STAGE_2, raw2))
                used_repair_any = used_repair_any or repaired2

                seeded_open_questions, notes, discard_count = self._build_open_questions_and_notes(enriched_stage2)
                stage_stats["stage_2_constraint_candidate_count"] = sum(
                    1 for item in enriched_stage2 if str(item.get("final_type", "")).strip() == "constraint"
                )
                stage_stats["stage_2_discard_count"] = discard_count
                stage_stats["stage_2_discard_rate"] = (
                    (discard_count / max(1, len(stage1_output.candidates)))
                )
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_2,
                    step_index=pipeline_step_index(STAGE_2),
                    total_steps=total_steps,
                    result_text=f"completed ({len(enriched_stage2)} labeled items)",
                )

                self.progress_reporter.stage_started(
                    stage_key=STAGE_3,
                    step_index=pipeline_step_index(STAGE_3),
                    total_steps=total_steps,
                )
                stage3_output, raw3, logs3, retry3, lat3, pt3, ct3, repaired3 = self.run_stage_3_requirement_rewriting(
                    conversation_units,
                    enriched_stage2,
                )
                stage_retry_counts[STAGE_3] = retry3
                retry_count_total += retry3
                total_latency += lat3
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt3)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct3)
                stage_attempt_logs.append(logs3)
                stage_outputs_raw.append((STAGE_3, raw3))
                used_repair_any = used_repair_any or repaired3
                stage_stats["stage_3_constraint_count"] = sum(
                    1 for item in stage3_output.rewritten_items if item.type == "constraint"
                )
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_3,
                    step_index=pipeline_step_index(STAGE_3),
                    total_steps=total_steps,
                    result_text=f"completed ({len(stage3_output.rewritten_items)} rewritten items)",
                )

                try:
                    self.progress_reporter.stage_started(
                        stage_key=STAGE_4,
                        step_index=pipeline_step_index(STAGE_4),
                        total_steps=total_steps,
                    )
                    stage4_output, raw4, logs4, retry4, lat4, pt4, ct4, repaired4 = self.run_stage_4_open_question_generation(
                        conversation_units,
                        enriched_stage2,
                        stage3_output,
                    )
                except StageRunError as exc:
                    if not self._stage_fallback_enabled():
                        raise
                    stage4_output = self._build_fallback_open_questions(enriched_stage2)
                    raw4 = exc.last_raw_output
                    logs4 = exc.attempt_logs
                    retry4 = exc.retry_count
                    lat4 = exc.latency_sec
                    pt4 = exc.prompt_tokens
                    ct4 = exc.completion_tokens
                    repaired4 = exc.used_repair
                    soft_warnings.append(f"{STAGE_4} used deterministic fallback: {exc}")
                    stage_stats["stage_4_fallback_used"] = True
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_4,
                        step_index=pipeline_step_index(STAGE_4),
                        total_steps=total_steps,
                        result_text="used deterministic fallback",
                    )
                stage_retry_counts[STAGE_4] = retry4
                retry_count_total += retry4
                total_latency += lat4
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt4)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct4)
                stage_attempt_logs.append(logs4)
                stage_outputs_raw.append((STAGE_4, raw4))
                used_repair_any = used_repair_any or repaired4
                open_questions = self._dedupe_question_items(
                    seeded_open_questions + list(stage4_output.open_questions)
                )
                stage_stats["stage_4_open_question_count"] = len(stage4_output.open_questions)
                if not stage_stats.get("stage_4_fallback_used"):
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_4,
                        step_index=pipeline_step_index(STAGE_4),
                        total_steps=total_steps,
                        result_text=f"completed ({len(stage4_output.open_questions)} generated questions)",
                    )

                try:
                    self.progress_reporter.stage_started(
                        stage_key=STAGE_5,
                        step_index=pipeline_step_index(STAGE_5),
                        total_steps=total_steps,
                    )
                    stage5_output, raw5, logs5, retry5, lat5, pt5, ct5, repaired5 = self.run_stage_5_followup_generation(
                        conversation_units,
                        enriched_stage2,
                        stage3_output,
                        open_questions,
                    )
                except StageRunError as exc:
                    if not self._stage_fallback_enabled():
                        raise
                    stage5_output = self._build_fallback_followups(
                        open_questions=open_questions,
                        notes=notes,
                        enriched_classified_candidates=enriched_stage2,
                    )
                    raw5 = exc.last_raw_output
                    logs5 = exc.attempt_logs
                    retry5 = exc.retry_count
                    lat5 = exc.latency_sec
                    pt5 = exc.prompt_tokens
                    ct5 = exc.completion_tokens
                    repaired5 = exc.used_repair
                    soft_warnings.append(f"{STAGE_5} used deterministic fallback: {exc}")
                    stage_stats["stage_5_fallback_used"] = True
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_5,
                        step_index=pipeline_step_index(STAGE_5),
                        total_steps=total_steps,
                        result_text="used deterministic fallback",
                    )
                stage_retry_counts[STAGE_5] = retry5
                retry_count_total += retry5
                total_latency += lat5
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt5)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct5)
                stage_attempt_logs.append(logs5)
                stage_outputs_raw.append((STAGE_5, raw5))
                used_repair_any = used_repair_any or repaired5
                stage_stats["stage_5_follow_up_count"] = len(stage5_output.follow_up_questions)
                if not stage_stats.get("stage_5_fallback_used"):
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_5,
                        step_index=pipeline_step_index(STAGE_5),
                        total_steps=total_steps,
                        result_text=f"completed ({len(stage5_output.follow_up_questions)} follow-up questions)",
                    )

                try:
                    self.progress_reporter.stage_started(
                        stage_key=STAGE_6,
                        step_index=pipeline_step_index(STAGE_6),
                        total_steps=total_steps,
                    )
                    stage6_output, raw6, logs6, retry6, lat6, pt6, ct6, repaired6 = self.run_stage_6_project_summary(
                        conversation_units,
                        stage3_output,
                        open_questions,
                        notes,
                    )
                except StageRunError as exc:
                    if not self._stage_fallback_enabled():
                        raise
                    stage6_output = self._build_fallback_summary(stage3_output, open_questions, notes)
                    raw6 = exc.last_raw_output
                    logs6 = exc.attempt_logs
                    retry6 = exc.retry_count
                    lat6 = exc.latency_sec
                    pt6 = exc.prompt_tokens
                    ct6 = exc.completion_tokens
                    repaired6 = exc.used_repair
                    soft_warnings.append(f"{STAGE_6} used deterministic fallback: {exc}")
                    stage_stats["stage_6_fallback_used"] = True
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_6,
                        step_index=pipeline_step_index(STAGE_6),
                        total_steps=total_steps,
                        result_text="used deterministic fallback",
                    )
                stage_retry_counts[STAGE_6] = retry6
                retry_count_total += retry6
                total_latency += lat6
                total_prompt_tokens = self._append_tokens(total_prompt_tokens, pt6)
                total_completion_tokens = self._append_tokens(total_completion_tokens, ct6)
                stage_attempt_logs.append(logs6)
                stage_outputs_raw.append((STAGE_6, raw6))
                used_repair_any = used_repair_any or repaired6
                if not stage_stats.get("stage_6_fallback_used"):
                    self.progress_reporter.stage_finished(
                        stage_key=STAGE_6,
                        step_index=pipeline_step_index(STAGE_6),
                        total_steps=total_steps,
                        result_text="completed",
                    )

                self.progress_reporter.stage_started(
                    stage_key=STAGE_FINAL,
                    step_index=pipeline_step_index(STAGE_FINAL),
                    total_steps=total_steps,
                )
                final_spec, semantic_warnings = self.assemble_final_spec(
                    conversation_units=conversation_units,
                    summary_output=stage6_output,
                    rewritten_output=stage3_output,
                    open_questions=open_questions,
                    follow_up_output=stage5_output,
                    notes=notes,
                )
                if soft_warnings:
                    final_spec.verification_warnings = sorted(
                        set(list(final_spec.verification_warnings) + soft_warnings)
                    )
                    semantic_warnings = sorted(set(semantic_warnings + soft_warnings))
                self.progress_reporter.stage_finished(
                    stage_key=STAGE_FINAL,
                    step_index=pipeline_step_index(STAGE_FINAL),
                    total_steps=total_steps,
                    result_text="completed",
                )
            except Exception as exc:
                stage_failure = stage_failure or (
                    str(exc).split(" failed after retries:", 1)[0] if " failed after retries:" in str(exc) else "unknown_stage"
                )
                error_message = str(exc)
                final_spec = self._build_fallback_spec(conversation_units, error_message)
                status = PIPELINE_STATUS_FAILED_INVALID_OUTPUT

            if final_spec is None:
                error_message = error_message or "Pipeline failed to produce final spec."
                final_spec = self._build_fallback_spec(conversation_units, error_message)
                status = PIPELINE_STATUS_FAILED_INVALID_OUTPUT

            if stage_failure is None:
                if semantic_warnings:
                    status = PIPELINE_STATUS_SEMANTIC_WARNING
                elif retry_count_total > 0:
                    status = PIPELINE_STATUS_RETRY_SUCCESS
                elif used_repair_any:
                    status = PIPELINE_STATUS_REPAIRED_SUCCESS
                else:
                    status = PIPELINE_STATUS_SUCCESS

            extraction_meta = ExtractionMeta(
                json_parse_ok=stage_failure is None,
                pydantic_validation_ok=stage_failure is None,
                used_repair=used_repair_any,
                semantic_warnings=semantic_warnings,
                validation_error=error_message if stage_failure is not None else None,
            )

            success = stage_failure is None
            latency_sec = total_latency if total_latency > 0 else (time.perf_counter() - started)
            merged_attempt_logs = self._aggregate_attempts(stage_attempt_logs)
            raw_output = "\n\n".join(f"[{stage}]\n{raw}" for stage, raw in stage_outputs_raw)

            run = PipelineRunResult(
                spec=final_spec,
                raw_output=raw_output,
                extraction_meta=extraction_meta,
                latency_sec=latency_sec,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                success=success,
                status=status,
                retry_count=retry_count_total,
                error_message=error_message,
                attempt_logs=merged_attempt_logs,
                stage_retry_counts=stage_retry_counts,
                stage_failure=stage_failure,
                stage_stats=stage_stats,
                pipeline_mode="chain",
                robustness_profile=self.robustness_profile,
            )

            output_json_path: Optional[Path] = None
            output_md_path: Optional[Path] = None
            debug_dir: Optional[Path] = None
            if output_dir is not None:
                ensure_dir(output_dir)
                output_json_path = output_dir / f"{output_basename}.json"
                output_md_path = output_dir / f"{output_basename}.md"
                write_json_file(output_json_path, model_dump_compat(final_spec))
                if save_markdown:
                    write_text_file(output_md_path, format_spec_markdown(final_spec))
                error_log_path = output_dir / "error.log"
                if not success:
                    write_text_file(error_log_path, error_message or "Invalid structured output.")
                elif error_log_path.exists():
                    error_log_path.unlink()
                debug_dir = output_dir / "debug" / output_basename
                self._write_debug_artifacts(debug_dir, run)

            run.output_json_path = output_json_path
            run.output_md_path = output_md_path
            run.debug_dir = debug_dir
            self.progress_reporter.pipeline_finished(status=run.status, elapsed_sec=run.latency_sec)
            return run
        finally:
            self.progress_reporter = previous_reporter

    def run_file(
        self,
        input_path: Path,
        output_dir: Path,
        output_basename: str = "spec",
        progress_reporter: Any | None = None,
    ) -> PipelineRunResult:
        text = load_conversation_text(input_path)
        return self.run_text(
            conversation_text=text,
            output_dir=output_dir,
            output_basename=output_basename,
            save_markdown=True,
            progress_reporter=progress_reporter,
        )
