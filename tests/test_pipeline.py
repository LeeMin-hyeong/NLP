import json
from pathlib import Path

from app.evaluation import evaluate_model
from app.model_runner import MockModelRunner
from app.pipeline import ConversationToSpecPipeline
from app.prompt_builder import load_prompt_config
from app.segmenter import segment_conversation


class FirstStage2FailRunner(MockModelRunner):
    def __init__(self) -> None:
        super().__init__()
        self._failed_once = False

    def generate(self, prompt: str, generation_config: dict) -> str:
        if "CHAIN_STAGE:2_CANDIDATE_CLASSIFICATION" in prompt and not self._failed_once:
            self._failed_once = True
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": 0.01,
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            return "not-json"
        return super().generate(prompt, generation_config)


class Stage5MalformedRunner(MockModelRunner):
    def generate(self, prompt: str, generation_config: dict) -> str:
        if "CHAIN_STAGE:5_FOLLOWUP_GENERATION" in prompt:
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": 0.01,
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            return '{"developer_questions_text":"What should we do next?"}'
        return super().generate(prompt, generation_config)


class Stage1PlaceholderRunner(MockModelRunner):
    def generate(self, prompt: str, generation_config: dict) -> str:
        if "CHAIN_STAGE:1_CANDIDATE_EXTRACTION" in prompt:
            self.last_generation_info = {
                "model_name": self.model_name,
                "latency_sec": 0.01,
                "prompt_tokens": None,
                "completion_tokens": None,
            }
            return json.dumps(
                {
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
                            "source_units": ["U1"],
                        },
                    ]
                }
            )
        return super().generate(prompt, generation_config)


class QualityAsFRRunner(MockModelRunner):
    def generate(self, prompt: str, generation_config: dict) -> str:
        self.last_generation_info = {
            "model_name": self.model_name,
            "latency_sec": 0.01,
            "prompt_tokens": None,
            "completion_tokens": None,
        }
        if "CHAIN_STAGE:1_CANDIDATE_EXTRACTION" in prompt:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "id": "C1",
                            "kind": "possible_requirement",
                            "text": "Customers should reserve tables online.",
                            "source_units": ["U1"],
                        },
                        {
                            "id": "C2",
                            "kind": "possible_requirement",
                            "text": "The website should load quickly on mobile devices.",
                            "source_units": ["U2"],
                        },
                    ]
                }
            )
        if "CHAIN_STAGE:2_CANDIDATE_CLASSIFICATION" in prompt:
            return json.dumps(
                {
                    "classified_candidates": [
                        {
                            "id": "C1",
                            "final_type": "functional_requirement",
                            "reason": "Capability",
                            "source_units": ["U1"],
                        },
                        {
                            "id": "C2",
                            "final_type": "functional_requirement",
                            "reason": "Model mislabeled quality as FR",
                            "source_units": ["U2"],
                        },
                    ]
                }
            )
        if "CHAIN_STAGE:3_REQUIREMENT_REWRITING" in prompt:
            return json.dumps(
                {
                    "rewritten_items": [
                        {
                            "id": "R1",
                            "type": "functional_requirement",
                            "text": "The system shall allow customers to reserve tables online.",
                            "source_units": ["U1"],
                        },
                        {
                            "id": "R2",
                            "type": "functional_requirement",
                            "text": "The website shall load quickly on mobile devices.",
                            "source_units": ["U2"],
                        },
                    ]
                }
            )
        if "CHAIN_STAGE:4_OPEN_QUESTION_GENERATION" in prompt:
            return json.dumps({"open_questions": []})
        if "CHAIN_STAGE:5_FOLLOWUP_GENERATION" in prompt:
            return json.dumps({"follow_up_questions": []})
        if "CHAIN_STAGE:6_PROJECT_SUMMARY" in prompt:
            return json.dumps({"project_summary": "Summary."})
        return super().generate(prompt, generation_config)


def _sample_text() -> str:
    return (
        "We need a cafe website with menu and hours.\n"
        "Customers should reserve tables online.\n"
        "Online payment is not part of the first release.\n"
        "Only staff should have access to the admin page.\n"
        "The design should feel clean and modern.\n"
        "We may add online payment later."
    )


def _build_pipeline(runner=None, generation_config=None) -> ConversationToSpecPipeline:
    return ConversationToSpecPipeline(
        runner=runner or MockModelRunner(),
        prompt_config=load_prompt_config(),
        generation_config=generation_config or {"max_retries": 1},
    )


def test_stage_1_returns_candidates_with_source_units():
    pipeline = _build_pipeline()
    units = segment_conversation(_sample_text())
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    assert stage1.candidates
    for candidate in stage1.candidates:
        assert candidate.source_units


def test_stage_2_classifies_candidates_with_allowed_labels():
    pipeline = _build_pipeline()
    units = segment_conversation(_sample_text())
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    stage2, enriched, *_ = pipeline.run_stage_2_candidate_classification(units, stage1)
    assert stage2.classified_candidates
    stage1_ids = {c.id for c in stage1.candidates}
    classified_ids = {c.id for c in stage2.classified_candidates}
    assert stage1_ids == classified_ids
    for item in stage2.classified_candidates:
        assert item.final_type in {
            "functional_requirement",
            "non_functional_requirement",
            "constraint",
            "open_question",
            "follow_up_trigger",
            "note",
            "discard",
        }
    assert enriched
    assert any(item.final_type == "constraint" for item in stage2.classified_candidates)
    assert any(item.final_type == "note" for item in stage2.classified_candidates)


def test_stage_3_rewrites_without_losing_traceability():
    pipeline = _build_pipeline()
    units = segment_conversation(_sample_text())
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    _, enriched, *_ = pipeline.run_stage_2_candidate_classification(units, stage1)
    stage3, *_ = pipeline.run_stage_3_requirement_rewriting(units, enriched)
    for item in stage3.rewritten_items:
        assert item.type in {"functional_requirement", "non_functional_requirement", "constraint"}
        assert item.text.strip()
        assert item.source_units
    assert any(item.type == "constraint" for item in stage3.rewritten_items)


def test_stage_4_generates_open_questions_tied_to_ambiguity():
    pipeline = _build_pipeline()
    units = segment_conversation(_sample_text())
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    _, enriched, *_ = pipeline.run_stage_2_candidate_classification(units, stage1)
    stage3, *_ = pipeline.run_stage_3_requirement_rewriting(units, enriched)
    stage4, *_ = pipeline.run_stage_4_open_question_generation(units, enriched, stage3)
    assert stage4.open_questions
    for question in stage4.open_questions:
        assert question.text.strip().endswith("?")
        assert question.source_units


def test_stage_5_generates_followup_questions_tied_to_ambiguity():
    pipeline = _build_pipeline()
    units = segment_conversation(_sample_text())
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    _, enriched, *_ = pipeline.run_stage_2_candidate_classification(units, stage1)
    stage3, *_ = pipeline.run_stage_3_requirement_rewriting(units, enriched)
    seeded_open_questions, _, _ = pipeline._build_open_questions_and_notes(enriched)
    stage4, *_ = pipeline.run_stage_4_open_question_generation(units, enriched, stage3)
    open_questions = pipeline._dedupe_question_items(
        seeded_open_questions + list(stage4.open_questions)
    )
    stage5, *_ = pipeline.run_stage_5_followup_generation(units, enriched, stage3, open_questions)
    assert stage5.follow_up_questions
    for question in stage5.follow_up_questions:
        assert question.text.strip().endswith("?")
        assert question.source_units


def test_constraint_budget_triggers_followup_question():
    text = (
        "We need a booking website.\n"
        "The first release must fit a limited budget.\n"
        "Online payment is not part of the first release."
    )
    pipeline = _build_pipeline()
    units = segment_conversation(text)
    stage1, *_ = pipeline.run_stage_1_candidate_extraction(units)
    _, enriched, *_ = pipeline.run_stage_2_candidate_classification(units, stage1)
    stage3, *_ = pipeline.run_stage_3_requirement_rewriting(units, enriched)
    seeded_open_questions, _, _ = pipeline._build_open_questions_and_notes(enriched)
    stage4, *_ = pipeline.run_stage_4_open_question_generation(units, enriched, stage3)
    open_questions = pipeline._dedupe_question_items(
        seeded_open_questions + list(stage4.open_questions)
    )
    stage5, *_ = pipeline.run_stage_5_followup_generation(units, enriched, stage3, open_questions)
    assert any("budget" in question.text.lower() for question in stage5.follow_up_questions)


def test_final_assembly_preserves_schema(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline()
    run = pipeline.run_file(input_path=input_path, output_dir=tmp_path)

    assert run.success is True
    assert run.spec.project_summary
    assert run.spec.conversation_units
    assert (tmp_path / "spec.json").exists()
    assert (tmp_path / "spec.md").exists()
    loaded = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert "functional_requirements" in loaded
    assert "non_functional_requirements" in loaded
    assert "constraints" in loaded
    assert "open_questions" in loaded
    assert "follow_up_questions" in loaded
    assert "notes" in loaded


def test_single_shot_pipeline_preserves_schema(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = ConversationToSpecPipeline(
        runner=MockModelRunner(),
        prompt_config=load_prompt_config(),
        generation_config={"max_retries": 1},
        pipeline_mode="single_shot",
    )
    run = pipeline.run_file(input_path=input_path, output_dir=tmp_path)

    assert run.success is True
    assert run.pipeline_mode == "single_shot"
    assert run.spec.project_summary
    assert (tmp_path / "spec.json").exists()
    loaded = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert "functional_requirements" in loaded
    assert "non_functional_requirements" in loaded
    assert "follow_up_questions" in loaded


def test_strict_raw_profile_disables_stage_fallback():
    pipeline = ConversationToSpecPipeline(
        runner=Stage1PlaceholderRunner(),
        prompt_config=load_prompt_config(),
        generation_config={"max_retries": 0},
        robustness_profile="StrictRaw",
    )
    run = pipeline.run_text(_sample_text())

    assert run.success is False
    assert run.stage_failure == "stage_1_candidate_extraction"
    assert run.stage_stats.get("stage_fallback_enabled") is False


def test_stage_failure_does_not_crash_batch_evaluation(tmp_path: Path):
    samples = [
        {
            "id": "s1",
            "conversation_text": _sample_text(),
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        },
        {
            "id": "s2",
            "conversation_text": "Need scheduling app.\nUsers should receive reminders.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        },
    ]
    pipeline = _build_pipeline(runner=FirstStage2FailRunner(), generation_config={"max_retries": 0})
    report = evaluate_model("mock-fail-once", pipeline, samples, tmp_path)
    assert report["metrics"]["sample_count"] == 2
    assert (tmp_path / "predictions" / "s1_error.txt").exists()
    assert (tmp_path / "predictions" / "s2_pred.json").exists()


def test_stage_5_failure_uses_fallback_and_keeps_final_spec(tmp_path: Path):
    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline(runner=Stage5MalformedRunner(), generation_config={"max_retries": 0})
    run = pipeline.run_file(input_path=input_path, output_dir=tmp_path)

    assert run.success is True
    assert run.spec.project_summary
    assert run.spec.follow_up_questions
    assert any("stage_5_followup_generation used deterministic fallback" in w for w in run.spec.verification_warnings)
    assert run.stage_stats.get("stage_5_fallback_used") is True
    loaded = json.loads((tmp_path / "spec.json").read_text(encoding="utf-8"))
    assert loaded["functional_requirements"] or loaded["non_functional_requirements"] or loaded["constraints"]


def test_debug_artifacts_cleanup_removes_stale_files(tmp_path: Path):
    output_dir = tmp_path / "out"
    stale_debug_dir = output_dir / "debug" / "spec"
    stale_debug_dir.mkdir(parents=True, exist_ok=True)
    stale_file = stale_debug_dir / "stale_from_old_run.txt"
    stale_file.write_text("stale", encoding="utf-8")
    stale_error_log = output_dir / "error.log"
    stale_error_log.write_text("old error", encoding="utf-8")

    input_path = tmp_path / "sample.txt"
    input_path.write_text(_sample_text(), encoding="utf-8")
    pipeline = _build_pipeline()
    run = pipeline.run_file(input_path=input_path, output_dir=output_dir)

    assert run.success is True
    assert run.debug_dir is not None
    assert not stale_file.exists()
    assert not stale_error_log.exists()
    assert (run.debug_dir / "summary.json").exists()


def test_mobile_performance_ends_as_nfr_not_fr_in_final_spec():
    text = (
        "Customers should reserve tables online.\n"
        "The website should load quickly on mobile devices."
    )
    pipeline = _build_pipeline(runner=QualityAsFRRunner(), generation_config={"max_retries": 0})
    run = pipeline.run_text(text)
    assert run.success is True
    assert any("load quickly on mobile" in item.text.lower() for item in run.spec.non_functional_requirements)
    assert not any("load quickly on mobile" in item.text.lower() for item in run.spec.functional_requirements)


def test_stage_1_placeholder_echo_uses_fallback_and_pipeline_succeeds():
    text = (
        "Customers should reserve tables online.\n"
        "The website should load quickly on mobile devices.\n"
        "The design should feel clean and modern."
    )
    pipeline = _build_pipeline(runner=Stage1PlaceholderRunner(), generation_config={"max_retries": 0})
    run = pipeline.run_text(text)
    assert run.success is True
    assert run.stage_stats.get("stage_1_fallback_used") is True
    assert any("stage_1_candidate_extraction used deterministic fallback" in w for w in run.spec.verification_warnings)
    assert any("load quickly on mobile" in item.text.lower() for item in run.spec.non_functional_requirements)
