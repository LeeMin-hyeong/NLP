from pathlib import Path

from app.evaluation import build_comparison_table, compute_metrics, evaluate_model
from app.model_runner import MockModelRunner
from app.pipeline import ConversationToSpecPipeline
from app.prompt_builder import load_prompt_config
from app.schemas import ConstraintItem, ConversationUnit, RequirementItem, SpecOutput


def test_compute_metrics_includes_chain_robustness_keys():
    samples = [
        {
            "id": "s1",
            "conversation_text": "Need booking.",
            "gold": {
                "functional_requirements": [{"text": "The system shall provide booking."}],
                "non_functional_requirements": [],
                "constraints": [{"text": "Online payment shall not be included in v1."}],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        }
    ]
    predicted = {
        "s1": SpecOutput(
            project_summary="Summary.",
            functional_requirements=[
                RequirementItem(
                    id="FR1",
                    text="The system shall provide booking.",
                    source_units=["U1"],
                )
            ],
            non_functional_requirements=[],
            constraints=[
                ConstraintItem(
                    id="CON1",
                    text="Online payment shall not be included in v1.",
                    source_units=["U1"],
                )
            ],
            open_questions=[],
            follow_up_questions=[],
            notes=[],
            conversation_units=[ConversationUnit(id="U1", text="Need booking.")],
            verification_warnings=[],
        )
    }
    statuses = {
        "s1": type(
            "Status",
            (),
            {
                "sample_id": "s1",
                "success": True,
                "json_parse_ok": True,
                "pydantic_validation_ok": True,
                "latency_sec": 0.5,
                "final_status": "success",
                "retry_count": 1,
                "retry_success": True,
                "retry_recovery": True,
                "fallback_rescue": False,
                "semantic_warning": False,
                "stage_failure": None,
                "stage_retry_counts": {"stage_2_candidate_classification": 1},
                "stage_stats": {
                    "stage_1_candidate_count": 5,
                    "stage_2_discard_rate": 0.4,
                    "stage_4_open_question_count": 2,
                    "stage_5_follow_up_count": 3,
                },
            },
        )()
    }
    metrics = compute_metrics(samples, predicted, statuses)
    assert metrics["functional_f1"] == 1.0
    assert metrics["constraint_f1"] == 1.0
    assert "json_parse_success_rate" in metrics
    assert "pydantic_validation_success_rate" in metrics
    assert "retry_success_rate" in metrics
    assert "retry_recovery_rate" in metrics
    assert "fallback_rescue_rate" in metrics
    assert "final_usable_output_rate" in metrics
    assert "semantic_warning_rate" in metrics
    assert "avg_stage_1_candidate_count" in metrics
    assert "avg_stage_2_discard_rate" in metrics
    assert "avg_stage_4_open_question_count" in metrics
    assert "avg_stage_5_follow_up_count" in metrics
    assert "stage_failure_counts" in metrics
    assert "constraint_semantic_warning_count" in metrics
    assert metrics["avg_stage_4_open_question_count"] == 2.0
    assert metrics["avg_stage_5_follow_up_count"] == 3.0


def test_compute_metrics_accepts_legacy_stage_key_fallback():
    samples = [
        {
            "id": "s1",
            "conversation_text": "Need booking.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "constraints": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
            },
        }
    ]
    predicted = {}
    statuses = {
        "s1": type(
            "Status",
            (),
            {
                "sample_id": "s1",
                "success": False,
                "json_parse_ok": False,
                "pydantic_validation_ok": False,
                "latency_sec": 0.0,
                "final_status": "failed",
                "retry_count": 0,
                "retry_success": False,
                "retry_recovery": False,
                "fallback_rescue": False,
                "semantic_warning": False,
                "stage_failure": None,
                "stage_retry_counts": {},
                "stage_stats": {"stage_4_follow_up_count": 4},
            },
        )()
    }
    metrics = compute_metrics(samples, predicted, statuses)
    assert metrics["avg_stage_4_open_question_count"] == 4.0
    assert metrics["avg_stage_5_follow_up_count"] == 4.0
    assert metrics["avg_stage_4_follow_up_count"] == 4.0


def test_comparison_table_uses_new_stage_diagnostic_columns():
    table = build_comparison_table(
        {
            "mock": {
                "metrics": {
                    "functional_f1": 1.0,
                    "non_functional_f1": 0.5,
                    "constraint_f1": 0.25,
                    "requirement_type_macro_f1": 0.75,
                    "open_question_recall": 0.1,
                    "follow_up_question_coverage": 0.2,
                    "hallucination_rate": 0.3,
                    "schema_validity_rate": 0.4,
                    "json_parse_success_rate": 0.5,
                    "pydantic_validation_success_rate": 0.6,
                    "retry_success_rate": 0.7,
                    "retry_recovery_rate": 0.65,
                    "fallback_rescue_rate": 0.15,
                    "final_usable_output_rate": 0.8,
                    "semantic_warning_rate": 0.9,
                    "avg_stage_1_candidate_count": 1.1,
                    "avg_stage_2_discard_rate": 1.2,
                    "avg_stage_4_open_question_count": 1.3,
                    "avg_stage_5_follow_up_count": 1.4,
                    "avg_latency_sec": 1.5,
                }
            }
        }
    )
    assert "stage4_open_questions" in table
    assert "stage5_follow_ups" in table
    assert "retry_recovery" in table
    assert "fallback_rescue" in table
    assert "1.3000" in table
    assert "1.4000" in table


def test_comparison_table_accepts_legacy_stage_metric_key():
    table = build_comparison_table(
        {
            "legacy": {
                "metrics": {
                    "avg_stage_4_follow_up_count": 2.5,
                }
            }
        }
    )
    assert "stage4_open_questions" in table
    assert "stage5_follow_ups" in table
    assert "2.5000 | 2.5000" in table


def test_comparison_table_marks_null_semantic_warning_as_na():
    table = build_comparison_table(
        {
            "NoSemanticVerify": {
                "metrics": {
                    "semantic_warning_rate": None,
                }
            }
        }
    )
    assert "N/A" in table


def test_evaluate_model_writes_debug_and_prediction_artifacts(tmp_path: Path):
    samples = [
        {
            "id": "mini",
            "conversation_text": "Need scheduling tool.\nIt should be fast.\nMaybe payment later.",
            "gold": {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "open_questions": [],
                "follow_up_questions": [],
                "notes": [],
                "constraints": [],
            },
        }
    ]
    pipeline = ConversationToSpecPipeline(
        runner=MockModelRunner(),
        prompt_config=load_prompt_config(),
        generation_config={},
    )
    report = evaluate_model(
        model_label="mock",
        pipeline=pipeline,
        samples=samples,
        output_dir=tmp_path,
        run_metadata={"pipeline_mode": "chain", "ablation_profile": "FullChain"},
    )
    assert "metrics" in report
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "run_config.json").exists()
    assert (tmp_path / "predictions" / "mini_pred.json").exists()
    assert (tmp_path / "debug" / "mini" / "summary.json").exists()
