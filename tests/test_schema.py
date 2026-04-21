import pytest
from pydantic import ValidationError

from app.schemas import QuestionDecisionItem, RequirementItem, RunMetadata, SpecOutput, Utterance


def test_requirement_item_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        RequirementItem(id="FR1", text="Users can login", priority="high", evidence=[])


def test_spec_output_validation_success() -> None:
    spec = SpecOutput(
        project_summary="Summary",
        functional_requirements=[
            RequirementItem(id="FR1", text="Users can login", priority="high", evidence=["U1"])
        ],
        non_functional_requirements=[],
        constraints=[],
        assumptions=[],
        open_questions=[],
        follow_up_questions=[],
        question_decisions=[
            QuestionDecisionItem(
                text="Clarify the response-time target.",
                decision="needs_follow_up",
                suggested_follow_up="What is the target response time in seconds?",
                evidence=["U1"],
            )
        ],
        utterances=[Utterance(id="U1", speaker="Client", text="Need login")],
        run_metadata=RunMetadata(
            execution_time_sec=1.23,
            model_latency_sec=0.98,
            model_name="mock",
            prompt_tokens=120,
            completion_tokens=90,
        ),
    )
    assert spec.project_summary == "Summary"
    assert spec.run_metadata is not None
    assert spec.run_metadata.execution_time_sec == 1.23
    assert len(spec.question_decisions) == 1
