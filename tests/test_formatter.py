from app.formatter import spec_to_markdown
from app.schemas import NoteItem, QuestionDecisionItem, RequirementItem, SpecOutput, Utterance


def test_spec_to_markdown_contains_required_sections() -> None:
    spec = SpecOutput(
        project_summary="Test summary",
        functional_requirements=[
            RequirementItem(
                id="FR1",
                text="Users can register.",
                priority="high",
                evidence=["U1"],
            )
        ],
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="System should respond quickly.",
                priority="medium",
                evidence=["U2"],
            )
        ],
        constraints=[NoteItem(text="Must use school SSO", evidence=["U3"])],
        assumptions=[NoteItem(text="Analytics deferred", evidence=["U4"])],
        open_questions=[NoteItem(text="What is target response time?", evidence=["U2"])],
        follow_up_questions=["What is the launch deadline?"],
        question_decisions=[
            QuestionDecisionItem(
                text="Response-time target remains ambiguous.",
                decision="needs_follow_up",
                suggested_follow_up="What is the target response time in seconds?",
                evidence=["U2"],
            )
        ],
        utterances=[Utterance(id="U1", speaker="Client", text="Need register")],
    )

    markdown = spec_to_markdown(spec)

    assert "## Project Summary" in markdown
    assert "## Functional Requirements" in markdown
    assert "## Non-functional Requirements" in markdown
    assert "## Constraints" in markdown
    assert "## Assumptions" in markdown
    assert "## Open Questions" in markdown
    assert "## Follow-up Questions" in markdown
    assert "## Question Decisions" in markdown
    assert "Evidence: U1" in markdown
