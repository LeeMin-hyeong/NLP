from app.evaluation import (
    compute_evidence_linking_counts,
    compute_evidence_jaccard,
    compute_hallucination_counts,
    compute_open_question_recall_counts,
    compute_requirement_extraction_counts,
    compute_type_classification_counts,
)
from app.schemas import NoteItem, RequirementItem, SpecOutput, Utterance


def _build_gold_spec() -> SpecOutput:
    return SpecOutput(
        project_summary="gold",
        functional_requirements=[
            RequirementItem(id="FR1", text="Users can book seats.", priority="high", evidence=["U1"])
        ],
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="System responds in under 2 seconds.",
                priority="high",
                evidence=["U2"],
            )
        ],
        constraints=[NoteItem(text="Must use Google login.", evidence=["U3"])],
        assumptions=[NoteItem(text="Payments are out of scope.", evidence=["U4"])],
        open_questions=[NoteItem(text="Which payment gateway is preferred?", evidence=["U4"])],
        follow_up_questions=["Which payment gateway should be used?"],
        utterances=[Utterance(id="U1", speaker="Client", text="Need booking")],
    )


def _build_pred_spec() -> SpecOutput:
    return SpecOutput(
        project_summary="pred",
        functional_requirements=[
            RequirementItem(id="FR1", text="Users can book seats.", priority="high", evidence=["U1"]),
            RequirementItem(
                id="FR2",
                text="Admins can export invoices.",
                priority="low",
                evidence=["U5"],
            ),
        ],
        non_functional_requirements=[
            RequirementItem(
                id="NFR1",
                text="System responds in under 2 seconds.",
                priority="high",
                evidence=["U2"],
            )
        ],
        constraints=[NoteItem(text="Must use Google login.", evidence=["U3"])],
        assumptions=[],
        open_questions=[NoteItem(text="Which payment gateway is preferred?", evidence=["U4"])],
        follow_up_questions=[],
        utterances=[Utterance(id="U1", speaker="Client", text="Need booking")],
    )


def test_requirement_precision_recall_counts() -> None:
    pred = _build_pred_spec()
    gold = _build_gold_spec()

    tp, fp, fn = compute_requirement_extraction_counts(pred, gold)
    assert (tp, fp, fn) == (2, 1, 0)

    exact_tp, exact_fp, exact_fn = compute_requirement_extraction_counts(
        pred,
        gold,
        match_mode="exact",
    )
    assert (exact_tp, exact_fp, exact_fn) == (2, 1, 0)


def test_type_classification_counts() -> None:
    pred = _build_pred_spec()
    gold = _build_gold_spec()

    counts = compute_type_classification_counts(pred, gold)
    assert counts["functional"]["tp"] >= 1
    assert counts["assumption"]["fn"] >= 1


def test_evidence_and_hallucination_metrics() -> None:
    pred = _build_pred_spec()
    gold = _build_gold_spec()

    ev_tp, ev_fp, ev_fn = compute_evidence_linking_counts(pred, gold)
    assert ev_tp >= 4
    assert ev_fp >= 1
    assert ev_fn >= 1

    unsupported, total_pred = compute_hallucination_counts(pred, gold)
    assert unsupported == 1
    assert total_pred == 3

    jaccard_sum, jaccard_count = compute_evidence_jaccard(pred, gold)
    assert jaccard_count > 0
    assert jaccard_sum > 0


def test_open_question_recall_counts() -> None:
    pred = _build_pred_spec()
    gold = _build_gold_spec()

    captured, total = compute_open_question_recall_counts(pred, gold)
    assert (captured, total) == (1, 1)
