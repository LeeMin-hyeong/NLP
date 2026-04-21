import json

from app.model_runner import BaseModelRunner, MockModelRunner
from app.pipeline import run_pipeline


def test_mock_pipeline_writes_json_and_markdown(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: We need users to reserve tables online.",
                "Client: It should be simple for first-time users.",
                "PM: What is the target response time?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = MockModelRunner()

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "ambiguous_phrases": ["simple", "easy to use"],
            "generation": {},
        },
    )

    assert result.json_path.exists()
    assert result.markdown_path.exists()

    payload = json.loads(result.json_path.read_text(encoding="utf-8"))
    assert "project_summary" in payload
    assert "functional_requirements" in payload
    assert "utterances" in payload
    assert "run_metadata" in payload
    assert payload["run_metadata"]["execution_time_sec"] >= 0


class _StaticRunner(BaseModelRunner):
    def __init__(self, payload: dict) -> None:
        super().__init__(runner_name="static")
        self._payload = payload

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        return json.dumps(self._payload, ensure_ascii=False)


class _FailThenRescueRunner(BaseModelRunner):
    def __init__(self) -> None:
        super().__init__(runner_name="fail-then-rescue")

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        stage = str(prompt_config.get("stage", "extract_chunk"))
        if stage == "utterance_rescue":
            text = str(utterances[0].get("text", "")).lower()
            if "?" in text:
                return "category: open_question\ntext: Need clarification\nfollow_up_question: What should happen?\npriority: null\n"
            return "category: functional\ntext: Users can reserve tables\npriority: medium\n"
        return "needs_follow_up>"


class _HybridFlowRunner(BaseModelRunner):
    def __init__(self) -> None:
        super().__init__(runner_name="hybrid-static")
        self.stages: list[str] = []

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        stage = str(prompt_config.get("stage", "extract_chunk"))
        self.stages.append(stage)

        if stage == "hybrid_role_inference":
            return json.dumps(
                {
                    "roles": [
                        {"id": "U1", "role": "requester"},
                        {"id": "U2", "role": "requester"},
                        {"id": "U3", "role": "requester"},
                        {"id": "U4", "role": "developer"},
                    ]
                },
                ensure_ascii=False,
            )

        if stage == "hybrid_candidate_extraction":
            return json.dumps(
                {
                    "project_intent": "School clubs scheduling app",
                    "candidates": [
                        {
                            "id": "C1",
                            "text": "Allow club leaders to create and update events.",
                            "evidence": ["U1"],
                            "ambiguity": "clear",
                        },
                        {
                            "id": "C2",
                            "text": "It should be easy to use for freshmen.",
                            "evidence": ["U2"],
                            "ambiguity": "ambiguous",
                        },
                        {
                            "id": "C3",
                            "text": "Maybe attendance analytics can be added later.",
                            "evidence": ["U3"],
                            "ambiguity": "ambiguous",
                        },
                        {
                            "id": "C4",
                            "text": "What is the analytics deadline?",
                            "evidence": ["U4"],
                            "ambiguity": "ambiguous",
                        },
                    ],
                },
                ensure_ascii=False,
            )

        if stage == "hybrid_spec_render":
            draft = dict(prompt_config.get("candidate_payload", {}))
            return json.dumps(
                {
                    "project_summary": "School clubs scheduling requirements draft.",
                    "functional_requirements": draft.get("functional_requirements", []),
                    "non_functional_requirements": draft.get("non_functional_requirements", []),
                    "constraints": draft.get("constraints", []),
                    "assumptions": draft.get("assumptions", []),
                    "open_questions": draft.get("open_questions", []),
                    "follow_up_questions": draft.get("follow_up_questions", []),
                    "question_decisions": draft.get("question_decisions", []),
                },
                ensure_ascii=False,
            )

        return "{}"


class _HybridEmptyCandidateRunner(BaseModelRunner):
    def __init__(self) -> None:
        super().__init__(runner_name="hybrid-empty-candidate")
        self.stages: list[str] = []

    def generate_structured_output(self, utterances: list[dict], prompt_config: dict) -> str:
        stage = str(prompt_config.get("stage", "extract_chunk"))
        self.stages.append(stage)

        if stage == "hybrid_role_inference":
            return json.dumps({"roles": [{"id": "U1", "role": "requester"}, {"id": "U2", "role": "requester"}]}, ensure_ascii=False)
        if stage == "hybrid_candidate_extraction":
            return json.dumps({"project_intent": "Test", "candidates": []}, ensure_ascii=False)
        if stage == "utterance_rescue":
            text = str(utterances[0].get("text", "")).lower()
            if "?" in text:
                return "category: open_question\ntext: Clarify details\nfollow_up_question: What should happen?\npriority: null\n"
            return "category: functional\ntext: Users can reserve tables online.\npriority: medium\n"
        if stage == "hybrid_spec_render":
            return json.dumps(
                {
                    "project_summary": "should not be used in this test",
                    "functional_requirements": [],
                    "non_functional_requirements": [],
                    "constraints": [],
                    "assumptions": [],
                    "open_questions": [],
                    "follow_up_questions": [],
                    "question_decisions": [],
                },
                ensure_ascii=False,
            )
        return "{}"


def test_pipeline_enforces_llm_question_decisions(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: It should be simple for freshmen.",
                "PM: What response time target do you need?",
                "Client: Under 2 seconds.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "Test",
            "functional_requirements": [],
            "non_functional_requirements": [
                {
                    "id": "NFR1",
                    "text": "The app should be simple for freshmen.",
                    "priority": "medium",
                    "evidence": ["U1"],
                }
            ],
            "constraints": [],
            "assumptions": [],
            "open_questions": [],
            "follow_up_questions": [
                "What response time target do you need?",
                "How should simplicity be measured?",
            ],
            "question_decisions": [
                {
                    "text": "Response-time target",
                    "decision": "already_asked",
                    "evidence": ["U2", "U3"],
                },
                {
                    "text": "Simplicity requirement",
                    "decision": "needs_follow_up",
                    "suggested_follow_up": "How should simplicity be measured?",
                    "evidence": ["U1"],
                },
            ],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_rule_postprocess": False,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assert result.spec.follow_up_questions == ["How should simplicity be measured?"]
    assert len(result.spec.question_decisions) == 2


def test_pipeline_recovers_when_all_chunks_fail_via_llm_rescue(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: We need users to reserve tables online.",
                "PM: What is the target response time?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _FailThenRescueRunner()

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_chunking": False,
            "enable_final_refinement": False,
            "enable_failure_rescue_on_total_failure": True,
            "enable_rule_postprocess": False,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assert len(result.spec.functional_requirements) >= 1
    assert any(item.decision in {"needs_follow_up", "already_asked"} for item in result.spec.question_decisions)


def test_pipeline_sanity_reclassifies_obvious_mislabels(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Our school clubs need an app to publish weekly meeting schedules.",
                "Client: Club leaders should create and update events.",
                "Client: Students should join events and receive reminder notifications.",
                "Client: It should be easy to use for freshmen.",
                "Client: Maybe attendance analytics can be added later.",
                "PM: Who is allowed to cancel events?",
                "Client: Only club leaders and the student council should cancel events.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [
                {"id": "FR1", "text": "Our school clubs need an app to publish weekly meeting schedules.", "priority": "medium", "evidence": ["U1"]},
                {"id": "FR2", "text": "Club leaders should create and update events.", "priority": "medium", "evidence": ["U2"]},
                {"id": "FR3", "text": "Students should join events and receive reminder notifications.", "priority": "medium", "evidence": ["U3"]},
            ],
            "non_functional_requirements": [
                {"id": "NFR1", "text": "It should be easy to use for freshmen.", "priority": "medium", "evidence": ["U4"]},
                {"id": "NFR2", "text": "Maybe attendance analytics can be added later.", "priority": "medium", "evidence": ["U5"]},
                {"id": "NFR3", "text": "Who is allowed to cancel events?", "priority": "medium", "evidence": ["U6"]},
                {"id": "NFR4", "text": "Only club leaders and the student council should cancel events.", "priority": "medium", "evidence": ["U7"]},
            ],
            "constraints": [],
            "assumptions": [],
            "open_questions": [],
            "follow_up_questions": [],
            "question_decisions": [],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assert any("only club leaders" in item.text.lower() for item in result.spec.constraints)
    assert any("attendance analytics" in item.text.lower() for item in result.spec.assumptions)
    assert not any("who is allowed to cancel events" in item.text.lower() for item in result.spec.open_questions)
    assert any("who is allowed to cancel events" in item.text.lower() for item in result.spec.question_decisions)
    assert any(item.decision in {"already_asked", "resolved"} for item in result.spec.question_decisions)


def test_assumption_decision_becomes_resolved_after_topic_follow_up(tmp_path) -> None:
    conversation_path = tmp_path / "conversation_no_label.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Our school clubs need an app to publish weekly meeting schedules.",
                "Maybe attendance analytics can be added later.",
                "You should give us specific deadline for attendance analytics.",
                "I think it's okay for May 3rd.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [
                {
                    "id": "FR1",
                    "text": "Our school clubs need an app to publish weekly meeting schedules.",
                    "priority": "medium",
                    "evidence": ["U1"],
                }
            ],
            "non_functional_requirements": [
                {
                    "id": "NFR1",
                    "text": "Maybe attendance analytics can be added later.",
                    "priority": "medium",
                    "evidence": ["U2"],
                }
            ],
            "constraints": [],
            "assumptions": [],
            "open_questions": [],
            "follow_up_questions": [],
            "question_decisions": [],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assumption_decisions = [
        item for item in result.spec.question_decisions if "attendance analytics" in item.text.lower()
    ]
    assert assumption_decisions
    assert assumption_decisions[0].decision == "resolved"
    assert assumption_decisions[0].suggested_follow_up is None


def test_assumption_without_answer_requires_follow_up_question(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Maybe attendance analytics can be added later.",
                "PM: What is the target deadline for attendance analytics?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [],
            "non_functional_requirements": [],
            "constraints": [],
            "assumptions": [
                {
                    "text": "Maybe attendance analytics can be added later.",
                    "evidence": ["U1"],
                }
            ],
            "open_questions": [],
            "follow_up_questions": [],
            "question_decisions": [
                {
                    "text": "Attendance analytics deadline",
                    "decision": "already_asked",
                    "evidence": ["U2"],
                }
            ],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_requester_source_only": True,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assumption_decisions = [
        item for item in result.spec.question_decisions if "attendance analytics" in item.text.lower()
    ]
    assert assumption_decisions
    assert assumption_decisions[0].decision == "needs_follow_up"
    assert result.spec.follow_up_questions
    assert any("attendance analytics" in item.lower() for item in result.spec.follow_up_questions)


def test_pipeline_filters_pm_sourced_spec_items_and_keeps_support_decision(tmp_path) -> None:
    conversation_path = tmp_path / "conversation_no_label.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Our school clubs need an app to publish weekly meeting schedules.",
                "PM: You should give us specific deadline for attendance analytics.",
                "Client: I think it's okay for May 3rd.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [
                {
                    "id": "FR1",
                    "text": "Our school clubs need an app to publish weekly meeting schedules.",
                    "priority": "medium",
                    "evidence": ["U1"],
                },
                {
                    "id": "FR2",
                    "text": "You should give us specific deadline for attendance analytics.",
                    "priority": "medium",
                    "evidence": ["U2"],
                },
            ],
            "non_functional_requirements": [],
            "constraints": [
                {
                    "text": "You should give us specific deadline for attendance analytics.",
                    "evidence": ["U2"],
                }
            ],
            "assumptions": [],
            "open_questions": [
                {
                    "text": "What is the deadline for attendance analytics?",
                    "evidence": ["U2"],
                }
            ],
            "follow_up_questions": ["What is the deadline for attendance analytics?"],
            "question_decisions": [],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_requester_source_only": True,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assert len(result.spec.functional_requirements) == 1
    assert not result.spec.constraints
    assert not result.spec.open_questions
    decision_texts = [item.text.lower() for item in result.spec.question_decisions]
    assert any("deadline for attendance analytics" in text for text in decision_texts)
    assert any(item.decision == "resolved" for item in result.spec.question_decisions)


def test_pipeline_consolidates_and_trims_question_outputs(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: It should be easy to use for freshmen.",
                "PM: What is the target response time?",
                "Client: Under 2 seconds.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [],
            "non_functional_requirements": [
                {
                    "id": "NFR1",
                    "text": "It should be easy to use for freshmen.",
                    "priority": "medium",
                    "evidence": ["U1"],
                }
            ],
            "constraints": [],
            "assumptions": [],
            "open_questions": [
                {"text": "What is the target response time?", "evidence": ["U2"]},
                {"text": "What response time target do you need?", "evidence": ["U2"]},
                {"text": "How should ease of use be measured?", "evidence": ["U1"]},
            ],
            "follow_up_questions": [
                "What response time target do you need?",
                "How should ease of use be measured?",
                "What measurable acceptance criteria define easy to use?",
            ],
            "question_decisions": [
                {
                    "text": "What is the target response time?",
                    "decision": "already_asked",
                    "evidence": ["U2"],
                },
                {
                    "text": "Response time target",
                    "decision": "resolved",
                    "evidence": ["U2", "U3"],
                },
                {
                    "text": "How should ease of use be measured?",
                    "decision": "needs_follow_up",
                    "suggested_follow_up": "How should ease of use be measured?",
                    "evidence": ["U1"],
                },
                {
                    "text": "What measurable criteria define easy to use?",
                    "decision": "needs_follow_up",
                    "suggested_follow_up": "What measurable acceptance criteria define easy to use?",
                    "evidence": ["U1"],
                },
            ],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_requester_source_only": True,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
            "question_topic_similarity_threshold": 0.45,
            "max_open_questions": 4,
            "max_question_decisions": 4,
            "max_follow_up_questions": 1,
        },
    )

    assert not any("response time" in item.text.lower() for item in result.spec.open_questions)
    assert len(result.spec.follow_up_questions) == 1
    assert any(item.decision == "resolved" for item in result.spec.question_decisions)


def test_pipeline_drops_follow_up_for_resolved_topics(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Maybe attendance analytics can be added later.",
                "PM: What is the deadline for attendance analytics?",
                "Client: May 3rd is okay.",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [],
            "non_functional_requirements": [],
            "constraints": [],
            "assumptions": [
                {
                    "text": "Maybe attendance analytics can be added later.",
                    "evidence": ["U1"],
                }
            ],
            "open_questions": [],
            "follow_up_questions": ["What is the deadline for attendance analytics?"],
            "question_decisions": [
                {
                    "text": "attendance analytics deadline",
                    "decision": "resolved",
                    "evidence": ["U2", "U3"],
                }
            ],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": False,
            "enforce_requester_source_only": True,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": False,
        },
    )

    assert any(
        item.decision == "resolved" and "attendance analytics" in item.text.lower()
        for item in result.spec.question_decisions
    )
    assert not result.spec.follow_up_questions


def test_pipeline_llm_centric_mode_skips_rule_heavy_postprocess(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Maybe attendance analytics can be added later.",
                "PM: Can you clarify the deadline?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _StaticRunner(
        {
            "project_summary": "raw model output",
            "functional_requirements": [],
            "non_functional_requirements": [],
            "constraints": [],
            "assumptions": [
                {
                    "text": "Maybe attendance analytics can be added later.",
                    "evidence": ["U1"],
                }
            ],
            "open_questions": [],
            "follow_up_questions": [],
            "question_decisions": [],
        }
    )

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "enable_final_refinement": False,
            "enable_chunking": False,
            "llm_centric_mode": True,
            "llm_centric_align_follow_up": False,
            "llm_centric_trim_questions": False,
            "enable_sanity_postprocess": True,
            "enable_rule_postprocess": True,
            "enforce_requester_source_only": True,
            "enforce_llm_question_decisions": True,
            "auto_follow_up_fallback": True,
        },
    )

    assert len(result.spec.assumptions) == 1
    assert result.spec.question_decisions == []
    assert result.spec.follow_up_questions == []


def test_pipeline_hybrid_flow_runs_role_candidate_classify_render(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Club leaders should create and update events.",
                "Client: It should be easy to use for freshmen.",
                "Client: Maybe attendance analytics can be added later.",
                "PM: What is the analytics deadline?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _HybridFlowRunner()

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "hybrid_flow_mode": True,
            "classification_use_mock": True,
            "llm_retry_on_invalid": 1,
            "llm_centric_mode": True,
        },
    )

    assert "hybrid_role_inference" in runner.stages
    assert "hybrid_candidate_extraction" in runner.stages
    assert "hybrid_spec_render" in runner.stages

    assert any("create and update events" in item.text.lower() for item in result.spec.functional_requirements)
    assert any("easy to use" in item.text.lower() for item in result.spec.non_functional_requirements)
    assert any("attendance analytics" in item.text.lower() for item in result.spec.assumptions)
    assert not any("analytics deadline" in item.text.lower() for item in result.spec.open_questions)
    assert result.spec.follow_up_questions


def test_pipeline_hybrid_empty_candidates_recovers_with_rescue(tmp_path) -> None:
    conversation_path = tmp_path / "conversation.txt"
    conversation_path.write_text(
        "\n".join(
            [
                "Client: Users should reserve tables online.",
                "Client: Who can cancel a reservation?",
            ]
        ),
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    runner = _HybridEmptyCandidateRunner()

    result = run_pipeline(
        input_path=conversation_path,
        output_dir=output_dir,
        runner=runner,
        prompt_config={
            "generation": {},
            "hybrid_flow_mode": True,
            "hybrid_enable_empty_rescue": True,
            "classification_use_mock": True,
            "llm_retry_on_invalid": 1,
            "llm_centric_mode": True,
        },
    )

    assert "hybrid_candidate_extraction" in runner.stages
    assert "utterance_rescue" in runner.stages
    assert len(result.spec.functional_requirements) >= 1
