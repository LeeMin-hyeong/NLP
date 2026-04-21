from app.extractor import extract_spec_output, parse_json_payload


def test_parse_json_payload_handles_extra_wrapped_text() -> None:
    raw = "prefix text\n{\"project_summary\": \"ok\"}\ntrailing explanation"
    payload = parse_json_payload(raw)
    assert payload["project_summary"] == "ok"


def test_extract_spec_output_recovers_from_code_fence_with_trailing_data() -> None:
    utterances = [
        {"id": "U1", "speaker": "Client", "text": "Need booking."},
    ]
    raw = (
        "```json\n"
        "{\n"
        "  \"project_summary\": \"Test\",\n"
        "  \"functional_requirements\": [\n"
        "    {\"id\": \"FR1\", \"text\": \"Users can book\", \"priority\": \"high\", \"evidence\": [\"U1\"]}\n"
        "  ],\n"
        "  \"non_functional_requirements\": [],\n"
        "  \"constraints\": [],\n"
        "  \"assumptions\": [],\n"
        "  \"open_questions\": [],\n"
        "  \"follow_up_questions\": []\n"
        "}\n"
        "```\n"
        "note: done"
    )

    result = extract_spec_output(raw_text=raw, utterances=utterances)
    assert result.spec is not None
    assert result.json_parse_ok is True
    assert result.pydantic_validation_ok is True


def test_extract_spec_output_normalizes_key_value_style_items() -> None:
    utterances = [
        {"id": "U1", "speaker": "Client", "text": "Our school clubs need an app to publish weekly meeting schedules."},
        {"id": "U2", "speaker": "Client", "text": "Club leaders should create and update events."},
        {"id": "U3", "speaker": "Client", "text": "Students should join events and receive reminder notifications."},
        {"id": "U4", "speaker": "Client", "text": "It should be easy to use for freshmen."},
        {"id": "U5", "speaker": "Client", "text": "Maybe attendance analytics can be added later."},
        {"id": "U6", "speaker": "PM", "text": "Who is allowed to cancel events?"},
        {"id": "U7", "speaker": "Client", "text": "Only club leaders and the student council should cancel events."},
    ]
    raw = (
        "Output:\n"
        "project_summary: School clubs require an app for publishing and managing weekly meeting schedules.\n"
        "functional_requirements:\n"
        "  - create_event: Club leaders must be able to create and update events.\n"
        "    join_event: Students must be able to join events.\n"
        "    receive_reminder: Students should receive reminder notifications for events.\n"
        "assumptions:\n"
        "  - ease_of_use: The app should be easy to use, particularly for freshmen.\n"
        "open_questions:\n"
        "  - attendance_analytics: Attendance analytics may be considered for future updates.\n"
        "constraints:\n"
        "  - event_cancellation: Only club leaders and the student council are permitted to cancel events.\n"
        "follow_up_questions:\n"
        "  - event_cancellation_process: Clarification needed on the process for event cancellation.\n"
    )

    result = extract_spec_output(raw_text=raw, utterances=utterances)
    assert result.spec is not None
    assert result.json_parse_ok is True
    assert result.pydantic_validation_ok is True
    assert len(result.spec.functional_requirements) >= 2
    assert len(result.spec.constraints) >= 1
    assert len(result.spec.assumptions) >= 1
    assert len(result.spec.open_questions) >= 1
    assert all(item.evidence for item in result.spec.functional_requirements)


def test_extract_spec_output_handles_narrative_plus_fenced_yaml() -> None:
    utterances = [
        {"id": "U1", "speaker": "Client", "text": "Our school clubs need an app to publish weekly meeting schedules."},
        {"id": "U2", "speaker": "Client", "text": "Club leaders should create and update events."},
        {"id": "U3", "speaker": "Client", "text": "Students should join events and receive reminder notifications."},
        {"id": "U4", "speaker": "Client", "text": "It should be easy to use for freshmen."},
        {"id": "U5", "speaker": "Client", "text": "Maybe attendance analytics can be added later."},
        {"id": "U7", "speaker": "Client", "text": "Only club leaders and the student council should cancel events."},
    ]
    raw = (
        "To extract structured requirements, let's reason step by step.\n"
        "Below is the output:\n"
        "```yaml\n"
        "project_summary:\n"
        "  summary: School clubs scheduling app\n"
        "functional_requirements:\n"
        "  meeting_schedule_app:\n"
        "    requirements:\n"
        "      - Publish weekly meeting schedules.\n"
        "      - Club leaders can create and update events.\n"
        "non_functional_requirements:\n"
        "  usability:\n"
        "    requirement: Easy to use for freshmen.\n"
        "assumptions:\n"
        "  - attendance_analytics: Attendance analytics may be added later.\n"
        "constraints:\n"
        "  - event_cancellation: Only club leaders and student council can cancel events.\n"
        "open_questions:\n"
        "  - reminder_channel: Which channel should reminders use?\n"
        "follow_up_questions:\n"
        "  - reminder_channel: Which channel should reminders use?\n"
        "```\n"
    )

    result = extract_spec_output(raw_text=raw, utterances=utterances)
    assert result.spec is not None
    assert result.pydantic_validation_ok is True
    assert len(result.spec.functional_requirements) >= 2
    assert len(result.spec.non_functional_requirements) >= 1
    assert len(result.spec.constraints) >= 1
    assert len(result.spec.assumptions) >= 1


def test_extract_spec_output_normalizes_question_decisions() -> None:
    utterances = [
        {"id": "U1", "speaker": "Client", "text": "It should be simple for freshmen."},
        {"id": "U2", "speaker": "PM", "text": "What response time target do you need?"},
        {"id": "U3", "speaker": "Client", "text": "Under 2 seconds would be good."},
    ]
    raw = (
        "{\n"
        "  \"project_summary\": \"Club app requirements.\",\n"
        "  \"functional_requirements\": [],\n"
        "  \"non_functional_requirements\": [\n"
        "    {\"id\": \"NFR1\", \"text\": \"The app should be simple for freshmen\", \"priority\": \"medium\", \"evidence\": [\"U1\"]}\n"
        "  ],\n"
        "  \"constraints\": [],\n"
        "  \"assumptions\": [],\n"
        "  \"open_questions\": [],\n"
        "  \"follow_up_questions\": [\"What response time target do you need?\"],\n"
        "  \"question_decisions\": [\n"
        "    {\"topic\": \"Response-time target\", \"status\": \"already asked\", \"evidence\": [\"U2\", \"U3\"]},\n"
        "    {\"text\": \"Simplicity requirement\", \"decision\": \"needs_follow_up\", \"suggested_follow_up\": \"How will simplicity be measured?\", \"evidence\": [\"U1\"]}\n"
        "  ]\n"
        "}\n"
    )

    result = extract_spec_output(raw_text=raw, utterances=utterances)
    assert result.spec is not None
    assert len(result.spec.question_decisions) == 2
    assert any(item.decision == "already_asked" for item in result.spec.question_decisions)
    assert any(item.decision == "needs_follow_up" for item in result.spec.question_decisions)
