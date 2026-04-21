from app.parser import parse_conversation_text


def test_parse_conversation_with_speakers_and_unknown_lines() -> None:
    text = """
    Client: We need a booking page.
    This should work for mobile too.
    PM: Any deadline?
    Around next month.
    """

    utterances = parse_conversation_text(text)

    assert [item.id for item in utterances] == ["U1", "U2", "U3", "U4"]
    assert utterances[0].speaker == "Client"
    assert utterances[0].text == "We need a booking page."
    assert utterances[1].speaker == "Client"
    assert utterances[1].text == "This should work for mobile too."
    assert utterances[2].speaker == "PM"
    assert utterances[3].speaker == "Client"
    assert utterances[3].text == "Around next month."


def test_parse_raises_for_empty_text() -> None:
    try:
        parse_conversation_text("   \n\n")
    except ValueError as exc:
        assert "No utterances" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty conversation")


def test_parse_infers_speakers_without_prefixes() -> None:
    text = """
    We need an app that publishes weekly meeting schedules.
    What should happen when two events overlap?
    Club leaders should edit events and students can join.
    Around next month would be ideal.
    """

    utterances = parse_conversation_text(text)

    assert [item.speaker for item in utterances] == ["Client", "PM", "Client", "Client"]


def test_parse_maps_developer_prefix_to_pm() -> None:
    text = """
    Developer: What acceptance criteria should we use?
    Client: It should be easy for freshmen.
    """
    utterances = parse_conversation_text(text)
    assert utterances[0].speaker == "PM"
    assert utterances[1].speaker == "Client"


def test_parse_maps_korean_developer_prefix_to_pm() -> None:
    text = """
    개발자: 취소 권한은 누가 가져야 하나요?
    고객: 동아리장과 학생회만 취소할 수 있어요.
    """
    utterances = parse_conversation_text(text)
    assert utterances[0].speaker == "PM"
    assert utterances[1].speaker == "Client"


def test_parse_infers_pm_guidance_line_in_unlabeled_dialogue() -> None:
    text = """
    Our school clubs need an app to publish weekly meeting schedules.
    Maybe attendance analytics can be added later.
    You should give us specific deadline for attendance analytics.
    I think it's okay for May 3rd.
    """
    utterances = parse_conversation_text(text)
    assert [item.speaker for item in utterances] == ["Client", "Client", "PM", "Client"]
