import pytest

from companion.adapters.events import (
    MAX_TOOL_RESULT_CONTENT_CHARS,
    Event,
    EventSequencer,
    truncate_tool_result_content,
)


def test_sequencer_assigns_monotonic_ids_per_session():
    seq = EventSequencer("session-1")
    e0 = seq.emit("session_started")
    e1 = seq.emit("assistant_message", text="hi")

    assert e0.event_id == 0
    assert e1.event_id == 1
    assert e0.session_id == e1.session_id == "session-1"


def test_two_sequencers_do_not_share_counters():
    a = EventSequencer("session-a")
    b = EventSequencer("session-b")
    assert a.emit("session_started").event_id == 0
    assert a.emit("session_started").event_id == 1
    assert b.emit("session_started").event_id == 0


def test_emit_rejects_unknown_event_type():
    seq = EventSequencer("session-1")
    with pytest.raises(ValueError):
        seq.emit("not_a_real_type")


def test_event_roundtrips_through_dict():
    seq = EventSequencer("session-1")
    event = seq.emit("tool_call", tool="Read", input={"path": "x.py"})

    restored = Event.from_dict(event.to_dict())

    assert restored == event


# --- U9: tool_result content truncation ----------------------------------

def test_truncate_short_string_content_is_unchanged():
    assert truncate_tool_result_content("short output") == "short output"


def test_truncate_long_string_content_is_marked():
    long_text = "x" * (MAX_TOOL_RESULT_CONTENT_CHARS + 500)

    result = truncate_tool_result_content(long_text)

    assert len(result) < len(long_text)
    assert result.startswith("x" * MAX_TOOL_RESULT_CONTENT_CHARS)
    assert "truncated" in result
    assert str(len(long_text)) in result  # original size is disclosed


def test_truncate_list_of_blocks_only_truncates_long_text_blocks():
    blocks = [
        {"type": "text", "text": "short"},
        {"type": "text", "text": "y" * (MAX_TOOL_RESULT_CONTENT_CHARS + 500)},
        {"type": "image", "data": "base64..."},
    ]

    result = truncate_tool_result_content(blocks)

    assert result[0] == {"type": "text", "text": "short"}
    assert "truncated" in result[1]["text"]
    assert result[2] == {"type": "image", "data": "base64..."}


def test_truncate_non_string_non_list_content_passes_through():
    structured = {"exit_code": 0}
    assert truncate_tool_result_content(structured) is structured


def test_truncate_none_content_passes_through():
    assert truncate_tool_result_content(None) is None
