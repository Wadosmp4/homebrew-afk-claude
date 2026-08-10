import pytest

from companion.adapters.events import Event, EventSequencer


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
