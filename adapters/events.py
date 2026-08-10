"""The shared normalized event model both adapters (U3 SDK-owned,
U4 observe-only) produce, per R6: the relay (U5) and mobile client are
adapter-agnostic - they only ever see this one shape, never SDK- or
hook-specific fields.

Event types (R6): assistant_message, tool_call, tool_result,
permission_request, user_message, and the lifecycle events
session_started, session_ended, waiting_for_input, error.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

EVENT_TYPES = frozenset(
    {
        "assistant_message",
        "tool_call",
        "tool_result",
        "permission_request",
        "user_message",
        "session_started",
        "session_ended",
        "waiting_for_input",
        "error",
    }
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Event:
    session_id: str
    event_id: int
    type: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "event_id": self.event_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Event":
        return cls(
            session_id=raw["session_id"],
            event_id=raw["event_id"],
            type=raw["type"],
            timestamp=raw["timestamp"],
            data=raw.get("data", {}),
        )


# U9 (R14): a very large tool result (e.g. a long test run's output) is
# truncated with an explicit marker rather than silently dropped or
# blowing up the relay's bounded event-replay cache (KTD3/KTD4) - both
# adapters call this before emitting `tool_result`.
MAX_TOOL_RESULT_CONTENT_CHARS = 8000


def truncate_tool_result_content(content: Any) -> Any:
    """Truncate a tool_result's `content` if it's a long string, or a list
    of content blocks whose text is long. Non-string/list content (already
    small/structured data) passes through unchanged - there's nothing
    generically safe to truncate there."""
    if isinstance(content, str):
        return _truncate_text(content)
    if isinstance(content, list):
        return [_truncate_block(block) for block in content]
    return content


def _truncate_text(text: str) -> str:
    if len(text) <= MAX_TOOL_RESULT_CONTENT_CHARS:
        return text
    return text[:MAX_TOOL_RESULT_CONTENT_CHARS] + f"\n… [truncated, {len(text)} chars total]"


def _truncate_block(block: Any) -> Any:
    if isinstance(block, dict) and isinstance(block.get("text"), str) and len(block["text"]) > MAX_TOOL_RESULT_CONTENT_CHARS:
        return {**block, "text": _truncate_text(block["text"])}
    return block


class EventSequencer:
    """Assigns monotonic per-session `event_id`s (U5's Approach: "Companion
    assigns event_id per session (monotonic)"). One instance per session -
    adapters must not share a sequencer across sessions or ids collide."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._next_id = 0

    def emit(self, type_: str, **data: Any) -> Event:
        if type_ not in EVENT_TYPES:
            raise ValueError(f"unknown event type: {type_!r}")
        event = Event(
            session_id=self.session_id,
            event_id=self._next_id,
            type=type_,
            timestamp=_now_iso(),
            data=data,
        )
        self._next_id += 1
        return event
