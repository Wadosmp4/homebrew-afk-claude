"""The shared normalized event model both adapters (U3 SDK-owned,
U4 observe-only) produce, per R6: the relay (U5) and mobile client are
adapter-agnostic - they only ever see this one shape, never SDK- or
hook-specific fields.

Event types (R6): assistant_message, tool_call, tool_result,
permission_request, user_message, and the lifecycle events
session_started, session_ended, waiting_for_input, error. See EVENT_TYPES
below for the full set including later additions.
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
        # U10 (R16): a git_status/git_diff request's result rides the same
        # per-session event stream as everything else (same relay caching,
        # replay, and mobile event listener) rather than a separate
        # request/response subprotocol - see companion/daemon.py's
        # _handle_action for how a request becomes one of these.
        "git_status",
        "git_diff",
        # AFK Digest plan (U2): an on-demand "Catch me up" request's result
        # rides the same per-session event stream, same shape as
        # git_status/git_diff above - see daemon.py's _handle_session_digest.
        "session_digest",
        # Sessions-screen picker (daemon.py's _handle_list_projects): not
        # per-session like the rest of this registry - sent on a fixed
        # sentinel session_id ("_projects") since it isn't scoped to one.
        "project_list",
        # Read-only past-session browsing (daemon.py's
        # _handle_list_project_sessions/_handle_read_session_history,
        # companion/history.py) - also sentinel session_ids, not real ones.
        "session_history_list",
        "session_history",
        # Which Claude Code clients the observe-only watcher surfaces
        # (daemon.py's _handle_get_observe_settings/_handle_set_observe_entrypoints,
        # ObserveAdapter.required_entrypoints) - also a sentinel session_id
        # ("_observe_settings").
        "observe_settings",
        # Per-session auto_approve/llm_judge override confirmation
        # (ObserveAdapter.set_session_auto_approve) - a real, scoped
        # session_id, emitted whenever a phone-issued override changes a
        # running session's own state independent of the adapter-wide
        # default. Observed sessions only as of the permission-mode-picker
        # plan's U3 - SDKAdapter's own set_session_auto_approve was
        # replaced there by set_session_permission_mode/set_session_model
        # below, since permission_mode/model are native-SDK concepts with
        # no auto_approve/llm_judge equivalent left to override.
        "session_auto_approve",
        # SDK-owned only (permission-mode-picker plan U3,
        # SDKAdapter.set_session_permission_mode/set_session_model): a
        # phone-issued live, no-reconnect change to a running session's
        # permission mode or model, applied via the SDK's own
        # control-request mechanism (session.client.set_permission_mode/
        # set_model) - these confirm it took effect, mirroring
        # session_auto_approve's shape for the SDK-owned equivalents.
        "session_permission_mode",
        "session_model",
        # SDK-owned only (SDKAdapter._Session._emit_context_usage): a
        # ClaudeSDKClient.get_context_usage() snapshot polled once per
        # completed turn - matches what the CLI's own /context command
        # shows. ObserveAdapter never emits this - an observed session
        # isn't driven by our own client, so we have no equivalent to poll.
        "context_usage",
        # SDK-owned only (SDKAdapter._Session._handle_rate_limit): pushed
        # by the CLI whenever a rate-limit window's status changes, not on
        # a fixed poll like context_usage - see claude_agent_sdk's
        # RateLimitEvent. ObserveAdapter never emits this, same reasoning
        # as context_usage above.
        "rate_limit",
        # Connection-resilience plan U1: emitted once a permission_request
        # is resolved - manually (respond_to_permission) or via an
        # interrupt's deny-pending loop - so resolution is a durable,
        # replayable fact rather than only held in the answering device's
        # own memory. Mirrors "permission_request"'s own auto_approved=True
        # self-documenting shape for the manual/interrupt-denied path.
        "permission_resolved",
        # Risk Explanation plan U1: emitted asynchronously, shortly after a
        # permission_request that reached the human, correlating back to it
        # by request_id the same way permission_resolved does above. Purely
        # additive - explanation: None means no explanation is available
        # (no API key configured, or the call failed), not an error.
        "permission_risk_explanation",
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
