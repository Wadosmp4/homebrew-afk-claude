"""Multi-Agent Adapter plan (009), U3: an OpenAI Codex CLI adapter proving
`AdapterProtocol` (companion/adapters/base.py) against a real second agent,
mirroring `SDKAdapter`/`_Session`'s two-level structure as closely as
Codex's actual SDK allows.

Spiked hands-on against the real `openai-codex` PyPI package (v0.147.0)
before writing this, per the plan's own execution note - two findings
that reshaped scope versus the plan's original assumption:

1. The SDK genuinely exists and is well-formed: `AsyncCodex.thread_start()`
   returns an `AsyncThread`; `thread.turn(TextInput(...))` returns an
   `AsyncTurnHandle` whose `.stream()` yields typed `Notification`s live as
   the turn progresses - `ItemStartedNotification`/`ItemCompletedNotification`
   carry a `ThreadItem` discriminated by a `type` string
   ("commandExecution", "mcpToolCall", "fileChange", "agentMessage", ...),
   which maps directly onto this app's existing `tool_call`/`tool_result`/
   `assistant_message` event shapes - no new EVENT_TYPES needed.
2. `ApprovalMode` is a coarse, thread-level, two-value enum
   (`deny_all`/`auto_review`) - there is no per-tool-call interactive
   callback like `claude_agent_sdk`'s `can_use_tool`. A real interactive
   approval loop only exists through Codex's separate hooks/Guardian-review
   system, which this plan explicitly scoped out of this round (that's
   `ObserveAdapter`'s model, not `SDKAdapter`'s). Per product decision:
   this adapter ships without interactive approval - Codex sessions run
   under `ApprovalMode.auto_review` (Codex's own built-in judgment,
   conceptually parallel to this app's `llm_judge`), and
   `respond_to_permission`/`set_session_auto_approve` are
   `UnsupportedOperation`, the same precedent `ObserveAdapter` already
   sets for a capability one adapter genuinely lacks.

`compact` IS implemented for real (`AsyncThread.compact()` exists on the
real SDK) - no reason to leave working functionality unsupported.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any, Callable, Optional

from .base import UnsupportedOperation
from .events import Event, EventSequencer

logger = logging.getLogger(__name__)

ClientFactory = Callable[[], Any]

# Item `type` discriminators this adapter maps onto tool_call/tool_result -
# see the module docstring's spike-finding #1. Anything else (reasoning,
# plan updates, web search, etc.) is not yet surfaced as its own event;
# R3 only requires "at minimum" tool-call/tool-result/permission-request-
# equivalent coverage, not an exhaustive mapping of Codex's full item set.
_TOOL_ITEM_TYPES = ("commandExecution", "mcpToolCall", "fileChange")


def _tool_name_for_item(item: Any) -> str:
    item_type = getattr(item, "type", None)
    if item_type == "commandExecution":
        return "Bash"
    if item_type == "mcpToolCall":
        return getattr(item, "tool", "McpTool")
    if item_type == "fileChange":
        return "ApplyPatch"
    return item_type or "unknown"


def _tool_input_for_item(item: Any) -> dict[str, Any]:
    item_type = getattr(item, "type", None)
    if item_type == "commandExecution":
        return {"command": getattr(item, "command", "")}
    if item_type == "mcpToolCall":
        return {"arguments": getattr(item, "arguments", None)}
    if item_type == "fileChange":
        return {"changes": [str(change) for change in getattr(item, "changes", [])]}
    return {}


def _sandbox_from_wire(sandbox: Optional[str]) -> Optional[Any]:
    """Codex Model & Sandbox Config plan (001), simplify pass: the one
    place the wire-string -> Sandbox enum conversion lives, called from both
    connect() and set_session_sandbox() - previously duplicated in each.
    `sandbox` is a wire string using the SDK's *member name* convention
    ("workspace_write"), not its hyphenated enum *value* ("workspace-write"),
    so this is always a name-based `Sandbox[sandbox]` lookup, never
    `Sandbox(sandbox)` - the latter raises ValueError for every valid wire
    string. Returns None unchanged for a None input (connect()'s own
    "no sandbox requested" case); raises KeyError for an unrecognized name,
    same as a bare `Sandbox[sandbox]` would - each call site decides for
    itself whether that should propagate (connect(), fail-closed) or be
    caught (set_session_sandbox(), fail-soft)."""
    from openai_codex import Sandbox

    return Sandbox[sandbox] if sandbox is not None else None


def _tool_result_for_item(item: Any) -> tuple[str, bool]:
    """Returns (content, is_error), matching sdk_adapter.py's own
    tool_result shape."""
    item_type = getattr(item, "type", None)
    status = getattr(item, "status", None)
    status_value = getattr(status, "value", status)
    is_error = status_value in ("failed", "declined")
    if item_type == "commandExecution":
        return getattr(item, "aggregated_output", None) or "", is_error
    if item_type == "mcpToolCall":
        error = getattr(item, "error", None)
        if error is not None:
            return str(error), True
        return str(getattr(item, "result", "")), is_error
    if item_type == "fileChange":
        return f"{len(getattr(item, 'changes', []))} file(s) changed", is_error
    return "", is_error


class CodexAdapter:
    """Owns every Codex-driven session this companion starts, mirroring
    SDKAdapter's own role for Claude Code sessions."""

    def __init__(self, client_factory: Optional[ClientFactory] = None, *, cwd: Optional[str] = None):
        self._sessions: dict[str, "_CodexSession"] = {}
        self._client_factory = client_factory
        self._cwd = cwd

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

    async def connect(
        self,
        session_id: str,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        sandbox: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """Codex Agent Integration plan (002), U2 (PKTD3): `api_key` is now
        an explicit, caller-resolved value rather than this adapter reading
        OPENAI_API_KEY out of its own process environment - the daemon is
        the one place that knows whether codex_active_account is "personal"
        (pass the configured codex_personal_api_key) or "vscode" (pass
        nothing, the default). Omitting it entirely keeps today's
        implicit-default "vscode" behavior - skip login_api_key() and let
        the ambient CLI/SDK login resolve itself, exactly as before.

        Codex Model & Sandbox Config plan (001), U1: `sandbox` is converted
        via the shared `_sandbox_from_wire()` helper above (name-based
        lookup, not value-based - see its own docstring for why). An
        unrecognized wire string raises (KeyError) here and propagates
        uncaught out of connect() - fails closed at daemon.py's existing
        start_session outer exception handler, no new handling needed in
        this adapter."""
        from openai_codex import ApprovalMode, AsyncCodex

        client = self._client_factory() if self._client_factory is not None else AsyncCodex()
        # Code-review fix: a failure anywhere in this setup sequence must
        # not leak the just-constructed client - nothing else can reach it
        # to close it otherwise, since it's never stored until the
        # _CodexSession below is actually created. This also covers an
        # unrecognized sandbox string's KeyError, which must close the
        # client identically to any other setup failure.
        try:
            sandbox_enum = _sandbox_from_wire(sandbox)
            if api_key:
                await client.login_api_key(api_key)
            resolved_cwd = cwd or self._cwd
            thread = await client.thread_start(
                approval_mode=ApprovalMode.auto_review, cwd=resolved_cwd, model=model, sandbox=sandbox_enum
            )
        except Exception:
            await client.close()
            raise

        session = _CodexSession(session_id, client=client, thread=thread)
        session.cwd = resolved_cwd
        session.model = model
        session.sandbox = sandbox_enum
        self._sessions[session_id] = session
        # KTD4: agent="codex" is the only shape difference from SDKAdapter's
        # own session_started - mobile reads it for KTD5's minimal labeling,
        # defaulting to "claude_code" everywhere else so no existing event
        # payload needs to change shape.
        session.emit("session_started", mode="sdk_owned", agent="codex", cwd=resolved_cwd)

    async def send_message(self, session_id: str, text: str) -> None:
        from openai_codex import TextInput

        session = self._get(session_id)
        session.emit("user_message", text=text)
        # Code-review fix: a still-running previous turn's background task
        # must not be silently overwritten/orphaned by this turn's own
        # current_turn/stream_task assignment below - cancel it first so
        # exactly one stream task is ever tracked per session.
        if session.stream_task is not None and not session.stream_task.done():
            session.stream_task.cancel()
        try:
            turn_handle = await session.thread.turn(
                TextInput(text=text), model=session.model, sandbox=session.sandbox
            )
        except Exception as exc:
            logger.warning("Codex turn failed to start for session %s: %s", session_id, exc)
            session.emit("error", message=str(exc))
            return
        session.current_turn = turn_handle
        session.stream_task = asyncio.create_task(self._drive_turn(session, turn_handle))

    async def interrupt(self, session_id: str) -> None:
        # Code-review fix: lenient lookup, matching SDKAdapter.interrupt()'s
        # own deliberate leniency - daemon.py's unordered per-action
        # asyncio.create_task dispatch can race a "Cancel" against an "End
        # Session" tapped on the same session in close succession, and the
        # session may already be gone from self._sessions by the time this
        # runs. The raising _get() reintroduced the exact KeyError
        # SDKAdapter's own test_interrupt_is_a_clean_noop_when_end_session_
        # already_won_the_race exists to pin.
        session = self._sessions.get(session_id)
        if session is None:
            return
        if session.current_turn is not None:
            await session.current_turn.interrupt()

    async def compact(self, session_id: str) -> None:
        session = self._get(session_id)
        try:
            await session.thread.compact()
        except Exception as exc:
            logger.warning("Codex compact failed for session %s: %s", session_id, exc)
            session.emit("error", message=str(exc))

    async def disconnect(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        if session.current_turn is not None:
            try:
                await session.current_turn.interrupt()
            except Exception:
                logger.exception("failed to interrupt Codex turn on disconnect for session_id=%r", session_id)
        if session.stream_task is not None:
            session.stream_task.cancel()
        # Code-review fix: close() must not skip session.end() - the
        # session is already popped from self._sessions above, so if close()
        # raises here, nothing else will ever emit session_ended, stranding
        # any forwarder/subscriber blocked on session.events.get().
        try:
            await session.client.close()
        except Exception:
            logger.exception("failed to close Codex client on disconnect for session_id=%r", session_id)
        session.end("disconnected")

    async def respond_to_permission(
        self, session_id: str, request_id: str, decision: str, *, message: str = ""
    ) -> UnsupportedOperation:
        # See module docstring's spike-finding #2: no per-call interactive
        # approval exists in the SDK surface this adapter integrates with.
        return UnsupportedOperation(
            operation="respond_to_permission", reason="not supported for Codex sessions"
        )

    def set_session_auto_approve(
        self, session_id: str, auto_approve: Optional[bool] = None, llm_judge: Optional[bool] = None
    ) -> bool:
        # Not part of AdapterProtocol (see base.py's own comment - permission-
        # mode-picker plan removed it from the shared interface once
        # SDKAdapter stopped implementing it), kept here anyway for parity
        # with ObserveAdapter's own method of this name. False is the same
        # "no-op, nothing to apply" signal ObserveAdapter's own
        # set_session_auto_approve already uses for an unknown session_id;
        # there's no per-session auto_approve/llm_judge concept for Codex
        # to actually set (only the coarse, thread-level ApprovalMode).
        return False

    def set_session_model(self, session_id: str, model: str) -> bool:
        """Codex Model & Sandbox Config plan (001), U1: no SDK round-trip -
        purely local state, applied starting with the session's next
        send_message()/turn() call (KTD2). Mirrors sdk_adapter.py's own
        set_session_model emit/return-bool shape, minus its
        asyncio.wait_for/CLIConnectionError handling, which this
        local-state-only path doesn't need."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.model = model
        session.emit("session_model", model=model)
        return True

    def set_session_sandbox(self, session_id: str, sandbox: str) -> bool:
        """Same shape as set_session_model above. Uses the same shared
        `_sandbox_from_wire()` helper connect() does, but an unrecognized
        wire string is caught here (unlike connect()'s deliberate fail-closed
        raise) since this is a live user action against an already-running
        session, not session creation - returns False and leaves the
        session's stored sandbox unchanged rather than raising."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        try:
            sandbox_enum = _sandbox_from_wire(sandbox)
        except KeyError:
            return False
        session.sandbox = sandbox_enum
        session.emit("session_sandbox", sandbox=sandbox)
        return True

    def get_cwd(self, session_id: str) -> Optional[str]:
        session = self._sessions.get(session_id)
        return session.cwd if session is not None else None

    def is_active(self, session_id: str) -> Optional[bool]:
        session = self._sessions.get(session_id)
        return None if session is None else not session._ended

    def emit_custom(self, session_id: str, type_: str, **data: Any) -> None:
        self._get(session_id).emit(type_, **data)

    async def subscribe(self, session_id: str) -> AsyncIterator[Event]:
        session = self._get(session_id)
        while True:
            event = await session.events.get()
            yield event
            if event.type in ("session_ended", "error"):
                return

    def _get(self, session_id: str) -> "_CodexSession":
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"no such Codex session: {session_id}")
        return session

    async def _drive_turn(self, session: "_CodexSession", turn_handle: Any) -> None:
        """Consumes the turn's live notification stream and translates each
        item into the shared Event model, mirroring how SDKAdapter's
        read_loop translates claude_agent_sdk messages. Runs as a background
        task (send_message returns as soon as the turn starts, not once it
        finishes) so the phone sees tool calls/results as they happen."""
        seen_item_ids: set[str] = set()
        try:
            async for notification in turn_handle.stream():
                await self._handle_notification(session, notification, seen_item_ids)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Codex turn stream crashed for session %s: %s", session.session_id, exc)
            session.emit("error", message=str(exc))

    async def _handle_notification(self, session: "_CodexSession", notification: Any, seen_item_ids: set[str]) -> None:
        from openai_codex.models import (
            ErrorNotification,
            ItemCompletedNotification,
            ItemStartedNotification,
            TurnCompletedNotification,
        )

        if isinstance(notification, TurnCompletedNotification):
            # Code-review fix (P0): the turn finishing was previously never
            # signaled at all - mirrors SDKAdapter's own ResultMessage ->
            # waiting_for_input emit (sdk_adapter.py), which is what tells
            # the phone "the agent is done, your turn" and clears its own
            # "thinking" indicator. Without this, every ordinary Codex turn
            # that completes without a fresh tool call as its last item left
            # the phone waiting forever with no signal the turn ever ended.
            session.emit("waiting_for_input", subtype="codex_turn_completed")
            return

        if isinstance(notification, ItemStartedNotification):
            item = notification.item.root if hasattr(notification.item, "root") else notification.item
            item_type = getattr(item, "type", None)
            item_id = getattr(item, "id", None)
            if item_type in _TOOL_ITEM_TYPES and item_id not in seen_item_ids:
                seen_item_ids.add(item_id)
                session.emit(
                    "tool_call",
                    tool_use_id=item_id,
                    tool=_tool_name_for_item(item),
                    input=_tool_input_for_item(item),
                )
            return

        if isinstance(notification, ItemCompletedNotification):
            item = notification.item.root if hasattr(notification.item, "root") else notification.item
            item_type = getattr(item, "type", None)
            item_id = getattr(item, "id", None)
            if item_type in _TOOL_ITEM_TYPES:
                content, is_error = _tool_result_for_item(item)
                session.emit("tool_result", tool_use_id=item_id, content=content, is_error=is_error)
            elif item_type == "agentMessage":
                session.emit("assistant_message", text=getattr(item, "text", ""))
            return

        if isinstance(notification, ErrorNotification):
            session.emit("error", message=getattr(notification.error, "message", "Codex turn failed"))
            return


class _CodexSession:
    def __init__(self, session_id: str, *, client: Any, thread: Any):
        self.session_id = session_id
        self.client = client
        self.thread = thread
        self.sequencer = EventSequencer(session_id)
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.cwd: Optional[str] = None
        self.model: Optional[str] = None
        self.sandbox: Optional[Any] = None
        self.current_turn: Optional[Any] = None
        self.stream_task: Optional[asyncio.Task] = None
        self._ended = False

    def emit(self, type_: str, **data: Any) -> Event:
        event = self.sequencer.emit(type_, **data)
        self.events.put_nowait(event)
        return event

    def end(self, reason: str) -> None:
        if self._ended:
            return
        self._ended = True
        self.emit("session_ended", reason=reason)
