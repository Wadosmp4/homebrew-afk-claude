"""SDK-owned Claude adapter (U3): sessions the companion itself starts and
drives via the Claude Agent SDK's bidirectional streaming client, with full
remote control (R4, R11) per KTD1 - `can_use_tool` wired to a per-session
pending-approvals queue, not `permission_prompt_tool_name` (the two are
mutually exclusive; only the callback form round-trips a decision from a
remote device).

Every SDK callback normalizes into the shared event model
(companion/adapters/events.py, R6) before the relay/mobile client ever see
it - see that module's docstring for why adapters may not leak SDK-specific
fields beyond what's captured in each event's `data`.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Callable
from time import monotonic
from typing import Any, Optional
from uuid import uuid4

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    PermissionResultAllow,
    PermissionResultDeny,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from .. import auto_approve as approval_policy
from .. import risk_judge
from .events import Event, EventSequencer, truncate_tool_result_content

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ClaudeAgentOptions], Any]


_is_structured_question = approval_policy.is_structured_question


class SDKAdapter:
    """Manages the SDK-owned sessions this companion process started.

    `client_factory` defaults to the real `ClaudeSDKClient` and exists so
    tests can substitute a fake that speaks the same connect/
    receive_messages/interrupt/disconnect interface without spawning a
    real `claude` subprocess.
    """

    def __init__(self, client_factory: Optional[ClientFactory] = None, *, cwd: Optional[str] = None):
        self._client_factory: ClientFactory = client_factory or (lambda options: ClaudeSDKClient(options))
        self._default_cwd = os.path.realpath(cwd) if cwd else None
        self._sessions: dict[str, "_Session"] = {}

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

    async def connect(
        self,
        session_id: str,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        auto_approve: bool = False,
        llm_judge: bool = False,
        resume: Optional[str] = None,
    ) -> None:
        if session_id in self._sessions:
            raise ValueError(f"session already connected: {session_id}")

        session = _Session(session_id)
        # Resolved (not just whatever string the phone sent) so this
        # matches what the real `claude` subprocess itself records as its
        # transcript's own cwd (it resolves symlinks internally - e.g.
        # macOS's /tmp -> /private/tmp - confirmed empirically). Companion's
        # own bookkeeping (recent-projects list, git status scope) and
        # history.py's later cwd-based transcript lookup both need to agree
        # with that resolved form, or a real session can end up invisible
        # in "past sessions for this project" purely from a path-string
        # mismatch, not any real absence of history.
        resolved_cwd = os.path.realpath(cwd) if cwd else self._default_cwd
        session.cwd = resolved_cwd  # U10 (R16): the git-status scope
        # Opt-in per session (defaults off) - see companion/auto_approve.py
        # for the policy this gates and why it's rule-based, not LLM-judged.
        # `llm_judge` is a separate, also-opt-in-defaults-off layer on top
        # (companion/risk_judge.py) - it costs real latency and API/
        # subscription usage per undecided call, unlike the free, instant
        # rule-based policy, so it's never bundled into the base toggle.
        session.auto_approve = auto_approve
        session.llm_judge = llm_judge
        self._sessions[session_id] = session
        # `model=None` leaves ClaudeAgentOptions' own default (whatever the
        # bundled CLI resolves on its own) untouched - the phone's model
        # picker only overrides it when the user actually picked one.
        # `resume` (a session_id, not a bool) reconnects to that session's
        # own existing transcript instead of starting a fresh one - used by
        # daemon.py's _try_resume_sdk_session after a companion restart
        # wiped this adapter's in-memory _sessions. fork_session defaults
        # to False, so the resumed conversation keeps the same session_id
        # rather than being copied to a new one - required here since our
        # own bookkeeping below keys on the session_id the caller passed in.
        options = ClaudeAgentOptions(
            cwd=session.cwd, model=model, can_use_tool=session.can_use_tool, resume=resume
        )
        session.client = self._client_factory(options)

        await session.client.connect(session.prompt_stream())
        # `mode` lets the mobile client (U7) tell an SDK-owned session
        # (full remote control) from an observe-only one (no
        # send_message/interrupt, per R5) without a side-channel - see
        # observe_adapter.py's matching emit for why this one bit matters.
        session.emit(
            "session_started",
            mode="sdk_owned",
            cwd=session.cwd,
            model=model,
            auto_approve=session.auto_approve,
            llm_judge=session.llm_judge,
        )
        session.reader_task = asyncio.create_task(session.read_loop())

    async def send_message(self, session_id: str, text: str) -> None:
        session = self._get(session_id)
        session.emit("user_message", text=text)
        await session.outbound.put(text)

    async def interrupt(self, session_id: str) -> None:
        session = self._get(session_id)
        # A tool call still awaiting an answer (can_use_tool blocked on its
        # own Future - see resolve_permission) never gets resolved by
        # client.interrupt() itself - the CLI then has to abandon a turn
        # that's still mid-await on our own permission callback, which it
        # reports back as a ResultMessage with subtype error_during_execution
        # rather than a clean cancellation. Deny anything pending first so
        # can_use_tool returns normally before the interrupt cuts the turn
        # off, same as if the phone had tapped Deny itself.
        for request_id in list(session.pending):
            session.resolve_permission(request_id, "deny", "Session interrupted before this was answered")
        await session.client.interrupt()
        session.end("interrupted")

    async def compact(self, session_id: str) -> None:
        """Trigger the CLI's own compaction on demand, rather than waiting
        for it to fire automatically once auto_compact_threshold is
        crossed (see _emit_context_usage) - there's no dedicated SDK
        control request for this (only interrupt/set_permission_mode/
        set_model/get_context_usage/etc., checked against
        claude_agent_sdk's internal Query class), so this relies on the
        same path the interactive CLI itself uses: the literal "/compact"
        slash command is intercepted by the CLI before reaching the model,
        same as typing it at the prompt. Goes straight to `outbound`, not
        through send_message, so it doesn't emit a `user_message` event -
        this is a maintenance action, not something the phone said to
        Claude, and shouldn't show up as a fake chat bubble."""
        session = self._get(session_id)
        await session.outbound.put("/compact")

    def set_session_auto_approve(
        self, session_id: str, auto_approve: Optional[bool] = None, llm_judge: Optional[bool] = None
    ) -> bool:
        """Override this one session's own auto_approve/llm_judge directly
        - lets the phone turn auto-approval off (or on) for a session
        already in progress, independent of whatever it was started with.
        `None` for either argument leaves that one field untouched, same
        convention as daemon.py's set_auto_approve_settings. Returns False
        (no-op) for an unknown session_id."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if auto_approve is not None:
            session.auto_approve = auto_approve
        if llm_judge is not None:
            session.llm_judge = llm_judge
        session.emit("session_auto_approve", auto_approve=session.auto_approve, llm_judge=session.llm_judge)
        return True

    async def respond_to_permission(
        self, session_id: str, request_id: str, decision: str, *, message: str = ""
    ) -> None:
        session = self._get(session_id)
        session.resolve_permission(request_id, decision, message)

    async def disconnect(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        if session.reader_task is not None:
            session.reader_task.cancel()
        await session.client.disconnect()
        session.end("disconnected")

    def get_cwd(self, session_id: str) -> Optional[str]:
        """U10 (R16): the working directory git_status/git_diff should
        run against for this session, or None if unknown (never raises -
        the daemon reports "no cwd known" rather than crashing an action)."""
        session = self._sessions.get(session_id)
        return session.cwd if session is not None else None

    def is_active(self, session_id: str) -> Optional[bool]:
        """For the Sessions screen's list_active_sessions snapshot
        (daemon.py) - interrupt() ends a session (_ended=True, a
        session_ended event) without removing it from _sessions (only
        disconnect() does that), so discover_sessions() alone can't tell
        an interrupted/stopped session from one still actually running.
        None for an unknown session_id, distinct from False."""
        session = self._sessions.get(session_id)
        return None if session is None else not session._ended

    def emit_custom(self, session_id: str, type_: str, **data: Any) -> None:
        """U10: lets the daemon inject a result it computed itself (e.g. a
        git_status snapshot) into this session's own event stream, so it
        gets a proper sequenced event_id and rides the same relay
        caching/replay/mobile-listener path as everything else."""
        self._get(session_id).emit(type_, **data)

    async def subscribe(self, session_id: str) -> AsyncIterator[Event]:
        """Yield this session's normalized events as they occur, until the
        session ends or errors (then the generator returns)."""
        session = self._get(session_id)
        while True:
            event = await session.events.get()
            yield event
            if event.type in ("session_ended", "error"):
                return

    def _get(self, session_id: str) -> "_Session":
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown SDK-owned session: {session_id}") from None


class _Session:
    """One SDK-owned session's mutable state: the SDK client, the shared
    event sequencer/queue, the outbound message stream driving the
    bidirectional connection, and the pending-approvals queue KTD1 calls
    for (respond_to_permission resolves a Future here, unblocking the
    can_use_tool callback the SDK is awaiting on)."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sequencer = EventSequencer(session_id)
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.outbound: asyncio.Queue[str] = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}
        self.auto_allowed_tools: set[str] = set()
        self.auto_approve: bool = False  # opt-in per session - set in SDKAdapter.connect
        self.llm_judge: bool = False  # opt-in per session, separate from auto_approve - see connect()
        self.tool_started_at: dict[str, float] = {}
        self.client: Any = None
        self.reader_task: Optional[asyncio.Task] = None
        self.cwd: Optional[str] = None
        self._ended = False

    def emit(self, type_: str, **data: Any) -> Event:
        event = self.sequencer.emit(type_, **data)
        self.events.put_nowait(event)
        return event

    def end(self, reason: str) -> None:
        """Emit `session_ended` at most once - interrupt(), disconnect(),
        and a natural end-of-stream in read_loop can all reach here."""
        if self._ended:
            return
        self._ended = True
        self.emit("session_ended", reason=reason)

    async def prompt_stream(self):
        """The AsyncIterable prompt the SDK's bidirectional streaming
        client consumes (KTD1) - each item is written verbatim as a JSON
        line to the CLI subprocess's stdin, so the shape must match the
        SDK's own streaming user-message format exactly."""
        while True:
            text = await self.outbound.get()
            yield {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": text},
                "parent_tool_use_id": None,
            }

    async def can_use_tool(self, tool_name: str, tool_input: dict, context: Any):
        """The KTD1 callback: block until `respond_to_permission` resolves
        the Future for this request, then translate the decision into the
        SDK's PermissionResultAllow/Deny.

        Once a tool has been allowed once this session, later calls to
        that same tool name skip the prompt entirely (`auto_allowed_tools`)
        - approving "Edit" once shouldn't mean re-approving every
        subsequent edit. A structured question is exempt: each one has
        different content, so "allowed" never applies to it as a category
        the way it does for Bash/Edit/etc.

        When the phone opted this session into policy auto-approval
        (`self.auto_approve`), a narrow rule-based policy
        (companion/auto_approve.py) can also skip the prompt - checked
        first, ahead of auto_allowed_tools, since it's the stricter gate
        and its own denylist must not be bypassable by anything else,
        including the LLM judge below. A policy-approved call still emits
        a permission_request (tagged `auto_approved`) so the phone can
        show it happened, rather than it silently never appearing -
        transparency over invisibility.

        A denylisted call never reaches the judge either - the judge only
        ever sees the gray area (not denylisted, not already rule-
        approved), and a REVIEW/ambiguous/failed verdict falls straight
        through to the normal human prompt below (fail closed)."""
        if not _is_structured_question(tool_input) and self.auto_approve:
            denylisted = approval_policy.is_denylisted(tool_name, tool_input)
            if not denylisted and approval_policy.is_auto_approvable(tool_name, tool_input, cwd=self.cwd):
                self.emit(
                    "permission_request",
                    request_id=str(uuid4()),
                    tool=tool_name,
                    input=tool_input,
                    auto_approved=True,
                    judged_by="policy",
                )
                return PermissionResultAllow()
            if not denylisted and self.llm_judge:
                if await risk_judge.judge_is_safe(tool_name, tool_input, self.cwd):
                    self.emit(
                        "permission_request",
                        request_id=str(uuid4()),
                        tool=tool_name,
                        input=tool_input,
                        auto_approved=True,
                        judged_by="llm",
                    )
                    return PermissionResultAllow()

        if tool_name in self.auto_allowed_tools and not _is_structured_question(tool_input):
            return PermissionResultAllow()

        request_id = getattr(context, "tool_use_id", None) or str(uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[request_id] = future
        self.emit("permission_request", request_id=request_id, tool=tool_name, input=tool_input)

        decision, message = await future
        if decision == "allow" and _is_structured_question(tool_input) and message:
            # A structured-choice tool call can't be "allowed" through in
            # the normal sense: allowing it just lets the real tool run,
            # and this is a headless SDK session with no interactive
            # terminal for it to read an answer from - it would come back
            # empty. The deny-reason message is the only channel this
            # callback has for returning free-form text to Claude, so the
            # phone's chosen option rides back through it instead. Gated on
            # decision == "allow" specifically - a real deny-with-reason on
            # a structured question (message set, decision == "deny") must
            # fall through to the plain deny path below with its own
            # message intact, not get misread as an answered question.
            return PermissionResultDeny(message=f"User answered: {message}")
        if decision == "allow":
            self.auto_allowed_tools.add(tool_name)
            return PermissionResultAllow()
        return PermissionResultDeny(message=message or "denied by user")

    def resolve_permission(self, request_id: str, decision: str, message: str = "") -> None:
        future = self.pending.pop(request_id, None)
        if future is None:
            raise KeyError(f"no pending permission request: {request_id}")
        future.set_result((decision, message))

    async def read_loop(self) -> None:
        try:
            async for message in self.client.receive_messages():
                self._handle_message(message)
                if isinstance(message, ResultMessage):
                    await self._emit_context_usage()
            self.end("completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # SDK/subprocess crash - surface, don't go quiet
            logger.warning("SDK-owned session %s crashed: %s", self.session_id, exc)
            self.emit("error", message=str(exc))

    async def _emit_context_usage(self) -> None:
        """Polled once per completed turn (ResultMessage), not on every
        message - the number is only meaningful once a turn has actually
        finished, matching when the CLI's own /context command would read
        it. Best-effort: a disconnected client, or a test double that
        doesn't implement get_context_usage, must never crash the read
        loop over what's a nice-to-have status update."""
        try:
            usage = await self.client.get_context_usage()
        except Exception as exc:
            logger.debug("context usage poll failed for %s: %s", self.session_id, exc)
            return
        self.emit(
            "context_usage",
            total_tokens=usage.get("totalTokens"),
            max_tokens=usage.get("maxTokens"),
            percentage=usage.get("percentage"),
            is_auto_compact_enabled=usage.get("isAutoCompactEnabled"),
            auto_compact_threshold=usage.get("autoCompactThreshold"),
            model=usage.get("model"),
        )

    def _handle_message(self, message: Any) -> None:
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    self.emit("assistant_message", text=block.text)
                elif isinstance(block, ToolUseBlock):
                    self.tool_started_at[block.id] = monotonic()
                    self.emit("tool_call", tool_use_id=block.id, tool=block.name, input=block.input)
                elif isinstance(block, ToolResultBlock):
                    self._handle_tool_result(block)
        elif isinstance(message, ResultMessage):
            if message.is_error:
                self.emit("error", message=message.result or f"turn ended with error: {message.subtype}")
            else:
                self.emit("waiting_for_input", subtype=message.subtype)

    def _handle_tool_result(self, block: ToolResultBlock) -> None:
        started_at = self.tool_started_at.pop(block.tool_use_id, None)
        duration_ms = (monotonic() - started_at) * 1000 if started_at is not None else None
        self.emit(
            "tool_result",
            tool_use_id=block.tool_use_id,
            content=truncate_tool_result_content(block.content),
            is_error=bool(block.is_error),
            duration_ms=duration_ms,
        )
