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
import contextlib
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
    CLIConnectionError,
    PermissionResultAllow,
    PermissionResultDeny,
    RateLimitEvent,
    ResultMessage,
    TaskNotificationMessage,
    TaskProgressMessage,
    TaskStartedMessage,
    TaskUpdatedMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from ..auto_approve import is_structured_question as _is_structured_question
from ..rate_limit_estimator import RateLimitEstimator
from .events import Event, EventSequencer, truncate_tool_result_content

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ClaudeAgentOptions], Any]

# code-review finding: set_session_permission_mode/set_session_model's own
# control-request awaits had no bound - an unresponsive CLI subprocess would
# hang the dispatching task forever, with no confirmation event and no
# registry write, leaving the phone's own client-side optimistic-UI timeout
# as the only thing that ever notices. Bounded the same way daemon.py's own
# comparable awaits already are (e.g. _handle_session_start's setup-token
# wait).
_LIVE_CONTROL_REQUEST_TIMEOUT_SECONDS = 10.0


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
        # Bug fix (user report): one estimator for this whole adapter, not
        # per-session - an account's rate limit is shared across every
        # session this companion happens to be driving concurrently, so the
        # token counter and per-window calibration must be too. See
        # rate_limit_estimator.py's own module docstring for why this
        # exists and what it can't do.
        self._rate_limit_estimator = RateLimitEstimator()

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

    async def connect(
        self,
        session_id: str,
        *,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        resume: Optional[str] = None,
        cli_path: Optional[str] = None,
        # Code-review fix (P3): a mutable {} default is created once at
        # function-definition time and shared across every call that omits
        # this argument - latent today (daemon.py's own callers always
        # pass an explicit fresh dict), but a landmine for any future/test
        # caller that doesn't. `None` (materialized to a fresh {} below)
        # is the safe sentinel instead.
        cli_env: Optional[dict[str, str]] = None,
    ) -> None:
        # Code-review fix (P1): a crashed session (read_loop's own except
        # branch, see its own comment) now marks itself `_ended` the same
        # way a normal completion already does, so this guard must allow
        # reconnecting under the same session_id once the previous
        # occupant is actually done - only a still-live entry blocks a
        # fresh connect(). Without this, a crash permanently blocked any
        # future reconnect attempt for that session_id until a full
        # daemon restart wiped this adapter's in-memory _sessions.
        existing = self._sessions.get(session_id)
        if existing is not None and not existing._ended:
            raise ValueError(f"session already connected: {session_id}")

        cli_env = cli_env if cli_env is not None else {}
        session = _Session(session_id, rate_limit_estimator=self._rate_limit_estimator)
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
        self._sessions[session_id] = session
        # `model=None`/`permission_mode=None` both leave ClaudeAgentOptions'
        # own default (whatever the bundled CLI resolves on its own)
        # untouched - the phone's model/permission-mode pickers only
        # override either when the user actually picked one. Permission-
        # mode-picker plan (R2/R3): this used to also be where
        # auto_approve/llm_judge/risk_judge_use_api were snapshotted onto
        # the session (a rule-based denylist plus an opt-in LLM judge,
        # both consulted from can_use_tool) - can_use_tool no longer
        # checks any app-side policy at all (U1), so a session's chosen
        # native permission_mode is the sole governor of tool-call
        # approval now, with nothing left here to snapshot.
        # `resume` (a session_id, not a bool) reconnects to that session's
        # own existing transcript instead of starting a fresh one - used by
        # daemon.py's _try_resume_sdk_session after a companion restart
        # wiped this adapter's in-memory _sessions. fork_session defaults
        # to False, so the resumed conversation keeps the same session_id
        # rather than being copied to a new one - required here since our
        # own bookkeeping below keys on the session_id the caller passed in.
        # `cli_env` must stay dict-shaped here, never collapsed to None -
        # ClaudeAgentOptions.env is dict-typed (default_factory=dict), not
        # Optional, on the SDK side, and the real subprocess transport
        # unconditionally dict-unpacks it (**self._options.env) when
        # building the child process's environment. Unlike `model=None`
        # (a legitimate "leave the SDK default alone" sentinel for an
        # Optional[str] field), env=None would raise a TypeError on every
        # single session start, not just when a CLI profile is configured.
        options = ClaudeAgentOptions(
            cwd=session.cwd,
            model=model,
            permission_mode=permission_mode,
            can_use_tool=session.can_use_tool,
            resume=resume,
            # Forces the CLI subprocess to file its transcript under this
            # exact session_id (see subprocess_cli.py's --session-id flag) -
            # without this, a fresh session's transcript is written under an
            # id the CLI invented itself, and every transcript lookup keyed
            # on this session_id (history.py) silently finds nothing.
            # NOT combined with resume: ClaudeAgentOptions.session_id's own
            # docstring forbids that unless fork_session is also set, and
            # fork_session forks onto a *new* id - the opposite of what a
            # resume needs. resume=resume alone already resolves correctly
            # once a fresh session's id lines up with its transcript per
            # this same fix.
            session_id=session_id if resume is None else None,
            cli_path=cli_path,
            env=cli_env,
        )
        session.client = self._client_factory(options)

        await session.client.connect(session.prompt_stream())
        # Code-review fix (P3, narrow race): a concurrently-dispatched
        # disconnect() (daemon.py dispatches every phone action as its own
        # unordered asyncio.create_task, same race set_session_permission_
        # mode/set_session_model/stop_task already guard against) can pop
        # this exact session back out of _sessions while the await above
        # was still in flight. Left unchecked, connect() would still
        # finish anyway: emitting session_started and creating a
        # reader_task for a session nothing else references any more -
        # untracked, never cancelled, and leaked for the life of the
        # process. Back out (and make sure the now-connected client
        # doesn't itself leak) rather than resurrecting it.
        if self._sessions.get(session_id) is not session:
            logger.info(
                "connect() for %s lost a race with a concurrent disconnect() - backing out", session_id
            )
            with contextlib.suppress(Exception):
                await session.client.disconnect()
            return
        # `mode` lets the mobile client (U7) tell an SDK-owned session
        # (full remote control) from an observe-only one (no
        # send_message/interrupt, per R5) without a side-channel - see
        # observe_adapter.py's matching emit for why this one bit matters.
        session.emit(
            "session_started",
            mode="sdk_owned",
            cwd=session.cwd,
            model=model,
            permission_mode=permission_mode,
        )
        session.reader_task = asyncio.create_task(session.read_loop())

    async def send_message(self, session_id: str, text: str) -> None:
        session = self._get(session_id)
        session.emit("user_message", text=text)
        await session.outbound.put(text)

    async def interrupt(self, session_id: str) -> None:
        """Cancels the in-flight turn only - the session itself stays open
        and usable afterward (U5/R7: "Cancel" is turn-only; ending the
        session entirely is a separate, explicit action - see disconnect()
        below)."""
        session = self._sessions.get(session_id)
        if session is None:
            # daemon.py dispatches every phone action as its own unordered
            # asyncio.create_task, so a concurrently-dispatched End Session
            # (disconnect(), which pops from _sessions) can beat this
            # Cancel here - the session is already gone, so there's no
            # turn left to cancel. Using _get() here would raise KeyError,
            # caught only by daemon.py's generic action-failure logger with
            # no signal back to the phone that tapped Cancel.
            return
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

    async def set_session_permission_mode(self, session_id: str, mode: str) -> bool:
        """R8/R9 (permission-mode-picker plan, U3): change this one
        session's native permission mode live, mid-session, via the SDK's
        own control-request mechanism - no reconnect, no fresh
        ClaudeAgentOptions. Replaces the old set_session_auto_approve,
        which this plan's U1/U2 had already made inert for tool-call
        behavior (can_use_tool stopped reading auto_approve/llm_judge
        entirely, and connect() stopped snapshotting either onto the
        session). Returns False (no-op) for an unknown session_id, same
        convention as every other method here.

        A concurrently-dispatched end_session (daemon.py dispatches every
        phone action as its own unordered asyncio.create_task) can
        disconnect this session's client between our own `session_id in
        _sessions` lookup and this control request actually reaching the
        subprocess - the SDK surfaces that as CLIConnectionError, not a
        KeyError, since the session_id lookup above already succeeded.
        Caught here (not left to propagate into daemon.py's generic
        per-action exception logger, which would log an indistinguishable
        crash) and logged identifiably so a real crash still stands out."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        try:
            await asyncio.wait_for(
                session.client.set_permission_mode(mode), timeout=_LIVE_CONTROL_REQUEST_TIMEOUT_SECONDS
            )
        except CLIConnectionError:
            logger.info(
                "set_session_permission_mode for %s raced a concurrent end_session and lost - client already disconnected",
                session_id,
            )
            return False
        except asyncio.TimeoutError:
            logger.warning("set_session_permission_mode for %s timed out waiting on the CLI subprocess", session_id)
            return False
        session.emit("session_permission_mode", permission_mode=mode)
        return True

    async def set_session_model(self, session_id: str, model: str) -> bool:
        """R10: model's counterpart to set_session_permission_mode above -
        same live control-request mechanism (no reconnect), same
        unknown-session/racing-end_session handling. See that method's
        docstring for both."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        try:
            await asyncio.wait_for(session.client.set_model(model), timeout=_LIVE_CONTROL_REQUEST_TIMEOUT_SECONDS)
        except CLIConnectionError:
            logger.info(
                "set_session_model for %s raced a concurrent end_session and lost - client already disconnected",
                session_id,
            )
            return False
        except asyncio.TimeoutError:
            logger.warning("set_session_model for %s timed out waiting on the CLI subprocess", session_id)
            return False
        session.emit("session_model", model=model)
        return True

    async def stop_task(self, session_id: str, task_id: str) -> bool:
        """Bug fix (user report): stops a background subagent this session
        dispatched via the "Task" tool - claude_agent_sdk.ClaudeSDKClient's
        own stop_task() control request, same live-control-request shape
        as set_session_permission_mode/set_session_model above (bounded
        wait, same unknown-session/racing-end_session handling, see that
        method's docstring for both).

        Does not emit anything itself - per stop_task()'s own docstring, a
        task_notification message (TaskNotificationMessage, status
        'stopped') follows asynchronously through the normal read_loop once
        the CLI subprocess actually stops it, which _handle_message already
        turns into a task_update event above. Returns False (no-op, same
        convention as every other method here) for an unknown session_id or
        an unknown/already-finished task_id - claude_agent_sdk's own
        stop_task() raises neither in that case (confirmed via its
        docstring/signature), but nothing here assumes what it does when
        the task_id is already terminal beyond "don't crash the read loop
        over it"."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        try:
            await asyncio.wait_for(session.client.stop_task(task_id), timeout=_LIVE_CONTROL_REQUEST_TIMEOUT_SECONDS)
        except CLIConnectionError:
            logger.info(
                "stop_task for %s/%s raced a concurrent end_session and lost - client already disconnected",
                session_id,
                task_id,
            )
            return False
        except asyncio.TimeoutError:
            logger.warning("stop_task for %s/%s timed out waiting on the CLI subprocess", session_id, task_id)
            return False
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
        # F3/R5: ending a session must durably deny any permission request
        # still open on it, same as interrupt()'s own deny-pending loop
        # above - otherwise a phone that replays this session's history
        # later sees a permission_request with no matching resolution and
        # reads it as still pending forever.
        for request_id in list(session.pending):
            session.resolve_permission(request_id, "deny", "Session ended before this was answered")
        if session.reader_task is not None:
            session.reader_task.cancel()
            # cancel() only *schedules* cancellation - without awaiting the
            # task afterward, disconnect() can return (and a test's event
            # loop can close) while read_loop() is still pending, which
            # asyncio reports as "Task was destroyed but it is pending!"
            # and, if it was itself awaiting a queue .get(), sometimes a
            # "RuntimeError: Event loop is closed" from that coroutine
            # being garbage-collected after the loop is gone.
            with contextlib.suppress(asyncio.CancelledError):
                await session.reader_task
        await session.client.disconnect()
        session.end("disconnected")

    async def disconnect_all(self) -> None:
        """Daemon shutdown (companion/daemon.py's run()): every still-active
        session's reader_task must be cancelled and awaited before the
        process exits, or it's leaked - orphaned, pending, and destroyed by
        the interpreter with no one left to await it. discover_sessions()'s
        own snapshot is taken once up front since disconnect() mutates
        _sessions as it goes."""
        for session_id in self.discover_sessions():
            await self.disconnect(session_id)

    def get_cwd(self, session_id: str) -> Optional[str]:
        """U10 (R16): the working directory git_status/git_diff should
        run against for this session, or None if unknown (never raises -
        the daemon reports "no cwd known" rather than crashing an action)."""
        session = self._sessions.get(session_id)
        return session.cwd if session is not None else None

    def is_active(self, session_id: str) -> Optional[bool]:
        """For the Sessions screen's list_active_sessions snapshot
        (daemon.py) - disconnect() ends a session (_ended=True, a
        session_ended event) and removes it from _sessions; interrupt()
        (U5: turn-only) leaves both untouched, so a cancelled turn still
        reads as active here. None for an unknown session_id, distinct
        from False."""
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

    def __init__(self, session_id: str, *, rate_limit_estimator: RateLimitEstimator):
        self.session_id = session_id
        self.sequencer = EventSequencer(session_id)
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.outbound: asyncio.Queue[str] = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}
        self.auto_allowed_tools: set[str] = set()
        self.tool_started_at: dict[str, float] = {}
        # Bug fix (user report): shared across every session this adapter
        # drives (see SDKAdapter.__init__'s own comment) - not this
        # session's own private tracker.
        self.rate_limit_estimator = rate_limit_estimator
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

        Permission-mode-picker plan (U1): this callback used to also check
        a rule-based denylist/allowlist policy (companion/auto_approve.py)
        and an opt-in LLM judge (companion/risk_judge.judge_is_safe)
        before falling through to the human prompt below, plus fire a
        fire-and-forget Risk Explanation call alongside it. All three are
        removed from this path - a session's own chosen native permission
        mode is now the sole governor of tool-call approval for an
        SDK-owned session, with nothing underneath it, including under a
        fully permissive mode. Neither `auto_approve.py` nor
        `risk_judge.judge_is_safe` is deleted from the codebase:
        observe_adapter.py's own, entirely separate mechanism for observed
        (non-SDK-owned) sessions still calls both directly."""
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
        # U1 (connection-resilience plan): the one call site every
        # resolution path shares - respond_to_permission and interrupt()'s
        # deny-pending loop both land here - so both get durable recording
        # for free, symmetric with permission_request's own auto_approved
        # self-documenting shape.
        self.emit("permission_resolved", request_id=request_id, decision=decision, message=message)

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
            # Code-review fix (P1): previously only the normal-completion
            # path below called self.end("completed") - this branch left
            # `_ended` False forever, so is_active() reported True forever,
            # the session stayed in SDKAdapter._sessions, and any reconnect
            # attempt for this session_id was permanently blocked by
            # connect()'s "already connected" guard until a full daemon
            # restart. "crashed" is a distinct reason (mirroring
            # disconnect()'s own "disconnected") so a durable session_ended
            # event still records how this session actually stopped.
            self.end("crashed")

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
            # Bug fix (user report): this turn's own token usage, recorded
            # regardless of is_error - a failed turn still spent tokens
            # getting there. See rate_limit_estimator.py's own docstring
            # for why this exists.
            self.rate_limit_estimator.record_turn_usage(message.model_usage)
            self._emit_rate_limit_estimates()
        elif isinstance(message, RateLimitEvent):
            self._handle_rate_limit(message)
        # Bug fix (user report): a background subagent dispatched via the
        # "Task" tool (claude_agent_sdk.ToolUseBlock with name "Task",
        # already rendered as "Running subagent" - see
        # mobile/lib/toolDescriptions.ts) keeps running after the
        # dispatching turn's own ResultMessage/waiting_for_input fires -
        # the two are unrelated: one SDK query() call completing has
        # nothing to do with a detached subagent it kicked off still being
        # active. These four message types (all claude_agent_sdk.SystemMessage
        # subclasses, so must be checked as their own isinstance branches,
        # not folded into a generic SystemMessage case that doesn't exist
        # here) are the CLI's own lifecycle reporting for that subagent -
        # forwarding them lets the phone track "is a task I dispatched
        # still running" independently of this turn's own waiting_for_input,
        # and lets stop_task() below give it something real to cancel.
        elif isinstance(message, TaskStartedMessage):
            self.emit(
                "task_update",
                phase="started",
                task_id=message.task_id,
                description=message.description,
                tool_use_id=message.tool_use_id,
            )
        elif isinstance(message, TaskProgressMessage):
            self.emit(
                "task_update",
                phase="progress",
                task_id=message.task_id,
                description=message.description,
                tool_use_id=message.tool_use_id,
            )
            # Bug fix (user report): a periodic progress tick is exactly
            # when a token-hungry subagent's usage would otherwise go
            # unnoticed until it finally finishes - record and re-estimate
            # here too, not only at the notification below.
            self.rate_limit_estimator.record_task_usage(message.task_id, message.usage)
            self._emit_rate_limit_estimates()
        elif isinstance(message, TaskUpdatedMessage):
            self.emit(
                "task_update",
                phase="updated",
                task_id=message.task_id,
                status=message.status,
            )
        elif isinstance(message, TaskNotificationMessage):
            # The authoritative "this task is over" signal - stop_task()'s
            # own docstring says a stopped task always reports here with
            # status 'stopped'; a naturally-finished one reports 'completed'
            # or 'failed'. All three are terminal (claude_agent_sdk's own
            # TERMINAL_TASK_STATUSES agrees, though that frozenset also
            # includes TaskUpdatedMessage's 'killed', not reachable from
            # this message type).
            self.emit(
                "task_update",
                phase="notification",
                task_id=message.task_id,
                status=message.status,
                summary=message.summary,
                tool_use_id=message.tool_use_id,
            )
            # Bug fix (user report): this task's final usage total - see
            # record_task_usage's own docstring for why only the delta
            # since the last progress tick (if any) actually gets added.
            self.rate_limit_estimator.record_task_usage(message.task_id, message.usage)
            self._emit_rate_limit_estimates()

    def _emit_rate_limit_estimates(self) -> None:
        """Bug fix (user report): a fresh guess for every window with
        enough real history to extrapolate from, sent right after new
        token usage was just recorded - see rate_limit_estimator.py's own
        module docstring for what this can and can't do. A window with no
        real reading yet, or only one (nothing to calibrate a slope from),
        is silently skipped rather than guessing at it - see estimate()'s
        own docstring for why None isn't "assume zero"."""
        for rate_limit_type in self.rate_limit_estimator.calibrated_types():
            utilization = self.rate_limit_estimator.estimate(rate_limit_type)
            if utilization is None:
                continue
            self.emit("rate_limit_estimate", rate_limit_type=rate_limit_type, utilization=utilization)

    def _handle_rate_limit(self, message: RateLimitEvent) -> None:
        # Pushed by the CLI only on a status transition (e.g.
        # allowed -> allowed_warning), one window (five_hour/seven_day/...)
        # per event - there's no on-demand "get current usage" call, so the
        # phone only ever learns what's changed since this session started,
        # not necessarily every window's live state. `utilization` is
        # nullable in the CLI's own payload (claude_agent_sdk's parser reads
        # it as `info.get("utilization")` with no default) - a null here
        # means the CLI itself didn't report a percentage for this
        # transition, not a bug on this side. Logged at info level (fires
        # rarely - only on an actual status transition) so a real payload
        # can be inspected without needing debug logging enabled.
        info = message.rate_limit_info
        logger.info("rate_limit_event for %s: %r", self.session_id, info.raw)
        self.emit(
            "rate_limit",
            rate_limit_type=info.rate_limit_type,
            status=info.status,
            utilization=info.utilization,
            resets_at=info.resets_at,
            overage_status=info.overage_status,
            overage_resets_at=info.overage_resets_at,
            raw=info.raw,
        )
        # Bug fix (user report): a real reading always wins - recalibrates
        # (or starts calibrating) this window's own estimate from here.
        # Skipped when rate_limit_type itself is nullable-and-null in the
        # CLI's own payload - nothing to calibrate for a window we don't
        # even know the identity of.
        if info.rate_limit_type is not None:
            self.rate_limit_estimator.record_real_reading(info.rate_limit_type, info.utilization)

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
