"""Observe-only adapter (U4): discovers and exposes Claude Code sessions
started manually in a terminal, read-only, with permission-response
support via hooks (R5, R6, KTD2).

Two independent data sources feed the same per-session event stream:

  1. JSONL transcript tailing (~/.claude/projects/*/*.jsonl) - the source
     of truth for conversation content: assistant messages, tool calls,
     tool results, and the human's own typed messages.
  2. The hooks Unix socket (hooks_installer.py installs the hook commands
     that reach it) - the source of truth for session lifecycle
     (SessionStart/SessionEnd), turn-boundary notifications (Stop/
     Notification), and permission requests, since none of those are
     represented as JSONL lines.

`send_message` and `interrupt` are unsupported here per R5 - they return an
`UnsupportedOperation` result rather than raising, so a phone tapping Stop
on an observed session gets a clear "can't do that" instead of a generic
error.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import stat
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .. import auto_approve as approval_policy
from .. import risk_judge
from .events import Event, EventSequencer, truncate_tool_result_content

logger = logging.getLogger(__name__)

DEFAULT_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
DEFAULT_SOCKET_PATH = os.path.expanduser("~/.config/remote-claude-companion/hooks.sock")
DEFAULT_WATCH_POLL_INTERVAL = 1.0
DEFAULT_TAIL_POLL_INTERVAL = 0.5
# Safeguards for pointing this at the real ~/.claude/projects, which can
# hold months of history across every project ever touched - without
# these, watching it reproduces the CPU-storm incident (tailing dozens of
# transcripts, some 70+MB, from byte 0, all at once). Neither limit
# applies to test fixtures (freshly-created files are never stale, and
# tests watch far fewer than the cap).
DEFAULT_MAX_WATCHED_SESSIONS = 10
DEFAULT_STALE_AFTER_SECONDS = 6 * 3600  # only watch transcripts touched recently

# ~/.claude/projects is shared machine-wide across every Claude Code
# client (Desktop app, VS Code extension, plain terminal CLI, a custom
# Agent SDK script) and every project ever opened in any of them -
# pointing the watcher at it means an unrelated repo worked on from, say,
# the VS Code extension shows up in the phone's Sessions list too. Each
# transcript line records which client wrote it (`entrypoint`), so
# `required_entrypoints` scopes discovery to a chosen subset - a session
# qualifies once at least one of its lines matches one of them (see
# _matches_required_entrypoints). None/empty (the class default) means no
# filtering, matching every existing caller/test that doesn't care about
# this. The actual choice is user-configurable at runtime (R: "give an
# ability to choose what clients to use") via set_required_entrypoints -
# companion/daemon.py persists it in CompanionConfig and applies it on
# both startup and a phone-sent set_observe_entrypoints action.
KNOWN_ENTRYPOINTS = ("claude-desktop", "claude-vscode", "sdk-py", "sdk-ts")
_ENTRYPOINT_SCAN_BYTES = 65536


def _matches_required_entrypoints(path: Path, required_entrypoints: frozenset[str]) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(_ENTRYPOINT_SCAN_BYTES)
    except OSError:
        return False
    return any(
        re.search(rb'"entrypoint"\s*:\s*"' + re.escape(entrypoint.encode()) + rb'"', chunk) is not None
        for entrypoint in required_entrypoints
    )

# Hook events that map directly onto a lifecycle/waiting event, per this
# unit's judgment call on KTD2's hook set (see module docstring): there's
# no dedicated "notification" event type in R6's model, so Stop/Notification
# both surface as `waiting_for_input` - the closest fit for "Claude isn't
# actively doing anything and is waiting on something".
_WAITING_HOOK_EVENTS = ("Stop", "Notification")


@dataclass(frozen=True)
class UnsupportedOperation:
    operation: str
    reason: str = "not supported for observed sessions"


class ObserveAdapter:
    def __init__(
        self,
        projects_dir: Optional[str] = None,
        socket_path: Optional[str] = None,
        watch_poll_interval: float = DEFAULT_WATCH_POLL_INTERVAL,
        tail_poll_interval: float = DEFAULT_TAIL_POLL_INTERVAL,
        max_watched_sessions: int = DEFAULT_MAX_WATCHED_SESSIONS,
        stale_after_seconds: float = DEFAULT_STALE_AFTER_SECONDS,
        required_entrypoints: Optional[frozenset[str]] = None,
        auto_approve: bool = False,
        llm_judge: bool = False,
    ):
        self.projects_dir = Path(projects_dir or DEFAULT_PROJECTS_DIR)
        self.socket_path = socket_path or DEFAULT_SOCKET_PATH
        self._watch_poll_interval = watch_poll_interval
        self._tail_poll_interval = tail_poll_interval
        self._max_watched_sessions = max_watched_sessions
        self._stale_after_seconds = stale_after_seconds
        self._required_entrypoints = required_entrypoints or frozenset()
        # Same opt-in-and-layered policy as SDKAdapter (see its _Session.
        # can_use_tool) - a global setting here rather than per-session,
        # since observed sessions aren't created by the phone at all, so
        # there's no start_session-time moment to attach a per-session flag
        # to. Checked live on every PermissionRequest hook (_dispatch_hook),
        # so toggling this on also takes effect for a session already
        # mid-conversation, not just ones discovered afterward.
        self._auto_approve = auto_approve
        self._llm_judge = llm_judge
        self._sessions: dict[str, "_ObserveSession"] = {}
        self._known_transcripts: set[Path] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self._watch_task: Optional[asyncio.Task] = None
        # True only for the very first watch-loop pass: a file discovered
        # then already existed before we started watching (possibly months
        # of real history), vs. one that appears later, which is by
        # definition new activity that just started - see
        # _start_watching_transcript's skip_existing_content param.
        self._initial_scan_done = False

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

    def get_required_entrypoints(self) -> frozenset[str]:
        return self._required_entrypoints

    def set_required_entrypoints(self, entrypoints: frozenset[str]) -> None:
        """Changes take effect only for transcripts discovered from now on
        - a session already being watched keeps streaming (KD7-style: don't
        yank a session out from under the phone mid-conversation just
        because a setting changed), and a transcript already rejected by
        the old filter stays rejected until next restart, same trade-off
        `_known_transcripts` already makes for the staleness/cap filters."""
        self._required_entrypoints = entrypoints

    def get_auto_approve(self) -> bool:
        return self._auto_approve

    def set_auto_approve(self, value: bool) -> None:
        self._auto_approve = value

    def get_llm_judge(self) -> bool:
        return self._llm_judge

    def set_llm_judge(self, value: bool) -> None:
        self._llm_judge = value

    async def start(self) -> None:
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)  # stale socket from a prior crashed run

        self._server = await asyncio.start_unix_server(self._handle_hook_connection, path=self.socket_path)
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - local privilege surface (Risk)
        self._watch_task = asyncio.create_task(self._watch_projects_dir())

    async def stop(self) -> None:
        if self._watch_task is not None:
            self._watch_task.cancel()
        for session in self._sessions.values():
            if session.tail_task is not None:
                session.tail_task.cancel()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    # --- unsupported remote-control operations (R5) ------------------------
    #
    # Returning `UnsupportedOperation` rather than raising, or emitting a
    # normal `error` event, is deliberate: `error` terminates this
    # session's `subscribe()` stream (see below), which is correct for a
    # fatal SDK/subprocess crash but wrong for "you tapped Stop on a
    # session you can't control" - that shouldn't kill the feed. U7's
    # mobile client instead disables the input bar proactively using
    # `session_started`'s `mode` field (R5's test scenario), so this
    # path is never actually reached from the app; it stays defensive for
    # any other action-sending client.

    def open_session(self, session_id: str) -> None:
        """R: 'do not connect automatically ... just show that it exists' -
        the phone calls this (daemon.py's open_session action) when the
        user actually taps into a discovered session; only then does its
        content start forwarding (see _ObserveSession.emit). Silently a
        no-op for an unknown session_id - a phone request racing ahead of
        this adapter's own discovery isn't an error to surface."""
        session = self._sessions.get(session_id)
        if session is not None:
            session.mark_opened()

    def set_session_auto_approve(
        self, session_id: str, auto_approve: Optional[bool] = None, llm_judge: Optional[bool] = None
    ) -> bool:
        """Override this one session's own (already-snapshotted, see
        _get_or_create) auto_approve/llm_judge directly - lets the phone
        turn auto-approval off (or on) for a session already in progress,
        independent of the global default and of any other session.
        `None` for either argument leaves that one field untouched, same
        convention as daemon.py's set_auto_approve_settings. Returns False
        (no-op) for an unknown session_id, matching open_session's own
        "a phone request racing ahead of discovery isn't an error"
        posture."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        if auto_approve is not None:
            session.auto_approve = auto_approve
        if llm_judge is not None:
            session.llm_judge = llm_judge
        session.emit("session_auto_approve", auto_approve=session.auto_approve, llm_judge=session.llm_judge)
        return True

    async def send_message(self, session_id: str, text: str) -> UnsupportedOperation:
        return UnsupportedOperation(operation="send_message")

    async def interrupt(self, session_id: str) -> UnsupportedOperation:
        return UnsupportedOperation(operation="interrupt")

    async def compact(self, session_id: str) -> UnsupportedOperation:
        """An observed session isn't driven by our own ClaudeSDKClient -
        there's no client here to send `/compact` through, and no
        context_usage was ever emitted for it to react to in the first
        place (see events.py's registration comment)."""
        return UnsupportedOperation(operation="compact")

    # --- permission round-trip ----------------------------------------------

    async def respond_to_permission(
        self, session_id: str, request_id: str, decision: str, *, message: str = ""
    ) -> None:
        session = self._get(session_id)
        session.resolve_permission(request_id, decision, message)

    async def subscribe(self, session_id: str) -> AsyncIterator[Event]:
        session = self._get(session_id)
        while True:
            event = await session.events.get()
            yield event
            if event.type in ("session_ended", "error"):
                return

    def get_cwd(self, session_id: str) -> Optional[str]:
        """U10 (R16): captured from the JSONL transcript entries' own
        `cwd` field (see _normalize_line) - None until at least one
        transcript line with a `cwd` has been seen for this session."""
        session = self._sessions.get(session_id)
        return session.cwd if session is not None else None

    def is_active(self, session_id: str) -> Optional[bool]:
        """For the Sessions screen's list_active_sessions snapshot
        (daemon.py) - None for an unknown session_id, distinct from False
        (a real, already-ended one) so the caller can tell "never existed"
        from "existed and ended" if it ever needs to."""
        session = self._sessions.get(session_id)
        return None if session is None else not session._ended

    def emit_custom(self, session_id: str, type_: str, **data: Any) -> None:
        """U10: see SDKAdapter.emit_custom's docstring - same purpose,
        same "lets the daemon inject a computed result into this
        session's own stream" contract, for observe-only sessions."""
        self._get(session_id).emit(type_, **data)

    def _get(self, session_id: str) -> "_ObserveSession":
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown observed session: {session_id}") from None

    def _get_or_create(self, session_id: str) -> "_ObserveSession":
        session = self._sessions.get(session_id)
        if session is None:
            # Snapshot, not a live reference - see _ObserveSession's own
            # comment on these two attributes for why.
            session = _ObserveSession(session_id, auto_approve=self._auto_approve, llm_judge=self._llm_judge)
            self._sessions[session_id] = session
        return session

    # --- JSONL transcript discovery + tailing -------------------------------

    async def _watch_projects_dir(self) -> None:
        while True:
            is_initial_scan = not self._initial_scan_done
            # Real usage points this at ~/.claude/projects, which can hold
            # hundreds of transcripts across every project ever touched -
            # the glob itself, and the per-file stat + up-to-64KB
            # entrypoint scan below, are synchronous filesystem I/O that
            # would otherwise block every other coroutine (message
            # handling, other sessions' tail loops) for the duration of
            # one poll tick. Offloaded to a thread rather than run inline.
            candidates = await asyncio.to_thread(lambda: sorted(self.projects_dir.glob("*/*.jsonl")))
            for path in candidates:
                if path in self._known_transcripts:
                    continue
                # Marked known unconditionally (even when skipped below) so
                # a huge/stale real projects_dir doesn't re-stat every file
                # on every poll forever - a skipped file stays skipped until
                # the next daemon restart, which is an acceptable trade-off
                # for a personal-use app.
                self._known_transcripts.add(path)
                if len(self._sessions) >= self._max_watched_sessions:
                    logger.info(
                        "not watching %s: already watching %d sessions (max %d)",
                        path, len(self._sessions), self._max_watched_sessions,
                    )
                    continue
                skip_reason = await asyncio.to_thread(
                    self._check_transcript_eligibility, path
                )
                if skip_reason is not None:
                    continue
                self._start_watching_transcript(path, skip_existing_content=is_initial_scan)
            self._initial_scan_done = True
            await asyncio.sleep(self._watch_poll_interval)

    def _check_transcript_eligibility(self, path: Path) -> Optional[str]:
        """The synchronous (stat + bounded file read) half of deciding
        whether to watch a newly-discovered transcript - run off the event
        loop via asyncio.to_thread by the caller. Returns None when
        eligible, else a short skip reason (unused beyond truthiness
        today, but named for whoever adds logging here next)."""
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return "unreadable"
        if time.time() - mtime > self._stale_after_seconds:
            return "stale"  # old history, not a live/recent session
        if self._required_entrypoints and not _matches_required_entrypoints(path, self._required_entrypoints):
            return "entrypoint-mismatch"  # not written by a client the user chose to see
        return None

    def _start_watching_transcript(self, path: Path, *, skip_existing_content: bool) -> None:
        session_id = path.stem
        session = self._get_or_create(session_id)
        session.start()
        initial_offset = 0
        if skip_existing_content:
            # Only for files that already existed at startup - they can
            # hold megabytes of real history from before this daemon ran,
            # and reading/emitting all of that as if it just happened is
            # both wasteful and not what "observe what's happening now"
            # means. A file that appears *after* startup is by definition
            # new activity, so it's read from the start like always.
            try:
                initial_offset = path.stat().st_size
            except OSError:
                initial_offset = 0
        session.tail_task = asyncio.create_task(self._tail_file(path, session, initial_offset))

    async def _tail_file(self, path: Path, session: "_ObserveSession", offset: int = 0) -> None:
        pending = b""
        try:
            while True:
                with open(path, "rb") as f:
                    f.seek(offset)
                    chunk = f.read()
                if chunk:
                    offset += len(chunk)
                    pending += chunk
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        if line.strip():
                            self._normalize_line(session, line)
                await asyncio.sleep(self._tail_poll_interval)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive, mirrors U3's crash handling
            logger.warning("transcript tail for %s failed: %s", session.session_id, exc)
            session.emit("error", message=str(exc))

    def _normalize_line(self, session: "_ObserveSession", raw_line: bytes) -> None:
        try:
            entry = json.loads(raw_line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # U10 (R16): every real transcript entry carries the session's cwd;
        # capture it regardless of whether this particular line has
        # message content, so git_status/git_diff work as soon as any
        # line has arrived, not only after the first content-bearing one.
        entry_cwd = entry.get("cwd")
        if entry_cwd:
            session.cwd = entry_cwd
            session.note_cwd_known()

        entry_type = entry.get("type")
        entry_timestamp = entry.get("timestamp")
        content = entry.get("message", {}).get("content")
        if content is None:
            return

        if entry_type == "assistant":
            for block in content if isinstance(content, list) else []:
                if block.get("type") == "text":
                    session.emit("assistant_message", text=block["text"])
                elif block.get("type") == "tool_use":
                    if entry_timestamp:
                        session.tool_started_at[block["id"]] = entry_timestamp
                    session.emit("tool_call", tool_use_id=block["id"], tool=block["name"], input=block.get("input"))
        elif entry_type == "user":
            if isinstance(content, str):
                session.emit("user_message", text=content)
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        tool_use_id = block.get("tool_use_id")
                        duration_ms = self._duration_since(session, tool_use_id, entry_timestamp)
                        session.emit(
                            "tool_result",
                            tool_use_id=tool_use_id,
                            content=truncate_tool_result_content(block.get("content")),
                            is_error=bool(block.get("is_error", False)),
                            duration_ms=duration_ms,
                        )
                    elif block.get("type") == "text":
                        session.emit("user_message", text=block["text"])

    @staticmethod
    def _duration_since(session: "_ObserveSession", tool_use_id: Optional[str], result_timestamp: Optional[str]) -> Optional[float]:
        started = session.tool_started_at.pop(tool_use_id, None) if tool_use_id else None
        if started is None or not result_timestamp:
            return None
        try:
            start = datetime.fromisoformat(started)
            end = datetime.fromisoformat(result_timestamp)
        except ValueError:
            return None
        return max((end - start).total_seconds() * 1000, 0.0)

    # --- hooks Unix socket ---------------------------------------------------

    async def _handle_hook_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        response: dict[str, Any] = {}
        try:
            raw = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            raw = exc.partial
        except asyncio.LimitOverrunError:
            raw = b""

        try:
            payload = json.loads(raw.decode("utf-8")) if raw.strip() else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = {}

        try:
            response = await self._dispatch_hook(payload)
        finally:
            writer.write(json.dumps(response).encode("utf-8") + b"\n")
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()

    async def _dispatch_hook(self, payload: dict[str, Any]) -> dict[str, Any]:
        event_name = payload.get("_hook_event")
        session_id = payload.get("session_id") or payload.get("sessionId")
        if not session_id:
            return {}

        if event_name == "SessionStart":
            session = self._get_or_create(session_id)
            session.start()
            return {}

        # Anything else must reference a session we're already tracking -
        # reject rather than acting on an unrecognized hook payload (Risk:
        # "reject hook payloads that don't match a ... session it is
        # actually tracking").
        session = self._sessions.get(session_id)
        if session is None:
            return {}

        if event_name == "SessionEnd":
            session.end("hook")
            return {}
        if event_name in _WAITING_HOOK_EVENTS:
            session.emit("waiting_for_input", source=event_name, message=payload.get("message"))
            return {}
        if event_name == "PermissionRequest":
            request_id = payload.get("tool_use_id") or str(uuid4())
            tool_name = payload.get("tool_name")
            tool_input = payload.get("tool_input")

            # Same layered policy as SDKAdapter._Session.can_use_tool: the
            # denylist always wins and is never consulted by the LLM judge;
            # a policy-approved or judge-approved call still emits this
            # event (tagged auto_approved) for transparency, just without
            # blocking on a phone response. Per-session (session.auto_approve/
            # session.llm_judge), snapshotted at discovery time from the
            # adapter's global default, not read live from self._auto_approve/
            # self._llm_judge here - see _ObserveSession's own comment on
            # why, and set_session_auto_approve for the per-session override.
            #
            # A structured question (AskUserQuestion) is excluded entirely,
            # same as SDKAdapter - "is this safe to run" is the wrong
            # question for it (asking a question is never "risky"), and
            # unlike a normal tool call, "allow" here doesn't carry the
            # phone's chosen answer back; only the blocking human-prompt
            # path below (respond_to_permission with the chosen label as
            # `message`) can. Auto-approving it here would silently
            # "answer" it with a bare allow-no-message before the phone
            # ever gets to choose - the exact regression this guard exists
            # to prevent.
            if not approval_policy.is_structured_question(tool_input) and session.auto_approve:
                denylisted = approval_policy.is_denylisted(tool_name, tool_input)
                if not denylisted and approval_policy.is_auto_approvable(tool_name, tool_input, cwd=session.cwd):
                    session.emit(
                        "permission_request",
                        request_id=request_id,
                        tool=tool_name,
                        input=tool_input,
                        auto_approved=True,
                        judged_by="policy",
                    )
                    return {"permissionDecision": "allow", "permissionDecisionReason": ""}
                if not denylisted and session.llm_judge:
                    if await risk_judge.judge_is_safe(tool_name, tool_input, session.cwd):
                        session.emit(
                            "permission_request",
                            request_id=request_id,
                            tool=tool_name,
                            input=tool_input,
                            auto_approved=True,
                            judged_by="llm",
                        )
                        return {"permissionDecision": "allow", "permissionDecisionReason": ""}

            future: asyncio.Future = asyncio.get_event_loop().create_future()
            session.pending[request_id] = future
            session.emit(
                "permission_request",
                request_id=request_id,
                tool=tool_name,
                input=tool_input,
            )
            decision, message = await future
            return {"permissionDecision": decision, "permissionDecisionReason": message}

        return {}


#  R: "do not connect automatically to opened session, just show that it
# exists" - discovering a transcript starts the internal tail (needed
# regardless, to learn its cwd for the Sessions list) but must not forward
# its full content to every phone before anyone has actually asked to look
# at it. Lifecycle/notification-worthy events always forward - the
# Sessions list needs session_started/session_ended, and a permission
# request or "waiting for input" must reach the phone even with the
# dashboard closed (that's the whole point of push notifications for
# those categories - see mobile/screens/SettingsScreen.tsx). Everything
# else (actual conversation content) only starts flowing once
# `mark_opened()` is called (ObserveAdapter.open_session, wired to a
# phone-sent `open_session` action - see daemon.py). Content from before
# that point isn't backfilled - companion/history.py's read-only
# transcript browsing covers "what happened before I looked."
_ALWAYS_FORWARD_TYPES = frozenset({"session_started", "session_ended", "error", "permission_request", "waiting_for_input"})


class _ObserveSession:
    """U9: `duration_ms` here comes from JSONL entries' own `timestamp`
    fields (wall-clock ISO strings every real transcript entry carries),
    not a monotonic clock like U3's SDK-owned session - `tool_started_at`
    tracks the entry timestamp captured when each tool_call was emitted."""

    def __init__(self, session_id: str, *, auto_approve: bool = False, llm_judge: bool = False):
        self.session_id = session_id
        self.sequencer = EventSequencer(session_id)
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}
        self.tail_task: Optional[asyncio.Task] = None
        self.tool_started_at: dict[str, str] = {}
        self.cwd: Optional[str] = None  # U10 (R16): captured from a JSONL entry's own `cwd` field
        self.opened = False
        self._cwd_announced = False
        self._started = False
        self._ended = False
        # Snapshotted from the adapter's global setting at the moment this
        # session was first discovered (see ObserveAdapter._get_or_create),
        # not re-checked live afterward - matches how a phone-started
        # session's own auto_approve/llm_judge are frozen at connect() time
        # (sdk_adapter.py), so toggling the global setting later has a
        # consistent, predictable effect: it changes what *new* sessions
        # get, not what an already-running one is doing. A specific
        # session can still be overridden individually - see
        # ObserveAdapter.set_session_auto_approve.
        self.auto_approve = auto_approve
        self.llm_judge = llm_judge

    def emit(self, type_: str, **data: Any) -> Event:
        event = self.sequencer.emit(type_, **data)
        if self.opened or type_ in _ALWAYS_FORWARD_TYPES:
            self.events.put_nowait(event)
        return event

    def mark_opened(self) -> None:
        self.opened = True

    def note_cwd_known(self) -> None:
        """Called once `cwd` is first set - re-announces `session_started`
        (always-forward) with it, so the Sessions list can show a real
        project name instead of a bare session id even before the phone
        opens this session (cwd isn't known yet at the original `start()`,
        which fires from mere file discovery)."""
        if self._cwd_announced or self.cwd is None:
            return
        self._cwd_announced = True
        self.emit(
            "session_started", mode="observe_only", cwd=self.cwd, auto_approve=self.auto_approve, llm_judge=self.llm_judge
        )

    def start(self) -> None:
        """Idempotent, mirroring `end()` - a session becomes known via
        *either* the JSONL file watcher or the SessionStart hook, often
        both, so both call sites route through here rather than emitting
        `session_started` directly (a real bug this guard fixes: without
        it, every session got two "Session started" events)."""
        if self._started:
            return
        self._started = True
        self.emit("session_started", mode="observe_only", auto_approve=self.auto_approve, llm_judge=self.llm_judge)

    def end(self, reason: str) -> None:
        if self._ended:
            return
        self._ended = True
        self.emit("session_ended", reason=reason)

    def resolve_permission(self, request_id: str, decision: str, message: str = "") -> None:
        future = self.pending.pop(request_id, None)
        if future is None:
            raise KeyError(f"no pending permission request: {request_id}")
        future.set_result((decision, message))
