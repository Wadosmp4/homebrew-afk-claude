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
import stat
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .events import Event, EventSequencer, truncate_tool_result_content

logger = logging.getLogger(__name__)

DEFAULT_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
DEFAULT_SOCKET_PATH = os.path.expanduser("~/.config/remote-claude-companion/hooks.sock")
DEFAULT_WATCH_POLL_INTERVAL = 1.0
DEFAULT_TAIL_POLL_INTERVAL = 0.5

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
    ):
        self.projects_dir = Path(projects_dir or DEFAULT_PROJECTS_DIR)
        self.socket_path = socket_path or DEFAULT_SOCKET_PATH
        self._watch_poll_interval = watch_poll_interval
        self._tail_poll_interval = tail_poll_interval
        self._sessions: dict[str, "_ObserveSession"] = {}
        self._known_transcripts: set[Path] = set()
        self._server: Optional[asyncio.AbstractServer] = None
        self._watch_task: Optional[asyncio.Task] = None

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

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

    async def send_message(self, session_id: str, text: str) -> UnsupportedOperation:
        return UnsupportedOperation(operation="send_message")

    async def interrupt(self, session_id: str) -> UnsupportedOperation:
        return UnsupportedOperation(operation="interrupt")

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

    def _get(self, session_id: str) -> "_ObserveSession":
        try:
            return self._sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown observed session: {session_id}") from None

    def _get_or_create(self, session_id: str) -> "_ObserveSession":
        session = self._sessions.get(session_id)
        if session is None:
            session = _ObserveSession(session_id)
            self._sessions[session_id] = session
        return session

    # --- JSONL transcript discovery + tailing -------------------------------

    async def _watch_projects_dir(self) -> None:
        while True:
            for path in sorted(self.projects_dir.glob("*/*.jsonl")):
                if path not in self._known_transcripts:
                    self._known_transcripts.add(path)
                    self._start_watching_transcript(path)
            await asyncio.sleep(self._watch_poll_interval)

    def _start_watching_transcript(self, path: Path) -> None:
        session_id = path.stem
        session = self._get_or_create(session_id)
        session.emit("session_started", mode="observe_only")
        session.tail_task = asyncio.create_task(self._tail_file(path, session))

    async def _tail_file(self, path: Path, session: "_ObserveSession") -> None:
        offset = 0
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
            session.emit("session_started", mode="observe_only")
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
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            session.pending[request_id] = future
            session.emit(
                "permission_request",
                request_id=request_id,
                tool=payload.get("tool_name"),
                input=payload.get("tool_input"),
            )
            decision, message = await future
            return {"permissionDecision": decision, "permissionDecisionReason": message}

        return {}


class _ObserveSession:
    """U9: `duration_ms` here comes from JSONL entries' own `timestamp`
    fields (wall-clock ISO strings every real transcript entry carries),
    not a monotonic clock like U3's SDK-owned session - `tool_started_at`
    tracks the entry timestamp captured when each tool_call was emitted."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.sequencer = EventSequencer(session_id)
        self.events: asyncio.Queue[Event] = asyncio.Queue()
        self.pending: dict[str, asyncio.Future] = {}
        self.tail_task: Optional[asyncio.Task] = None
        self.tool_started_at: dict[str, str] = {}
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

    def resolve_permission(self, request_id: str, decision: str, message: str = "") -> None:
        future = self.pending.pop(request_id, None)
        if future is None:
            raise KeyError(f"no pending permission request: {request_id}")
        future.set_result((decision, message))
