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

from .events import Event, EventSequencer

logger = logging.getLogger(__name__)

ClientFactory = Callable[[ClaudeAgentOptions], Any]


class SDKAdapter:
    """Manages the SDK-owned sessions this companion process started.

    `client_factory` defaults to the real `ClaudeSDKClient` and exists so
    tests can substitute a fake that speaks the same connect/
    receive_messages/interrupt/disconnect interface without spawning a
    real `claude` subprocess.
    """

    def __init__(self, client_factory: Optional[ClientFactory] = None, *, cwd: Optional[str] = None):
        self._client_factory: ClientFactory = client_factory or (lambda options: ClaudeSDKClient(options))
        self._default_cwd = cwd
        self._sessions: dict[str, "_Session"] = {}

    def discover_sessions(self) -> list[str]:
        return list(self._sessions)

    async def connect(self, session_id: str, *, cwd: Optional[str] = None) -> None:
        if session_id in self._sessions:
            raise ValueError(f"session already connected: {session_id}")

        session = _Session(session_id)
        self._sessions[session_id] = session
        options = ClaudeAgentOptions(cwd=cwd or self._default_cwd, can_use_tool=session.can_use_tool)
        session.client = self._client_factory(options)

        await session.client.connect(session.prompt_stream())
        # `mode` lets the mobile client (U7) tell an SDK-owned session
        # (full remote control) from an observe-only one (no
        # send_message/interrupt, per R5) without a side-channel - see
        # observe_adapter.py's matching emit for why this one bit matters.
        session.emit("session_started", mode="sdk_owned")
        session.reader_task = asyncio.create_task(session.read_loop())

    async def send_message(self, session_id: str, text: str) -> None:
        session = self._get(session_id)
        session.emit("user_message", text=text)
        await session.outbound.put(text)

    async def interrupt(self, session_id: str) -> None:
        session = self._get(session_id)
        await session.client.interrupt()
        session.end("interrupted")

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
        self.tool_started_at: dict[str, float] = {}
        self.client: Any = None
        self.reader_task: Optional[asyncio.Task] = None
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
        SDK's PermissionResultAllow/Deny."""
        request_id = getattr(context, "tool_use_id", None) or str(uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.pending[request_id] = future
        self.emit("permission_request", request_id=request_id, tool=tool_name, input=tool_input)

        decision, message = await future
        if decision == "allow":
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
            self.end("completed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # SDK/subprocess crash - surface, don't go quiet
            logger.warning("SDK-owned session %s crashed: %s", self.session_id, exc)
            self.emit("error", message=str(exc))

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
            content=block.content,
            is_error=bool(block.is_error),
            duration_ms=duration_ms,
        )
