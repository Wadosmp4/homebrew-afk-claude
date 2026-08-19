"""Tests for companion/adapters/codex_adapter.py.

The real `openai-codex` package is installed in this dev environment (used
for the plan's own hands-on spike - see codex_adapter.py's module
docstring), but no test here makes a real subprocess or network call: a
fake client/thread/turn-handle stand in for AsyncCodex/AsyncThread/
AsyncTurnHandle (mirroring test_sdk_adapter.py's FakeSDKClient
convention), while every *event/notification shape* is a real
openai_codex.models/generated dataclass - only the transport boundary is
faked, matching that file's own module docstring rationale."""
from __future__ import annotations

import asyncio

import pytest
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    CommandExecutionStatus,
    CommandExecutionThreadItem,
    McpToolCallStatus,
    McpToolCallThreadItem,
    ThreadItem,
    TurnError,
)
from openai_codex.models import ErrorNotification, ItemCompletedNotification, ItemStartedNotification

from companion.adapters.base import AdapterProtocol
from companion.adapters.codex_adapter import CodexAdapter, UnsupportedOperation

_STOP = object()


class FakeTurnHandle:
    """Stands in for AsyncTurnHandle - stream() is the real surface
    _drive_turn consumes; tests push notifications onto it directly,
    the same way test_sdk_adapter.py's FakeSDKClient.push feeds
    receive_messages()."""

    def __init__(self):
        self._notifications: asyncio.Queue = asyncio.Queue()
        self.interrupted = False

    def push(self, notification) -> None:
        self._notifications.put_nowait(notification)

    def stop(self) -> None:
        self._notifications.put_nowait(_STOP)

    async def stream(self):
        while True:
            notification = await self._notifications.get()
            if notification is _STOP:
                return
            yield notification

    async def interrupt(self) -> None:
        self.interrupted = True
        self.stop()


class FakeThread:
    """Stands in for AsyncThread."""

    def __init__(self):
        self.turns: list[FakeTurnHandle] = []
        self.compact_calls = 0

    async def turn(self, input, **kwargs):
        handle = FakeTurnHandle()
        self.turns.append(handle)
        return handle

    async def compact(self):
        self.compact_calls += 1


class FakeCodexClient:
    """Stands in for AsyncCodex."""

    def __init__(self):
        self.threads: list[FakeThread] = []
        self.thread_start_kwargs: list[dict] = []
        self.logged_in_with: str | None = None
        self.closed = False

    async def login_api_key(self, api_key: str) -> None:
        self.logged_in_with = api_key

    async def thread_start(self, **kwargs):
        thread = FakeThread()
        self.threads.append(thread)
        self.thread_start_kwargs.append(kwargs)
        return thread

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    clients: dict[str, FakeCodexClient] = {}

    def factory():
        client = FakeCodexClient()
        clients["latest"] = client
        return client

    a = CodexAdapter(client_factory=factory)
    a._test_clients = clients  # type: ignore[attr-defined]
    return a


def test_codex_adapter_conforms_to_adapter_protocol(adapter):
    assert isinstance(adapter, AdapterProtocol)


@pytest.mark.asyncio
async def test_connect_creates_a_session_and_discover_sessions_reflects_it(adapter):
    await adapter.connect("s1", cwd="/repo")

    assert adapter.discover_sessions() == ["s1"]
    client = adapter._test_clients["latest"]
    assert len(client.threads) == 1


@pytest.mark.asyncio
async def test_connect_emits_session_started_with_agent_codex(adapter):
    await adapter.connect("s1", cwd="/repo")

    events = adapter.subscribe("s1")
    event = await events.__anext__()

    assert event.type == "session_started"
    assert event.data["agent"] == "codex"
    assert event.data["cwd"] == "/repo"


@pytest.mark.asyncio
async def test_send_message_reports_a_tool_call_event_shaped_like_claude_codes(adapter):
    """Covers AE2: a command-execution item started notification becomes a
    tool_call event identically shaped (tool/input/tool_use_id) to a
    Claude Code tool_call event."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.send_message("s1", "run the tests")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    item = CommandExecutionThreadItem(
        id="item-1",
        command="pytest",
        commandActions=[],
        cwd="/repo",
        status=CommandExecutionStatus.in_progress,
        type="commandExecution",
    )
    turn_handle.push(ItemStartedNotification(item=ThreadItem(item), startedAtMs=0, threadId="t1", turnId="turn-1"))

    tool_call_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert tool_call_event.type == "tool_call"
    assert tool_call_event.data["tool"] == "Bash"
    assert tool_call_event.data["input"] == {"command": "pytest"}
    assert tool_call_event.data["tool_use_id"] == "item-1"


@pytest.mark.asyncio
async def test_a_completed_command_execution_item_reports_a_tool_result_event(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "run the tests")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    completed_item = CommandExecutionThreadItem(
        id="item-1",
        command="pytest",
        commandActions=[],
        cwd="/repo",
        status=CommandExecutionStatus.completed,
        aggregatedOutput="3 passed",
        exitCode=0,
        type="commandExecution",
    )
    turn_handle.push(
        ItemCompletedNotification(item=ThreadItem(completed_item), completedAtMs=1, threadId="t1", turnId="turn-1")
    )

    tool_result_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert tool_result_event.type == "tool_result"
    assert tool_result_event.data["tool_use_id"] == "item-1"
    assert tool_result_event.data["content"] == "3 passed"
    assert tool_result_event.data["is_error"] is False


@pytest.mark.asyncio
async def test_a_failed_mcp_tool_call_reports_an_error_tool_result(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "search the docs")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    failed_item = McpToolCallThreadItem(
        id="item-2",
        arguments={"query": "foo"},
        server="docs",
        tool="search",
        status=McpToolCallStatus.failed,
        type="mcpToolCall",
    )
    turn_handle.push(ItemCompletedNotification(item=ThreadItem(failed_item), completedAtMs=1, threadId="t1", turnId="turn-1"))

    tool_result_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert tool_result_event.type == "tool_result"
    assert tool_result_event.data["is_error"] is True


@pytest.mark.asyncio
async def test_a_completed_agent_message_item_reports_an_assistant_message_event(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    message_item = AgentMessageThreadItem(id="item-3", text="Hi, how can I help?", type="agentMessage")
    turn_handle.push(
        ItemCompletedNotification(item=ThreadItem(message_item), completedAtMs=1, threadId="t1", turnId="turn-1")
    )

    assistant_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert assistant_event.type == "assistant_message"
    assert assistant_event.data["text"] == "Hi, how can I help?"


@pytest.mark.asyncio
async def test_an_error_notification_reports_an_error_event(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    turn_handle.push(
        ErrorNotification(
            error=TurnError(message="rate limited"), threadId="t1", turnId="turn-1", willRetry=False
        )
    )

    error_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert error_event.type == "error"
    assert error_event.data["message"] == "rate limited"


@pytest.mark.asyncio
async def test_interrupt_stops_the_underlying_turn(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    await adapter.interrupt("s1")

    assert turn_handle.interrupted is True


@pytest.mark.asyncio
async def test_disconnect_closes_the_client_and_is_active_reflects_it(adapter):
    await adapter.connect("s1", cwd="/repo")

    await adapter.disconnect("s1")

    client = adapter._test_clients["latest"]
    assert client.closed is True
    assert adapter.is_active("s1") is None  # unknown after disconnect, same as SDKAdapter's own contract
    assert adapter.discover_sessions() == []


@pytest.mark.asyncio
async def test_is_active_is_true_for_a_connected_session(adapter):
    await adapter.connect("s1", cwd="/repo")

    assert adapter.is_active("s1") is True


@pytest.mark.asyncio
async def test_compact_calls_the_real_thread_compact(adapter):
    await adapter.connect("s1", cwd="/repo")

    await adapter.compact("s1")

    thread = adapter._test_clients["latest"].threads[0]
    assert thread.compact_calls == 1


@pytest.mark.asyncio
async def test_respond_to_permission_is_unsupported(adapter):
    await adapter.connect("s1", cwd="/repo")

    result = await adapter.respond_to_permission("s1", "req-1", "allow")

    assert isinstance(result, UnsupportedOperation)
    assert result.operation == "respond_to_permission"


@pytest.mark.asyncio
async def test_set_session_auto_approve_is_unsupported(adapter):
    await adapter.connect("s1", cwd="/repo")

    result = adapter.set_session_auto_approve("s1", auto_approve=True)

    assert isinstance(result, UnsupportedOperation)
    assert result.operation == "set_session_auto_approve"


@pytest.mark.asyncio
async def test_connect_logs_in_with_an_api_key_when_one_is_configured(adapter, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-codex-test")

    await adapter.connect("s1", cwd="/repo")

    client = adapter._test_clients["latest"]
    assert client.logged_in_with == "sk-codex-test"


@pytest.mark.asyncio
async def test_connect_skips_login_with_no_api_key_configured(adapter):
    await adapter.connect("s1", cwd="/repo")

    client = adapter._test_clients["latest"]
    assert client.logged_in_with is None
