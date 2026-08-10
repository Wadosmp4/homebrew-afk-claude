"""Tests for companion/adapters/sdk_adapter.py.

There's no `claude` CLI in this environment to spawn a real subprocess
against, so these tests inject a fake client that speaks the same async
interface as `claude_agent_sdk.ClaudeSDKClient` (connect/receive_messages/
interrupt/disconnect) and yields *real* SDK message dataclasses
(AssistantMessage, ToolUseBlock, ToolResultBlock, ResultMessage) - only the
transport boundary is faked, not the message shapes the adapter has to
handle, so the normalization logic under test is exercised against the
real contract.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from companion.adapters.sdk_adapter import SDKAdapter


class FakeSDKClient:
    """Stands in for ClaudeSDKClient. `options.can_use_tool` is the real
    callback the adapter wired up - tests invoke it directly to simulate
    what a real subprocess does when a tool needs permission."""

    def __init__(self, options):
        self.options = options
        self._outbox: asyncio.Queue = asyncio.Queue()
        self.interrupted = False
        self.disconnected = False
        self.connected_prompt = None

    async def connect(self, prompt=None) -> None:
        self.connected_prompt = prompt

    def push(self, message) -> None:
        """Test helper: queue a message for receive_messages() to yield."""
        self._outbox.put_nowait(message)

    async def receive_messages(self):
        while True:
            message = await self._outbox.get()
            if message is _STOP:
                return
            if isinstance(message, BaseException):
                raise message
            yield message

    async def interrupt(self) -> None:
        self.interrupted = True

    async def disconnect(self) -> None:
        self.disconnected = True


_STOP = object()


@pytest.fixture
async def adapter():
    clients: dict[str, FakeSDKClient] = {}

    def factory(options):
        client = FakeSDKClient(options)
        clients["latest"] = client
        return client

    a = SDKAdapter(client_factory=factory)
    a._test_clients = clients  # type: ignore[attr-defined]
    yield a

    for session_id in list(a.discover_sessions()):
        await a.disconnect(session_id)


async def _next_event(adapter: SDKAdapter, session_id: str):
    gen = adapter.subscribe(session_id)
    return await gen.__anext__(), gen


@pytest.mark.asyncio
async def test_connect_emits_session_started(adapter):
    await adapter.connect("s1")
    assert adapter.discover_sessions() == ["s1"]

    event, _gen = await _next_event(adapter, "s1")
    assert event.type == "session_started"
    assert event.session_id == "s1"
    assert event.data["mode"] == "sdk_owned"  # U7 uses this to enable full remote control


@pytest.mark.asyncio
async def test_send_message_then_assistant_reply_then_waiting_for_input(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]

    events = adapter.subscribe("s1")
    started = await events.__anext__()
    assert started.type == "session_started"

    await adapter.send_message("s1", "hello claude")
    user_event = await events.__anext__()
    assert user_event.type == "user_message"
    assert user_event.data["text"] == "hello claude"

    client.push(AssistantMessage(content=[TextBlock(text="hi there")], model="claude"))
    assistant_event = await events.__anext__()
    assert assistant_event.type == "assistant_message"
    assert assistant_event.data["text"] == "hi there"

    client.push(
        ResultMessage(
            subtype="success",
            duration_ms=10,
            duration_api_ms=8,
            is_error=False,
            num_turns=1,
            session_id="s1",
        )
    )
    lifecycle_event = await events.__anext__()
    assert lifecycle_event.type == "waiting_for_input"


@pytest.mark.asyncio
async def test_tool_call_and_result_produce_events_with_duration(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    client.push(
        AssistantMessage(
            content=[ToolUseBlock(id="tool-1", name="Read", input={"path": "x.py"})],
            model="claude",
        )
    )
    call_event = await events.__anext__()
    assert call_event.type == "tool_call"
    assert call_event.data["tool"] == "Read"
    assert call_event.data["tool_use_id"] == "tool-1"

    client.push(
        AssistantMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content="file contents", is_error=False)],
            model="claude",
        )
    )
    result_event = await events.__anext__()
    assert result_event.type == "tool_result"
    assert result_event.data["tool_use_id"] == "tool-1"
    assert result_event.data["duration_ms"] is not None
    assert result_event.data["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_permission_request_blocks_until_respond_to_permission_allow(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    # Simulate the real subprocess asking the wired-up can_use_tool callback
    # for permission - run it as a background task since it blocks until
    # respond_to_permission resolves it.
    from claude_agent_sdk import ToolPermissionContext

    call_task = asyncio.create_task(
        client.options.can_use_tool(
            "Bash", {"command": "rm -rf /tmp/x"}, ToolPermissionContext(tool_use_id="tool-9")
        )
    )

    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"
    assert permission_event.data["tool"] == "Bash"
    request_id = permission_event.data["request_id"]

    assert not call_task.done()
    await adapter.respond_to_permission("s1", request_id, "allow")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "allow"


@pytest.mark.asyncio
async def test_permission_request_deny_rejects_tool_back_to_claude(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    call_task = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-9", "deny", message="not allowed")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "deny"
    assert result.message == "not allowed"


@pytest.mark.asyncio
async def test_interrupt_stops_session_and_emits_lifecycle_event(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.interrupt("s1")
    assert client.interrupted is True

    lifecycle_event = await events.__anext__()
    assert lifecycle_event.type == "session_ended"
    assert lifecycle_event.data["reason"] == "interrupted"


@pytest.mark.asyncio
async def test_client_crash_emits_error_event_not_silence(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    client.push(RuntimeError("subprocess died"))  # simulate the SDK stream blowing up

    error_event = await events.__anext__()
    assert error_event.type == "error"
    assert "subprocess died" in error_event.data["message"]
