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
    Turn,
    TurnError,
    TurnStatus,
)
from openai_codex.models import (
    ErrorNotification,
    ItemCompletedNotification,
    ItemStartedNotification,
    TurnCompletedNotification,
)

from openai_codex import Sandbox

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
        self.turn_kwargs: list[dict] = []
        self.compact_calls = 0
        self.raise_on_turn: Exception | None = None
        self.raise_on_compact: Exception | None = None

    async def turn(self, input, **kwargs):
        if self.raise_on_turn is not None:
            raise self.raise_on_turn
        self.turn_kwargs.append(kwargs)
        handle = FakeTurnHandle()
        self.turns.append(handle)
        return handle

    async def compact(self):
        self.compact_calls += 1
        if self.raise_on_compact is not None:
            raise self.raise_on_compact


class FakeCodexClient:
    """Stands in for AsyncCodex."""

    def __init__(self):
        self.threads: list[FakeThread] = []
        self.thread_start_kwargs: list[dict] = []
        self.logged_in_with: str | None = None
        self.closed = False
        self.close_calls = 0
        self.raise_on_thread_start: Exception | None = None
        self.raise_on_close: Exception | None = None

    async def login_api_key(self, api_key: str) -> None:
        self.logged_in_with = api_key

    async def thread_start(self, **kwargs):
        if self.raise_on_thread_start is not None:
            raise self.raise_on_thread_start
        thread = FakeThread()
        self.threads.append(thread)
        self.thread_start_kwargs.append(kwargs)
        return thread

    async def close(self) -> None:
        self.close_calls += 1
        if self.raise_on_close is not None:
            raise self.raise_on_close
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
async def test_connect_includes_model_and_sandbox_in_session_started(adapter):
    """code-review finding: unlike SDKAdapter's own session_started (which
    carries model/permission_mode), CodexAdapter's originally didn't - the
    dashboard's derivedModel/derivedSandbox both read the latest of
    session_started or a live session_model/session_sandbox event, so a
    freshly created Codex session with a non-default model/sandbox showed
    "Default"/"Workspace write" until some later live change fired its own
    event."""
    await adapter.connect("s1", cwd="/repo", model="gpt-5.2-codex", sandbox="full_access")

    events = adapter.subscribe("s1")
    event = await events.__anext__()

    assert event.type == "session_started"
    assert event.data["model"] == "gpt-5.2-codex"
    assert event.data["sandbox"] == "full_access"


@pytest.mark.asyncio
async def test_connect_with_no_model_or_sandbox_emits_session_started_with_both_none(adapter):
    await adapter.connect("s1", cwd="/repo")

    events = adapter.subscribe("s1")
    event = await events.__anext__()

    assert event.type == "session_started"
    assert event.data["model"] is None
    assert event.data["sandbox"] is None


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
async def test_set_session_auto_approve_returns_false_not_unsupported_operation(adapter):
    """Not part of AdapterProtocol (see base.py's own comment) - kept for
    parity with ObserveAdapter's own method of this name. False is its
    "no-op, nothing to apply" signal, matching ObserveAdapter's own
    convention for an unknown session_id, rather than UnsupportedOperation
    (there's no per-session auto_approve/llm_judge concept for Codex to
    actually set, only the coarse, thread-level ApprovalMode)."""
    await adapter.connect("s1", cwd="/repo")

    result = adapter.set_session_auto_approve("s1", auto_approve=True)

    assert result is False


@pytest.mark.asyncio
async def test_connect_logs_in_with_an_api_key_when_one_is_configured(adapter):
    """Codex Agent Integration plan (002), U2 (PKTD3): connect() now takes
    an explicit, caller-resolved api_key instead of reading OPENAI_API_KEY
    out of this process's own environment - the daemon is the one place
    that resolves codex_active_account/codex_personal_api_key into this
    parameter (personal mode: test scenario 4)."""
    await adapter.connect("s1", cwd="/repo", api_key="sk-codex-test")

    client = adapter._test_clients["latest"]
    assert client.logged_in_with == "sk-codex-test"


@pytest.mark.asyncio
async def test_connect_skips_login_with_no_api_key_configured(adapter):
    """"vscode" mode (test scenario 3): no api_key passed at all means
    login_api_key is never called - the default, unchanged behavior."""
    await adapter.connect("s1", cwd="/repo")

    client = adapter._test_clients["latest"]
    assert client.logged_in_with is None


@pytest.mark.asyncio
async def test_connect_skips_login_when_api_key_is_explicitly_none(adapter):
    """Same as the no-argument case above, but for a caller (the daemon,
    under "vscode" mode) that passes api_key=None explicitly rather than
    omitting it - both must be equally inert."""
    await adapter.connect("s1", cwd="/repo", api_key=None)

    client = adapter._test_clients["latest"]
    assert client.logged_in_with is None


# --- Code-review fixes ---------------------------------------------------


@pytest.mark.asyncio
async def test_a_normal_turn_completion_reports_waiting_for_input(adapter):
    """Covers P0 fix: TurnCompletedNotification previously fell through
    unhandled, so a phone never learned an ordinary Codex turn had
    finished."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started
    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    turn_handle = adapter._test_clients["latest"].threads[0].turns[0]
    turn = Turn(id="turn-1", items=[], status=TurnStatus.completed)
    turn_handle.push(TurnCompletedNotification(threadId="t1", turn=turn))

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "waiting_for_input"


@pytest.mark.asyncio
async def test_interrupt_is_a_clean_noop_when_the_session_is_already_gone(adapter):
    """Covers P2 fix: interrupt() must not reintroduce the raising-_get()
    KeyError SDKAdapter's own equivalent regression test exists to pin -
    a Cancel racing an already-completed End Session must be a silent
    no-op, not an exception."""
    await adapter.connect("s1", cwd="/repo")
    await adapter.disconnect("s1")

    await adapter.interrupt("s1")  # must not raise


@pytest.mark.asyncio
async def test_disconnect_still_ends_the_session_when_client_close_raises(adapter):
    """Covers P1 fix: a failing client.close() must not skip session.end(),
    or nothing ever emits session_ended and subscribers hang."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    client = adapter._test_clients["latest"]
    client.raise_on_close = RuntimeError("connection reset")

    await adapter.disconnect("s1")

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "session_ended"


@pytest.mark.asyncio
async def test_send_message_reports_an_error_event_when_turn_start_fails(adapter):
    """Covers P1 fix: a thread.turn() failure must not vanish silently
    after the message already shows as sent."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    thread = adapter._test_clients["latest"].threads[0]
    thread.raise_on_turn = RuntimeError("turn failed to start")

    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "error"
    assert "turn failed to start" in event.data["message"]


@pytest.mark.asyncio
async def test_a_second_send_message_cancels_the_previous_turns_still_running_stream_task(adapter):
    """Covers P1 fix: a still-streaming previous turn's background task
    must not be silently orphaned when a new turn starts."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.send_message("s1", "first")
    await events.__anext__()  # user_message
    session = adapter._sessions["s1"]
    first_stream_task = session.stream_task

    await adapter.send_message("s1", "second")
    await events.__anext__()  # user_message

    assert first_stream_task.cancelled() or first_stream_task.cancelling() > 0
    assert session.stream_task is not first_stream_task


@pytest.mark.asyncio
async def test_compact_reports_an_error_event_on_failure_instead_of_raising(adapter):
    """Covers P2 fix: compact() previously had zero error handling."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    thread = adapter._test_clients["latest"].threads[0]
    thread.raise_on_compact = RuntimeError("compact failed")

    await adapter.compact("s1")  # must not raise

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "error"
    assert "compact failed" in event.data["message"]


@pytest.mark.asyncio
async def test_connect_closes_the_client_and_reraises_when_thread_start_fails(adapter):
    """Covers P2 fix: a failed thread_start() previously leaked the
    already-constructed client with nothing left to close it."""
    clients: dict[str, FakeCodexClient] = {}

    def factory():
        client = FakeCodexClient()
        client.raise_on_thread_start = RuntimeError("thread_start failed")
        clients["latest"] = client
        return client

    failing_adapter = CodexAdapter(client_factory=factory)

    with pytest.raises(RuntimeError, match="thread_start failed"):
        await failing_adapter.connect("s1", cwd="/repo")

    assert clients["latest"].close_calls == 1


@pytest.mark.asyncio
async def test_unsupported_operation_reasons_differ_between_adapters():
    """Covers maintainability fix: UnsupportedOperation is now shared from
    base.py (no duplicate class), with each adapter passing its own
    explicit reason rather than relying on a class-level default."""
    from companion.adapters.observe_adapter import ObserveAdapter

    observe_result = await ObserveAdapter().send_message("s1", "hi")
    codex_adapter = CodexAdapter(client_factory=lambda: FakeCodexClient())
    await codex_adapter.connect("s1", cwd="/repo")
    codex_result = await codex_adapter.respond_to_permission("s1", "req-1", "allow")

    assert observe_result.reason == "not supported for observed sessions"
    assert codex_result.reason == "not supported for Codex sessions"


# --- Codex Model & Sandbox Config plan (2026-08-27-001), U1 --------------


@pytest.mark.asyncio
async def test_connect_converts_sandbox_wire_string_to_enum_by_name(adapter):
    """The SDK's real Sandbox enum values are hyphenated
    ("workspace-write") while this app's wire strings are the underscored
    member names ("workspace_write") - a value-based Sandbox(value) lookup
    raises ValueError for every valid wire string. This asserts the exact
    enum member reached thread_start(), so a regression back to
    value-based lookup fails the test rather than passing on a
    coincidentally-equal string."""
    await adapter.connect("s1", cwd="/repo", sandbox="workspace_write")

    client = adapter._test_clients["latest"]
    assert client.thread_start_kwargs[0]["sandbox"] is Sandbox.workspace_write


@pytest.mark.asyncio
async def test_set_session_model_applies_starting_with_the_next_send_message(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    result = adapter.set_session_model("s1", "gpt-5.1-codex-mini")
    assert result is True

    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    thread = adapter._test_clients["latest"].threads[0]
    assert thread.turn_kwargs[-1]["model"] == "gpt-5.1-codex-mini"


@pytest.mark.asyncio
async def test_set_session_sandbox_applies_starting_with_the_next_send_message(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    result = adapter.set_session_sandbox("s1", "full_access")
    assert result is True

    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    thread = adapter._test_clients["latest"].threads[0]
    assert thread.turn_kwargs[-1]["sandbox"] is Sandbox.full_access


@pytest.mark.asyncio
async def test_connect_with_an_unrecognized_sandbox_string_raises_and_propagates(adapter):
    """Fails closed at the daemon.py layer (start_session's existing outer
    exception handler) - not silently swallowed inside connect()."""
    with pytest.raises(KeyError):
        await adapter.connect("s1", cwd="/repo", sandbox="not-a-real-value")


@pytest.mark.asyncio
async def test_set_session_sandbox_with_an_unrecognized_string_returns_false_and_leaves_sandbox_unchanged(adapter):
    await adapter.connect("s1", cwd="/repo", sandbox="read_only")

    result = adapter.set_session_sandbox("s1", "not-a-real-value")

    assert result is False
    session = adapter._sessions["s1"]
    assert session.sandbox is Sandbox.read_only


@pytest.mark.asyncio
async def test_set_session_sandbox_with_none_returns_false_and_leaves_sandbox_unchanged(adapter):
    """code-review finding: the simplify pass's shared _sandbox_from_wire()
    helper returns None (no exception) for a None input - correct for
    connect()'s own "no sandbox requested" semantics, but set_session_sandbox
    must still fail closed for None here (matching the unrecognized-string
    case above), not silently clear session.sandbox to None and report
    success - that would be a fail-open regression on a safety-relevant
    control the next turn() call would inherit."""
    await adapter.connect("s1", cwd="/repo", sandbox="read_only")

    result = adapter.set_session_sandbox("s1", None)

    assert result is False
    session = adapter._sessions["s1"]
    assert session.sandbox is Sandbox.read_only


@pytest.mark.asyncio
async def test_set_session_model_against_an_unknown_session_id_returns_false(adapter):
    result = adapter.set_session_model("does-not-exist", "gpt-5.1-codex-mini")

    assert result is False


@pytest.mark.asyncio
async def test_set_session_sandbox_against_an_unknown_session_id_returns_false(adapter):
    result = adapter.set_session_sandbox("does-not-exist", "full_access")

    assert result is False


@pytest.mark.asyncio
async def test_connect_with_no_model_or_sandbox_still_calls_turn_with_both_none(adapter):
    """Regression: a session connected with neither param set keeps today's
    behavior - turn() is still called, now explicitly carrying
    model=None, sandbox=None rather than omitting the kwargs."""
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.send_message("s1", "hello")
    await events.__anext__()  # user_message

    thread = adapter._test_clients["latest"].threads[0]
    assert thread.turn_kwargs[-1]["model"] is None
    assert thread.turn_kwargs[-1]["sandbox"] is None


@pytest.mark.asyncio
async def test_set_session_model_emits_a_session_model_event(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    adapter.set_session_model("s1", "gpt-5.1-codex-mini")

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "session_model"
    assert event.data["model"] == "gpt-5.1-codex-mini"


@pytest.mark.asyncio
async def test_set_session_sandbox_emits_a_session_sandbox_event(adapter):
    await adapter.connect("s1", cwd="/repo")
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    adapter.set_session_sandbox("s1", "full_access")

    event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert event.type == "session_sandbox"
    assert event.data["sandbox"] == "full_access"
