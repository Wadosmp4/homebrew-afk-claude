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
    RateLimitEvent,
    RateLimitInfo,
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
async def test_connect_passes_the_requested_model_to_claude_agent_options(adapter):
    await adapter.connect("s1", model="claude-opus-5")

    client = adapter._test_clients["latest"]
    assert client.options.model == "claude-opus-5"

    event, _gen = await _next_event(adapter, "s1")
    assert event.data["model"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_connect_with_no_model_leaves_claude_agent_options_default(adapter):
    await adapter.connect("s1")

    client = adapter._test_clients["latest"]
    assert client.options.model is None


@pytest.mark.asyncio
async def test_connect_with_resume_passes_the_same_session_id_through_to_claude_agent_options(adapter):
    """daemon.py's _try_resume_sdk_session reconnects a session_id that
    survived only as a transcript on disk - fork_session defaults to
    False, so this must keep the same session_id rather than forking a
    new one, or the phone's existing reference to it would go stale too."""
    await adapter.connect("s1", resume="s1")

    client = adapter._test_clients["latest"]
    assert client.options.resume == "s1"


@pytest.mark.asyncio
async def test_connect_with_no_resume_leaves_claude_agent_options_default(adapter):
    await adapter.connect("s1")

    client = adapter._test_clients["latest"]
    assert client.options.resume is None


@pytest.mark.asyncio
async def test_connect_with_no_resume_forces_the_daemons_session_id_on_the_cli(adapter):
    """Root-cause fix: without this, the CLI subprocess spawned for a fresh
    session invents its own internal session id and writes its transcript
    under that different id, so every transcript-based lookup keyed on the
    daemon's own session_id (companion/history.py) silently finds nothing -
    confirmed via real-device testing (a session tracked as one uuid had its
    real transcript filed under a completely different one)."""
    await adapter.connect("s1")

    client = adapter._test_clients["latest"]
    assert client.options.session_id == "s1"


@pytest.mark.asyncio
async def test_connect_with_resume_leaves_claude_agent_options_session_id_unset(adapter):
    """claude_agent_sdk's own ClaudeAgentOptions.session_id docstring: this
    field "Cannot be used with continue_conversation or resume unless
    fork_session is also set" - and fork_session forks the resumed
    conversation onto a *new* id, which is exactly wrong here (this
    adapter's own bookkeeping keys on the caller-supplied session_id staying
    the same across a resume). Combining the two would either be rejected
    by the CLI or silently fork - so session_id must stay unset whenever
    resume is set; `resume` alone already resolves to the correct
    transcript once a fresh session's id lines up per the test above."""
    await adapter.connect("s1", resume="s1")

    client = adapter._test_clients["latest"]
    assert client.options.session_id is None


@pytest.mark.asyncio
async def test_connect_passes_the_configured_cli_path_to_claude_agent_options(adapter):
    """U4/KTD5: a Mac-level CLI binary/profile setting, threaded straight
    into ClaudeAgentOptions.cli_path."""
    await adapter.connect("s1", cli_path="/usr/local/bin/claude-custom")

    client = adapter._test_clients["latest"]
    assert client.options.cli_path == "/usr/local/bin/claude-custom"


@pytest.mark.asyncio
async def test_connect_with_no_cli_path_leaves_claude_agent_options_default(adapter):
    await adapter.connect("s1")

    client = adapter._test_clients["latest"]
    assert client.options.cli_path is None


@pytest.mark.asyncio
async def test_connect_passes_the_configured_cli_env_to_claude_agent_options(adapter):
    await adapter.connect("s1", cli_env={"ANTHROPIC_API_KEY": "sk-test"})

    client = adapter._test_clients["latest"]
    assert client.options.env == {"ANTHROPIC_API_KEY": "sk-test"}


@pytest.mark.asyncio
async def test_connect_with_no_cli_env_passes_an_empty_dict_never_none(adapter):
    """CRITICAL regression coverage: ClaudeAgentOptions.env is dict-typed
    (default_factory=dict), not Optional, on the SDK side - the real
    subprocess transport unconditionally dict-unpacks it (**self._options.env)
    when building the child process's environment. Passing env=None would
    raise a TypeError on every single session start, not just when a CLI
    profile is configured. This must stay a dict even when cli_env is
    unset - never collapse an empty cli_env to None anywhere in the chain."""
    await adapter.connect("s1")

    client = adapter._test_clients["latest"]
    assert client.options.env is not None
    assert client.options.env == {}
    assert isinstance(client.options.env, dict)


@pytest.mark.asyncio
async def test_connect_resolves_a_symlinked_cwd(adapter, tmp_path):
    """Regression test: the real `claude` CLI subprocess resolves symlinks
    before recording its transcript's own cwd (confirmed empirically -
    macOS's own /tmp resolves to /private/tmp). session.cwd (exposed via
    get_cwd, and what later feeds companion/history.py's transcript
    lookup) must agree with that resolved form, or a real session's
    history becomes unfindable purely from a path-string mismatch - see
    companion/tests/test_history.py's matching regression test."""
    real_dir = tmp_path / "real-project"
    real_dir.mkdir()
    symlinked_dir = tmp_path / "project-via-symlink"
    symlinked_dir.symlink_to(real_dir)

    await adapter.connect("s1", cwd=str(symlinked_dir))

    assert adapter.get_cwd("s1") == str(real_dir.resolve())


@pytest.mark.asyncio
async def test_connect_snapshots_auto_approve_and_llm_judge_into_session_started(adapter):
    """A phone-started session's auto_approve/llm_judge are fixed at
    connect() time (opt-in kwargs, defaults off) and echoed back in
    session_started so the phone knows the state without a round-trip -
    see set_session_auto_approve for how it can change later."""
    await adapter.connect("s1", auto_approve=True, llm_judge=True)

    event, _gen = await _next_event(adapter, "s1")
    assert event.data["auto_approve"] is True
    assert event.data["llm_judge"] is True


@pytest.mark.asyncio
async def test_set_session_auto_approve_overrides_a_connected_sessions_state(adapter):
    await adapter.connect("s1", auto_approve=False, llm_judge=False)
    event, gen = await _next_event(adapter, "s1")
    assert event.type == "session_started"

    assert adapter.set_session_auto_approve("s1", auto_approve=True) is True

    confirm = await gen.__anext__()
    assert confirm.type == "session_auto_approve"
    assert confirm.data["auto_approve"] is True
    assert confirm.data["llm_judge"] is False  # untouched - only auto_approve was passed


@pytest.mark.asyncio
async def test_set_session_auto_approve_none_leaves_that_field_untouched(adapter):
    await adapter.connect("s1", auto_approve=True, llm_judge=True)
    event, gen = await _next_event(adapter, "s1")
    assert event.type == "session_started"

    adapter.set_session_auto_approve("s1", auto_approve=None, llm_judge=False)

    confirm = await gen.__anext__()
    assert confirm.type == "session_auto_approve"
    assert confirm.data["auto_approve"] is True  # untouched
    assert confirm.data["llm_judge"] is False


@pytest.mark.asyncio
async def test_set_session_auto_approve_for_unknown_session_returns_false_without_raising(adapter):
    assert adapter.set_session_auto_approve("no-such-session", auto_approve=True) is False


@pytest.mark.asyncio
async def test_context_usage_is_emitted_after_each_completed_turn(adapter):
    """SDKAdapter._emit_context_usage polls ClaudeSDKClient.get_context_usage()
    once a turn finishes (ResultMessage) - matches what the CLI's own
    /context command would report at that point."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]

    async def fake_get_context_usage():
        return {
            "totalTokens": 5000,
            "maxTokens": 200000,
            "percentage": 2.5,
            "isAutoCompactEnabled": True,
            "autoCompactThreshold": 180000,
            "model": "claude-opus-5",
        }

    client.get_context_usage = fake_get_context_usage

    events = adapter.subscribe("s1")
    started = await events.__anext__()
    assert started.type == "session_started"

    client.push(
        ResultMessage(
            subtype="success", duration_ms=10, duration_api_ms=8, is_error=False, num_turns=1, session_id="s1"
        )
    )

    # waiting_for_input (the turn's own lifecycle signal) goes out first,
    # ahead of the context usage poll - the phone shouldn't wait on an
    # extra round-trip just to know Claude is ready for the next message.
    waiting_event = await events.__anext__()
    assert waiting_event.type == "waiting_for_input"

    usage_event = await events.__anext__()
    assert usage_event.type == "context_usage"
    assert usage_event.data["percentage"] == 2.5
    assert usage_event.data["total_tokens"] == 5000
    assert usage_event.data["max_tokens"] == 200000
    assert usage_event.data["is_auto_compact_enabled"] is True
    assert usage_event.data["auto_compact_threshold"] == 180000
    assert usage_event.data["model"] == "claude-opus-5"


@pytest.mark.asyncio
async def test_context_usage_poll_failure_does_not_crash_the_read_loop(adapter):
    """FakeSDKClient doesn't implement get_context_usage by default (same
    as an older real CLI that predates this control request) - the read
    loop must swallow that and keep the session alive rather than crash
    over what's a nice-to-have status update."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    started = await events.__anext__()
    assert started.type == "session_started"

    client.push(
        ResultMessage(
            subtype="success", duration_ms=10, duration_api_ms=8, is_error=False, num_turns=1, session_id="s1"
        )
    )
    waiting_event = await events.__anext__()
    assert waiting_event.type == "waiting_for_input"

    # No context_usage follows, and the session is still alive - confirmed
    # by sending another message and getting a reply, rather than racing
    # events.__anext__() with a timeout.
    await adapter.send_message("s1", "still alive?")
    user_event = await events.__anext__()
    assert user_event.type == "user_message"


@pytest.mark.asyncio
async def test_rate_limit_event_is_emitted_when_the_cli_reports_one(adapter):
    """The CLI pushes a RateLimitEvent whenever a rate-limit window's status
    changes (not on a fixed poll, unlike context_usage) - the adapter just
    has to forward it, normalized into the shared event model."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    started = await events.__anext__()
    assert started.type == "session_started"

    client.push(
        RateLimitEvent(
            rate_limit_info=RateLimitInfo(
                status="allowed_warning",
                rate_limit_type="five_hour",
                utilization=0.82,
                resets_at=1700000000,
            ),
            uuid="rl-1",
            session_id="s1",
        )
    )

    rate_limit_event = await events.__anext__()
    assert rate_limit_event.type == "rate_limit"
    assert rate_limit_event.data["rate_limit_type"] == "five_hour"
    assert rate_limit_event.data["status"] == "allowed_warning"
    assert rate_limit_event.data["utilization"] == 0.82
    assert rate_limit_event.data["resets_at"] == 1700000000


@pytest.mark.asyncio
async def test_compact_sends_slash_compact_without_a_user_message_event(adapter):
    """Unlike send_message, compact must not show up as a fake chat bubble
    - it goes straight to the outbound queue that feeds prompt_stream(),
    bypassing send_message's own user_message emit entirely."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    started = await events.__anext__()
    assert started.type == "session_started"

    await adapter.compact("s1")

    # Drains the same generator connect() was given - exercises the real
    # prompt_stream() shape rather than reaching into private state.
    queued = await client.connected_prompt.__anext__()
    assert queued["message"]["content"] == "/compact"


@pytest.mark.asyncio
async def test_compact_for_unknown_session_raises_key_error(adapter):
    with pytest.raises(KeyError):
        await adapter.compact("no-such-session")


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
async def test_large_tool_result_content_is_truncated(adapter):
    """U9 (R14): a very large tool result (e.g. a long test run's output)
    is truncated with an explicit marker rather than reaching the relay's
    bounded event cache untruncated."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    client.push(
        AssistantMessage(
            content=[ToolUseBlock(id="tool-1", name="Bash", input={"command": "pytest"})], model="claude"
        )
    )
    await events.__anext__()  # tool_call

    huge_output = "PASS\n" * 5000  # well over the truncation threshold
    client.push(
        AssistantMessage(
            content=[ToolResultBlock(tool_use_id="tool-1", content=huge_output, is_error=False)], model="claude"
        )
    )
    result_event = await events.__anext__()

    assert len(result_event.data["content"]) < len(huge_output)
    assert "truncated" in result_event.data["content"]


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
async def test_respond_to_permission_allow_emits_a_durable_resolution_event(adapter):
    """U1 (connection-resilience plan): resolving a permission must be a
    durable, replayable fact (R5/R6), not only held in the phone's own
    memory - respond_to_permission's existing allow path now also emits
    permission_resolved, in order right after permission_request."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext(tool_use_id="tool-9"))
    )
    request_event = await events.__anext__()
    assert request_event.type == "permission_request"

    await adapter.respond_to_permission("s1", "tool-9", "allow")

    resolved_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["request_id"] == "tool-9"
    assert resolved_event.data["decision"] == "allow"


@pytest.mark.asyncio
async def test_respond_to_permission_deny_emits_a_durable_resolution_event(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request

    await adapter.respond_to_permission("s1", "tool-9", "deny", message="not allowed")

    resolved_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["decision"] == "deny"
    assert resolved_event.data["message"] == "not allowed"


@pytest.mark.asyncio
async def test_structured_question_answer_carries_through_onto_the_resolution_event(adapter):
    """The chosen option text (sent as decision="allow", message=<option>)
    must reach permission_resolved verbatim - resolve_permission emits
    before can_use_tool's own structured-question reinterpretation of the
    Future's result, so the durable record reflects what the phone
    actually sent, not the internal deny-translation."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    tool_input = {"questions": [{"question": "Red or blue?", "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", tool_input, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request

    await adapter.respond_to_permission("s1", "tool-9", "allow", message="Red")

    resolved_event = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["decision"] == "allow"
    assert resolved_event.data["message"] == "Red"


def test_sdk_adapter_no_longer_uses_the_removed_approval_stack():
    """Regression test (permission-mode-picker plan U1): can_use_tool no
    longer calls into the rule-based denylist/allowlist
    (auto_approve.is_denylisted/is_auto_approvable), the AI judge
    (risk_judge.judge_is_safe), or Risk Explanation (risk_judge.explain_risk).
    Checked against the module's actual namespace rather than a raw source
    substring search - a docstring here is allowed to keep *mentioning*
    risk_judge.judge_is_safe/explain_risk by name to explain what was
    removed and why (harmless prose), so the real regression signal is
    that the module itself no longer *binds* either name: no more
    `from .. import risk_judge` (only observe_adapter.py imports it now)
    and no more `from .. import auto_approve as approval_policy` (only a
    direct, still-needed `is_structured_question` import survives, for the
    untouched structured-question handling below in can_use_tool).
    `auto_approve` as a bare word still appears elsewhere in this module
    (connect()'s own opt-in kwarg, session_started's echo of it,
    set_session_auto_approve) - none of that is in this unit's scope; only
    the approval-stack calls actually reachable from can_use_tool are gone.
    observe_adapter.py's own, separate use of auto_approve.py/judge_is_safe
    (R11) is untouched - see test_observe_adapter.py's own unmodified suite
    for that coverage."""
    from companion.adapters import sdk_adapter

    assert not hasattr(sdk_adapter, "risk_judge")
    assert not hasattr(sdk_adapter, "approval_policy")
    assert sdk_adapter._is_structured_question is not None


@pytest.mark.asyncio
async def test_a_tool_allowed_once_is_auto_allowed_on_later_calls_this_session(adapter):
    """Once the phone allows a tool once, later calls to that same tool
    name skip the prompt entirely for the rest of the session - approving
    one Edit shouldn't mean re-approving every subsequent edit."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    first_call = asyncio.create_task(
        client.options.can_use_tool("Edit", {"file_path": "a.py"}, ToolPermissionContext(tool_use_id="tool-1"))
    )
    first_permission = await events.__anext__()
    assert first_permission.type == "permission_request"
    await adapter.respond_to_permission("s1", "tool-1", "allow")
    first_result = await asyncio.wait_for(first_call, timeout=1)
    assert first_result.behavior == "allow"

    # Second Edit call: no permission_request this time - resolves
    # immediately without the phone doing anything.
    second_result = await asyncio.wait_for(
        client.options.can_use_tool("Edit", {"file_path": "b.py"}, ToolPermissionContext(tool_use_id="tool-2")),
        timeout=1,
    )
    assert second_result.behavior == "allow"


@pytest.mark.asyncio
async def test_auto_allow_is_scoped_to_the_specific_tool_name(adapter):
    """Allowing Edit doesn't auto-allow Bash - the shortcut is per tool
    name, not "the phone said allow once, allow everything forever"."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    edit_call = asyncio.create_task(
        client.options.can_use_tool("Edit", {"file_path": "a.py"}, ToolPermissionContext(tool_use_id="tool-1"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-1", "allow")
    await events.__anext__()  # permission_resolved
    await asyncio.wait_for(edit_call, timeout=1)

    bash_call = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext(tool_use_id="tool-2"))
    )
    bash_permission = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert bash_permission.type == "permission_request"
    assert bash_permission.data["tool"] == "Bash"
    await adapter.respond_to_permission("s1", "tool-2", "allow")
    await asyncio.wait_for(bash_call, timeout=1)


@pytest.mark.asyncio
async def test_denying_a_tool_does_not_auto_allow_it_later(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    first_call = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="tool-1"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-1", "deny", message="no")
    await events.__anext__()  # permission_resolved
    first_result = await asyncio.wait_for(first_call, timeout=1)
    assert first_result.behavior == "deny"

    second_call = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "ls"}, ToolPermissionContext(tool_use_id="tool-2"))
    )
    second_permission = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert second_permission.type == "permission_request"  # still prompts
    await adapter.respond_to_permission("s1", "tool-2", "allow")
    await asyncio.wait_for(second_call, timeout=1)


@pytest.mark.asyncio
async def test_structured_questions_are_never_auto_allowed(adapter):
    """Each AskUserQuestion has different content, so "allowed" never
    applies to it as a tool-name category the way it does for Bash/Edit -
    every one must still prompt even after a prior one was answered."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    first_input = {"questions": [{"question": "Red or blue?", "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    first_call = asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", first_input, ToolPermissionContext(tool_use_id="tool-1"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-1", "allow", message="Red")
    await events.__anext__()  # permission_resolved
    await asyncio.wait_for(first_call, timeout=1)

    second_input = {"questions": [{"question": "Cat or dog?", "options": [{"label": "Cat"}, {"label": "Dog"}]}]}
    second_call = asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", second_input, ToolPermissionContext(tool_use_id="tool-2"))
    )
    second_permission = await asyncio.wait_for(events.__anext__(), timeout=1)
    assert second_permission.type == "permission_request"  # still prompts
    await adapter.respond_to_permission("s1", "tool-2", "allow", message="Cat")
    second_result = await asyncio.wait_for(second_call, timeout=1)
    assert second_result.message == "User answered: Cat"


@pytest.mark.asyncio
async def test_structured_question_answer_rides_back_as_a_deny_reason(adapter):
    """Regression test: AskUserQuestion can't be truly "allowed" through in
    a headless SDK session (no interactive terminal for it to read a real
    answer from - allowing it just returns empty). The phone's chosen
    option must come back as the deny-reason message instead, which is
    the only channel this callback has for returning free-form text."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    tool_input = {"questions": [{"question": "Red or blue?", "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    call_task = asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", tool_input, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request

    # The mobile client always sends decision="allow" for a tapped option
    # (StructuredOptions.tsx) - the fix is that this callback must not
    # honor that literally for a structured question.
    await adapter.respond_to_permission("s1", "tool-9", "allow", message="Red")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "deny"
    assert result.message == "User answered: Red"


@pytest.mark.asyncio
async def test_structured_question_with_no_answer_falls_back_to_plain_allow_deny(adapter):
    """A regular Allow/Deny tap (no chosen option) on a structured-question
    tool call - e.g. the raw permission card, if ever shown for one - still
    behaves like any other tool: allow really allows."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    tool_input = {"questions": [{"question": "Red or blue?", "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    call_task = asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", tool_input, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-9", "allow")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "allow"


@pytest.mark.asyncio
async def test_structured_question_deny_with_a_reason_is_not_misread_as_an_answer(adapter):
    """Regression test: a real deny-with-reason on a structured question
    (message set, decision == "deny") must fall through to the plain deny
    path with its own reason intact - the earlier `if _is_structured_question(...)
    and message` check (no decision guard) would have rewritten a deny
    reason as "User answered: <deny reason>", misrepresenting a rejection
    as an answered question."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    from claude_agent_sdk import ToolPermissionContext

    tool_input = {"questions": [{"question": "Red or blue?", "options": [{"label": "Red"}, {"label": "Blue"}]}]}
    call_task = asyncio.create_task(
        client.options.can_use_tool("AskUserQuestion", tool_input, ToolPermissionContext(tool_use_id="tool-9"))
    )
    await events.__anext__()  # permission_request
    await adapter.respond_to_permission("s1", "tool-9", "deny", message="not right now")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "deny"
    assert result.message == "not right now"


@pytest.mark.asyncio
async def test_interrupt_cancels_the_turn_but_leaves_the_session_open(adapter):
    """U5: Cancel is turn-only now - unlike disconnect() (see below), it
    must not end the session, so a subsequent send_message still works."""
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.interrupt("s1")
    assert client.interrupted is True

    await adapter.send_message("s1", "still there?")
    user_message_event = await events.__anext__()
    assert user_message_event.type == "user_message"
    assert user_message_event.data["text"] == "still there?"


@pytest.mark.asyncio
async def test_interrupt_is_a_clean_noop_when_end_session_already_won_the_race(adapter):
    """Regression test: daemon.py dispatches every phone action as its own
    unordered asyncio.create_task - if a concurrently-dispatched End
    Session's disconnect() pops the session first, a losing Cancel tap's
    interrupt() must not raise (previously KeyError via _get(), silently
    swallowed by daemon.py with no signal back to the phone)."""
    await adapter.connect("s1")
    await adapter.disconnect("s1")

    await adapter.interrupt("s1")  # must not raise


@pytest.mark.asyncio
async def test_disconnect_stops_session_and_emits_lifecycle_event(adapter):
    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    await adapter.disconnect("s1")
    assert client.disconnected is True

    lifecycle_event = await events.__anext__()
    assert lifecycle_event.type == "session_ended"
    assert lifecycle_event.data["reason"] == "disconnected"


@pytest.mark.asyncio
async def test_disconnect_denies_a_pending_permission_request_first(adapter):
    """F3/R5: ending a session must durably deny any still-open permission
    request on it, same as interrupt()'s own deny-pending loop."""
    from claude_agent_sdk import ToolPermissionContext

    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    call_task = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="tool-1"))
    )
    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"

    await adapter.disconnect("s1")

    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "deny"

    resolved_event = await events.__anext__()
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["request_id"] == "tool-1"
    assert resolved_event.data["decision"] == "deny"

    lifecycle_event = await events.__anext__()
    assert lifecycle_event.type == "session_ended"
    assert lifecycle_event.data["reason"] == "disconnected"


@pytest.mark.asyncio
async def test_is_active_is_true_after_interrupt_since_the_session_stays_open(adapter):
    """U5: interrupt() (unlike disconnect()) no longer ends the session or
    removes it from _sessions - a cancelled turn still reads as active."""
    await adapter.connect("s1")
    assert adapter.is_active("s1") is True

    await adapter.interrupt("s1")

    assert "s1" in adapter.discover_sessions()
    assert adapter.is_active("s1") is True


@pytest.mark.asyncio
async def test_is_active_is_false_after_disconnect(adapter):
    await adapter.connect("s1")
    assert adapter.is_active("s1") is True

    await adapter.disconnect("s1")

    assert "s1" not in adapter.discover_sessions()
    assert adapter.is_active("s1") is None


@pytest.mark.asyncio
async def test_is_active_for_unknown_session_returns_none(adapter):
    assert adapter.is_active("no-such-session") is None


@pytest.mark.asyncio
async def test_interrupt_denies_a_pending_permission_request_first(adapter):
    """Regression test: tapping Stop while a tool call (e.g. a structured
    question) is still awaiting an answer used to leave can_use_tool's own
    Future dangling through the interrupt, which the real CLI reported
    back as a ResultMessage with subtype error_during_execution instead of
    a clean cancellation. interrupt() must resolve it (deny) first."""
    from claude_agent_sdk import ToolPermissionContext

    await adapter.connect("s1")
    client = adapter._test_clients["latest"]
    events = adapter.subscribe("s1")
    await events.__anext__()  # session_started

    call_task = asyncio.create_task(
        client.options.can_use_tool("Bash", {"command": "rm -rf /"}, ToolPermissionContext(tool_use_id="tool-1"))
    )
    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"

    await adapter.interrupt("s1")

    # can_use_tool returns (denied) rather than staying blocked through
    # the interrupt.
    result = await asyncio.wait_for(call_task, timeout=1)
    assert result.behavior == "deny"

    # U1 (connection-resilience plan): the deny-pending loop's own
    # resolve_permission call durably records the denial too.
    resolved_event = await events.__anext__()
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["request_id"] == "tool-1"
    assert resolved_event.data["decision"] == "deny"

    # U5: Cancel is turn-only now - no session_ended follows.
    assert adapter.is_active("s1") is True


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
