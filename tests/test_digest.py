"""Tests for companion/digest.py - the real claude_agent_sdk.query() is
never invoked here (that would spawn a real `claude` CLI subprocess);
`generate_digest`'s `query_fn` parameter is injected with a fake async
generator that yields the same message shapes the real SDK does, mirroring
test_risk_judge.py's own convention. The direct-API fast path is exercised
the same way, via an injected fake `api_client`.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from companion import digest, risk_judge


@pytest.fixture(autouse=True)
def _reset_api_client_cache():
    """get_api_client() (imported from risk_judge) caches its resolved
    client at module scope - reset it before and after every test here so
    one test's state never leaks into the next, same reasoning as
    test_risk_judge.py's own fixture of the same name."""
    risk_judge._api_client = None
    yield
    risk_judge._api_client = None


def _fake_query(response_text: str, *, delay: float = 0.0):
    async def query_fn(*, prompt, options):
        if delay:
            await asyncio.sleep(delay)
        yield AssistantMessage(content=[TextBlock(text=response_text)], model="claude")

    return query_fn


class _FakeApiTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeApiResponse:
    def __init__(self, text: str):
        self.content = [_FakeApiTextBlock(text)]


class _FakeApiClient:
    def __init__(self, *, response_text=None, raises=None, delay: float = 0.0):
        self._response_text = response_text
        self._raises = raises
        self._delay = delay
        self.calls: list[dict] = []
        self.messages = self

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._raises:
            raise self._raises
        return _FakeApiResponse(self._response_text)


_SAMPLE_EVENTS = [
    {"type": "assistant_message", "timestamp": "t1", "data": {"text": "Adding tests for the parser"}},
    {"type": "tool_call", "timestamp": "t2", "data": {"tool": "Bash", "input": {"command": "npm test"}}},
    {"type": "tool_result", "timestamp": "t3", "data": {"content": "3 passed"}},
]


@pytest.mark.asyncio
async def test_a_successful_response_returns_the_summary_text():
    result = await digest.generate_digest(_SAMPLE_EVENTS, query_fn=_fake_query("Added tests, all passing."))
    assert result == "Added tests, all passing."


@pytest.mark.asyncio
async def test_an_exception_from_query_fails_closed_to_none():
    async def failing_query(*, prompt, options):
        raise RuntimeError("subprocess spawn failed")
        yield  # pragma: no cover - makes this a generator function

    result = await digest.generate_digest(_SAMPLE_EVENTS, query_fn=failing_query)
    assert result is None


@pytest.mark.asyncio
async def test_a_timeout_fails_closed_to_none():
    result = await digest.generate_digest(
        _SAMPLE_EVENTS, timeout_seconds=0.05, query_fn=_fake_query("slow summary", delay=1.0)
    )
    assert result is None


@pytest.mark.asyncio
async def test_an_empty_response_fails_closed_to_none():
    result = await digest.generate_digest(_SAMPLE_EVENTS, query_fn=_fake_query(""))
    assert result is None


@pytest.mark.asyncio
async def test_no_tools_are_offered_to_the_digest_session():
    """Same reasoning as risk_judge's own judge session - this summarizes,
    it never acts."""
    captured = {}

    async def query_fn(*, prompt, options):
        captured["tools"] = options.tools
        yield AssistantMessage(content=[TextBlock(text="a summary")], model="claude")

    await digest.generate_digest(_SAMPLE_EVENTS, query_fn=query_fn)
    assert captured["tools"] == []


@pytest.mark.asyncio
async def test_a_very_long_event_list_is_bounded_to_the_tail_in_the_prompt():
    long_events = [
        {"type": "tool_call", "timestamp": f"t{i}", "data": {"tool": "Bash", "input": {"command": f"cmd-{i}"}}}
        for i in range(500)
    ]
    captured = {}

    async def query_fn(*, prompt, options):
        captured["prompt"] = prompt
        yield AssistantMessage(content=[TextBlock(text="a summary")], model="claude")

    await digest.generate_digest(long_events, query_fn=query_fn)

    # The bound keeps the prompt from growing unbounded with session length -
    # the earliest events (cmd-0) must be dropped in favor of the most
    # recent ones (cmd-499), not the other way around.
    assert "cmd-499" in captured["prompt"]
    assert "cmd-0" not in captured["prompt"]


@pytest.mark.asyncio
async def test_a_safe_verdict_via_the_api_fast_path_returns_the_text_without_touching_the_cli_path():
    def cli_path_should_not_be_called(*, prompt, options):
        raise AssertionError("the CLI-subprocess path must not run when the API fast path succeeds")
        yield  # pragma: no cover - makes this a generator function

    api_client = _FakeApiClient(response_text="Added tests via the API path.")

    result = await digest.generate_digest(
        _SAMPLE_EVENTS, query_fn=cli_path_should_not_be_called, api_client=api_client
    )

    assert result == "Added tests via the API path."
    assert len(api_client.calls) == 1


@pytest.mark.asyncio
async def test_an_api_failure_falls_back_to_the_cli_subprocess_path():
    api_client = _FakeApiClient(raises=RuntimeError("connection reset"))

    result = await digest.generate_digest(
        _SAMPLE_EVENTS, query_fn=_fake_query("Added tests via the CLI path."), api_client=api_client
    )

    assert result == "Added tests via the CLI path."
    assert len(api_client.calls) == 1


@pytest.mark.asyncio
async def test_without_an_api_client_argument_the_cli_subprocess_path_runs_as_before():
    result = await digest.generate_digest(_SAMPLE_EVENTS, query_fn=_fake_query("A plain CLI-path summary."))
    assert result == "A plain CLI-path summary."


@pytest.mark.asyncio
async def test_the_configured_cli_env_is_threaded_into_the_cli_subprocess_call(monkeypatch):
    """fix(review): discovered via real-device testing - generate_digest's
    ClaudeAgentOptions never carried an `env` at all, so its one-shot CLI
    subprocess got none of _effective_cli_env()'s personal-account
    CLAUDE_CODE_OAUTH_TOKEN override - it fell back to the CLI's own
    default keychain session instead, which can be (and, live, was)
    expired even though the same account's token works fine for a real
    session's own connect(). "Catch me up" specifically failed with an
    auth error while regular messaging worked, because only send_message's
    path (SDKAdapter.connect) ever passed cli_env through."""
    captured_env = {}

    async def _capturing_query(*, prompt, options):
        captured_env.update(options.env)
        yield AssistantMessage(content=[TextBlock(text="summary")], model="claude")

    result = await digest.generate_digest(
        _SAMPLE_EVENTS, query_fn=_capturing_query, env={"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}
    )

    assert result == "summary"
    assert captured_env == {"CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-test"}


@pytest.mark.asyncio
async def test_with_no_env_argument_the_cli_subprocess_gets_an_empty_dict_never_none():
    """Mirrors sdk_adapter.py's own CRITICAL regression coverage - the real
    subprocess transport unconditionally dict-unpacks ClaudeAgentOptions.env
    (**self._options.env); passing env=None would raise a TypeError."""
    captured_env = {}

    async def _capturing_query(*, prompt, options):
        captured_env["value"] = options.env
        yield AssistantMessage(content=[TextBlock(text="summary")], model="claude")

    await digest.generate_digest(_SAMPLE_EVENTS, query_fn=_capturing_query)

    assert captured_env["value"] == {}


def test_generate_digest_reuses_risk_judges_get_api_client():
    """digest.py must not redefine get_api_client - it imports risk_judge's
    (KTD1's own reasoning: no duplicated ANTHROPIC_API_KEY-gating logic)."""
    assert digest.get_api_client is risk_judge.get_api_client
