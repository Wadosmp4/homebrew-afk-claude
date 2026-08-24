"""Tests for companion/risk_judge.py - the real claude_agent_sdk.query()
is never invoked here (that would spawn a real `claude` CLI subprocess);
`judge_is_safe`'s `query_fn` parameter is injected with a fake async
generator that yields the same message shapes the real SDK does. The
direct-API fast path is exercised the same way, via an injected fake
`api_client` - the real `anthropic` package's network client is never
constructed or called from this file.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from companion import risk_judge
from companion.risk_judge import judge_is_safe


@pytest.fixture(autouse=True)
def _reset_api_client_cache():
    """get_api_client() caches its resolved client at module scope - reset
    it before and after every test here so one test's state (real None, or
    anything a future test might set) never leaks into the next."""
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
    """Stands in for `anthropic.AsyncAnthropic()` - `client.messages.create`
    is the only surface `_judge_via_api` actually calls."""

    def __init__(self, *, response_text: Optional[str] = None, raises: Optional[Exception] = None, delay: float = 0.0):
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


@pytest.mark.asyncio
async def test_a_safe_verdict_returns_true():
    result = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=_fake_query("SAFE: small, ordinary edit")
    )
    assert result is True


@pytest.mark.asyncio
async def test_a_review_verdict_returns_false():
    result = await judge_is_safe(
        "Bash", {"command": "some-custom-script.sh"}, "/repo", query_fn=_fake_query("REVIEW: unfamiliar script")
    )
    assert result is False


@pytest.mark.asyncio
async def test_an_ambiguous_response_fails_closed():
    result = await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=_fake_query("Sure, go ahead!"))
    assert result is False


@pytest.mark.asyncio
async def test_an_exception_from_query_fails_closed():
    async def failing_query(*, prompt, options):
        raise RuntimeError("subprocess spawn failed")
        yield  # pragma: no cover - makes this a generator function

    result = await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=failing_query)
    assert result is False


@pytest.mark.asyncio
async def test_a_timeout_fails_closed():
    result = await judge_is_safe(
        "Edit",
        {"file_path": "a.py"},
        "/repo",
        timeout_seconds=0.05,
        query_fn=_fake_query("SAFE: fine", delay=1.0),
    )
    assert result is False


@pytest.mark.asyncio
async def test_no_tools_are_offered_to_the_judge_session():
    """The judge classifies, it never acts - options.tools must be an
    empty list, not the default full toolset."""
    captured = {}

    async def query_fn(*, prompt, options):
        captured["tools"] = options.tools
        yield AssistantMessage(content=[TextBlock(text="SAFE: fine")], model="claude")

    await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=query_fn)
    assert captured["tools"] == []


@pytest.mark.asyncio
async def test_the_real_project_cwd_is_never_passed_to_the_judge_session():
    """Regression test: passing the real project cwd to ClaudeAgentOptions
    made every judged call spawn a genuine `claude` subprocess that wrote
    its own transcript into *that project's* real ~/.claude/projects
    directory - indistinguishable from an actual session to
    companion/history.py's discovery, so a project's "past sessions" list
    filled up with the judge's own internal classification exchanges. The
    judge has tools=[] (it can never touch the filesystem) and already
    gets the working directory as plain text in the prompt, so cwd buys
    it nothing and must stay unset."""
    captured = {}

    async def query_fn(*, prompt, options):
        captured["cwd"] = options.cwd
        captured["prompt"] = prompt
        yield AssistantMessage(content=[TextBlock(text="SAFE: fine")], model="claude")

    await judge_is_safe("Bash", {"command": "ls"}, "/Users/x/some-real-project", query_fn=query_fn)

    assert captured["cwd"] is None
    # The working directory is still communicated to the model - just as
    # prompt text, not as a real subprocess cwd.
    assert "/Users/x/some-real-project" in captured["prompt"]


# --- Mobile UX follow-up #3b: per-session caching -----------------------


@pytest.mark.asyncio
async def test_an_identical_call_hits_the_cache_and_the_judge_is_never_consulted_again():
    cache: dict = {}
    call_count = {"n": 0}

    def counting_query(response_text: str):
        async def query_fn(*, prompt, options):
            call_count["n"] += 1
            yield AssistantMessage(content=[TextBlock(text=response_text)], model="claude")

        return query_fn

    first = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=counting_query("SAFE: fine"), cache=cache
    )
    second = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=counting_query("REVIEW: should never be seen"), cache=cache
    )

    assert first is True
    # The second call's own query_fn would have answered REVIEW - if it
    # were actually invoked, `second` would be False. It stays True only
    # because the cache short-circuited before query_fn ran at all.
    assert second is True
    assert call_count["n"] == 1


@pytest.mark.asyncio
async def test_a_different_tool_input_does_not_share_a_cache_entry():
    cache: dict = {}

    first = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=_fake_query("SAFE: fine"), cache=cache
    )
    second = await judge_is_safe(
        "Edit", {"file_path": "b.py"}, "/repo", query_fn=_fake_query("REVIEW: different file"), cache=cache
    )

    assert first is True
    assert second is False
    assert len(cache) == 2


@pytest.mark.asyncio
async def test_without_a_cache_argument_every_call_is_judged_fresh():
    call_count = {"n": 0}

    def counting_query(response_text: str):
        async def query_fn(*, prompt, options):
            call_count["n"] += 1
            yield AssistantMessage(content=[TextBlock(text=response_text)], model="claude")

        return query_fn

    await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=counting_query("SAFE: fine"))
    await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=counting_query("SAFE: fine"))

    assert call_count["n"] == 2


# --- Mobile UX follow-up #3b: direct-API fast path -----------------------


@pytest.mark.asyncio
async def test_a_safe_verdict_via_the_api_fast_path_returns_true_without_touching_the_cli_path():
    def cli_path_should_not_be_called(*, prompt, options):
        raise AssertionError("the CLI-subprocess path must not run when the API fast path succeeds")
        yield  # pragma: no cover - makes this a generator function

    api_client = _FakeApiClient(response_text="SAFE: small, ordinary edit")

    result = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=cli_path_should_not_be_called, api_client=api_client
    )

    assert result is True
    assert len(api_client.calls) == 1


@pytest.mark.asyncio
async def test_a_review_verdict_via_the_api_fast_path_returns_false():
    api_client = _FakeApiClient(response_text="REVIEW: unfamiliar script")

    result = await judge_is_safe("Bash", {"command": "some-script.sh"}, "/repo", api_client=api_client)

    assert result is False


@pytest.mark.asyncio
async def test_an_api_failure_falls_back_to_the_cli_subprocess_path_rather_than_failing_closed_immediately():
    """The API call erroring is not the same thing as the judge answering
    REVIEW - the CLI-subprocess path can still answer the question, so it
    must still run rather than the whole judgment short-circuiting to
    False right at the API layer."""
    api_client = _FakeApiClient(raises=RuntimeError("connection reset"))

    result = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=_fake_query("SAFE: fine"), api_client=api_client
    )

    assert result is True
    assert len(api_client.calls) == 1


@pytest.mark.asyncio
async def test_an_api_timeout_falls_back_to_the_cli_subprocess_path(monkeypatch):
    # DEFAULT_API_TIMEOUT_SECONDS (the API path's own internal timeout, not
    # the outer `timeout_seconds` argument) is lowered here so a fake delay
    # can genuinely exceed it without making this test slow.
    monkeypatch.setattr(risk_judge, "DEFAULT_API_TIMEOUT_SECONDS", 0.05)
    api_client = _FakeApiClient(response_text="SAFE: fine", delay=1.0)

    result = await judge_is_safe(
        "Edit",
        {"file_path": "a.py"},
        "/repo",
        query_fn=_fake_query("SAFE: fine (from the CLI path)"),
        api_client=api_client,
        timeout_seconds=5.0,
    )

    assert result is True


@pytest.mark.asyncio
async def test_an_ambiguous_api_response_fails_closed_without_falling_back_to_the_cli_path():
    # Unlike a mechanism failure (timeout, exception, non-text content),
    # an ambiguous-but-real text response is a legitimate answer from the
    # model that just isn't unambiguously SAFE - same "when in doubt,
    # REVIEW" fail-closed philosophy as the CLI-only path already applies
    # (see test_an_ambiguous_response_fails_closed above), not a signal to
    # retry via a different mechanism.
    def cli_path_should_not_be_called(*, prompt, options):
        raise AssertionError("an ambiguous (not a failure) API response must fail closed directly")
        yield  # pragma: no cover - makes this a generator function

    api_client = _FakeApiClient(response_text="Sure, sounds fine to me!")

    result = await judge_is_safe(
        "Edit", {"file_path": "a.py"}, "/repo", query_fn=cli_path_should_not_be_called, api_client=api_client
    )

    assert result is False


@pytest.mark.asyncio
async def test_without_an_api_client_argument_the_cli_subprocess_path_runs_as_before():
    result = await judge_is_safe("Edit", {"file_path": "a.py"}, "/repo", query_fn=_fake_query("SAFE: fine"))
    assert result is True


# --- Mobile UX follow-up #3b: get_api_client's own gating ----------------


def test_get_api_client_returns_none_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert risk_judge.get_api_client() is None


def test_get_api_client_returns_none_when_anthropic_is_not_installed(monkeypatch):
    # `anthropic` is a deliberately soft/optional dependency (see
    # CompanionConfig.risk_judge_use_api's own docstring) - not installed
    # in this project's own test environment, which is exactly the case
    # this asserts against: a key is set, but the package isn't there.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-key")
    assert risk_judge.get_api_client() is None


def test_get_api_client_returns_the_cached_client_without_re_checking_the_environment():
    """Only a *successful* resolution is cached (see get_api_client's own
    early `if _api_client is not None` check) - a "no key"/"not installed"
    miss is cheap enough to just re-check next time rather than needing its
    own negative-caching path. This directly exercises the cache-hit branch
    by seeding it, rather than depending on constructing a real client
    (not possible in this environment, since `anthropic` isn't installed)."""
    sentinel = object()
    risk_judge._api_client = sentinel

    # Even with no ANTHROPIC_API_KEY at all, the cached sentinel wins -
    # the environment is never consulted once _api_client is already set.
    result = risk_judge.get_api_client()

    assert result is sentinel
