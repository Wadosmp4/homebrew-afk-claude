"""Tests for companion/risk_judge.py - the real claude_agent_sdk.query()
is never invoked here (that would spawn a real `claude` CLI subprocess);
`judge_is_safe`'s `query_fn` parameter is injected with a fake async
generator that yields the same message shapes the real SDK does.
"""
from __future__ import annotations

import asyncio

import pytest
from claude_agent_sdk import AssistantMessage, TextBlock

from companion.risk_judge import judge_is_safe


def _fake_query(response_text: str, *, delay: float = 0.0):
    async def query_fn(*, prompt, options):
        if delay:
            await asyncio.sleep(delay)
        yield AssistantMessage(content=[TextBlock(text=response_text)], model="claude")

    return query_fn


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
