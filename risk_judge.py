"""LLM-assisted risk judgment for tool calls the rule-based policy
(auto_approve.py) can't confidently resolve on its own - a real content
question ("is this specific edit/command actually risky") that no static
pattern can answer. Opt-in and layered ON TOP of the rule-based policy,
never a replacement for it: auto_approve.py's denylist always wins
regardless of what this returns, and sdk_adapter.py only ever consults
this for a call the rule-based allowlist did not already approve.

Uses claude_agent_sdk.query() - a one-shot, stateless call, not a
persistent ClaudeSDKClient - since each judgment is independent by design
and doesn't need conversation history. The judge session has zero tool
access (tools=[]): it classifies, it never acts.

This is the exact tradeoff auto_approve.py's own docstring names as the
reason the rule-based layer exists at all: real latency (a fresh `claude`
CLI subprocess per undecided call) and real reliance on a model's
judgment for a security-relevant decision. Fails closed on any error,
timeout, or unparseable response - "ask the human" is always the safe
fallback, never "assume it's fine".
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0

_SYSTEM_PROMPT = (
    "You are a strict risk classifier for one tool call a coding agent wants "
    "to make. You are given the tool name and its input. Respond with EXACTLY "
    "one line: either 'SAFE: <brief reason>' if this specific call is "
    "low-risk and easily reversible (e.g. a small, ordinary code edit; a "
    "benign informational command), or 'REVIEW: <brief reason>' if it could "
    "be risky, destructive, hard to reverse, or you are at all unsure. "
    "When in doubt, always answer REVIEW - a human is always available as a "
    "fallback, so a wrong REVIEW just costs a mild prompt, but a wrong SAFE "
    "could cause real harm. You cannot use any tools yourself; only answer."
)

QueryFn = Callable[..., Any]


async def judge_is_safe(
    tool_name: str,
    tool_input: Optional[dict[str, Any]],
    cwd: Optional[str],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    query_fn: Optional[QueryFn] = None,
) -> bool:
    """True only if the judge explicitly answers SAFE - any error, timeout,
    or response that isn't unambiguously SAFE returns False (fail closed).
    `query_fn` is injectable for tests (see test_risk_judge.py) - the real
    default spawns a real `claude` CLI subprocess, which tests must not do.
    Resolved as `query_fn or query` here (not a bound default parameter)
    so a caller like sdk_adapter.py's can_use_tool, which never passes
    query_fn explicitly, still picks up a monkeypatched `risk_judge.query`
    in tests - a bound default would keep pointing at the original
    function object regardless of any later patch."""
    query_fn = query_fn or query
    prompt = f"Tool: {tool_name}\nInput: {tool_input!r}\nWorking directory: {cwd or 'unknown'}"
    try:
        async with asyncio.timeout(timeout_seconds):
            # No `cwd` here, deliberately - the judge has tools=[] (it can
            # never touch the filesystem) and already has the working
            # directory as plain text in the prompt above, so a real cwd
            # buys it nothing. Passing the real project cwd used to spawn
            # a genuine `claude` subprocess for every judged call, which
            # wrote its own transcript into *that project's* real
            # ~/.claude/projects directory - completely indistinguishable
            # from an actual session to history.py's discovery, so a
            # project's "past sessions" list filled up with the judge's
            # own internal "Tool: Bash / SAFE: ..." classification
            # exchanges. Omitting cwd keeps the judge's own housekeeping
            # sessions out of any real project's history entirely.
            async for message in query_fn(
                prompt=prompt,
                options=ClaudeAgentOptions(system_prompt=_SYSTEM_PROMPT, tools=[]),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            return block.text.strip().upper().startswith("SAFE")
    except Exception:
        logger.exception("risk judge failed for tool=%r - falling back to prompting the user", tool_name)
    return False
