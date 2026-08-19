"""LLM-assisted risk judgment for tool calls the rule-based policy
(auto_approve.py) can't confidently resolve on its own - a real content
question ("is this specific edit/command actually risky") that no static
pattern can answer. Opt-in and layered ON TOP of the rule-based policy,
never a replacement for it: auto_approve.py's denylist always wins
regardless of what this returns, and sdk_adapter.py only ever consults
this for a call the rule-based allowlist did not already approve.

Two independent latency mitigations on top of the base design (mobile UX
follow-up #3b - a judged call was, by design, real seconds of CLI
subprocess cold-start, not milliseconds):

- Per-session caching (`cache` param): an identical (tool_name, tool_input)
  pair judged once within a session is never re-judged - see `_cache_key`.
- An optional direct-Anthropic-API fast path (`api_client` param, see
  `get_api_client`), skipping the CLI subprocess entirely for a plain
  HTTPS call to a fast/cheap model. Off unless a caller explicitly passes
  a resolved client - see get_api_client's own docstring for why this is
  config-gated (CompanionConfig.risk_judge_use_api) rather than a bare
  os.environ read inside this module, and never a bound default here the
  way `query_fn` is (a test that never passes `api_client` must never
  accidentally make a real network call just because the process
  happened to inherit an ANTHROPIC_API_KEY from somewhere).

The CLI-subprocess path (claude_agent_sdk.query() - a one-shot, stateless
call, not a persistent ClaudeSDKClient, since each judgment is independent
by design and doesn't need conversation history) remains every existing
installation's unchanged default and fallback: it's the only path that
needs no separate API key, working equally for a subscription-only
`claude` CLI login as for an API-key one. The judge session has zero tool
access (tools=[]) either way: it classifies, it never acts.

This is the exact tradeoff auto_approve.py's own docstring names as the
reason the rule-based layer exists at all: real latency/cost and real
reliance on a model's judgment for a security-relevant decision. Fails
closed on any error, timeout, or unparseable response from either path -
"ask the human" is always the safe fallback, never "assume it's fine".
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
# The direct-API path's own timeout - much lower than the CLI-subprocess
# path's, since a plain HTTPS call to a fast model has none of the
# subprocess-startup latency that default exists to tolerate.
DEFAULT_API_TIMEOUT_SECONDS = 8.0
# A fast, cheap model for this narrow one-line classification - not
# whatever model the user's own session is using, which may be a slower/
# pricier one better spent on the actual agent turn than a background
# risk check.
DEFAULT_API_MODEL = "claude-haiku-4-5-20251001"

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

_EXPLAIN_SYSTEM_PROMPT = (
    "You are explaining one tool call a coding agent wants to make to the "
    "human who has to approve or deny it. You are given the tool name, its "
    "input, and the working directory. Respond with EXACTLY one short "
    "sentence naming the specific, concrete risk or blast radius of this "
    "particular call (what it touches, whether it's reversible) - not a "
    "generic risk-level label. Do not say SAFE or REVIEW or give a "
    "verdict; the human is already deciding that themselves. You cannot "
    "use any tools yourself; only answer."
)

QueryFn = Callable[..., Any]

_api_client: Optional[Any] = None


def _cache_key(tool_name: str, tool_input: Optional[dict[str, Any]], cwd: Optional[str]) -> tuple[str, str, str]:
    """A stable, hashable cache key - json.dumps with sort_keys so two
    structurally-identical inputs built in a different key order (not
    guaranteed against, even if unlikely for a given tool's own schema)
    still hit the same entry, unlike a plain repr(). Includes `cwd`: the
    judgment prompt embeds the working directory text (see
    `_judge_uncached`), so an identical (tool_name, tool_input) pair judged
    once in one directory must not short-circuit a later call for the same
    literal command after the session's persistent shell has `cd`'d
    somewhere else - the verdict genuinely can depend on where it runs."""
    return (tool_name, json.dumps(tool_input, sort_keys=True, default=str), cwd or "")


def get_api_client() -> Optional[Any]:
    """Lazily constructs and caches a single AsyncAnthropic client, reused
    across every fast-path judgment call rather than opening a fresh HTTP
    connection pool per call. Returns None if `anthropic` isn't installed
    (a deliberately soft/optional dependency - see CompanionConfig's own
    risk_judge_use_api docstring for why it isn't a listed companion
    dependency) or ANTHROPIC_API_KEY isn't set.

    Only ever called from sdk_adapter.py, and only when
    CompanionConfig.risk_judge_use_api is explicitly on - never called
    just because this process happens to have ANTHROPIC_API_KEY in its
    environment. That indirection is what keeps a bare `pytest` run (which
    inherits the invoking shell's environment like any subprocess) from
    ever making a real, billed network call: every test either injects its
    own fake `api_client` directly or leaves it unset, and this function
    is never invoked in between."""
    global _api_client
    if _api_client is not None:
        return _api_client
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning(
            "risk_judge_use_api is on but the `anthropic` package isn't installed - "
            "run `pip install anthropic` to use the fast path. Falling back to the CLI subprocess path."
        )
        return None
    _api_client = anthropic.AsyncAnthropic()
    return _api_client


async def _judge_via_api(
    prompt: str, *, api_client: Any, timeout_seconds: float
) -> tuple[Optional[bool], Optional[str]]:
    """The fast path: a direct Anthropic API call, skipping the CLI
    subprocess entirely. Returns (None, None) on any failure - a problem
    with the API call itself is not evidence the tool call is risky, so
    the caller falls back to the CLI-subprocess path rather than treating
    an API-layer failure as a REVIEW verdict. On a real text response,
    also returns the judge's own verbatim "SAFE: ..."/"REVIEW: ..." line
    alongside the bool, so a caller building an audit trail doesn't have
    to re-parse the response text itself."""
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await api_client.messages.create(
                model=DEFAULT_API_MODEL,
                max_tokens=100,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text.strip()
                return text.upper().startswith("SAFE"), text
        return None, None
    except Exception:
        logger.exception("direct API risk judgment failed - falling back to the CLI subprocess path")
        return None, None


async def judge_is_safe(
    tool_name: str,
    tool_input: Optional[dict[str, Any]],
    cwd: Optional[str],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    query_fn: Optional[QueryFn] = None,
    cache: Optional[dict[tuple[str, str, str], bool]] = None,
    api_client: Optional[Any] = None,
    reason_out: Optional[dict[str, Optional[str]]] = None,
) -> bool:
    """True only if the judge explicitly answers SAFE - any error, timeout,
    or response that isn't unambiguously SAFE returns False (fail closed).

    `cache` (optional, keyed by `_cache_key`) short-circuits a repeated
    identical (tool_name, tool_input, cwd) triple within whatever scope the
    caller's dict lives in - sdk_adapter.py passes one dict per session, so
    the same lint/test/build command run twice in one session (from the
    same working directory) is judged once. `api_client` (optional, see
    `get_api_client`) tries a direct Anthropic API call first when
    provided, falling back to the CLI-subprocess path below on any
    API-layer failure. `query_fn` is injectable for tests (see
    test_risk_judge.py) - the real default spawns a real `claude` CLI
    subprocess, which tests must not do. Resolved as `query_fn or query`
    here (not a bound default parameter) so a caller like sdk_adapter.py's
    can_use_tool, which never passes query_fn explicitly, still picks up a
    monkeypatched `risk_judge.query` in tests - a bound default would keep
    pointing at the original function object regardless of any later
    patch.

    `reason_out` (optional, a caller-supplied dict) receives the judge's
    own verbatim "SAFE: <reason>"/"REVIEW: <reason>" line under the
    "reason" key - but only for a *fresh* judgment. Left untouched on a
    cache hit (the original judgment's reason text isn't retained in the
    cache) and on any failure/timeout path (a fail-closed default isn't
    something the judge actually said, so there's no rationale to report)."""
    key = _cache_key(tool_name, tool_input, cwd)
    if cache is not None and key in cache:
        return cache[key]

    result, reason = await _judge_uncached(
        tool_name, tool_input, cwd, timeout_seconds=timeout_seconds, query_fn=query_fn, api_client=api_client
    )
    if reason_out is not None:
        reason_out["reason"] = reason

    if cache is not None:
        cache[key] = result
    return result


async def _judge_uncached(
    tool_name: str,
    tool_input: Optional[dict[str, Any]],
    cwd: Optional[str],
    *,
    timeout_seconds: float,
    query_fn: Optional[QueryFn],
    api_client: Optional[Any],
) -> tuple[bool, Optional[str]]:
    prompt = f"Tool: {tool_name}\nInput: {tool_input!r}\nWorking directory: {cwd or 'unknown'}"

    if api_client is not None:
        api_safe, api_reason = await _judge_via_api(
            prompt, api_client=api_client, timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS
        )
        if api_safe is not None:
            return api_safe, api_reason
        # Falls through to the CLI-subprocess path below rather than
        # returning False here - the API call itself failing is not the
        # same thing as the judge answering REVIEW, and the CLI path can
        # still answer the question perfectly well.

    query_fn = query_fn or query
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
                            text = block.text.strip()
                            return text.upper().startswith("SAFE"), text
    except Exception:
        logger.exception("risk judge failed for tool=%r - falling back to prompting the user", tool_name)
    return False, None


async def _explain_via_api(prompt: str, *, api_client: Any, timeout_seconds: float) -> Optional[str]:
    """The fast path for explain_risk, mirroring _judge_via_api's shape but
    returning the raw explanation text (or None on any failure) rather than
    a SAFE/REVIEW bool - there's no verdict to parse here, the response
    text itself IS the explanation."""
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await api_client.messages.create(
                model=DEFAULT_API_MODEL,
                max_tokens=100,
                system=_EXPLAIN_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text.strip()
                return text or None
        return None
    except Exception:
        logger.exception("direct API risk explanation failed - falling back to the CLI subprocess path")
        return None


async def explain_risk(
    tool_name: str,
    tool_input: Optional[dict[str, Any]],
    cwd: Optional[str],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    query_fn: Optional[QueryFn] = None,
    api_client: Optional[Any] = None,
) -> Optional[str]:
    """Generates a plain-English, specific explanation of a pending tool
    call's risk/blast-radius for a permission request that already reached
    the human - a sibling to judge_is_safe sharing its exact prompt-
    construction and fallback shape, but asking for an explanation rather
    than a SAFE/REVIEW verdict. Fails closed to None on any error, timeout,
    or empty response - "no explanation shown" is always safe, since this
    is purely additive to the existing Allow/Deny decision."""
    prompt = f"Tool: {tool_name}\nInput: {tool_input!r}\nWorking directory: {cwd or 'unknown'}"

    if api_client is not None:
        api_explanation = await _explain_via_api(prompt, api_client=api_client, timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS)
        if api_explanation is not None:
            return api_explanation
        # Falls through to the CLI-subprocess path below, same as
        # judge_is_safe - an API-layer failure isn't evidence there's no
        # explanation to give.

    query_fn = query_fn or query
    try:
        async with asyncio.timeout(timeout_seconds):
            async for message in query_fn(
                prompt=prompt,
                options=ClaudeAgentOptions(system_prompt=_EXPLAIN_SYSTEM_PROMPT, tools=[]),
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            return text or None
    except Exception:
        logger.exception("risk explanation failed for tool=%r", tool_name)
    return None
