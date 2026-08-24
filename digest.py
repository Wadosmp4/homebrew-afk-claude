"""AFK catch-up digest: a one-shot LLM call that summarizes a session's
recent transcript events into a few sentences for someone who was away.
Same companion-side, ANTHROPIC_API_KEY-gated, fail-closed shape as
risk_judge.py's judge_is_safe - this module reuses risk_judge.get_api_client
directly rather than redefining it, since both share the same "spawn a
one-shot claude_agent_sdk.query() with tools=[]" plumbing. Kept as a
separate module rather than folded into risk_judge.py: summarizing a
transcript and judging a tool call's safety are different questions that
only happen to share their LLM-call mechanics, not their logic.

Failure mode is always `None` (never raise), consumed by
daemon.py's _handle_session_digest as "no digest available" - the same
"ask the human instead" philosophy risk_judge.py's own docstring names,
adapted here to "just show the raw feed instead" since there's no human
question this module could otherwise force.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Optional

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock, query

from .risk_judge import DEFAULT_API_TIMEOUT_SECONDS, get_api_client

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 15.0
# A fast, cheap model for this narrow summarization task - same reasoning
# and same model choice as risk_judge.py's own DEFAULT_API_MODEL.
DEFAULT_API_MODEL = "claude-haiku-4-5-20251001"
# Bounds prompt size for a very long session - only the most recent events
# are summarized, matching what someone catching up actually cares about.
MAX_EVENTS_IN_PROMPT = 100

_SYSTEM_PROMPT = (
    "You are summarizing a coding agent's session for someone who just returned "
    "after being away. You are given a list of recent events (assistant messages, "
    "tool calls, tool results) in order. Respond with a short summary - 2 to 4 "
    "sentences - covering what changed, what the agent is currently doing or "
    "waiting on, and any errors hit. Be concrete (name files/commands when "
    "useful), not generic. You cannot use any tools yourself; only answer."
)

QueryFn = Callable[..., Any]


def _format_events(transcript_events: list[dict[str, Any]]) -> str:
    tail = transcript_events[-MAX_EVENTS_IN_PROMPT:]
    lines = [f"{event.get('type', 'unknown')}: {event.get('data', {})!r}" for event in tail]
    return "\n".join(lines)


async def _via_api(prompt: str, *, api_client: Any, timeout_seconds: float) -> Optional[str]:
    """Mirrors risk_judge.py's own _judge_via_api: returns None on any
    failure so the caller falls back to the CLI-subprocess path rather
    than treating an API-layer failure as "no summary exists"."""
    try:
        async with asyncio.timeout(timeout_seconds):
            response = await api_client.messages.create(
                model=DEFAULT_API_MODEL,
                max_tokens=300,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                text = block.text.strip()
                return text or None
        return None
    except Exception:
        logger.exception("direct API digest generation failed - falling back to the CLI subprocess path")
        return None


async def generate_digest(
    transcript_events: list[dict[str, Any]],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    query_fn: Optional[QueryFn] = None,
    api_client: Optional[Any] = None,
) -> Optional[str]:
    """Returns a short natural-language summary of transcript_events, or
    None on any failure/timeout/empty response (fail closed - the caller
    shows the raw feed with no digest, never an error). `query_fn` is
    injectable for tests, resolved as `query_fn or query` (not a bound
    default) for the same reason risk_judge.judge_is_safe does - a caller
    that never passes query_fn still picks up a monkeypatched
    `digest.query` in tests."""
    prompt = f"Recent session events, oldest first:\n{_format_events(transcript_events)}"

    if api_client is not None:
        api_result = await _via_api(prompt, api_client=api_client, timeout_seconds=DEFAULT_API_TIMEOUT_SECONDS)
        if api_result is not None:
            return api_result
        # Falls through to the CLI-subprocess path below, same reasoning as
        # risk_judge.py's own fallback - an API-layer failure isn't
        # evidence there's nothing to summarize.

    query_fn = query_fn or query
    # fix(review): discovered via real-device testing - a query that
    # completes normally but never yields an AssistantMessage/TextBlock
    # (as opposed to timing out or raising) fell through to `return None`
    # completely silently, indistinguishable from every other
    # fail-closed reason. Tracking what actually came back makes that
    # case diagnosable without reverting to throwaway debug prints.
    message_types_seen: list[str] = []
    try:
        async with asyncio.timeout(timeout_seconds):
            async for message in query_fn(
                prompt=prompt,
                options=ClaudeAgentOptions(system_prompt=_SYSTEM_PROMPT, tools=[]),
            ):
                message_types_seen.append(type(message).__name__)
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            text = block.text.strip()
                            return text or None
    except Exception:
        logger.exception("digest generation failed")
        return None
    logger.warning(
        "digest generation produced no usable text block - message types seen: %r", message_types_seen
    )
    return None
