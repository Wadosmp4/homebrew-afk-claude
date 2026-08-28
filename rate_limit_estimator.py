"""Bug fix (user report): the CLI only pushes a fresh RateLimitEvent on its
own status-transition schedule - tied to whenever it happens to notice one,
not on every turn, and with no guaranteed relationship to background-
subagent token consumption (a dispatched "Task" can burn real tokens
against the same account-wide quota with no signal reaching the phone until
the main session's own next request happens to trigger a fresh check).

Between two real readings, RateLimitEstimator extrapolates a live-updating
guess from tokens this companion has actually observed being consumed -
main-session turns (ResultMessage.model_usage) and background tasks
(TaskProgressMessage/TaskNotificationMessage.usage). It is explicitly NOT
exact, for two reasons that don't go away no matter how this is tuned:

1. It's a linear extrapolation calibrated from this account's own two most
   recent real readings, not Anthropic's real (undocumented - checked
   docs.claude.com's own rate-limits page, which 404s) rate-limit formula.
2. Rate limits are account-wide, but this only sees tokens consumed by
   sessions *this companion process* drives. Any other concurrent usage on
   the same account (another machine, the web app, direct API calls) is
   invisible here and will make the estimate undercount.

A real RateLimitEvent always wins the moment one arrives - see
SDKAdapter._handle_rate_limit, which calls record_real_reading() and then
emits the real "rate_limit" event unchanged. This class only fills the gap
between real readings, and the resulting "rate_limit_estimate" event is
rendered distinctly on the phone (see mobile's own isEstimate flag) so it's
never mistaken for a confirmed figure.

Scoped one instance per SDKAdapter (a daemon-wide singleton for all SDK-
owned sessions - see SDKAdapter.__init__), not per-session: an account's
rate limit is shared across every session this companion happens to be
driving concurrently, so the token counter and per-window calibration must
be too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class _WindowEstimate:
    last_real_utilization: float
    # This estimator's own _cumulative_tokens value at the moment
    # last_real_utilization was reported - the baseline every subsequent
    # estimate() call for this window measures forward from.
    last_real_tokens: int
    # None until a second, distinct real reading lets us compute a slope -
    # a single reading has nothing to calibrate against yet.
    tokens_per_utilization: Optional[float] = None


class RateLimitEstimator:
    def __init__(self) -> None:
        self._cumulative_tokens = 0
        self._windows: dict[str, _WindowEstimate] = {}
        # task_id -> last-seen cumulative total_tokens for that task, so a
        # later progress/notification reading for the same task_id
        # contributes only its own delta - TaskUsage.total_tokens is each
        # task's own running total, not an increment since the last one.
        self._task_totals: dict[str, int] = {}

    def record_turn_usage(self, model_usage: Optional[dict[str, Any]]) -> None:
        """ResultMessage.model_usage: keyed by model name, each a
        claude_agent_sdk.ModelUsage dict - per-turn (this ResultMessage's
        own completed turn), not cumulative for the session, so every call
        adds directly to the running counter. None on a CLI/SDK version
        that doesn't report it (or a test double) is a no-op, not an
        error - an estimate that can't advance for a given turn is exactly
        as honest as this class gets when the input data isn't there."""
        if not model_usage:
            return
        for entry in model_usage.values():
            self._cumulative_tokens += (
                (entry.get("inputTokens") or 0)
                + (entry.get("outputTokens") or 0)
                + (entry.get("cacheReadInputTokens") or 0)
                + (entry.get("cacheCreationInputTokens") or 0)
            )

    def record_task_usage(self, task_id: str, usage: Optional[dict[str, Any]]) -> None:
        """TaskProgressMessage/TaskNotificationMessage.usage
        (claude_agent_sdk.TaskUsage) - total_tokens is that task's own
        running cumulative total, so only the delta since the last known
        reading for this exact task_id is added, whether this call came
        from a progress tick or the final notification."""
        if not usage:
            return
        total = usage.get("total_tokens")
        if total is None:
            return
        previous = self._task_totals.get(task_id, 0)
        if total > previous:
            self._cumulative_tokens += total - previous
        self._task_totals[task_id] = total

    def record_real_reading(self, rate_limit_type: str, utilization: Optional[float]) -> None:
        """A genuine RateLimitEvent for this window. Always replaces the
        stored baseline reading; only recalibrates the slope
        (tokens_per_utilization) when this reading's utilization and the
        estimator's own token counter both moved forward from the previous
        reading - a repeat of the same utilization (nothing changed) or a
        window reset (utilization drops even though tokens rose) would
        poison the slope with a zero/negative denominator instead of
        genuinely recalibrating it. On a reset, the stale pre-reset slope
        is kept rather than discarded - it's a reasonable estimate of this
        account's own token-cost-per-percent to keep extrapolating from
        until the next post-reset pair of readings recalibrates it
        properly; explicit reset detection isn't needed for this to
        self-heal."""
        if utilization is None:
            return
        previous = self._windows.get(rate_limit_type)
        tokens_per_utilization = previous.tokens_per_utilization if previous else None
        if previous is not None:
            token_delta = self._cumulative_tokens - previous.last_real_tokens
            utilization_delta = utilization - previous.last_real_utilization
            if token_delta > 0 and utilization_delta > 0:
                tokens_per_utilization = token_delta / utilization_delta
        self._windows[rate_limit_type] = _WindowEstimate(
            last_real_utilization=utilization,
            last_real_tokens=self._cumulative_tokens,
            tokens_per_utilization=tokens_per_utilization,
        )

    def estimate(self, rate_limit_type: str) -> Optional[float]:
        """Best current guess at this window's utilization (0.0-1.0), or
        None when no real reading has ever arrived for it, or one has but
        hasn't calibrated a slope yet (needs a second distinct reading
        first) - both cases mean there is nothing honest to extrapolate,
        not a zero. Clamped to 1.0: this window's own utilization can't
        exceed "fully consumed" (a separate "overage" rate_limit_type
        tracks pay-as-you-go usage beyond it, not this window going over
        100%), and an unbounded compounding extrapolation over a long gap
        between real readings could otherwise drift well past it."""
        window = self._windows.get(rate_limit_type)
        if window is None or window.tokens_per_utilization is None:
            return None
        token_delta = self._cumulative_tokens - window.last_real_tokens
        if token_delta <= 0:
            return window.last_real_utilization
        return min(1.0, window.last_real_utilization + token_delta / window.tokens_per_utilization)

    def calibrated_types(self) -> list[str]:
        """Every rate_limit_type with enough real history to extrapolate
        from right now - the set SDKAdapter should emit a fresh
        rate_limit_estimate for after recording new usage."""
        return [rate_limit_type for rate_limit_type, window in self._windows.items() if window.tokens_per_utilization is not None]
