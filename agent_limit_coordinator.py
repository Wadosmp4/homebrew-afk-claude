"""Agent Limit Handoff plan (U2/U3): derives, escalates, and resolves one
`agent_limit_offer` per `(agent, rate_limit_type)` from real `"rate_limit"`
events across both agent adapters, and orchestrates the actual handoff once
an offer is accepted - composing the origin session's context (digest +
diff), starting a fresh session with the other agent, seeding it, and
reporting the outcome back on the origin session's own event stream.

Code-review fix (whole-codebase SOLID/DRY audit): extracted from
`CompanionDaemon`, which had grown into a ~2,200-line god object accreting
one feature plan's worth of methods after another. This was already the
single most self-contained concern in that file - its three state
dicts/sets are read/written only by the methods below, plus one call site
each from `CompanionDaemon._forward_events` (reacting to a `"rate_limit"`
event) and `CompanionDaemon._dispatch_session_scoped_action` (the
`resolve_agent_limit_offer` action) - so this class owns exactly that
block, unchanged in behavior. `CompanionDaemon` holds one instance
(`self.agent_limits`) and delegates.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from . import digest, git_status, history
from .adapters.base import AdapterProtocol
from .adapters.observe_adapter import ObserveAdapter
from .config import CompanionConfig

logger = logging.getLogger(__name__)

# Agent Limit Handoff plan, U3 (KTD6): a short, generic line substituted
# for an actual digest whenever generate_digest is skipped (Codex-
# originated handoff - no persisted on-disk transcript to summarize) or
# returns None (Claude-originated, fail-closed) - see
# AgentLimitCoordinator._compose_handoff_seed_message. Never blocks the
# handoff; only degrades its seed message.
_HANDOFF_NO_DIGEST_LINE = (
    "No summary is available for this handoff - see the diff below for what's changed so far."
)

# Agent Limit Handoff plan, U3 (KTD6): caps the *characters* of
# concatenated diff content folded into a handoff's seed message. There's
# no existing diff-sized bounding precedent to reuse directly - digest.py's
# MAX_EVENTS_IN_PROMPT bounds a transcript by event count, not character
# count, and git_status.py has no whole-tree diff concept at all - so this
# is a fresh, explicitly-chosen constant: generous enough for a normal
# in-progress changeset, small enough that a huge one degrades to a
# file-list summary (AgentLimitCoordinator._compose_handoff_diff) instead
# of ballooning the new session's very first turn.
_MAX_HANDOFF_DIFF_CHARS = 20_000

# Agent Limit Handoff plan, U3: human-readable labels for the handoff seed
# message and its user-facing failure reasons - the wire values
# ("claude_code"/"codex") are internal identifiers, not something to show
# the person reading the new session's first message or a rejection reason.
_AGENT_DISPLAY_NAMES = {"claude_code": "Claude Code", "codex": "Codex"}


class AgentLimitCoordinator:
    def __init__(
        self,
        *,
        sdk_adapter: AdapterProtocol,
        codex_adapter: AdapterProtocol,
        observe_adapter: ObserveAdapter,
        config: CompanionConfig,
        get_active_ws: Callable[[], Optional[Any]],
        start_session_for_agent: Callable[..., Awaitable[str]],
        effective_cli_env: Callable[[], dict[str, str]],
    ) -> None:
        self.sdk_adapter = sdk_adapter
        self.codex_adapter = codex_adapter
        self.observe_adapter = observe_adapter
        self.config = config
        # R5/KTD9: reuses whichever connection is currently serving actions
        # at all, read fresh at the moment a handoff actually needs it
        # (CompanionDaemon's own `_active_ws` docstring explains why this
        # is instance state rather than a threaded-through parameter) -
        # this callable is how that daemon-owned, mutates-over-time value
        # reaches here without this class holding a stale copy.
        self._get_active_ws = get_active_ws
        self._start_session_for_agent = start_session_for_agent
        self._effective_cli_env = effective_cli_env
        # (agent, rate_limit_type) -> the rate_limit `status`
        # ("allowed_warning"/"rejected") that raised/last-escalated the
        # currently-active, still-unresolved offer for that key; absent
        # entirely once there is no active offer (cleared or never raised).
        self._agent_limit_offers: dict[tuple[str, str], str] = {}
        # (agent, rate_limit_type) -> "dismissed"/"accepted" once a phone
        # has resolved that key via resolve_agent_limit_offer - suppresses
        # re-raising until KTD5's own reset conditions clear this entry
        # (see _clear_agent_limit_offer). Deliberately a separate dict from
        # the one above rather than folding "resolved" into the same value
        # space: a key can be resolved while genuinely having no entry in
        # _agent_limit_offers at all (the offer was popped the moment it
        # was resolved), so they track two independent facts.
        self._agent_limit_resolutions: dict[tuple[str, str], str] = {}
        # Keys with an accept currently mid-flight (added right before
        # _start_agent_handoff is awaited, removed in its finally once that
        # await completes, success or failure) - resolve_agent_limit_offer's
        # own accept branch uses this to reject a duplicate accept for the
        # same key (a Retry tap, or any other re-dispatch) while one is
        # already running, so a slow handoff can never be raced into
        # starting a second destination session. A retry AFTER the first
        # handoff has already finished is unaffected - the key is gone from
        # this set by then.
        self._agent_limit_handoffs_in_progress: set[tuple[str, str]] = set()

    def agent_for_adapter(self, adapter) -> Optional[str]:
        """Agent Limit Handoff plan, U2 (KTD3): the `agent` half of an
        offer's (agent, rate_limit_type) key. Only sdk_adapter/codex_adapter
        ever drive a session through a real agent process - observe_adapter
        watches a transcript it doesn't itself own and never emits
        "rate_limit" (events.py's own comment on that event type), so it
        deliberately has no mapping here; a caller that somehow reaches this
        for an unrecognized adapter gets None back rather than a guess."""
        if adapter is self.sdk_adapter:
            return "claude_code"
        if adapter is self.codex_adapter:
            return "codex"
        return None

    def _adapter_for_agent(self, agent: str) -> AdapterProtocol:
        return self.sdk_adapter if agent == "claude_code" else self.codex_adapter

    def _broadcast_agent_limit_event(self, agent: str, type_: str, **data: Any) -> None:
        """Agent Limit Handoff plan, U2 (KTD3): "Raising, escalating, or
        resolving an offer for a key emits the corresponding event to
        EVERY session id currently tracked under that key, not only the
        session whose reading triggered the change" - since the offer is
        one account-wide condition, every currently-live session driven by
        this `agent` counts as "tracked under that key", not just sessions
        that happen to have seen this exact rate_limit_type before.
        `discover_sessions()` is read fresh here rather than cached, so a
        session that ended between the triggering reading and this call is
        never handed a doomed emit_custom (KeyError on an already-popped
        session)."""
        adapter = self._adapter_for_agent(agent)
        for session_id in adapter.discover_sessions():
            adapter.emit_custom(session_id, type_, agent=agent, **data)

    def _emit_agent_limit_offer(
        self, agent: str, rate_limit_type: str, status: str, utilization: Optional[float], resets_at: Optional[int]
    ) -> None:
        offered_agent = "codex" if agent == "claude_code" else "claude_code"
        self._broadcast_agent_limit_event(
            agent,
            "agent_limit_offer",
            offered_agent=offered_agent,
            rate_limit_type=rate_limit_type,
            status=status,
            utilization=utilization,
            resets_at=resets_at,
        )

    def _emit_agent_limit_offer_resolved(self, agent: str, rate_limit_type: str, resolution: str) -> None:
        self._broadcast_agent_limit_event(
            agent, "agent_limit_offer_resolved", rate_limit_type=rate_limit_type, resolution=resolution
        )

    @staticmethod
    def _agent_limit_window_has_reset(resets_at: Any) -> bool:
        """KTD5: "a later rate_limit reading for that key carries a
        resets_at in the past" - resets_at is the CLI/Codex's own raw
        epoch-seconds field (see sdk_adapter.py's RateLimitEvent handling
        and codex_adapter.py's RateLimitWindow.resetsAt), never an ISO
        string, so this is a plain numeric comparison. A missing/malformed
        value (None, or any non-numeric payload) is never treated as
        "passed" - there's nothing honest to compare against."""
        if not isinstance(resets_at, (int, float)) or isinstance(resets_at, bool):
            return False
        return resets_at <= datetime.now(timezone.utc).timestamp()

    def _clear_agent_limit_offer(self, agent: str, rate_limit_type: str) -> None:
        """KTD5: clears both the active-offer state and any remembered
        dismissal/acceptance for this key - called when a real rate_limit
        reading shows either the window has reset or a fresh "allowed"
        status, "regardless of whether the offer was ever resolved". Design
        judgment call (state-diagram's own bracketed note): when an offer
        was actively showing and still unresolved, an explicit
        agent_limit_offer_resolved with resolution="cleared" is broadcast
        too, so mobile's replay-safe scan (the same findPendingPermission-
        style pattern permission_resolved already supports) has a positive
        signal to retire a previously-shown card that neither dismiss nor
        accept ever touched. When the key was already resolved (no active
        offer left to clear), only the remembered resolution itself is
        dropped - silently, per the state diagram's own "no new event
        needed" note for that branch."""
        key = (agent, rate_limit_type)
        had_active_offer = self._agent_limit_offers.pop(key, None) is not None
        self._agent_limit_resolutions.pop(key, None)
        if had_active_offer:
            self._emit_agent_limit_offer_resolved(agent, rate_limit_type, "cleared")

    def handle_rate_limit_reading(self, agent: str, data: dict[str, Any]) -> None:
        """Agent Limit Handoff plan, U2 (KTD3/KTD5): derives, escalates, or
        clears one agent_limit_offer for (agent, data["rate_limit_type"])
        from a single real "rate_limit" event's data. Callers must already
        have filtered to event.type == "rate_limit" - this has no way to
        tell a real reading from an estimate on its own once it only has
        the bare data dict, so that check has to happen upstream (see
        CompanionDaemon._forward_events)."""
        rate_limit_type = data.get("rate_limit_type")
        status = data.get("status")
        if not rate_limit_type or not status:
            return  # nothing to key an offer on - see sdk_adapter.py's own note that rate_limit_type/status can be nullable
        resets_at = data.get("resets_at")
        if status == "allowed" or self._agent_limit_window_has_reset(resets_at):
            self._clear_agent_limit_offer(agent, rate_limit_type)
            if status == "allowed":
                return
            # Judgment call: resets_at already passed, but this very
            # reading's own status is still allowed_warning/rejected (a
            # brand-new window already past its own threshold at read
            # time) - fall through and evaluate it fresh below rather than
            # silently swallowing a genuinely elevated current reading just
            # because the same event also cleared stale pre-reset state.
        if status not in ("allowed_warning", "rejected"):
            return  # an unrecognized/future status value - fail closed, no offer
        key = (agent, rate_limit_type)
        if key in self._agent_limit_resolutions:
            return  # already dismissed/accepted for this window - suppressed until it resets (KTD5)
        current = self._agent_limit_offers.get(key)
        if current == status or current == "rejected":
            # Identical reading (nothing changed), or already at max
            # severity - "rejected" never downgrades in place back to
            # "allowed_warning" from a harder reading (the diagram only
            # ever escalates upward; a softer subsequent reading before the
            # window resets is treated as noise, not a real de-escalation).
            return
        self._agent_limit_offers[key] = status
        self._emit_agent_limit_offer(agent, rate_limit_type, status, data.get("utilization"), resets_at)

    async def resolve_agent_limit_offer(
        self, adapter, session_id: str, rate_limit_type: Optional[str], decision: Optional[str]
    ) -> None:
        """Agent Limit Handoff plan, U2 (KTD4): the `resolve_agent_limit_offer`
        action's handler - mirrors respond_to_permission's own
        decision-parameter shape. A "dismiss" only emits
        agent_limit_offer_resolved; "accept" emits it too, then hands off
        to _start_agent_handoff below."""
        agent = self.agent_for_adapter(adapter)
        if agent is None or not rate_limit_type or decision not in ("accept", "dismiss"):
            logger.warning(
                "rejected resolve_agent_limit_offer: session_id=%r agent=%r rate_limit_type=%r decision=%r",
                session_id,
                agent,
                rate_limit_type,
                decision,
            )
            return
        key = (agent, rate_limit_type)
        if decision == "accept" and key in self._agent_limit_handoffs_in_progress:
            # Code-review fix (P1): a Retry tap (AgentLimitOfferCard's own
            # handleRetry has no double-tap guard) or a second accept
            # arriving for a key whose handoff hasn't finished yet must not
            # start a second, fully independent _start_agent_handoff - that
            # would spawn two live destination-agent sessions for one
            # logical accept, with the second result silently dropped by
            # mobile's single-slot correlation ref. A genuine retry AFTER
            # the first handoff has already finished (removed from this set
            # in the finally below) is unaffected and proceeds normally.
            logger.info(
                "ignored duplicate resolve_agent_limit_offer(accept) for %r - a handoff is already in flight",
                key,
            )
            return
        resolution = "accepted" if decision == "accept" else "dismissed"
        self._agent_limit_offers.pop(key, None)
        self._agent_limit_resolutions[key] = resolution
        self._emit_agent_limit_offer_resolved(agent, rate_limit_type, resolution)
        if decision == "accept":
            self._agent_limit_handoffs_in_progress.add(key)
            try:
                await self._start_agent_handoff(session_id, agent, rate_limit_type)
            finally:
                self._agent_limit_handoffs_in_progress.discard(key)

    async def _compose_handoff_diff(self, cwd: str) -> str:
        """Agent Limit Handoff plan, U3 (KTD6): git_status.py has no
        whole-tree diff helper - get_diff is single-file only (confirmed in
        this plan's own research) - so this aggregates get_status(cwd)'s
        changed-path lists through one get_diff(cwd, path) call per path,
        concatenating the results up to _MAX_HANDOFF_DIFF_CHARS. Beyond the
        cap, falls back to a plain file-list summary instead of a
        half-truncated diff blob that would leave the handed-off session
        guessing at a cut-off hunk."""
        status = await asyncio.to_thread(git_status.get_status, cwd)
        changed_paths = [*status.modified, *status.added, *status.deleted]
        if not changed_paths:
            return "No local changes (working tree is clean)."
        chunks: list[str] = []
        total_chars = 0
        for path in changed_paths:
            diff = await asyncio.to_thread(git_status.get_diff, cwd, path)
            if diff.is_binary:
                chunk = f"diff --git a/{path} b/{path}\nBinary file changed.\n"
            else:
                chunk = diff.diff or f"diff --git a/{path} b/{path}\n(no textual diff available)\n"
            total_chars += len(chunk)
            if total_chars > _MAX_HANDOFF_DIFF_CHARS:
                file_list = "\n".join(f"- {p}" for p in changed_paths)
                return (
                    f"{len(changed_paths)} changed file(s) - diff omitted, total size exceeds "
                    f"the {_MAX_HANDOFF_DIFF_CHARS}-character handoff cap:\n{file_list}"
                )
            chunks.append(chunk)
        return "\n".join(chunks)

    async def _compose_handoff_seed_message(self, session_id: str, agent: str, cwd: str) -> str:
        """Agent Limit Handoff plan, U3 (KTD6): the handed-off session's
        first message - a short intro, a digest section, and a diff
        section. The digest is only ever attempted for a Claude-originated
        handoff (`agent == "claude_code"`), mirroring
        CompanionDaemon._handle_session_digest's own argument-gathering
        (read_session_history against observe_adapter.projects_dir, then
        generate_digest with the same risk_judge_use_api-gated api_client
        and effective_cli_env()) - a Codex-originated handoff skips
        straight to diff-only, permanently, not as a failure case, since
        Codex sessions have no persisted on-disk transcript generate_digest
        could read. Either a missing transcript, a fail-closed
        generate_digest (None), or a skipped Codex origin all degrade to
        _HANDOFF_NO_DIGEST_LINE rather than blocking the handoff - never a
        reason to fail it."""
        digest_text: Optional[str] = None
        if agent == "claude_code":
            events = await asyncio.to_thread(
                history.read_session_history, session_id, str(self.observe_adapter.projects_dir)
            )
            if events:
                api_client = digest.get_api_client() if self.config.risk_judge_use_api else None
                digest_text = await digest.generate_digest(
                    events, api_client=api_client, env=self._effective_cli_env()
                )
        diff_text = await self._compose_handoff_diff(cwd)
        origin_label = _AGENT_DISPLAY_NAMES.get(agent, agent)
        return (
            f"You're picking up a session handed off from {origin_label} after it hit its usage limit.\n\n"
            f"## Summary\n{digest_text or _HANDOFF_NO_DIGEST_LINE}\n\n"
            f"## Uncommitted changes\n{diff_text}"
        )

    def _emit_handoff_result(
        self,
        adapter,
        session_id: str,
        *,
        ok: bool,
        reason: Optional[str] = None,
        new_session_id: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Agent Limit Handoff plan, U3: the one place "agent_handoff_result"
        is emitted, so every path through _start_agent_handoff below reports
        the same shape on the *origin* session's own stream (R6) - never
        leaves an accepted offer unanswered. Wrapped in its own try/except:
        emit_custom raises for an origin session_id that's no longer known
        (e.g. it ended in the window between the accept and this call) -
        there's no one left to answer at that point, and the caller
        (resolve_agent_limit_offer, itself inside
        CompanionDaemon._dispatch_session_scoped_action's own broad
        exception handler) would otherwise just log-and-drop the very same
        failure anyway, so swallowing it here (with its own explicit log
        line) is no less visible."""
        try:
            if ok:
                adapter.emit_custom(
                    session_id, "agent_handoff_result", ok=True, new_session_id=new_session_id, agent=agent
                )
            else:
                adapter.emit_custom(session_id, "agent_handoff_result", ok=False, reason=reason)
        except Exception:
            logger.exception("failed to emit agent_handoff_result for origin session %r", session_id)

    async def _start_agent_handoff(self, session_id: str, agent: str, rate_limit_type: str) -> None:
        """Agent Limit Handoff plan, U3: orchestrates R5's actual handoff
        once an offer is accepted (called from resolve_agent_limit_offer's
        accept branch above). Composes the origin session's context (a
        digest, Claude-originated only, plus a whole-session git diff - see
        _compose_handoff_seed_message), starts a fresh session with the
        *other* agent on the same cwd (KTD9's shared
        start_session_for_agent callback), seeds it with that content as
        its first message, and reports the outcome back on the *origin*
        session's own event stream (R6) so the phone never sees an
        unexplained hang - `ok: False` with a specific, user-facing
        `reason` on any failure, never a raw exception message (see
        events.py's own "agent_handoff_result" registration comment)."""
        source_adapter = self._adapter_for_agent(agent)
        destination_agent = "codex" if agent == "claude_code" else "claude_code"
        destination_label = _AGENT_DISPLAY_NAMES.get(destination_agent, destination_agent)

        cwd = source_adapter.get_cwd(session_id)
        if cwd is None:
            self._emit_handoff_result(
                source_adapter,
                session_id,
                ok=False,
                reason="Could not determine this session's working directory.",
            )
            return

        try:
            seed_message = await self._compose_handoff_seed_message(session_id, agent, cwd)
        except Exception:
            logger.exception("failed to compose handoff seed content for session %r", session_id)
            self._emit_handoff_result(
                source_adapter, session_id, ok=False, reason="Could not prepare the handoff context."
            )
            return

        ws = self._get_active_ws()
        if ws is None:
            self._emit_handoff_result(
                source_adapter,
                session_id,
                ok=False,
                reason="No active connection to the phone to start the new session on.",
            )
            return

        try:
            new_session_id = await self._start_session_for_agent(ws, destination_agent, cwd=cwd)
        except Exception:
            logger.exception(
                "agent handoff failed to start a %r session for cwd=%r", destination_agent, cwd
            )
            self._emit_handoff_result(
                source_adapter,
                session_id,
                ok=False,
                reason=f"Could not start a new {destination_label} session.",
            )
            return

        destination_adapter = self._adapter_for_agent(destination_agent)
        try:
            await destination_adapter.send_message(new_session_id, seed_message)
        except Exception:
            logger.exception("failed to seed handed-off session %r", new_session_id)
            self._emit_handoff_result(
                source_adapter,
                session_id,
                ok=False,
                reason=f"Started the new {destination_label} session but could not seed it with context.",
            )
            return

        self._emit_handoff_result(
            source_adapter, session_id, ok=True, new_session_id=new_session_id, agent=destination_agent
        )
