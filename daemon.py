"""Companion daemon: persistent, reconnecting, authenticated connection to
the relay's /ws/companion gateway (see relay/app.py's WebSocket auth
contract), with periodic heartbeats - plus the wiring that turns the SDK
(U3) and observe-only (U4) adapters from standalone classes into a live
remote-control loop:

  - Inbound `{"type": "action", "action": {...}}` messages (a phone's
    send_message/interrupt/respond_to_permission/start_session, forwarded
    verbatim by the relay - see relay/app.py) are routed to whichever
    adapter owns the named `session_id`.
  - `start_session` is the one action with no `session_id` yet: it always
    targets the SDK adapter (KD5 - only companion-started sessions are
    SDK-owned), which mints a fresh session_id and connects with the
    caller-supplied `cwd`.
  - Every adapter's event stream (`subscribe(session_id)`) is forwarded to
    the relay as `{"type": "event", "event": {...}}` - forwarding starts
    right after `start_session` for a new SDK-owned session, and
    `_watch_active_sessions` (re)starts it for any other session (an
    observe-only one U4 just discovered, or an SDK-owned one whose
    forwarder died with a previous connection) by polling both adapters'
    `discover_sessions()`.

State machine (`CompanionDaemon.state`):

    connecting -> connected -> reconnecting -> connecting -> ... -> stopped

A failed or dropped connection never raises out of `run()` - it always
falls back to `reconnecting` with exponential backoff (reset to
`reconnect_initial_delay` on every successful connect), per R1 (persistent
connection with automatic reconnect and heartbeats) and the System-Wide
Impact note that a companion restart must re-arm connectivity rather than
require manual intervention. Losing the relay connection cancels the
current connection's heartbeat/receive/observe-scan tasks, but SDK-owned
sessions and their adapter-side state are untouched - only the outward
forwarding pauses until reconnect (KTD1/R4 own session lifecycle, not this
module).

Runs as a macOS `launchd` user agent per KTD6 - see companion/launchd/ and
companion/README.md for install instructions. `main()` below is the
entrypoint the plist invokes (`python3 -m companion.daemon`).
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib.metadata
import json
import logging
import os
import pathlib
import pty
import re
import stat
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import websockets

from . import digest, git_status, history, projects, session_registry
from .agent_limit_coordinator import AgentLimitCoordinator
from .adapters.base import AdapterProtocol
from .adapters.codex_adapter import CodexAdapter
from .adapters.events import Event
from .adapters.observe_adapter import KNOWN_ENTRYPOINTS, ObserveAdapter
from .adapters.sdk_adapter import SDKAdapter
from .config import DEFAULT_SESSION_REGISTRY_PATH, CompanionConfig, load_config, save_config

logger = logging.getLogger(__name__)


def _companion_version() -> str:
    """Update Safety plan (U1), KTD1: reads this companion's own version
    via the installed package's metadata rather than a hand-maintained
    `__version__` string that could drift from the two pyproject.toml
    files (pyproject.toml, companion/pyproject.toml) already kept in sync
    manually per the release runbook - the installed package's own
    metadata is authoritative regardless of Homebrew, pip, or editable
    install.

    A real Homebrew/pip install always has this metadata, but this
    project's own local dev/test workflow runs straight from a source
    checkout with no `pip install -e .` step (confirmed: no CI workflow
    runs the Python test suite at all, and every daemon-integration test
    here starts a real `CompanionDaemon` against a real relay expecting it
    to connect) - `importlib.metadata.version` raises
    `PackageNotFoundError` in exactly that case. Falling back straight to
    "0.0.0" there would make a source checkout's own daemon permanently
    fail relay/app.py's version floor against every relay, including the
    embedded ones this test suite spins up - not a hypothetical, this
    broke `test_daemon_connects_authenticates_and_sends_heartbeats` and
    friends the first time this landed. So: fall back to reading this
    version straight out of `companion/pyproject.toml` (a sibling of this
    file when running from source) before falling back further to "0.0.0"
    - itself always below any relay version floor - only if that file is
    also missing or unparseable (some other packaging layout this hasn't
    anticipated), so a broken lookup still degrades to "rejected as too
    old" rather than crashing every connection attempt."""
    try:
        return importlib.metadata.version("remote-claude-companion")
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        pyproject_text = (pathlib.Path(__file__).parent / "pyproject.toml").read_text()
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject_text)
        if match:
            return match.group(1)
    except OSError:
        pass
    return "0.0.0"


# Read once at import time (daemon startup) and cached for the process
# lifetime (KTD1) - every hello this companion ever sends (_connect_and_serve
# below) reports this same value, rather than re-reading package metadata on
# every reconnect.
COMPANION_VERSION = _companion_version()

# Update Safety plan (U3), R4/R5/R6, KTD10: a second, separate Unix socket
# from ObserveAdapter's own hooks.sock (adapters/observe_adapter.py) - this
# one exists purely so the `companion precheck` CLI command (companion/cli.py)
# can ask a running daemon "how many sessions do you consider active right
# now" before an operator runs `brew upgrade`. Homebrew has no pre-upgrade
# hook to wire this into (see this unit's plan), so it's a standalone
# command the operator runs themselves - never something `brew upgrade`
# itself depends on or blocks on (R5).
DEFAULT_STATUS_SOCKET_PATH = os.path.expanduser("~/.config/remote-claude-companion/status.sock")

# Dynamic-linker/interpreter injection vectors - a phone-supplied cli_env
# containing any of these could hijack the spawned CLI subprocess (e.g.
# loading an attacker-controlled shared library into it) regardless of
# what cli_path itself points to. Rejected outright rather than stripped,
# so a rejected change is visible (the confirm-back below still reports
# the last-accepted cli_env) instead of silently landing minus one key.
_DANGEROUS_CLI_ENV_KEYS = frozenset({
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "DYLD_FRAMEWORK_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
})


def _redact_secrets(value: Any, secrets: tuple[str, ...]) -> Any:
    """KTD8: exact-substring redaction of currently-configured secret
    values (device_token, personal_oauth_token) from anything forwarded to
    the phone. Recurses through dicts/lists so it covers every event
    shape uniformly - tool_result content (str or list of content
    blocks), tool_call input (dict), assistant_message text (str) - without
    needing per-event-type handling. Applied at `_send_event` (the one
    function every live-forwarded event already passes through) and at the
    history-replay path, which bypasses `_send_event` entirely.

    `secrets` must already have falsy/empty values filtered out - an empty
    string as a `str.replace` target would insert the placeholder between
    every character."""
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        return value
    if isinstance(value, dict):
        return {key: _redact_secrets(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(item, secrets) for item in value]
    return value


# KTD5: `claude setup-token` performs its own full OAuth flow (opens a
# browser, waits for the user to approve there) - 90s gives a human enough
# room to notice the browser, log in if needed, and approve, unlike the
# near-instant SDK-connect case elsewhere in this module.
_SETUP_TOKEN_TIMEOUT_SECONDS = 90.0
_VALID_ACCOUNTS = frozenset({"vscode", "personal"})
# Codex Agent Integration plan (002), U2 (R10): Codex's own independent
# account-mode allowlist. Deliberately a distinct frozenset from
# _VALID_ACCOUNTS above, not a shared alias - the two agents' reject paths
# must stay independent even though the values happen to read the same.
_VALID_CODEX_ACCOUNTS = frozenset({"vscode", "personal"})
# Multi-Agent Adapter plan (009), U1 (R2): which adapter start_session
# routes a new session to. "claude_code" is the default so every existing
# caller that doesn't send `agent` at all keeps landing on sdk_adapter,
# byte-for-byte unchanged from before this field existed.
_VALID_AGENTS = frozenset({"claude_code", "codex"})

# code-review finding (P1): the post-restart SDK-resume fast path below
# (`_adapter_for(session_id, include_observed=False)` before ever trying
# `_try_resume_sdk_session`) previously applied to every action kind on
# every session_id, not just the narrow "just restarted, an SDK session's
# in-memory state was wiped" race it was built for. A session_id that was
# never SDK-owned at all (a real, ordinary observed session - e.g. a
# `claude` CLI session running directly in a terminal) has no
# session_registry row and no prior sdk_adapter history, so
# `_try_resume_sdk_session` had nothing to distinguish it from a genuinely
# resumable one - any action, including a read-only one like git_status,
# could trigger a real SDK resume attempt as a side effect, converting a
# passive observed session into an SDK-owned one and risking a resume
# against a transcript a separate terminal process is actively using.
# Restricting the fast path to the two action kinds that actually need a
# live, resumed SDK connection to do anything meaningful - starting a new
# turn, or answering a pending permission prompt mid-turn - means every
# other action kind resolves via the normal include_observed=True lookup
# first, so an already-known observed session's git_status/interrupt/
# end_session/etc. just reaches observe_adapter (a no-op there for the
# write-shaped ones) exactly as it did before this diff, with no resume
# side effect. The restart race this fast path exists for only matters for
# these two kinds anyway - nothing else needs a *live* SDK client to have
# an effect.
_RESUME_WORTHY_ACTION_KINDS = frozenset({"send_message", "respond_to_permission"})

# KTD4: set_personal_account_token's raw pasted token rides straight in the
# action dict - _handle_action's generic exception logger (below) logs the
# whole action on any dispatch failure, which would otherwise print it even
# though it was never yet written to self.config (and so isn't caught by
# _redact_secrets, which only knows about already-configured values).
#
# code-review finding: `cli_env` (set_cli_settings) is included too - a
# secret placed there by misuse could otherwise get logged in full at ERROR
# level if dispatch later raises for an unrelated reason, the same exposure
# `token` above exists to close.
_ACTION_KEYS_NEVER_LOGGED = ("token", "cli_env")


def _scrub_action_for_logging(action: dict[str, Any]) -> dict[str, Any]:
    return {
        key: ("[redacted]" if key in _ACTION_KEYS_NEVER_LOGGED and value is not None else value)
        for key, value in action.items()
    }


async def _run_setup_token_under_pty(cli_path: Optional[str]) -> Optional[str]:
    """KTD5's automated setup attempt: spawn `claude setup-token` (or a
    configured custom cli_path) under a pseudo-terminal so the CLI believes
    it has a real terminal, wait up to _SETUP_TOKEN_TIMEOUT_SECONDS for it
    to exit, and return the captured token if the output looks like exactly
    one clean token line - or None on any failure (binary not found, spawn
    error, timeout, non-zero exit, or ambiguous output), which the caller
    treats as "automation unavailable, fall back to manual entry".

    Per security review: never logs the raw pty buffer or the returned
    token, at any level - only the caller's success/failure boolean is
    ever observable outside this function.

    Whether `claude setup-token` actually tolerates running under a pty
    non-interactively is unverified in any environment this plan's
    research could reach (see Planning Contract Assumptions) - this
    function is written to the documented contract (opens its own OAuth
    flow, prints the token, exits) and fails closed to "unavailable" on
    anything it doesn't recognize, rather than guessing at a wrong token.
    """
    binary = cli_path or "claude"
    master_fd, slave_fd = pty.openpty()
    output_chunks: list[bytes] = []
    loop = asyncio.get_event_loop()

    def _drain() -> None:
        try:
            chunk = os.read(master_fd, 4096)
        except OSError:
            chunk = b""
        if chunk:
            output_chunks.append(chunk)
        else:
            loop.remove_reader(master_fd)

    try:
        process = await asyncio.create_subprocess_exec(
            binary, "setup-token", stdin=slave_fd, stdout=slave_fd, stderr=slave_fd
        )
    except OSError:
        os.close(master_fd)
        os.close(slave_fd)
        return None
    os.close(slave_fd)  # only the child needs the slave end

    os.set_blocking(master_fd, False)
    loop.add_reader(master_fd, _drain)
    try:
        try:
            await asyncio.wait_for(process.wait(), timeout=_SETUP_TOKEN_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return None
        if process.returncode != 0:
            return None
    finally:
        try:
            loop.remove_reader(master_fd)
        except (ValueError, OSError):
            pass
        os.close(master_fd)

    raw = b"".join(output_chunks).decode("utf-8", errors="ignore")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    candidate = lines[0]
    if " " in candidate or len(candidate) < 20:
        return None
    return candidate


# KTD3: outrank CLAUDE_CODE_OAUTH_TOKEN in the CLI's own auth precedence,
# so an ambient value (this daemon process's own inherited os.environ, or
# a phone-set cli_env entry) could otherwise silently defeat the "personal"
# identity switch.
_PRECEDENCE_ENV_KEYS_TO_NEUTRALIZE_FOR_PERSONAL = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def _is_valid_remote_cli_path(cli_path: str) -> bool:
    """A phone-supplied cli_path is a code-execution-adjacent primitive
    (it's what gets spawned for every subsequent session on this Mac) -
    this doesn't try to whitelist a specific binary name (users legitimately
    point this at a custom claude build or wrapper script - KTD5), but does
    require it to be an absolute path to a file that already exists on this
    machine and is executable. That closes the sharpest edge of "any phone
    token can point the companion at an arbitrary path" - it can only ever
    select among executables already present on disk, not stage a brand
    new one via this action alone."""
    if not os.path.isabs(cli_path):
        return False
    try:
        return os.path.isfile(cli_path) and os.access(cli_path, os.X_OK)
    except OSError:
        return False


class CompanionDaemon:
    """Owns one persistent (reconnecting) connection to the relay, plus the
    SDK-owned, observe-only, and Codex-owned adapters that connection's
    actions/events flow through.

    `sdk_adapter`/`observe_adapter`/`codex_adapter` are injectable so tests
    can substitute fakes (mirroring each adapter's own `client_factory`
    injection point) without spawning a real `claude`/Codex subprocess or
    touching the real `~/.claude/projects` / hooks socket paths.
    """

    def __init__(
        self,
        config: CompanionConfig,
        *,
        sdk_adapter: Optional[SDKAdapter] = None,
        observe_adapter: Optional[ObserveAdapter] = None,
        codex_adapter: Optional[CodexAdapter] = None,
        recents_path: Optional[str] = None,
        config_path: Optional[str] = None,
        session_registry_path: Optional[str] = None,
        status_socket_path: Optional[str] = None,
    ):
        self.config = config
        # Companion-Local Token Entry & SQLite Config plan, KTD2: the
        # shared session_registry.db file, not the legacy JSON path -
        # config.py's own load_config/save_config now read/write a table
        # inside it.
        self.config_path = config_path or DEFAULT_SESSION_REGISTRY_PATH
        self.sdk_adapter = sdk_adapter or SDKAdapter()
        self.observe_adapter = observe_adapter or ObserveAdapter(
            required_entrypoints=frozenset(config.observe_entrypoints),
            auto_approve=config.observe_auto_approve,
            llm_judge=config.observe_llm_judge,
        )
        # Multi-Agent Adapter plan (009), U1: the third adapter this
        # registry was always meant to grow into (see U2's comment below) -
        # CodexAdapter finally goes live here, wired into start_session's
        # dispatch (below) rather than sitting orphaned.
        self.codex_adapter = codex_adapter or CodexAdapter()
        # Multi-Agent Adapter plan (009), U2: dispatch/lookup/broadcast call
        # sites iterate this registry instead of naming self.sdk_adapter/
        # self.observe_adapter directly, so adding a third adapter (e.g.
        # CodexAdapter) only means appending it here - the named attributes
        # above stay too, for the call sites that intentionally mean "the
        # Claude-Code-specific adapter" (persistence gated on `adapter is
        # self.sdk_adapter`, observe-only settings actions, etc.), which
        # this plan deliberately leaves untouched (KTD2).
        self.adapters: list[AdapterProtocol] = [self.sdk_adapter, self.observe_adapter, self.codex_adapter]
        # Injectable for the same reason as the adapters above: tests must
        # never write into the developer's real recent-projects file
        # (projects.DEFAULT_RECENTS_PATH) just by exercising start_session.
        self.recents_path = recents_path
        # Same injectability reason, for session_registry.py's own default
        # path - see _try_resume_sdk_session for why this exists at all.
        self.session_registry_path = session_registry_path
        # Update Safety plan (U3), R4/R6, KTD10: same injectability reason
        # as the paths above - tests must never bind the real
        # ~/.config/remote-claude-companion/status.sock, and a machine
        # running two daemons in parallel (dev + real) needs distinct
        # sockets too.
        self.status_socket_path = status_socket_path or DEFAULT_STATUS_SOCKET_PATH
        self._status_server: Optional[asyncio.AbstractServer] = None
        self.state = "connecting"
        self.connect_attempts = 0
        self.heartbeats_sent = 0
        self._stop_event = asyncio.Event()
        self._backoff = config.reconnect_initial_delay
        # session_id -> the task currently forwarding its events, bound to
        # whichever connection spawned it. See _serve_connection's cleanup
        # for why these must be torn down on disconnect, not left to
        # discover their own dead socket.
        self._forwarding: dict[str, asyncio.Task] = {}
        # session_id -> which adapter object owns the entry above - lets
        # _watch_active_sessions detect and correct the one case where the
        # wrong adapter won a session_id's forwarder slot (see that
        # method's own comment on ObserveAdapter/SDKAdapter session_id
        # collisions) rather than silently forwarding the wrong session's
        # events forever.
        self._forwarding_adapter: dict[str, object] = {}
        # _receive_loop dispatches every inbound action as its own
        # asyncio.create_task, unordered - so two actions that each read-
        # modify-write the same on-disk file (session_registry.py's
        # upsert_session, projects.py's record_recent) can interleave and
        # silently drop one write. Each lock serializes one file's
        # read-modify-write sequence across concurrently-dispatched actions
        # on this daemon instance; separate locks since the two files are
        # unrelated and there's no reason to block one on the other.
        self._session_registry_lock = asyncio.Lock()
        self._recents_lock = asyncio.Lock()
        # code-review finding (P1): write_session_file's own read-compare-
        # write conflict check (git_status.py) is race-safe only *within*
        # one call - nothing serializes two concurrently-dispatched saves
        # for the same (session_id, path), so each can pass its own check
        # against the same pre-write content and one silently overwrites
        # the other. (session_id, path) -> asyncio.Lock, created lazily via
        # _write_session_file_lock_for below and never removed - the same
        # per-key-forever tradeoff session_registry.py's own lack of
        # pruning already accepts, and one lock per distinct edited file
        # this daemon has ever seen is not a real memory concern.
        self._write_session_file_locks: dict[tuple[str, str], asyncio.Lock] = {}
        # Agent Limit Handoff plan, U3: the one currently-serving connection
        # (or None while connecting/reconnecting) - set/cleared by
        # _serve_connection. AgentLimitCoordinator._start_agent_handoff
        # needs a `ws` to spawn the handed-off session's own forwarder on
        # (via _start_session_for_agent), but it's reached from
        # resolve_agent_limit_offer, several calls below the one place `ws`
        # is naturally in scope (_dispatch_session_scoped_action) -
        # threading it down through every intermediate signature was
        # rejected in favor of this, since there is only ever at most one
        # live connection at a time (this module's own docstring:
        # "connecting -> connected -> ..."). Read via the get_active_ws
        # callback passed to AgentLimitCoordinator below, rather than that
        # class holding a stale copy.
        self._active_ws: Optional["websockets.WebSocketClientProtocol"] = None
        # Code-review fix (whole-codebase SOLID/DRY audit): the
        # (agent, rate_limit_type)-keyed offer-tracking state and the
        # handoff orchestration logic (KTD3/KTD6 - deriving/escalating/
        # resolving agent_limit_offer, composing a digest+diff, starting
        # and seeding the destination session) used to live directly on
        # CompanionDaemon as ~345 lines of its own methods - the single
        # most self-contained concern in this already-large file, so it's
        # now its own collaborator instead. See agent_limit_coordinator.py.
        self.agent_limits = AgentLimitCoordinator(
            sdk_adapter=self.sdk_adapter,
            codex_adapter=self.codex_adapter,
            observe_adapter=self.observe_adapter,
            config=self.config,
            get_active_ws=lambda: self._active_ws,
            start_session_for_agent=self._start_session_for_agent,
            effective_cli_env=self._effective_cli_env,
        )

    def stop(self) -> None:
        """Request a graceful stop. `run()` returns once the current
        connect/heartbeat cycle notices, typically within one sleep tick."""
        self._stop_event.set()

    async def run(self) -> None:
        await self.observe_adapter.start()
        try:
            # Update Safety plan (U3), R4/R6, code review finding: started
            # here (inside the try, ahead of the while loop) rather than
            # before it - a failure in _start_status_socket() (e.g. a
            # permission error, or two daemon instances contending for the
            # same default status_socket_path, exactly the scenario
            # status_socket_path's own constructor comment calls out) must
            # still reach the finally below and tear down the
            # already-started observe_adapter, not leak its socket. Started
            # ahead of the connect/reconnect loop, not inside it, because
            # the precheck must keep answering "how many active sessions"
            # even while this daemon is offline or stuck reconnecting to
            # the relay - exactly the situation an operator most needs it
            # to still work in.
            await self._start_status_socket()
            while not self._stop_event.is_set():
                self.state = "connecting"
                self.connect_attempts += 1
                try:
                    await self._connect_and_serve()
                except (OSError, websockets.exceptions.WebSocketException) as exc:
                    # Update Safety plan (U2/R10/KTD9): a 4402 close (a
                    # ConnectionClosed, which is itself a WebSocketException
                    # - already caught by this clause) names this
                    # companion's own version as below the relay's
                    # supported floor (KTD6) - retrying would just get
                    # rejected again forever, with the rejection message
                    # never surfacing anywhere an operator would see it
                    # (this daemon has no interactive UI of its own). A
                    # plain `return` here is not enough: main() has no
                    # sys.exit after asyncio.run(daemon.run()), so exiting
                    # cleanly would just have launchd/brew's KeepAlive
                    # respawn the process into the exact retry loop this
                    # exists to eliminate (see this file's module docstring
                    # and Formula/afk-claude-companion.rb's service block).
                    # Blocking on _stop_event instead keeps the process
                    # alive - satisfying KeepAlive's "long-running service"
                    # expectation - while genuinely never reconnecting; only
                    # an explicit stop() call or the next deliberate service
                    # restart ends it.
                    if isinstance(exc, websockets.exceptions.ConnectionClosed) and exc.code == 4402:
                        self.state = "version_blocked"
                        logger.error(
                            "relay rejected this companion's connection for being below "
                            "the supported version floor - it will NOT retry until "
                            "upgraded and restarted: %s",
                            exc.reason,
                        )
                        await self._stop_event.wait()
                    else:
                        logger.info("companion connection lost/failed: %s", exc)
                if self._stop_event.is_set():
                    break
                self.state = "reconnecting"
                await self._sleep_or_stop(self._backoff)
                self._backoff = min(self._backoff * 2, self.config.reconnect_max_delay)
        finally:
            await self.observe_adapter.stop()
            await self._stop_status_socket()
            # A still-active SDK-owned/Codex session's own background task
            # (SDKAdapter's reader_task, CodexAdapter's stream_task) would
            # otherwise outlive this daemon shutdown, leaked - orphaned,
            # pending, and destroyed by the interpreter with no one left to
            # await it (asyncio's own "Task was destroyed but it is
            # pending!" warning). disconnect_all() cancels and awaits each.
            await self.sdk_adapter.disconnect_all()
            await self.codex_adapter.disconnect_all()
        self.state = "stopped"

    async def _connect_and_serve(self) -> None:
        async with websockets.connect(self.config.relay_url) as ws:
            # Update Safety plan (U1/R1): reports this companion's own
            # version at connect time - the relay is the sole enforcement
            # point for version compatibility (R2), so this daemon makes no
            # attempt to interpret or validate the relay's own version in
            # return, only to report its own honestly.
            await ws.send(
                json.dumps(
                    {"token": self.config.device_token, "type": "hello", "version": COMPANION_VERSION}
                )
            )
            reply = json.loads(await ws.recv())
            if reply.get("type") != "ack":
                raise websockets.exceptions.WebSocketException(
                    f"relay did not acknowledge companion auth: {reply!r}"
                )

            self._backoff = self.config.reconnect_initial_delay  # reset on success
            self.state = "connected"
            await self._serve_connection(ws)

    async def _serve_connection(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """Run heartbeat, inbound-action handling, and observe-session
        discovery concurrently until the connection drops or a stop is
        requested - whichever happens first tears down the others."""
        self._active_ws = ws
        tasks = [
            asyncio.create_task(self._heartbeat_loop(ws)),
            asyncio.create_task(self._receive_loop(ws)),
            asyncio.create_task(self._watch_active_sessions(ws)),
            asyncio.create_task(self._stop_event.wait()),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                exc = task.exception() if not task.cancelled() else None
                if exc is not None:
                    raise exc
        finally:
            self._active_ws = None
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Every entry in self._forwarding was spawned against *this*
            # connection's `ws`. Left alone, a still-blocked forwarder
            # (idle on adapter.subscribe()'s queue, having sent nothing
            # yet) keeps its session_id marked "being forwarded"
            # indefinitely - so a reconnect's _watch_active_sessions
            # would see the session_id already present and never restart
            # it, silently dropping events sent to it after the reconnect
            # (regression: see test_sdk_session_forwarding_resumes_after_relay_reconnect).
            # Cancelling explicitly here, rather than waiting for each
            # forwarder to discover the dead socket on its own next send,
            # is what actually makes forwarding resume on the new connection.
            forwarder_tasks = list(self._forwarding.values())
            for task in forwarder_tasks:
                task.cancel()
            await asyncio.gather(*forwarder_tasks, return_exceptions=True)
            self._forwarding.clear()
            self._forwarding_adapter.clear()

    async def _heartbeat_loop(self, ws: "websockets.WebSocketClientProtocol") -> None:
        while True:
            await self._sleep_or_stop(self.config.heartbeat_interval)
            if self._stop_event.is_set():
                return
            await ws.send(
                json.dumps(
                    {"token": self.config.device_token, "type": "heartbeat", "seq": self.heartbeats_sent}
                )
            )
            self.heartbeats_sent += 1

    async def _receive_loop(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """The single reader of this connection. Acks for messages the
        daemon itself sent (hello/heartbeat/event) need no handling here;
        `{"type": "action", ...}` is the relay pushing a phone's action
        (see relay/app.py's action-forwarding contract)."""
        async for raw in ws:
            try:
                message = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                logger.warning("dropping malformed message from relay: %r", raw)
                continue
            if not isinstance(message, dict):
                continue
            if message.get("type") == "action":
                asyncio.create_task(self._handle_action(ws, message.get("action") or {}))

    async def _handle_action(self, ws: "websockets.WebSocketClientProtocol", action: dict[str, Any]) -> None:
        # `_serve_connection` spawns this via asyncio.create_task (fire-and-
        # forget, per action) - an exception here would otherwise surface
        # only as asyncio's own "exception was never retrieved" warning
        # instead of our own logging, and would silently drop that one
        # action rather than the intended log-and-continue.
        try:
            await self._dispatch_action(ws, action)
        except Exception:
            logger.exception("action %r failed: %r", action.get("kind"), _scrub_action_for_logging(action))

    async def _dispatch_action(self, ws: "websockets.WebSocketClientProtocol", action: dict[str, Any]) -> None:
        kind = action.get("kind")

        if kind == "start_session":
            # Multi-Agent Adapter plan (009), U1 (R2): "claude_code" default
            # keeps every pre-existing caller (nothing sends `agent` yet)
            # routed to sdk_adapter exactly as before - only an explicit
            # "codex" opts into the new adapter. An unrecognized value is
            # rejected outright (logged, no session created), mirroring
            # _handle_set_active_account's own reject-unknown-value
            # convention below - there's no start_session-specific confirm-
            # back event to reply on, so silently declining to create a
            # session (like the missing-session_id case elsewhere in this
            # method) is the equivalent "reply".
            agent = action.get("agent") or "claude_code"
            if agent not in _VALID_AGENTS:
                logger.warning(
                    "rejected start_session agent %r: must be one of %r", agent, sorted(_VALID_AGENTS)
                )
                return
            try:
                await self._start_session_for_agent(
                    ws,
                    agent,
                    cwd=action.get("cwd"),
                    model=action.get("model"),
                    permission_mode=action.get("permission_mode"),
                    sandbox=action.get("sandbox"),
                    # Bug fix (user report): echoed back verbatim on
                    # session_started so the requesting phone can recognize
                    # its own new session without comparing cwd strings -
                    # see sdk_adapter.py's connect() for why cwd-string
                    # matching broke (realpath resolves through symlinks).
                    client_request_id=action.get("client_request_id"),
                )
            except Exception:
                logger.exception("start_session failed for action %r", action)
            return

        if kind == "list_projects":
            await self._handle_list_projects(ws)
            return

        if kind == "list_active_sessions":
            await self._handle_list_active_sessions(ws)
            return

        if kind == "list_project_sessions":
            await self._handle_list_project_sessions(ws, action.get("cwd"))
            return

        if kind == "read_session_history":
            await self._handle_read_session_history(ws, action.get("session_id"))
            return

        if kind == "open_session":
            # R: "do not connect automatically to opened session, just show
            # that it exists" - an observe-only session's content only
            # starts forwarding once the phone actually opens it (see
            # ObserveAdapter.open_session). No-op for an SDK-owned session,
            # which already forwards from the moment start_session created it.
            self.observe_adapter.open_session(action.get("session_id"))
            return

        if kind == "get_observe_settings":
            await self._handle_get_observe_settings(ws)
            return

        if kind == "set_observe_entrypoints":
            await self._handle_set_observe_entrypoints(ws, action.get("entrypoints"))
            return

        if kind == "get_auto_approve_settings":
            await self._handle_get_auto_approve_settings(ws)
            return

        if kind == "set_auto_approve_settings":
            await self._handle_set_auto_approve_settings(
                ws, action.get("auto_approve"), action.get("llm_judge")
            )
            return

        if kind == "get_cli_settings":
            await self._handle_get_cli_settings(ws)
            return

        if kind == "set_cli_settings":
            await self._handle_set_cli_settings(ws, action.get("cli_path"), action.get("cli_env"))
            return

        if kind == "get_account_settings":
            await self._handle_get_account_settings(ws)
            return

        if kind == "set_active_account":
            await self._handle_set_active_account(ws, action.get("active_account"))
            return

        if kind == "set_personal_account_token":
            await self._handle_set_personal_account_token(ws, action.get("token"))
            return

        if kind == "start_personal_account_setup":
            await self._handle_start_personal_account_setup(ws)
            return

        if kind == "set_active_codex_account":
            await self._handle_set_active_codex_account(ws, action.get("codex_active_account"))
            return

        if kind == "set_personal_codex_account_token":
            await self._handle_set_personal_codex_account_token(ws, action.get("token"))
            return

        if kind == "compose_digest":
            await self._handle_compose_digest(ws, action.get("session_summaries"))
            return

        # code-review finding (P2): everything past this point used to stay
        # inline in this one method, alongside every other action-kind
        # branch above - splitting the session-scoped actions (the largest
        # single cohesive group: turn control, git/file access, and the
        # various set_session_* live-config actions) into their own
        # dispatcher gives this growing set of actions a dedicated place to
        # land instead of adding yet another branch to an already-long
        # do-everything router.
        await self._dispatch_session_scoped_action(ws, kind, action)

    async def _start_session_for_agent(
        self,
        ws: "websockets.WebSocketClientProtocol",
        agent: str,
        *,
        cwd: Optional[str],
        model: Optional[str] = None,
        permission_mode: Optional[str] = None,
        sandbox: Optional[str] = None,
        client_request_id: Optional[str] = None,
    ) -> str:
        """KTD9 (Agent Limit Handoff plan, U3): the connect/_spawn_forwarder/
        recents/registry sequence that used to live inline in the
        `start_session` action's own dispatch branch above - extracted so
        that branch (unchanged behavior - every kwarg here is threaded
        straight from the action payload the same way it always was) and
        U3's handoff orchestration (AgentLimitCoordinator._start_agent_handoff,
        agent_limit_coordinator.py, via the start_session_for_agent callback
        it's constructed with below) share one copy instead of risking two
        ~30-line copies drifting apart.

        Mints a fresh session_id, connects the chosen adapter (codex_adapter
        for "codex", sdk_adapter for anything else - callers must have
        already validated `agent` against _VALID_AGENTS; this trusts that),
        spawns its event forwarder on `ws`, records the resolved cwd as a
        recent project, and (sdk_adapter only) upserts a session_registry
        row. Returns the new session_id. Raises whatever adapter.connect()
        raises on failure - deliberately not caught here, so each caller
        decides how to surface it (start_session's dispatch logs and drops
        the action; U3's handoff path reports agent_handoff_result with
        ok=False and a user-facing reason)."""
        session_id = str(uuid4())
        adapter: AdapterProtocol = self.codex_adapter if agent == "codex" else self.sdk_adapter
        if agent == "codex":
            # No permission_mode: Codex has no interactive per-tool-call
            # approval concept (see codex_adapter.py's module docstring,
            # spike-finding #2) - it runs under its own
            # ApprovalMode.auto_review instead.
            #
            # U2 (PKTD3): api_key mirrors _effective_cli_env's own
            # active_account branch below, one level simpler - Codex has no
            # ambient-env-precedence-stripping concern to replicate (KTD3),
            # just "pass the configured personal key under 'personal', pass
            # nothing under 'vscode'". Only ever reads codex_active_account/
            # codex_personal_api_key, never Claude's own active_account/
            # personal_oauth_token (R10).
            #
            # Merge (main): client_request_id (bug fix, user report -
            # 2aba83b) applies equally to Codex-started sessions - mobile's
            # reduceEvent() matches ANY session_started event by
            # client_request_id regardless of agent, so omitting it here
            # would silently regress that fix back to cwd-string matching
            # for Codex sessions specifically.
            await self.codex_adapter.connect(
                session_id,
                cwd=cwd,
                model=model,
                sandbox=sandbox,
                api_key=(
                    self.config.codex_personal_api_key
                    if self.config.codex_active_account == "personal"
                    else None
                ),
                client_request_id=client_request_id,
            )
        else:
            await self.sdk_adapter.connect(
                session_id,
                cwd=cwd,
                model=model,
                permission_mode=permission_mode,
                # Unlike model/permission_mode above, cli_path/cli_env
                # (KTD5) are Mac-level settings with no phone-side
                # per-request equivalent - always sourced from this
                # companion's own config, never the action payload. cli_env
                # goes through _effective_cli_env (R2/KTD1/KTD3), not the
                # raw config field, so a "personal"-identity spawn gets
                # CLAUDE_CODE_OAUTH_TOKEN merged in.
                cli_path=self.config.cli_path,
                cli_env=self._effective_cli_env(),
                client_request_id=client_request_id,
            )
        self._spawn_forwarder(ws, adapter, session_id)
        resolved_cwd = adapter.get_cwd(session_id)
        if resolved_cwd:
            async with self._recents_lock:
                await asyncio.to_thread(projects.record_recent, resolved_cwd, self.recents_path)
            # Code-review fix: only sdk_adapter sessions get a
            # session_registry row. That row exists purely to support
            # _try_resume_sdk_session after a restart, which resolves cwd
            # via observe_adapter.projects_dir (~/.claude/projects) - a
            # Codex session_id can never match there, so a Codex row could
            # never actually be resumed. Writing one anyway would leave it
            # permanently 'active' forever, since the end-marking gates
            # elsewhere are (correctly) sdk_adapter-only - simpler to never
            # create the unresumable row than to widen every end-marking
            # gate just to clean up after it.
            if adapter is self.sdk_adapter:
                async with self._session_registry_lock:
                    await asyncio.to_thread(
                        session_registry.upsert_session,
                        session_id,
                        cwd=resolved_cwd,
                        model=model,
                        permission_mode=permission_mode,
                        path=self.session_registry_path,
                    )
        return session_id

    async def _dispatch_session_scoped_action(
        self, ws: "websockets.WebSocketClientProtocol", kind: Optional[str], action: dict[str, Any]
    ) -> None:
        session_id = action.get("session_id")
        if not session_id:
            logger.warning("action missing session_id: %r", action)
            return

        if kind in _RESUME_WORTHY_ACTION_KINDS:
            adapter = self._adapter_for(session_id, include_observed=False)
            if adapter is None:
                adapter = await self._try_resume_sdk_session(session_id)
            if adapter is None:
                adapter = self._adapter_for(session_id)
        else:
            adapter = self._adapter_for(session_id)
            if adapter is None:
                adapter = await self._try_resume_sdk_session(session_id)
        if adapter is None:
            logger.warning("action for unknown session_id=%r: %r", session_id, action)
            # fix(review): discovered via real-device testing - previously
            # this just logged and dropped the action, leaving the phone
            # with zero signal (a real user hitting this after e.g. a Mac
            # reboot mid-session would see "End Session"/"Catch me up" just
            # silently do nothing forever, with no way to tell what
            # happened). No adapter exists for this session_id (that's the
            # whole reason we're here), so this can't go through
            # adapter.emit_custom - sent directly on ws, same shape every
            # other one-off/sentinel event in this file already uses.
            event = {
                "session_id": session_id,
                "event_id": 0,
                "type": "session_unavailable",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": {},
            }
            await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))
            return

        try:
            if kind == "send_message":
                await adapter.send_message(session_id, action.get("text", ""))
            elif kind == "interrupt":
                await adapter.interrupt(session_id)
            elif kind == "compact":
                await adapter.compact(session_id)
            elif kind == "end_session":
                # code-review finding: marking the registry 'ended' had
                # previously only happened reactively, inside
                # _forward_events, once it observed the session_ended event
                # disconnect() below eventually emits. That left a real
                # window open between adapter.disconnect() popping this
                # session out of _sessions (synchronous, the adapter's very
                # first statement) and the registry write actually landing
                # (several awaits later) - any other action for this same
                # session_id dispatched by _receive_loop's own unordered
                # asyncio.create_task in that window would find no adapter,
                # fall into _try_resume_sdk_session, see the registry still
                # 'active', and silently resurrect the session the user
                # just explicitly ended. Doing the write here, synchronously
                # and before the pop, closes that window for the primary
                # trigger - _forward_events' own reactive mark stays in
                # place as the only mechanism for a session that ends on
                # its own (no end_session action to hook into).
                if adapter is self.sdk_adapter:
                    async with self._session_registry_lock:
                        await asyncio.to_thread(
                            session_registry.mark_session_ended, session_id, self.session_registry_path
                        )
                await adapter.disconnect(session_id)
            elif kind == "respond_to_permission":
                await adapter.respond_to_permission(
                    session_id,
                    action.get("request_id"),
                    action.get("decision"),
                    message=action.get("message", ""),
                )
            elif kind == "resolve_agent_limit_offer":
                # Agent Limit Handoff plan, U2 (KTD4): mirrors
                # respond_to_permission's own decision-parameter shape one
                # line up - dispatched here rather than through `adapter`
                # itself (unlike respond_to_permission, this isn't an
                # adapter capability - the offer state lives on
                # self.agent_limits, not on any one session's adapter).
                await self.agent_limits.resolve_agent_limit_offer(
                    adapter, session_id, action.get("rate_limit_type"), action.get("decision")
                )
            elif kind == "git_status":
                await self._handle_git_status(adapter, session_id)
            elif kind == "git_diff":
                await self._handle_git_diff(adapter, session_id, action.get("path"))
            elif kind == "get_file_content":
                await self._handle_get_file_content(adapter, session_id, action.get("path"))
            elif kind == "write_session_file":
                await self._handle_write_session_file(
                    adapter,
                    session_id,
                    action.get("path"),
                    action.get("previous_content"),
                    action.get("new_content"),
                )
            elif kind == "get_session_digest":
                await self._handle_session_digest(adapter, session_id)
            elif kind == "set_session_auto_approve":
                # Permission-mode-picker plan (U2): this action's
                # session_registry persistence side effect for an
                # SDK-owned session is retired along with auto_approve/
                # llm_judge as registry concepts (R7) - the action payload
                # never carried a permission_mode to persist instead.
                # code-review finding: SDKAdapter no longer defines
                # set_session_auto_approve at all (U1/U3 replaced it with
                # set_session_permission_mode/set_session_model) - only
                # ObserveAdapter still has it, for its own separate,
                # untouched auto_approve/llm_judge mechanism (R11). Guard
                # explicitly rather than relying on both adapters sharing a
                # method name, matching the two new sibling branches below.
                if adapter is self.observe_adapter:
                    adapter.set_session_auto_approve(
                        session_id, action.get("auto_approve"), action.get("llm_judge")
                    )
            elif kind == "set_session_permission_mode":
                # R8/R9: live, no-reconnect mode change - SDK-owned
                # sessions only. Unlike set_session_auto_approve above,
                # this has no ObserveAdapter equivalent to fall back to
                # (permission_mode is a native-SDK concept; an observed
                # session has no `client` of ours to send a control
                # request to at all) - so it needs its own explicit guard
                # rather than relying on both adapters sharing a method
                # name. Persisted only on success (adapter.
                # set_session_permission_mode already returned False, with
                # its own distinct log, for an unknown session_id or a
                # racing end_session) - this write is purely so a later
                # companion-restart resume picks up the live change; it is
                # not what applies it live.
                if adapter is self.sdk_adapter:
                    mode = action.get("permission_mode")
                    if await adapter.set_session_permission_mode(session_id, mode):
                        await self._persist_sdk_session_registry(session_id, permission_mode=mode)
            elif kind == "set_session_model":
                # R10: model's counterpart to set_session_permission_mode
                # directly above - same guard, same persist-on-success-only
                # shape.
                #
                # Codex Model & Sandbox Config plan (001), U2 (KTD5): widened
                # to branch by adapter instead of introducing a Codex-
                # specific action name for the same concept - a Codex-owned
                # session's set_session_model reaches CodexAdapter directly.
                # No `await` (CodexAdapter.set_session_model is sync - purely
                # local state, no SDK round-trip) and no session_registry
                # persistence (that registry only exists to support
                # sdk_adapter's own _try_resume_sdk_session; Codex sessions
                # never get a row there - see start_session's own comment on
                # this above).
                if adapter is self.sdk_adapter:
                    model = action.get("model")
                    if await adapter.set_session_model(session_id, model):
                        await self._persist_sdk_session_registry(session_id, model=model)
                elif adapter is self.codex_adapter:
                    self.codex_adapter.set_session_model(session_id, action.get("model"))
            elif kind == "set_session_sandbox":
                # Codex Model & Sandbox Config plan (001), U2 (R6/R7/KTD5):
                # Codex-only - Claude has no sandbox concept, so this is
                # guarded to CodexAdapter alone (not a shared-name branch
                # like set_session_model above) and is a no-op for any other
                # adapter, matching the existing observe-only-session no-op
                # convention.
                if adapter is self.codex_adapter:
                    self.codex_adapter.set_session_sandbox(session_id, action.get("sandbox"))
            elif kind == "stop_task":
                # Bug fix (user report): stops a background subagent this
                # session dispatched via the "Task" tool - SDK-owned only,
                # same guard as set_session_permission_mode/set_session_model
                # above (ObserveAdapter/CodexAdapter have no client of ours
                # to send this control request to). Nothing to persist on
                # success - unlike permission_mode/model, there's no
                # resume-time state a stopped task needs to carry forward.
                if adapter is self.sdk_adapter:
                    await adapter.stop_task(session_id, action.get("task_id"))
            else:
                logger.warning("unknown action kind: %r", kind)
        except Exception:
            logger.exception("action %r failed for session %r", kind, session_id)

    async def _handle_list_projects(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """Sessions-screen picker (no session_id, unlike git_status/
        git_diff): sent as a synthetic event on a fixed sentinel
        session_id rather than through an adapter's `emit_custom`, since
        that requires an already-connected session and this isn't scoped
        to one. `event_id` is always 0 - the relay's replay cache keys on
        (session_id, event_id) and just overwrites, which is exactly what
        we want for a point-in-time snapshot with no history to replay."""
        result = await asyncio.to_thread(projects.list_projects, None, self.recents_path)
        event = {
            "session_id": "_projects",
            "event_id": 0,
            "type": "project_list",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"projects": result},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_list_active_sessions(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """SessionListScreen.tsx has no other way to learn what's already
        running - it normally only ever finds out about a session by
        witnessing its session_started event live, which a fresh screen
        instance (e.g. navigated back to after unmounting - React
        Navigation only keeps a *covered* screen alive, not one you've
        actually navigated back past) never gets to see. This snapshot
        fills that gap on mount, the same way _handle_list_projects fills
        it for the project picker."""
        sessions = await self._gather_active_sessions()
        event = {
            "session_id": "_active_sessions",
            "event_id": 0,
            "type": "active_sessions",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"sessions": sessions},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _gather_active_sessions(self) -> list[dict[str, Any]]:
        """The combined snapshot _handle_list_active_sessions sends to the
        phone - extracted (Update Safety plan U3, KTD11) so
        _count_active_sessions below can share it rather than separately
        re-deriving its own view of which sessions exist. Mixes
        genuinely-live sessions from all three adapters (`active: True`,
        from each adapter's own discover_sessions()/is_active()) with
        restart-recovered-but-not-yet-reconnected sessions the
        session_registry still remembers from before a restart, which are
        deliberately appended below with `active: False` (R4/KTD3) - see
        that section's own comment."""
        live_sdk_session_ids = set(self.sdk_adapter.discover_sessions())
        sessions = [
            {
                "session_id": sid,
                "cwd": self.sdk_adapter.get_cwd(sid),
                "mode": "sdk_owned",
                "active": bool(self.sdk_adapter.is_active(sid)),
                # Cross-Session Digest plan (010): mirrors KTD4's "claude_code"
                # default (also session_started's own default, per Multi-Agent
                # Adapter plan KTD4) - this snapshot path was the one gap that
                # plan's own review flagged (correctness residual risk) as
                # never carrying agent, unlike the live session_started event.
                "agent": "claude_code",
            }
            for sid in live_sdk_session_ids
        ] + [
            {
                "session_id": sid,
                "cwd": self.codex_adapter.get_cwd(sid),
                "mode": "sdk_owned",
                "active": bool(self.codex_adapter.is_active(sid)),
                # Code-review fix: this snapshot previously only knew
                # about sdk_adapter/observe_adapter (from before
                # codex_adapter existed) - a live Codex session was
                # entirely absent from it, disappearing from the phone's
                # list on any remount that lost live-witnessed state.
                "agent": "codex",
            }
            for sid in self.codex_adapter.discover_sessions()
        ] + [
            {
                "session_id": sid,
                "cwd": self.observe_adapter.get_cwd(sid),
                "mode": "observe_only",
                "active": bool(self.observe_adapter.is_active(sid)),
                # Code-review fix: an observed session can genuinely be
                # Codex now (ObserveAdapter.get_agent) - the "claude_code"
                # literal here used to be correct unconditionally, back
                # when this was the only agent ObserveAdapter could ever
                # discover.
                "agent": self.observe_adapter.get_agent(sid) or "claude_code",
            }
            for sid in self.observe_adapter.discover_sessions()
        ]
        # R4/KTD3: a session known from before a restart isn't in
        # sdk_adapter.discover_sessions() at all yet (that's purely
        # in-memory, wiped by the restart) - list it anyway, unreconnected
        # (active=False), so it's visible without "stay lazy" requiring it
        # to be proactively reconnected to a live CLI process first.
        try:
            async with self._session_registry_lock:
                known = await asyncio.to_thread(
                    session_registry.list_known_sessions, status="active", path=self.session_registry_path
                )
        except Exception:
            # code-review finding: unlike the resume-safety read above, a
            # failure here only affects *visibility* of restart-recovered
            # sessions, not R5's resume-safety guarantee - degrade to just
            # the live in-memory sessions already gathered above rather
            # than dropping this whole response.
            logger.exception("session_registry read failed while listing active sessions")
            known = []
        sessions.extend(
            {"session_id": record.session_id, "cwd": record.cwd, "mode": "sdk_owned", "active": False}
            for record in known
            if record.session_id not in live_sdk_session_ids
        )
        return sessions

    async def _count_active_sessions(self) -> int:
        """Update Safety plan (U3), R4/R5/R6, KTD11: how many sessions this
        daemon currently considers active - the pre-upgrade check an
        operator runs themselves before `brew upgrade` (never a hook
        Homebrew runs on its own - see this unit's plan). Shares
        _gather_active_sessions's combined list rather than separately
        re-deriving "what counts as a session" - that list deliberately
        mixes truly-live sessions with restart-recovered-but-not-yet-
        reconnected ones appended with `active: False` (see that method's
        own docstring), so filtering on the `active` flag here (not a bare
        `len()`) is what keeps this count from overstating how many
        sessions an upgrade could actually disrupt."""
        sessions = await self._gather_active_sessions()
        return sum(1 for session in sessions if session["active"])

    async def _start_status_socket(self) -> None:
        """Update Safety plan (U3), R4/R6, KTD10: a Unix socket serving
        exactly one purpose - answering `companion precheck` (companion/
        cli.py). Same lifecycle shape as ObserveAdapter.start()'s own hooks
        socket (adapters/observe_adapter.py): stale-socket cleanup from a
        prior crashed run, then 0600 permissions (local-only - no reason
        for another user on this machine to learn session counts)."""
        os.makedirs(os.path.dirname(self.status_socket_path) or ".", exist_ok=True)
        if os.path.exists(self.status_socket_path):
            os.remove(self.status_socket_path)  # stale socket from a prior crashed run
        self._status_server = await asyncio.start_unix_server(
            self._handle_status_connection, path=self.status_socket_path
        )
        os.chmod(self.status_socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - local privilege surface (Risk)

    async def _stop_status_socket(self) -> None:
        """Mirrors ObserveAdapter.stop()'s own socket teardown."""
        if self._status_server is not None:
            self._status_server.close()
            await self._status_server.wait_closed()
        if os.path.exists(self.status_socket_path):
            os.remove(self.status_socket_path)

    async def _handle_status_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """R4/R5/R6: there's only ever one question a client can ask here,
        so unlike ObserveAdapter's hooks socket this never reads a request
        payload - it just answers as soon as a client connects. Writes the
        response unconditionally from a `finally`, mirroring
        ObserveAdapter._handle_hook_connection's own shape, so a bug in
        _count_active_sessions can't leave `companion precheck` hanging on
        a connection that's never going to answer.

        Code review finding (testing/correctness, U3): an internal failure
        here used to answer with the bare `{"active_sessions": null}` a
        client also gets from a plain OSError (socket doesn't exist,
        connection refused) - `precheck` couldn't tell "the daemon is up
        but something broke counting sessions" apart from "the daemon
        isn't reachable at all", which matters to an operator deciding
        whether to upgrade anyway. The `error` field, present only on this
        branch, is precheck's signal to report the distinction."""
        response: dict[str, Any] = {"active_sessions": None}
        try:
            response = {"active_sessions": await self._count_active_sessions()}
        except Exception as exc:
            logger.exception("status socket: failed to count active sessions")
            response = {"active_sessions": None, "error": str(exc) or exc.__class__.__name__}
        finally:
            writer.write(json.dumps(response).encode("utf-8") + b"\n")
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()

    async def _handle_list_project_sessions(
        self, ws: "websockets.WebSocketClientProtocol", cwd: Optional[str]
    ) -> None:
        """Past-session browsing (R: "see previous history of the
        project"), read-only - see companion/history.py. Sentinel
        session_id embeds the requested cwd so a phone browsing two
        different projects' histories in quick succession gets two
        distinguishable cached responses rather than one clobbering the
        other (unlike _handle_list_projects, which has only ever needed
        one global snapshot)."""
        if not cwd:
            logger.warning("list_project_sessions action missing cwd")
            return
        result = await asyncio.to_thread(history.list_project_sessions, cwd)
        event = {
            "session_id": f"_history_list:{cwd}",
            "event_id": 0,
            "type": "session_history_list",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"cwd": cwd, "sessions": [asdict(s) for s in result]},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_read_session_history(
        self, ws: "websockets.WebSocketClientProtocol", session_id: Optional[str]
    ) -> None:
        if not session_id:
            logger.warning("read_session_history action missing session_id")
            return
        events = await asyncio.to_thread(history.read_session_history, session_id)
        # KTD8: this path sends directly over the websocket and bypasses
        # _send_event entirely, so it needs its own redaction pass rather
        # than inheriting _send_event's.
        redacted_events = _redact_secrets(events if events is not None else [], self._secret_values())
        event = {
            "session_id": f"_history:{session_id}",
            "event_id": 0,
            "type": "session_history",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"session_id": session_id, "events": redacted_events},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_get_observe_settings(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """R: "give an ability to choose what clients to use" - reports
        every entrypoint value this build knows about plus which ones are
        currently selected, so the phone can render a set of toggles
        without hardcoding the list on both sides."""
        event = {
            "session_id": "_observe_settings",
            "event_id": 0,
            "type": "observe_settings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "known_entrypoints": list(KNOWN_ENTRYPOINTS),
                "selected_entrypoints": sorted(self.observe_adapter.get_required_entrypoints()),
            },
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_set_observe_entrypoints(
        self, ws: "websockets.WebSocketClientProtocol", entrypoints: Optional[list]
    ) -> None:
        if entrypoints is None:
            logger.warning("set_observe_entrypoints action missing entrypoints")
            return
        # code-review finding: a bare string (or any non-list-of-strings
        # payload) must be rejected outright, not passed to frozenset() -
        # frozenset() happily iterates a str's individual characters
        # instead of raising, which would silently corrupt the configured
        # entrypoint set rather than failing loudly. Reject-and-confirm-back
        # matches this file's own convention elsewhere (e.g.
        # _handle_set_active_account) for an invalid phone-supplied value.
        if not isinstance(entrypoints, list) or not all(isinstance(e, str) for e in entrypoints):
            logger.warning(
                "rejected set_observe_entrypoints %r: expected a list of strings", entrypoints
            )
            await self._handle_get_observe_settings(ws)
            return
        chosen = frozenset(entrypoints)
        self.observe_adapter.set_required_entrypoints(chosen)
        self.config.observe_entrypoints = sorted(chosen)
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_observe_settings(ws)  # confirm back with the new state

    async def _handle_get_auto_approve_settings(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """The auto-approve/AI-judgment policy (companion/auto_approve.py,
        companion/risk_judge.py) applies to *every* session once enabled
        here - both ones the phone starts (via start_session's own
        auto_approve/llm_judge flags, which default to this same value -
        see mobile/screens/SessionListScreen.tsx) and ones discovered from
        a terminal-started `claude` session (ObserveAdapter's hook path).
        This is a single global switch, not per-session, since an observed
        session was never "started" by the phone at all."""
        event = {
            "session_id": "_auto_approve_settings",
            "event_id": 0,
            "type": "auto_approve_settings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "auto_approve": self.observe_adapter.get_auto_approve(),
                "llm_judge": self.observe_adapter.get_llm_judge(),
            },
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_set_auto_approve_settings(
        self, ws: "websockets.WebSocketClientProtocol", auto_approve: Optional[bool], llm_judge: Optional[bool]
    ) -> None:
        if auto_approve is not None:
            self.observe_adapter.set_auto_approve(bool(auto_approve))
            self.config.observe_auto_approve = bool(auto_approve)
        if llm_judge is not None:
            self.observe_adapter.set_llm_judge(bool(llm_judge))
            self.config.observe_llm_judge = bool(llm_judge)
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_auto_approve_settings(ws)  # confirm back with the new state

    async def _handle_get_cli_settings(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """R7/KD5/KTD5: which Claude Code CLI binary/profile this Mac's
        companion invokes for new SDK-owned sessions - a Mac-level setting
        (not per-session, not per-request), so this is the whole state
        rather than anything scoped to a session_id."""
        event = {
            "session_id": "_cli_settings",
            "event_id": 0,
            "type": "cli_settings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "cli_path": self.config.cli_path,
                "cli_env": self.config.cli_env,
            },
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_set_cli_settings(
        self, ws: "websockets.WebSocketClientProtocol", cli_path: Optional[str], cli_env: Optional[dict]
    ) -> None:
        """`None` for either argument leaves that field as it was already
        saved, same convention as _handle_set_auto_approve_settings - the
        phone sends only the field(s) it's actually changing.

        Unlike the other settings this daemon accepts over the wire,
        cli_path/cli_env feed straight into the subprocess spawned for
        every subsequent session (sdk_adapter.py's ClaudeAgentOptions) -
        entirely outside the can_use_tool permission gating everything
        else goes through. A rejected value is dropped silently from this
        call (logged, not applied) rather than raising - the confirm-back
        below still reports the last-accepted state either way, so the
        phone can tell a rejection happened by comparing what it sent
        against what comes back."""
        if cli_path is not None:
            if _is_valid_remote_cli_path(cli_path):
                self.config.cli_path = cli_path
            else:
                logger.warning(
                    "rejected set_cli_settings cli_path %r: not an absolute path to an existing executable file",
                    cli_path,
                )
        if cli_env is not None:
            # code-review finding: cli_env must be dict-shaped (with str
            # keys/values, matching ClaudeAgentOptions.env's own typing)
            # before `dict(cli_env)` ever runs - a list or other non-dict
            # payload can raise ValueError/TypeError there uncaught,
            # dropping this whole action with no confirm-back event at all
            # (unlike every other rejection path in this method, which logs
            # and still falls through to the confirm-back below).
            if not isinstance(cli_env, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in cli_env.items()
            ):
                logger.warning("rejected set_cli_settings cli_env %r: expected a dict of str to str", cli_env)
            else:
                dangerous = _DANGEROUS_CLI_ENV_KEYS.intersection(cli_env)
                if not dangerous:
                    self.config.cli_env = dict(cli_env)
                else:
                    logger.warning("rejected set_cli_settings cli_env: disallowed key(s) %r", sorted(dangerous))
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_cli_settings(ws)  # confirm back with the (possibly-unchanged) state

    async def _handle_get_account_settings(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """R1/R2 (session-account-switch): which identity new SDK-owned
        sessions launch under. Never reports personal_oauth_token's raw
        value (KTD4) - only whether it's configured, so the phone can show
        the "Custom auth token" option as available/unavailable without the
        secret ever riding this response.

        Codex Agent Integration plan (002), U2 (R8): the same event now
        also carries Codex's own, independent codex_active_account/
        codex_personal_configured fields alongside Claude's - one wider
        payload rather than a second event, since both describe the same
        "what does this companion report as its account settings" moment
        and the phone already re-fetches/re-renders this whole event on any
        one change (R10: reads self.config.codex_* only, never
        active_account/personal_oauth_token)."""
        event = {
            "session_id": "_account_settings",
            "event_id": 0,
            "type": "account_settings",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "active_account": self.config.active_account,
                "personal_configured": self.config.personal_oauth_token is not None,
                "codex_active_account": self.config.codex_active_account,
                "codex_personal_configured": self.config.codex_personal_api_key is not None,
            },
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_set_active_account(
        self, ws: "websockets.WebSocketClientProtocol", active_account: Optional[str]
    ) -> None:
        """R2: rejects an unknown value, and rejects switching to
        "personal" before a token exists (the daemon-side backstop for
        what U4's UI already prevents by disabling that option) - same
        reject-outright-but-still-confirm-back convention as
        _handle_set_cli_settings."""
        if active_account is None:
            logger.warning("set_active_account action missing active_account")
            await self._handle_get_account_settings(ws)
            return
        if active_account not in _VALID_ACCOUNTS:
            logger.warning("rejected set_active_account %r: must be one of %r", active_account, sorted(_VALID_ACCOUNTS))
            await self._handle_get_account_settings(ws)
            return
        if active_account == "personal" and self.config.personal_oauth_token is None:
            logger.warning("rejected set_active_account 'personal': no personal_oauth_token configured yet")
            await self._handle_get_account_settings(ws)
            return
        self.config.active_account = active_account
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_account_settings(ws)

    async def _store_personal_account_token(self, token: str) -> None:
        """Shared persistence path for both the manual paste-in
        (_handle_set_personal_account_token) and the automated pty-capture
        (_handle_start_personal_account_setup) flows, so success is
        recorded identically either way (U2's own test scenario)."""
        self.config.personal_oauth_token = token
        await asyncio.to_thread(save_config, self.config_path, self.config)

    async def _handle_set_personal_account_token(
        self, ws: "websockets.WebSocketClientProtocol", token: Optional[str]
    ) -> None:
        """KTD5's manual-fallback entry point."""
        if token is None or not token.strip():
            logger.warning("rejected set_personal_account_token: empty or missing token")
            await self._handle_get_account_settings(ws)
            return
        await self._store_personal_account_token(token.strip())
        await self._handle_get_account_settings(ws)

    async def _handle_start_personal_account_setup(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """KTD5's automated-first attempt. On success, confirms back
        through the same _handle_get_account_settings every other change
        does - the phone infers success from personal_configured flipping
        true, no separate "it worked" signal needed. On failure, emits an
        explicit personal_account_setup_result(available=False) - unlike
        success, there is nothing else that would tell the phone "an
        attempt just failed, show the manual fallback" versus "nothing has
        happened yet"."""
        token = await _run_setup_token_under_pty(self.config.cli_path)
        if token is not None:
            await self._store_personal_account_token(token)
            await self._handle_get_account_settings(ws)
            return
        event = {
            "session_id": "_account_settings",
            "event_id": 0,
            "type": "personal_account_setup_result",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"available": False},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _handle_set_active_codex_account(
        self, ws: "websockets.WebSocketClientProtocol", codex_active_account: Optional[str]
    ) -> None:
        """Codex Agent Integration plan (002), U2 (R9/R10): mirrors
        _handle_set_active_account exactly, one level over - rejects an
        unknown value, and rejects switching to "personal" before a
        codex_personal_api_key exists, same reject-outright-but-still-
        confirm-back convention. Only ever reads/writes
        self.config.codex_active_account - never Claude's own
        active_account (R10)."""
        if codex_active_account is None:
            logger.warning("set_active_codex_account action missing codex_active_account")
            await self._handle_get_account_settings(ws)
            return
        if codex_active_account not in _VALID_CODEX_ACCOUNTS:
            logger.warning(
                "rejected set_active_codex_account %r: must be one of %r",
                codex_active_account,
                sorted(_VALID_CODEX_ACCOUNTS),
            )
            await self._handle_get_account_settings(ws)
            return
        if codex_active_account == "personal" and self.config.codex_personal_api_key is None:
            logger.warning(
                "rejected set_active_codex_account 'personal': no codex_personal_api_key configured yet"
            )
            await self._handle_get_account_settings(ws)
            return
        self.config.codex_active_account = codex_active_account
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_account_settings(ws)

    async def _store_personal_codex_account_token(self, token: str) -> None:
        """Codex's own persistence path, mirroring
        _store_personal_account_token - only ever writes
        self.config.codex_personal_api_key, never Claude's own
        personal_oauth_token (R10). Unlike Claude's pair of callers, there
        is no automated pty-capture flow to share this with: Codex has no
        `claude setup-token`-equivalent CLI command to shell out to (see
        _handle_start_personal_account_setup's own docstring) - a manual
        paste is the only entry point."""
        self.config.codex_personal_api_key = token
        await asyncio.to_thread(save_config, self.config_path, self.config)

    async def _handle_set_personal_codex_account_token(
        self, ws: "websockets.WebSocketClientProtocol", token: Optional[str]
    ) -> None:
        """Codex's own manual-entry point, mirroring
        _handle_set_personal_account_token."""
        if token is None or not token.strip():
            logger.warning("rejected set_personal_codex_account_token: empty or missing token")
            await self._handle_get_account_settings(ws)
            return
        await self._store_personal_codex_account_token(token.strip())
        await self._handle_get_account_settings(ws)

    async def _handle_git_status(self, adapter, session_id: str) -> None:
        """U10 (R16): computes git status for the session's cwd off the
        event loop (subprocess calls block) and injects the result into
        the session's own event stream via `emit_custom` - it then rides
        the same forwarding path (_forward_events) as every other event."""
        cwd = adapter.get_cwd(session_id)
        if cwd is None:
            adapter.emit_custom(session_id, "git_status", is_git_repo=False)
            return
        status = await asyncio.to_thread(git_status.get_status, cwd)
        adapter.emit_custom(session_id, "git_status", **status.to_dict())

    async def _handle_git_diff(self, adapter, session_id: str, path: Optional[str]) -> None:
        if not path:
            logger.warning("git_diff action missing path for session %r", session_id)
            return
        cwd = adapter.get_cwd(session_id)
        if cwd is None:
            adapter.emit_custom(session_id, "git_diff", is_git_repo=False, path=path)
            return
        diff = await asyncio.to_thread(git_status.get_diff, cwd, path)
        adapter.emit_custom(session_id, "git_diff", path=path, **diff.to_dict())

    async def _handle_get_file_content(self, adapter, session_id: str, path: Optional[str]) -> None:
        """Quick-Edit Changed Files plan, U1 (KTD10): unlike _handle_git_diff
        above, a missing/invalid `path` always emits a specific rejection
        event rather than a silent log-and-drop - F3 requires every
        rejection to reach the client with a reason."""
        if not path:
            adapter.emit_custom(session_id, "file_content", path=path, ok=False, content=None, reason="missing_path")
            return
        cwd = adapter.get_cwd(session_id)
        if cwd is None:
            adapter.emit_custom(session_id, "file_content", path=path, ok=False, content=None, reason="no_cwd")
            return
        result = await asyncio.to_thread(git_status.get_file_content, cwd, path)
        adapter.emit_custom(session_id, "file_content", path=path, **result.to_dict())

    async def _handle_write_session_file(
        self,
        adapter,
        session_id: str,
        path: Optional[str],
        previous_content: Optional[str],
        new_content: Optional[str],
    ) -> None:
        """Quick-Edit Changed Files plan, U2 (KTD10): same always-emit-a-
        rejection convention as _handle_get_file_content above. `is None`,
        not falsy - an empty string is a legitimate previous/new content
        (an emptied file), not a missing field."""
        if not path or previous_content is None or new_content is None:
            adapter.emit_custom(session_id, "write_session_file", path=path, ok=False, reason="missing_input")
            return
        cwd = adapter.get_cwd(session_id)
        if cwd is None:
            adapter.emit_custom(session_id, "write_session_file", path=path, ok=False, reason="no_cwd")
            return
        # code-review finding (P1): serializes concurrent dispatch of the
        # same (session_id, path) - see this lock's own docstring in
        # __init__ - so the second of two near-simultaneous saves sees the
        # first one's already-written content in its own read-compare-write
        # check (and correctly reports "conflict") instead of both racing
        # past that check against the same stale content.
        lock = self._write_session_file_locks.setdefault((session_id, path), asyncio.Lock())
        async with lock:
            result = await asyncio.to_thread(
                git_status.write_session_file, cwd, path, previous_content, new_content
            )
        adapter.emit_custom(session_id, "write_session_file", path=path, **result.to_dict())

    async def _handle_session_digest(self, adapter, session_id: str) -> None:
        """AFK Digest plan (U2): on-demand "Catch me up" - same off-loop
        I/O + emit_custom shape as _handle_git_status/_handle_git_diff
        above. No cwd means no transcript to read (mirrors
        _handle_git_status's own early return); a missing transcript or a
        fail-closed digest.generate_digest (no ANTHROPIC_API_KEY, timeout,
        any other failure) both report available=False - never an
        exception out of the dispatch."""
        cwd = adapter.get_cwd(session_id)
        if cwd is None:
            adapter.emit_custom(session_id, "session_digest", available=False, text=None)
            return
        events = await asyncio.to_thread(history.read_session_history, session_id, str(self.observe_adapter.projects_dir))
        if not events:
            adapter.emit_custom(session_id, "session_digest", available=False, text=None)
            return
        # Mirrors sdk_adapter.py's own risk_judge.get_api_client() gating
        # (code-review finding: this was previously never wired, leaving
        # digest.py's direct-API fast path unreachable in production).
        api_client = digest.get_api_client() if self.config.risk_judge_use_api else None
        # fix(review): discovered via real-device testing - without this,
        # digest generation's own one-shot CLI subprocess never got the
        # personal-account OAuth override a real session's connect() call
        # already threads through _effective_cli_env().
        text = await digest.generate_digest(events, api_client=api_client, env=self._effective_cli_env())
        adapter.emit_custom(session_id, "session_digest", available=text is not None, text=text)

    async def _handle_compose_digest(
        self, ws: "websockets.WebSocketClientProtocol", session_summaries: Optional[list]
    ) -> None:
        """Cross-Session Digest plan (010), U1: the phone has already
        collected one per-session digest per active session across every
        online paired Mac (its own compose-then-collect orchestration,
        reusing AFK Digest's get_session_digest action per session) and
        sends the collected texts here for one final merge call. Not
        scoped to a single session_id, so this responds the same way
        _handle_get_account_settings does - a raw event on a sentinel
        session_id, matching the existing "_account_settings"/
        "_observe_settings" convention - rather than through an adapter's
        emit_custom, which requires an already-connected real session."""
        summaries = session_summaries or []
        api_client = digest.get_api_client() if self.config.risk_judge_use_api else None
        text = await digest.compose_cross_session_digest(summaries, api_client=api_client)
        event = {
            "session_id": "_cross_session_digest",
            "event_id": 0,
            "type": "cross_session_digest",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"available": text is not None, "text": text},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

    async def _persist_sdk_session_registry(
        self, session_id: str, *, model: Optional[str] = None, permission_mode: Optional[str] = None
    ) -> None:
        """None (the default) for either argument leaves that one field as
        it was already saved, same convention set_session_auto_approve's
        own auto_approve/llm_judge args used to follow - a mode-only change
        must not clobber a model already on file (and vice versa), since
        session_registry.upsert_session takes a full row, not a partial
        update.

        Permission-mode-picker plan (U3): called by the
        set_session_permission_mode/set_session_model dispatch branches
        above, on success only, purely so a later companion-restart resume
        picks up the live change - this is not what applies either change
        live (that's the direct session.client.set_permission_mode/
        set_model call inside SDKAdapter itself).

        R5/code-review finding: SDKAdapter.set_session_permission_mode/
        set_session_model can still win their own race against a
        concurrently-dispatched end_session (their CLIConnectionError
        catch only covers the client already having been disconnected -
        it says nothing about which of the two _sessions-lookups ran
        first), successfully applying the change and returning True just
        before end_session's disconnect() actually pops the session.
        upsert_session always sets status='active' unconditionally (by
        design, for the ordinary resume path), so without the status check
        below, that legitimately-successful live change's own persistence
        write could silently flip an end_session that raced ahead of it
        (registry already marked 'ended', synchronously, before disconnect
        - see end_session's own dispatch above) back to 'active' - the
        exact silent resurrection R5 exists to prevent, just via a
        different door than _try_resume_sdk_session's own status check."""
        cwd = self.sdk_adapter.get_cwd(session_id)
        if cwd is None:
            return  # shouldn't happen for a live SDK session, but never crash persistence over it
        # The load and save below must be one atomic read-modify-write, not
        # just the save - two concurrently-dispatched live changes for this
        # session each read the pre-change file, so locking only the write
        # would still let the second save silently overwrite the first
        # change's data with stale data.
        async with self._session_registry_lock:
            saved = await asyncio.to_thread(session_registry.get_session, session_id, self.session_registry_path)
            if saved is not None and saved.status == "ended":
                return
            await asyncio.to_thread(
                session_registry.upsert_session,
                session_id,
                cwd=cwd,
                model=model if model is not None else (saved.model if saved is not None else None),
                permission_mode=(
                    permission_mode if permission_mode is not None else (saved.permission_mode if saved is not None else None)
                ),
                path=self.session_registry_path,
            )

    def _adapter_for(self, session_id: str, *, include_observed: bool = True) -> Optional[AdapterProtocol]:
        # U2 (KTD2): iterates the registry rather than naming sdk_adapter/
        # observe_adapter - preserves today's short-circuit order (SDK-owned
        # sessions resolve before observed ones) since self.adapters is
        # constructed in that same order in __init__.
        #
        # `include_observed=False` restricts this to actively-controlled
        # adapters (sdk_adapter/codex_adapter), skipping observe_adapter
        # entirely. The action dispatch below calls it this way *before*
        # attempting an SDK resume: observe_adapter polls the same on-disk
        # transcript directory an SDK-owned session's own transcript lives
        # in, and can independently discover a just-restarted session's
        # file before anything has resumed it - with observe_adapter left
        # in the normal priority list at that point, this method would
        # return it directly (list order only matters when a session_id
        # already appears in *some* adapter's discover_sessions() - it
        # doesn't make the SDK resume path get tried first), permanently
        # routing every action for that session_id to observe_adapter
        # (a no-op there for send_message and friends) instead of ever
        # calling _try_resume_sdk_session.
        adapters = self.adapters if include_observed else [a for a in self.adapters if a is not self.observe_adapter]
        for adapter in adapters:
            if session_id in adapter.discover_sessions():
                return adapter
        return None

    async def _try_resume_sdk_session(self, session_id: str) -> Optional[SDKAdapter]:
        """sdk_adapter's in-memory _sessions is wiped on every companion
        restart (see SDKAdapter.connect's own docstring) - a session_id the
        phone still references from before the restart isn't a phone error,
        it's a real, disk-backed transcript the daemon just doesn't know
        about anymore. Reconnect via the SDK's own `resume` support (same
        session_id, no fork - see connect()'s comment) instead of silently
        dropping whatever action prompted this, so a message sent right
        after a restart is delivered rather than vanishing.

        model/permission_mode come from session_registry.py's saved record
        when there is one (written by start_session) - falling back to the
        transcript's own cwd plus permission_mode=None (SDK default) only
        for a session that predates this being tracked at all, or that
        predates permission_mode existing as a concept (R7/AE3: no mode is
        retroactively assigned just because an old row still has its
        now-unused auto_approve/llm_judge columns set). Returns None (falls
        through to the normal
        "unknown session_id" warning) if no matching transcript exists, the
        registry already marks this session 'ended' (R5 - an ended session
        must not be silently resurrected just because its transcript still
        exists on disk; the explicit end action needs to stay durable), or
        the resume attempt itself fails - e.g. a session id that was never
        real, or one already too old for the CLI to load."""
        cwd = await asyncio.to_thread(
            history.find_transcript_cwd, session_id, str(self.observe_adapter.projects_dir)
        )
        if cwd is None:
            return None
        try:
            async with self._session_registry_lock:
                saved = await asyncio.to_thread(
                    session_registry.get_session, session_id, self.session_registry_path
                )
        except Exception:
            # code-review finding: a registry I/O failure here must fail
            # closed (refuse to resume), not fail open by falling through
            # as if this were simply a never-tracked session - the whole
            # point of this read is R5's "don't resume something that was
            # explicitly ended" check, and permitting a resume we couldn't
            # actually verify defeats that guarantee rather than degrading
            # gracefully.
            logger.exception("session_registry read failed while resuming %r - refusing to resume", session_id)
            return None
        if saved is not None and saved.status == "ended":
            return None
        try:
            await self.sdk_adapter.connect(
                session_id,
                cwd=cwd,
                resume=session_id,
                model=saved.model if saved is not None else None,
                permission_mode=saved.permission_mode if saved is not None else None,
                # Same as the start_session call site above: cli_path/
                # cli_env are Mac-level settings, always read from
                # self.config directly - there's no saved
                # session_registry.py equivalent for either. cli_env goes
                # through _effective_cli_env, same reasoning as
                # start_session.
                cli_path=self.config.cli_path,
                cli_env=self._effective_cli_env(),
            )
        except Exception:
            logger.exception("failed to resume SDK-owned session %r", session_id)
            return None
        resolved_cwd = self.sdk_adapter.get_cwd(session_id) or cwd
        async with self._session_registry_lock:
            await asyncio.to_thread(
                session_registry.upsert_session,
                session_id,
                cwd=resolved_cwd,
                model=saved.model if saved is not None else None,
                permission_mode=saved.permission_mode if saved is not None else None,
                path=self.session_registry_path,
            )
        return self.sdk_adapter

    async def _watch_active_sessions(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """Restarts forwarding for any session not currently being
        forwarded on *this* connection - both directions matter:

        - U4 has no "new session discovered" callback, only the growing
          `discover_sessions()` list, so an observe-only session is only
          ever noticed by polling.
        - An SDK-owned session's forwarding is wired to one specific `ws`
          at `start_session` time. If the relay connection drops and
          `_serve_connection` reconnects on a new `ws`, that session is
          still running (session.events keeps queuing) but nothing is
          re-forwarding it - _serve_connection's cleanup cancels the old
          connection's forwarder tasks and clears `self._forwarding` on
          disconnect specifically so this loop sees every still-active
          session as needing a fresh forwarder here, on the new `ws`.

        A session_id can (rarely, but really) be discoverable by both
        ObserveAdapter and sdk_adapter/codex_adapter at once - ObserveAdapter
        scans the same on-disk transcript directory an SDK-owned session's
        own transcript lives in, and its poll can occasionally notice a
        fresh file before that session_id's own resume/connect() has
        finished registering it. _spawn_forwarder's "already forwarding"
        guard means whichever adapter's tick wins that race would otherwise
        keep the session_id's forwarder slot *permanently* - if
        ObserveAdapter won, the SDK-owned session's own events would never
        be forwarded at all, not even after the resume completes, since the
        guard never re-checks who actually owns a session_id once claimed.
        sdk_adapter/codex_adapter are authoritative for a session_id they
        actively control; an ObserveAdapter forwarder found holding that
        slot is reclaimed and replaced below rather than left to win by
        having gotten there first."""
        while True:
            await self._sleep_or_stop(self.config.observe_scan_interval)
            if self._stop_event.is_set():
                return
            authoritative_ids = set(self.sdk_adapter.discover_sessions()) | set(self.codex_adapter.discover_sessions())
            for adapter in self.adapters:
                for session_id in adapter.discover_sessions():
                    if (
                        adapter is not self.observe_adapter
                        and session_id in authoritative_ids
                        and self._forwarding_adapter.get(session_id) is self.observe_adapter
                    ):
                        await self._reclaim_forwarder(session_id)
                    self._spawn_forwarder(ws, adapter, session_id)

    async def _reclaim_forwarder(self, session_id: str) -> None:
        """Cancels and awaits an ObserveAdapter forwarder wrongly holding a
        session_id that sdk_adapter/codex_adapter now also claims (see
        _watch_active_sessions's own comment), then clears both bookkeeping
        dicts so the immediately-following _spawn_forwarder call below can
        claim it for the authoritative adapter instead."""
        task = self._forwarding.pop(session_id, None)
        self._forwarding_adapter.pop(session_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _spawn_forwarder(self, ws: "websockets.WebSocketClientProtocol", adapter, session_id: str) -> None:
        """The only place a forwarder task is created, so `self._forwarding`
        always reflects reality: every entry is a task bound to `ws`,
        removed either when its session naturally ends (`_forward_events`'s
        `finally`) or when `_serve_connection` tears down the connection
        it was spawned against (see that method's cleanup)."""
        if session_id in self._forwarding:
            return
        self._forwarding[session_id] = asyncio.create_task(self._forward_events(ws, adapter, session_id))
        self._forwarding_adapter[session_id] = adapter

    async def _forward_events(self, ws: "websockets.WebSocketClientProtocol", adapter, session_id: str) -> None:
        try:
            async for event in adapter.subscribe(session_id):
                await self._send_event(ws, event)
                # R5: `session_ended` (SDKAdapter.end()'s own docstring:
                # "interrupt(), disconnect(), and a natural end-of-stream in
                # read_loop can all reach here") is the one funnel point for
                # every way an SDK-owned session actually stops - hooking
                # here durably marks the registry 'ended' whichever path
                # got here, not just an explicit end_session action.
                # Code-review fix (stale comment): this used to claim a
                # crash (read_loop's except-branch) never reaches end() and
                # so leaves the registry row 'active' - no longer true,
                # read_loop's own except-branch calls self.end("crashed"),
                # which reaches this same "session_ended" funnel like every
                # other end path. Observed sessions have no registry row at
                # all (KTD4) - that's the one real "not tracked here" case.
                if adapter is self.sdk_adapter and event.type == "session_ended":
                    async with self._session_registry_lock:
                        await asyncio.to_thread(
                            session_registry.mark_session_ended, session_id, self.session_registry_path
                        )
                # Agent Limit Handoff plan, U2 (KTD3): this is the one
                # point every adapter-emitted event already crosses on its
                # way to the phone, so it's also the one place to watch for
                # a REAL "rate_limit" reading (never "rate_limit_estimate" -
                # a calibrated guess alone must never raise an offer, R2's
                # own key decision) and derive/escalate/clear this key's
                # offer from it. `agent_for_adapter` returns None for any
                # adapter other than sdk_adapter/codex_adapter (ObserveAdapter
                # never emits "rate_limit" at all - see events.py's own
                # comment on that event type), so this is unreachable there.
                if event.type == "rate_limit":
                    agent = self.agent_limits.agent_for_adapter(adapter)
                    if agent is not None:
                        self.agent_limits.handle_rate_limit_reading(agent, event.data)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("event forwarding failed for session %r", session_id)
        finally:
            self._forwarding.pop(session_id, None)
            self._forwarding_adapter.pop(session_id, None)

    def _effective_cli_env(self) -> dict[str, str]:
        """KTD1/KTD3: the env a new SDK-owned session actually launches
        with. "vscode" (the default) or "personal" with no token configured
        yet both pass self.config.cli_env through completely unchanged -
        today's behavior, byte for byte. "personal" with a configured
        token merges CLAUDE_CODE_OAUTH_TOKEN in.

        claude_agent_sdk's own subprocess transport merges this
        dict ON TOP OF this daemon process's own inherited os.environ
        (`{**inherited_env, **options.env}` - see
        claude_agent_sdk._internal.transport.subprocess_cli), not the
        other way around - so removing a precedence-key from cli_env only
        helps when that key was never ambient in the daemon's own
        environment to begin with. When it IS ambient (e.g. this
        companion's own ANTHROPIC_API_KEY, set for risk_judge_use_api -
        see CompanionConfig's docstring), only an explicit override in the
        returned dict can take precedence over it; an empty string is the
        best currently-verifiable neutralization available. Whether the
        `claude` CLI itself treats an empty value as "unset" rather than
        "set to nothing" is not confirmed outside a real machine - a
        residual uncertainty of the same shape as KTD1/KTD5's, not
        something this method can close further from here."""
        if self.config.active_account != "personal" or self.config.personal_oauth_token is None:
            return self.config.cli_env
        env = dict(self.config.cli_env)
        for key in _PRECEDENCE_ENV_KEYS_TO_NEUTRALIZE_FOR_PERSONAL:
            env.pop(key, None)
            if key in os.environ:
                env[key] = ""
        env["CLAUDE_CODE_OAUTH_TOKEN"] = self.config.personal_oauth_token
        return env

    def _secret_values(self) -> tuple[str, ...]:
        """KTD8: current secret values to redact from forwarded content -
        filters out personal_oauth_token when unset (None) rather than
        passing an empty-string redaction target. codex_personal_api_key
        (U2) gets the same treatment - a Codex session's own forwarded
        tool output is just as capable of echoing a configured secret back
        verbatim as an SDK-owned Claude session is."""
        return tuple(
            value
            for value in (
                self.config.device_token,
                self.config.personal_oauth_token,
                self.config.codex_personal_api_key,
            )
            if value
        )

    async def _send_event(self, ws: "websockets.WebSocketClientProtocol", event: Event) -> None:
        wire = event.to_dict()
        wire["data"] = _redact_secrets(wire["data"], self._secret_values())
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": wire}))

    async def _sleep_or_stop(self, delay: float) -> None:
        """Sleep for `delay` seconds, but wake immediately if `stop()` is called."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


def main(config_path: Optional[str] = None, status_socket_path: Optional[str] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    resolved_config_path = config_path or DEFAULT_SESSION_REGISTRY_PATH
    config = load_config(resolved_config_path)
    daemon = CompanionDaemon(config, config_path=resolved_config_path, status_socket_path=status_socket_path)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
