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
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import websockets

from . import git_status, history, projects
from .adapters.events import Event
from .adapters.observe_adapter import KNOWN_ENTRYPOINTS, ObserveAdapter
from .adapters.sdk_adapter import SDKAdapter
from .config import DEFAULT_CONFIG_PATH, CompanionConfig, load_config, save_config
from .session_settings import SessionSettings, load_session_settings, save_session_settings

logger = logging.getLogger(__name__)


class CompanionDaemon:
    """Owns one persistent (reconnecting) connection to the relay, plus the
    SDK-owned and observe-only adapters that connection's actions/events
    flow through.

    `sdk_adapter`/`observe_adapter` are injectable so tests can substitute
    fakes (mirroring each adapter's own `client_factory` injection point)
    without spawning a real `claude` subprocess or touching the real
    `~/.claude/projects` / hooks socket paths.
    """

    def __init__(
        self,
        config: CompanionConfig,
        *,
        sdk_adapter: Optional[SDKAdapter] = None,
        observe_adapter: Optional[ObserveAdapter] = None,
        recents_path: Optional[str] = None,
        config_path: Optional[str] = None,
        session_settings_path: Optional[str] = None,
    ):
        self.config = config
        self.config_path = config_path or DEFAULT_CONFIG_PATH
        self.sdk_adapter = sdk_adapter or SDKAdapter()
        self.observe_adapter = observe_adapter or ObserveAdapter(
            required_entrypoints=frozenset(config.observe_entrypoints),
            auto_approve=config.observe_auto_approve,
            llm_judge=config.observe_llm_judge,
        )
        # Injectable for the same reason as the adapters above: tests must
        # never write into the developer's real recent-projects file
        # (projects.DEFAULT_RECENTS_PATH) just by exercising start_session.
        self.recents_path = recents_path
        # Same injectability reason, for session_settings.py's own default
        # path - see _try_resume_sdk_session for why this exists at all.
        self.session_settings_path = session_settings_path
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

    def stop(self) -> None:
        """Request a graceful stop. `run()` returns once the current
        connect/heartbeat cycle notices, typically within one sleep tick."""
        self._stop_event.set()

    async def run(self) -> None:
        await self.observe_adapter.start()
        try:
            while not self._stop_event.is_set():
                self.state = "connecting"
                self.connect_attempts += 1
                try:
                    await self._connect_and_serve()
                except (OSError, websockets.exceptions.WebSocketException) as exc:
                    logger.info("companion connection lost/failed: %s", exc)
                if self._stop_event.is_set():
                    break
                self.state = "reconnecting"
                await self._sleep_or_stop(self._backoff)
                self._backoff = min(self._backoff * 2, self.config.reconnect_max_delay)
        finally:
            await self.observe_adapter.stop()
        self.state = "stopped"

    async def _connect_and_serve(self) -> None:
        async with websockets.connect(self.config.relay_url) as ws:
            await ws.send(json.dumps({"token": self.config.device_token, "type": "hello"}))
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
            logger.exception("action %r failed: %r", action.get("kind"), action)

    async def _dispatch_action(self, ws: "websockets.WebSocketClientProtocol", action: dict[str, Any]) -> None:
        kind = action.get("kind")

        if kind == "start_session":
            session_id = str(uuid4())
            try:
                await self.sdk_adapter.connect(
                    session_id,
                    cwd=action.get("cwd"),
                    model=action.get("model"),
                    auto_approve=bool(action.get("auto_approve", False)),
                    llm_judge=bool(action.get("llm_judge", False)),
                    # Unlike model/auto_approve/llm_judge above, cli_path/
                    # cli_env (KTD5) are a Mac-level setting with no
                    # phone-side per-request equivalent - always sourced
                    # from this companion's own config, never the action
                    # payload.
                    cli_path=self.config.cli_path,
                    cli_env=self.config.cli_env,
                )
            except Exception:
                logger.exception("start_session failed for action %r", action)
                return
            self._spawn_forwarder(ws, self.sdk_adapter, session_id)
            resolved_cwd = self.sdk_adapter.get_cwd(session_id)
            if resolved_cwd:
                await asyncio.to_thread(projects.record_recent, resolved_cwd, self.recents_path)
                await asyncio.to_thread(
                    save_session_settings,
                    session_id,
                    SessionSettings(
                        cwd=resolved_cwd,
                        model=action.get("model"),
                        auto_approve=bool(action.get("auto_approve", False)),
                        llm_judge=bool(action.get("llm_judge", False)),
                    ),
                    self.session_settings_path,
                )
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

        session_id = action.get("session_id")
        if not session_id:
            logger.warning("action missing session_id: %r", action)
            return

        adapter = self._adapter_for(session_id)
        if adapter is None:
            adapter = await self._try_resume_sdk_session(session_id)
        if adapter is None:
            logger.warning("action for unknown session_id=%r: %r", session_id, action)
            return

        try:
            if kind == "send_message":
                await adapter.send_message(session_id, action.get("text", ""))
            elif kind == "interrupt":
                await adapter.interrupt(session_id)
            elif kind == "compact":
                await adapter.compact(session_id)
            elif kind == "respond_to_permission":
                await adapter.respond_to_permission(
                    session_id,
                    action.get("request_id"),
                    action.get("decision"),
                    message=action.get("message", ""),
                )
            elif kind == "git_status":
                await self._handle_git_status(adapter, session_id)
            elif kind == "git_diff":
                await self._handle_git_diff(adapter, session_id, action.get("path"))
            elif kind == "set_session_auto_approve":
                adapter.set_session_auto_approve(
                    session_id, action.get("auto_approve"), action.get("llm_judge")
                )
                if adapter is self.sdk_adapter:
                    # Keep the saved record in sync with this live override,
                    # so a later restart-triggered resume
                    # (_try_resume_sdk_session) picks it up instead of
                    # whatever was saved at start_session time. Observed
                    # sessions have no equivalent - they're never resumed
                    # this way, since ObserveAdapter has no client to
                    # reconnect.
                    await self._persist_sdk_session_settings(
                        session_id, action.get("auto_approve"), action.get("llm_judge")
                    )
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
        sessions = [
            {
                "session_id": sid,
                "cwd": self.sdk_adapter.get_cwd(sid),
                "mode": "sdk_owned",
                "active": bool(self.sdk_adapter.is_active(sid)),
            }
            for sid in self.sdk_adapter.discover_sessions()
        ] + [
            {
                "session_id": sid,
                "cwd": self.observe_adapter.get_cwd(sid),
                "mode": "observe_only",
                "active": bool(self.observe_adapter.is_active(sid)),
            }
            for sid in self.observe_adapter.discover_sessions()
        ]
        event = {
            "session_id": "_active_sessions",
            "event_id": 0,
            "type": "active_sessions",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"sessions": sessions},
        }
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event}))

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
        event = {
            "session_id": f"_history:{session_id}",
            "event_id": 0,
            "type": "session_history",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"session_id": session_id, "events": events if events is not None else []},
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
        phone sends only the field(s) it's actually changing."""
        if cli_path is not None:
            self.config.cli_path = cli_path
        if cli_env is not None:
            self.config.cli_env = dict(cli_env)
        await asyncio.to_thread(save_config, self.config_path, self.config)
        await self._handle_get_cli_settings(ws)  # confirm back with the new state

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

    async def _persist_sdk_session_settings(
        self, session_id: str, auto_approve: Optional[bool], llm_judge: Optional[bool]
    ) -> None:
        """None for either argument leaves that field as it was already
        saved, same convention as set_session_auto_approve itself."""
        cwd = self.sdk_adapter.get_cwd(session_id)
        if cwd is None:
            return  # shouldn't happen for a live SDK session, but never crash persistence over it
        saved = await asyncio.to_thread(load_session_settings, session_id, self.session_settings_path)
        await asyncio.to_thread(
            save_session_settings,
            session_id,
            SessionSettings(
                cwd=cwd,
                model=saved.model if saved is not None else None,
                auto_approve=auto_approve if auto_approve is not None else (saved.auto_approve if saved is not None else False),
                llm_judge=llm_judge if llm_judge is not None else (saved.llm_judge if saved is not None else False),
            ),
            self.session_settings_path,
        )

    def _adapter_for(self, session_id: str):
        if session_id in self.sdk_adapter.discover_sessions():
            return self.sdk_adapter
        if session_id in self.observe_adapter.discover_sessions():
            return self.observe_adapter
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

        model/auto_approve/llm_judge come from session_settings.py's saved
        record when there is one (written by start_session and
        set_session_auto_approve) - falling back to the transcript's own
        cwd plus opt-in-off defaults only for a session that predates this
        being tracked at all. Returns None (falls through to the normal
        "unknown session_id" warning) if no matching transcript exists, or
        the resume attempt itself fails - e.g. a session id that was never
        real, or one already too old for the CLI to load."""
        cwd = await asyncio.to_thread(
            history.find_transcript_cwd, session_id, str(self.observe_adapter.projects_dir)
        )
        if cwd is None:
            return None
        saved = await asyncio.to_thread(load_session_settings, session_id, self.session_settings_path)
        try:
            await self.sdk_adapter.connect(
                session_id,
                cwd=cwd,
                resume=session_id,
                model=saved.model if saved is not None else None,
                auto_approve=saved.auto_approve if saved is not None else False,
                llm_judge=saved.llm_judge if saved is not None else False,
                # Same as the start_session call site above: cli_path/
                # cli_env are a Mac-level setting, always read from
                # self.config directly - there's no saved SessionSettings
                # equivalent for it.
                cli_path=self.config.cli_path,
                cli_env=self.config.cli_env,
            )
        except Exception:
            logger.exception("failed to resume SDK-owned session %r", session_id)
            return None
        resolved_cwd = self.sdk_adapter.get_cwd(session_id) or cwd
        await asyncio.to_thread(
            save_session_settings,
            session_id,
            SessionSettings(
                cwd=resolved_cwd,
                model=saved.model if saved is not None else None,
                auto_approve=saved.auto_approve if saved is not None else False,
                llm_judge=saved.llm_judge if saved is not None else False,
            ),
            self.session_settings_path,
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
          session as needing a fresh forwarder here, on the new `ws`."""
        while True:
            await self._sleep_or_stop(self.config.observe_scan_interval)
            if self._stop_event.is_set():
                return
            for adapter in (self.sdk_adapter, self.observe_adapter):
                for session_id in adapter.discover_sessions():
                    self._spawn_forwarder(ws, adapter, session_id)

    def _spawn_forwarder(self, ws: "websockets.WebSocketClientProtocol", adapter, session_id: str) -> None:
        """The only place a forwarder task is created, so `self._forwarding`
        always reflects reality: every entry is a task bound to `ws`,
        removed either when its session naturally ends (`_forward_events`'s
        `finally`) or when `_serve_connection` tears down the connection
        it was spawned against (see that method's cleanup)."""
        if session_id in self._forwarding:
            return
        self._forwarding[session_id] = asyncio.create_task(self._forward_events(ws, adapter, session_id))

    async def _forward_events(self, ws: "websockets.WebSocketClientProtocol", adapter, session_id: str) -> None:
        try:
            async for event in adapter.subscribe(session_id):
                await self._send_event(ws, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("event forwarding failed for session %r", session_id)
        finally:
            self._forwarding.pop(session_id, None)

    async def _send_event(self, ws: "websockets.WebSocketClientProtocol", event: Event) -> None:
        await ws.send(json.dumps({"token": self.config.device_token, "type": "event", "event": event.to_dict()}))

    async def _sleep_or_stop(self, delay: float) -> None:
        """Sleep for `delay` seconds, but wake immediately if `stop()` is called."""
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


def main(config_path: Optional[str] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    resolved_config_path = config_path or DEFAULT_CONFIG_PATH
    config = load_config(resolved_config_path)
    daemon = CompanionDaemon(config, config_path=resolved_config_path)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
