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
    the relay as `{"type": "event", "event": {...}}` - for an SDK-owned
    session, forwarding starts right after `start_session`; for an
    observe-only session, `_watch_observe_sessions` notices it the same
    way U4 itself notices it (polling `discover_sessions()`, since U4 has
    no push-style "new session" callback).

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
from typing import Any, Optional
from uuid import uuid4

import websockets

from .adapters.events import Event
from .adapters.observe_adapter import ObserveAdapter
from .adapters.sdk_adapter import SDKAdapter
from .config import CompanionConfig, load_config

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
    ):
        self.config = config
        self.sdk_adapter = sdk_adapter or SDKAdapter()
        self.observe_adapter = observe_adapter or ObserveAdapter()
        self.state = "connecting"
        self.connect_attempts = 0
        self.heartbeats_sent = 0
        self._stop_event = asyncio.Event()
        self._backoff = config.reconnect_initial_delay
        self._forwarding: set[str] = set()

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
            asyncio.create_task(self._watch_observe_sessions(ws)),
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
        kind = action.get("kind")

        if kind == "start_session":
            session_id = str(uuid4())
            try:
                await self.sdk_adapter.connect(session_id, cwd=action.get("cwd"))
            except Exception:
                logger.exception("start_session failed for action %r", action)
                return
            asyncio.create_task(self._forward_events(ws, self.sdk_adapter, session_id))
            return

        session_id = action.get("session_id")
        if not session_id:
            logger.warning("action missing session_id: %r", action)
            return

        adapter = self._adapter_for(session_id)
        if adapter is None:
            logger.warning("action for unknown session_id=%r: %r", session_id, action)
            return

        try:
            if kind == "send_message":
                await adapter.send_message(session_id, action.get("text", ""))
            elif kind == "interrupt":
                await adapter.interrupt(session_id)
            elif kind == "respond_to_permission":
                await adapter.respond_to_permission(
                    session_id,
                    action.get("request_id"),
                    action.get("decision"),
                    message=action.get("message", ""),
                )
            else:
                logger.warning("unknown action kind: %r", kind)
        except Exception:
            logger.exception("action %r failed for session %r", kind, session_id)

    def _adapter_for(self, session_id: str):
        if session_id in self.sdk_adapter.discover_sessions():
            return self.sdk_adapter
        if session_id in self.observe_adapter.discover_sessions():
            return self.observe_adapter
        return None

    async def _watch_observe_sessions(self, ws: "websockets.WebSocketClientProtocol") -> None:
        """U4 has no "new session discovered" callback, only the growing
        `discover_sessions()` list - poll it and start forwarding anything
        this connection isn't already forwarding."""
        while True:
            await self._sleep_or_stop(self.config.observe_scan_interval)
            if self._stop_event.is_set():
                return
            for session_id in self.observe_adapter.discover_sessions():
                if session_id not in self._forwarding:
                    asyncio.create_task(self._forward_events(ws, self.observe_adapter, session_id))

    async def _forward_events(self, ws: "websockets.WebSocketClientProtocol", adapter, session_id: str) -> None:
        if session_id in self._forwarding:
            return
        self._forwarding.add(session_id)
        try:
            async for event in adapter.subscribe(session_id):
                await self._send_event(ws, event)
        except Exception:
            logger.exception("event forwarding failed for session %r", session_id)
        finally:
            self._forwarding.discard(session_id)

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
    config = load_config(config_path) if config_path else load_config()
    daemon = CompanionDaemon(config)
    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        daemon.stop()


if __name__ == "__main__":
    main()
