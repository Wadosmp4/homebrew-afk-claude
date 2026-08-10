"""Companion daemon: persistent, reconnecting, authenticated connection to
the relay's /ws/companion gateway (see relay/app.py's WebSocket auth
contract), with periodic heartbeats.

State machine (`CompanionDaemon.state`):

    connecting -> connected -> reconnecting -> connecting -> ... -> stopped

A failed or dropped connection never raises out of `run()` - it always
falls back to `reconnecting` with exponential backoff (reset to
`reconnect_initial_delay` on every successful connect), per KTD... no KTD
number for this behavior specifically, but it's required by R1 (persistent
connection with automatic reconnect and heartbeats) and the System-Wide
Impact note that a companion restart must re-arm connectivity rather than
require manual intervention.

Runs as a macOS `launchd` user agent per KTD6 - see companion/launchd/ and
companion/README.md for install instructions. `main()` below is the
entrypoint the plist invokes (`python3 -m companion.daemon`).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import websockets

from .config import CompanionConfig, load_config

logger = logging.getLogger(__name__)


class CompanionDaemon:
    """Owns one persistent (reconnecting) connection to the relay.

    `state`, `connect_attempts`, and `heartbeats_sent` are plain public
    attributes rather than an event-callback API - this is the only
    consumer in-process (tests, and later U3/U4 adapters that will hang off
    the same daemon instance), so polling them is simpler than a pub/sub
    layer this unit doesn't need yet.
    """

    def __init__(self, config: CompanionConfig):
        self.config = config
        self.state = "connecting"
        self.connect_attempts = 0
        self.heartbeats_sent = 0
        self._stop_event = asyncio.Event()
        self._backoff = config.reconnect_initial_delay

    def stop(self) -> None:
        """Request a graceful stop. `run()` returns once the current
        connect/heartbeat cycle notices, typically within one sleep tick."""
        self._stop_event.set()

    async def run(self) -> None:
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

            while not self._stop_event.is_set():
                await self._sleep_or_stop(self.config.heartbeat_interval)
                if self._stop_event.is_set():
                    return
                await ws.send(
                    json.dumps(
                        {
                            "token": self.config.device_token,
                            "type": "heartbeat",
                            "seq": self.heartbeats_sent,
                        }
                    )
                )
                await ws.recv()  # ack
                self.heartbeats_sent += 1

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
