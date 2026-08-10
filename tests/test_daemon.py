"""Tests for companion/daemon.py.

These run the real relay app (U1) as a live uvicorn server on localhost and
connect the real CompanionDaemon over a real websocket - no mocks for the
connection/auth chain, per the plan's own integration test scenarios.
"""
from __future__ import annotations

import asyncio
import socket

import pytest
import pytest_asyncio
import uvicorn

from companion.config import CompanionConfig
from companion.daemon import CompanionDaemon
from relay import auth
from relay.app import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _start_server(app, port: int) -> uvicorn.Server:
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server


async def _stop_server(server: uvicorn.Server) -> None:
    server.should_exit = True
    await server.task


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(_poll(), timeout=timeout)


@pytest_asyncio.fixture
async def relay(tmp_path):
    db_path = str(tmp_path / "relay.db")
    app = create_app(db_path)
    port = _free_port()
    server = await _start_server(app, port)
    yield app, port, db_path
    if not server.should_exit:
        await _stop_server(server)


@pytest_asyncio.fixture
async def companion_token(relay):
    app, _port, _db_path = relay
    _, token = auth.bootstrap_companion_device(app.state.db)
    return token


def _fast_config(port: int, token: str) -> CompanionConfig:
    return CompanionConfig(
        relay_url=f"ws://127.0.0.1:{port}/ws/companion",
        device_token=token,
        heartbeat_interval=0.05,
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.05,
    )


async def _stop_daemon(daemon: CompanionDaemon, task: asyncio.Task) -> None:
    daemon.stop()
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_daemon_connects_authenticates_and_sends_heartbeats(relay, companion_token):
    _app, port, _db_path = relay
    daemon = CompanionDaemon(_fast_config(port, companion_token))
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.state == "connected")
    await _wait_until(lambda: daemon.heartbeats_sent >= 2)

    await _stop_daemon(daemon, task)
    assert daemon.state == "stopped"


@pytest.mark.asyncio
async def test_daemon_retries_with_backoff_when_relay_unreachable():
    unreachable_port = _free_port()  # nothing listening on it
    config = _fast_config(unreachable_port, token="does-not-matter-yet")
    daemon = CompanionDaemon(config)
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.connect_attempts >= 3)
    assert not task.done()  # retried without crashing
    assert daemon.state in ("connecting", "reconnecting")

    await _stop_daemon(daemon, task)


@pytest.mark.asyncio
async def test_daemon_reconnects_after_relay_restart_without_losing_identity(tmp_path):
    # Owns its own relay server lifecycle (rather than the `relay` fixture)
    # so it can stop and restart it mid-test to simulate a real drop.
    db_path = str(tmp_path / "relay.db")
    app = create_app(db_path)
    _, token = auth.bootstrap_companion_device(app.state.db)
    port = _free_port()

    server = await _start_server(app, port)
    daemon = CompanionDaemon(_fast_config(port, token))
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.state == "connected")
    attempts_before = daemon.connect_attempts

    # Mid-session drop: the relay process goes away entirely.
    await _stop_server(server)
    await _wait_until(lambda: daemon.state != "connected")

    # Relay comes back on the same port, same db -> same device/token.
    app2 = create_app(db_path)
    server2 = await _start_server(app2, port)
    try:
        await _wait_until(
            lambda: daemon.state == "connected" and daemon.connect_attempts > attempts_before,
            timeout=5,
        )
        assert daemon.config.device_token == token  # identity/config preserved
    finally:
        await _stop_daemon(daemon, task)
        await _stop_server(server2)
