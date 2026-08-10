"""Tests for companion/daemon.py.

These run the real relay app (U1) as a live uvicorn server on localhost and
connect the real CompanionDaemon over a real websocket - no mocks for the
connection/auth chain, per the plan's own integration test scenarios.

Every daemon under test gets an explicit tmp_path-scoped ObserveAdapter
(never the real default `~/.claude/projects` / hooks socket paths) and, for
action-routing tests, a fake-backed SDKAdapter (see test_sdk_adapter.py's
FakeSDKClient) so nothing here spawns a real `claude` subprocess or touches
the developer's actual Claude Code state.
"""
from __future__ import annotations

import asyncio
import json
import socket
import uuid

import pytest
import pytest_asyncio
import uvicorn
import websockets

from companion.adapters.observe_adapter import ObserveAdapter
from companion.adapters.sdk_adapter import SDKAdapter
from companion.config import CompanionConfig
from companion.daemon import CompanionDaemon
from companion.tests.test_sdk_adapter import FakeSDKClient
from relay import auth
from relay.app import create_app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _short_socket_path() -> str:
    # AF_UNIX paths are capped at ~104 bytes on macOS/BSD - pytest's
    # tmp_path is nested too deep to use directly (see test_observe_adapter.py).
    return f"/tmp/rc-daemon-{uuid.uuid4().hex[:8]}.sock"


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


@pytest_asyncio.fixture
async def phone_token(relay):
    app, _port, _db_path = relay
    _, token = auth.create_device(app.state.db, "phone")
    return token


def _fast_config(port: int, token: str) -> CompanionConfig:
    return CompanionConfig(
        relay_url=f"ws://127.0.0.1:{port}/ws/companion",
        device_token=token,
        heartbeat_interval=0.05,
        reconnect_initial_delay=0.01,
        reconnect_max_delay=0.05,
        observe_scan_interval=0.02,
    )


def _test_observe_adapter(tmp_path) -> ObserveAdapter:
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir(exist_ok=True)
    return ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=0.02,
        tail_poll_interval=0.02,
    )


def _fake_sdk_adapter() -> tuple[SDKAdapter, dict[str, FakeSDKClient]]:
    clients: dict[str, FakeSDKClient] = {}

    def factory(options):
        client = FakeSDKClient(options)
        clients[len(clients)] = client
        return client

    return SDKAdapter(client_factory=factory), clients


async def _stop_daemon(daemon: CompanionDaemon, task: asyncio.Task) -> None:
    daemon.stop()
    await asyncio.wait_for(task, timeout=2)


@pytest.mark.asyncio
async def test_daemon_connects_authenticates_and_sends_heartbeats(relay, companion_token, tmp_path):
    _app, port, _db_path = relay
    daemon = CompanionDaemon(_fast_config(port, companion_token), observe_adapter=_test_observe_adapter(tmp_path))
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.state == "connected")
    await _wait_until(lambda: daemon.heartbeats_sent >= 2)

    await _stop_daemon(daemon, task)
    assert daemon.state == "stopped"


@pytest.mark.asyncio
async def test_daemon_retries_with_backoff_when_relay_unreachable(tmp_path):
    unreachable_port = _free_port()  # nothing listening on it
    config = _fast_config(unreachable_port, token="does-not-matter-yet")
    daemon = CompanionDaemon(config, observe_adapter=_test_observe_adapter(tmp_path))
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
    daemon = CompanionDaemon(_fast_config(port, token), observe_adapter=_test_observe_adapter(tmp_path))
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


# --- Action routing + event forwarding (daemon <-> SDK adapter <-> phone) --


class _FakePhone:
    """A second real WebSocket connection to /ws/phone, standing in for
    the mobile client - actions are sent phone-shaped
    (`{"token", "type": "action", "action": {...}}`), and forwarded
    companion events arrive the same way U7's dashboard will receive them.
    """

    def __init__(self, ws, token: str):
        self._ws = ws
        self._token = token

    @classmethod
    async def connect(cls, port: int, token: str) -> "_FakePhone":
        ws = await websockets.connect(f"ws://127.0.0.1:{port}/ws/phone")
        await ws.send(json.dumps({"token": token}))
        reply = json.loads(await ws.recv())
        assert reply["type"] == "ack"
        return cls(ws, token)

    async def send_action(self, action: dict) -> None:
        await self._ws.send(json.dumps({"token": self._token, "type": "action", "action": action}))

    async def next_event(self, timeout: float = 2.0) -> dict:
        """Read messages until an `event` arrives, skipping `ack`s for the
        action this phone itself just sent."""

        async def _read():
            while True:
                message = json.loads(await self._ws.recv())
                if message.get("type") == "event":
                    return message["event"]

        return await asyncio.wait_for(_read(), timeout=timeout)

    async def close(self) -> None:
        await self._ws.close()


@pytest_asyncio.fixture
async def running_daemon(relay, companion_token, tmp_path):
    _app, port, _db_path = relay
    sdk_adapter, fake_clients = _fake_sdk_adapter()
    daemon = CompanionDaemon(
        _fast_config(port, companion_token),
        sdk_adapter=sdk_adapter,
        observe_adapter=_test_observe_adapter(tmp_path),
    )
    task = asyncio.create_task(daemon.run())
    await _wait_until(lambda: daemon.state == "connected")
    yield daemon, port, fake_clients
    await _stop_daemon(daemon, task)


@pytest.mark.asyncio
async def test_start_session_action_creates_sdk_session_and_forwards_session_started(
    running_daemon, phone_token
):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})

        event = await phone.next_event()
        assert event["type"] == "session_started"
        await _wait_until(lambda: len(daemon.sdk_adapter.discover_sessions()) == 1)
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_send_message_action_routes_to_the_owning_sdk_session(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        session_id = started["session_id"]

        await phone.send_action({"kind": "send_message", "session_id": session_id, "text": "hello claude"})

        event = await phone.next_event()
        assert event["type"] == "user_message"
        assert event["data"]["text"] == "hello claude"
        assert event["session_id"] == session_id
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_action_for_unknown_session_is_ignored_without_crashing(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "send_message", "session_id": "no-such-session", "text": "hi"})
        # No crash, no event - just confirm the daemon connection survives
        # (still answers a subsequent heartbeat-triggered ack cycle) rather
        # than asserting a negative on timing.
        await asyncio.sleep(0.1)
        assert daemon.state == "connected"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_observe_session_is_auto_discovered_and_forwarded(running_daemon, phone_token, tmp_path):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        projects_dir = daemon.observe_adapter.projects_dir
        project_dir = projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "session-xyz.jsonl").touch()

        event = await phone.next_event(timeout=3.0)
        assert event["type"] == "session_started"
        assert event["session_id"] == "session-xyz"
    finally:
        await phone.close()
