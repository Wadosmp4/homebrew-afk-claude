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
import os
import socket
import subprocess
import sys
import uuid

import pytest
import pytest_asyncio
import uvicorn
import websockets

from companion.adapters.observe_adapter import KNOWN_ENTRYPOINTS, ObserveAdapter
from companion.adapters.sdk_adapter import SDKAdapter
from companion.config import CompanionConfig, load_config
from companion.daemon import CompanionDaemon
from companion.session_settings import load_session_settings
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
async def relay(tmp_path, pg_url):
    db_path = pg_url
    app = create_app(db_path)
    port = _free_port()
    server = await _start_server(app, port)
    yield app, port, db_path
    if not server.should_exit:
        await _stop_server(server)


@pytest_asyncio.fixture
async def companion_token(relay):
    app, _port, _db_path = relay
    device_id, token = auth.bootstrap_companion_device(app.state.db)
    # U2 (R8/KD6): every phone-sent action must now carry a device_id
    # naming its target companion. Every test in this file predates
    # multi-Mac support and only ever connects one companion, so rather
    # than threading device_id through every one of _FakePhone's ~40 call
    # sites, _FakePhone.send_action auto-fills it from this one companion
    # whenever a test's action dict doesn't already set one - scoped by
    # this fixture's own teardown, so it never leaks between tests.
    _FakePhone.default_device_id = device_id
    yield token
    _FakePhone.default_device_id = None


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
    daemon = CompanionDaemon(
        _fast_config(port, companion_token),
        observe_adapter=_test_observe_adapter(tmp_path),
        recents_path=str(tmp_path / "recent_projects.json"),
        config_path=str(tmp_path / "companion_config.json"),
        session_settings_path=str(tmp_path / "session_settings.json"),
    )
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.state == "connected")
    await _wait_until(lambda: daemon.heartbeats_sent >= 2)

    await _stop_daemon(daemon, task)
    assert daemon.state == "stopped"


@pytest.mark.asyncio
async def test_daemon_retries_with_backoff_when_relay_unreachable(tmp_path):
    unreachable_port = _free_port()  # nothing listening on it
    config = _fast_config(unreachable_port, token="does-not-matter-yet")
    daemon = CompanionDaemon(
        config,
        observe_adapter=_test_observe_adapter(tmp_path),
        recents_path=str(tmp_path / "recent_projects.json"),
        config_path=str(tmp_path / "companion_config.json"),
        session_settings_path=str(tmp_path / "session_settings.json"),
    )
    task = asyncio.create_task(daemon.run())

    await _wait_until(lambda: daemon.connect_attempts >= 3)
    assert not task.done()  # retried without crashing
    assert daemon.state in ("connecting", "reconnecting")

    await _stop_daemon(daemon, task)


@pytest.mark.asyncio
async def test_daemon_reconnects_after_relay_restart_without_losing_identity(tmp_path, pg_url):
    # Owns its own relay server lifecycle (rather than the `relay` fixture)
    # so it can stop and restart it mid-test to simulate a real drop.
    db_path = pg_url
    app = create_app(db_path)
    _, token = auth.bootstrap_companion_device(app.state.db)
    port = _free_port()

    server = await _start_server(app, port)
    daemon = CompanionDaemon(
        _fast_config(port, token),
        observe_adapter=_test_observe_adapter(tmp_path),
        recents_path=str(tmp_path / "recent_projects.json"),
        config_path=str(tmp_path / "companion_config.json"),
        session_settings_path=str(tmp_path / "session_settings.json"),
    )
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


@pytest.mark.asyncio
async def test_sdk_session_forwarding_resumes_after_relay_reconnect(tmp_path, pg_url):
    """Regression test: an SDK-owned session started before a relay drop
    must keep forwarding events after the daemon reconnects - previously
    only observe-only sessions were rediscovered on reconnect, so an
    SDK-owned session's forwarder silently died with the old connection
    and was never restarted (see _watch_active_sessions)."""
    db_path = pg_url
    app = create_app(db_path)
    companion_device_id, companion_token = auth.bootstrap_companion_device(app.state.db)
    _, phone_token = auth.create_device(app.state.db, "phone")
    port = _free_port()

    sdk_adapter, _fake_clients = _fake_sdk_adapter()
    server = await _start_server(app, port)
    daemon = CompanionDaemon(
        _fast_config(port, companion_token),
        sdk_adapter=sdk_adapter,
        observe_adapter=_test_observe_adapter(tmp_path),
        recents_path=str(tmp_path / "recent_projects.json"),
        config_path=str(tmp_path / "companion_config.json"),
        session_settings_path=str(tmp_path / "session_settings.json"),
    )
    task = asyncio.create_task(daemon.run())
    await _wait_until(lambda: daemon.state == "connected")

    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "start_session", "cwd": "/tmp/some-repo", "device_id": companion_device_id}
        )
        started = await phone.next_event()
        session_id = started["session_id"]
        await phone.close()

        # Relay drops entirely, then comes back on the same port/db.
        await _stop_server(server)
        await _wait_until(lambda: daemon.state != "connected")
        app2 = create_app(db_path)
        server2 = await _start_server(app2, port)
        try:
            await _wait_until(lambda: daemon.state == "connected", timeout=5)

            # The SDK session survived the drop (adapter-side state is
            # untouched) - sending it a message must still reach a phone
            # reconnected to the relay's new process.
            phone2 = await _FakePhone.connect(port, phone_token)
            try:
                await phone2.send_action(
                    {
                        "kind": "send_message",
                        "session_id": session_id,
                        "text": "still there?",
                        "device_id": companion_device_id,
                    }
                )
                event = await phone2.next_event(timeout=3.0)
                assert event["type"] == "user_message"
                assert event["data"]["text"] == "still there?"
            finally:
                await phone2.close()
        finally:
            await _stop_server(server2)
    finally:
        await _stop_daemon(daemon, task)


# --- Action routing + event forwarding (daemon <-> SDK adapter <-> phone) --


class _FakePhone:
    """A second real WebSocket connection to /ws/phone, standing in for
    the mobile client - actions are sent phone-shaped
    (`{"token", "type": "action", "action": {...}}`), and forwarded
    companion events arrive the same way U7's dashboard will receive them.
    """

    # Set/cleared by the companion_token fixture for the duration of each
    # test - see that fixture's own comment for why this exists.
    default_device_id: "str | None" = None

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
        if "device_id" not in action and _FakePhone.default_device_id is not None:
            action = {**action, "device_id": _FakePhone.default_device_id}
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
        recents_path=str(tmp_path / "recent_projects.json"),
        config_path=str(tmp_path / "companion_config.json"),
        session_settings_path=str(tmp_path / "session_settings.json"),
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
async def test_start_session_action_passes_the_requested_model_through(running_daemon, phone_token):
    daemon, port, fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo", "model": "claude-opus-5"})

        event = await phone.next_event()
        assert event["type"] == "session_started"
        assert event["data"]["model"] == "claude-opus-5"
        assert fake_clients[0].options.model == "claude-opus-5"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_session_action_with_auto_approve_skips_the_prompt_for_a_read(
    running_daemon, phone_token
):
    """End-to-end: the phone's start_session action turns on policy
    auto-approval for the session, and a Read call resolves without the
    phone ever having to send respond_to_permission."""
    from claude_agent_sdk import ToolPermissionContext

    daemon, port, fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo", "auto_approve": True})
        await phone.next_event()  # session_started
        client = fake_clients[0]

        result = await asyncio.wait_for(
            client.options.can_use_tool("Read", {"file_path": "a.py"}, ToolPermissionContext(tool_use_id="tool-1")),
            timeout=1,
        )
        assert result.behavior == "allow"

        event = await phone.next_event()
        assert event["type"] == "permission_request"
        assert event["data"]["auto_approved"] is True
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_session_action_with_llm_judge_skips_the_prompt_on_a_safe_verdict(
    running_daemon, phone_token, monkeypatch
):
    """End-to-end: the phone's start_session action turns on both
    auto_approve and llm_judge for the session, and a Bash command the
    rule-based policy doesn't cover on its own resolves without the phone
    ever having to send respond_to_permission, once the (faked) LLM judge
    answers SAFE."""
    from claude_agent_sdk import AssistantMessage, TextBlock, ToolPermissionContext

    from companion import risk_judge

    async def fake_query(*, prompt, options):
        yield AssistantMessage(content=[TextBlock(text="SAFE: ordinary command")], model="claude")

    monkeypatch.setattr(risk_judge, "query", fake_query)

    daemon, port, fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "start_session", "cwd": "/tmp/some-repo", "auto_approve": True, "llm_judge": True}
        )
        await phone.next_event()  # session_started
        client = fake_clients[0]

        result = await asyncio.wait_for(
            client.options.can_use_tool(
                "Bash", {"command": "some-custom-script.sh"}, ToolPermissionContext(tool_use_id="tool-1")
            ),
            timeout=1,
        )
        assert result.behavior == "allow"

        event = await phone.next_event()
        assert event["type"] == "permission_request"
        assert event["data"]["auto_approved"] is True
        assert event["data"]["judged_by"] == "llm"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_session_action_with_llm_judge_off_by_default_still_prompts(running_daemon, phone_token):
    """llm_judge is opt-in, separate from auto_approve - omitting it from
    the start_session action must not consult the judge, same as before
    this feature existed."""
    from claude_agent_sdk import ToolPermissionContext

    daemon, port, fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo", "auto_approve": True})
        await phone.next_event()  # session_started
        client = fake_clients[0]

        call_task = asyncio.create_task(
            client.options.can_use_tool(
                "Bash", {"command": "some-custom-script.sh"}, ToolPermissionContext(tool_use_id="tool-1")
            )
        )
        event = await phone.next_event()
        assert event["type"] == "permission_request"
        assert "auto_approved" not in event["data"]

        await phone.send_action(
            {"kind": "respond_to_permission", "session_id": event["session_id"], "request_id": "tool-1", "decision": "allow"}
        )
        result = await asyncio.wait_for(call_task, timeout=1)
        assert result.behavior == "allow"
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
async def test_start_session_persists_its_settings_for_a_later_resume(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {
                "kind": "start_session",
                "cwd": "/tmp/some-repo",
                "model": "claude-opus-5",
                "auto_approve": True,
                "llm_judge": True,
            }
        )
        started = await phone.next_event()
        session_id = started["session_id"]

        saved = load_session_settings(session_id, daemon.session_settings_path)
        assert saved is not None
        assert saved.cwd == os.path.realpath("/tmp/some-repo")
        assert saved.model == "claude-opus-5"
        assert saved.auto_approve is True
        assert saved.llm_judge is True
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_session_auto_approve_updates_the_persisted_record(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        session_id = started["session_id"]
        assert load_session_settings(session_id, daemon.session_settings_path).auto_approve is False

        await phone.send_action(
            {"kind": "set_session_auto_approve", "session_id": session_id, "auto_approve": True}
        )
        await phone.next_event()  # session_auto_approve confirmation

        saved = load_session_settings(session_id, daemon.session_settings_path)
        assert saved.auto_approve is True
        assert saved.llm_judge is False  # untouched - only auto_approve was passed
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_resuming_a_stale_session_restores_its_previously_saved_settings(
    running_daemon, phone_token
):
    """session_settings.py's saved record (written by start_session and
    kept in sync by set_session_auto_approve) is what lets a restart-
    triggered resume come back with the same auto_approve/llm_judge/model
    the phone had actually chosen, instead of always falling back to
    opt-in-off defaults."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {
                "kind": "start_session",
                "cwd": "/tmp/some-repo",
                "model": "claude-opus-5",
                "auto_approve": True,
                "llm_judge": True,
            }
        )
        started = await phone.next_event()
        session_id = started["session_id"]
        real_cwd = daemon.sdk_adapter.get_cwd(session_id)

        # Simulate a companion restart: sdk_adapter's in-memory state is
        # wiped (the FakeSDKClient never wrote a real transcript, so one is
        # written here by hand), but the saved settings survive on disk.
        project_dir = daemon.observe_adapter.projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / f"{session_id}.jsonl").write_text(
            json.dumps({"type": "user", "cwd": real_cwd, "message": {"role": "user", "content": "hi"}}) + "\n"
        )
        del daemon.sdk_adapter._sessions[session_id]
        # A real restart starts self._forwarding empty too (it's a fresh
        # CompanionDaemon instance) - without also clearing this stale
        # entry, _spawn_forwarder's "already forwarding" guard would skip
        # wiring up the resumed session, and the events below would queue
        # forever with nothing reading them.
        old_forwarder = daemon._forwarding.pop(session_id, None)
        if old_forwarder is not None:
            old_forwarder.cancel()

        await phone.send_action({"kind": "send_message", "session_id": session_id, "text": "still there?"})

        resumed_started = await phone.next_event()
        assert resumed_started["type"] == "session_started"
        assert resumed_started["data"]["auto_approve"] is True
        assert resumed_started["data"]["llm_judge"] is True
        assert resumed_started["data"]["model"] == "claude-opus-5"

        message_event = await phone.next_event()
        assert message_event["type"] == "user_message"
        assert message_event["data"]["text"] == "still there?"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_send_message_for_a_stale_session_id_resumes_it_from_its_own_transcript(
    running_daemon, phone_token
):
    """Regression test: sdk_adapter's in-memory _sessions is wiped by every
    companion restart, but the phone doesn't know that - it can still send
    an action for a session_id from before the restart. Rather than
    silently dropping it, the daemon should reconnect via the SDK's own
    resume support (using the transcript's own recorded cwd) and then
    deliver the action against the now-live session."""
    daemon, port, fake_clients = running_daemon
    projects_dir = daemon.observe_adapter.projects_dir
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "old-session-1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/tmp/some-repo", "message": {"role": "user", "content": "hi"}}) + "\n"
    )

    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "send_message", "session_id": "old-session-1", "text": "still there?"}
        )

        # connect()'s own session_started fires first (the reconnect), then
        # the original action actually goes through.
        started = await phone.next_event()
        assert started["type"] == "session_started"
        assert started["session_id"] == "old-session-1"

        message_event = await phone.next_event()
        assert message_event["type"] == "user_message"
        assert message_event["data"]["text"] == "still there?"
        assert message_event["session_id"] == "old-session-1"

        assert fake_clients[0].options.resume == "old-session-1"
        # os.path.realpath'd by connect() itself (macOS resolves /tmp ->
        # /private/tmp) - same regression this whole resolution exists to
        # avoid, see sdk_adapter.py's connect() comment.
        assert fake_clients[0].options.cwd == os.path.realpath("/tmp/some-repo")
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_send_message_for_a_session_id_with_no_matching_transcript_still_warns_and_drops(
    running_daemon, phone_token
):
    """No transcript anywhere means this was never a real session on this
    machine - resume must not be attempted, and the action is dropped the
    same way it always was (logged, no crash, no event to the phone)."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "send_message", "session_id": "never-existed", "text": "hello?"}
        )
        with pytest.raises(asyncio.TimeoutError):
            await phone.next_event(timeout=0.5)
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_compact_action_routes_to_the_owning_sdk_session_without_a_user_message_event(
    running_daemon, phone_token
):
    """Unlike send_message, compact must not surface as a fake chat
    bubble - it goes straight into the session's outbound queue."""
    daemon, port, fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        session_id = started["session_id"]
        client = fake_clients[0]

        await phone.send_action({"kind": "compact", "session_id": session_id})

        queued = await client.connected_prompt.__anext__()
        assert queued["message"]["content"] == "/compact"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_git_status_action_returns_a_real_status_snapshot(running_daemon, phone_token, tmp_path):
    """U10 (R16): the git_status action's result rides the session's own
    event stream (emit_custom), computed against the session's real cwd -
    companion/git_status.py's own tests cover the git plumbing itself."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "README.md").write_text("hi\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Initial"], cwd=repo, check=True)
    (repo / "README.md").write_text("changed\n")

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": str(repo)})
        started = await phone.next_event()
        session_id = started["session_id"]

        await phone.send_action({"kind": "git_status", "session_id": session_id})

        event = await phone.next_event()
        assert event["type"] == "git_status"
        assert event["data"]["is_git_repo"] is True
        assert "README.md" in event["data"]["modified"]
        assert event["data"]["last_commit"] == "Initial"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_git_diff_action_returns_a_real_diff(running_daemon, phone_token, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    (repo / "README.md").write_text("original\n")
    subprocess.run(["git", "add", "README.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Initial"], cwd=repo, check=True)
    (repo / "README.md").write_text("updated\n")

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": str(repo)})
        started = await phone.next_event()
        session_id = started["session_id"]

        await phone.send_action({"kind": "git_diff", "session_id": session_id, "path": "README.md"})

        event = await phone.next_event()
        assert event["type"] == "git_diff"
        assert event["data"]["is_binary"] is False
        assert "+updated" in event["data"]["diff"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_git_status_with_no_known_cwd_reports_not_a_repo_rather_than_crashing(
    running_daemon, phone_token
):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        # start_session with no cwd at all - SDKAdapter.get_cwd returns None.
        await phone.send_action({"kind": "start_session", "cwd": None})
        started = await phone.next_event()
        session_id = started["session_id"]

        await phone.send_action({"kind": "git_status", "session_id": session_id})

        event = await phone.next_event()
        assert event["type"] == "git_status"
        assert event["data"]["is_git_repo"] is False
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_session_records_the_resolved_cwd_as_recent(running_daemon, phone_token, tmp_path):
    # Same tmp_path the running_daemon fixture used to build daemon.recents_path.
    recents_path = str(tmp_path / "recent_projects.json")

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        await phone.next_event()  # session_started

        await _wait_until(lambda: os.path.exists(recents_path))
        with open(recents_path) as f:
            # realpath-resolved (see sdk_adapter.py's connect()) - macOS's
            # /tmp is itself a symlink to /private/tmp, which is exactly
            # the class of mismatch this resolution exists to avoid (see
            # history.py's list_project_sessions).
            assert json.load(f) == [os.path.realpath("/tmp/some-repo")]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_list_active_sessions_action_returns_a_snapshot_of_both_adapters(running_daemon, phone_token):
    """Regression test: SessionListScreen.tsx has no other way to learn
    what's already running once its own component instance has been
    unmounted and remounted (it only ever finds out about a session by
    witnessing its session_started event live) - this snapshot is what
    lets it repopulate on mount instead of showing nothing."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        sdk_session_id = started["session_id"]

        project_dir = daemon.observe_adapter.projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "observed-session-1.jsonl").touch()
        await phone.next_event()  # that transcript's own session_started

        await phone.send_action({"kind": "list_active_sessions"})
        event = await phone.next_event()

        assert event["type"] == "active_sessions"
        assert event["session_id"] == "_active_sessions"
        by_id = {s["session_id"]: s for s in event["data"]["sessions"]}
        assert by_id[sdk_session_id]["mode"] == "sdk_owned"
        assert by_id[sdk_session_id]["cwd"] == os.path.realpath("/tmp/some-repo")
        assert by_id[sdk_session_id]["active"] is True
        assert by_id["observed-session-1"]["mode"] == "observe_only"
        assert by_id["observed-session-1"]["active"] is True
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_list_active_sessions_reports_an_interrupted_sdk_session_as_inactive(
    running_daemon, phone_token
):
    """Regression test: interrupt() ends a session (session_ended,
    _ended=True) without removing it from SDKAdapter's own _sessions dict
    - only disconnect() does that - so discover_sessions() alone still
    lists it. Without checking is_active(), the snapshot reported it as
    still running, which looked to the phone like a stopped session
    reappeared/"restarted" the next time the Sessions screen remounted."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        session_id = started["session_id"]

        await phone.send_action({"kind": "interrupt", "session_id": session_id})
        ended = await phone.next_event()
        assert ended["type"] == "session_ended"

        await phone.send_action({"kind": "list_active_sessions"})
        event = await phone.next_event()

        by_id = {s["session_id"]: s for s in event["data"]["sessions"]}
        assert by_id[session_id]["active"] is False
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_list_projects_action_returns_the_picker_list(running_daemon, phone_token, tmp_path, monkeypatch):
    from companion import projects

    projects_dir = tmp_path / "claude-projects"
    real_project = tmp_path / "Projects" / "app"
    real_project.mkdir(parents=True)
    project_dir = projects_dir / "-Users-x-Projects-app"
    project_dir.mkdir(parents=True)
    (project_dir / "session-1.jsonl").write_text(
        json.dumps({"type": "assistant", "cwd": str(real_project)}) + "\n"
    )
    monkeypatch.setattr(projects, "DEFAULT_PROJECTS_DIR", str(projects_dir))

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "list_projects"})

        event = await phone.next_event()
        assert event["type"] == "project_list"
        assert event["session_id"] == "_projects"
        assert event["data"]["projects"] == [
            {"path": str(real_project), "last_used_at": event["data"]["projects"][0]["last_used_at"], "recent": False}
        ]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_list_project_sessions_action_returns_past_sessions_for_that_cwd(
    running_daemon, phone_token, tmp_path, monkeypatch
):
    from companion import history

    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / "-Users-x-app"
    project_dir.mkdir(parents=True)
    (project_dir / "session-1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "fix the bug"}}) + "\n"
    )
    monkeypatch.setattr(history, "DEFAULT_PROJECTS_DIR", str(projects_dir))

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "list_project_sessions", "cwd": "/Users/x/app"})

        event = await phone.next_event()
        assert event["type"] == "session_history_list"
        assert event["data"]["cwd"] == "/Users/x/app"
        assert event["data"]["sessions"][0]["session_id"] == "session-1"
        assert event["data"]["sessions"][0]["preview"] == "fix the bug"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_get_observe_settings_action_reports_known_and_currently_selected_entrypoints(
    running_daemon, phone_token
):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "get_observe_settings"})

        event = await phone.next_event()
        assert event["type"] == "observe_settings"
        assert set(event["data"]["known_entrypoints"]) == set(KNOWN_ENTRYPOINTS)
        assert event["data"]["selected_entrypoints"] == sorted(daemon.observe_adapter.get_required_entrypoints())
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_observe_entrypoints_action_updates_the_adapter_and_persists_to_config(
    running_daemon, phone_token
):
    """R: 'give an ability to choose what clients to use' - the choice
    must survive a daemon restart, not just apply in memory."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_observe_entrypoints", "entrypoints": ["claude-desktop", "claude-vscode"]})

        event = await phone.next_event()
        assert event["type"] == "observe_settings"
        assert event["data"]["selected_entrypoints"] == ["claude-desktop", "claude-vscode"]
        assert daemon.observe_adapter.get_required_entrypoints() == frozenset({"claude-desktop", "claude-vscode"})

        persisted = load_config(daemon.config_path)
        assert sorted(persisted.observe_entrypoints) == ["claude-desktop", "claude-vscode"]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_get_auto_approve_settings_action_reports_the_current_global_policy(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "get_auto_approve_settings"})

        event = await phone.next_event()
        assert event["type"] == "auto_approve_settings"
        assert event["data"]["auto_approve"] is False
        assert event["data"]["llm_judge"] is False
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_auto_approve_settings_action_applies_globally_and_persists_to_config(
    running_daemon, phone_token
):
    """This is a global switch, not per-session - it must apply to
    ObserveAdapter (terminal-started sessions) immediately, and survive a
    daemon restart, not just apply in memory for this connection."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_auto_approve_settings", "auto_approve": True, "llm_judge": True})

        event = await phone.next_event()
        assert event["type"] == "auto_approve_settings"
        assert event["data"]["auto_approve"] is True
        assert event["data"]["llm_judge"] is True
        assert daemon.observe_adapter.get_auto_approve() is True
        assert daemon.observe_adapter.get_llm_judge() is True

        persisted = load_config(daemon.config_path)
        assert persisted.observe_auto_approve is True
        assert persisted.observe_llm_judge is True
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_auto_approve_settings_action_can_toggle_auto_approve_without_touching_llm_judge(
    running_daemon, phone_token
):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_auto_approve_settings", "auto_approve": True})

        event = await phone.next_event()
        assert event["data"]["auto_approve"] is True
        assert event["data"]["llm_judge"] is False
    finally:
        await phone.close()


# --- U4: per-Mac CLI binary/profile setting -------------------------------


@pytest.mark.asyncio
async def test_get_cli_settings_action_reports_the_current_defaults(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "get_cli_settings"})

        event = await phone.next_event()
        assert event["type"] == "cli_settings"
        assert event["data"]["cli_path"] is None
        assert event["data"]["cli_env"] == {}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_cli_settings_action_persists_to_config_and_reflects_back(running_daemon, phone_token):
    # sys.executable (the running Python interpreter) is a real, absolute,
    # executable path on any machine these tests run on - _is_valid_remote_cli_path
    # (companion/daemon.py) now requires that of a phone-supplied cli_path,
    # so a fixed made-up path like "/usr/local/bin/claude-custom" would be
    # rejected here rather than exercising the persist-and-reflect-back path
    # this test is actually about (see the rejection tests below for that).
    daemon, port, _fake_clients = running_daemon
    real_cli_path = sys.executable
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {
                "kind": "set_cli_settings",
                "cli_path": real_cli_path,
                "cli_env": {"ANTHROPIC_API_KEY": "sk-test"},
            }
        )

        event = await phone.next_event()
        assert event["type"] == "cli_settings"
        assert event["data"]["cli_path"] == real_cli_path
        assert event["data"]["cli_env"] == {"ANTHROPIC_API_KEY": "sk-test"}
        assert daemon.config.cli_path == real_cli_path
        assert daemon.config.cli_env == {"ANTHROPIC_API_KEY": "sk-test"}

        persisted = load_config(daemon.config_path)
        assert persisted.cli_path == real_cli_path
        assert persisted.cli_env == {"ANTHROPIC_API_KEY": "sk-test"}

        # A subsequent get_cli_settings reflects the persisted values too,
        # not just the immediate confirmation reply above.
        await phone.send_action({"kind": "get_cli_settings"})
        confirm = await phone.next_event()
        assert confirm["data"]["cli_path"] == real_cli_path
        assert confirm["data"]["cli_env"] == {"ANTHROPIC_API_KEY": "sk-test"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_cli_settings_rejects_a_path_that_does_not_exist_on_this_mac(running_daemon, phone_token):
    """A phone action is a code-execution-adjacent primitive once accepted
    (cli_path feeds straight into the subprocess spawned for every later
    session) - a path with nothing on disk at it is rejected outright
    rather than persisted and left to fail confusingly at the next
    start_session."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "set_cli_settings", "cli_path": "/definitely/not/a/real/path/claude"}
        )

        confirm = await phone.next_event()
        assert confirm["data"]["cli_path"] is None
        assert daemon.config.cli_path is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_cli_settings_rejects_a_relative_path(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_cli_settings", "cli_path": "claude"})

        confirm = await phone.next_event()
        assert confirm["data"]["cli_path"] is None
        assert daemon.config.cli_path is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_cli_settings_rejects_a_dynamic_linker_injection_env_key(running_daemon, phone_token):
    """DYLD_INSERT_LIBRARIES et al. could hijack the spawned CLI subprocess
    regardless of what cli_path itself points to - rejected outright
    (not stripped) so the rejection is visible rather than the change
    silently landing minus one key."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {
                "kind": "set_cli_settings",
                "cli_env": {"DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib", "SAFE_VAR": "ok"},
            }
        )

        confirm = await phone.next_event()
        assert confirm["data"]["cli_env"] == {}
        assert daemon.config.cli_env == {}
    finally:
        await phone.close()


# --- U2: account-settings actions and custom-token setup ---


@pytest.mark.asyncio
async def test_get_account_settings_action_reports_the_current_defaults(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "get_account_settings"})

        event = await phone.next_event()
        assert event["type"] == "account_settings"
        assert event["data"]["active_account"] == "vscode"
        assert event["data"]["personal_configured"] is False
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_personal_account_token_persists_and_reports_configured(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_personal_account_token", "token": "sk-ant-oat-test-token"})

        confirm = await phone.next_event()
        assert confirm["type"] == "account_settings"
        assert confirm["data"]["personal_configured"] is True
        # The raw token value never rides this confirmation.
        assert "token" not in confirm["data"]
        assert daemon.config.personal_oauth_token == "sk-ant-oat-test-token"

        persisted = load_config(daemon.config_path)
        assert persisted.personal_oauth_token == "sk-ant-oat-test-token"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_personal_account_token_rejects_empty_or_whitespace(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_personal_account_token", "token": "   "})

        confirm = await phone.next_event()
        assert confirm["data"]["personal_configured"] is False
        assert daemon.config.personal_oauth_token is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_active_account_switches_once_a_token_is_configured(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_personal_account_token", "token": "sk-ant-oat-test-token"})
        await phone.next_event()

        await phone.send_action({"kind": "set_active_account", "active_account": "personal"})

        confirm = await phone.next_event()
        assert confirm["data"]["active_account"] == "personal"
        assert daemon.config.active_account == "personal"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_active_account_rejects_an_unknown_value(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_active_account", "active_account": "bogus"})

        confirm = await phone.next_event()
        assert confirm["data"]["active_account"] == "vscode"
        assert daemon.config.active_account == "vscode"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_active_account_rejects_personal_before_any_token_is_configured(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "set_active_account", "active_account": "personal"})

        confirm = await phone.next_event()
        assert confirm["data"]["active_account"] == "vscode"
        assert daemon.config.active_account == "vscode"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_personal_account_setup_reports_unavailable_when_automation_fails(
    running_daemon, phone_token, monkeypatch
):
    from companion import daemon as daemon_module

    async def _fake_run_setup_token_under_pty(cli_path):
        return None

    monkeypatch.setattr(daemon_module, "_run_setup_token_under_pty", _fake_run_setup_token_under_pty)

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_personal_account_setup"})

        event = await phone.next_event()
        assert event["type"] == "personal_account_setup_result"
        assert event["data"] == {"available": False}
        assert daemon.config.personal_oauth_token is None
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_start_personal_account_setup_stores_the_captured_token_via_the_shared_path(
    running_daemon, phone_token, monkeypatch
):
    """Same persistence path as set_personal_account_token - no duplicated
    logic (U2's own test scenario)."""
    from companion import daemon as daemon_module

    async def _fake_run_setup_token_under_pty(cli_path):
        return "sk-ant-oat-captured-token"

    monkeypatch.setattr(daemon_module, "_run_setup_token_under_pty", _fake_run_setup_token_under_pty)

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_personal_account_setup"})

        event = await phone.next_event()
        assert event["type"] == "account_settings"
        assert event["data"]["personal_configured"] is True
        assert daemon.config.personal_oauth_token == "sk-ant-oat-captured-token"

        persisted = load_config(daemon.config_path)
        assert persisted.personal_oauth_token == "sk-ant-oat-captured-token"
    finally:
        await phone.close()


def _fake_claude_binary(tmp_path, script_body: str):
    """A standalone executable that stands in for `claude` -
    _run_setup_token_under_pty always execs `[cli_path, "setup-token"]`
    directly (no shell), so the fake has to be one real executable file,
    not an "interpreter + script" string."""
    script = tmp_path / "fake-claude"
    script.write_text(f"#!{sys.executable}\n{script_body}\n")
    script.chmod(0o755)
    return str(script)


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_returns_none_when_the_binary_does_not_exist():
    from companion.daemon import _run_setup_token_under_pty

    result = await _run_setup_token_under_pty("/definitely/not/a/real/claude/binary")

    assert result is None


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_returns_none_on_multi_line_output(tmp_path):
    from companion.daemon import _run_setup_token_under_pty

    binary = _fake_claude_binary(tmp_path, "print('line one')\nprint('line two')\n")

    result = await _run_setup_token_under_pty(binary)

    assert result is None


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_returns_none_on_nonzero_exit(tmp_path):
    from companion.daemon import _run_setup_token_under_pty

    binary = _fake_claude_binary(tmp_path, "import sys\nprint('sk-ant-oat-should-be-discarded')\nsys.exit(1)\n")

    result = await _run_setup_token_under_pty(binary)

    assert result is None


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_captures_a_single_clean_token_line(tmp_path):
    from companion.daemon import _run_setup_token_under_pty

    binary = _fake_claude_binary(tmp_path, "print('sk-ant-oat-fake-captured-token-value')\n")

    result = await _run_setup_token_under_pty(binary)

    assert result == "sk-ant-oat-fake-captured-token-value"


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_returns_none_when_output_is_too_short_to_be_a_token(tmp_path):
    from companion.daemon import _run_setup_token_under_pty

    binary = _fake_claude_binary(tmp_path, "print('short')\n")

    result = await _run_setup_token_under_pty(binary)

    assert result is None


@pytest.mark.asyncio
async def test_run_setup_token_under_pty_times_out_on_a_hanging_process(tmp_path, monkeypatch):
    from companion import daemon as daemon_module

    monkeypatch.setattr(daemon_module, "_SETUP_TOKEN_TIMEOUT_SECONDS", 0.2)
    binary = _fake_claude_binary(tmp_path, "import time\ntime.sleep(5)\n")

    result = await daemon_module._run_setup_token_under_pty(binary)

    assert result is None


@pytest.mark.asyncio
async def test_start_session_action_threads_the_configured_cli_settings_into_the_sdk_adapter(
    running_daemon, phone_token
):
    """Unlike model/auto_approve/llm_judge, cli_path/cli_env have no
    phone-side per-request equivalent (KTD5) - start_session must read
    them straight from self.config, not from the action payload."""
    daemon, port, fake_clients = running_daemon
    daemon.config.cli_path = "/usr/local/bin/claude-custom"
    daemon.config.cli_env = {"ANTHROPIC_API_KEY": "sk-test"}

    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        await phone.next_event()  # session_started

        assert fake_clients[0].options.cli_path == "/usr/local/bin/claude-custom"
        assert fake_clients[0].options.env == {"ANTHROPIC_API_KEY": "sk-test"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_resumed_sdk_session_also_threads_the_configured_cli_settings(running_daemon, phone_token):
    """_try_resume_sdk_session is the other of the two connect() call
    sites (KTD5) - it must also read cli_path/cli_env from self.config
    directly, since there's no saved SessionSettings equivalent for a
    Mac-level setting."""
    daemon, port, fake_clients = running_daemon
    daemon.config.cli_path = "/usr/local/bin/claude-custom"
    daemon.config.cli_env = {"ANTHROPIC_API_KEY": "sk-test"}

    projects_dir = daemon.observe_adapter.projects_dir
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "old-session-1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/tmp/some-repo", "message": {"role": "user", "content": "hi"}}) + "\n"
    )

    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action(
            {"kind": "send_message", "session_id": "old-session-1", "text": "still there?"}
        )
        started = await phone.next_event()
        assert started["type"] == "session_started"

        assert fake_clients[0].options.cli_path == "/usr/local/bin/claude-custom"
        assert fake_clients[0].options.env == {"ANTHROPIC_API_KEY": "sk-test"}
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_read_session_history_action_returns_the_normalized_conversation(
    running_daemon, phone_token, tmp_path, monkeypatch
):
    from companion import history

    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / "-Users-x-app"
    project_dir.mkdir(parents=True)
    (project_dir / "session-1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hello"}}) + "\n"
    )
    monkeypatch.setattr(history, "DEFAULT_PROJECTS_DIR", str(projects_dir))

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "read_session_history", "session_id": "session-1"})

        event = await phone.next_event()
        assert event["type"] == "session_history"
        assert event["data"]["session_id"] == "session-1"
        assert event["data"]["events"] == [{"type": "user_message", "timestamp": "", "data": {"text": "hello"}}]
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_read_session_history_action_for_unknown_session_returns_empty_events(
    running_daemon, phone_token, tmp_path, monkeypatch
):
    from companion import history

    monkeypatch.setattr(history, "DEFAULT_PROJECTS_DIR", str(tmp_path / "claude-projects"))

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "read_session_history", "session_id": "no-such-session"})

        event = await phone.next_event()
        assert event["type"] == "session_history"
        assert event["data"]["events"] == []
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


@pytest.mark.asyncio
async def test_open_session_action_lets_a_discovered_observe_session_start_forwarding_content(
    running_daemon, phone_token
):
    """R: 'do not connect automatically to opened session, just show that
    it exists' - discovery alone only gets the phone session_started;
    content requires the phone to explicitly open_session first."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        projects_dir = daemon.observe_adapter.projects_dir
        project_dir = projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        transcript = project_dir / "session-open.jsonl"
        transcript.touch()

        started = await phone.next_event(timeout=3.0)
        assert started["type"] == "session_started"

        await phone.send_action({"kind": "open_session", "session_id": "session-open"})
        await asyncio.sleep(0.05)  # let the action reach the adapter before writing content

        with open(transcript, "a") as f:
            f.write(
                json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}})
                + "\n"
            )

        content_event = await phone.next_event(timeout=3.0)
        assert content_event["type"] == "assistant_message"
        assert content_event["data"]["text"] == "hi"
    finally:
        await phone.close()


def _send_hook(socket_path: str, event: str, payload: dict) -> dict:
    body = dict(payload)
    body["_hook_event"] = event
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(socket_path)
        s.sendall(json.dumps(body).encode() + b"\n")
        s.shutdown(socket.SHUT_WR)
        data = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            data += chunk
    return json.loads(data.decode()) if data else {}


async def _send_hook_async(socket_path: str, event: str, payload: dict) -> dict:
    return await asyncio.to_thread(_send_hook, socket_path, event, payload)


@pytest.mark.asyncio
async def test_set_auto_approve_settings_does_not_retroactively_affect_an_already_discovered_session(
    running_daemon, phone_token
):
    """End-to-end: an observed session's auto_approve/llm_judge are
    snapshotted once, at discovery time (ObserveAdapter._get_or_create) -
    not read live from the global setting on every hook call anymore.
    Turning the global switch on from the phone, via
    set_auto_approve_settings, changes what *new* sessions get; a session
    already discovered before that keeps prompting until either it's
    individually overridden (see the next test) or it's re-discovered
    fresh. This matches how a phone-started session's own auto_approve
    has always worked (frozen at start_session time) - the two session
    types are now consistent instead of one being "live" and one not."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        projects_dir = daemon.observe_adapter.projects_dir
        project_dir = projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "session-live.jsonl").touch()

        started = await phone.next_event(timeout=3.0)
        assert started["type"] == "session_started"
        assert started["data"]["auto_approve"] is False

        await phone.send_action({"kind": "set_auto_approve_settings", "auto_approve": True})
        settings_event = await phone.next_event(timeout=2.0)
        assert settings_event["type"] == "auto_approve_settings"
        assert settings_event["data"]["auto_approve"] is True

        hook_task = asyncio.create_task(
            _send_hook_async(
                daemon.observe_adapter.socket_path,
                "PermissionRequest",
                {"session_id": "session-live", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "pytest"}},
            )
        )

        permission_event = await phone.next_event(timeout=2.0)
        assert permission_event["type"] == "permission_request"
        assert "auto_approved" not in permission_event["data"]  # still a real prompt, not auto-approved

        await phone.send_action(
            {"kind": "respond_to_permission", "session_id": "session-live", "request_id": "tool-1", "decision": "allow"}
        )
        response = await asyncio.wait_for(hook_task, timeout=2.0)
        assert response["permissionDecision"] == "allow"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_set_session_auto_approve_overrides_just_that_one_observed_session(running_daemon, phone_token):
    """The per-session override (companion/adapters/observe_adapter.py's
    set_session_auto_approve) is how you actually change an
    already-discovered session's behavior mid-conversation - unlike the
    global setting (previous test), this takes effect immediately for
    the specific session it targets."""
    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        projects_dir = daemon.observe_adapter.projects_dir
        project_dir = projects_dir / "my-repo"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "session-live.jsonl").touch()

        started = await phone.next_event(timeout=3.0)
        assert started["type"] == "session_started"

        # Matches the real mobile flow (SessionDashboardScreen sends this on
        # mount, before the auto-approve toggle is ever reachable) - a
        # session must be opened before non-lifecycle events like
        # session_auto_approve forward to the phone (_ObserveSession.emit).
        await phone.send_action({"kind": "open_session", "session_id": "session-live"})

        await phone.send_action(
            {"kind": "set_session_auto_approve", "session_id": "session-live", "auto_approve": True}
        )
        confirm_event = await phone.next_event(timeout=2.0)
        assert confirm_event["type"] == "session_auto_approve"
        assert confirm_event["data"]["auto_approve"] is True

        response = await _send_hook_async(
            daemon.observe_adapter.socket_path,
            "PermissionRequest",
            {"session_id": "session-live", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "pytest"}},
        )
        assert response == {"permissionDecision": "allow", "permissionDecisionReason": ""}

        permission_event = await phone.next_event(timeout=2.0)
        assert permission_event["type"] == "permission_request"
        assert permission_event["data"]["auto_approved"] is True
        assert permission_event["data"]["judged_by"] == "policy"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_an_exception_in_one_of_the_new_sentinel_actions_does_not_kill_the_connection(
    running_daemon, phone_token, monkeypatch
):
    """Regression test: list_project_sessions/read_session_history/
    open_session/get_observe_settings/set_observe_entrypoints previously
    ran outside any try/except, unlike the session-scoped actions below
    them - an exception there would propagate out of the fire-and-forget
    _handle_action task uncaught (only asyncio's own "exception was never
    retrieved" warning, never our logging) rather than being caught and
    logged, and would have no test coverage proving the connection itself
    survives it."""
    from companion import history

    def _boom(cwd, projects_dir=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(history, "list_project_sessions", _boom)

    daemon, port, _fake_clients = running_daemon
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "list_project_sessions", "cwd": "/Users/x/app"})
        await asyncio.sleep(0.05)  # let the failing action run and get caught

        # The connection itself must still be usable afterward.
        await phone.send_action({"kind": "get_observe_settings"})
        event = await phone.next_event(timeout=3.0)
        assert event["type"] == "observe_settings"
    finally:
        await phone.close()


# --- U6: secret redaction from forwarded event content ---

from companion.daemon import _redact_secrets  # noqa: E402


def test_redact_secrets_replaces_an_exact_substring_match_in_a_string():
    assert _redact_secrets("token=sk-secret-1 end", ("sk-secret-1",)) == "token=[redacted] end"


def test_redact_secrets_recurses_through_nested_dicts_and_lists():
    value = {
        "content": [{"type": "text", "text": "leaked: sk-secret-1"}],
        "nested": {"deep": ["sk-secret-2", "unrelated"]},
    }

    redacted = _redact_secrets(value, ("sk-secret-1", "sk-secret-2"))

    assert redacted == {
        "content": [{"type": "text", "text": "leaked: [redacted]"}],
        "nested": {"deep": ["[redacted]", "unrelated"]},
    }


def test_redact_secrets_leaves_content_with_no_secret_substring_unchanged():
    value = {"text": "nothing sensitive here"}

    assert _redact_secrets(value, ("sk-secret-1",)) == value


def test_redact_secrets_is_a_no_op_with_no_secrets_configured():
    value = {"text": "some content"}

    assert _redact_secrets(value, ()) == value


def test_redact_secrets_passes_through_non_string_leaf_values():
    value = {"count": 3, "active": True, "missing": None}

    assert _redact_secrets(value, ("sk-secret-1",)) == value


@pytest.mark.asyncio
async def test_send_event_redacts_the_configured_secrets_from_forwarded_content(running_daemon, phone_token):
    daemon, port, _fake_clients = running_daemon
    daemon.config.personal_oauth_token = "sk-ant-oat-leaked"
    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "start_session", "cwd": "/tmp/some-repo"})
        started = await phone.next_event()
        session_id = started["session_id"]

        secret_bearing_text = f"my device token is {daemon.config.device_token} and oauth is sk-ant-oat-leaked"
        await phone.send_action({"kind": "send_message", "session_id": session_id, "text": secret_bearing_text})

        event = await phone.next_event()
        assert event["type"] == "user_message"
        assert daemon.config.device_token not in event["data"]["text"]
        assert "sk-ant-oat-leaked" not in event["data"]["text"]
        assert event["data"]["text"] == "my device token is [redacted] and oauth is [redacted]"
    finally:
        await phone.close()


@pytest.mark.asyncio
async def test_read_session_history_redacts_secrets_from_the_replayed_transcript(
    running_daemon, phone_token, tmp_path, monkeypatch
):
    from companion import history

    daemon, port, _fake_clients = running_daemon
    daemon.config.personal_oauth_token = "sk-ant-oat-leaked"

    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / "-Users-x-app"
    project_dir.mkdir(parents=True)
    secret_bearing_message = f"leaked token: {daemon.config.device_token}"
    (project_dir / "session-1.jsonl").write_text(
        json.dumps({"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": secret_bearing_message}})
        + "\n"
    )
    monkeypatch.setattr(history, "DEFAULT_PROJECTS_DIR", str(projects_dir))

    phone = await _FakePhone.connect(port, phone_token)
    try:
        await phone.send_action({"kind": "read_session_history", "session_id": "session-1"})

        event = await phone.next_event()
        assert event["type"] == "session_history"
        replayed_text = event["data"]["events"][0]["data"]["text"]
        assert daemon.config.device_token not in replayed_text
        assert replayed_text == "leaked token: [redacted]"
    finally:
        await phone.close()
