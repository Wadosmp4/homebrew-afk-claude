"""Tests for companion/adapters/observe_adapter.py.

Uses a real filesystem transcript directory and a real Unix domain socket
(no mocks for the tail/socket boundary) - only the wall-clock polling
intervals are sped up for test speed.
"""
from __future__ import annotations

import asyncio
import json
import socket
import uuid

import pytest

from companion.adapters.observe_adapter import ObserveAdapter

FAST_POLL = 0.01


def _short_socket_path() -> str:
    # AF_UNIX paths are capped at ~104 bytes on macOS/BSD - pytest's
    # tmp_path is nested too deep to use directly for the socket file
    # (though it's fine for the projects_dir, which is opened normally).
    return f"/tmp/rc-{uuid.uuid4().hex[:8]}.sock"


async def _wait_until(predicate, timeout: float = 2.0) -> None:
    async def _poll():
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout=timeout)


def _write_line(path, obj) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


@pytest.fixture
async def adapter(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    socket_path = _short_socket_path()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=socket_path,
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
    )
    await a.start()
    yield a
    await a.stop()


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
async def test_session_started_is_not_duplicated_when_both_hook_and_file_discovery_fire(adapter, tmp_path):
    """Regression test: a real session is normally noticed by *both* the
    SessionStart hook and the JSONL file watcher for the same session_id
    (hooks_installer.py registers SessionStart for every watched repo, and
    _watch_projects_dir independently discovers the new transcript file
    shortly after) - previously each path emitted its own session_started
    unconditionally, so every real session showed two "Session started"
    events in the mobile feed."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-dup.jsonl"

    # Hook fires first (as it would in practice - SessionStart precedes
    # the first transcript write).
    await _send_hook_async(adapter.socket_path, "SessionStart", {"session_id": "session-dup"})
    events = adapter.subscribe("session-dup")
    started = await events.__anext__()
    assert started.type == "session_started"

    # File discovery now also notices the same session.
    transcript.touch()
    await _wait_until(lambda: "session-dup" in adapter.discover_sessions())
    _write_line(
        transcript,
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}},
    )

    # The next event must be the real content, not a second session_started.
    next_event = await events.__anext__()
    assert next_event.type == "assistant_message"


@pytest.mark.asyncio
async def test_discovers_transcript_and_streams_normalized_events(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-abc.jsonl"
    transcript.touch()

    await _wait_until(lambda: "session-abc" in adapter.discover_sessions())

    events = adapter.subscribe("session-abc")
    started = await events.__anext__()
    assert started.type == "session_started"
    assert started.data["mode"] == "observe_only"  # U7 disables send/interrupt using this

    _write_line(
        transcript,
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}},
    )
    assistant_event = await events.__anext__()
    assert assistant_event.type == "assistant_message"
    assert assistant_event.data["text"] == "hi"

    _write_line(
        transcript,
        {"type": "user", "message": {"role": "user", "content": "what's up"}},
    )
    user_event = await events.__anext__()
    assert user_event.type == "user_message"
    assert user_event.data["text"] == "what's up"

    _write_line(
        transcript,
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "Read", "input": {"path": "x.py"}}],
            },
        },
    )
    call_event = await events.__anext__()
    assert call_event.type == "tool_call"
    assert call_event.data["tool"] == "Read"

    _write_line(
        transcript,
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "contents"}],
            },
        },
    )
    result_event = await events.__anext__()
    assert result_event.type == "tool_result"
    assert result_event.data["tool_use_id"] == "tool-1"


@pytest.mark.asyncio
async def test_tool_result_duration_computed_from_entry_timestamps(adapter, tmp_path):
    """U9 (R14): unlike U3's SDK-owned session (a monotonic clock),
    duration here comes from the JSONL entries' own `timestamp` fields."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-timed.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-timed" in adapter.discover_sessions())
    events = adapter.subscribe("session-timed")
    await events.__anext__()  # session_started

    _write_line(
        transcript,
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "pytest"}}],
            },
        },
    )
    await events.__anext__()  # tool_call

    _write_line(
        transcript,
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:02.500Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "14 passed"}],
            },
        },
    )
    result_event = await events.__anext__()

    assert result_event.data["duration_ms"] == pytest.approx(2500, abs=1)


@pytest.mark.asyncio
async def test_tool_result_without_a_matching_tool_call_has_no_duration(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-untimed.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-untimed" in adapter.discover_sessions())
    events = adapter.subscribe("session-untimed")
    await events.__anext__()  # session_started

    _write_line(
        transcript,
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:00.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "orphan-tool-use-id", "content": "x"}],
            },
        },
    )
    result_event = await events.__anext__()

    assert result_event.data["duration_ms"] is None


@pytest.mark.asyncio
async def test_large_tool_result_content_is_truncated(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-huge.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-huge" in adapter.discover_sessions())
    events = adapter.subscribe("session-huge")
    await events.__anext__()  # session_started

    _write_line(
        transcript,
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "pytest"}}],
            },
        },
    )
    await events.__anext__()  # tool_call

    huge_output = "PASS\n" * 5000
    _write_line(
        transcript,
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": huge_output}],
            },
        },
    )
    result_event = await events.__anext__()

    assert len(result_event.data["content"]) < len(huge_output)
    assert "truncated" in result_event.data["content"]


@pytest.mark.asyncio
async def test_concurrent_partial_write_does_not_double_emit_or_drop(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-partial.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-partial" in adapter.discover_sessions())
    events = adapter.subscribe("session-partial")
    await events.__anext__()  # session_started

    line = json.dumps({"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "partial-safe"}]}})
    # Write in two chunks with no trailing newline on the first, simulating
    # a reader catching the file mid-write.
    with open(transcript, "a") as f:
        f.write(line[: len(line) // 2])
    await asyncio.sleep(FAST_POLL * 3)
    with open(transcript, "a") as f:
        f.write(line[len(line) // 2 :] + "\n")

    event = await asyncio.wait_for(events.__anext__(), timeout=2)
    assert event.type == "assistant_message"
    assert event.data["text"] == "partial-safe"


@pytest.mark.asyncio
async def test_permission_request_hook_round_trips_through_socket(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-perm.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-perm" in adapter.discover_sessions())
    events = adapter.subscribe("session-perm")
    await events.__anext__()  # session_started

    hook_task = asyncio.create_task(
        _send_hook_async(
            adapter.socket_path,
            "PermissionRequest",
            {"session_id": "session-perm", "tool_use_id": "tool-9", "tool_name": "Bash", "tool_input": {}},
        )
    )

    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"
    request_id = permission_event.data["request_id"]

    await adapter.respond_to_permission("session-perm", request_id, "allow")
    response = await asyncio.wait_for(hook_task, timeout=2)
    assert response["permissionDecision"] == "allow"


@pytest.mark.asyncio
async def test_hook_for_unknown_session_is_rejected(adapter):
    response = await _send_hook_async(
        adapter.socket_path, "Stop", {"session_id": "no-such-session"}
    )
    assert response == {}
    assert "no-such-session" not in adapter.discover_sessions()


@pytest.mark.asyncio
async def test_send_message_and_interrupt_return_unsupported_result(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-ro.jsonl").touch()
    await _wait_until(lambda: "session-ro" in adapter.discover_sessions())

    send_result = await adapter.send_message("session-ro", "hello")
    assert send_result.operation == "send_message"
    assert "not supported" in send_result.reason

    interrupt_result = await adapter.interrupt("session-ro")
    assert interrupt_result.operation == "interrupt"
    assert "not supported" in interrupt_result.reason
