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
    adapter.open_session("session-dup")

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
async def test_content_events_are_not_forwarded_until_the_session_is_opened(adapter, tmp_path):
    """R: 'do not connect automatically to opened session, just show that
    it exists' - a discovered session's conversation content must not
    reach a phone that never asked to look at it. Only session_started/
    session_ended (the Sessions list's 'this exists' signal) forward
    unconditionally."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-unopened.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-unopened" in adapter.discover_sessions())

    events = adapter.subscribe("session-unopened")
    started = await events.__anext__()
    assert started.type == "session_started"

    _write_line(
        transcript,
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "should stay unforwarded"}]}},
    )
    await asyncio.sleep(FAST_POLL * 20)  # give the tail loop time to pick up and process the line
    # Never queued at all - checking the queue directly (rather than
    # racing events.__anext__() against a timeout) avoids cancelling the
    # subscribe() generator, which would leave it unusable afterward.
    assert adapter._sessions["session-unopened"].events.qsize() == 0

    # Opening now only surfaces *new* content from this point forward -
    # the earlier, unforwarded line never arrives.
    adapter.open_session("session-unopened")
    _write_line(
        transcript,
        {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "after opening"}]}},
    )
    next_event = await events.__anext__()
    assert next_event.type == "assistant_message"
    assert next_event.data["text"] == "after opening"


@pytest.mark.asyncio
async def test_session_started_is_re_announced_with_cwd_once_known_even_if_unopened(adapter, tmp_path):
    """The Sessions list needs a project name (R: 'name would be good to
    have to differentiate') even before the phone opens the session - cwd
    isn't known at the original session_started (mere file discovery), so
    it rides a second, always-forwarded session_started instead."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-named.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-named" in adapter.discover_sessions())

    events = adapter.subscribe("session-named")
    started = await events.__anext__()
    assert started.type == "session_started"
    assert started.data.get("cwd") is None

    _write_line(
        transcript,
        {"type": "user", "cwd": "/Users/x/my-repo", "message": {"role": "user", "content": "hi"}},
    )
    announced = await events.__anext__()
    assert announced.type == "session_started"
    assert announced.data["cwd"] == "/Users/x/my-repo"


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
    adapter.open_session("session-abc")  # R: content only forwards once opened

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
    adapter.open_session("session-timed")

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
    adapter.open_session("session-untimed")

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
    adapter.open_session("session-huge")

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
    adapter.open_session("session-partial")

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
async def test_respond_to_permission_emits_a_durable_resolution_event(adapter, tmp_path):
    """U1 (connection-resilience plan): parity with SDKAdapter - an
    observe-only session's manually-resolved permission is durable too,
    not only held in the answering phone's own memory."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-perm-resolved.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-perm-resolved" in adapter.discover_sessions())
    events = adapter.subscribe("session-perm-resolved")
    await events.__anext__()  # session_started

    hook_task = asyncio.create_task(
        _send_hook_async(
            adapter.socket_path,
            "PermissionRequest",
            {"session_id": "session-perm-resolved", "tool_use_id": "tool-9", "tool_name": "Bash", "tool_input": {}},
        )
    )
    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"

    await adapter.respond_to_permission("session-perm-resolved", "tool-9", "allow")
    await asyncio.wait_for(hook_task, timeout=2)

    resolved_event = await asyncio.wait_for(events.__anext__(), timeout=2)
    assert resolved_event.type == "permission_resolved"
    assert resolved_event.data["request_id"] == "tool-9"
    assert resolved_event.data["decision"] == "allow"


@pytest.mark.asyncio
async def test_auto_approve_policy_allows_an_allowlisted_command_without_a_phone_round_trip(tmp_path):
    """A terminal-started session's PermissionRequest hook gets the same
    treatment as a phone-started one once the observed-session policy is
    enabled (ObserveAdapter(auto_approve=True)) - the hook response comes
    back "allow" immediately, with no respond_to_permission call needed,
    for a command companion/auto_approve.py's allowlist covers."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-auto.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        auto_approve=True,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-auto" in a.discover_sessions())
        events = a.subscribe("session-auto")
        await events.__anext__()  # session_started (no cwd yet)

        # Content written before start() is treated as pre-existing history
        # and skipped (see test_preexisting_transcript_content_is_not_replayed)
        # - the cwd-bearing line has to land after discovery, like any other
        # "new" transcript content, and re-announces session_started once
        # picked up (note_cwd_known).
        _write_line(transcript, {"type": "assistant", "cwd": str(project_dir), "message": {"role": "assistant", "content": []}})
        await _wait_until(lambda: a.get_cwd("session-auto") == str(project_dir))
        await events.__anext__()  # session_started (cwd now known)

        response = await _send_hook_async(
            a.socket_path,
            "PermissionRequest",
            {"session_id": "session-auto", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "pytest"}},
        )
        assert response == {"permissionDecision": "allow", "permissionDecisionReason": ""}

        event = await events.__anext__()
        assert event.type == "permission_request"
        assert event.data["auto_approved"] is True
        assert event.data["judged_by"] == "policy"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_auto_approve_policy_still_blocks_a_denylisted_command_for_observed_sessions(tmp_path):
    """The denylist wins for observed sessions too - git push always
    prompts, regardless of the observed-session auto_approve setting."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-denylist.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        auto_approve=True,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-denylist" in a.discover_sessions())
        events = a.subscribe("session-denylist")
        await events.__anext__()  # session_started

        hook_task = asyncio.create_task(
            _send_hook_async(
                a.socket_path,
                "PermissionRequest",
                {"session_id": "session-denylist", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "git push"}},
            )
        )
        permission_event = await events.__anext__()
        assert permission_event.type == "permission_request"
        assert "auto_approved" not in permission_event.data

        await a.respond_to_permission("session-denylist", "tool-1", "allow")
        response = await asyncio.wait_for(hook_task, timeout=2)
        assert response["permissionDecision"] == "allow"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_auto_approve_off_by_default_still_prompts_for_observed_sessions(adapter, tmp_path):
    """The default ObserveAdapter() (as built by the `adapter` fixture,
    with no auto_approve kwarg) behaves exactly as before this feature -
    every permission request blocks on a phone response."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-default.jsonl"
    transcript.touch()
    await _wait_until(lambda: "session-default" in adapter.discover_sessions())
    events = adapter.subscribe("session-default")
    await events.__anext__()  # session_started

    hook_task = asyncio.create_task(
        _send_hook_async(
            adapter.socket_path,
            "PermissionRequest",
            {"session_id": "session-default", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "pytest"}},
        )
    )
    permission_event = await events.__anext__()
    assert permission_event.type == "permission_request"
    assert "auto_approved" not in permission_event.data

    await adapter.respond_to_permission("session-default", "tool-1", "allow")
    await asyncio.wait_for(hook_task, timeout=2)


@pytest.mark.asyncio
async def test_llm_judge_auto_approves_an_observed_session_command_on_a_safe_verdict(tmp_path, monkeypatch):
    """llm_judge is opt-in and separate from auto_approve, same as the
    phone-started path - a command the allowlist doesn't cover still gets
    auto-approved when the (faked) judge answers SAFE."""
    from claude_agent_sdk import AssistantMessage, TextBlock

    from companion import risk_judge

    async def fake_query(*, prompt, options):
        yield AssistantMessage(content=[TextBlock(text="SAFE: ordinary command")], model="claude")

    monkeypatch.setattr(risk_judge, "query", fake_query)

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-llm.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        auto_approve=True,
        llm_judge=True,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-llm" in a.discover_sessions())
        events = a.subscribe("session-llm")
        await events.__anext__()  # session_started

        response = await _send_hook_async(
            a.socket_path,
            "PermissionRequest",
            {"session_id": "session-llm", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "some-custom-script.sh"}},
        )
        assert response == {"permissionDecision": "allow", "permissionDecisionReason": ""}

        event = await events.__anext__()
        assert event.type == "permission_request"
        assert event.data["auto_approved"] is True
        assert event.data["judged_by"] == "llm"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_llm_judge_is_never_consulted_for_a_denylisted_observed_session_command(tmp_path, monkeypatch):
    from companion import risk_judge

    async def judge_should_not_be_called(*, prompt, options):
        raise AssertionError("the LLM judge must never be consulted for a denylisted command")
        yield  # pragma: no cover - makes this a generator function

    monkeypatch.setattr(risk_judge, "query", judge_should_not_be_called)

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-llm-deny.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        auto_approve=True,
        llm_judge=True,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-llm-deny" in a.discover_sessions())
        events = a.subscribe("session-llm-deny")
        await events.__anext__()  # session_started

        hook_task = asyncio.create_task(
            _send_hook_async(
                a.socket_path,
                "PermissionRequest",
                {"session_id": "session-llm-deny", "tool_use_id": "tool-1", "tool_name": "Bash", "tool_input": {"command": "git push"}},
            )
        )
        permission_event = await events.__anext__()
        assert permission_event.type == "permission_request"
        assert "auto_approved" not in permission_event.data

        await a.respond_to_permission("session-llm-deny", "tool-1", "allow")
        await asyncio.wait_for(hook_task, timeout=2)
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_auto_approve_never_resolves_a_structured_question_even_with_llm_judge_on(tmp_path, monkeypatch):
    """Regression: AskUserQuestion must always fall through to the
    blocking human-prompt path, even with both auto_approve and llm_judge
    enabled - "is this safe" is the wrong question for it (the real
    answer is the phone's chosen option, which only respond_to_permission
    can carry back). Before this guard, a SAFE verdict from the judge
    (or, in principle, a permissive is_auto_approvable) would silently
    resolve the question with a bare allow before the phone ever chose,
    breaking the "choosing options" feature entirely for observed
    sessions with auto-approve on."""
    from companion import risk_judge

    async def judge_always_says_safe(*, prompt, options):
        from claude_agent_sdk import AssistantMessage, TextBlock

        yield AssistantMessage(content=[TextBlock(text="SAFE: just a question")], model="claude")

    monkeypatch.setattr(risk_judge, "query", judge_always_says_safe)

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-question.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        auto_approve=True,
        llm_judge=True,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-question" in a.discover_sessions())
        events = a.subscribe("session-question")
        await events.__anext__()  # session_started

        question_input = {
            "questions": [{"question": "Which files?", "options": [{"label": "Last commit"}, {"label": "Last 5"}]}]
        }
        hook_task = asyncio.create_task(
            _send_hook_async(
                a.socket_path,
                "PermissionRequest",
                {
                    "session_id": "session-question",
                    "tool_use_id": "tool-1",
                    "tool_name": "AskUserQuestion",
                    "tool_input": question_input,
                },
            )
        )

        permission_event = await events.__anext__()
        assert permission_event.type == "permission_request"
        assert "auto_approved" not in permission_event.data
        assert not hook_task.done()  # still blocked - not silently auto-resolved

        await a.respond_to_permission("session-question", "tool-1", "allow", message="Last commit")
        response = await asyncio.wait_for(hook_task, timeout=2)
        assert response == {"permissionDecision": "allow", "permissionDecisionReason": "Last commit"}
    finally:
        await a.stop()


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

    compact_result = await adapter.compact("session-ro")
    assert compact_result.operation == "compact"
    assert "not supported" in compact_result.reason

    # U5: End Session isn't this phone's call to make for a session it's
    # only observing - server-side backstop alongside the mobile gate.
    disconnect_result = await adapter.disconnect("session-ro")
    assert disconnect_result.operation == "end_session"
    assert "not supported" in disconnect_result.reason


# --- Safeguards for watching the real ~/.claude/projects (not just test
# fixtures) - reproducing the CPU-storm incident at unit-test scale would
# mean tailing a real multi-MB history, so these instead assert the
# mechanism directly: existing content is skipped, old files are ignored,
# and the watched-session count is capped. -------------------------------


@pytest.mark.asyncio
async def test_preexisting_transcript_content_is_not_replayed(tmp_path):
    """Only new lines written after discovery show up - a transcript that
    already has megabytes of history by the time we notice it must not be
    read and emitted as if it just happened."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-old-content.jsonl"
    _write_line(transcript, {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "old, pre-existing turn"}]}})

    a = ObserveAdapter(projects_dir=str(projects_dir), socket_path=_short_socket_path(), watch_poll_interval=FAST_POLL, tail_poll_interval=FAST_POLL)
    await a.start()
    try:
        await _wait_until(lambda: "session-old-content" in a.discover_sessions())
        events = a.subscribe("session-old-content")
        started = await events.__anext__()
        assert started.type == "session_started"
        a.open_session("session-old-content")

        _write_line(transcript, {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "new turn"}]}})
        new_event = await events.__anext__()
        assert new_event.type == "assistant_message"
        assert new_event.data["text"] == "new turn"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_stale_transcript_is_never_watched(tmp_path):
    """A transcript untouched for longer than stale_after_seconds is old
    history, not a live session - real usage points this at ~/.claude/projects,
    which can hold months of dormant sessions across every project ever
    touched."""
    import os
    import time

    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-stale.jsonl"
    transcript.touch()
    old = time.time() - 1000
    os.utime(transcript, (old, old))

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        stale_after_seconds=500,
    )
    await a.start()
    try:
        await asyncio.sleep(FAST_POLL * 10)
        assert "session-stale" not in a.discover_sessions()
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_watched_session_count_is_capped(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-a.jsonl").touch()
    (project_dir / "session-b.jsonl").touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        max_watched_sessions=1,
    )
    await a.start()
    try:
        await _wait_until(lambda: len(a.discover_sessions()) >= 1)
        await asyncio.sleep(FAST_POLL * 10)
        assert len(a.discover_sessions()) == 1
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_a_session_from_a_client_not_in_required_entrypoints_is_never_watched(tmp_path):
    """~/.claude/projects is shared machine-wide across every client and
    project - required_entrypoints (R: 'give an ability to choose what
    clients to use') lets a session from an unrelated repo opened in a
    different client (e.g. the VS Code extension) stay out of the phone's
    Sessions list."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "other-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-vscode.jsonl"
    _write_line(transcript, {"type": "user", "entrypoint": "claude-vscode", "cwd": str(project_dir)})

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        required_entrypoints=frozenset({"claude-desktop"}),
    )
    await a.start()
    try:
        await asyncio.sleep(FAST_POLL * 10)
        assert "session-vscode" not in a.discover_sessions()
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_a_session_matching_one_of_several_required_entrypoints_is_watched(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-desktop.jsonl"
    _write_line(transcript, {"type": "user", "entrypoint": "claude-desktop", "cwd": str(project_dir)})

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        required_entrypoints=frozenset({"claude-desktop", "claude-vscode"}),
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-desktop" in a.discover_sessions())
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_no_required_entrypoints_means_no_filtering(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    transcript = project_dir / "session-anything.jsonl"
    _write_line(transcript, {"type": "user", "entrypoint": "sdk-ts", "cwd": str(project_dir)})

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
    )
    await a.start()
    try:
        await _wait_until(lambda: "session-anything" in a.discover_sessions())
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_set_required_entrypoints_changes_filtering_for_transcripts_discovered_after(tmp_path):
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
        required_entrypoints=frozenset({"claude-desktop"}),
    )
    await a.start()
    try:
        assert a.get_required_entrypoints() == frozenset({"claude-desktop"})
        a.set_required_entrypoints(frozenset({"claude-vscode"}))
        assert a.get_required_entrypoints() == frozenset({"claude-vscode"})

        transcript = project_dir / "session-vscode.jsonl"
        _write_line(transcript, {"type": "user", "entrypoint": "claude-vscode", "cwd": str(project_dir)})
        await _wait_until(lambda: "session-vscode" in a.discover_sessions())
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_is_active_reflects_a_discovered_sessions_live_state(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-a.jsonl").touch()
    await _wait_until(lambda: "session-a" in adapter.discover_sessions())

    assert adapter.is_active("session-a") is True

    await _send_hook_async(adapter.socket_path, "SessionEnd", {"session_id": "session-a"})

    assert adapter.is_active("session-a") is False


def test_is_active_for_unknown_session_returns_none(adapter):
    assert adapter.is_active("no-such-session") is None


@pytest.mark.asyncio
async def test_set_session_auto_approve_overrides_a_specific_sessions_snapshotted_state(adapter, tmp_path):
    """The per-session override lets the phone flip a single already-
    discovered session's behavior without touching the adapter-wide
    default or any other session - see ObserveAdapter.set_session_auto_approve."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-a.jsonl").touch()
    await _wait_until(lambda: "session-a" in adapter.discover_sessions())

    session = adapter._sessions["session-a"]
    assert session.auto_approve is False  # snapshotted from the adapter default (False)
    assert session.llm_judge is False

    assert adapter.set_session_auto_approve("session-a", auto_approve=True) is True
    assert session.auto_approve is True
    assert session.llm_judge is False  # untouched - only auto_approve was passed


@pytest.mark.asyncio
async def test_set_session_auto_approve_none_leaves_that_field_untouched(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-b.jsonl").touch()
    await _wait_until(lambda: "session-b" in adapter.discover_sessions())

    session = adapter._sessions["session-b"]
    adapter.set_session_auto_approve("session-b", auto_approve=True, llm_judge=True)
    assert session.auto_approve is True
    assert session.llm_judge is True

    # Passing None for auto_approve leaves it at its current value while
    # still changing llm_judge - same convention as daemon.py's
    # set_auto_approve_settings.
    adapter.set_session_auto_approve("session-b", auto_approve=None, llm_judge=False)
    assert session.auto_approve is True
    assert session.llm_judge is False


@pytest.mark.asyncio
async def test_set_session_auto_approve_only_affects_the_targeted_session(adapter, tmp_path):
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-c.jsonl").touch()
    (project_dir / "session-d.jsonl").touch()
    await _wait_until(lambda: "session-c" in adapter.discover_sessions() and "session-d" in adapter.discover_sessions())

    adapter.set_session_auto_approve("session-c", auto_approve=True)
    assert adapter._sessions["session-c"].auto_approve is True
    assert adapter._sessions["session-d"].auto_approve is False


@pytest.mark.asyncio
async def test_set_session_auto_approve_for_unknown_session_returns_false_without_raising(adapter):
    assert adapter.set_session_auto_approve("no-such-session", auto_approve=True) is False


@pytest.mark.asyncio
async def test_set_session_auto_approve_emits_a_confirmation_event_once_opened(adapter, tmp_path):
    """The confirmation event only reaches the phone once the session is
    opened (same forwarding rule as any other non-lifecycle event, see
    _ObserveSession.emit) - matches the real mobile flow, where the
    per-session toggle only appears inside an already-opened session."""
    project_dir = tmp_path / "projects" / "my-repo"
    project_dir.mkdir()
    (project_dir / "session-e.jsonl").touch()
    await _wait_until(lambda: "session-e" in adapter.discover_sessions())

    events = adapter.subscribe("session-e")
    started = await events.__anext__()
    assert started.type == "session_started"
    adapter.open_session("session-e")

    adapter.set_session_auto_approve("session-e", auto_approve=True, llm_judge=True)
    confirm = await events.__anext__()
    assert confirm.type == "session_auto_approve"
    assert confirm.data["auto_approve"] is True
    assert confirm.data["llm_judge"] is True


# --- U6 (Codex agent integration plan): observe-only Codex discovery -------
#
# A Codex session started outside the app (e.g. in a terminal) writes its
# transcript to ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl
# - structurally analogous to Claude Code's own ~/.claude/projects/*/*.jsonl,
# one directory level deeper for the year/month/day nesting. These tests use
# the same tmp_path fixture-directory-injection convention as the
# projects_dir tests above, via the new codex_sessions_dir constructor param.


@pytest.mark.asyncio
async def test_discovers_a_codex_rollout_transcript_and_tags_it_agent_codex(tmp_path):
    """R12/KTD5: an externally-started Codex session is discoverable the
    same way an externally-started Claude Code session already is - surfaced
    by discover_sessions() and tagged agent="codex" on its session_started
    event, mirroring how the live CodexAdapter tags its own sessions (see
    codex_adapter.py), for consistency with the mobile agent badge that
    reads this field."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    codex_sessions_dir = tmp_path / "codex-sessions"
    day_dir = codex_sessions_dir / "2026" / "08" / "26"
    day_dir.mkdir(parents=True)
    transcript = day_dir / "rollout-2026-08-26T12-00-00-abc123.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        codex_sessions_dir=str(codex_sessions_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
    )
    await a.start()
    try:
        session_id = "rollout-2026-08-26T12-00-00-abc123"
        await _wait_until(lambda: session_id in a.discover_sessions())

        events = a.subscribe(session_id)
        started = await events.__anext__()
        assert started.type == "session_started"
        assert started.data["agent"] == "codex"
        assert started.data["mode"] == "observe_only"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_codex_session_content_normalizes_user_and_assistant_messages(tmp_path):
    """Best-effort parsing of a rollout line's response_item wrapping
    (session's own comment on _normalize_codex_line explains this mapping
    is not hands-on verified against a real Codex install - none was
    available in this dev environment). This test pins down the shape this
    adapter expects so a future correction against a real transcript has a
    regression test to update alongside the parser."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    codex_sessions_dir = tmp_path / "codex-sessions"
    day_dir = codex_sessions_dir / "2026" / "08" / "26"
    day_dir.mkdir(parents=True)
    transcript = day_dir / "rollout-2026-08-26T12-00-00-def456.jsonl"
    transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        codex_sessions_dir=str(codex_sessions_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
    )
    await a.start()
    try:
        session_id = "rollout-2026-08-26T12-00-00-def456"
        await _wait_until(lambda: session_id in a.discover_sessions())

        events = a.subscribe(session_id)
        started = await events.__anext__()
        assert started.type == "session_started"
        a.open_session(session_id)

        _write_line(transcript, {"type": "session_meta", "payload": {"cwd": "/Users/x/codex-repo"}})
        announced = await events.__anext__()
        assert announced.type == "session_started"
        assert announced.data["cwd"] == "/Users/x/codex-repo"
        assert announced.data["agent"] == "codex"

        _write_line(
            transcript,
            {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}},
        )
        user_event = await events.__anext__()
        assert user_event.type == "user_message"
        assert user_event.data["text"] == "hi"

        _write_line(
            transcript,
            {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hello there"}]}},
        )
        assistant_event = await events.__anext__()
        assert assistant_event.type == "assistant_message"
        assert assistant_event.data["text"] == "hello there"

        _write_line(
            transcript,
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "call-1", "name": "shell", "arguments": {"command": "ls"}}},
        )
        call_event = await events.__anext__()
        assert call_event.type == "tool_call"
        assert call_event.data["tool"] == "shell"

        _write_line(
            transcript,
            {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call-1", "output": "file.txt"}},
        )
        result_event = await events.__anext__()
        assert result_event.type == "tool_result"
        assert result_event.data["tool_use_id"] == "call-1"
    finally:
        await a.stop()


@pytest.mark.asyncio
async def test_claude_code_and_codex_discovery_coexist_without_interfering(tmp_path):
    """Regression test: the Codex watch (codex_sessions_dir) is a second
    root watched alongside the existing ~/.claude/projects watch, not a
    replacement for it - both a Claude Code transcript and a Codex rollout
    transcript, dropped at the same time, must each be discovered and
    tagged correctly (Claude sessions carry no `agent` field at all, same
    as before this unit; only Codex sessions get agent="codex")."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    project_dir = projects_dir / "my-repo"
    project_dir.mkdir()
    claude_transcript = project_dir / "session-claude-1.jsonl"
    claude_transcript.touch()

    codex_sessions_dir = tmp_path / "codex-sessions"
    day_dir = codex_sessions_dir / "2026" / "08" / "26"
    day_dir.mkdir(parents=True)
    codex_transcript = day_dir / "rollout-2026-08-26T12-00-00-ghi789.jsonl"
    codex_transcript.touch()

    a = ObserveAdapter(
        projects_dir=str(projects_dir),
        codex_sessions_dir=str(codex_sessions_dir),
        socket_path=_short_socket_path(),
        watch_poll_interval=FAST_POLL,
        tail_poll_interval=FAST_POLL,
    )
    await a.start()
    try:
        claude_session_id = "session-claude-1"
        codex_session_id = "rollout-2026-08-26T12-00-00-ghi789"
        await _wait_until(
            lambda: claude_session_id in a.discover_sessions() and codex_session_id in a.discover_sessions()
        )

        claude_events = a.subscribe(claude_session_id)
        claude_started = await claude_events.__anext__()
        assert claude_started.type == "session_started"
        assert "agent" not in claude_started.data or claude_started.data["agent"] is None

        codex_events = a.subscribe(codex_session_id)
        codex_started = await codex_events.__anext__()
        assert codex_started.type == "session_started"
        assert codex_started.data["agent"] == "codex"

        # Content still flows correctly for the pre-existing Claude path too.
        a.open_session(claude_session_id)
        _write_line(
            claude_transcript,
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "still works"}]}},
        )
        claude_content = await claude_events.__anext__()
        assert claude_content.type == "assistant_message"
        assert claude_content.data["text"] == "still works"
    finally:
        await a.stop()
