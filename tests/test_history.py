"""Tests for companion/history.py against real transcript fixtures - no
mocking of file I/O, per the project's convention for git_status.py's and
projects.py's tests.
"""
from __future__ import annotations

import json

import pytest

from companion.history import find_transcript_cwd, list_project_sessions, read_session_history


def _write_transcript(path, entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_list_project_sessions_finds_transcripts_matching_the_given_cwd(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )
    _write_transcript(
        projects_dir / "-b" / "session-2.jsonl",
        [{"type": "user", "cwd": "/Users/x/other", "message": {"role": "user", "content": "hi"}}],
    )

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert [s.session_id for s in found] == ["session-1"]


def test_list_project_sessions_finds_cwd_past_a_large_early_entry(tmp_path):
    """Same regression as companion/projects.py's identical case: a large
    early entry with no cwd must not push the real cwd-bearing line past
    the scan window."""
    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / "-a"
    project_dir.mkdir(parents=True)
    large_entry = {"type": "system", "text": "x" * 20000}
    with open(project_dir / "session-1.jsonl", "w") as f:
        f.write(json.dumps(large_entry) + "\n")
        f.write(json.dumps({"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}) + "\n")

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert [s.session_id for s in found] == ["session-1"]


def test_list_project_sessions_captures_first_user_message_as_preview(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [
            {"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "fix the login bug"}},
            {"type": "assistant", "cwd": "/Users/x/app", "message": {"role": "assistant", "content": [{"type": "text", "text": "sure"}]}},
        ],
    )

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert found[0].preview == "fix the login bug"


def test_list_project_sessions_truncates_a_long_preview(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    long_text = "x" * 500
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": long_text}}],
    )

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert len(found[0].preview) == 140


def test_list_project_sessions_orders_newest_first(tmp_path):
    import os
    import time

    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-old.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )
    old_path = projects_dir / "-a" / "session-old.jsonl"
    os.utime(old_path, (time.time() - 3600, time.time() - 3600))
    _write_transcript(
        projects_dir / "-a" / "session-new.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert [s.session_id for s in found] == ["session-new", "session-old"]


def test_last_modified_at_uses_the_transcripts_own_last_timestamp_not_file_mtime(tmp_path):
    """Regression test: a real session's "last used" time reading wrong -
    the filesystem's mtime can be skewed by anything that touches the file
    without real new conversation content (backups, sync tools, etc.), so
    it must not be the primary signal when the transcript's own last
    recorded activity timestamp is available and parseable."""
    import os
    import time

    projects_dir = tmp_path / "claude-projects"
    transcript = projects_dir / "-a" / "session-1.jsonl"
    _write_transcript(
        transcript,
        [
            {"type": "user", "cwd": "/Users/x/app", "timestamp": "2026-08-01T10:00:00.000Z", "message": {"role": "user", "content": "hi"}},
            {"type": "assistant", "timestamp": "2026-08-01T10:05:00.000Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]}},
        ],
    )
    # Simulate a filesystem mtime that has drifted far from the real
    # conversation activity - e.g. a backup tool touching the file.
    stale_mtime = time.time() - 3600 * 24 * 30
    os.utime(transcript, (stale_mtime, stale_mtime))

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert found[0].last_modified_at == "2026-08-01T10:05:00+00:00"


def test_last_modified_at_falls_back_to_file_mtime_with_no_parseable_timestamp(tmp_path):
    import os
    import time

    projects_dir = tmp_path / "claude-projects"
    transcript = projects_dir / "-a" / "session-1.jsonl"
    _write_transcript(
        transcript,
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],  # no timestamp field
    )
    known_mtime = time.time() - 3600
    os.utime(transcript, (known_mtime, known_mtime))

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    from datetime import datetime, timezone

    assert found[0].last_modified_at == datetime.fromtimestamp(known_mtime, tz=timezone.utc).isoformat()


def test_last_modified_at_finds_the_last_timestamp_past_a_large_entry_near_the_tail(tmp_path, monkeypatch):
    """Mirrors the existing cwd-scan regression test's shape, but for the
    tail-scan: a large entry sitting near (but not at) the very end must
    not push the real last timestamp outside the scanned window."""
    import companion.history as history_module

    monkeypatch.setattr(history_module, "_TAIL_SCAN_BYTES", 200)

    projects_dir = tmp_path / "claude-projects"
    transcript = projects_dir / "-a" / "session-1.jsonl"
    transcript.parent.mkdir(parents=True)
    with open(transcript, "w") as f:
        f.write(json.dumps({"type": "user", "cwd": "/Users/x/app", "timestamp": "2026-08-01T00:00:00.000Z", "message": {"role": "user", "content": "hi"}}) + "\n")
        f.write(json.dumps({"type": "system", "text": "x" * 5000}) + "\n")  # large, no timestamp field
        f.write(json.dumps({"type": "assistant", "timestamp": "2026-08-01T00:05:00.000Z", "message": {"role": "assistant", "content": []}}) + "\n")

    found = list_project_sessions("/Users/x/app", str(projects_dir))

    assert found[0].last_modified_at == "2026-08-01T00:05:00+00:00"


def test_list_project_sessions_matches_cwd_through_a_symlink(tmp_path):
    """Regression test: a real session vanishing from history purely from
    a path-string mismatch, not any real absence of history. The real
    `claude` CLI subprocess resolves symlinks before writing `cwd` into
    its transcript (confirmed empirically - macOS's own /tmp resolves to
    /private/tmp), but a caller here may still be holding the unresolved
    form (e.g. whatever path the phone originally sent to start the
    session) - a raw string comparison would silently drop a real,
    matching session."""
    projects_dir = tmp_path / "claude-projects"
    real_dir = tmp_path / "real-project"
    real_dir.mkdir()
    symlinked_dir = tmp_path / "project-via-symlink"
    symlinked_dir.symlink_to(real_dir)

    # The transcript records the *resolved* cwd, same as the real
    # subprocess does.
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": str(real_dir), "message": {"role": "user", "content": "hi"}}],
    )

    # The caller passes the *unresolved* (symlinked) form.
    found = list_project_sessions(str(symlinked_dir), str(projects_dir))

    assert [s.session_id for s in found] == ["session-1"]


def test_list_project_sessions_for_unknown_cwd_returns_empty(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )

    assert list_project_sessions("/Users/x/does-not-exist", str(projects_dir)) == []


def test_read_session_history_normalizes_the_full_conversation(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [
            {"type": "user", "cwd": "/Users/x/app", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": "add tests"}},
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "On it"},
                        {"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "pytest"}},
                    ],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "3 passed"}]},
            },
        ],
    )

    events = read_session_history("session-1", str(projects_dir))

    assert [e["type"] for e in events] == ["user_message", "assistant_message", "tool_call", "tool_result"]
    assert events[0]["data"]["text"] == "add tests"
    assert events[2]["data"]["tool"] == "Bash"
    assert events[3]["data"]["tool_use_id"] == "tool-1"
    assert events[3]["data"]["content"] == "3 passed"


def test_read_session_history_for_unknown_session_id_returns_none(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    projects_dir.mkdir()

    assert read_session_history("no-such-session", str(projects_dir)) is None


def test_find_transcript_cwd_locates_the_cwd_for_a_resumable_session(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )

    assert find_transcript_cwd("session-1", str(projects_dir)) == "/Users/x/app"


def test_find_transcript_cwd_for_unknown_session_id_returns_none(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    projects_dir.mkdir()

    assert find_transcript_cwd("no-such-session", str(projects_dir)) is None


def test_read_session_history_skips_unparseable_lines(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    transcript = projects_dir / "-a" / "session-1.jsonl"
    transcript.parent.mkdir(parents=True)
    with open(transcript, "w") as f:
        f.write("not json at all\n")
        f.write(json.dumps({"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}) + "\n")

    events = read_session_history("session-1", str(projects_dir))

    assert [e["type"] for e in events] == ["user_message"]


def test_read_session_history_computes_duration_ms_for_tool_results(tmp_path):
    """Regression test: tool_started_at was populated on tool_use but
    never read back - every historical tool_result silently had no
    duration_ms, unlike the live observe/SDK adapters' equivalent events."""
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [
            {
                "type": "assistant",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tool-1", "name": "Bash", "input": {"command": "pytest"}}],
                },
            },
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:02.5Z",
                "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "3 passed"}]},
            },
        ],
    )

    events = read_session_history("session-1", str(projects_dir))

    result_event = next(e for e in events if e["type"] == "tool_result")
    assert result_event["data"]["duration_ms"] == pytest.approx(2500, abs=1)


def test_read_session_history_truncates_when_over_the_event_cap(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    entries = [
        {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"role": "user", "content": f"turn {i}"}}
        for i in range(2100)
    ]
    _write_transcript(projects_dir / "-a" / "session-1.jsonl", entries)

    events = read_session_history("session-1", str(projects_dir))

    assert len(events) == 2001  # cap + one truncation marker
    assert events[0]["type"] == "error"
    assert "truncated" in events[0]["data"]["message"]
    assert events[-1]["data"]["text"] == "turn 2099"  # kept the most recent, not the oldest


def test_read_session_history_returns_none_if_the_transcript_becomes_unreadable(tmp_path, monkeypatch):
    projects_dir = tmp_path / "claude-projects"
    _write_transcript(
        projects_dir / "-a" / "session-1.jsonl",
        [{"type": "user", "cwd": "/Users/x/app", "message": {"role": "user", "content": "hi"}}],
    )

    real_open = open

    def _flaky_open(path, *args, **kwargs):
        if str(path).endswith("session-1.jsonl"):
            raise OSError("disappeared mid-read")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _flaky_open)

    assert read_session_history("session-1", str(projects_dir)) is None
