"""Tests for companion/projects.py against a real filesystem fixture
shaped like ~/.claude/projects - no mocking of file I/O, per the
project's convention for git_status.py's tests.
"""
from __future__ import annotations

import json
import os
import time

from companion.projects import discover_projects, list_projects, record_recent


def _write_transcript(project_dir, name, cwd, lines=1) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    transcript = project_dir / f"{name}.jsonl"
    with open(transcript, "w") as f:
        for _ in range(lines):
            f.write(json.dumps({"type": "assistant", "cwd": cwd, "message": {}}) + "\n")


def test_discover_projects_reads_cwd_from_newest_transcript_not_dirname(tmp_path):
    """The real bug a naive dash-decode would hit: this directory name
    can't be reversed to the real path by splitting on '-' alone."""
    projects_dir = tmp_path / "claude-projects"
    real_project = tmp_path / "Projects" / "yt-music-organizer"
    real_project.mkdir(parents=True)
    _write_transcript(projects_dir / "-Users-x-Projects-yt-music-organizer", "session-1", str(real_project))

    found = discover_projects(str(projects_dir))

    assert [p.path for p in found] == [str(real_project)]


def test_discover_projects_dedupes_multiple_transcripts_for_the_same_project(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    real_project = tmp_path / "Projects" / "app"
    real_project.mkdir(parents=True)
    project_dir = projects_dir / "-Users-x-Projects-app"
    _write_transcript(project_dir, "session-1", str(real_project))
    _write_transcript(project_dir, "session-2", str(real_project))

    found = discover_projects(str(projects_dir))

    assert len(found) == 1


def test_discover_projects_orders_newest_touched_first(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    older_project = tmp_path / "older"
    newer_project = tmp_path / "newer"
    older_project.mkdir()
    newer_project.mkdir()

    _write_transcript(projects_dir / "-older", "s1", str(older_project))
    older_transcript = projects_dir / "-older" / "s1.jsonl"
    os.utime(older_transcript, (time.time() - 3600, time.time() - 3600))

    _write_transcript(projects_dir / "-newer", "s2", str(newer_project))

    found = discover_projects(str(projects_dir))

    assert [p.path for p in found] == [str(newer_project), str(older_project)]


def test_discover_projects_finds_cwd_past_a_large_early_entry(tmp_path):
    """Regression test: a real transcript can carry a large early entry
    (e.g. a system-prompt line) with no cwd, before the first line that
    actually has one - a byte-limited read of the file's start missed cwd
    entirely on a real transcript where that line didn't start until byte
    ~40000. Scanning by line count instead is robust to how big any one
    line is."""
    projects_dir = tmp_path / "claude-projects"
    real_project = tmp_path / "Projects" / "app"
    real_project.mkdir(parents=True)
    project_dir = projects_dir / "-Users-x-Projects-app"
    project_dir.mkdir(parents=True)
    large_entry = {"type": "system", "text": "x" * 20000}  # no cwd, larger than the old byte-limited scan window
    with open(project_dir / "session-1.jsonl", "w") as f:
        f.write(json.dumps(large_entry) + "\n")
        f.write(json.dumps({"type": "assistant", "cwd": str(real_project), "message": {}}) + "\n")

    found = discover_projects(str(projects_dir))

    assert [p.path for p in found] == [str(real_project)]


def test_discover_projects_skips_transcript_with_no_cwd_field(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    project_dir = projects_dir / "-Users-x-Projects-app"
    project_dir.mkdir(parents=True)
    (project_dir / "session-1.jsonl").write_text(json.dumps({"type": "assistant"}) + "\n")

    assert discover_projects(str(projects_dir)) == []


def test_discover_projects_skips_a_path_that_no_longer_exists_on_disk(tmp_path):
    """A transcript can outlive its working directory - e.g. an ephemeral
    git worktree that's since been cleaned up. Surfacing it would be a
    dead entry that fails silently when tapped (see daemon.py)."""
    projects_dir = tmp_path / "claude-projects"
    gone_path = str(tmp_path / "worktree-abc123" / "repo")  # never created

    _write_transcript(projects_dir / "-private-var-worktree-abc123-repo", "s1", gone_path)

    assert discover_projects(str(projects_dir)) == []


def test_discover_projects_on_missing_directory_returns_empty(tmp_path):
    assert discover_projects(str(tmp_path / "does-not-exist")) == []


def test_record_recent_persists_and_dedupes(tmp_path):
    recents_path = str(tmp_path / "recent_projects.json")

    record_recent("/Users/x/a", recents_path)
    record_recent("/Users/x/b", recents_path)
    record_recent("/Users/x/a", recents_path)  # re-starting "a" bumps it back to the front

    with open(recents_path) as f:
        assert json.load(f) == ["/Users/x/a", "/Users/x/b"]


def test_record_recent_caps_at_max_length(tmp_path):
    recents_path = str(tmp_path / "recent_projects.json")

    for i in range(25):
        record_recent(f"/Users/x/project-{i}", recents_path)

    with open(recents_path) as f:
        saved = json.load(f)
    assert len(saved) == 20
    assert saved[0] == "/Users/x/project-24"  # most recent first


def test_list_projects_pins_recents_first_then_other_known_projects(tmp_path):
    projects_dir = tmp_path / "claude-projects"
    recents_path = str(tmp_path / "recent_projects.json")
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()

    _write_transcript(projects_dir / "-a", "s1", str(project_a))
    _write_transcript(projects_dir / "-b", "s2", str(project_b))
    record_recent(str(project_b), recents_path)  # "b" started via Remote Claude - should come first

    result = list_projects(str(projects_dir), recents_path)

    assert [p["path"] for p in result] == [str(project_b), str(project_a)]
    assert result[0]["recent"] is True
    assert result[1]["recent"] is False


def test_list_projects_includes_a_recent_path_claude_code_has_no_record_of(tmp_path):
    """A path started via Remote Claude is still surfaced even if it has
    since fallen out of (or never had) a ~/.claude/projects transcript -
    unlike discover_projects, the recents list is never existence-checked
    (it's a plain "you were just here" breadcrumb, not a claim Claude Code
    itself vouches for)."""
    projects_dir = tmp_path / "claude-projects"
    recents_path = str(tmp_path / "recent_projects.json")
    record_recent("/Users/x/gone", recents_path)

    result = list_projects(str(projects_dir), recents_path)

    assert result == [{"path": "/Users/x/gone", "last_used_at": None, "recent": True}]
