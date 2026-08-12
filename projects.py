"""Project discovery for the Sessions screen's picker: every project
Claude Code has ever touched, plus this companion's own record of
projects recently started via Remote Claude specifically.

Discovery reads each `~/.claude/projects/<dir>/` directory's own newest
transcript for its `cwd` field, rather than decoding the directory's own
dash-encoded name (e.g. `-Users-x-Projects-yt-music-organizer`) - that
encoding is ambiguous whenever the real path itself contains a hyphen, as
that example does. Each directory costs one bounded read of its newest
transcript's first few KB (cwd is on every line) - a one-shot lookup per
directory, not the continuous tailing/watching ObserveAdapter does for
active sessions.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_PROJECTS_DIR = os.path.expanduser("~/.claude/projects")
DEFAULT_RECENTS_PATH = os.path.expanduser("~/.config/remote-claude-companion/recent_projects.json")
MAX_RECENTS = 20
# A byte-limited read of the file's start can miss cwd entirely: a single
# early entry (e.g. a large "system" prompt line) can run past any fixed
# byte budget before the first cwd-bearing line ever appears - confirmed
# on a real transcript where that line didn't start until byte ~40000.
# Scanning by line count instead is robust to how big any one entry is.
_CWD_SCAN_MAX_LINES = 50


@dataclass(frozen=True)
class KnownProject:
    path: str
    last_used_at: str  # ISO8601, from the transcript file's own mtime


def discover_projects(projects_dir: Optional[str] = None) -> list[KnownProject]:
    """Every real project path Claude Code has ever run in, newest first."""
    root = Path(projects_dir if projects_dir is not None else DEFAULT_PROJECTS_DIR)
    if not root.is_dir():
        return []

    by_path: dict[str, tuple[str, float]] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        found = _read_cwd(entry)
        if found is None:
            continue
        cwd, mtime = found
        if not os.path.isdir(cwd):
            # A transcript can outlive its working directory - e.g. an
            # ephemeral git worktree (ce-worktree, a sandboxed test run)
            # that's since been cleaned up. Surfacing it would just be a
            # dead entry that fails silently when tapped.
            continue
        existing = by_path.get(cwd)
        if existing is None or mtime > existing[1]:
            by_path[cwd] = (cwd, mtime)

    ordered = sorted(by_path.values(), key=lambda pair: pair[1], reverse=True)
    return [KnownProject(path=path, last_used_at=_iso(mtime)) for path, mtime in ordered]


def _read_cwd(project_dir: Path) -> Optional[tuple[str, float]]:
    """The newest transcript's `cwd` field, paired with that transcript's
    mtime (used as this project's "last touched" time) - every JSONL line
    Claude Code writes carries the session's cwd, so the first of the
    first `_CWD_SCAN_MAX_LINES` lines that has one is enough."""
    transcripts = sorted(project_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not transcripts:
        return None
    newest = transcripts[0]
    try:
        with open(newest, "rb") as f:
            for _ in range(_CWD_SCAN_MAX_LINES):
                line = f.readline()
                if not line:
                    break
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                cwd = entry.get("cwd")
                if isinstance(cwd, str) and cwd:
                    return cwd, newest.stat().st_mtime
    except OSError:
        return None
    return None


def _iso(mtime: float) -> str:
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def record_recent(path: str, recents_path: Optional[str] = None) -> None:
    """Bump `path` to the front of the recently-started-via-Remote-Claude
    list - called every time `start_session` fires (see daemon.py)."""
    recents_path = recents_path if recents_path is not None else DEFAULT_RECENTS_PATH
    recents = _load_recents(recents_path)
    recents = [p for p in recents if p != path]
    recents.insert(0, path)
    del recents[MAX_RECENTS:]
    os.makedirs(os.path.dirname(recents_path), exist_ok=True)
    with open(recents_path, "w") as f:
        json.dump(recents, f)


def _load_recents(recents_path: str) -> list[str]:
    try:
        with open(recents_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return []


def list_projects(projects_dir: Optional[str] = None, recents_path: Optional[str] = None) -> list[dict]:
    """The Sessions screen's picker list: paths recently started via
    Remote Claude pinned to the top (most recent first), then every other
    Claude-Code-known project by last-touched time."""
    recents_path = recents_path if recents_path is not None else DEFAULT_RECENTS_PATH
    recents = _load_recents(recents_path)
    discovered = {p.path: p for p in discover_projects(projects_dir)}

    result: list[dict] = []
    seen: set[str] = set()
    for path in recents:
        seen.add(path)
        known = discovered.get(path)
        result.append({"path": path, "last_used_at": known.last_used_at if known else None, "recent": True})
    for known in discovered.values():
        if known.path in seen:
            continue
        result.append({"path": known.path, "last_used_at": known.last_used_at, "recent": False})
    return result
