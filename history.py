"""Read-only browsing of past Claude Code sessions for a project.

Unlike observe_adapter.py's live tailer, this reads a transcript once, in
full, on request - there's no streaming/incremental-offset concern here,
so the parsing below is a separate, simpler pass rather than reusing
_normalize_line's stateful, queue-emitting logic. It produces the same
normalized event shapes (assistant_message, tool_call, tool_result,
user_message) so the mobile client's existing EventFeed can render history
the same way it renders a live session - permission_request and
waiting_for_input never appear here, since those are hook-only concepts,
never written into the transcript itself.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .adapters.events import truncate_tool_result_content
from .projects import DEFAULT_PROJECTS_DIR

_CWD_SCAN_MAX_LINES = 50  # see companion/projects.py's identical constant for why line count, not bytes
_PREVIEW_MAX_CHARS = 140
_TAIL_SCAN_BYTES = 65536  # last ~64KB - enough to find a recent, valid timestamp without reading a huge file
# A real transcript on this machine can run 300+MB - reading one fully
# into memory is fine (line-by-line, like the live tailers), but returning
# every event as one WebSocket payload to the phone is not. Capped to the
# most recent events, matching truncate_tool_result_content's "truncate
# with a marker, don't silently drop everything" posture (U9/R14) - recent
# activity is what a "what happened in this session" browse actually needs.
_MAX_HISTORY_EVENTS = 2000


@dataclass(frozen=True)
class SessionSummary:
    session_id: str
    last_modified_at: str  # ISO8601 - the transcript's own last recorded activity timestamp, falling back to file mtime only if that's unparseable
    preview: Optional[str]  # first human message, truncated - for a readable list row


def list_project_sessions(project_path: str, projects_dir: Optional[str] = None) -> list[SessionSummary]:
    """Every past session transcript for `project_path`, newest first.

    Matches on the *resolved* form of both `project_path` and each
    transcript's own recorded `cwd` - the real `claude` CLI subprocess
    resolves symlinks before writing `cwd` into its transcript (confirmed
    empirically: passing cwd="/tmp/x" on macOS produces a transcript whose
    own `cwd` field reads "/private/tmp/x", since /tmp is itself a
    symlink), while callers of this function (daemon.py's
    _handle_list_project_sessions, sdk_adapter.py's session.cwd) may still
    be holding the unresolved form. A raw string comparison silently
    excludes a real, matching session whenever the two differ only by
    symlink resolution - exactly the "a session I was just in vanished
    from history" bug this fixes."""
    resolved_project_path = os.path.realpath(project_path)
    root = Path(projects_dir if projects_dir is not None else DEFAULT_PROJECTS_DIR)
    if not root.is_dir():
        return []

    summaries = []
    for project_dir in root.iterdir():
        if not project_dir.is_dir():
            continue
        for transcript in project_dir.glob("*.jsonl"):
            transcript_cwd = _read_cwd(transcript)
            if transcript_cwd is None or os.path.realpath(transcript_cwd) != resolved_project_path:
                continue
            summaries.append(_summarize(transcript))
    summaries.sort(key=lambda s: s.last_modified_at, reverse=True)
    return summaries


def _find_transcript(session_id: str, root: Path) -> Optional[Path]:
    """Session ids are unique UUIDs, so which project subdirectory holds
    this one doesn't need to be known ahead of time - shared by
    read_session_history and find_transcript_cwd."""
    return next(
        (p / f"{session_id}.jsonl" for p in root.iterdir() if p.is_dir() and (p / f"{session_id}.jsonl").is_file()),
        None,
    )


def find_transcript_cwd(session_id: str, projects_dir: Optional[str] = None) -> Optional[str]:
    """The `cwd` to reconnect an SDK-owned session with after a companion
    restart wiped it from SDKAdapter's in-memory state (see daemon.py's
    _try_resume_sdk_session) - None if no transcript with this session_id
    exists anywhere under projects_dir, or it never recorded a cwd."""
    root = Path(projects_dir if projects_dir is not None else DEFAULT_PROJECTS_DIR)
    if not root.is_dir():
        return None
    transcript = _find_transcript(session_id, root)
    if transcript is None:
        return None
    return _read_cwd(transcript)


def read_session_history(session_id: str, projects_dir: Optional[str] = None) -> Optional[list[dict]]:
    """Every normalized event for one past session, in order - None if no
    transcript with this session_id exists anywhere under projects_dir."""
    root = Path(projects_dir if projects_dir is not None else DEFAULT_PROJECTS_DIR)
    if not root.is_dir():
        return None

    transcript = _find_transcript(session_id, root)
    if transcript is None:
        return None

    events: list[dict] = []
    tool_started_at: dict[str, str] = {}
    try:
        with open(transcript, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.extend(_events_from_entry(entry, tool_started_at))
    except OSError:
        # The transcript could be deleted/rotated between discovery and
        # this on-demand read - matches _read_cwd/_summarize's own
        # OSError handling a few lines below, rather than propagating an
        # unhandled exception into daemon.py's action dispatch.
        return None

    if len(events) > _MAX_HISTORY_EVENTS:
        events = [{"type": "error", "timestamp": "", "data": {"message": f"History truncated - showing the most recent {_MAX_HISTORY_EVENTS} of {len(events)} events."}}] + events[-_MAX_HISTORY_EVENTS:]
    return events


def _read_cwd(transcript: Path) -> Optional[str]:
    try:
        with open(transcript, "rb") as f:
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
                    return cwd
    except OSError:
        return None
    return None


def _summarize(transcript: Path) -> SessionSummary:
    preview = None
    try:
        with open(transcript, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "user":
                    continue
                text = _first_text(entry.get("message", {}).get("content"))
                if text:
                    preview = text[:_PREVIEW_MAX_CHARS]
                    break
    except OSError:
        pass

    last_activity = _read_last_timestamp(transcript)
    if last_activity is None:
        # Fallback only - the filesystem's mtime can be skewed by anything
        # that touches the file without real new conversation content
        # (backups, sync tools, etc.), showing a "last used" time that
        # doesn't match when the conversation actually happened. The
        # transcript's own last recorded activity timestamp (above) is the
        # more trustworthy signal when it's parseable at all.
        try:
            last_activity = transcript.stat().st_mtime
        except OSError:
            last_activity = 0.0
    return SessionSummary(
        session_id=transcript.stem,
        last_modified_at=datetime.fromtimestamp(last_activity, tz=timezone.utc).isoformat(),
        preview=preview,
    )


def _read_last_timestamp(transcript: Path) -> Optional[float]:
    """The transcript's own last recorded activity time (epoch seconds),
    scanned from the tail of the file rather than reading it in full - a
    real transcript can run hundreds of MB, and this runs once per listed
    session, not just once per read. Only the last _TAIL_SCAN_BYTES are
    read; the first line found there may be a partial line cut off by the
    seek point, hence scanning backwards and skipping anything that
    doesn't parse instead of trusting the first line found."""
    try:
        size = transcript.stat().st_size
        with open(transcript, "rb") as f:
            f.seek(max(0, size - _TAIL_SCAN_BYTES))
            tail = f.read()
    except OSError:
        return None

    for raw_line in reversed(tail.split(b"\n")):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        timestamp = entry.get("timestamp")
        if not isinstance(timestamp, str):
            continue
        try:
            return datetime.fromisoformat(timestamp).timestamp()
        except ValueError:
            continue
    return None


def _first_text(content: Any) -> Optional[str]:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text")
    return None


def _events_from_entry(entry: dict, tool_started_at: dict[str, str]) -> list[dict]:
    entry_type = entry.get("type")
    timestamp = entry.get("timestamp") or ""
    content = entry.get("message", {}).get("content")
    if content is None:
        return []

    events: list[dict] = []
    if entry_type == "assistant":
        for block in content if isinstance(content, list) else []:
            if block.get("type") == "text":
                events.append({"type": "assistant_message", "timestamp": timestamp, "data": {"text": block["text"]}})
            elif block.get("type") == "tool_use":
                if timestamp:
                    tool_started_at[block["id"]] = timestamp
                events.append(
                    {
                        "type": "tool_call",
                        "timestamp": timestamp,
                        "data": {"tool_use_id": block["id"], "tool": block["name"], "input": block.get("input")},
                    }
                )
    elif entry_type == "user":
        if isinstance(content, str):
            events.append({"type": "user_message", "timestamp": timestamp, "data": {"text": content}})
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    events.append(
                        {
                            "type": "tool_result",
                            "timestamp": timestamp,
                            "data": {
                                "tool_use_id": tool_use_id,
                                "content": truncate_tool_result_content(block.get("content")),
                                "is_error": bool(block.get("is_error", False)),
                                "duration_ms": _duration_since(tool_started_at, tool_use_id, timestamp),
                            },
                        }
                    )
                elif block.get("type") == "text":
                    events.append({"type": "user_message", "timestamp": timestamp, "data": {"text": block["text"]}})
    return events


def _duration_since(tool_started_at: dict[str, str], tool_use_id: Optional[str], result_timestamp: str) -> Optional[float]:
    """Mirrors observe_adapter.py's _ObserveSession._duration_since (U9,
    R14) - duration here also comes from the JSONL entries' own timestamp
    fields, not a monotonic clock, since this reads a finished transcript
    rather than watching a live one."""
    started = tool_started_at.pop(tool_use_id, None) if tool_use_id else None
    if started is None or not result_timestamp:
        return None
    try:
        start = datetime.fromisoformat(started)
        end = datetime.fromisoformat(result_timestamp)
    except ValueError:
        return None
    return max((end - start).total_seconds() * 1000, 0.0)
