"""Durable, enumerable registry of known SDK-owned sessions - replaces
session_settings.py's flat JSON file, which could only answer "what were
this known session's settings" (a point lookup by session_id), not "what
sessions does this companion know about at all" (enumeration). A restart
wipes SDKAdapter._sessions (plain in-memory dict); this is what lets a
previously-known session still show up in the phone's session list (see
daemon.py's _handle_list_active_sessions) before the phone has touched it
again, not just be resumable on demand.

A small local SQLite file, not the relay's Postgres - the companion runs on
the user's own machine while the relay runs on a separate server (see
relay/db.py's DATABASE_URL), so companion-side state can't live there
anyway. Same `~/.config/remote-claude-companion/` directory convention as
config.py/projects.py/session_settings.py. Python's stdlib `sqlite3` avoids
adding a new dependency; each call opens a short-lived connection and closes
it, matching the low-frequency access pattern (session start/settings-
change/end) - no long-lived connection held across the daemon's own
restarts or test runs, no WAL mode needed (single-process access, same as
the JSON file this replaces).

Unlike the JSON file, queries here read by session_id/status without
loading the whole table into memory, so there's no equivalent performance
reason to cap row count the way session_settings.py's MAX_SESSION_SETTINGS
did - retention/pruning of old 'ended' rows is a deliberately deferred
follow-up, not added here.
"""
from __future__ import annotations

import contextlib
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

DEFAULT_SESSION_REGISTRY_PATH = os.path.expanduser("~/.config/remote-claude-companion/session_registry.db")

# code-review finding: shared across the INSERT and both SELECT statements
# below so adding/renaming a column can't drift between them and
# _row_to_record's positional unpacking.
_COLUMNS = "session_id, cwd, model, auto_approve, llm_judge, status, created_at, updated_at"


@dataclass
class SessionRecord:
    session_id: str
    cwd: str
    model: Optional[str]
    auto_approve: bool
    llm_judge: bool
    status: str  # "active" | "ended"
    created_at: str
    updated_at: str


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            cwd TEXT NOT NULL,
            model TEXT,
            auto_approve INTEGER NOT NULL,
            llm_judge INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    return conn


def _row_to_record(row: tuple) -> SessionRecord:
    session_id, cwd, model, auto_approve, llm_judge, status, created_at, updated_at = row
    return SessionRecord(
        session_id=session_id,
        cwd=cwd,
        model=model,
        auto_approve=bool(auto_approve),
        llm_judge=bool(llm_judge),
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def upsert_session(
    session_id: str,
    *,
    cwd: str,
    model: Optional[str],
    auto_approve: bool,
    llm_judge: bool,
    path: Optional[str] = None,
) -> None:
    """Insert or update this session's row, always setting status='active'
    - even over a previously-'ended' row (U3's _try_resume_sdk_session is
    what guards against silently resurrecting an ended session; this
    function's job is just to record current settings, not enforce that
    policy). `created_at` is preserved across an update; `updated_at`
    always bumps to now."""
    path = path if path is not None else DEFAULT_SESSION_REGISTRY_PATH
    now = datetime.now(timezone.utc).isoformat()
    # code-review finding: `with conn:` only commits/rolls back the
    # transaction on exit - sqlite3.Connection's context-manager protocol
    # does not close() the connection. contextlib.closing is what actually
    # releases it, matching this module's own "short-lived, closed"
    # documented behavior above.
    with contextlib.closing(_connect(path)) as conn, conn:
        existing = conn.execute("SELECT created_at FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        created_at = existing[0] if existing is not None else now
        conn.execute(
            f"""
            INSERT INTO sessions ({_COLUMNS})
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cwd = excluded.cwd,
                model = excluded.model,
                auto_approve = excluded.auto_approve,
                llm_judge = excluded.llm_judge,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (session_id, cwd, model, int(auto_approve), int(llm_judge), created_at, now),
        )


def get_session(session_id: str, path: Optional[str] = None) -> Optional[SessionRecord]:
    path = path if path is not None else DEFAULT_SESSION_REGISTRY_PATH
    if not os.path.exists(path):
        return None
    with contextlib.closing(_connect(path)) as conn, conn:
        row = conn.execute(
            f"SELECT {_COLUMNS} FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def mark_session_ended(session_id: str, path: Optional[str] = None) -> None:
    """No-op (not an error) for an unknown session_id, matching
    SDKAdapter.set_session_auto_approve's own "unknown id is a silent
    no-op" convention elsewhere in this codebase."""
    path = path if path is not None else DEFAULT_SESSION_REGISTRY_PATH
    now = datetime.now(timezone.utc).isoformat()
    with contextlib.closing(_connect(path)) as conn, conn:
        conn.execute(
            "UPDATE sessions SET status = 'ended', updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def list_known_sessions(*, status: Optional[str] = None, path: Optional[str] = None) -> list[SessionRecord]:
    """Every known session, newest-first by creation. `status=None` returns
    both 'active' and 'ended' rows."""
    path = path if path is not None else DEFAULT_SESSION_REGISTRY_PATH
    if not os.path.exists(path):
        return []
    with contextlib.closing(_connect(path)) as conn, conn:
        if status is None:
            rows = conn.execute(f"SELECT {_COLUMNS} FROM sessions ORDER BY created_at DESC").fetchall()
        else:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM sessions WHERE status = ? ORDER BY created_at DESC",
                (status,),
            ).fetchall()
    return [_row_to_record(row) for row in rows]
