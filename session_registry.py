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
_COLUMNS = "session_id, cwd, model, permission_mode, status, created_at, updated_at"


@dataclass
class SessionRecord:
    session_id: str
    cwd: str
    model: Optional[str]
    permission_mode: Optional[str]
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
            permission_mode TEXT,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    # Permission-mode-picker plan (U2), KTD4: CREATE TABLE IF NOT EXISTS
    # above is a no-op against a `sessions` table this companion already
    # created on some earlier run - real installations have one on disk
    # under the OLD schema (auto_approve/llm_judge INTEGER NOT NULL
    # columns, no permission_mode column at all). This ALTER TABLE is the
    # only piece of this migration that actually reaches such a file; the
    # CREATE TABLE statement above only ever fires for a genuinely fresh
    # install with no `sessions` table yet. auto_approve/llm_judge
    # themselves are deliberately left in place on an upgraded table,
    # simply unused from here on - SQLite's ALTER TABLE DROP COLUMN
    # support is version-dependent/fragile, and R7 explicitly does not
    # require translating an old row's auto_approve/llm_judge into an
    # equivalent permission_mode: a session resuming under the old system
    # just loses that setting.
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
    if "permission_mode" not in existing_columns:
        conn.execute("ALTER TABLE sessions ADD COLUMN permission_mode TEXT")
    return conn


def _row_to_record(row: tuple) -> SessionRecord:
    session_id, cwd, model, permission_mode, status, created_at, updated_at = row
    return SessionRecord(
        session_id=session_id,
        cwd=cwd,
        model=model,
        permission_mode=permission_mode,
        status=status,
        created_at=created_at,
        updated_at=updated_at,
    )


def upsert_session(
    session_id: str,
    *,
    cwd: str,
    model: Optional[str],
    permission_mode: Optional[str],
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
        # A table migrated from the pre-permission_mode schema (KTD4, see
        # _connect()) still has its old auto_approve/llm_judge columns -
        # deliberately left in place, not dropped - but they're NOT NULL
        # with no default, unlike a freshly-created table, which never has
        # them at all. SQLite enforces NOT NULL while constructing the
        # candidate row for INSERT ... ON CONFLICT regardless of whether
        # the conflict resolution ends up redirecting into DO UPDATE - so
        # omitting them here would fail *every* upsert against a migrated
        # table, including one that's really just updating an existing
        # row's permission_mode. A fresh table must NOT reference columns
        # it was never given, or this would fail the other way around
        # ("no such column"). Detected per call (not cached) since this is
        # already a short-lived, per-call connection with no long-lived
        # state to stash it in - matching this module's own low-frequency
        # access pattern.
        table_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        legacy_columns = [name for name in ("auto_approve", "llm_judge") if name in table_columns]
        extra_columns = "".join(f", {name}" for name in legacy_columns)
        extra_values = ", 0" * len(legacy_columns)  # dummy - never read back, see SessionRecord/_COLUMNS
        conn.execute(
            f"""
            INSERT INTO sessions ({_COLUMNS}{extra_columns})
            VALUES (?, ?, ?, ?, 'active', ?, ?{extra_values})
            ON CONFLICT(session_id) DO UPDATE SET
                cwd = excluded.cwd,
                model = excluded.model,
                permission_mode = excluded.permission_mode,
                status = 'active',
                updated_at = excluded.updated_at
            """,
            (session_id, cwd, model, permission_mode, created_at, now),
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
