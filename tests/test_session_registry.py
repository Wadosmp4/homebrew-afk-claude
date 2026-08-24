"""Tests for companion/session_registry.py - a real local SQLite file, no
mocking of file I/O, per the project's convention for git_status.py's,
projects.py's, and session_settings.py's tests.
"""
from __future__ import annotations

import sqlite3

from companion.session_registry import (
    get_session,
    list_known_sessions,
    mark_session_ended,
    upsert_session,
)


def test_upsert_then_get_round_trips(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model="claude-opus-5", permission_mode="bypassPermissions", path=path)

    record = get_session("s1", path)

    assert record.session_id == "s1"
    assert record.cwd == "/tmp/app"
    assert record.model == "claude-opus-5"
    assert record.permission_mode == "bypassPermissions"
    assert record.status == "active"


def test_get_for_unknown_session_id_returns_none(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None, path=path)

    assert get_session("no-such-session", path) is None


def test_get_from_a_missing_database_returns_none(tmp_path):
    path = str(tmp_path / "does-not-exist.db")

    assert get_session("s1", path) is None


def test_upserting_again_updates_in_place_not_a_duplicate_row(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None, path=path)
    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode="acceptEdits", path=path)

    record = get_session("s1", path)
    assert record.permission_mode == "acceptEdits"
    assert len(list_known_sessions(path=path)) == 1


def test_upserting_again_preserves_created_at_but_bumps_updated_at(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None, path=path)
    first = get_session("s1", path)

    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode="acceptEdits", path=path)
    second = get_session("s1", path)

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_upsert_always_sets_status_active_even_over_an_ended_row(tmp_path):
    """Matches U3's own reasoning for why _try_resume_sdk_session must check
    status *before* attempting a resume: a bare upsert would otherwise
    silently resurrect an ended session the moment it's touched again."""
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None, path=path)
    mark_session_ended("s1", path)

    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None, path=path)

    assert get_session("s1", path).status == "active"


def test_mark_session_ended_flips_status_without_touching_other_fields(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model="claude-opus-5", permission_mode="bypassPermissions", path=path)

    mark_session_ended("s1", path)

    record = get_session("s1", path)
    assert record.status == "ended"
    assert record.cwd == "/tmp/app"
    assert record.model == "claude-opus-5"
    assert record.permission_mode == "bypassPermissions"


def test_mark_session_ended_for_unknown_session_id_is_a_no_op(tmp_path):
    path = str(tmp_path / "session_registry.db")

    mark_session_ended("no-such-session", path)  # must not raise

    assert get_session("no-such-session", path) is None


def test_list_known_sessions_filtered_by_status_excludes_the_other_status(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, permission_mode=None, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, permission_mode=None, path=path)
    mark_session_ended("s2", path)

    active = list_known_sessions(status="active", path=path)
    assert [r.session_id for r in active] == ["s1"]

    ended = list_known_sessions(status="ended", path=path)
    assert [r.session_id for r in ended] == ["s2"]


def test_list_known_sessions_with_no_filter_returns_both_statuses(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, permission_mode=None, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, permission_mode=None, path=path)
    mark_session_ended("s2", path)

    assert {r.session_id for r in list_known_sessions(path=path)} == {"s1", "s2"}


def test_list_known_sessions_from_a_missing_database_returns_empty_list(tmp_path):
    path = str(tmp_path / "does-not-exist.db")

    assert list_known_sessions(path=path) == []


def test_list_known_sessions_returns_newest_first(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, permission_mode=None, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, permission_mode=None, path=path)

    assert [r.session_id for r in list_known_sessions(path=path)] == ["s2", "s1"]


def test_defaults_to_the_process_wide_config_dir_when_no_path_given(tmp_path, monkeypatch):
    """Matches config.py/projects.py/session_settings.py's own convention:
    an explicit path argument is for tests; production code leaves it unset
    and gets DEFAULT_SESSION_REGISTRY_PATH."""
    import companion.session_registry as module

    monkeypatch.setattr(module, "DEFAULT_SESSION_REGISTRY_PATH", str(tmp_path / "session_registry.db"))

    upsert_session("s1", cwd="/tmp/app", model=None, permission_mode=None)

    assert get_session("s1").cwd == "/tmp/app"


def test_an_existing_database_with_the_old_schema_gains_permission_mode_without_losing_data(tmp_path):
    """Regression test for the additive migration inside _connect() (KTD4).

    A real installation's on-disk session_registry.db predates
    permission_mode entirely - this companion has been running for a
    while, so `CREATE TABLE IF NOT EXISTS` is a no-op against the
    already-existing table (it only ever fires for a brand-new file).
    _connect() must therefore ALTER TABLE the column onto a table that
    already exists, not just rely on the updated CREATE TABLE statement.

    This writes the OLD schema directly via a bare sqlite3 connection -
    bypassing this module entirely - to prove the migration path itself
    survives real pre-existing data, rather than only exercising a fresh
    empty database (which CREATE TABLE IF NOT EXISTS would already handle
    trivially, proving nothing about ALTER TABLE)."""
    path = str(tmp_path / "session_registry.db")
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE sessions (
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
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-session",
                "/tmp/pre-existing",
                "claude-opus-5",
                1,
                0,
                "active",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # The next real access to this file (not a fresh CREATE TABLE IF NOT
    # EXISTS - that path is a no-op here) must add permission_mode without
    # disturbing the pre-existing row, and R7 says the old row's
    # auto_approve/llm_judge values are NOT translated into a mode - the
    # migrated column reads as None (SDK default), not resurrected.
    record = get_session("old-session", path)
    assert record is not None
    assert record.session_id == "old-session"
    assert record.cwd == "/tmp/pre-existing"
    assert record.model == "claude-opus-5"
    assert record.status == "active"
    assert record.permission_mode is None

    # A write against the migrated table must also succeed - ALTER TABLE
    # alone isn't enough if _COLUMNS/the INSERT statement don't line up
    # with the now-7-column schema.
    upsert_session(
        "old-session", cwd="/tmp/pre-existing", model="claude-opus-5", permission_mode="acceptEdits", path=path
    )
    updated = get_session("old-session", path)
    assert updated.permission_mode == "acceptEdits"

    # Raw SQL check, independent of this module's own read path, that the
    # original row's data actually survived the migration (ALTER TABLE ADD
    # COLUMN, not some destructive drop-and-recreate) rather than being
    # lost.
    conn = sqlite3.connect(path)
    try:
        rows = conn.execute("SELECT session_id, cwd FROM sessions").fetchall()
    finally:
        conn.close()
    assert ("old-session", "/tmp/pre-existing") in rows
