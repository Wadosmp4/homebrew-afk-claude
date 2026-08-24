"""Tests for companion/session_registry.py - a real local SQLite file, no
mocking of file I/O, per the project's convention for git_status.py's,
projects.py's, and session_settings.py's tests.
"""
from __future__ import annotations

from companion.session_registry import (
    get_session,
    list_known_sessions,
    mark_session_ended,
    upsert_session,
)


def test_upsert_then_get_round_trips(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model="claude-opus-5", auto_approve=True, llm_judge=True, path=path)

    record = get_session("s1", path)

    assert record.session_id == "s1"
    assert record.cwd == "/tmp/app"
    assert record.model == "claude-opus-5"
    assert record.auto_approve is True
    assert record.llm_judge is True
    assert record.status == "active"


def test_get_for_unknown_session_id_returns_none(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False, path=path)

    assert get_session("no-such-session", path) is None


def test_get_from_a_missing_database_returns_none(tmp_path):
    path = str(tmp_path / "does-not-exist.db")

    assert get_session("s1", path) is None


def test_upserting_again_updates_in_place_not_a_duplicate_row(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False, path=path)
    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=True, llm_judge=False, path=path)

    record = get_session("s1", path)
    assert record.auto_approve is True
    assert len(list_known_sessions(path=path)) == 1


def test_upserting_again_preserves_created_at_but_bumps_updated_at(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False, path=path)
    first = get_session("s1", path)

    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=True, llm_judge=False, path=path)
    second = get_session("s1", path)

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


def test_upsert_always_sets_status_active_even_over_an_ended_row(tmp_path):
    """Matches U3's own reasoning for why _try_resume_sdk_session must check
    status *before* attempting a resume: a bare upsert would otherwise
    silently resurrect an ended session the moment it's touched again."""
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False, path=path)
    mark_session_ended("s1", path)

    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False, path=path)

    assert get_session("s1", path).status == "active"


def test_mark_session_ended_flips_status_without_touching_other_fields(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/app", model="claude-opus-5", auto_approve=True, llm_judge=True, path=path)

    mark_session_ended("s1", path)

    record = get_session("s1", path)
    assert record.status == "ended"
    assert record.cwd == "/tmp/app"
    assert record.model == "claude-opus-5"
    assert record.auto_approve is True
    assert record.llm_judge is True


def test_mark_session_ended_for_unknown_session_id_is_a_no_op(tmp_path):
    path = str(tmp_path / "session_registry.db")

    mark_session_ended("no-such-session", path)  # must not raise

    assert get_session("no-such-session", path) is None


def test_list_known_sessions_filtered_by_status_excludes_the_other_status(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, auto_approve=False, llm_judge=False, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, auto_approve=False, llm_judge=False, path=path)
    mark_session_ended("s2", path)

    active = list_known_sessions(status="active", path=path)
    assert [r.session_id for r in active] == ["s1"]

    ended = list_known_sessions(status="ended", path=path)
    assert [r.session_id for r in ended] == ["s2"]


def test_list_known_sessions_with_no_filter_returns_both_statuses(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, auto_approve=False, llm_judge=False, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, auto_approve=False, llm_judge=False, path=path)
    mark_session_ended("s2", path)

    assert {r.session_id for r in list_known_sessions(path=path)} == {"s1", "s2"}


def test_list_known_sessions_returns_newest_first(tmp_path):
    path = str(tmp_path / "session_registry.db")
    upsert_session("s1", cwd="/tmp/a", model=None, auto_approve=False, llm_judge=False, path=path)
    upsert_session("s2", cwd="/tmp/b", model=None, auto_approve=False, llm_judge=False, path=path)

    assert [r.session_id for r in list_known_sessions(path=path)] == ["s2", "s1"]


def test_defaults_to_the_process_wide_config_dir_when_no_path_given(tmp_path, monkeypatch):
    """Matches config.py/projects.py/session_settings.py's own convention:
    an explicit path argument is for tests; production code leaves it unset
    and gets DEFAULT_SESSION_REGISTRY_PATH."""
    import companion.session_registry as module

    monkeypatch.setattr(module, "DEFAULT_SESSION_REGISTRY_PATH", str(tmp_path / "session_registry.db"))

    upsert_session("s1", cwd="/tmp/app", model=None, auto_approve=False, llm_judge=False)

    assert get_session("s1").cwd == "/tmp/app"
