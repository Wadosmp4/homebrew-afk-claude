"""Tests for companion/session_settings.py - a real local JSON file, no
mocking of file I/O, per the project's convention for git_status.py's and
projects.py's tests.
"""
from __future__ import annotations

import json

from companion.session_settings import (
    MAX_SESSION_SETTINGS,
    SessionSettings,
    load_session_settings,
    save_session_settings,
)


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "session_settings.json")
    save_session_settings("s1", SessionSettings(cwd="/tmp/app", model="claude-opus-5", auto_approve=True, llm_judge=True), path)

    loaded = load_session_settings("s1", path)

    assert loaded == SessionSettings(cwd="/tmp/app", model="claude-opus-5", auto_approve=True, llm_judge=True)


def test_load_for_unknown_session_id_returns_none(tmp_path):
    path = str(tmp_path / "session_settings.json")
    save_session_settings("s1", SessionSettings(cwd="/tmp/app"), path)

    assert load_session_settings("no-such-session", path) is None


def test_load_from_a_missing_file_returns_none(tmp_path):
    path = str(tmp_path / "does-not-exist.json")

    assert load_session_settings("s1", path) is None


def test_saving_again_overwrites_the_previous_record_for_that_session(tmp_path):
    path = str(tmp_path / "session_settings.json")
    save_session_settings("s1", SessionSettings(cwd="/tmp/app", auto_approve=False), path)
    save_session_settings("s1", SessionSettings(cwd="/tmp/app", auto_approve=True), path)

    loaded = load_session_settings("s1", path)

    assert loaded.auto_approve is True
    with open(path) as f:
        assert len(json.load(f)) == 1


def test_saving_two_different_sessions_keeps_both(tmp_path):
    path = str(tmp_path / "session_settings.json")
    save_session_settings("s1", SessionSettings(cwd="/tmp/a"), path)
    save_session_settings("s2", SessionSettings(cwd="/tmp/b"), path)

    assert load_session_settings("s1", path).cwd == "/tmp/a"
    assert load_session_settings("s2", path).cwd == "/tmp/b"


def test_saving_past_the_cap_drops_the_oldest_entries(tmp_path):
    path = str(tmp_path / "session_settings.json")
    for i in range(MAX_SESSION_SETTINGS + 10):
        save_session_settings(f"s{i}", SessionSettings(cwd=f"/tmp/{i}"), path)

    with open(path) as f:
        all_settings = json.load(f)
    assert len(all_settings) == MAX_SESSION_SETTINGS
    assert "s0" not in all_settings  # the oldest, dropped
    assert f"s{MAX_SESSION_SETTINGS + 9}" in all_settings  # the newest, kept


def test_defaults_to_the_process_wide_config_dir_when_no_path_given(tmp_path, monkeypatch):
    """Matches config.py/projects.py's own convention: an explicit path
    argument is for tests; production code leaves it unset and gets
    DEFAULT_SESSION_SETTINGS_PATH."""
    import companion.session_settings as module

    monkeypatch.setattr(module, "DEFAULT_SESSION_SETTINGS_PATH", str(tmp_path / "session_settings.json"))

    save_session_settings("s1", SessionSettings(cwd="/tmp/app"))

    assert load_session_settings("s1").cwd == "/tmp/app"
