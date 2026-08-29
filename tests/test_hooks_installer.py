import json
import os

import pytest

from companion.hooks_installer import HOOK_EVENTS, _save_settings, install_hooks, uninstall_hooks


def test_install_creates_settings_file_with_all_hook_events(tmp_path):
    settings_path = str(tmp_path / ".claude" / "settings.json")
    install_hooks(settings_path, socket_path="/tmp/x.sock", python_bin="/usr/bin/python3")

    settings = json.loads((tmp_path / ".claude" / "settings.json").read_text())
    for event in HOOK_EVENTS:
        assert event in settings["hooks"]
        command = settings["hooks"][event][0]["hooks"][0]["command"]
        assert "companion.hook_bridge" in command
        assert event in command
        assert "/tmp/x.sock" in command


def test_install_is_idempotent(tmp_path):
    settings_path = str(tmp_path / "settings.json")
    install_hooks(settings_path, socket_path="/tmp/x.sock")
    install_hooks(settings_path, socket_path="/tmp/x.sock")

    settings = json.loads((tmp_path / "settings.json").read_text())
    for event in HOOK_EVENTS:
        assert len(settings["hooks"][event]) == 1


def test_install_merges_with_existing_unrelated_hooks(tmp_path):
    settings_path = tmp_path / "settings.json"
    existing = {
        "hooks": {
            "PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo audit"}]}],
            "SessionStart": [{"hooks": [{"type": "command", "command": "echo hello"}]}],
        },
        "otherSetting": True,
    }
    settings_path.write_text(json.dumps(existing))

    install_hooks(str(settings_path), socket_path="/tmp/x.sock")

    settings = json.loads(settings_path.read_text())
    assert settings["otherSetting"] is True
    # Pre-existing, unrelated hook entries survive untouched.
    assert settings["hooks"]["PreToolUse"] == existing["hooks"]["PreToolUse"]
    session_start_commands = [h["command"] for e in settings["hooks"]["SessionStart"] for h in e["hooks"]]
    assert "echo hello" in session_start_commands
    assert any("companion.hook_bridge" in c for c in session_start_commands)


def test_uninstall_removes_only_our_entries(tmp_path):
    settings_path = tmp_path / "settings.json"
    existing = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo hello"}]}]}}
    settings_path.write_text(json.dumps(existing))

    install_hooks(str(settings_path), socket_path="/tmp/x.sock")
    uninstall_hooks(str(settings_path))

    settings = json.loads(settings_path.read_text())
    commands = [h["command"] for e in settings["hooks"]["SessionStart"] for h in e["hooks"]]
    assert commands == ["echo hello"]
    for event in HOOK_EVENTS:
        if event != "SessionStart":
            assert event not in settings["hooks"]


# --- Code review fix: _save_settings writes atomically (temp file + rename)
# instead of truncating the target file in place, so a crash mid-write can't
# leave `.claude/settings.json` corrupt for other tooling to read. ----------


def test_save_settings_writes_valid_complete_json(tmp_path):
    settings_path = str(tmp_path / "settings.json")
    _save_settings(settings_path, {"hooks": {"SessionStart": []}})

    assert json.loads((tmp_path / "settings.json").read_text()) == {"hooks": {"SessionStart": []}}


def test_save_settings_leaves_no_tmp_file_behind_on_success(tmp_path):
    settings_path = str(tmp_path / "settings.json")
    _save_settings(settings_path, {"a": 1})

    assert not os.path.exists(settings_path + ".tmp")
    assert os.path.exists(settings_path)


def test_save_settings_does_not_touch_original_if_the_write_fails(tmp_path, monkeypatch):
    """Simulates a crash partway through writing the new content: the
    original file must be left completely untouched (not truncated), since
    the write happens to a `.tmp` sibling that's only ever renamed onto the
    real target after it's fully written."""
    settings_path = tmp_path / "settings.json"
    original = {"hooks": {"SessionStart": [{"hooks": [{"type": "command", "command": "echo original"}]}]}}
    settings_path.write_text(json.dumps(original))

    real_dump = json.dump

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr(json, "dump", _boom)

    with pytest.raises(RuntimeError):
        _save_settings(str(settings_path), {"hooks": {"SessionStart": []}})

    monkeypatch.setattr(json, "dump", real_dump)

    # The original target is untouched - still valid, still the old content
    # - because the write happened to a `.tmp` sibling that never got
    # renamed onto it.
    assert json.loads(settings_path.read_text()) == original
