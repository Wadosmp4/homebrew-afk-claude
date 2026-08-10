import json

from companion.hooks_installer import HOOK_EVENTS, install_hooks, uninstall_hooks


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
