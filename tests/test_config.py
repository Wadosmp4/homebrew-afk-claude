import json
import stat

from companion.config import CompanionConfig, load_config, save_config


def test_loading_a_config_file_from_before_observe_entrypoints_existed_still_works(tmp_path):
    path = str(tmp_path / "config.json")
    path_obj = tmp_path / "config.json"
    path_obj.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))

    loaded = load_config(path)

    assert loaded.observe_entrypoints == ["claude-desktop"]


def test_loading_a_config_file_from_before_cli_settings_existed_still_works(tmp_path):
    """U4/KTD5: cli_path/cli_env are new fields - a config.json written
    before they existed must still load under
    CompanionConfig(**json.load(f))'s strict field match, defaulting to
    cli_path=None and cli_env={} (never None)."""
    path = str(tmp_path / "config.json")
    path_obj = tmp_path / "config.json"
    path_obj.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))

    loaded = load_config(path)

    assert loaded.cli_path is None
    assert loaded.cli_env == {}


def test_cli_settings_round_trip_through_save_and_load(tmp_path):
    path = str(tmp_path / "config.json")
    config = CompanionConfig(
        relay_url="ws://127.0.0.1:9000/ws/companion",
        device_token="secret-token",
        cli_path="/usr/local/bin/claude-custom",
        cli_env={"ANTHROPIC_API_KEY": "sk-test"},
    )

    save_config(path, config)
    loaded = load_config(path)

    assert loaded.cli_path == "/usr/local/bin/claude-custom"
    assert loaded.cli_env == {"ANTHROPIC_API_KEY": "sk-test"}


def test_save_then_load_roundtrips(tmp_path):
    path = str(tmp_path / "config.json")
    config = CompanionConfig(relay_url="ws://127.0.0.1:9000/ws/companion", device_token="secret-token")

    save_config(path, config)
    loaded = load_config(path)

    assert loaded == config


def test_save_config_is_owner_only_readable(tmp_path):
    path = str(tmp_path / "config.json")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    mode = stat.S_IMODE(__import__("os").stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_loading_a_config_file_from_before_account_switch_existed_still_works(tmp_path):
    """U1 of the session-account-switch plan: active_account/personal_oauth_token
    are new fields - a config.json written before they existed must still load
    under CompanionConfig(**json.load(f))'s strict field match, defaulting to
    active_account="vscode" and personal_oauth_token=None."""
    path = str(tmp_path / "config.json")
    path_obj = tmp_path / "config.json"
    path_obj.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))

    loaded = load_config(path)

    assert loaded.active_account == "vscode"
    assert loaded.personal_oauth_token is None


def test_account_switch_fields_round_trip_through_save_and_load(tmp_path):
    path = str(tmp_path / "config.json")
    config = CompanionConfig(
        relay_url="ws://127.0.0.1:9000/ws/companion",
        device_token="secret-token",
        active_account="personal",
        personal_oauth_token="sk-ant-oat-test",
    )

    save_config(path, config)
    loaded = load_config(path)

    assert loaded.active_account == "personal"
    assert loaded.personal_oauth_token == "sk-ant-oat-test"


def test_save_config_stays_owner_only_readable_with_personal_oauth_token_set(tmp_path):
    path = str(tmp_path / "config.json")
    save_config(
        path,
        CompanionConfig(relay_url="ws://x/ws/companion", device_token="t", personal_oauth_token="sk-ant-oat-test"),
    )

    mode = stat.S_IMODE(__import__("os").stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR
