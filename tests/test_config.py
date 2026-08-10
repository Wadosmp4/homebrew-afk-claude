import stat

from companion.config import CompanionConfig, load_config, save_config


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
