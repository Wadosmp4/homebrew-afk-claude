"""Tests for companion/cli.py - the argparse wiring and the `configure`
subcommand's config-writing behavior. `run` just delegates straight to
companion.daemon.main (already covered by that module's own tests), so it's
verified here only via a monkeypatched call-through, not a real daemon run.

`pair`'s tests monkeypatch `_request_companion_pairing`/`_poll_claim`
directly (rather than faking urllib/HTTP) - those two functions are the
seam deliberately kept small for exactly this purpose (see cli.py's own
comment on them).
"""
from __future__ import annotations

import json
import os
import socket
import threading
import urllib.error
import uuid

import pytest

import companion.cli as cli
from companion.cli import main
from companion.config import load_config


def test_configure_writes_a_config_file_that_is_owner_only_readable(tmp_path):
    """Companion-Local Token Entry & SQLite Config plan, KTD4: the db file
    `configure` writes to is owner-only readable from the moment it's
    created - see config.py's own module docstring. The written content
    itself round-trips through load_config, checked separately below."""
    import stat

    config_path = tmp_path / "session_registry.db"
    main(
        [
            "configure",
            "--relay-url",
            "wss://relay.example.com/ws/companion",
            "--device-token",
            "tok-abc123",
            "--config-path",
            str(config_path),
        ]
    )

    assert config_path.exists()
    mode = stat.S_IMODE(config_path.stat().st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_configure_written_config_round_trips_through_load_config(tmp_path):
    config_path = tmp_path / "config.json"
    main(
        [
            "configure",
            "--relay-url",
            "wss://relay.example.com/ws/companion",
            "--device-token",
            "tok-abc123",
            "--config-path",
            str(config_path),
        ]
    )

    loaded = load_config(str(config_path))
    assert loaded.relay_url == "wss://relay.example.com/ws/companion"
    assert loaded.device_token == "tok-abc123"


def test_configure_defaults_observe_auto_approve_and_llm_judge_off(tmp_path):
    config_path = tmp_path / "config.json"
    main(
        [
            "configure",
            "--relay-url",
            "wss://relay.example.com/ws/companion",
            "--device-token",
            "tok-abc123",
            "--config-path",
            str(config_path),
        ]
    )

    loaded = load_config(str(config_path))
    assert loaded.observe_auto_approve is False
    assert loaded.observe_llm_judge is False


def test_run_delegates_to_the_daemon_main_with_the_given_config_path(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "companion.cli.run_daemon",
        lambda config_path=None, status_socket_path=None: calls.append((config_path, status_socket_path)),
    )

    config_path = str(tmp_path / "config.json")
    main(["run", "--config-path", config_path])

    assert calls == [(config_path, None)]


def test_run_defaults_to_the_standard_config_path(monkeypatch):
    from companion.config import DEFAULT_SESSION_REGISTRY_PATH

    calls = []
    monkeypatch.setattr(
        "companion.cli.run_daemon",
        lambda config_path=None, status_socket_path=None: calls.append((config_path, status_socket_path)),
    )

    main(["run"])

    assert calls == [(DEFAULT_SESSION_REGISTRY_PATH, None)]


def test_run_forwards_an_explicit_status_socket_path(monkeypatch, tmp_path):
    """Code review finding (correctness, U3): `run` had no way to point a
    second daemon instance at a different status socket, unlike `precheck`
    which already accepts one - two daemons started with different
    --config-path values (e.g. a dev instance alongside a real one) would
    collide on the same default status.sock, with the second silently
    stealing it (see status_socket_path's own constructor comment in
    companion/daemon.py)."""
    calls = []
    monkeypatch.setattr(
        "companion.cli.run_daemon",
        lambda config_path=None, status_socket_path=None: calls.append((config_path, status_socket_path)),
    )

    from companion.config import DEFAULT_SESSION_REGISTRY_PATH

    status_socket_path = str(tmp_path / "status.sock")
    main(["run", "--status-socket-path", status_socket_path])

    assert calls == [(DEFAULT_SESSION_REGISTRY_PATH, status_socket_path)]


def test_no_subcommand_is_a_usage_error():
    with pytest.raises(SystemExit):
        main([])


# --- pair ------------------------------------------------------------------


@pytest.mark.parametrize(
    "https_url,expected_ws_url",
    [
        ("https://relay.example.com", "wss://relay.example.com/ws/companion"),
        ("https://relay.example.com/", "wss://relay.example.com/ws/companion"),
        ("http://localhost:8765", "ws://localhost:8765/ws/companion"),
    ],
)
def test_derive_ws_url_converts_the_http_scheme(https_url, expected_ws_url):
    assert cli._derive_ws_url(https_url) == expected_ws_url


def test_derive_ws_url_rejects_a_non_http_scheme():
    with pytest.raises(ValueError):
        cli._derive_ws_url("ftp://relay.example.com")


def test_pair_writes_config_once_approved(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cli, "_request_companion_pairing", lambda relay_url, label: {"code": "482913", "expires_at": "later"}
    )
    responses = iter([{"status": "pending"}, {"status": "approved", "device_id": "d1", "token": "tok-xyz"}])
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: next(responses))

    config_path = tmp_path / "config.json"
    main(
        [
            "pair",
            "--relay-url",
            "https://relay.example.com",
            "--poll-interval",
            "0.01",
            "--timeout",
            "5",
            "--config-path",
            str(config_path),
        ]
    )

    loaded = load_config(str(config_path))
    assert loaded.relay_url == "wss://relay.example.com/ws/companion"
    assert loaded.device_token == "tok-xyz"


def test_pair_sends_the_requested_label(monkeypatch, tmp_path):
    captured = {}

    def fake_request(relay_url, label):
        captured["label"] = label
        return {"code": "111111", "expires_at": "later"}

    monkeypatch.setattr(cli, "_request_companion_pairing", fake_request)
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: {"status": "approved", "device_id": "d1", "token": "t"})

    main(
        [
            "pair",
            "--relay-url",
            "https://relay.example.com",
            "--label",
            "MacBook Pro",
            "--config-path",
            str(tmp_path / "config.json"),
        ]
    )

    assert captured["label"] == "MacBook Pro"


def test_pair_defaults_the_label_to_the_hostname(monkeypatch, tmp_path):
    import socket

    captured = {}

    def fake_request(relay_url, label):
        captured["label"] = label
        return {"code": "111111", "expires_at": "later"}

    monkeypatch.setattr(cli, "_request_companion_pairing", fake_request)
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: {"status": "approved", "device_id": "d1", "token": "t"})

    main(["pair", "--relay-url", "https://relay.example.com", "--config-path", str(tmp_path / "config.json")])

    assert captured["label"] == socket.gethostname()


def test_pair_times_out_without_writing_a_config_if_never_approved(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_request_companion_pairing", lambda relay_url, label: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: {"status": "pending"})

    config_path = tmp_path / "config.json"
    main(
        [
            "pair",
            "--relay-url",
            "https://relay.example.com",
            "--timeout",
            "0.05",
            "--poll-interval",
            "0.01",
            "--config-path",
            str(config_path),
        ]
    )

    assert not config_path.exists()
    assert "Timed out" in capsys.readouterr().out


def test_pair_stops_cleanly_on_an_http_error_requesting_a_code(monkeypatch, tmp_path, capsys):
    def raise_error(relay_url, label):
        raise urllib.error.HTTPError(relay_url, 429, "rate limited", None, None)

    monkeypatch.setattr(cli, "_request_companion_pairing", raise_error)

    config_path = tmp_path / "config.json"
    main(["pair", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    assert not config_path.exists()
    assert "Could not request a pairing code" in capsys.readouterr().out


def test_pair_stops_cleanly_on_an_http_error_while_polling(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_request_companion_pairing", lambda relay_url, label: {"code": "111111", "expires_at": "later"})

    def raise_error(relay_url, code):
        raise urllib.error.HTTPError(relay_url, 400, "invalid_code", None, None)

    monkeypatch.setattr(cli, "_poll_claim", raise_error)

    config_path = tmp_path / "config.json"
    main(["pair", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    assert not config_path.exists()
    assert "Pairing failed" in capsys.readouterr().out


# --- pairing-code ------------------------------------------------------------
#
# The first phone's onboarding step: this companion already has its own
# device token (from `pair`/`configure`), so it authenticates
# POST /pairing/register itself instead of requiring a raw curl call.


@pytest.mark.parametrize(
    "ws_url,expected_http_url",
    [
        ("wss://relay.example.com/ws/companion", "https://relay.example.com"),
        ("wss://relay.example.com/ws/companion/", "https://relay.example.com"),
        ("ws://localhost:8765/ws/companion", "http://localhost:8765"),
    ],
)
def test_derive_http_url_converts_the_ws_scheme(ws_url, expected_http_url):
    assert cli._derive_http_url(ws_url) == expected_http_url


def test_derive_http_url_rejects_a_non_ws_scheme():
    with pytest.raises(ValueError):
        cli._derive_http_url("ftp://relay.example.com/ws/companion")


def _write_config(config_path, relay_url="wss://relay.example.com/ws/companion", device_token="tok-abc123"):
    """Companion-Local Token Entry & SQLite Config plan: `config_path` is a
    SQLite db path now (config.py's own save_config/load_config), not a
    JSON file - writing raw JSON here would leave a file `load_config`
    can't open as a database at all. Writes through the real save_config
    so these tests simulate "an already-configured companion" under the
    current storage scheme, not the pre-migration one."""
    from companion.config import CompanionConfig, save_config

    save_config(str(config_path), CompanionConfig(relay_url=relay_url, device_token=device_token))


def test_pairing_code_prints_the_relay_url_and_code(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_request(relay_url, device_token):
        captured["relay_url"] = relay_url
        captured["device_token"] = device_token
        return {"code": "482913", "expires_at": "later"}

    monkeypatch.setattr(cli, "_request_phone_pairing_code", fake_request)

    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(["pairing-code", "--config-path", str(config_path)])

    assert captured["relay_url"] == "https://relay.example.com"
    assert captured["device_token"] == "tok-abc123"
    out = capsys.readouterr().out
    assert "482913" in out
    assert "https://relay.example.com" in out


def test_pairing_code_with_no_config_prints_guidance_instead_of_crashing(tmp_path, capsys):
    main(["pairing-code", "--config-path", str(tmp_path / "does-not-exist.json")])

    out = capsys.readouterr().out
    assert "pair" in out or "configure" in out


def test_pairing_code_with_a_corrupted_db_prints_guidance_instead_of_crashing(tmp_path, capsys):
    """code-review finding: load_config's SQLite backing store can raise
    sqlite3.Error (a corrupted file, e.g. truncated or overwritten by
    something else), not just OSError/ValueError/TypeError - this must
    produce the same friendly message as a missing config, not a raw
    traceback."""
    config_path = tmp_path / "config.json"
    config_path.write_bytes(b"not a valid sqlite database, just garbage bytes")

    main(["pairing-code", "--config-path", str(config_path)])

    out = capsys.readouterr().out
    assert "Could not read config" in out
    assert "setup" in out


def test_pairing_code_stops_cleanly_on_an_http_error(monkeypatch, tmp_path, capsys):
    def raise_error(relay_url, device_token):
        raise urllib.error.HTTPError(relay_url, 401, "invalid_companion_token", None, None)

    monkeypatch.setattr(cli, "_request_phone_pairing_code", raise_error)

    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(["pairing-code", "--config-path", str(config_path)])

    assert "Could not request a pairing code" in capsys.readouterr().out


# --- setup -------------------------------------------------------------
#
# The one command for any Mac, in any state - already configured, an
# additional Mac joining an already-paired phone, or the very first Mac
# ever.


def _fake_input(*responses):
    """Each successive input() call returns the next response in order -
    `setup`'s unconfigured branch asks a y/n question, then (only on
    "yes") a second question for the actual bootstrap code."""
    it = iter(responses)
    return lambda prompt: next(it)


def test_setup_on_an_already_configured_mac_just_prints_a_phone_pairing_code(monkeypatch, tmp_path, capsys):
    captured = {}

    def fake_request(relay_url, device_token):
        captured["relay_url"] = relay_url
        captured["device_token"] = device_token
        return {"code": "482913", "expires_at": "later"}

    monkeypatch.setattr(cli, "_request_phone_pairing_code", fake_request)

    config_path = tmp_path / "config.json"
    _write_config(config_path)
    main(["setup", "--config-path", str(config_path)])

    assert captured["relay_url"] == "https://relay.example.com"
    assert captured["device_token"] == "tok-abc123"
    assert "482913" in capsys.readouterr().out


def test_setup_with_a_bootstrap_code_claims_it_then_prints_a_phone_code(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_claim_bootstrap_code", lambda relay_url, code: {"device_id": "d1", "token": "tok-new"})
    monkeypatch.setattr(cli, "_request_phone_pairing_code", lambda relay_url, device_token: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr("builtins.input", _fake_input("y", "555444"))

    config_path = tmp_path / "config.json"
    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    loaded = load_config(str(config_path))
    assert loaded.relay_url == "wss://relay.example.com/ws/companion"
    assert loaded.device_token == "tok-new"
    out = capsys.readouterr().out
    assert "111111" in out


def test_setup_sends_the_bootstrap_code_the_user_typed(monkeypatch, tmp_path):
    captured = {}

    def fake_claim(relay_url, code):
        captured["code"] = code
        return {"device_id": "d1", "token": "tok-new"}

    monkeypatch.setattr(cli, "_claim_bootstrap_code", fake_claim)
    monkeypatch.setattr(cli, "_request_phone_pairing_code", lambda relay_url, device_token: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr("builtins.input", _fake_input("y", "  999888  "))

    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(tmp_path / "config.json")])

    assert captured["code"] == "999888"


def test_setup_without_a_bootstrap_code_runs_the_phone_approval_flow_instead(monkeypatch, tmp_path):
    captured = {}

    def fake_request(relay_url, label):
        captured["label"] = label
        return {"code": "111111", "expires_at": "later"}

    monkeypatch.setattr(cli, "_request_companion_pairing", fake_request)
    monkeypatch.setattr(
        cli, "_poll_claim", lambda relay_url, code: {"status": "approved", "device_id": "d1", "token": "tok-xyz"}
    )
    monkeypatch.setattr("builtins.input", _fake_input("n"))

    config_path = tmp_path / "config.json"
    main(["setup", "--relay-url", "https://relay.example.com", "--label", "MacBook Pro", "--config-path", str(config_path)])

    assert captured["label"] == "MacBook Pro"
    loaded = load_config(str(config_path))
    assert loaded.relay_url == "wss://relay.example.com/ws/companion"
    assert loaded.device_token == "tok-xyz"


def test_setup_without_a_bootstrap_code_never_claims_one(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_request_companion_pairing", lambda relay_url, label: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: {"status": "approved", "device_id": "d1", "token": "tok-xyz"})

    def fail_if_called(relay_url, code):
        raise AssertionError("setup should not ask for/claim a bootstrap code when the user said no")

    monkeypatch.setattr(cli, "_claim_bootstrap_code", fail_if_called)
    monkeypatch.setattr("builtins.input", _fake_input("N"))

    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(tmp_path / "config.json")])


def test_setup_without_a_bootstrap_code_times_out_without_writing_a_config(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_request_companion_pairing", lambda relay_url, label: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr(cli, "_poll_claim", lambda relay_url, code: {"status": "pending"})
    monkeypatch.setattr(cli, "DEFAULT_PAIR_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(cli, "DEFAULT_PAIR_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("builtins.input", _fake_input("n"))

    config_path = tmp_path / "config.json"
    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    # A timed-out approval must not persist a config row. The db file
    # itself may still exist - `load_config`'s own "is this Mac already
    # configured" check opens a connection, and sqlite3 creates the file
    # on connect even when nothing is ever inserted into it (unlike the
    # old config.json, which only existed once actually written).
    with pytest.raises(FileNotFoundError):
        load_config(str(config_path))
    assert "Timed out" in capsys.readouterr().out


def test_setup_falls_back_to_the_afk_relay_url_env_var(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DEFAULT_RELAY_URL", "https://relay.example.com")
    monkeypatch.setattr(cli, "_claim_bootstrap_code", lambda relay_url, code: {"device_id": "d1", "token": "tok-new"})
    monkeypatch.setattr(cli, "_request_phone_pairing_code", lambda relay_url, device_token: {"code": "111111", "expires_at": "later"})
    monkeypatch.setattr("builtins.input", _fake_input("y", "555444"))

    config_path = tmp_path / "config.json"
    main(["setup", "--config-path", str(config_path)])

    loaded = load_config(str(config_path))
    assert loaded.relay_url == "wss://relay.example.com/ws/companion"


def test_setup_with_no_address_and_no_invitation_points_to_the_app_instead_of_crashing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DEFAULT_RELAY_URL", None)
    monkeypatch.setattr("builtins.input", _fake_input("n"))

    main(["setup", "--config-path", str(tmp_path / "does-not-exist.json")])

    assert "Add a computer" in capsys.readouterr().out


# --- invitations: the address travels inside the code ----------------------------------


def test_setup_with_an_invitation_carries_its_own_address_and_claims_its_code(monkeypatch, tmp_path, capsys):
    from companion.invite import make_invite

    monkeypatch.setattr(cli, "DEFAULT_RELAY_URL", None)
    seen = {}

    def claim(relay_url, code):
        seen["claim"] = (relay_url, code)
        return {"token": "device-token"}

    monkeypatch.setattr(cli, "_claim_bootstrap_code", claim)
    monkeypatch.setattr(cli, "_print_phone_pairing_code", lambda url, token: print("PHONE CODE"))
    monkeypatch.setattr("builtins.input", _fake_input("y", make_invite("https://relay.example.com", "123456")))

    main(["setup", "--config-path", str(tmp_path / "config.json")])

    assert seen["claim"] == ("https://relay.example.com", "123456")
    assert "PHONE CODE" in capsys.readouterr().out
    assert cli.load_config(str(tmp_path / "config.json")).relay_url == "wss://relay.example.com/ws/companion"


def test_setup_invite_option_without_a_code_goes_straight_to_the_phone_approval(monkeypatch, tmp_path):
    from companion.invite import make_invite

    monkeypatch.setattr(cli, "DEFAULT_RELAY_URL", None)
    asked = {}

    def approval(relay_url, label, timeout, poll_interval, config_path):
        asked["relay_url"] = relay_url
        return True

    monkeypatch.setattr(cli, "_request_and_await_companion_approval", approval)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("no question should be asked"))

    main(["setup", "--invite", make_invite("https://relay.example.com"), "--config-path", str(tmp_path / "config.json")])

    assert asked["relay_url"] == "https://relay.example.com"


def test_setup_rejects_a_garbled_invitation_without_asking_or_connecting(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "_claim_bootstrap_code", lambda *a: pytest.fail("must not connect"))

    main(["setup", "--invite", "AFK1.%%%not-base64", "--config-path", str(tmp_path / "config.json")])

    assert "Could not use that invitation" in capsys.readouterr().out


def test_setup_refuses_an_invitation_that_points_at_an_insecure_address(monkeypatch, tmp_path, capsys):
    import base64

    evil = "AFK1." + base64.urlsafe_b64encode(b"http://evil.example.com").decode().rstrip("=")
    monkeypatch.setattr(cli, "_claim_bootstrap_code", lambda *a: pytest.fail("must not connect"))

    main(["setup", "--invite", evil, "--config-path", str(tmp_path / "config.json")])

    assert "secure" in capsys.readouterr().out


def test_setup_stops_cleanly_on_an_invalid_bootstrap_code(monkeypatch, tmp_path, capsys):
    def raise_error(relay_url, code):
        raise urllib.error.HTTPError(relay_url, 400, "invalid_code", None, None)

    monkeypatch.setattr(cli, "_claim_bootstrap_code", raise_error)
    monkeypatch.setattr("builtins.input", _fake_input("y", "000000"))

    config_path = tmp_path / "config.json"
    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    # See the sibling timeout test above for why this checks for an
    # absent config row rather than raw file non-existence.
    with pytest.raises(FileNotFoundError):
        load_config(str(config_path))
    assert "Could not claim that code" in capsys.readouterr().out


# --- `token set` (U2) -------------------------------------------------------


def test_token_set_claude_saves_the_token_masked_and_notes_the_restart(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "session_registry.db"
    _write_config(config_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-ant-oat-secret")

    main(["token", "set", "claude", "--config-path", str(config_path)])

    loaded = load_config(str(config_path))
    assert loaded.personal_oauth_token == "sk-ant-oat-secret"

    out = capsys.readouterr().out
    assert "sk-ant-oat-secret" not in out
    assert "restart" in out.lower()


def test_token_set_codex_saves_the_api_key_masked_and_notes_the_restart(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "session_registry.db"
    _write_config(config_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-codex-secret")

    main(["token", "set", "codex", "--config-path", str(config_path)])

    loaded = load_config(str(config_path))
    assert loaded.codex_personal_api_key == "sk-codex-secret"

    out = capsys.readouterr().out
    assert "sk-codex-secret" not in out
    assert "restart" in out.lower()


def test_token_set_does_not_revert_a_field_changed_concurrently_by_the_daemon(monkeypatch, tmp_path, capsys):
    """code-review finding (P1): _token_set used to load_config, mutate one
    field on that in-memory snapshot, and save_config the whole thing back
    - a concurrent daemon-triggered save_config landing in between (e.g.
    from set_observe_entrypoints) would be silently reverted by
    _token_set's own save, since save_config's UPSERT always writes every
    column from whatever CompanionConfig it's handed. Simulates that race
    directly: change observe_entrypoints (standing in for "the daemon
    committed something else while token set's getpass prompt was up")
    after _write_config's initial write but before token set runs, and
    confirm token set's targeted single-column update doesn't touch it."""
    from companion.config import load_config as _load, save_config as _save

    config_path = tmp_path / "session_registry.db"
    _write_config(config_path)

    # Stand in for the daemon committing an unrelated change concurrently.
    config = _load(str(config_path))
    config.observe_entrypoints = ["claude", "codex"]
    _save(str(config_path), config)

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("getpass.getpass", lambda prompt="": "sk-ant-oat-secret")

    main(["token", "set", "claude", "--config-path", str(config_path)])

    loaded = load_config(str(config_path))
    assert loaded.personal_oauth_token == "sk-ant-oat-secret"
    assert loaded.observe_entrypoints == ["claude", "codex"]


def test_token_set_rejects_an_unknown_agent_before_the_handler_runs(monkeypatch, tmp_path):
    def fail_if_called(prompt=""):
        raise AssertionError("getpass should never be called for an invalid agent choice")

    monkeypatch.setattr("getpass.getpass", fail_if_called)

    with pytest.raises(SystemExit):
        main(["token", "set", "nonsense", "--config-path", str(tmp_path / "session_registry.db")])


def test_token_set_refuses_non_interactive_stdin_without_calling_save_config(monkeypatch, tmp_path, capsys):
    """KTD5/R2: the isatty check must happen before getpass.getpass is ever
    called, since getpass's own fallback would otherwise silently accept a
    piped value."""
    config_path = tmp_path / "session_registry.db"
    _write_config(config_path)

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    def fail_if_called(prompt=""):
        raise AssertionError("getpass should never be called for non-interactive stdin")

    monkeypatch.setattr("getpass.getpass", fail_if_called)

    def fail_if_saved(path, config):
        raise AssertionError("save_config should never be called for non-interactive stdin")

    monkeypatch.setattr("companion.cli.save_config", fail_if_saved)

    with pytest.raises(SystemExit):
        main(["token", "set", "claude", "--config-path", str(config_path)])

    assert "interactive" in capsys.readouterr().out.lower()


# --- precheck ----------------------------------------------------------------
#
# Update Safety plan (U3), R4-R6: this is a standalone pre-check the operator
# runs themselves before `brew upgrade` - never a `brew upgrade` hook (Homebrew
# has none to wire into), and never something that blocks or fails the
# upgrade regardless of what it reports (R5). It talks to the running
# daemon's status socket (companion/daemon.py's _start_status_socket) as a
# plain client, matching hook_bridge.py's own fail-open-on-OSError posture -
# see test_hook_bridge.py for that pattern's original test.


def _short_socket_path() -> str:
    return f"/tmp/rc-cli-precheck-{uuid.uuid4().hex[:8]}.sock"


def test_precheck_reports_the_active_session_count_from_a_running_daemon(capsys):
    """AE3: the pre-check reports however many active sessions the running
    daemon says it has - here 2 - regardless of what that number is."""
    socket_path = _short_socket_path()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn:
            conn.sendall(json.dumps({"active_sessions": 2}).encode("utf-8") + b"\n")

    thread = threading.Thread(target=serve_one)
    thread.start()
    try:
        main(["precheck", "--status-socket-path", socket_path])
    finally:
        thread.join(timeout=2)
        server.close()
        os.remove(socket_path)

    out = capsys.readouterr().out
    assert "2" in out
    assert "active session" in out.lower()


def test_precheck_reports_daemon_unreachable_when_the_socket_does_not_exist(capsys):
    """AE4: no running daemon (or a stale/gone socket) is reported plainly,
    not as an error - and, per R5, must never look like it blocked
    anything. main() must not raise SystemExit here (implicit exit 0)."""
    main(["precheck", "--status-socket-path", "/tmp/definitely-not-a-real-status-socket.sock"])

    out = capsys.readouterr().out.lower()
    assert "unreachable" in out


def test_precheck_distinguishes_a_daemon_internal_error_from_unreachable(capsys):
    """Code review finding (correctness): a daemon that answered but failed
    internally counting sessions (companion/daemon.py's `error` field) must
    print a message distinguishable from "unreachable" - an operator seeing
    "unreachable" would reasonably conclude the daemon isn't running at
    all, which is a different situation (and a different next step) than a
    running daemon that hit an internal bug."""
    socket_path = _short_socket_path()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn:
            conn.sendall(
                json.dumps({"active_sessions": None, "error": "session registry is locked"}).encode("utf-8")
                + b"\n"
            )

    thread = threading.Thread(target=serve_one)
    thread.start()
    try:
        main(["precheck", "--status-socket-path", socket_path])
    finally:
        thread.join(timeout=2)
        server.close()
        os.remove(socket_path)

    out = capsys.readouterr().out.lower()
    assert "unreachable" not in out
    assert "session registry is locked" in out


def test_precheck_reports_unreachable_on_a_malformed_response(capsys):
    """A daemon that answers with garbage (not the {"active_sessions": ...}
    shape at all) must fail open the same way an actually-unreachable
    daemon does, not raise past AE4's "never look like it blocked
    anything" contract."""
    socket_path = _short_socket_path()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn:
            conn.sendall(b"not json at all\n")

    thread = threading.Thread(target=serve_one)
    thread.start()
    try:
        main(["precheck", "--status-socket-path", socket_path])
    finally:
        thread.join(timeout=2)
        server.close()
        os.remove(socket_path)

    out = capsys.readouterr().out.lower()
    assert "unreachable" in out

