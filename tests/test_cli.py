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
import urllib.error

import pytest

import companion.cli as cli
from companion.cli import main
from companion.config import load_config


def test_configure_writes_a_config_file_from_relay_url_and_token(tmp_path):
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

    written = json.loads(config_path.read_text())
    assert written["relay_url"] == "wss://relay.example.com/ws/companion"
    assert written["device_token"] == "tok-abc123"


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
    monkeypatch.setattr("companion.cli.run_daemon", lambda config_path=None: calls.append(config_path))

    config_path = str(tmp_path / "config.json")
    main(["run", "--config-path", config_path])

    assert calls == [config_path]


def test_run_defaults_to_the_standard_config_path(monkeypatch):
    from companion.config import DEFAULT_CONFIG_PATH

    calls = []
    monkeypatch.setattr("companion.cli.run_daemon", lambda config_path=None: calls.append(config_path))

    main(["run"])

    assert calls == [DEFAULT_CONFIG_PATH]


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
    config_path.write_text(json.dumps({"relay_url": relay_url, "device_token": device_token}))


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

    assert not config_path.exists()
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


def test_setup_with_no_relay_url_anywhere_prints_guidance_instead_of_crashing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "DEFAULT_RELAY_URL", None)

    main(["setup", "--config-path", str(tmp_path / "does-not-exist.json")])

    out = capsys.readouterr().out
    assert "AFK_RELAY_URL" in out or "--relay-url" in out


def test_setup_stops_cleanly_on_an_invalid_bootstrap_code(monkeypatch, tmp_path, capsys):
    def raise_error(relay_url, code):
        raise urllib.error.HTTPError(relay_url, 400, "invalid_code", None, None)

    monkeypatch.setattr(cli, "_claim_bootstrap_code", raise_error)
    monkeypatch.setattr("builtins.input", _fake_input("y", "000000"))

    config_path = tmp_path / "config.json"
    main(["setup", "--relay-url", "https://relay.example.com", "--config-path", str(config_path)])

    assert not config_path.exists()
    assert "Could not claim that code" in capsys.readouterr().out
