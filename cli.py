"""Command-line entry point for the companion, installed as `afk-claude-companion`
(see the repo-root pyproject.toml's [project.scripts] and the Homebrew formula).

Four subcommands:

- `run`       - starts the daemon (companion.daemon.main), same as the
  existing `python3 -m companion.daemon`.
- `pair`      - the recommended way to connect a new Mac: requests a
  short code from the relay, prints it for you to enter in the mobile app,
  polls until an already-paired phone approves it, then writes the config
  itself. No database access, no admin secret, no manually copying a
  token between two commands - see relay/pairing.py's module comment for
  the full request/approve/claim design this drives.
- `configure` - the lower-level fallback: writes the config directly from
  a relay URL and device token you already have some other way (scripting,
  a token minted via relay/bootstrap_companion.py, etc).
- `pairing-code` - the *first* phone's onboarding step: this companion is
  already configured (it has its own device token), so it authenticates
  the relay's POST /pairing/register itself and prints the resulting code
  + relay address for the mobile app's Pair screen. Before this existed,
  minting that first code required a raw authenticated curl call - the
  same mismatch `pair` fixed for connecting additional Macs.

Only stdlib (`urllib`) for the HTTP calls these need - deliberately not
adding `httpx`/`requests` as a dependency just for this, since every
dependency here is one more entry in the Homebrew formula's resource list.
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

from .config import CompanionConfig, DEFAULT_CONFIG_PATH, load_config, save_config
from .daemon import main as run_daemon

DEFAULT_PAIR_TIMEOUT_SECONDS = 300.0  # matches the relay's own PAIRING_CODE_TTL_SECONDS
DEFAULT_PAIR_POLL_INTERVAL_SECONDS = 2.0


def _configure(args: argparse.Namespace) -> None:
    save_config(
        args.config_path,
        CompanionConfig(relay_url=args.relay_url, device_token=args.device_token),
    )
    print(f"Config written to {args.config_path}")
    print("Run `afk-claude-companion run` to start the daemon.")


def _run(args: argparse.Namespace) -> None:
    run_daemon(config_path=args.config_path)


def _derive_ws_url(relay_http_url: str) -> str:
    """The pairing HTTP calls and the daemon's own WebSocket connection use
    the same relay, but different URL schemes - `pair` only ever asks for
    the one, ordinary https:// URL a person would actually type."""
    base = relay_http_url.rstrip("/")
    if base.startswith("https://"):
        ws_base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        ws_base = "ws://" + base[len("http://"):]
    else:
        raise ValueError(f"--relay-url must start with http:// or https://, got: {relay_http_url!r}")
    return f"{ws_base}/ws/companion"


def _derive_http_url(relay_ws_url: str) -> str:
    """Inverse of _derive_ws_url - the companion's own saved config stores
    the WebSocket URL its daemon connects with, but pairing-code's HTTP
    call (and the address a person types into the app) needs the ordinary
    https:// form instead."""
    base = relay_ws_url.rstrip("/")
    if base.endswith("/ws/companion"):
        base = base[: -len("/ws/companion")]
    if base.startswith("wss://"):
        return "https://" + base[len("wss://"):]
    if base.startswith("ws://"):
        return "http://" + base[len("ws://"):]
    raise ValueError(f"stored relay_url must start with ws:// or wss://, got: {relay_ws_url!r}")


def _http_post_json(url: str, body: dict) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _http_post_authed(url: str, token: str) -> dict:
    req = urllib.request.Request(url, data=b"", headers={"Authorization": f"Bearer {token}"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read())


def _request_companion_pairing(relay_url: str, label: Optional[str]) -> dict:
    """Kept as its own small function (rather than inlined in _pair) so
    tests can monkeypatch it directly instead of faking urllib."""
    return _http_post_json(f"{relay_url}/pairing/companion-request", {"label": label})


def _poll_claim(relay_url: str, code: str) -> dict:
    return _http_get_json(f"{relay_url}/pairing/companion-claim/{code}")


def _request_phone_pairing_code(relay_url: str, device_token: str) -> dict:
    """Kept as its own small function (rather than inlined in
    _pairing_code) so tests can monkeypatch it directly instead of faking
    urllib - same convention as _request_companion_pairing above."""
    return _http_post_authed(f"{relay_url}/pairing/register", device_token)


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return json.loads(exc.read()).get("detail", "unknown_error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return "unknown_error"


def _pair(args: argparse.Namespace) -> None:
    relay_url = args.relay_url.rstrip("/")
    label = args.label or socket.gethostname()

    try:
        result = _request_companion_pairing(relay_url, label)
    except urllib.error.HTTPError as exc:
        print(f"Could not request a pairing code: {_error_detail(exc)}")
        return
    except urllib.error.URLError as exc:
        print(f"Could not reach the relay at {relay_url}: {exc.reason}")
        return

    code = result["code"]
    print(f"Enter this code in the AFK Claude app to connect this Mac: {code}")
    print(f"(expires at {result['expires_at']})")
    print("Waiting for approval...")

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        try:
            body = _poll_claim(relay_url, code)
        except urllib.error.HTTPError as exc:
            print(f"Pairing failed: {_error_detail(exc)}")
            return
        except urllib.error.URLError as exc:
            print(f"Could not reach the relay at {relay_url}: {exc.reason}")
            return

        if body.get("status") == "approved":
            ws_url = _derive_ws_url(relay_url)
            save_config(args.config_path, CompanionConfig(relay_url=ws_url, device_token=body["token"]))
            print(f"Paired. Config written to {args.config_path}")
            print("Run `afk-claude-companion run` to start the daemon.")
            return

        time.sleep(args.poll_interval)

    print("Timed out waiting for approval - run `afk-claude-companion pair` again.")


def _pairing_code(args: argparse.Namespace) -> None:
    try:
        config = load_config(args.config_path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Could not read config at {args.config_path}: {exc}")
        print("Run `afk-claude-companion pair` or `configure` first.")
        return

    relay_url = _derive_http_url(config.relay_url)
    try:
        result = _request_phone_pairing_code(relay_url, config.device_token)
    except urllib.error.HTTPError as exc:
        print(f"Could not request a pairing code: {_error_detail(exc)}")
        return
    except urllib.error.URLError as exc:
        print(f"Could not reach the relay at {relay_url}: {exc.reason}")
        return

    print("Enter these in the AFK Claude app's Pair screen:")
    print(f"  Relay address: {relay_url}")
    print(f"  Pairing code:  {result['code']}")
    print(f"(expires at {result['expires_at']})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afk-claude-companion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start the companion daemon")
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_CONFIG_PATH})",
    )
    run_parser.set_defaults(func=_run)

    pair_parser = subparsers.add_parser(
        "pair", help="Connect this Mac to a relay - approve the printed code from the mobile app"
    )
    pair_parser.add_argument("--relay-url", required=True, help="The relay's https:// URL, e.g. https://your-relay.herokuapp.com")
    pair_parser.add_argument("--label", default=None, help="Human-readable label for this Mac (default: hostname)")
    pair_parser.add_argument(
        "--timeout", type=float, default=DEFAULT_PAIR_TIMEOUT_SECONDS, help="Seconds to wait for approval before giving up"
    )
    pair_parser.add_argument(
        "--poll-interval", type=float, default=DEFAULT_PAIR_POLL_INTERVAL_SECONDS, help="Seconds between approval checks"
    )
    pair_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Where to write the resulting config (default: {DEFAULT_CONFIG_PATH})",
    )
    pair_parser.set_defaults(func=_pair)

    configure_parser = subparsers.add_parser(
        "configure", help="Write this machine's companion config from a relay URL and device token"
    )
    configure_parser.add_argument(
        "--relay-url", required=True, help="The relay's WebSocket URL, e.g. wss://your-relay.example.com/ws/companion"
    )
    configure_parser.add_argument(
        "--device-token",
        required=True,
        help="This companion's device token - mint one with relay/bootstrap_companion.py, run against the relay you operate",
    )
    configure_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Where to write the resulting config (default: {DEFAULT_CONFIG_PATH})",
    )
    configure_parser.set_defaults(func=_configure)

    pairing_code_parser = subparsers.add_parser(
        "pairing-code",
        help="Print a relay address + short code to pair a phone with the AFK Claude app",
    )
    pairing_code_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_CONFIG_PATH})",
    )
    pairing_code_parser.set_defaults(func=_pairing_code)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
