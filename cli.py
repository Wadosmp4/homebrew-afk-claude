"""Command-line entry point for the companion, installed as `afk-claude-companion`
(see the repo-root pyproject.toml's [project.scripts] and the Homebrew formula).

Five subcommands:

- `run`       - starts the daemon (companion.daemon.main), same as the
  existing `python3 -m companion.daemon`.
- `setup`     - the one command for any Mac, in any state: if it already
  has a device token, this is just `pairing-code` below; if not, it asks
  one question to tell apart the two ways an unconfigured Mac can get its
  first token - either it's an additional Mac and some phone is already
  paired (same request/approve/claim flow as `pair` below, minus needing
  to already know that's the right command), or it's the very first Mac
  ever and needs the one-time bootstrap code a relay operator generated
  locally (see relay/bootstrap_companion.py). Either way it finishes by
  printing a phone pairing code if one is actually needed - no relay URL,
  database, or token ever typed by hand (`--relay-url` falls back to the
  AFK_RELAY_URL env var - see companion/config.py's DEFAULT_RELAY_URL).
- `pair`      - the direct form of `setup`'s additional-Mac branch, for
  when you already know a phone is paired and don't want the extra
  question: requests a short code from the relay, prints it for you to
  enter in the mobile app, polls until that already-paired phone approves
  it, then writes the config itself. No database access, no admin
  secret, no manually copying a token between two commands - see
  relay/pairing.py's module comment for the full request/approve/claim
  design this drives.
- `configure` - the lower-level fallback: writes the config directly from
  a relay URL and device token you already have some other way
  (scripting, a token claimed some other way, etc).
- `pairing-code` - print a relay address + short code for another phone,
  once this companion already has its own device token (from `setup`,
  `pair`, or `configure`) - authenticates the relay's POST
  /pairing/register itself instead of requiring a raw curl call.

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

from .config import CompanionConfig, DEFAULT_CONFIG_PATH, DEFAULT_RELAY_URL, load_config, save_config
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


def _claim_bootstrap_code(relay_url: str, code: str) -> dict:
    """Kept as its own small function so tests can monkeypatch it directly
    instead of faking urllib - same convention as the other _request_*/
    _poll_* helpers above. Exchanges the one-time code a relay operator
    generated locally (relay/bootstrap_companion.py) for this companion's
    first device token - see relay/pairing.py's claim_bootstrap_code."""
    return _http_post_json(f"{relay_url}/pairing/bootstrap-claim", {"code": code})


def _error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        return json.loads(exc.read()).get("detail", "unknown_error")
    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
        return "unknown_error"


def _request_and_await_companion_approval(
    relay_url: str, label: Optional[str], timeout: float, poll_interval: float, config_path: str
) -> Optional[bool]:
    """The `pair` flow: request a companion-request code, print it, and
    poll until an already-paired phone approves it. Shared by `pair`
    itself and by `setup`'s branch for a Mac that isn't the very first
    one (some phone is already paired and can vouch for it) - see
    relay/pairing.py's module comment for the full request/approve/claim
    design.

    Returns True (config written), False (the request or a poll hit a
    real error - already printed here), or None (timed out, nothing
    wrong, just nobody approved it in time) - callers add their own
    finishing message for True/None either way."""
    try:
        result = _request_companion_pairing(relay_url, label)
    except urllib.error.HTTPError as exc:
        print(f"Could not request a pairing code: {_error_detail(exc)}")
        return False
    except urllib.error.URLError as exc:
        print(f"Could not reach the relay at {relay_url}: {exc.reason}")
        return False

    code = result["code"]
    print(f"Enter this code in the AFK Claude app to connect this Mac: {code}")
    print(f"(expires at {result['expires_at']})")
    print("Waiting for approval...")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            body = _poll_claim(relay_url, code)
        except urllib.error.HTTPError as exc:
            print(f"Pairing failed: {_error_detail(exc)}")
            return False
        except urllib.error.URLError as exc:
            print(f"Could not reach the relay at {relay_url}: {exc.reason}")
            return False

        if body.get("status") == "approved":
            ws_url = _derive_ws_url(relay_url)
            save_config(config_path, CompanionConfig(relay_url=ws_url, device_token=body["token"]))
            print(f"Paired. Config written to {config_path}")
            return True

        time.sleep(poll_interval)

    return None


def _pair(args: argparse.Namespace) -> None:
    relay_url = args.relay_url.rstrip("/")
    label = args.label or socket.gethostname()

    approved = _request_and_await_companion_approval(
        relay_url, label, args.timeout, args.poll_interval, args.config_path
    )
    if approved is True:
        print("Run `afk-claude-companion run` to start the daemon.")
    elif approved is None:
        print("Timed out waiting for approval - run `afk-claude-companion pair` again.")


def _print_phone_pairing_code(relay_url: str, device_token: str) -> None:
    """Shared by `pairing-code` and `setup` (once a companion has, or just
    got, its own device token) - requests and prints a phone pairing
    code."""
    try:
        result = _request_phone_pairing_code(relay_url, device_token)
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


def _pairing_code(args: argparse.Namespace) -> None:
    try:
        config = load_config(args.config_path)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Could not read config at {args.config_path}: {exc}")
        print("Run `afk-claude-companion setup` first.")
        return

    relay_url = _derive_http_url(config.relay_url)
    _print_phone_pairing_code(relay_url, config.device_token)


def _setup(args: argparse.Namespace) -> None:
    """The one command for any Mac, in any state - already configured,
    an additional Mac joining a phone that's already paired, or the very
    first Mac ever. See this module's docstring."""
    try:
        config = load_config(args.config_path)
    except (OSError, ValueError, TypeError):
        config = None

    if config is not None:
        relay_url = _derive_http_url(config.relay_url)
        _print_phone_pairing_code(relay_url, config.device_token)
        return

    relay_url = args.relay_url or DEFAULT_RELAY_URL
    if not relay_url:
        print("No relay address known - set the AFK_RELAY_URL environment variable, or pass --relay-url.")
        return
    relay_url = relay_url.rstrip("/")

    has_bootstrap_code = (
        input("Do you have a one-time bootstrap code from your relay? [y/N]: ").strip().lower().startswith("y")
    )

    if not has_bootstrap_code:
        # Not the very first Mac - some phone is already paired and can
        # vouch for this one, same as `pair`.
        label = args.label or socket.gethostname()
        approved = _request_and_await_companion_approval(
            relay_url, label, DEFAULT_PAIR_TIMEOUT_SECONDS, DEFAULT_PAIR_POLL_INTERVAL_SECONDS, args.config_path
        )
        if approved is True:
            print("Run `afk-claude-companion run` to start the daemon.")
        elif approved is None:
            print("Timed out waiting for approval - run `afk-claude-companion setup` again.")
        return

    code = input("Enter the bootstrap code shown by your relay: ").strip()
    try:
        result = _claim_bootstrap_code(relay_url, code)
    except urllib.error.HTTPError as exc:
        print(f"Could not claim that code: {_error_detail(exc)}")
        return
    except urllib.error.URLError as exc:
        print(f"Could not reach the relay at {relay_url}: {exc.reason}")
        return

    ws_url = _derive_ws_url(relay_url)
    save_config(args.config_path, CompanionConfig(relay_url=ws_url, device_token=result["token"]))
    print(f"This Mac is now paired with the relay. Config written to {args.config_path}")
    _print_phone_pairing_code(relay_url, result["token"])


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

    setup_parser = subparsers.add_parser(
        "setup",
        help="One command for any Mac in any state: prints a phone pairing code, claiming a bootstrap code first if needed",
    )
    setup_parser.add_argument(
        "--relay-url",
        default=DEFAULT_RELAY_URL,
        help="The relay's https:// URL - only needed before this Mac has a config (default: AFK_RELAY_URL env var)",
    )
    setup_parser.add_argument(
        "--label", default=None, help="Human-readable label for this Mac if it needs a phone's approval (default: hostname)"
    )
    setup_parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_CONFIG_PATH})",
    )
    setup_parser.set_defaults(func=_setup)

    pair_parser = subparsers.add_parser(
        "pair", help="Connect this Mac to a relay - approve the printed code from the mobile app"
    )
    pair_parser.add_argument(
        "--relay-url",
        default=DEFAULT_RELAY_URL,
        required=DEFAULT_RELAY_URL is None,
        help="The relay's https:// URL, e.g. https://your-relay.herokuapp.com (default: AFK_RELAY_URL env var)",
    )
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
        help="This companion's device token, already obtained some other way (scripting, etc) - `setup` is the normal path",
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
