"""Command-line entry point for the companion, installed as `afk-claude-companion`
(see the repo-root pyproject.toml's [project.scripts] and the Homebrew formula).

Six subcommands:

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
- `token set <claude|codex>` - interactively set this Mac's own personal
  Claude/Codex token with masked input (`getpass`), stored locally with
  zero network transit. Refuses to run with piped/non-interactive stdin
  rather than silently accepting a plaintext value. A running companion
  needs restarting to pick up the change.
- `precheck`   - Update Safety plan (U3), R4-R6: reports how many sessions
  the currently-running daemon considers active, before an operator runs
  `brew upgrade` - purely informational (R5), never blocks or fails the
  upgrade itself, and there is no way to make `brew upgrade` pause on it
  (Homebrew has no pre-upgrade hook - see this unit's plan). Talks to the
  daemon's status socket (companion/daemon.py's `_start_status_socket`) as
  a plain client; if that daemon isn't running or doesn't answer, this
  reports "unreachable" plainly (R6) rather than raising or exiting
  nonzero, following hook_bridge.py's own fail-open-on-OSError posture.

Only stdlib (`urllib`) for the HTTP calls these need - deliberately not
adding `httpx`/`requests` as a dependency just for this, since every
dependency here is one more entry in the Homebrew formula's resource list.
"""
from __future__ import annotations

import argparse
import getpass
import json
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from typing import Optional, Sequence

from . import hook_bridge
from .invite import looks_like_invite, parse_invite
from .config import (
    CompanionConfig,
    DEFAULT_RELAY_URL,
    DEFAULT_SESSION_REGISTRY_PATH,
    load_config,
    save_config,
    update_config_field,
)
from .daemon import DEFAULT_STATUS_SOCKET_PATH, main as run_daemon

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
    run_daemon(config_path=args.config_path, status_socket_path=args.status_socket_path)


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
    # code-review finding: load_config's SQLite backing store can raise
    # sqlite3.Error (a corrupted or locked db file) - this and the other two
    # load_config call sites below all catch it alongside the pre-existing
    # (OSError, ValueError, TypeError) trio, so a corrupted db produces this
    # same friendly "run setup first" message instead of a raw traceback.
    try:
        config = load_config(args.config_path)
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        print(f"Could not read config at {args.config_path}: {exc}")
        print("Run `afk-claude-companion setup` first.")
        return

    relay_url = _derive_http_url(config.relay_url)
    _print_phone_pairing_code(relay_url, config.device_token)


def _token_set(args: argparse.Namespace) -> None:
    """Companion-Local Token Entry & SQLite Config plan (R1, R2, R10; KTD5,
    KTD8): sets `personal_oauth_token`/`codex_personal_api_key` locally,
    with masked input and zero network transit. The stdin-isatty check
    happens before `getpass.getpass()` is ever called (KTD5) - `getpass`'s
    own fallback silently accepts a piped value instead of refusing, which
    could land a token in shell history or script output (R2)."""
    if not sys.stdin.isatty():
        print("`token set` requires an interactive terminal - refusing to read a piped or redirected value.")
        sys.exit(1)

    label = "Claude" if args.agent == "claude" else "Codex"
    token = getpass.getpass(f"Enter your personal {label} token (input hidden): ").strip()

    try:
        load_config(args.config_path)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        print(f"No existing config at {args.config_path} - run `afk-claude-companion setup` first.")
        sys.exit(1)

    # code-review finding (P1): the load-mutate-save pattern this used to
    # follow (load the whole config, set one field on the in-memory
    # dataclass, save the whole thing back) races a concurrent daemon-
    # triggered save_config - whichever save lands second would silently
    # revert every field the other one just changed. A targeted single-
    # column UPDATE can't clobber a concurrent change to a different
    # field, because it never touches any column but this one.
    field = "personal_oauth_token" if args.agent == "claude" else "codex_personal_api_key"
    update_config_field(args.config_path, **{field: token})
    print(f"{label} personal token saved locally.")
    print("A running companion won't pick this up until it's restarted (`afk-claude-companion run`).")


def _precheck(args: argparse.Namespace) -> None:
    """Update Safety plan (U3), R4-R6: this is a standalone pre-check the
    operator runs themselves before `brew upgrade` - Homebrew has no
    pre-upgrade hook to wire this into (see this module's docstring), so it
    is never invoked automatically and never blocks the upgrade regardless
    of what it reports (R5). Deliberately does not import anything from
    `relay/` - companion/pyproject.toml's own packages list already keeps
    `relay/` out of a Homebrew install entirely, so this stays a plain
    client of the daemon's own local status socket (companion/daemon.py's
    `_start_status_socket`), sharing hook_bridge.py's connect+recv-until-
    close helper and its fail-open-on-OSError posture: an unreachable
    daemon (not running, socket gone, unresponsive) is reported plainly
    rather than raised as an error (R6), and this function never calls
    sys.exit with a nonzero code - AE4 is explicit that "unreachable" must
    not look like it blocked anything.

    Code review finding (correctness): a daemon that IS reachable but
    fails internally counting sessions (companion/daemon.py's
    `_handle_status_connection` exception branch) used to print the exact
    same "unreachable" message as a daemon that isn't running at all -
    genuinely different situations for an operator deciding whether to
    upgrade anyway. That branch's response now carries an `error` field;
    this function reports it distinctly rather than folding it into
    "unreachable"."""
    try:
        response = hook_bridge.read_unix_socket_response(args.status_socket_path, timeout=5.0)
    except OSError:
        print("Companion daemon is unreachable - could not check active sessions.")
        return

    try:
        body = json.loads(response.decode("utf-8"))
        count = body["active_sessions"]
        error = body.get("error") if isinstance(body, dict) else None
    except (json.JSONDecodeError, UnicodeDecodeError, KeyError, TypeError):
        count = None
        error = None

    if count is None:
        if error:
            print(f"Companion daemon is running but failed to count active sessions: {error}")
        else:
            print("Companion daemon is unreachable - could not check active sessions.")
        return

    plural = "s" if count != 1 else ""
    print(f"The running companion daemon reports {count} active session{plural}.")
    print("This is informational only - `brew upgrade` will proceed regardless of this count.")


def _setup(args: argparse.Namespace) -> None:
    """The one command for any Mac, in any state - already configured,
    an additional Mac joining a phone that's already paired, or the very
    first Mac ever. See this module's docstring."""
    try:
        config = load_config(args.config_path)
    except (OSError, ValueError, TypeError, sqlite3.Error):
        config = None

    if config is not None:
        relay_url = _derive_http_url(config.relay_url)
        _print_phone_pairing_code(relay_url, config.device_token)
        return

    # An invitation (--invite) carries the address, so nothing has to be typed or looked up.
    try:
        invite = parse_invite(args.invite) if args.invite else None
    except ValueError as exc:
        print(f"Could not use that invitation: {exc}.")
        return

    relay_url = (invite.relay_url if invite else None) or args.relay_url or DEFAULT_RELAY_URL
    code: Optional[str] = invite.code if invite else None

    if invite is None:
        has_code = input("Do you have an invitation code? [y/N]: ").strip().lower().startswith("y")
        if has_code:
            pasted = input("Paste your invitation code: ").strip()
            if looks_like_invite(pasted):
                try:
                    invite = parse_invite(pasted)
                except ValueError as exc:
                    print(f"Could not use that invitation: {exc}.")
                    return
                relay_url, code = invite.relay_url, invite.code
            else:
                code = pasted  # a bare one-time code: the address has to come from elsewhere (--relay-url / env)
        elif not relay_url:
            print(
                "To add this computer to an existing account, use the command shown in the app: "
                "Home > Add a computer."
            )
            return

    if not relay_url:
        print("No address known for this code - ask for the full invitation code.")
        return
    relay_url = relay_url.rstrip("/")

    if code is None:
        # No one-time code: an already-paired phone vouches for this computer, same as `pair`.
        label = args.label or socket.gethostname()
        approved = _request_and_await_companion_approval(
            relay_url, label, DEFAULT_PAIR_TIMEOUT_SECONDS, DEFAULT_PAIR_POLL_INTERVAL_SECONDS, args.config_path
        )
        if approved is True:
            print("Run `afk-claude-companion run` to start the daemon.")
        elif approved is None:
            print("Timed out waiting for approval - run `afk-claude-companion setup` again.")
        return

    try:
        result = _claim_bootstrap_code(relay_url, code)
    except urllib.error.HTTPError as exc:
        print(f"Could not claim that code: {_error_detail(exc)}")
        return
    except urllib.error.URLError as exc:
        print(f"Could not reach the service: {exc.reason}")
        return

    ws_url = _derive_ws_url(relay_url)
    save_config(args.config_path, CompanionConfig(relay_url=ws_url, device_token=result["token"]))
    print(f"This computer is now connected. Config written to {args.config_path}")
    _print_phone_pairing_code(relay_url, result["token"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="afk-claude-companion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Start the companion daemon")
    run_parser.add_argument(
        "--config-path",
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_SESSION_REGISTRY_PATH})",
    )
    run_parser.add_argument(
        "--status-socket-path",
        default=None,
        help=(
            "Path to this daemon's status socket, answered by `precheck` (default: "
            f"{DEFAULT_STATUS_SOCKET_PATH}). Only needed to run a second daemon "
            "instance alongside one already using the default path."
        ),
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
        "--invite",
        default=None,
        help="An invitation code (it carries the service address, so no --relay-url is needed)",
    )
    setup_parser.add_argument(
        "--label", default=None, help="Human-readable label for this Mac if it needs a phone's approval (default: hostname)"
    )
    setup_parser.add_argument(
        "--config-path",
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_SESSION_REGISTRY_PATH})",
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
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Where to write the resulting config (default: {DEFAULT_SESSION_REGISTRY_PATH})",
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
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Where to write the resulting config (default: {DEFAULT_SESSION_REGISTRY_PATH})",
    )
    configure_parser.set_defaults(func=_configure)

    pairing_code_parser = subparsers.add_parser(
        "pairing-code",
        help="Print a relay address + short code to pair a phone with the AFK Claude app",
    )
    pairing_code_parser.add_argument(
        "--config-path",
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_SESSION_REGISTRY_PATH})",
    )
    pairing_code_parser.set_defaults(func=_pairing_code)

    token_parser = subparsers.add_parser("token", help="Manage this Mac's personal Claude/Codex tokens")
    token_subparsers = token_parser.add_subparsers(dest="token_command", required=True)
    token_set_parser = token_subparsers.add_parser(
        "set", help="Interactively set a personal token for an agent, stored locally with zero network transit"
    )
    token_set_parser.add_argument("agent", choices=["claude", "codex"], help="Which agent's personal token to set")
    token_set_parser.add_argument(
        "--config-path",
        default=DEFAULT_SESSION_REGISTRY_PATH,
        help=f"Path to the companion config file (default: {DEFAULT_SESSION_REGISTRY_PATH})",
    )
    token_set_parser.set_defaults(func=_token_set)

    precheck_parser = subparsers.add_parser(
        "precheck",
        help="Report how many sessions the running companion daemon considers active - run this before `brew upgrade` (never blocks it)",
    )
    precheck_parser.add_argument(
        "--status-socket-path",
        default=DEFAULT_STATUS_SOCKET_PATH,
        help=f"Path to the running daemon's status socket (default: {DEFAULT_STATUS_SOCKET_PATH})",
    )
    precheck_parser.set_defaults(func=_precheck)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
