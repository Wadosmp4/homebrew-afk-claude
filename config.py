"""Companion configuration: relay URL and device identity.

Contract for U1 (relay): `device_token` is a companion-kind bearer token.
The normal path to getting one is `afk-claude-companion setup` (cli.py) -
claiming a one-time bootstrap code a relay operator generated locally
via `relay/bootstrap_companion.py` (an operator action against the
database, not an HTTP call - see relay/auth.py's module docstring for
why minting stays off the network).

The config file holds a secret (`device_token`), so it's written with
owner-only permissions (mode 0600), the same posture the plan calls for
on the observe-only adapter's Unix socket (U4) - anything readable only by
the companion's own user.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass, field
from typing import Optional

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/remote-claude-companion/config.json")

# `setup`/`pair` (cli.py) fall back to this when --relay-url isn't passed,
# mirroring how the mobile app bakes in EXPO_PUBLIC_RELAY_URL at build time
# (mobile/screens/PairingScreen.tsx) - this is a personal, single-relay
# app, so there's no real reason to type the same hostname every time.
# None (unset) is a legitimate value: the CLI just asks for --relay-url
# explicitly in that case instead of silently failing.
DEFAULT_RELAY_URL = os.environ.get("AFK_RELAY_URL")


@dataclass
class CompanionConfig:
    relay_url: str
    device_token: str
    heartbeat_interval: float = 5.0
    reconnect_initial_delay: float = 1.0
    reconnect_max_delay: float = 30.0
    # How often the daemon polls ObserveAdapter.discover_sessions() for a
    # session it hasn't started forwarding yet (U4 has no "new session"
    # callback, only the growing list - see daemon.py's _watch_active_sessions).
    observe_scan_interval: float = 1.0
    # Which Claude Code clients' sessions the observe-only watcher surfaces
    # (ObserveAdapter's required_entrypoints, R: "give an ability to choose
    # what clients to use") - ~/.claude/projects is shared machine-wide
    # across every client and project, so without this, an unrelated repo
    # opened from a different client shows up in the phone's Sessions list
    # too. Defaults to just the Desktop app; the phone can change this via
    # set_observe_entrypoints (daemon.py), which persists back here.
    observe_entrypoints: list[str] = field(default_factory=lambda: ["claude-desktop"])
    # Same policy used for phone-started sessions (companion/auto_approve.py,
    # companion/risk_judge.py), applied to observed/hook-based sessions too -
    # a terminal-started `claude` session's PermissionRequest hook checks
    # these live (ObserveAdapter._dispatch_hook), not just sessions the phone
    # itself spawned via start_session. Defaults off; the phone toggles both
    # via set_auto_approve_settings (daemon.py), which persists back here.
    observe_auto_approve: bool = False
    observe_llm_judge: bool = False
    # R7/KD5/KTD5: which Claude Code CLI binary/profile this Mac's companion
    # invokes for new SDK-owned sessions (companion/adapters/sdk_adapter.py's
    # connect(), threaded into ClaudeAgentOptions' own cli_path/env). A
    # separate mechanism from the per-session `model` field above - not a
    # repurposing of it. Defaults keep a pre-existing on-disk config.json
    # (written before this field existed) loading under
    # CompanionConfig(**json.load(f))'s strict field match; cli_env must
    # stay dict-shaped (never None) since ClaudeAgentOptions.env is
    # dict-typed, not Optional, on the SDK side.
    cli_path: Optional[str] = None
    cli_env: dict[str, str] = field(default_factory=dict)
    # Mobile UX follow-up #3b: opt-in fast path for AI-judgment
    # (companion/risk_judge.py) - when true AND this daemon process's own
    # ANTHROPIC_API_KEY environment variable is set, judge_is_safe skips
    # the CLI-subprocess path entirely for a direct Anthropic API call to
    # a fast/cheap model instead (no subprocess cold-start per judged
    # call, the dominant cost the CLI-subprocess path pays every time).
    # Defaults off, and deliberately a persisted config field rather than
    # risk_judge.py reading the environment variable on its own - the
    # CLI-subprocess fallback (works for subscription-only `claude` auth,
    # no separate API key needed) stays every existing installation's
    # unchanged behavior unless this is turned on explicitly; it must
    # never flip on as a side effect of the daemon process merely
    # inheriting an ANTHROPIC_API_KEY some other tool set in the same
    # shell. Enabling this also requires `pip install anthropic`
    # separately (see companion/README.md) - deliberately not a listed
    # companion dependency, so the default install stays exactly as lean
    # for the vast majority of users who never turn this on.
    risk_judge_use_api: bool = False
    # Which Claude identity new SDK-owned sessions launch under (U1 of the
    # session-account-switch plan). "vscode" reuses the CLI's existing
    # default login (the same one the VS Code extension shares) with no
    # extra config; "personal" requires personal_oauth_token to be set,
    # merged into cli_env as CLAUDE_CODE_OAUTH_TOKEN only for that spawn
    # (see daemon.py's session-start call sites and _send_event redaction -
    # KTD1/KTD8). personal_oauth_token is a `claude setup-token`-minted
    # credential, not the device_token's bearer-token shape - same 0600
    # file, same "never echoed back over the wire" posture as device_token.
    active_account: str = "vscode"
    personal_oauth_token: Optional[str] = None


def save_config(path: str, config: CompanionConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - contains device_token


def load_config(path: str = DEFAULT_CONFIG_PATH) -> CompanionConfig:
    with open(path) as f:
        data = json.load(f)
    return CompanionConfig(**data)
