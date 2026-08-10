"""Companion configuration: relay URL and device identity.

Contract for U1 (relay): `device_token` is a companion-kind bearer token
minted once via `relay.auth.bootstrap_companion_device` (an operator
action, not an HTTP call - see relay/auth.py's module docstring) and
handed to the companion out-of-band at install time.

The config file holds a secret (`device_token`), so it's written with
owner-only permissions (mode 0600), the same posture the plan calls for
on the observe-only adapter's Unix socket (U4) - anything readable only by
the companion's own user.
"""
from __future__ import annotations

import json
import os
import stat
from dataclasses import asdict, dataclass

DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/remote-claude-companion/config.json")


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


def save_config(path: str, config: CompanionConfig) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - contains device_token


def load_config(path: str = DEFAULT_CONFIG_PATH) -> CompanionConfig:
    with open(path) as f:
        data = json.load(f)
    return CompanionConfig(**data)
