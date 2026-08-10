"""Installs/removes the observe-only adapter's (U4) hooks block in a
watched repo's `.claude/settings.json`, per KTD2.

Each hook command shells out to `companion.hook_bridge`, which forwards the
hook's stdin JSON to the companion's local Unix socket and prints the
response back to stdout for Claude Code to interpret (e.g. a
`permissionDecision` for PermissionRequest).

Risk mitigation (see plan's System-Wide Impact note): `.claude/settings.json`
is shared, mutable repo state other tooling (the developer, `ce-setup`)
may also edit - `install_hooks`/`uninstall_hooks` merge into the existing
`hooks` block rather than overwriting it, and only ever touch entries this
module itself created (identified by the `companion.hook_bridge` marker in
the command string).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

HOOK_EVENTS = ("PermissionRequest", "Notification", "Stop", "SessionStart", "SessionEnd")
_BRIDGE_MARKER = "companion.hook_bridge"


def _load_settings(path: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        content = f.read().strip()
    return json.loads(content) if content else {}


def _save_settings(path: str, settings: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")


def _our_command(event: str, socket_path: str, python_bin: str) -> str:
    return f"{python_bin} -m companion.hook_bridge {event} {socket_path}"


def _is_our_entry(entry: dict[str, Any]) -> bool:
    return any(_BRIDGE_MARKER in h.get("command", "") for h in entry.get("hooks", []))


def install_hooks(settings_path: str, socket_path: str, python_bin: str | None = None) -> None:
    """Idempotent: safe to call repeatedly (e.g. once per watched repo per
    daemon start) without duplicating entries."""
    python_bin = python_bin or sys.executable
    settings = _load_settings(settings_path)
    hooks = settings.setdefault("hooks", {})

    for event in HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        if any(_is_our_entry(e) for e in entries):
            continue
        entries.append(
            {
                "matcher": "*",
                "hooks": [{"type": "command", "command": _our_command(event, socket_path, python_bin)}],
            }
        )

    _save_settings(settings_path, settings)


def uninstall_hooks(settings_path: str) -> None:
    """Removes only this module's own entries, leaving any other tooling's
    hooks (e.g. `ce-setup`'s) untouched."""
    if not os.path.exists(settings_path):
        return
    settings = _load_settings(settings_path)
    hooks = settings.get("hooks", {})

    for event in HOOK_EVENTS:
        if event not in hooks:
            continue
        hooks[event] = [e for e in hooks[event] if not _is_our_entry(e)]
        if not hooks[event]:
            del hooks[event]

    if not hooks:
        settings.pop("hooks", None)

    _save_settings(settings_path, settings)
