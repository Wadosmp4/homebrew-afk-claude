"""Per-session settings (cwd, model, auto_approve, llm_judge) that need to
survive a companion restart - SDKAdapter._sessions itself doesn't (it's a
plain in-memory dict, wiped whenever the daemon process dies), so without
this, a resumed session (daemon.py's _try_resume_sdk_session) always came
back with the opt-in defaults, silently discarding whatever the phone had
actually chosen for it.

A small local JSON file, not the relay's Postgres - the companion runs on
the user's own machine while the relay runs on a separate server (see
relay/db.py's DATABASE_URL), so companion-side state can't live there
anyway. This follows the same single-file-per-concern convention already
used for the global config (config.py) and the recent-projects list
(projects.py), rather than introducing a database dependency for a
handful of small per-session fields.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

DEFAULT_SESSION_SETTINGS_PATH = os.path.expanduser("~/.config/remote-claude-companion/session_settings.json")
# Oldest dropped past this - same cap style as projects.py's MAX_RECENTS,
# so years of daily use don't grow this file without bound.
MAX_SESSION_SETTINGS = 500


@dataclass
class SessionSettings:
    cwd: str
    model: Optional[str] = None
    auto_approve: bool = False
    llm_judge: bool = False


def save_session_settings(
    session_id: str, settings: SessionSettings, path: Optional[str] = None
) -> None:
    path = path if path is not None else DEFAULT_SESSION_SETTINGS_PATH
    all_settings = _load_all(path)
    all_settings.pop(session_id, None)  # re-insert at the end - keeps insertion order recency for the cap below
    all_settings[session_id] = asdict(settings)
    if len(all_settings) > MAX_SESSION_SETTINGS:
        for stale_id in list(all_settings)[: len(all_settings) - MAX_SESSION_SETTINGS]:
            del all_settings[stale_id]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(all_settings, f, indent=2)


def load_session_settings(session_id: str, path: Optional[str] = None) -> Optional[SessionSettings]:
    path = path if path is not None else DEFAULT_SESSION_SETTINGS_PATH
    data = _load_all(path).get(session_id)
    return SessionSettings(**data) if data is not None else None


def _load_all(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
