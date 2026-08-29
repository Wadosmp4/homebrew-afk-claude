"""Shared fixtures for companion/tests/.

Companion-Local Token Entry & SQLite Config plan: config.py's load_config()
falls back to reading DEFAULT_CONFIG_PATH (the legacy config.json location)
for one-time migration whenever the requested db has no config row yet -
regardless of what `path` argument was actually passed in. Without this
autouse fixture, any test that exercises that fallback (e.g. a "is this Mac
already configured yet" check, or a "no config" error-guidance path) would
silently read - and in the worst case, act on - this machine's own real
companion config at ~/.config/remote-claude-companion/config.json, which can
mean a live network call using a real device token. Tests that want to
exercise migration explicitly monkeypatch DEFAULT_CONFIG_PATH themselves
afterward (see test_config.py), which safely overrides this default.
"""
from __future__ import annotations

import pytest

import companion.config as config_module


@pytest.fixture(autouse=True)
def _isolate_default_config_path(tmp_path, monkeypatch):
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(tmp_path / "unused-legacy-config.json"))
