# Companion

Background daemon that authenticates to the relay and holds a persistent,
auto-reconnecting connection with heartbeats (U2). Adapters that discover
and drive/observe Claude Code sessions (U3, U4) hang off this daemon in
later units.

## Setup

```bash
python3 -m venv .venv          # from the repo root, shared with relay/
source .venv/bin/activate
pip install -r companion/requirements.txt
```

## Config

The daemon reads `~/.config/remote-claude-companion/config.json` (override
with `main(config_path=...)` or by pointing `load_config` elsewhere). Create
it once per machine:

```python
from companion.config import CompanionConfig, save_config

save_config(
    "/Users/you/.config/remote-claude-companion/config.json",
    CompanionConfig(
        relay_url="wss://your-relay-host/ws/companion",
        device_token="<minted via relay.auth.bootstrap_companion_device>",
    ),
)
```

`device_token` comes from the relay's `bootstrap_companion_device` — an
operator action run once against the relay's db file (see
`relay/auth.py`'s module docstring), not an HTTP endpoint.

## Running manually

```bash
python3 -m companion.daemon
```

## Running as a `launchd` user agent

1. Copy `companion/launchd/com.remoteclaude.companion.plist` to
   `~/Library/LaunchAgents/`.
2. Replace `PYTHON_BIN` with your venv's `python3` path and `REPO_ROOT` with
   this checkout's absolute path.
3. `launchctl load ~/Library/LaunchAgents/com.remoteclaude.companion.plist`

The daemon should appear connected in the relay's device registry within
one heartbeat interval.

## Tests

```bash
pip install -r companion/requirements-dev.txt
pytest companion/
```
