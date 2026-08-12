# Companion

Background daemon that authenticates to the relay and holds a persistent,
auto-reconnecting connection with heartbeats (U2). Adapters that discover
and drive/observe Claude Code sessions (U3, U4) hang off this daemon in
later units.

## Setup

### Via Homebrew (recommended for a fresh Mac)

```bash
brew install <tap>/afk-claude-companion
```

### From a checkout

```bash
python3 -m venv .venv          # from the repo root, shared with relay/
source .venv/bin/activate
pip install -r companion/requirements.txt
pip install .                  # from the repo root - installs the afk-claude-companion CLI
```

## Pairing a new machine

### `pair` (recommended)

One command on the new Mac - no database access, no token to copy by hand:

```bash
afk-claude-companion pair --relay-url https://your-relay-host
```

This prints a short code and waits. Open the mobile app, go to "Connect a
new Mac," and enter the code - your phone (already trusted) vouches for
this companion, `pair` picks up its new token automatically, and writes
`~/.config/remote-claude-companion/config.json` itself. See
`relay/pairing.py`'s module comment for the full request/approve/claim
design.

### `configure` (manual fallback)

If you already have a device token some other way - e.g. minted directly
against the relay's storage, which stays a relay-side operator action (see
`relay/auth.py`'s module docstring for why that's deliberately not an HTTP
endpoint):

```bash
python3 -m relay.bootstrap_companion \
  --db postgresql://user:pass@host/dbname \
  --relay-url wss://your-relay-host/ws/companion
```

This prints the exact command to run on the new Mac:

```bash
afk-claude-companion configure --relay-url wss://your-relay-host/ws/companion --device-token <token>
```

which writes the same config file (override the path with
`--config-path`, or `load_config`'s own argument if calling it directly).

### Getting your phone its first pairing code

Once a companion is configured (either path above), it has its own device
token - so it can request a phone pairing code itself, rather than you
hand-writing an authenticated `curl` call against `/pairing/register`:

```bash
afk-claude-companion pairing-code
```

Prints a relay address and a 6-digit code - enter both in the mobile
app's Pair screen. See `relay/README.md`'s "First-time setup" for the
full walkthrough from a fresh deploy through a paired phone.

## Running manually

```bash
afk-claude-companion run
# or, from a checkout without installing:
python3 -m companion.daemon
```

## Running as a background service

**Via Homebrew** (recommended): `brew services start afk-claude-companion` -
the formula's `service` block handles `launchd` registration for you.

**Manually**, from a checkout:

1. Copy `companion/launchd/com.remoteclaude.companion.plist` to
   `~/Library/LaunchAgents/`.
2. Replace `PYTHON_BIN` with your venv's `python3` path and `REPO_ROOT` with
   this checkout's absolute path.
3. `launchctl load ~/Library/LaunchAgents/com.remoteclaude.companion.plist`

Either way, the daemon should appear connected in the relay's device
registry within one heartbeat interval.

## Tests

```bash
pip install -r companion/requirements-dev.txt
pytest companion/
```
