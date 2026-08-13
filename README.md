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

### `setup` (recommended - any Mac, in any state)

One command, whether this is your very first Mac or your fifth - no
relay URL, database, or token ever typed by hand (`--relay-url` falls
back to the `AFK_RELAY_URL` environment variable if you've set one):

```bash
afk-claude-companion setup
```

- Already has a config? Just prints your phone's pairing code.
- No config, and you have a one-time bootstrap code your relay operator
  generated locally (`python3 -m relay.bootstrap_companion` - see
  `relay/README.md`'s "First-time setup")? Answer "y" when asked, enter
  it, and `setup` claims this companion's first device token, writes
  `~/.config/remote-claude-companion/config.json` itself, then
  immediately prints your phone's pairing code too.
- No config, and no bootstrap code - just an already-paired phone?
  Answer "n" and `setup` requests a short code, prints it, and waits for
  that phone to approve it from "Connect a new Mac," same as `pair`
  below.

### `pair` (the direct form, additional Macs)

If you already know a phone is paired and want to skip `setup`'s
question:

```bash
afk-claude-companion pair --relay-url https://your-relay-host
```

This prints a short code and waits. Open the mobile app, go to "Connect a
new Mac," and enter the code - your phone (already trusted) vouches for
this companion, `pair` picks up its new token automatically, and writes
the config itself. See `relay/pairing.py`'s module comment for the full
request/approve/claim design.

### `configure` (manual fallback)

If you already have a device token some other way (scripting, etc):

```bash
afk-claude-companion configure --relay-url wss://your-relay-host/ws/companion --device-token <token>
```

which writes the same config file (override the path with
`--config-path`, or `load_config`'s own argument if calling it directly).

### Getting another phone a pairing code

Once a companion is configured (any path above), it has its own device
token - so it can request a phone pairing code itself, rather than you
hand-writing an authenticated `curl` call against `/pairing/register`.
`setup` already does this automatically; to get another one later
without re-running `setup`:

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
