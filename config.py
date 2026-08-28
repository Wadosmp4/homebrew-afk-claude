"""Companion configuration: relay URL and device identity.

Contract for U1 (relay): `device_token` is a companion-kind bearer token.
The normal path to getting one is `afk-claude-companion setup` (cli.py) -
claiming a one-time bootstrap code a relay operator generated locally
via `relay/bootstrap_companion.py` (an operator action against the
database, not an HTTP call - see relay/auth.py's module docstring for
why minting stays off the network).

Companion-Local Token Entry & SQLite Config plan: storage moved from a
standalone JSON file to a `config` table inside the companion's existing
`session_registry.db` (KD2/KTD1-KTD4) - one more local SQLite table,
following the exact `PRAGMA table_info` + `ALTER TABLE ... ADD COLUMN`
migration shape and `INSERT ... ON CONFLICT DO UPDATE` upsert shape
session_registry.py's own `sessions` table already uses, rather than a
second, differently-shaped local storage mechanism. `load_config`/
`save_config` keep their exact existing signatures and call shape (R5) -
`path` now means the SQLite db path (defaulting to
session_registry.DEFAULT_SESSION_REGISTRY_PATH), not a JSON file path.
`DEFAULT_CONFIG_PATH` is kept as a named constant purely as the legacy
JSON migration-source path `load_config` checks on its very first call
after an upgrade (see load_config's own docstring) - it is no longer what
`load_config`/`save_config` read or write themselves.

The stored data holds secrets (`device_token`, `personal_oauth_token`,
`codex_personal_api_key`), so every file this module creates carries the
same owner-only-readable posture the old JSON file had: a process-wide
`os.umask(0o077)` set once at import time (before this module's own first
connection, so the db file and its `-journal` sidecar are both born at
safe permissions - a post-hoc chmod alone only protects the success path,
not a crash exactly mid-transaction that leaves a journal page exposed at
the OS default instead), backed up by an explicit `os.chmod(0o600)` on
both files after every write.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sqlite3
import stat
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .session_registry import DEFAULT_SESSION_REGISTRY_PATH

# Legacy JSON location (pre-migration) - see this module's own docstring
# and load_config's migration branch. Not read or written by save_config
# at all anymore; load_config only ever reads it once, lazily, the first
# time no SQLite row exists yet.
#
# Deliberately read as the bare module-level name at call time inside
# save_config/load_config below (via `DEFAULT_CONFIG_PATH`, not a
# precomputed `DEFAULT_CONFIG_PATH + ".migrated"` constant) - a test
# monkeypatching this name (mirroring test_session_registry.py's own
# DEFAULT_SESSION_REGISTRY_PATH convention) needs the migrated-path
# derivation to follow the patched value, which a module-load-time-frozen
# constant would not.
DEFAULT_CONFIG_PATH = os.path.expanduser("~/.config/remote-claude-companion/config.json")

# `setup`/`pair` (cli.py) fall back to this when --relay-url isn't passed,
# mirroring how the mobile app bakes in EXPO_PUBLIC_RELAY_URL at build time
# (mobile/screens/PairingScreen.tsx) - this is a personal, single-relay
# app, so there's no real reason to type the same hostname every time.
# None (unset) is a legitimate value: the CLI just asks for --relay-url
# explicitly in that case instead of silently failing.
DEFAULT_RELAY_URL = os.environ.get("AFK_RELAY_URL")

# Set once, at module import time, before this module's own first
# connection - see this module's own docstring for why. Process-wide by
# design (os.umask has no per-module scope); a companion daemon/CLI
# process has no other file-creation code this could conflict with.
os.umask(0o077)


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
    # Codex Agent Integration plan (002), U2 (R8/R9/R10/PKTD3): Codex's own
    # independent active_account/token pair, deliberately never read or
    # written by any Claude-account code path (and vice versa) - mirrors
    # active_account/personal_oauth_token's shape exactly, one level over,
    # so switching one agent's identity can never bleed into the other's.
    codex_active_account: str = "vscode"
    codex_personal_api_key: Optional[str] = None


# --- SQLite-backed storage (Companion-Local Token Entry & SQLite Config plan) ---

# KTD1: which fields need non-TEXT column affinity / value conversion.
# Every other field is a plain str/Optional[str] and round-trips through
# TEXT with no conversion. Hand-maintained rather than reflected from
# CompanionConfig's own dataclass field types - `from __future__ import
# annotations` makes those types plain strings at runtime (postponed
# evaluation), which would need fragile parsing to use reliably here.
_JSON_FIELDS = frozenset({"observe_entrypoints", "cli_env"})
_BOOL_FIELDS = frozenset({"observe_auto_approve", "observe_llm_judge", "risk_judge_use_api"})
_FLOAT_FIELDS = frozenset({"heartbeat_interval", "reconnect_initial_delay", "reconnect_max_delay", "observe_scan_interval"})

# Shared across CREATE TABLE, INSERT, and SELECT below so adding/renaming a
# CompanionConfig field can't drift between them - same discipline
# session_registry.py's own _COLUMNS constant documents for its table.
_CONFIG_FIELDS: list[str] = [f.name for f in dataclasses.fields(CompanionConfig)]
_CONFIG_COLUMNS = ", ".join(_CONFIG_FIELDS)
_CONFIG_PLACEHOLDERS = ", ".join("?" for _ in _CONFIG_FIELDS)
_CONFIG_UPDATE_SET = ", ".join(f"{name} = excluded.{name}" for name in _CONFIG_FIELDS)


def _column_type(name: str) -> str:
    if name in _BOOL_FIELDS:
        return "INTEGER"
    if name in _FLOAT_FIELDS:
        return "REAL"
    return "TEXT"  # plain str/Optional[str] fields, and the two JSON-serialized ones


def _to_column_value(name: str, value: Any) -> Any:
    if name in _JSON_FIELDS:
        return json.dumps(value)
    if name in _BOOL_FIELDS:
        return int(value)
    return value


def _from_column_value(name: str, value: Any) -> Any:
    if name in _JSON_FIELDS:
        # A freshly-migrated or never-written row can't reach here with a
        # NULL JSON column (the migration/save path always writes a real
        # value for every field), but a defensive fallback to the
        # dataclass's own default is cheap insurance against a malformed
        # row rather than a confusing json.loads(None) crash.
        return json.loads(value) if value is not None else None
    if name in _BOOL_FIELDS:
        return bool(value) if value is not None else None
    return value


def _row_to_config(row: tuple) -> CompanionConfig:
    data = {name: _from_column_value(name, value) for name, value in zip(_CONFIG_FIELDS, row)}
    return CompanionConfig(**data)


def _connect(path: str) -> sqlite3.Connection:
    """KTD1/KTD4: mirrors session_registry.py's own _connect() shape - a
    short-lived connection, secure_delete + journal_mode=DELETE set
    explicitly (this module's own posture, independent of whatever
    session_registry.py leaves at SQLite's factory default), a synthetic
    single-row primary key (`id INTEGER PRIMARY KEY CHECK (id = 1)` - no
    natural key like session_registry's own `session_id` exists for a
    single config row), and the same PRAGMA table_info + ALTER TABLE ADD
    COLUMN forward-migration check for a field added after this table
    already existed on some earlier run."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA secure_delete = ON")
    conn.execute("PRAGMA journal_mode = DELETE")
    columns_ddl = ", ".join(f"{name} {_column_type(name)}" for name in _CONFIG_FIELDS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS config (id INTEGER PRIMARY KEY CHECK (id = 1), {columns_ddl})")
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(config)")}
    for name in _CONFIG_FIELDS:
        if name not in existing_columns:
            conn.execute(f"ALTER TABLE config ADD COLUMN {name} {_column_type(name)}")
    return conn


def _chmod_if_exists(path: str) -> None:
    if os.path.exists(path):
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)


def save_config(path: str, config: CompanionConfig) -> None:
    """KTD1: a real single-row invariant enforced by the database itself
    (the `id = 1` CHECK constraint plus this upsert's own ON CONFLICT
    clause), not just by convention - calling this twice in a row updates
    the same row rather than accumulating duplicates.

    KTD7/R11: deletes the migration recovery copy (see load_config's own
    migration branch) once it's confirmed durable - the very next
    successful save after a migration means the SQLite row is the
    account's own living copy of the data from here on, so the JSON
    recovery copy's job is done."""
    data = asdict(config)
    values = tuple(_to_column_value(name, data[name]) for name in _CONFIG_FIELDS)
    with contextlib.closing(_connect(path)) as conn, conn:
        conn.execute(
            f"INSERT INTO config (id, {_CONFIG_COLUMNS}) VALUES (1, {_CONFIG_PLACEHOLDERS}) "
            f"ON CONFLICT(id) DO UPDATE SET {_CONFIG_UPDATE_SET}",
            values,
        )
    _chmod_if_exists(path)
    _chmod_if_exists(path + "-journal")
    migrated_path = DEFAULT_CONFIG_PATH + ".migrated"
    if os.path.exists(migrated_path):
        os.remove(migrated_path)


def load_config(path: str = DEFAULT_SESSION_REGISTRY_PATH) -> CompanionConfig:
    """KTD3: the existing-row check, the legacy-JSON read, and the
    migration INSERT all happen inside one BEGIN IMMEDIATE transaction, so
    two processes racing this same first-post-upgrade call can't both
    insert - the loser's INSERT hits the `id = 1` conflict and becomes a
    no-op DO UPDATE against data the winner already committed, rather than
    a second row (SQLite's own default busy-timeout, 5s, is what makes the
    loser wait for the winner's transaction instead of immediately raising
    "database is locked" - comfortably longer than this transaction's own
    read-JSON-then-INSERT work ever takes). The JSON-file rename that
    follows is a separate, non-transactional filesystem step, deliberately
    outside the transaction - if the process is killed between the commit
    and the rename, the SQLite row is already durable and correct (every
    subsequent load_config call reads it directly, never re-touching the
    JSON), but the original config.json is left un-renamed at its original
    path indefinitely. This is an accepted, harmless outcome (the
    "recovery fallback" the JSON file exists for is still met - it's still
    there, just not renamed), not a bug this function tries to eliminate.

    Raises FileNotFoundError when neither a SQLite row nor a legacy JSON
    file exists - matching this function's own pre-migration behavior for
    a config that has genuinely never been written yet (the original
    JSON-only load_config raised the same error opening a nonexistent
    file); there is no meaningful "fresh default" CompanionConfig to
    return instead, since relay_url/device_token have no default value at
    all (every real caller already treats a missing config as "not set up
    yet," not as a construction the loader can synthesize itself)."""
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(f"SELECT {_CONFIG_COLUMNS} FROM config WHERE id = 1").fetchone()
        if row is not None:
            conn.commit()
            return _row_to_config(row)

        if not os.path.exists(DEFAULT_CONFIG_PATH):
            conn.commit()
            raise FileNotFoundError(
                f"No companion config found (checked SQLite table at {path!r} and legacy file at "
                f"{DEFAULT_CONFIG_PATH!r})"
            )

        with open(DEFAULT_CONFIG_PATH) as f:
            legacy_data = json.load(f)
        config = CompanionConfig(**legacy_data)
        data = asdict(config)
        values = tuple(_to_column_value(name, data[name]) for name in _CONFIG_FIELDS)
        conn.execute(
            f"INSERT INTO config (id, {_CONFIG_COLUMNS}) VALUES (1, {_CONFIG_PLACEHOLDERS}) "
            f"ON CONFLICT(id) DO UPDATE SET {_CONFIG_UPDATE_SET}",
            values,
        )
        conn.commit()
    finally:
        conn.close()

    _chmod_if_exists(path)
    _chmod_if_exists(path + "-journal")
    try:
        os.rename(DEFAULT_CONFIG_PATH, DEFAULT_CONFIG_PATH + ".migrated")
    except OSError:
        pass
    return config
