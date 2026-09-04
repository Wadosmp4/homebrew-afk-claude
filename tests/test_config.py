"""Tests for companion/config.py's SQLite-backed storage (Companion-Local
Token Entry & SQLite Config plan). `path` throughout means the SQLite db
path (mirroring session_registry.py's own test conventions), not a JSON
file - see config.py's own module docstring for why."""
import json
import os
import sqlite3
import stat
import threading
import time

import pytest

import companion.config as config_module
from companion.config import CompanionConfig, load_config, save_config, update_config_field


def test_save_then_load_roundtrips_every_field(tmp_path):
    path = str(tmp_path / "session_registry.db")
    config = CompanionConfig(
        relay_url="ws://127.0.0.1:9000/ws/companion",
        device_token="secret-token",
        heartbeat_interval=7.5,
        cli_path="/usr/local/bin/claude-custom",
        cli_env={"ANTHROPIC_API_KEY": "sk-test"},
        observe_entrypoints=["claude-desktop", "claude-vscode"],
        observe_auto_approve=True,
        active_account="personal",
        personal_oauth_token="sk-ant-oat-test",
        codex_active_account="personal",
        codex_personal_api_key="sk-codex-test",
    )

    save_config(path, config)
    loaded = load_config(path)

    assert loaded == config


def test_calling_save_config_twice_updates_the_same_row_not_a_second_one(tmp_path):
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t1"))
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t2"))

    conn = sqlite3.connect(path)
    rows = conn.execute("SELECT id, device_token FROM config").fetchall()
    conn.close()

    assert rows == [(1, "t2")]
    assert load_config(path).device_token == "t2"


def test_a_json_fixture_missing_fields_added_later_still_loads_with_dataclass_defaults(tmp_path, monkeypatch):
    """Forward-compatibility: a config.json written before observe_entrypoints/
    cli_path/active_account/codex_* existed must still migrate cleanly,
    defaulting every missing field - same guarantee the pre-migration
    CompanionConfig(**json.load(f)) construction already provided."""
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))

    db_path = str(tmp_path / "session_registry.db")
    loaded = load_config(db_path)

    assert loaded.observe_entrypoints == ["claude-desktop"]
    assert loaded.cli_path is None
    assert loaded.cli_env == {}
    assert loaded.active_account == "vscode"
    assert loaded.personal_oauth_token is None
    assert loaded.codex_active_account == "vscode"
    assert loaded.codex_personal_api_key is None


def test_load_config_with_no_row_and_no_legacy_json_raises_file_not_found(tmp_path, monkeypatch):
    """Matches the pre-migration behavior exactly: opening a config that
    has genuinely never been written yet raised FileNotFoundError (a
    nonexistent JSON file) - there is no meaningful "fresh default"
    CompanionConfig to synthesize instead, since relay_url/device_token
    have no default value at all."""
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(tmp_path / "config.json"))
    db_path = str(tmp_path / "session_registry.db")

    with pytest.raises(FileNotFoundError):
        load_config(db_path)


# --- Migration (AE2) -------------------------------------------------------


def test_migrating_an_existing_json_config_preserves_device_token_and_personal_oauth_token(tmp_path, monkeypatch):
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(
        json.dumps(
            {
                "relay_url": "ws://x/ws/companion",
                "device_token": "real-device-token",
                "personal_oauth_token": "sk-ant-oat-real",
            }
        )
    )
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))
    db_path = str(tmp_path / "session_registry.db")

    loaded = load_config(db_path)

    assert loaded.device_token == "real-device-token"
    assert loaded.personal_oauth_token == "sk-ant-oat-real"

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id FROM config").fetchall()
    conn.close()
    assert rows == [(1,)]

    # The original file exists renamed to config.json.migrated immediately
    # after migration.
    assert not legacy_json_path.exists()
    migrated_path = tmp_path / "config.json.migrated"
    assert migrated_path.exists()


def test_the_migrated_recovery_copy_is_deleted_after_the_next_successful_save(tmp_path, monkeypatch):
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))
    db_path = str(tmp_path / "session_registry.db")

    loaded = load_config(db_path)
    migrated_path = tmp_path / "config.json.migrated"
    assert migrated_path.exists()

    save_config(db_path, loaded)

    assert not migrated_path.exists()


def test_a_second_load_config_call_after_migration_reads_directly_from_sqlite(tmp_path, monkeypatch):
    """Simulates a daemon restart after migration already happened - no
    re-read or re-migration of the (now-renamed) JSON file."""
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "original"}))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))
    db_path = str(tmp_path / "session_registry.db")

    first = load_config(db_path)
    assert first.device_token == "original"

    # Mutate the SQLite row directly (bypassing save_config) so a second
    # load_config that re-read the (gone) JSON instead would return the
    # stale "original" value rather than this one.
    conn = sqlite3.connect(db_path)
    conn.execute("UPDATE config SET device_token = ? WHERE id = 1", ("changed-after-migration",))
    conn.commit()
    conn.close()

    second = load_config(db_path)
    assert second.device_token == "changed-after-migration"


def test_os_rename_failing_after_the_migration_insert_committed_still_returns_correct_config(
    tmp_path, monkeypatch
):
    """KTD3: the SQLite row is already durable and correct the moment the
    transaction commits - a rename failure right after (process killed,
    permission error, whatever) must not lose or corrupt the migrated
    data, and a subsequent load_config must not attempt to re-migrate."""
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "t"}))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))
    db_path = str(tmp_path / "session_registry.db")

    with monkeypatch.context() as m:
        m.setattr(os, "rename", lambda *a, **k: (_ for _ in ()).throw(OSError("simulated rename failure")))
        loaded = load_config(db_path)
        assert loaded.device_token == "t"
        # The un-renamed config.json is left in place - an accepted outcome.
        assert legacy_json_path.exists()
    # os.rename is back to normal here (the `with` block above scoped the
    # patch to just the failing call) - DEFAULT_CONFIG_PATH is still
    # patched, from the outer monkeypatch fixture.

    second = load_config(db_path)
    assert second.device_token == "t"


def test_two_concurrent_load_config_calls_racing_the_first_migration_produce_exactly_one_row(tmp_path, monkeypatch):
    """KTD3's own race-safety guarantee: two processes calling load_config
    for the first time after an upgrade, at roughly the same moment,
    cannot both insert a row."""
    legacy_json_path = tmp_path / "config.json"
    legacy_json_path.write_text(json.dumps({"relay_url": "ws://x/ws/companion", "device_token": "raced-value"}))
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG_PATH", str(legacy_json_path))
    db_path = str(tmp_path / "session_registry.db")

    results: list[CompanionConfig] = []
    errors: list[Exception] = []

    def worker():
        try:
            results.append(load_config(db_path))
        except Exception as exc:  # pragma: no cover - only fires on a genuine race bug
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1] == CompanionConfig(relay_url="ws://x/ws/companion", device_token="raced-value")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT id FROM config").fetchall()
    conn.close()
    assert rows == [(1,)]


# --- update_config_field (code-review finding, P1) --------------------------


def test_update_config_field_sets_only_the_named_column(tmp_path):
    path = str(tmp_path / "session_registry.db")
    save_config(
        path,
        CompanionConfig(relay_url="ws://x/ws/companion", device_token="t", observe_entrypoints=["claude"]),
    )

    update_config_field(path, personal_oauth_token="sk-ant-oat-new")

    loaded = load_config(path)
    assert loaded.personal_oauth_token == "sk-ant-oat-new"
    # Every other field survives untouched - the whole point versus
    # save_config's full-row upsert.
    assert loaded.relay_url == "ws://x/ws/companion"
    assert loaded.device_token == "t"
    assert loaded.observe_entrypoints == ["claude"]


def test_update_config_field_does_not_clobber_a_concurrent_change_to_a_different_field(tmp_path):
    """The exact race the finding describes: two independent single-field
    updates for different columns, in either order, must both survive -
    neither one reads or rewrites a column it wasn't asked to touch."""
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    update_config_field(path, personal_oauth_token="sk-ant-oat-a")
    update_config_field(path, observe_entrypoints=["claude", "codex"])

    loaded = load_config(path)
    assert loaded.personal_oauth_token == "sk-ant-oat-a"
    assert loaded.observe_entrypoints == ["claude", "codex"]


def test_update_config_field_is_owner_only_readable(tmp_path):
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    update_config_field(path, personal_oauth_token="sk-ant-oat-new")

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


# --- Permissions (KTD4) -----------------------------------------------------


def test_save_config_is_owner_only_readable(tmp_path):
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_save_config_stays_owner_only_readable_with_personal_tokens_set(tmp_path):
    path = str(tmp_path / "session_registry.db")
    save_config(
        path,
        CompanionConfig(
            relay_url="ws://x/ws/companion",
            device_token="t",
            personal_oauth_token="sk-ant-oat-test",
            codex_personal_api_key="sk-codex-test",
        ),
    )

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_a_freshly_created_db_file_is_owner_only_readable_from_the_umask_alone(tmp_path, monkeypatch):
    """Regression: verifies the module-load-time os.umask(0o077) is doing
    real work, independent of save_config's own explicit chmod backstop -
    without the umask, a crash between the file being created and that
    chmod call would leave it at the OS-default (world-readable) mode."""
    monkeypatch.setattr(config_module, "_chmod_if_exists", lambda _path: None)
    path = str(tmp_path / "session_registry.db")

    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


def test_a_leftover_journal_sidecar_is_also_chmodded_after_a_write(tmp_path):
    """secure_delete + journal_mode=DELETE (KTD4) mean the -journal file is
    normally deleted at commit, not left behind - this asserts the chmod
    call site itself is correct (a no-op via _chmod_if_exists's own
    existence check) rather than depending on a journal file actually
    surviving to be found, which would be flaky/implementation-detail-
    dependent across SQLite versions."""
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    # journal_mode=DELETE removes the sidecar on a clean commit - confirms
    # this test's premise (nothing should be left over to leak permissions
    # on) rather than asserting a specific mode on a file that may not exist.
    assert not os.path.exists(path + "-journal")


def test_load_config_chmods_the_db_on_the_existing_row_fast_path(tmp_path):
    """code-review finding (P0-equivalent): load_config's existing-row
    fast path - the one every daemon startup actually hits - must chmod the
    db back to owner-only-readable even if its permissions were loosened
    by something outside this module's own write paths (a backup/restore,
    `cp -p`, a manual edit), not just leave it as found."""
    path = str(tmp_path / "session_registry.db")
    save_config(path, CompanionConfig(relay_url="ws://x/ws/companion", device_token="t"))

    # Loosen the permissions to simulate an externally-touched file.
    os.chmod(path, 0o644)
    assert stat.S_IMODE(os.stat(path).st_mode) != (stat.S_IRUSR | stat.S_IWUSR)

    load_config(path)  # existing-row fast path, not the migration branch

    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == stat.S_IRUSR | stat.S_IWUSR


# --- Duplicate-column ALTER race (P2) ---------------------------------------


def test_two_concurrent_connects_racing_a_forward_migration_do_not_raise(tmp_path, monkeypatch):
    """Two processes (simulated here as two threads) racing _connect()'s
    forward-migration ALTER TABLE ADD COLUMN for the same missing column
    can both see "column doesn't exist yet" - the loser must treat SQLite's
    "duplicate column name" error as a harmless no-op (someone else already
    added it), not crash."""
    path = str(tmp_path / "session_registry.db")

    # Pre-create the table with every column except one, simulating an
    # on-disk db from before that field existed - exactly the scenario
    # _connect's own forward-migration loop exists for.
    missing_field = "codex_personal_api_key"
    present_fields = [f for f in config_module._CONFIG_FIELDS if f != missing_field]
    columns_ddl = ", ".join(f"{name} {config_module._column_type(name)}" for name in present_fields)
    setup_conn = sqlite3.connect(path)
    setup_conn.execute(f"CREATE TABLE config (id INTEGER PRIMARY KEY CHECK (id = 1), {columns_ddl})")
    setup_conn.commit()
    setup_conn.close()

    # Force both threads' ALTER TABLE calls to line up at (almost) the same
    # instant, so one genuinely races the other's already-committed change
    # instead of just happening to run sequentially. sqlite3.Connection is
    # an immutable C type (can't monkeypatch its `execute` directly), so a
    # thin subclass plugged in via sqlite3.connect's own `factory` param -
    # itself reached through a patched `sqlite3.connect`, since config.py
    # calls that as a plain module-level function - does the same job.
    barrier = threading.Barrier(2)
    real_connect = sqlite3.connect

    class _BarrierConnection(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if f"ALTER TABLE config ADD COLUMN {missing_field}" in sql:
                barrier.wait(timeout=5)
            return super().execute(sql, *args, **kwargs)

    def patched_connect(*args, **kwargs):
        kwargs.setdefault("factory", _BarrierConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", patched_connect)

    errors: list[Exception] = []

    def worker():
        try:
            conn = config_module._connect(path)
            conn.close()
        except Exception as exc:  # pragma: no cover - only fires on a genuine race bug
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []

    # And the column really is there afterwards - the race didn't silently
    # skip adding it altogether.
    conn = sqlite3.connect(path)
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(config)")}
    conn.close()
    assert missing_field in existing_columns


# --- Schema migration data preservation (R7/R8/AE5) -------------------------


def test_an_existing_config_db_with_the_old_schema_gains_codex_personal_api_key_without_losing_data(tmp_path):
    """Regression test mirroring test_session_registry.py's own additive-
    migration test (KTD4): a real installation's on-disk config table can
    predate codex_personal_api_key entirely - it's the newest field on
    CompanionConfig, added by the Codex Agent Integration plan's U2 (see
    config.py's own dataclass comment) - so `CREATE TABLE IF NOT EXISTS` is
    a no-op against the already-existing table (it only ever fires for a
    brand-new file). _connect()'s ensure_table_columns call must therefore
    ALTER TABLE the column onto a table that already exists, not just rely
    on the updated CREATE TABLE statement.

    This writes the OLD schema directly via a bare sqlite3 connection -
    bypassing config.py's own _connect()/save_config entirely - to prove
    the migration path itself survives real pre-existing data, rather than
    only exercising a fresh empty database (which CREATE TABLE IF NOT
    EXISTS would already handle trivially, proving nothing about ALTER
    TABLE)."""
    path = str(tmp_path / "session_registry.db")

    # The immediately-prior schema: every current column except the newest
    # one (codex_personal_api_key).
    old_fields = [f for f in config_module._CONFIG_FIELDS if f != "codex_personal_api_key"]
    columns_ddl = ", ".join(f"{name} {config_module._column_type(name)}" for name in old_fields)

    old_data = {
        "relay_url": "ws://127.0.0.1:9000/ws/companion",
        "device_token": "pre-existing-token",
        "heartbeat_interval": 7.5,
        "reconnect_initial_delay": 2.0,
        "reconnect_max_delay": 45.0,
        "observe_scan_interval": 2.5,
        "observe_entrypoints": ["claude-desktop", "claude-vscode"],
        "observe_auto_approve": True,
        "observe_llm_judge": False,
        "cli_path": "/usr/local/bin/claude-custom",
        "cli_env": {"ANTHROPIC_API_KEY": "sk-test"},
        "risk_judge_use_api": False,
        "active_account": "personal",
        "personal_oauth_token": "sk-ant-oat-pre-existing",
        "codex_active_account": "personal",
    }
    values = tuple(config_module._to_column_value(name, old_data[name]) for name in old_fields)

    conn = sqlite3.connect(path)
    try:
        conn.execute(f"CREATE TABLE config (id INTEGER PRIMARY KEY CHECK (id = 1), {columns_ddl})")
        placeholders = ", ".join("?" for _ in old_fields)
        conn.execute(
            f"INSERT INTO config (id, {', '.join(old_fields)}) VALUES (1, {placeholders})",
            values,
        )
        conn.commit()
    finally:
        conn.close()

    # The next real access to this file (not a fresh CREATE TABLE IF NOT
    # EXISTS - that path is a no-op here) must add codex_personal_api_key
    # without disturbing the pre-existing row, and R7 says the old row's
    # data is never translated into a value for a column that didn't exist
    # when it was written - the migrated column reads as None (dataclass
    # default), not backfilled.
    loaded = load_config(path)
    expected = CompanionConfig(**old_data)  # codex_personal_api_key defaults to None
    assert loaded == expected


def test_load_configs_own_default_parameter_points_at_the_shared_session_registry_path():
    """R5/KTD2: load_config's `path` default is a literal default-argument
    expression (`path: str = DEFAULT_SESSION_REGISTRY_PATH`), matching the
    original JSON-era signature's exact style (`path: str =
    DEFAULT_CONFIG_PATH`) - only which constant it points at changed. A
    literal default-argument expression binds once, at function-definition
    time (module import), so it can't be exercised via monkeypatch-then-
    call-with-no-args the way session_registry.py's own Optional[str] =
    None + body-level lookup can - this checks the actual bound default
    value directly instead, which is what R5's "unchanged call shape,
    changed default path" really requires."""
    import inspect

    default = inspect.signature(load_config).parameters["path"].default
    assert default == config_module.DEFAULT_SESSION_REGISTRY_PATH
