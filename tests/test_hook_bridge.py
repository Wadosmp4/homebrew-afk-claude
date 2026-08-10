"""Tests companion/hook_bridge.py as a real subprocess talking to a real
Unix socket server - this is the actual code path Claude Code invokes as a
hook command, so it's worth exercising as a subprocess rather than only
via a Python import."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _short_socket_path() -> str:
    return f"/tmp/rc-bridge-{uuid.uuid4().hex[:8]}.sock"


@pytest.mark.asyncio
async def test_bridge_forwards_stdin_and_prints_response():
    socket_path = _short_socket_path()
    received = {}

    async def handle(reader, writer):
        raw = await reader.readuntil(b"\n")
        received["payload"] = json.loads(raw.decode())
        writer.write(json.dumps({"permissionDecision": "allow"}).encode() + b"\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_unix_server(handle, path=socket_path)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "companion.hook_bridge",
            "PermissionRequest",
            socket_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            cwd=str(REPO_ROOT),
        )
        stdin_payload = json.dumps({"session_id": "s1", "tool_name": "Bash"}).encode()
        stdout, _ = await proc.communicate(stdin_payload)

        assert json.loads(stdout.decode()) == {"permissionDecision": "allow"}
        assert received["payload"]["_hook_event"] == "PermissionRequest"
        assert received["payload"]["session_id"] == "s1"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_bridge_fails_open_when_socket_missing():
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "companion.hook_bridge",
        "SessionStart",
        "/tmp/definitely-not-a-real-socket.sock",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd=str(REPO_ROOT),
    )
    stdout, _ = await proc.communicate(json.dumps({"session_id": "s1"}).encode())
    assert json.loads(stdout.decode()) == {}
    assert proc.returncode == 0
