"""Tiny bridge invoked as a Claude Code hook command (installed by
hooks_installer.py): reads the hook's JSON payload from stdin, forwards it
plus the event name to the companion's local Unix socket, and prints
whatever the companion sends back to stdout - that's what Claude Code
reads as the hook's structured output (e.g. `permissionDecision`).

Kept dependency-free (stdlib only) since it runs as a subprocess Claude
Code spawns per hook firing, not as part of the companion's own process.
"""
from __future__ import annotations

import json
import socket
import sys
from typing import Optional


def read_unix_socket_response(
    socket_path: str, payload: Optional[bytes] = None, timeout: Optional[float] = None
) -> bytes:
    """Connects to a local Unix-domain socket, optionally sends `payload`
    (half-closing the write side afterward so the server sees EOF), and
    reads until the server closes its end. Shared by this module's own
    `main()` below and companion/cli.py's `precheck` (U3) - both are plain
    request/response clients of a companion-owned Unix socket and were
    hand-rolling the same connect+recv-until-close loop before this
    extraction, which had already let the two drift (only one had a
    `timeout`)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        if timeout is not None:
            sock.settimeout(timeout)
        sock.connect(socket_path)
        if payload is not None:
            sock.sendall(payload)
            sock.shutdown(socket.SHUT_WR)
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        return response


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(json.dumps({}), file=sys.stdout)
        return 1

    event_name, socket_path = argv[1], argv[2]
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        payload = {}
    payload["_hook_event"] = event_name

    try:
        response = read_unix_socket_response(socket_path, json.dumps(payload).encode("utf-8") + b"\n")
    except OSError:
        # Companion isn't running / socket gone - fail open with an empty
        # decision rather than crashing the hook and blocking Claude Code.
        print(json.dumps({}))
        return 0

    sys.stdout.write(response.decode("utf-8") or "{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
