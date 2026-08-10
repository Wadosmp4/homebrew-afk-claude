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
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(socket_path)
            sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
            sock.shutdown(socket.SHUT_WR)
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
    except OSError:
        # Companion isn't running / socket gone - fail open with an empty
        # decision rather than crashing the hook and blocking Claude Code.
        print(json.dumps({}))
        return 0

    sys.stdout.write(response.decode("utf-8") or "{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
