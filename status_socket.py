"""Update Safety plan (U3), R4/R6, KTD10: a small Unix-domain socket server
answering exactly one question - how many sessions does the running
companion consider active - for `companion precheck` (companion/cli.py) to
ask before an operator runs `brew upgrade`.

Code review finding (maintainability): this used to live inline on
CompanionDaemon in companion/daemon.py, which was already the single
largest, most-contended file in this package before this plan's additions.
Extracted into its own module - dependencies threaded through the
constructor (a plain async callable) rather than ambient `self` access,
mirroring this codebase's own AgentLimitCoordinator extraction pattern
(companion/agent_limit_coordinator.py) - so CompanionDaemon composes this
rather than implementing it inline.

Deliberately NOT sharing implementation with ObserveAdapter's own hooks
socket (companion/adapters/observe_adapter.py), even though the two
lifecycles are close to identical (stale-socket cleanup, chmod 0600,
close+wait_closed+remove teardown) - consolidating them would mean
touching ObserveAdapter's connection-lifecycle code, which this repo's own
history (a whole-codebase review earlier in this project) flagged as
high-risk given past production incidents there. A cosmetic dedup was not
worth that risk.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


class StatusSocketServer:
    """Owns the status socket's full lifecycle. `count_active_sessions` is
    an async callable returning the current active-session count (see
    CompanionDaemon._count_active_sessions) - injected rather than taking a
    reference to the daemon itself, so this class's only dependency is the
    one value it actually needs."""

    def __init__(self, socket_path: str, count_active_sessions: Callable[[], Awaitable[int]]) -> None:
        self.socket_path = socket_path
        self._count_active_sessions = count_active_sessions
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        """Same lifecycle shape as ObserveAdapter.start()'s own hooks
        socket (adapters/observe_adapter.py): stale-socket cleanup from a
        prior crashed run, then 0600 permissions (local-only - no reason
        for another user on this machine to learn session counts)."""
        os.makedirs(os.path.dirname(self.socket_path) or ".", exist_ok=True)
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)  # stale socket from a prior crashed run
        self._server = await asyncio.start_unix_server(self._handle_connection, path=self.socket_path)
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600 - local privilege surface (Risk)

    async def stop(self) -> None:
        """Mirrors ObserveAdapter.stop()'s own socket teardown."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """R4/R5/R6: there's only ever one question a client can ask here,
        so unlike ObserveAdapter's hooks socket this never reads a request
        payload - it just answers as soon as a client connects. Writes the
        response unconditionally from a `finally`, mirroring
        ObserveAdapter._handle_hook_connection's own shape, so a bug in
        count_active_sessions can't leave `companion precheck` hanging on
        a connection that's never going to answer.

        Code review finding (testing/correctness, U3): an internal failure
        here used to answer with the bare `{"active_sessions": null}` a
        client also gets from a plain OSError (socket doesn't exist,
        connection refused) - `precheck` couldn't tell "the daemon is up
        but something broke counting sessions" apart from "the daemon
        isn't reachable at all", which matters to an operator deciding
        whether to upgrade anyway. The `error` field, present only on this
        branch, is precheck's signal to report the distinction."""
        response: dict[str, Any] = {"active_sessions": None}
        try:
            response = {"active_sessions": await self._count_active_sessions()}
        except Exception as exc:
            logger.exception("status socket: failed to count active sessions")
            response = {"active_sessions": None, "error": str(exc) or exc.__class__.__name__}
        finally:
            writer.write(json.dumps(response).encode("utf-8") + b"\n")
            with contextlib.suppress(Exception):
                await writer.drain()
            writer.close()
