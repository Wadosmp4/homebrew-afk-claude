"""Multi-Agent Adapter plan (009), U1: the formal interface extracted from
`SDKAdapter` and `ObserveAdapter`'s real, already-shared shape - not
designed abstractly ahead of a concrete second implementation. A
`typing.Protocol` (structural typing) rather than an ABC: neither existing
class inherits from a common base today, and this codebase has no existing
`Protocol`/ABC precedent to follow instead (checked before choosing), so
`Protocol` is the option that requires zero changes to either class.

`daemon.py`'s `self.adapters: list[AdapterProtocol]` registry (U2) is the
only consumer that needs this as a real type, not just documentation - see
its own comment for why dispatch generalizes but adapter-specific call
sites don't.

Methods `ObserveAdapter` implements as `UnsupportedOperation` (e.g.
`send_message`, `interrupt`, `compact`, `disconnect`) stay in this
protocol rather than being split into a smaller required subset - that
already-established "this adapter can't do that, and says so" precedent
extends naturally to any future adapter with its own partial support
(this plan's own `CodexAdapter` uses it too, for `respond_to_permission`).

`set_session_auto_approve` deliberately is NOT part of this protocol
(permission-mode-picker plan, later than this one): `SDKAdapter` no
longer implements it at all, since permission behavior for an SDK-owned
session is now `set_session_permission_mode`/`set_session_model` instead
(see daemon.py's own explicit `if adapter is self.observe_adapter:`
guard around it, rather than dispatching through this protocol) - it's
`ObserveAdapter`'s own separate, untouched auto-approve/AI-judge
mechanism now, not a shared adapter capability. `ObserveAdapter` and
`CodexAdapter` may still implement it themselves; the protocol just
doesn't require it."""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from .events import Event


@dataclass(frozen=True)
class UnsupportedOperation:
    """A capability a given adapter genuinely doesn't have, reported rather
    than raised, so a phone tapping an unavailable control gets a clear
    "can't do that" instead of a generic error. Shared by every adapter with
    partial support (code-review fix: previously ObserveAdapter and
    CodexAdapter each redeclared an identically-shaped class under this same
    name) - `reason` has no default so each call site states its own,
    since "not supported for observed sessions" and "not supported for
    Codex sessions" genuinely differ."""

    operation: str
    reason: str


async def drain_session_events(session: Any) -> AsyncIterator[Event]:
    """Shared body for every adapter's `subscribe()` - each adapter's own
    `subscribe` delegates here (`async for event in drain_session_events(session):
    yield event`) rather than reimplementing this loop, given a `session`
    object exposing a `.events` queue (an `asyncio.Queue[Event]`; duck-typed
    rather than a shared base class, since `_Session`/`_CodexSession`/
    `_ObserveSession` are otherwise unrelated classes).

    Code-review fix: yielding a terminal `"session_ended"`/`"error"` event
    used to return immediately afterward, which could silently orphan any
    event *already sitting in the queue* behind it - concretely, a
    reactive handler (e.g. daemon.py's own rate-limit-offer logic) can
    append a new event to this same session's queue while a preceding
    event from the SAME queue is still being processed by the caller, and
    that new event can land behind an already-enqueued terminal one, where
    it would otherwise never be dequeued (the generator returns the
    instant it yields the terminal event). Originally fixed only in
    `CodexAdapter.subscribe()` (for a Codex hard-rejection's `"rate_limit"`
    immediately followed by `"error"`, both enqueued before the daemon's
    reaction to `"rate_limit"` could run) - a wider audit found the same
    latent bug in `SDKAdapter`/`ObserveAdapter`, unrelated to that specific
    scenario but sharing the identical root cause, so the fix now lives
    here once instead of being re-copied a third time. Only items ALREADY
    in the queue at this exact moment are drained - `get_nowait()` on an
    empty queue raises immediately rather than blocking, so this can never
    turn into an unbounded wait."""
    while True:
        event = await session.events.get()
        yield event
        if event.type in ("session_ended", "error"):
            while not session.events.empty():
                yield session.events.get_nowait()
            return


async def cancel_and_wait(task: Optional[Any]) -> None:
    """Shared teardown idiom repeated across every adapter's disconnect
    path and daemon.py's own forwarder teardown: cancel a background task
    and await it, swallowing the `CancelledError` that cancellation itself
    raises - without the `await`, the task stays pending past its owner's
    own teardown and surfaces later as a misattributed "Task was destroyed
    but it is pending!" warning against whatever runs next (see individual
    call sites' own prior comments on this exact failure mode). A `None`
    task (never started, or already cleared) is a no-op - callers don't
    each need their own guard for that."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


@runtime_checkable
class AdapterProtocol(Protocol):
    def discover_sessions(self) -> list[str]: ...

    async def send_message(self, session_id: str, text: str) -> Any: ...

    async def interrupt(self, session_id: str) -> Any: ...

    async def compact(self, session_id: str) -> Any: ...

    async def disconnect(self, session_id: str) -> Any: ...

    async def respond_to_permission(
        self, session_id: str, request_id: str, decision: str, *, message: str = ""
    ) -> Any: ...

    def get_cwd(self, session_id: str) -> Optional[str]: ...

    def is_active(self, session_id: str) -> Optional[bool]: ...

    def emit_custom(self, session_id: str, type_: str, **data: Any) -> None: ...

    def subscribe(self, session_id: str) -> AsyncIterator[Event]: ...
