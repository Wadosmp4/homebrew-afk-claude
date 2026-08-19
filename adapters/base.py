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
(this plan's own `CodexAdapter` uses it too, for `respond_to_permission`)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional, Protocol, runtime_checkable

from .events import Event


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

    def set_session_auto_approve(
        self, session_id: str, auto_approve: Optional[bool] = None, llm_judge: Optional[bool] = None
    ) -> bool: ...

    def get_cwd(self, session_id: str) -> Optional[str]: ...

    def is_active(self, session_id: str) -> Optional[bool]: ...

    def emit_custom(self, session_id: str, type_: str, **data: Any) -> None: ...

    def subscribe(self, session_id: str) -> AsyncIterator[Event]: ...
