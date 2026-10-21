"""Invitation codes: one string that carries both where to connect and (optionally) a one-time code.

    AFK1.<relay address, base64url, no padding>[.<one-time code>]

The service address travels inside the invitation, so it is never typed, never printed in the guide and
never written into the public source. (It is not a secret from the person who receives it - their own
computer has to know it - the point is that it stays out of public places.) The same format is made on the
relay side (relay/invite.py) and by the phone app (mobile/lib/invite.ts): they are separate copies of this
few-line format, kept in step by tests.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

PREFIX = "AFK1"


@dataclass(frozen=True)
class Invite:
    relay_url: str
    code: Optional[str] = None


def _check_url(url: str) -> str:
    parts = urlsplit(url)
    local = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme not in ("https", "http") or not parts.hostname or parts.username or parts.password:
        raise ValueError("the invitation does not contain a valid address")
    if parts.scheme == "http" and not local:
        raise ValueError("the invitation must use a secure (https) address")
    return url.rstrip("/")


def make_invite(relay_url: str, code: Optional[str] = None) -> str:
    encoded = base64.urlsafe_b64encode(_check_url(relay_url).encode()).decode().rstrip("=")
    return f"{PREFIX}.{encoded}" + (f".{code}" if code else "")


def parse_invite(text: str) -> Invite:
    """Raises ValueError for anything that is not a well-formed invitation."""
    parts = text.strip().split(".")
    if len(parts) not in (2, 3) or parts[0] != PREFIX or not parts[1]:
        raise ValueError("that is not an invitation code")
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        url = base64.urlsafe_b64decode(padded.encode()).decode()
    except Exception as exc:  # noqa: BLE001 - any decoding problem is just "malformed"
        raise ValueError("that is not an invitation code") from exc
    code = parts[2] if len(parts) == 3 and parts[2] else None
    return Invite(relay_url=_check_url(url), code=code)


def looks_like_invite(text: str) -> bool:
    return text.strip().startswith(PREFIX + ".")
