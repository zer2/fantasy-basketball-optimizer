"""Small HTTP-layer helpers shared across routers."""

from __future__ import annotations

from datetime import datetime, timezone


def iso_expires(ttl_seconds: int) -> str:
    """Return an ISO-8601 UTC timestamp `ttl_seconds` from now (for `expires_at` fields)."""
    expires = datetime.now(timezone.utc).timestamp() + ttl_seconds
    return datetime.fromtimestamp(expires, tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
