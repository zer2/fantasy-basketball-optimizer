"""
Persistent per-client credential storage for auth-based platforms.

Yahoo's OAuth tokens are stored as yfpy's token.json + private.json inside a
per-client directory; YahooFantasySportsQuery reads and refreshes them there. The
directory IS the persistence (survives restarts), keyed by the frontend-minted
client id. No secrets are held in memory.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional


# Base directory for persisted platform credentials (gitignored). Override with the
# PLATFORM_CREDENTIAL_DIR env var in deployment.
_CREDENTIAL_BASE = Path(os.environ.get('PLATFORM_CREDENTIAL_DIR', '.platform_credentials'))

# client_id is user-supplied and lands in a filesystem path, so constrain it hard to
# prevent path traversal.
_SAFE_CLIENT_ID = re.compile(r'^[A-Za-z0-9_-]{1,64}$')


def _validate_client_id(client_id: str) -> None:
    if not isinstance(client_id, str) or not _SAFE_CLIENT_ID.match(client_id):
        raise ValueError(f'Invalid client_id {client_id!r}; expected 1-64 chars of [A-Za-z0-9_-].')


def yahoo_auth_dir(client_id: str) -> str:
    """Return (creating if needed) the directory holding this client's Yahoo token files."""
    _validate_client_id(client_id)
    auth_dir = _CREDENTIAL_BASE / 'yahoo' / client_id
    auth_dir.mkdir(parents=True, exist_ok=True)
    return str(auth_dir)


def has_yahoo_credentials(client_id: str) -> bool:
    """True once a Yahoo token has been exchanged and persisted for this client."""
    _validate_client_id(client_id)
    return (_CREDENTIAL_BASE / 'yahoo' / client_id / 'token.json').exists()


# ── ESPN (s2 + SWID cookies) ──────────────────────────────────────────────────

def _espn_credentials_path(client_id: str) -> Path:
    _validate_client_id(client_id)
    return _CREDENTIAL_BASE / 'espn' / client_id / 'credentials.json'


def store_espn_credentials(client_id: str, s2: str, swid: str) -> None:
    """Persist this client's ESPN s2 + SWID cookies."""
    path = _espn_credentials_path(client_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'s2': s2, 'swid': swid}))


def get_espn_credentials(client_id: str) -> Optional[dict]:
    """Return {'s2': ..., 'swid': ...} for this client, or None if not stored."""
    path = _espn_credentials_path(client_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def has_espn_credentials(client_id: str) -> bool:
    """True once this client's ESPN cookies have been stored."""
    return _espn_credentials_path(client_id).exists()
