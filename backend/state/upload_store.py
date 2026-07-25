"""In-memory store for user-uploaded projection CSVs (HTB / BBM), TTL-expiring.

Application state, a peer of the session store: the /data/upload route stashes an
uploaded file here and returns a short id; session creation/patch later pulls it back
out by that id (see session_management._resolve_csv / _resolve_uploaded_dfs).
"""

from __future__ import annotations

import time
import threading
from typing import Optional

UPLOAD_TTL     = 2 * 3600          # 2 hours
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

_upload_store: dict[str, dict] = {}
_upload_lock = threading.Lock()


def store_upload(data_id: str, csv_bytes: bytes, file_type: str, n_players: int) -> None:
    with _upload_lock:
        _upload_store[data_id] = {
            'bytes':      csv_bytes,
            'file_type':  file_type,
            'n_players':  n_players,
            'created_at': time.time(),
        }


def get_upload(data_id: str) -> Optional[dict]:
    with _upload_lock:
        entry = _upload_store.get(data_id)
        if entry is None:
            return None
        if time.time() - entry['created_at'] > UPLOAD_TTL:
            del _upload_store[data_id]
            return None
        return entry
