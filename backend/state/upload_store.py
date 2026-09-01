"""Store for user-uploaded projection CSVs, memory + disk, TTL-expiring.

Application state, a peer of the session store: the /data/upload route stashes an
uploaded file here and returns a short id; session creation/patch later pulls it back
out by that id (see _resolve_csv / _resolve_uploaded_dfs in api/routers/sessions.py).

Lifetime rules mirror the session store, because an upload must outlive every session
that references it — a session whose upload has vanished cannot be rebuilt at all
(session creation 404s), which strands the app until the page is reloaded:
  - the TTL is SLIDING (refreshed on every read), so a file in active use never dies
    mid-draft the way a fixed created_at expiry did;
  - the window is long (a day), since drafts and auctions routinely run for hours;
  - entries are persisted to disk, so a process restart does not strand live sessions.
    In dev that means an auto-reload (any backend edit) no longer wipes uploads; in
    production it means a cold start or a second instance still resolves the id.
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional

# Sliding (see the module docstring): the clock restarts on every read, so this bounds
# idle time, not total lifetime. Comfortably outlives the 4h session TTL.
UPLOAD_TTL     = 24 * 3600
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

_upload_store: dict[str, dict] = {}
_upload_lock = threading.Lock()

# Same DISK_CACHE_DIR convention as the Snowflake view cache and the headshot proxy;
# without the env var uploads still persist, in a gitignored local directory.
_UPLOAD_DISK_DIR = (
    Path(os.environ['DISK_CACHE_DIR']) / 'uploads'
    if 'DISK_CACHE_DIR' in os.environ else Path('.cache') / 'uploads'
)


def _upload_paths(data_id: str) -> tuple[Path, Path]:
    """(csv bytes path, metadata path) for an upload. The csv stays raw bytes so it is
    re-parsed exactly as it was received."""
    return (_UPLOAD_DISK_DIR / f'{data_id}.csv', _UPLOAD_DISK_DIR / f'{data_id}.json')


def _write_upload_to_disk(data_id: str, entry: dict) -> None:
    _UPLOAD_DISK_DIR.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = _upload_paths(data_id)
    csv_path.write_bytes(entry['bytes'])
    meta_path.write_text(json.dumps({
        'n_players':     entry['n_players'],
        'last_accessed': entry['last_accessed'],
    }), encoding='utf-8')


def _touch_upload_on_disk(data_id: str, last_accessed: float) -> None:
    """Persist a refreshed access time so the sliding window survives a restart."""
    _, meta_path = _upload_paths(data_id)
    if not meta_path.exists():
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return
    metadata['last_accessed'] = last_accessed
    meta_path.write_text(json.dumps(metadata), encoding='utf-8')


def _read_upload_from_disk(data_id: str) -> Optional[dict]:
    """Rehydrate an upload the current process never saw (restart, or another instance).
    Returns None when either file is missing or unreadable — treated exactly like an
    upload that was never stored."""
    csv_path, meta_path = _upload_paths(data_id)
    if not (csv_path.exists() and meta_path.exists()):
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding='utf-8'))
        return {
            'bytes':         csv_path.read_bytes(),
            'n_players':     metadata['n_players'],
            'last_accessed': float(metadata['last_accessed']),
        }
    except (OSError, ValueError, KeyError):
        return None


def _discard_upload(data_id: str) -> None:
    _upload_store.pop(data_id, None)
    for path in _upload_paths(data_id):
        path.unlink(missing_ok=True)


def _evict_expired_uploads(now: float) -> None:
    """Drop every upload past its idle window, memory and disk. Called on store (the
    same reclaim-on-write policy the session store uses), so abandoned uploads cannot
    accumulate on disk indefinitely."""
    expired = [data_id for data_id, entry in _upload_store.items()
               if now - entry['last_accessed'] > UPLOAD_TTL]
    for data_id in expired:
        _discard_upload(data_id)

    if not _UPLOAD_DISK_DIR.exists():
        return
    for meta_path in _UPLOAD_DISK_DIR.glob('*.json'):
        try:
            last_accessed = float(json.loads(meta_path.read_text(encoding='utf-8'))['last_accessed'])
        except (OSError, ValueError, KeyError):
            continue   # unreadable sidecar: left alone rather than guessed at
        if now - last_accessed > UPLOAD_TTL:
            _discard_upload(meta_path.stem)


def store_upload(data_id: str, csv_bytes: bytes, n_players: int) -> None:
    entry = {
        'bytes':         csv_bytes,
        'n_players':     n_players,
        'last_accessed': time.time(),
    }
    with _upload_lock:
        _evict_expired_uploads(entry['last_accessed'])
        _upload_store[data_id] = entry
        _write_upload_to_disk(data_id, entry)


def get_upload(data_id: str) -> Optional[dict]:
    """The stored upload, or None when it is unknown or has gone idle past the TTL.
    Reading refreshes the entry's clock (sliding window), so an upload backing an
    active session stays alive for as long as that session keeps using it."""
    now = time.time()
    with _upload_lock:
        entry = _upload_store.get(data_id)
        if entry is None:
            entry = _read_upload_from_disk(data_id)
            if entry is None:
                return None
            _upload_store[data_id] = entry
        if now - entry['last_accessed'] > UPLOAD_TTL:
            _discard_upload(data_id)
            return None
        entry['last_accessed'] = now
        _touch_upload_on_disk(data_id, now)
        return entry
