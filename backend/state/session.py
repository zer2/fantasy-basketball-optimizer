"""
In-memory session store for the FastAPI backend.

Each Session holds:
  - scorer: the built Scorer (HAgent + info + baseline cache) — the pipeline's output
  - current_params: snapshot of all mutable parameters, used to diff PATCH bodies
  - v0_clean: immutable copy of raw player stats (before any transformations)
  - v1_clean: immutable copy after injured players dropped (before upsilon)
  - v2: DataFrame after upsilon adjustment (input to process_player_data)
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.platform_integration.base import PlatformConfig
from backend.state.scorer import Scorer


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Snapshot of current parameters — used to diff PATCH requests
    current_params: dict = field(default_factory=dict)

    # Pipeline intermediate DataFrames (kept for resumable PATCH re-runs from a step)
    v0_clean: Optional[pd.DataFrame] = None   # raw player stats
    v1_clean: Optional[pd.DataFrame] = None   # after drop_injured
    v2:       Optional[pd.DataFrame] = None   # after upsilon adjustment

    # The built scoring model (HAgent + info + baseline cache); None until the pipeline runs.
    scorer: Optional[Scorer] = None

    # Live-platform connection (None for 'Enter your own data'). Set during session
    # creation when a live platform is selected; used by the draft-state poll.
    platform_config: Optional[PlatformConfig] = None

    # {platform player name -> canonical 'Name (POS)'} lookup, rebuilt from scorer.info
    # whenever the data changes (see refresh_platform_name_lookup). None until a
    # live platform is connected.
    platform_name_lookup: Optional[dict[str, str]] = None

# ── Store ────────────────────────────────────────────────────────────────────
# Grouped here rather than above Session because the _store annotation below is
# evaluated at import time, so Session must already be defined.

SESSION_TTL = 4 * 3600  # seconds

_store: dict[str, Session] = {}
_lock = threading.Lock()


# ── CRUD ─────────────────────────────────────────────────────────────────────
# Public API is in CRUD order (create / get / delete); _evict_expired_sessions
# leads because it belongs beside create_session, its only caller (same
# workflow-over-visibility grouping the platform integrations use).

def _evict_expired_sessions(now: float) -> None:
    """Remove every session past its TTL. Caller must hold _lock."""
    expired_ids = [
        sid for sid, session in _store.items()
        if now - session.last_accessed > SESSION_TTL
    ]
    for sid in expired_ids:
        del _store[sid]


def create_session() -> Session:
    sid = uuid.uuid4().hex[:8]
    session = Session(id=sid)
    with _lock:
        # Reclaim abandoned sessions on each create. get_session only evicts a
        # session when it is actively looked up, so sessions that are never
        # queried again (e.g. the user closed the tab) would otherwise pin their
        # DataFrames in memory forever. Sweeping here bounds the store to the
        # active set plus whatever expired since the last create.
        _evict_expired_sessions(time.time())
        _store[sid] = session
    return session


def get_session(sid: str) -> Optional[Session]:
    with _lock:
        session = _store.get(sid)
        if session is None:
            return None
        if time.time() - session.last_accessed > SESSION_TTL:
            # Expired on read: never serve a stale session. Bulk reclamation of
            # abandoned sessions happens in create_session via _evict_expired_sessions.
            del _store[sid]
            return None
        session.last_accessed = time.time()
        return session


def delete_session(sid: str) -> bool:
    with _lock:
        if sid in _store:
            del _store[sid]
            return True
        return False
