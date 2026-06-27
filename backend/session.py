"""
In-memory session store for the FastAPI backend.

Each Session holds:
  - H: the HAgent (set after step 5)
  - info: the info dict (set after step 4)
  - current_params: snapshot of all mutable parameters, used to diff PATCH bodies
  - v0_clean: immutable copy of raw player stats (before any transformations)
  - v1_clean: immutable copy after injured players dropped (before upsilon)
  - v2: DataFrame after upsilon adjustment (input to process_player_data)

Also exports PositionConfig / build_position_config — pre-computed position
structure derived from sport params + slot_counts, passed explicitly to backend
math functions instead of reading from st.session_state.
"""

import time
import threading
import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.platform_integration.base import PlatformConfig


# ── Position config ───────────────────────────────────────────────────────────

@dataclass
class PositionConfig:
    position_structure: dict
    position_numbers:   dict[str, int]
    position_ranges:    dict[str, dict]
    position_indices:   dict[str, list[int]]


def build_position_config(params: dict, slot_counts: dict) -> PositionConfig:
    """Build a PositionConfig from sport-level params and the session's slot_counts."""
    position_structure = params['position_structure']
    base_list  = position_structure['base_list']
    flex_list  = position_structure['flex_list']
    all_positions = base_list + flex_list

    position_numbers = {pos: slot_counts.get(pos, 0) for pos in all_positions}

    start = 0
    position_ranges: dict[str, dict] = {}
    for pos in all_positions:
        end = start + position_numbers[pos]
        position_ranges[pos] = {'start': start, 'end': end}
        start = end

    flex_info = position_structure['flex']
    position_indices = {
        pos: [i for i, val in enumerate(base_list) if val in flex_info[pos]['bases']]
        for pos in flex_list
    }

    return PositionConfig(
        position_structure = position_structure,
        position_numbers   = position_numbers,
        position_ranges    = position_ranges,
        position_indices   = position_indices,
    )

SESSION_TTL = 4 * 3600  # seconds


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Snapshot of current parameters — used to diff PATCH requests
    current_params: dict = field(default_factory=dict)

    # Pipeline intermediate DataFrames
    v0_clean: Optional[pd.DataFrame] = None   # raw player stats
    v1_clean: Optional[pd.DataFrame] = None   # after drop_injured
    v2:       Optional[pd.DataFrame] = None   # after upsilon adjustment

    # Pipeline outputs
    info: Optional[dict]   = None   # process_player_data result
    H:    Optional[object] = None   # HAgent instance

    # Cached baseline H-scores from a run with no players taken.
    # Used for gnrc_dollar / orig_dollar auction values.
    # Invalidated whenever run_step5 rebuilds the HAgent.
    generic_h_scores: Optional[object] = None  # pd.Series when populated

    # Live-platform connection (None for 'Enter your own data'). Set during session
    # creation when a live platform is selected; used by the draft-state poll.
    platform_config: Optional[PlatformConfig] = None

    # {platform player name -> canonical 'Name (POS)'} lookup, rebuilt from info
    # whenever the data changes (see _refresh_platform_name_lookup). None until a
    # live platform is connected.
    platform_name_lookup: Optional[dict[str, str]] = None


_store: dict[str, Session] = {}
_lock = threading.Lock()


# ── CRUD ─────────────────────────────────────────────────────────────────────

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
