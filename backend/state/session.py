"""
In-memory session store for the FastAPI backend.

Each Session holds:
  - agent: the built HAgent — the whole scoring model (retains info; owns the default baseline)
  - info: the step-4 processed player data (pipeline intermediate; also readable via agent.info)
  - current_settings: snapshot of all mutable parameters, used to diff PATCH bodies
  - v0_clean: immutable copy of raw player stats (before any transformations)
  - v1_clean: immutable copy after injured players dropped (before upsilon)
  - v2: DataFrame after upsilon adjustment (input to process_player_data)
"""

import time
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.platform_integration.base import PlatformConfig


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Snapshot of current parameters — used to diff PATCH requests
    current_settings: dict = field(default_factory=dict)

    # Pipeline intermediate DataFrames (kept for resumable PATCH re-runs from a step).
    # All three are indexed by player id; display names live only in player_registry.
    v0_clean: Optional[pd.DataFrame] = None   # raw player stats
    v1_clean: Optional[pd.DataFrame] = None   # after drop_injured
    v2:       Optional[pd.DataFrame] = None   # after upsilon adjustment

    # The session's player identities: {player_id -> PlayerIdentity (name, last name,
    # positions, headshot availability)}. Built by load_player_pool alongside v0_clean — the two
    # travel together through every cache — and includes the RP sentinel. The ONLY place
    # names live server-side; everything else keys by id.
    player_registry: Optional[dict] = None

    # Step-4 processed player data (G-scores, positions, covariance, ...). A pipeline intermediate
    # kept for from_step==5 patches; consumers read it via session.agent.info.
    info: Optional[dict] = None

    # The built HAgent — the whole scoring model (retains info; owns the neutral baseline).
    # None until the pipeline runs.
    agent: Optional[object] = None

    # Recently built pipelines, keyed by the full parameter snapshot that produced them —
    # {key: {'agent', 'info', 'v0_clean', 'v1_clean', 'v2'}}. A PATCH stashes the current
    # build here before rebuilding, so returning to a configuration this session already
    # built (e.g. toggling blend weights back mid-draft) restores it instead of re-running
    # the pipeline and the expensive baseline H-scoring pass. Session-private: agents are
    # stateful and must never be shared across sessions. See session_management.apply_patch.
    pipeline_cache: "OrderedDict[tuple, dict]" = field(default_factory=OrderedDict)

    # Live-platform connection (None for 'Enter your own data'). Set during session
    # creation when a live platform is selected; used by the draft-state poll.
    platform_config: Optional[PlatformConfig] = None

    # {platform player id/name -> session player id} lookup, rebuilt from the registry
    # whenever the data changes (see refresh_platform_player_id_lookup). None until a
    # live platform is connected.
    platform_player_id_lookup: Optional[dict[str, int]] = None

    # Serialises work on this one session. An evaluate mutates shared agent state (warm-start
    # weights via get_h_scores), and a PATCH rebuilds the pipeline in place, so two overlapping
    # requests on the same session corrupt each other and 500. The frontend aborts its previous
    # request before firing the next, but an aborted fetch does not stop the server-side handler,
    # so the two still overlap here. Every request that reads or mutates agent/pipeline state holds
    # this lock, restoring the one-operation-at-a-time-per-session invariant the Streamlit app had.
    # Per session (not global) so different sessions still run fully in parallel.
    lock: threading.Lock = field(default_factory=threading.Lock)

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
        session_id for session_id, session in _store.items()
        if now - session.last_accessed > SESSION_TTL
    ]
    for session_id in expired_ids:
        del _store[session_id]


def create_session() -> Session:
    session_id = uuid.uuid4().hex[:8]
    session = Session(id=session_id)
    with _lock:
        # Reclaim abandoned sessions on each create. get_session only evicts a
        # session when it is actively looked up, so sessions that are never
        # queried again (e.g. the user closed the tab) would otherwise pin their
        # DataFrames in memory forever. Sweeping here bounds the store to the
        # active set plus whatever expired since the last create.
        _evict_expired_sessions(time.time())
        _store[session_id] = session
    return session


def get_session(session_id: str) -> Optional[Session]:
    with _lock:
        session = _store.get(session_id)
        if session is None:
            return None
        if time.time() - session.last_accessed > SESSION_TTL:
            # Expired on read: never serve a stale session. Bulk reclamation of
            # abandoned sessions happens in create_session via _evict_expired_sessions.
            del _store[session_id]
            return None
        session.last_accessed = time.time()
        return session


def delete_session(session_id: str) -> bool:
    with _lock:
        if session_id in _store:
            del _store[session_id]
            return True
        return False
