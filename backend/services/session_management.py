"""Session lifecycle orchestration: build a session, and re-run it on patch.

The request-driven application logic behind POST/PATCH /sessions. It stays HTTP-free and
never imports the request schemas: the sessions router maps the request into plain dicts,
resolves inputs (raising any 4xx), and shapes the response; here we only assemble the
session, run the pipeline, and maintain the platform name lookup. The low-level store
(create/get/delete) lives in session.py — this is the tier above it.
"""

from __future__ import annotations

from typing import Optional

from backend.state.session import Session, create_session, delete_session
from backend.services.build_agent import build_agent
from backend.data_retrieval import get_unified_player_table
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.helpers import build_platform_name_lookup


def refresh_platform_name_lookup(session: Session) -> None:
    """Rebuild the session's platform name lookup from its current info.

    Precondition: a live platform is connected (session.platform_config is set) — there is nothing
    to refresh otherwise, so callers guard on it. Lives here, not in build_agent, so the pipeline
    stays platform-agnostic. Call after the pipeline runs when the player set may have changed
    (session creation and data/injured patches, from_step <= 2); model/category/slot patches leave
    info['Positions'] untouched.
    """
    session.platform_name_lookup = build_platform_name_lookup(
        session.agent.info,
        session.platform_config.player_name_column,
        get_unified_player_table(),
    )


def build_session(
    current_params: dict
    , platform_config: Optional[PlatformConfig]
    , csv_bytes: Optional[bytes]
    , file_type: Optional[str]
    , uploaded_dfs: Optional[dict]
) -> Session:
    """Create a session from already-built params + resolved inputs, run the pipeline, return it.

    HTTP-free: on pipeline failure it cleans up its own session and re-raises; the router maps
    that to a 500.
    """
    session = create_session()
    session.current_params = current_params
    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())
    try:
        build_agent(session, from_step=1, csv_bytes=csv_bytes, file_type=file_type,
                     uploaded_dfs=uploaded_dfs)
        if session.platform_config is not None:
            refresh_platform_name_lookup(session)
    except Exception:
        delete_session(session.id)
        raise
    return session


def apply_patch(
    session: Session
    , patch: dict
    , from_step: int
    , platform_config: Optional[PlatformConfig]
    , csv_bytes: Optional[bytes]
    , file_type: Optional[str]
    , uploaded_dfs: Optional[dict]
) -> None:
    """Apply an already-built patch dict to a session and re-run the pipeline from from_step.

    platform_config feeds only the draft-state poll + name lookup, never the pipeline, so
    connecting a platform needs no pipeline rerun of its own.
    """
    session.current_params.update(patch)

    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())

    build_agent(session, from_step=from_step, csv_bytes=csv_bytes, file_type=file_type,
                 uploaded_dfs=uploaded_dfs)

    # Rebuild the lookup when either of its inputs changed: the player set (from_step <= 2)
    # or player_name_column (a platform_config was just set).
    if session.platform_config is not None and (from_step <= 2 or platform_config is not None):
        refresh_platform_name_lookup(session)
