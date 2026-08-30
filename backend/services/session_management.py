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
from backend.platform_integration.helpers import build_platform_player_id_lookup


def refresh_platform_player_id_lookup(session: Session) -> None:
    """Rebuild the session's platform-key -> player-id lookup from its current registry.

    Precondition: a live platform is connected (session.platform_config is set) — there is nothing
    to refresh otherwise, so callers guard on it. Lives here, not in build_agent, so the pipeline
    stays platform-agnostic. Call after the pipeline runs when the player set may have changed
    (session creation and data/injured patches, from_step <= 2); model/category/slot patches leave
    the registry untouched.
    """
    session.platform_player_id_lookup = build_platform_player_id_lookup(
        session.player_registry,
        session.platform_config.player_name_column,
        get_unified_player_table(),
    )


def normalize_objective_settings(current_params: dict) -> None:
    """Settle the Head-to-Head objective dial and tiebreaker against the format, in place.

    Under Rotisserie the dial is pinned to None: that format ignores it, HAgent rejects a number
    there, and the pipeline cache is keyed on the whole parameter snapshot — so a leftover slider
    value would split the cache into entries that build identical Rotisserie agents. Under Head to
    Head the dial is required, and its absence raises rather than defaulting, since guessing would
    mean drafting to an objective the caller never chose.

    The tiebreaker is pinned to None wherever it cannot bite — under Rotisserie, at dial 0 (each
    category scored on its own has no ties to break), and with an odd number of categories, where
    the majority is already decided. Pinned rather than rejected because a client may legitimately
    keep the choice while the count is briefly odd, so that it returns when the count is even
    again; what must not happen is that inert value splitting the cache.

    Applied wherever current_params is assembled or patched, so no caller has to remember.
    """
    if current_params['scoring_format'] == 'Rotisserie':
        current_params['most_categories_weight'] = None
        current_params['tiebreaker_category']    = None
        return

    weight = current_params['most_categories_weight']
    if weight is None or not 0.0 <= weight <= 1.0:
        raise ValueError('most_categories_weight must be between 0 and 1 for Head to Head '
                         f'(0 = Each Category, 1 = Most Categories). Got {weight!r}.')

    categories = current_params['categories']
    tiebreaker = current_params['tiebreaker_category']
    if tiebreaker is not None and (weight == 0
                                   or len(categories) % 2 == 1
                                   or tiebreaker not in categories):
        current_params['tiebreaker_category'] = None


def build_session(
    current_params: dict
    , platform_config: Optional[PlatformConfig]
    , csv_bytes: Optional[bytes]
    , uploaded_dfs: Optional[dict]
) -> Session:
    """Create a session from already-built params + resolved inputs, run the pipeline, return it.

    HTTP-free: on pipeline failure it cleans up its own session and re-raises; the router maps
    that to a 500.
    """
    session = create_session()
    normalize_objective_settings(current_params)
    session.current_params = current_params
    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())
    try:
        build_agent(session, from_step=1, csv_bytes=csv_bytes, uploaded_dfs=uploaded_dfs)
        if session.platform_config is not None:
            refresh_platform_player_id_lookup(session)
    except Exception:
        delete_session(session.id)
        raise
    return session


# How many previously built pipelines each session keeps (the current one is not counted).
# Sized for the toggling-a-setting-back-and-forth pattern, small because each entry holds
# the agent and every pipeline intermediate.
_PIPELINE_CACHE_ENTRIES = 4


def _build_pipeline_cache_key(current_params: dict) -> tuple:
    """A hashable snapshot of the full parameter set that produced a pipeline build."""
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(v) for v in value)
        return value
    return freeze(current_params)


def apply_patch(
    session: Session
    , patch: dict
    , from_step: int
    , platform_config: Optional[PlatformConfig]
    , csv_bytes: Optional[bytes]
    , uploaded_dfs: Optional[dict]
) -> None:
    """Apply an already-built patch dict to a session and re-run the pipeline from from_step.

    Pipelines this session already built are restored from its cache instead of re-running —
    the common case being a user toggling a setting (e.g. blend weights mid-draft) back to a
    value they had before. The current build is stashed under its parameter snapshot before
    the patch lands; a snapshot match after the patch restores agent + intermediates wholesale.

    platform_config feeds only the draft-state poll + name lookup, never the pipeline, so
    connecting a platform needs no pipeline rerun of its own.
    """
    if session.agent is not None:
        outgoing_key = _build_pipeline_cache_key(session.current_params)
        session.pipeline_cache[outgoing_key] = {
            'agent':           session.agent,
            'info':            session.info,
            'v0_clean':        session.v0_clean,
            'v1_clean':        session.v1_clean,
            'v2':              session.v2,
            'player_registry': session.player_registry,
        }
        session.pipeline_cache.move_to_end(outgoing_key)
        while len(session.pipeline_cache) > _PIPELINE_CACHE_ENTRIES:
            session.pipeline_cache.popitem(last=False)

    session.current_params.update(patch)
    # After the merge, not before: a patch can change the format, the weight, or both, and the
    # rule is about the pair that results.
    normalize_objective_settings(session.current_params)

    if platform_config is not None:
        session.platform_config = platform_config
        session.current_params['team_names'] = list(platform_config.teams_dict.keys())

    cached_pipeline = session.pipeline_cache.get(_build_pipeline_cache_key(session.current_params))
    if cached_pipeline is not None:
        session.agent           = cached_pipeline['agent']
        session.info            = cached_pipeline['info']
        session.v0_clean        = cached_pipeline['v0_clean']
        session.v1_clean        = cached_pipeline['v1_clean']
        session.v2              = cached_pipeline['v2']
        session.player_registry = cached_pipeline['player_registry']
        session.pipeline_cache.move_to_end(_build_pipeline_cache_key(session.current_params))
    else:
        build_agent(session, from_step=from_step, csv_bytes=csv_bytes, uploaded_dfs=uploaded_dfs)

    # Rebuild the lookup when either of its inputs changed: the player set (from_step <= 2)
    # or player_name_column (a platform_config was just set).
    if session.platform_config is not None and (from_step <= 2 or platform_config is not None):
        refresh_platform_player_id_lookup(session)
