# testing_files/benchmark_helpers.py
# Shared constants, client, and session-request builder used across all benchmark files.

import os
import yaml
from fastapi.testclient import TestClient

from backend.main import app
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

client = TestClient(app)

_PARAMS_PATH = 'parameters.yaml'
_SEASON      = '2024-25'
_SCORE_TOL   = 0.05   # allowed deviation from expected H-score (percentage points)

with open(_PARAMS_PATH) as _f:
    _NBA_PARAMS = yaml.safe_load(_f)['NBA']

_DEFAULT_CATEGORIES = _NBA_PARAMS['default-categories']
_NO_TO_CATEGORIES   = [c for c in _DEFAULT_CATEGORIES if c != 'Turnovers']

_ratio_names    = list(_NBA_PARAMS.get('ratio-statistics', {}).keys())
_count_names    = _NBA_PARAMS.get('counting-statistics', [])
_ALL_CATEGORIES = _ratio_names + [c for c in _count_names if c not in _ratio_names]


def _build_session_request(
    scoring_format: str = 'Head to Head: Most Categories'
    , categories: list = None
    , n_drafters: int = None
    , cash_per_team: int = None
) -> dict:
    """Construct a session request using all default parameters from parameters.yaml."""
    with open(_PARAMS_PATH) as f:
        all_params = yaml.safe_load(f)

    nba              = all_params['NBA']
    nba_options      = nba['options']
    n_picks          = nba_options['n_picks']['default']
    if n_drafters is None:
        n_drafters   = nba_options['n_drafters']['default']
    positions_config = nba_options['positions'][n_picks]
    slot_counts      = {**positions_config['base'], **positions_config['flex']}

    league: dict = {
        'sport':          'NBA',
        'n_drafters':     n_drafters,
        'n_picks':        n_picks,
        'scoring_format': scoring_format,
        'categories':     categories if categories is not None else nba['default-categories'],
    }
    if cash_per_team is not None:
        league['cash_per_team'] = cash_per_team

    return {
        'league': league,
        # The league TYPE is a top-level session field, not inferred from cash: without it the build
        # populates in draft mode and auction evaluates run against draft-shaped anchors (the app's
        # router enforces this consistency; service-level tests must declare it themselves).
        'is_auction': cash_per_team is not None,
        'slot_counts': slot_counts,
        'parameters': {
            'omega':           nba_options['omega']['default'],
            'gamma':           nba_options['gamma']['default'],
            'n_iterations':    nba_options['n_iterations']['default'],
            'beth':            nba_options['beth']['default'],
            'upsilon':         nba_options['upsilon']['default'],
            'psi':             nba_options['psi']['default'],
            'chi':             nba_options['chi']['default'],
            'aleph':           nba_options['aleph']['default'],
            # kappa follows the app default (parameters.yaml): goldens and benchmarks encode exactly what
            # ships. The one place kappa is deliberately pinned to 0 is the G-score season-sim harness
            # (simulate.py) -- against a non-punting G-drafter field the anti-crowded-punt penalty has no
            # crowd to defect from, so it would only distort that comparison.
            'kappa':           nba_options['kappa']['default'],
            # Follows the app default like kappa: goldens, benchmarks, and experiments all encode
            # exactly the opponent-punt softening the app ships with.
            'behavior_model_confidence': nba_options['behavior_model_confidence']['default'],
            'streaming_noise': nba_options['S']['default'],
        },
        'data_source': {
            'type':   'historical',
            'season': _SEASON,
        },
    }


def resolve_player_ids(session, player_names):
    """Resolve human-readable test-fixture names to the session's player ids.

    Accepts bare names, 'Name (POS)' labels, or prefixes (mirroring check_top_scores'
    startswith semantics). Raises on no match or ambiguity — a fixture that silently
    resolved to the wrong player would corrupt the test, so fail loudly instead.
    """
    registry = session.player_registry
    resolved = []
    for player_name in player_names:
        bare_name = player_name.split(' (')[0]
        matches = [identity.player_id for identity in registry.values()
                   if identity.name == bare_name]
        if not matches:
            matches = [identity.player_id for identity in registry.values()
                       if identity.name.startswith(bare_name)]
        if len(matches) != 1:
            raise ValueError(
                f'{player_name!r} resolved to {len(matches)} registry entries: {matches}')
        resolved.append(matches[0])
    return resolved


def resolve_player_assignments(session, player_assignments_by_name):
    """resolve_player_ids over a whole {team: [name, ...]} board."""
    return {
        team: resolve_player_ids(session, names)
        for team, names in player_assignments_by_name.items()
    }


def resolve_display_labels(session, player_ids):
    """Player ids -> the legacy 'Name (POS)' labels, for comparing service-level results
    against human-readable fixture snapshots."""
    from backend.player_identity import build_legacy_display_label

    return [build_legacy_display_label(session.player_registry[player_id])
            for player_id in player_ids]


def check_top_scores(label, expected_top_scores, candidates):
    """Assert each expected player's H-score is within _SCORE_TOL of its golden. With REGEN_GOLDENS set,
    print the actuals in paste-ready form and skip the assertions instead (golden regeneration)."""
    by_name  = {c.name: c for c in candidates}
    resolved = []
    for expected_name, expected_score in expected_top_scores:
        match = next((n for n in by_name if n.startswith(expected_name)), None)
        assert match is not None, f'{expected_name} not found in candidates'
        resolved.append((expected_name, expected_score, by_name[match].h_score, match))
    if os.environ.get('REGEN_GOLDENS'):
        rows = ',\n'.join(f"            ({repr(name) + ',':38} {round(actual, 1)})"
                          for name, _, actual, _ in resolved)
        print(f'\n# REGEN [{label}]\n{rows}')
        return
    for expected_name, expected_score, actual_score, match in resolved:
        assert abs(actual_score - expected_score) <= _SCORE_TOL, \
            f'{match} ({label}): expected H-score {expected_score}, got {actual_score:.1f}'
