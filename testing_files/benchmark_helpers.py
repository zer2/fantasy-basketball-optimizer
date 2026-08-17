# testing_files/benchmark_helpers.py
# Shared constants, client, and session-request builder used across all benchmark files.

import datetime
import json
import os
import subprocess
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from backend.main import app
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

client = TestClient(app)

# Append-only local benchmark history (gitignored): every [benchmark] measurement lands here
# with its commit, so "did this get faster or slower?" is answerable from data instead of
# memory. One JSON object per line: {timestamp, commit, label, seconds}.
_BENCHMARK_HISTORY_PATH = Path(__file__).parent / 'benchmark_history.jsonl'
_current_commit_cache: str | None = None


def _current_commit() -> str:
    global _current_commit_cache
    if _current_commit_cache is None:
        _current_commit_cache = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip() or 'unknown'
    return _current_commit_cache


def record_benchmark(label: str, seconds: float) -> None:
    """Print a [benchmark] line AND append it to the local benchmark history."""
    print(f'\n[benchmark] {label}: {seconds:.2f}s')
    entry = {
        'timestamp': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'commit':    _current_commit(),
        'label':     label,
        'seconds':   round(seconds, 3),
    }
    with open(_BENCHMARK_HISTORY_PATH, 'a', encoding='utf-8') as history_file:
        history_file.write(json.dumps(entry) + '\n')

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


# Head to Head is one format with an objective dial (0 = every category its own contest,
# 1 = only the majority matters), so the two names it used to have are now presets on that dial.
# Tests name the objective they mean and this maps it onto the wire fields; adding a mixed
# objective is a line here rather than a new format everywhere.
OBJECTIVE_PRESETS: dict[str, tuple[str, float | None]] = {
    'Each Category':   ('Head to Head', 0.0),
    'Half and Half':   ('Head to Head', 0.5),
    'Most Categories': ('Head to Head', 1.0),
    'Rotisserie':      ('Rotisserie',   None),
}


def resolve_objective(objective: str) -> tuple[str, float | None]:
    """The (scoring_format, most_categories_weight) pair a preset name means. Unknown names raise:
    a typo would otherwise quietly score a test against a different objective than it claims."""
    if objective not in OBJECTIVE_PRESETS:
        raise ValueError(f'Unknown objective {objective!r}. '
                         f'Known: {sorted(OBJECTIVE_PRESETS)}')
    return OBJECTIVE_PRESETS[objective]


def _build_session_request(
    objective: str = 'Most Categories'
    , categories: list = None
    , n_drafters: int = None
    , cash_per_team: int = None
    , tiebreaker_category: str = None
) -> dict:
    """Construct a session request using all default parameters from parameters.yaml.

    `objective` names one of OBJECTIVE_PRESETS. The default keeps every caller that never named
    one on Most Categories, the objective they were built against. A tiebreaker needs an even
    number of categories, so callers passing one pass `categories` too.
    """
    scoring_format, most_categories_weight = resolve_objective(objective)
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
        'most_categories_weight': most_categories_weight,
        'tiebreaker_category': tiebreaker_category,
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


def name_candidates(session, candidates):
    """{registry display name: candidate} over an evaluate result — the human-readable view
    of the id-keyed candidate payload for fixture matching."""
    registry = session.player_registry
    return {registry[c.player_id].name: c for c in candidates}


def check_top_scores(session, label, expected_top_scores, candidates):
    """Assert each expected player's H-score is within _SCORE_TOL of its golden. With REGEN_GOLDENS set,
    print the actuals in paste-ready form and skip the assertions instead (golden regeneration).
    Expected names are prefixes of registry display names."""
    by_name  = name_candidates(session, candidates)
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
