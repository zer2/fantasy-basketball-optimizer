# testing_files/season_simulation/simulate.py
# Part 1 of the season-simulation harness: run the drafts and SAVE all data. No HTML here.
#
# For each (season, scoring format) we build one backend session (the expensive pipeline runs once)
# and then run one snake draft per seat. In each draft exactly one seat is the "H-score drafter"
# (picks the top candidate from run_evaluate); the other eleven are "G-score drafters" (pick the
# highest-Total-G-score available player that keeps their roster position-eligible). We record the
# H-score drafter's final team H-score + per-category rates, plus, for each of its picks, the ranked
# candidate table (top N) and the picked player's expand-view detail — i.e. the tables the website
# would have shown at that pick. Results are written to one JSON file per (season, format).
#
# Run (small scope first):
#   python -m testing_files.season_simulation.simulate --seasons 2024-25 --formats EC --seats 0 1
#   python -m testing_files.season_simulation.simulate                       # all seasons, EC/MC/Roto

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault('SESSION_SECRET_KEY', 'season-sim-only')
logging.disable(logging.CRITICAL)   # silence Snowflake / app logs for a clean console

import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))   # repo root -> `backend`
sys.path.insert(0, str(_HERE.parent))           # testing_files -> `benchmark_helpers`

from benchmark_helpers import client, _build_session_request
from backend.session import get_session
from backend.evaluate import run_evaluate
from backend.data_retrieval import get_available_seasons
from backend.math.position_optimization import check_single_player_eligibility


# scoring-format key -> the exact string the backend expects.
SCORING_FORMATS = {
    'EC':   'Head to Head: Each Category',
    'MC':   'Head to Head: Most Categories',
    'Roto': 'Rotisserie',
}

_OUTPUT_DIR = _HERE / 'output' / 'data'
_TOP_N_DEFAULT = 40   # how many ranked candidates to keep per pick for the report's candidate table


def _create_session(season: str, scoring_format: str):
    """Build one backend session for a (season, format) and return (session, session_id)."""
    request = _build_session_request(scoring_format=scoring_format)
    request['data_source']['season'] = season
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'Session creation failed ({season}, {scoring_format}): {response.text}'
    session_id = response.json()['session_id']
    return get_session(session_id), session_id


def _has_position_data(session) -> bool:
    """Historical seasons before 2000-01 carry no positions; the optimiser then skips position work
    and every player fits every roster. Detect it so the G-score drafter can skip eligibility checks
    (and so the report can flag the row)."""
    positions = session.info.get('Positions')
    if positions is None or len(positions) == 0:
        return False
    # Positions map name -> list of position codes (or a '?'/'NP' sentinel when unknown).
    sample = next(iter(positions.values())) if isinstance(positions, dict) else positions.iloc[0]
    text = ','.join(sample) if isinstance(sample, list) else str(sample)
    return 'NP' not in text and '?' not in text


def _gscore_ranking(session) -> list[str]:
    """Player names ordered by descending total G-score (the G-score drafter's preference order)."""
    g_scores = session.info['G-scores']
    return list(g_scores.sort_values('Total', ascending=False).index)


def _pick_gscore_player(
    gscore_order: list[str]
    , drafted: set[str]
    , team_so_far: list[str]
    , position_config
    , has_positions: bool
) -> str:
    """Highest-Total-G-score available player that keeps the roster position-eligible. Walking the
    ranking top-down and taking the first eligible name is cheap: early picks the top name almost
    always fits, so it is usually a single eligibility check."""
    for name in gscore_order:
        if name in drafted:
            continue
        if not has_positions or check_single_player_eligibility(name, team_so_far, position_config):
            return name
    raise RuntimeError('No eligible G-score player remains — pool exhausted unexpectedly.')


def _light_candidate(candidate) -> dict:
    """Compact ranked-table row (no expand views) — what the candidate table shows at a glance."""
    return {
        'name':             candidate.name,
        'position':         candidate.position,
        'h_score':          candidate.h_score,
        'h_rank':           candidate.h_rank,
        'win_rates':        candidate.win_rates,
        'category_weights': candidate.category_weights,
    }


def _picked_detail(candidate) -> dict:
    """The picked player's expand-view tables (G-score rows / flex / roster) — the per-pick detail.
    Uses Pydantic's json dump so the nested dataclasses serialise exactly as the API sends them."""
    dump = candidate.model_dump(mode='json')
    return {
        'g_score_rows':     dump.get('g_score_rows'),
        'flex_allocations': dump.get('flex_allocations'),
        'roster':           dump.get('roster'),
    }


def _score_team(session, assignments: dict[str, list[str]], team_name: str, n_iterations: int) -> tuple[float, list[float]]:
    """Final H-score + per-category rates for a completed team, via the same extraction the trade
    analyzer uses (backend/math/trading.py). A full roster yields a single-row result."""
    result = session.H.get_h_scores(assignments, team_name, n_iterations)
    scores = result['Scores']
    rates  = result['Rates']
    index  = scores.idxmax()
    return float(scores[index]), [float(x) for x in rates.loc[index].tolist()]


def _simulate_one_seat(
    session
    , hscore_seat: int
    , n_drafters: int
    , n_picks: int
    , gscore_order: list[str]
    , has_positions: bool
    , top_n: int
) -> dict:
    """Run one snake draft where `hscore_seat` drafts by H-score and the rest by G-score. Returns the
    H-score drafter's final score, its roster, and its per-pick candidate tables."""
    position_config = session.H._pos_cfg
    n_iterations    = session.current_params['n_iterations']
    team_names      = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments     = {name: [] for name in team_names}
    drafted: set[str] = set()
    hscore_name     = team_names[hscore_seat]

    picks: list[dict] = []

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            drafter_index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)   # serpentine
            drafter_name  = team_names[drafter_index]

            if drafter_index == hscore_seat:
                result = run_evaluate(session, assignments, drafter_name, [], None, 0, None)
                assert result.candidates, f'No candidates for {drafter_name}, round {pick_row + 1}'
                top_candidate = result.candidates[0]
                picks.append({
                    'round':         pick_row + 1,
                    'picked':        top_candidate.name,
                    'candidates':    [_light_candidate(c) for c in result.candidates[:top_n]],
                    'picked_detail': _picked_detail(top_candidate),
                })
                chosen = top_candidate.name
            else:
                chosen = _pick_gscore_player(
                    gscore_order, drafted, assignments[drafter_name], position_config, has_positions,
                )

            assignments[drafter_name].append(chosen)
            drafted.add(chosen)

    team_h_score, team_rates = _score_team(session, assignments, hscore_name, n_iterations)
    return {
        'hscore_seat':  hscore_seat,
        'roster':       assignments[hscore_name],
        'team_h_score': round(team_h_score * 100, 2),
        'team_rates':   [round(x * 100, 2) for x in team_rates],
        'picks':        picks,
    }


def simulate_season_format(season: str, format_key: str, seats: list[int], top_n: int) -> dict:
    """Simulate all requested seats for one (season, format) and return the full result record."""
    scoring_format = SCORING_FORMATS[format_key]
    session, _ = _create_session(season, scoring_format)
    n_drafters   = session.current_params['n_drafters']
    n_picks      = session.current_params['n_picks']
    categories   = session.current_params['categories']
    has_positions = _has_position_data(session)
    gscore_order = _gscore_ranking(session)

    seat_results = []
    for seat in seats:
        start = time.perf_counter()
        seat_results.append(
            _simulate_one_seat(session, seat, n_drafters, n_picks, gscore_order, has_positions, top_n)
        )
        elapsed = time.perf_counter() - start
        print(f'    seat {seat + 1:2d}/{n_drafters}: '
              f'H={seat_results[-1]["team_h_score"]:5.1f}%  ({elapsed:4.1f}s)', flush=True)

    return {
        'season':             season,
        'format_key':         format_key,
        'scoring_format':     scoring_format,
        'categories':         list(categories),
        'n_drafters':         n_drafters,
        'n_picks':            n_picks,
        'has_position_data':  has_positions,
        'seats':              seat_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Simulate historical seasons and save H-score results.')
    parser.add_argument('--seasons', nargs='*', default=None, help='Seasons (default: all available).')
    parser.add_argument('--formats', nargs='*', default=list(SCORING_FORMATS), help='EC MC Roto.')
    parser.add_argument('--seats', nargs='*', type=int, default=None, help='Seat indices 0..11 (default: all).')
    parser.add_argument('--top-n', type=int, default=_TOP_N_DEFAULT, help='Ranked candidates kept per pick.')
    parser.add_argument('--out', default=str(_OUTPUT_DIR), help='Output directory for JSON files.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip any (season, format) whose JSON already exists (for restarts).')
    args = parser.parse_args()

    seasons = args.seasons if args.seasons else get_available_seasons()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        for format_key in args.formats:
            path = out_dir / f'{season}__{format_key}.json'
            if args.resume and path.exists():
                print(f'[{season} / {format_key}] skip (exists)', flush=True)
                continue
            # Seats default to all drafters; resolve per session since n_drafters could vary.
            print(f'[{season} / {format_key}] simulating...', flush=True)
            record = simulate_season_format(
                season, format_key,
                seats = args.seats if args.seats is not None else list(range(12)),
                top_n = args.top_n,
            )
            # Write atomically (tmp then replace) so a concurrent render.py never reads a half file.
            tmp = path.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(record), encoding='utf-8')
            tmp.replace(path)
            print(f'  wrote {path}  ({path.stat().st_size // 1024} KB)', flush=True)


if __name__ == '__main__':
    main()
