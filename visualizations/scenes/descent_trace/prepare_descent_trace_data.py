"""Record one candidate's gradient descent, exactly as the app runs it, for the descent-trace scene.

The app scores every seed in its menu before descending (one objective call per seed, each over the whole candidate
batch), keeps each candidate's best, and then descends from it for its iteration budget (one objective call per
iteration). This script runs the app's own evaluate on a board -- the same rank_candidates path the evaluate route takes,
at the app's default settings -- with a recorder around the agent's objective, and keeps, for one candidate:

  - every seed's H-score before any descent, and which seed the app chose: the gentle punts and the balanced build,
    in the menu's own order (HAgent._select_starting_weights);
  - after every iteration: the category weights (in the app's display units, 100 = neutral), the H-score, each
    category's win probability, the flex shares, the roster's slot assignment, and the expand view's G-score
    breakdown (the candidate, Current diff, Future diff, Total diff), built by the app's own _build_g_score_rows from
    that iteration's projected future picks. The final iteration's breakdown is checked against the one the evaluate
    returned.

Nothing in the algorithm is changed or re-implemented: the recorder only reads the arguments and results of calls the
evaluate makes anyway. An evaluate can descend more than once (the opponent model's replays); the descent kept is the
last, which is the one that produces the candidate's ranking.

The board is the one after --rounds snake rounds taken in default-ranking order, seat 1 to pick; --rounds 0 is the
first pick (see record_board).

    python visualizations/scenes/descent_trace/prepare_descent_trace_data.py --rounds 3 --player "Ausar Thompson"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))   # the repository
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))          # visualizations/

from backend.api.routers.sessions import _build_current_settings            # noqa: E402
from backend.services.ranking import _build_g_score_rows, make_slot_names, rank_candidates  # noqa: E402
from backend.services.session_management import build_session               # noqa: E402
from shared.prepare_season_data import (                                     # noqa: E402
    HEADSHOT_PIXELS, SEASON, SPORT, build_default_session_request, write_circular_headshot,
)

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent.parent
_PREPARED_DATA_DIR = _VISUALIZATIONS_DIR / 'prepared_data'
_HEADSHOT_DIR = _VISUALIZATIONS_DIR / 'prepared_assets' / 'headshots'
_HEADSHOT_SOURCE_DIR = _VISUALIZATIONS_DIR.parent / '.cache' / 'headshots'


def build_board(session, n_rounds: int) -> tuple[dict, str]:
    """The board after n_rounds snake rounds in default-ranking order, and the seat now picking (the first)."""
    teams = session.current_settings['team_names']
    ranked = [int(player) for player in session.agent.default_h_scores.index]
    board = {team: [] for team in teams}
    for pick, team in enumerate(((teams + teams[::-1]) * n_rounds)[:len(teams) * n_rounds]):
        board[team].append(ranked[pick])
    return board, teams[0]


def describe_menu(categories: list[str], menu_size: int) -> list[dict]:
    """What each seed of the app's menu is, in its order: a gentle punt per category, then the balanced build
    (HAgent._select_starting_weights)."""
    n_categories = len(categories)
    if menu_size != n_categories + 1:
        raise ValueError(f'A menu of {menu_size} seeds is not the {n_categories} gentle punts and the balanced build.')
    return ([{'kind': 'gentle punt', 'category': category} for category in categories]
            + [{'kind': 'balanced', 'category': None}])


def record_evaluate(session, board: dict, seat: str) -> tuple[list[dict], object]:
    """Run the app's evaluate with a recorder around the agent's descent and objective. Returns one entry per descent
    (its candidate ids, the board's differentials against the field and the opponents' expected future tilts it was
    given, its seed-scoring calls and its iteration calls, each the call's arguments and result), and the evaluate's
    response."""
    agent = session.agent
    descents: list[dict] = []
    original_objective, original_descent = agent.get_objective_and_gradient, agent.perform_iterations

    def recording_descent(*args, **kwargs):
        descents.append({'candidate_ids': [int(player) for player in args[7]], 'diff_means': np.array(args[4]),
                         'opponent_future_tilts': kwargs['opponent_future_tilts'], 'seed_calls': [],
                         'iteration_calls': []})
        return original_descent(*args, **kwargs)

    def recording_objective(*args, **kwargs):
        # The flex shares are copied as they stand at this call: the descent updates them IN PLACE between iterations,
        # so a kept reference would read whatever they became later.
        shares = None if args[1] is None else {flex_code: None if frame is None else frame.copy()
                                               for flex_code, frame in args[1].items()}
        result = original_objective(*args, **kwargs)
        call = {'weights': np.array(args[0], dtype=float), 'shares': shares, 'iteration': args[9], 'result': result}
        descent = descents[-1]
        if kwargs.get('correction_mode') == 'skip':
            descent['seed_calls'].append(call)
        else:
            descent['iteration_calls'].append(call)
        return result

    agent.get_objective_and_gradient, agent.perform_iterations = recording_objective, recording_descent
    try:
        response = rank_candidates(session=session, player_assignments=board, my_team_id=seat, exclusion_list=[],
                                   remaining_cash=None)
    finally:
        del agent.get_objective_and_gradient, agent.perform_iterations
    return descents, response


def read_row(values, row: int) -> list[float]:
    """One candidate's row of a per-candidate array or frame, as floats."""
    array = values.to_numpy() if hasattr(values, 'to_numpy') else np.asarray(values)
    return [float(value) for value in np.asarray(array[row], dtype=float).reshape(-1)]


def read_shares(shares, row: int) -> dict:
    """The candidate's flex shares, {flex slot: {base position: share}}, from the descent's per-candidate frames."""
    if shares is None:
        return {}
    readable = {}
    for flex_code, frame in shares.items():
        if frame is None:
            continue
        columns = list(frame.columns) if hasattr(frame, 'columns') else None
        values = read_row(frame, row)
        if columns is None:
            raise TypeError(f'Flex shares for {flex_code} carry no base-position labels.')
        readable[flex_code] = dict(zip(columns, values))
    return readable


def build_diff_rows(session, descent: dict, result: dict, row: int) -> list[dict]:
    """The expand view's G-score breakdown for candidate `row` as it stands at one iteration (`result`, that
    iteration's objective result), built exactly as rank_candidates builds it from the descent's final result: the
    team's differential is the board's (averaged over the field) plus the projected future picks', all in G-score
    units, and the Future row nets out the opponents' expected future tilts."""
    agent = session.agent
    n_candidates = len(descent['candidate_ids'])
    categories = list(agent.x_scores.columns)
    original_v = np.array(agent.original_v)
    candidate = descent['candidate_ids'][row]
    board_diff = np.broadcast_to(descent['diff_means'].mean(axis=2), (n_candidates, len(categories)))[row]
    future_diffs = result['Future-Diffs']
    future_diff = None if future_diffs is None else np.asarray(future_diffs).mean(axis=2)[row]
    team_diff = board_diff if future_diff is None else future_diff + board_diff
    tilts = descent['opponent_future_tilts']
    opponent_tilt = (None if tilts is None or future_diff is None
                     else np.broadcast_to(tilts, (n_candidates, len(categories)))[row])
    own = agent.x_scores.reindex([candidate])[categories].fillna(0.0).to_numpy().astype(float) * original_v
    rows = _build_g_score_rows(
        [session.player_registry[candidate].last_name]
        , own
        , team_diff.reshape(1, -1)
        , None if future_diff is None else future_diff.reshape(1, -1)
        , original_v
        , None if opponent_tilt is None else opponent_tilt.reshape(1, -1)
    )[0]
    # The expand view lists the candidate's own row first (expand_view.ts).
    ordered = [entry for entry in rows if entry.is_candidate] + [entry for entry in rows if not entry.is_candidate]
    return [{'label': entry.label, 'values': entry.values, 'total': entry.total, 'is_candidate': entry.is_candidate}
            for entry in ordered]


def check_diff_rows(recorded: list[dict], response, candidate: int) -> None:
    """The final iteration's breakdown must be the one the evaluate returned for the candidate."""
    returned = next((entry for entry in response.candidates if entry.player_id == candidate), None)
    if returned is None:
        raise RuntimeError(f'The evaluate returned no row for player {candidate}.')
    expected = {entry.label: (entry.values, entry.total) for entry in returned.g_score_rows}
    actual = {entry['label']: (entry['values'], entry['total']) for entry in recorded}
    if expected != actual:
        raise RuntimeError(f"The recorded final G-score breakdown differs from the evaluate's:\n"
                           f'  recorded {actual}\n  returned {expected}')


def build_recording(session, board: dict, seat: str, descent: dict, response, row: int) -> dict:
    """The scene's recording of one candidate's descent (row `row` of `descent`)."""
    agent = session.agent
    registry = session.player_registry
    ids = descent['candidate_ids']
    categories = list(agent.x_scores.columns)
    neutral = agent.v.reshape(-1)
    neutral = neutral / neutral.sum()

    def to_display(weights) -> list[float]:
        shares = np.asarray(weights, dtype=float) / np.sum(weights)
        return [round(float(value), 2) for value in shares / neutral * 100]

    # The head-to-head formats scan a seed menu; Rotisserie starts every candidate from its one lowvar seed and scans
    # nothing (HAgent.seed_mode), so its recording has no seeds and names its start instead.
    if descent['seed_calls']:
        menu = describe_menu(categories, len(descent['seed_calls']))
        seeds = [{**seed, 'weights': to_display(call['weights'][row]),
                  'h_score': round(100 * float(np.asarray(call['result']['Score'])[row]), 4)}
                 for seed, call in zip(menu, descent['seed_calls'])]
        chosen = int(np.argmax([seed['h_score'] for seed in seeds]))
    elif agent.seed_mode == 'lowvar':
        seeds, chosen = [], None
    else:
        raise RuntimeError(f'The descent scanned no seeds, which only the lowvar seed mode does; this is '
                           f'{agent.seed_mode}.')

    slot_names = make_slot_names(agent.position_config)
    candidate = ids[row]
    roster_players = board[seat] + [candidate]
    # CDF-Estimates has one column per opponent per snapshot of the opponent model: the field is built once per
    # remembered snapshot, each block listing the opponents in seat order (HAgent's field_blocks).
    opponents = [team for team in session.current_settings['team_names'] if team != seat]
    iterations = []
    for call in descent['iteration_calls']:
        result = call['result']
        cdf_row = np.asarray(read_row(result['CDF-Estimates'], row))
        n_snapshots, remainder = divmod(cdf_row.size, len(categories) * len(opponents))
        if remainder:
            raise ValueError(f'{cdf_row.size} win estimates do not split into {len(categories)} categories x '
                             f'{len(opponents)} opponents x whole snapshots.')
        cdf_estimates = cdf_row.reshape(len(categories), n_snapshots, len(opponents))
        # Which player fills each roster slot (None: a slot left for the picks to come). Rosters lists, per player in
        # the order [team so far..., candidate, future picks...], the slot it is assigned.
        roster_row = read_row(result['Rosters'], row)
        if roster_row[0] < 0:
            raise ValueError('The descent carried no roster assignment (no position data).')
        occupants = {slot_names[int(roster_row[column])]: player for column, player in enumerate(roster_players)}
        iterations.append({
            'iteration': int(call['iteration']) + 1,
            'weights': to_display(call['weights'][row]),
            'h_score': round(100 * float(np.asarray(result['Score'])[row]), 4),
            # Each category's win probability against each opponent, averaged over the opponent model's snapshots:
            # rows are categories, columns opponents. The H-score is the mean of the whole grid.
            'category_opponent_win_rates': [[round(100 * float(value), 1) for value in category_row]
                                            for category_row in cdf_estimates.mean(axis=1)],
            'flex_shares': read_shares(call['shares'], row),
            'slots': [{'slot': slot, 'player_id': occupants.get(slot)} for slot in slot_names],
            'diff_rows': build_diff_rows(session, descent, result, row),
        })
    check_diff_rows(iterations[-1]['diff_rows'], response, candidate)

    return {
        'season': SEASON, 'seat': seat, 'player': registry[candidate].name, 'player_id': candidate,
        'roster': [registry[player].name for player in board[seat]],
        'players': {str(player): {'name': registry[player].name, 'last_name': registry[player].last_name}
                    for player in roster_players},
        'scoring_format': session.current_settings['scoring_format'],
        'categories': categories, 'opponents': opponents, 'seeds': seeds, 'chosen_seed': chosen,
        # Where the descent starts: the chosen seed, or (no seed scan) the format's own seed, scored by the descent's
        # first objective call, which is made at the starting weights.
        'start': ({'label': describe_seed(seeds[chosen]), 'h_score': seeds[chosen]['h_score']} if seeds
                  else {'label': 'low-variance seed', 'h_score': iterations[0]['h_score']}),
        'iterations': iterations,
    }


def write_missing_headshots(player_ids: list[int]) -> None:
    """The circular headshot of every player the scene draws, built from the app's headshot cache when missing."""
    missing = []
    for player in player_ids:
        destination = _HEADSHOT_DIR / f'{player}.png'
        if destination.exists():
            continue
        source = _HEADSHOT_SOURCE_DIR / f'{player}.png'
        if not source.exists():
            missing.append(player)
            continue
        write_circular_headshot(source, destination, HEADSHOT_PIXELS)
    if missing:
        raise SystemExit(f'No cached headshot for player ids {missing}. Start the app and load {SEASON} once to '
                         f'populate .cache/headshots, then re-run.')


def describe_seed(seed: dict) -> str:
    """A seed's name: 'Threes at 50%' or 'balanced'."""
    return f"{seed['category']} at 50%" if seed['kind'] == 'gentle punt' else seed['kind']


def build_app_session(scoring_format: str):
    """A session on SEASON in this scoring format at the app's own default settings."""
    request = build_default_session_request(SEASON, SPORT)
    request = request.model_copy(update={'league': request.league.model_copy(update={'scoring_format': scoring_format})})
    return build_session(
        current_settings = _build_current_settings(request),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )


def record_board(session, n_rounds: int) -> tuple[dict, str, dict, object]:
    """(board, seat, the evaluate's final descent, its response) for the board after n_rounds rounds.

    On the first pick (n_rounds 0) the evaluate does not descend: it returns the board the session build already
    solved (HAgent._default_result, the get_h_scores short-circuit). That board is the same cold solve -- seed scan,
    then descent, against the same field -- so the stored result is hidden for this one evaluate to make it solve
    again where the recorder can see it, and is restored straight after."""
    board, seat = build_board(session, n_rounds)
    agent = session.agent
    default_result = agent._default_result
    if n_rounds == 0:
        agent._default_result = None
    try:
        descents, response = record_evaluate(session, board, seat)
    finally:
        agent._default_result = default_result
    if not descents:
        raise RuntimeError('The evaluate made no descent.')
    descent = descents[-1]
    if not descent['seed_calls'] and agent.seed_mode != 'lowvar':
        raise RuntimeError('The final descent scored no seed menu.')
    return board, seat, descent, response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--rounds', type=int, default=2,
                        help='snake rounds already drafted; 0 is the first pick')
    parser.add_argument('--player', default=None,
                        help="part of the candidate's name; the top-ranked candidate when left out")
    parser.add_argument('--format', default='Head to Head', choices=('Head to Head', 'Rotisserie'),
                        help='the scoring format; Rotisserie scans no seeds, so its scene skips the seed act')
    arguments = parser.parse_args()
    if arguments.rounds < 0:
        raise SystemExit('--rounds cannot be negative.')

    print(f'Building a {SEASON} {arguments.format} session at the app defaults...')
    session = build_app_session(arguments.format)
    print(f'Evaluating the board after {arguments.rounds} rounds with the recorder attached...')
    board, seat, descent, response = record_board(session, arguments.rounds)
    ids = descent['candidate_ids']
    names = {player: session.player_registry[player].name for player in ids}
    if arguments.player is None:
        row = int(np.argmax(np.asarray(descent['iteration_calls'][-1]['result']['Score'], dtype=float)))
    else:
        matches = [index for index, player in enumerate(ids) if arguments.player.lower() in names[player].lower()]
        if len(matches) != 1:
            raise SystemExit(f'"{arguments.player}" matches {len(matches)} candidates: '
                             f'{[names[ids[index]] for index in matches][:10]}')
        row = matches[0]

    out = build_recording(session, board, seat, descent, response, row)
    out['rounds'] = arguments.rounds
    write_missing_headshots([int(player) for player in out['players']])
    slug = re.sub(r'[^a-z0-9]+', '_', out['player'].lower()).strip('_')
    format_part = '_rotisserie' if arguments.format == 'Rotisserie' else ''
    path = (_PREPARED_DATA_DIR
            / f'descent_trace_{SEASON.replace("-", "_")}{format_part}_round{arguments.rounds}_{slug}.json')
    path.write_text(json.dumps(out, indent=1), encoding='utf-8')
    start, iterations = out['start'], out['iterations']
    print(f"\n{out['player']}: {len(out['seeds'])} seeds, started from {start['label']}, {len(iterations)} "
          f"iterations; H-score {start['h_score']:.2f} -> {iterations[-1]['h_score']:.2f}")
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
