"""How big a pool the future-pick model's picks behave like: its predicted tilts against what the same weights would
actually have taken from the recorded drafts.

The truncated-max model (backend/math/truncated_max_pick_model.py) prices every future pick as the best, by the
team's weights w, of a window of `pick_pool_size` (M) surviving players, relative to the best by generic value v: a
predicted tilt x(w; M), the same for every remaining pick. The draft-order drafts (draft_order.py, saved by
test_draft_order_bias to experiment_results/draft_order_picks.jsonl) record every pick's weights. For each pick before
a team's last, and each of that team's later picks, the realised tilt is

    x[the best available player by w . x]  -  x[the best available player by v . x]

over the players still undrafted at that later pick of the recorded draft, x the agent's X-scores (the units of the
model's covariance L). The effective M at a point in the draft is the M whose prediction is closest (least squares
over categories and picks) to the realised tilts there.

The realised tilt holds the decision's weights fixed for every later pick, as the model does; the field around it is
the recorded draft's, not the model's idealised pool. Two fields:
  - H-scoring: the draft-order drafts, twelve H-scoring seats.
  - G-score: draft_g_field_and_record_weights, twelve seats each taking the best G-score player its roster can place
    (the H vs G experiment's G drafters). At every pick the seat's own H-scoring session also ranks the board, at the H
    vs G conventions (opponent model off), and the weights behind its top candidate are recorded; it does not pick.
"""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

import numpy as np

from .draft_order import DRAFTER_SETTINGS, OBJECTIVES
from .league_sessions import build_league_session

# The candidate pool sizes. The app's default is 25 (backend/api/schemas.py).
POOL_SIZE_GRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200, 300, 400)
APP_POOL_SIZE = 25
# The H vs G conventions for the H-scoring sessions that weigh a G-score field's picks: modelling G drafters as
# punt-pursuers is a knowably wrong belief, so the opponent model is off.
G_FIELD_WEIGHER_SETTINGS = {**DRAFTER_SETTINGS, 'opponent_model_confidence': 0}
G_FIELD_ORDER = 'G-score field'
CANDIDATE_LIMIT = 40


def read_draft_picks(path: Path) -> list[dict]:
    """Every recorded draft (one per format, season and order)."""
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def measure_realised_tilts(draft: dict
                           , x_scores
                           , value_direction: np.ndarray) -> list[dict]:
    """For every pick with weights and every later pick of the same team: the decision's raw weights, its round, the
    later pick's round, and the realised tilt (C,) there."""
    from backend.player_identity import RP_PLAYER_ID
    player_ids = np.array([player for player in x_scores.index if player != RP_PLAYER_ID])
    scores = x_scores.loc[player_ids].to_numpy()
    position_by_player = {player: index for index, player in enumerate(player_ids)}
    picks = draft['picks']
    taken_before = np.zeros((len(picks) + 1, len(player_ids)), dtype=bool)
    for index, pick in enumerate(picks):
        taken_before[index + 1] = taken_before[index]
        taken_before[index + 1, position_by_player[pick['player_id']]] = True
    generic_value = scores @ value_direction
    records = []
    for index, decision in enumerate(picks):
        if decision['weights'] is None:
            continue
        weights = np.asarray(decision['weights']) / 100 * value_direction
        weighted_value = scores @ weights
        for later_index in range(index + 1, len(picks)):
            if picks[later_index]['team'] != decision['team']:
                continue
            available = ~taken_before[later_index]
            best_weighted = np.flatnonzero(available)[np.argmax(weighted_value[available])]
            best_generic = np.flatnonzero(available)[np.argmax(generic_value[available])]
            records.append({'decision_round': decision['round'], 'later_round': picks[later_index]['round'],
                            'weights': weights, 'tilt': scores[best_weighted] - scores[best_generic]})
    return records


def draft_g_field_and_record_weights(format_key: str
                                     , season: str) -> dict:
    """One snake draft of twelve G-score seats. Every pick records, as draft_order's drafts do, its number, round, team
    and player, and the weights the seat's own H-scoring session would pick with (None at a team's last pick)."""
    from backend.services.ranking import rank_candidates
    sessions = [build_league_session(OBJECTIVES[format_key], season, G_FIELD_WEIGHER_SETTINGS)[2] for _ in range(12)]
    from simulate import _has_position_data, _pick_gscore_player   # importable once a session build has set the path
    from self_play import list_snake_order
    session = sessions[0]
    settings = session.current_settings
    n_drafters, n_picks = settings['n_drafters'], settings['n_picks']
    if n_drafters != len(sessions):
        raise ValueError(f'The league has {n_drafters} drafters; this experiment builds {len(sessions)} sessions.')
    g_ranking = list(session.agent.info['G-scores'].sort_values('Total', ascending=False).index)
    teams = [f'Drafter {seat + 1}' for seat in range(n_drafters)]
    assignments = {team: [] for team in teams}
    drafted: set = set()
    has_positions = _has_position_data(session)
    positions_by_player = {player_id: identity.positions for player_id, identity in session.player_registry.items()}
    picks = []
    for pick_number, seat in enumerate(list_snake_order(n_drafters, n_picks)):
        team = teams[seat]
        result = rank_candidates(sessions[seat], assignments, team, [], None, 0, CANDIDATE_LIMIT)
        if not result.candidates:
            raise RuntimeError(f'No candidates for {team} at pick {pick_number + 1}.')
        is_last_pick = len(assignments[team]) == n_picks - 1
        weights = result.candidates[0].category_weights
        if weights is None and not is_last_pick:
            raise RuntimeError(f'No category weights for {team} at pick {pick_number + 1}, before its last pick.')
        chosen = _pick_gscore_player(g_ranking, drafted, assignments[team], session.agent.position_config,
                                     has_positions, positions_by_player)
        picks.append({'pick': pick_number + 1, 'round': pick_number // n_drafters + 1, 'team': team,
                      'player_id': int(chosen),
                      'weights': None if weights is None else [float(weight) for weight in weights]})
        assignments[team].append(chosen)
        drafted.add(chosen)
    return {'format': format_key, 'season': season, 'order': G_FIELD_ORDER,
            'categories': list(settings['categories']),
            'value_direction': [float(value) for value in np.asarray(session.agent.v).reshape(-1)], 'picks': picks}


def _draft_reporting_failure(*job) -> tuple:
    """draft_g_field_and_record_weights, returning ('drafted', draft) or ('failed', job, traceback)."""
    import traceback
    try:
        return 'drafted', draft_g_field_and_record_weights(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def draft_g_fields_in_parallel(format_key: str
                               , seasons: list[str]
                               , n_workers: int) -> tuple[list[dict], list[dict]]:
    """A G-field draft for every season of this format, one per job across n_workers processes (each job holds twelve
    sessions). Returns (drafts, failures)."""
    jobs = [(format_key, season) for season in seasons]
    with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool:
        outcomes = pool.starmap(_draft_reporting_failure, jobs, chunksize=1)
    drafts = [outcome[1] for outcome in outcomes if outcome[0] == 'drafted']
    failures = [{'job': outcome[1], 'traceback': outcome[2]} for outcome in outcomes if outcome[0] == 'failed']
    return drafts, failures


def predict_tilts(weights: np.ndarray
                  , covariance: np.ndarray
                  , value_direction: np.ndarray) -> np.ndarray:
    """The model's tilt for each row of weights (P, C) at every grid pool size: (G, P, C)."""
    from backend.math.truncated_max_pick_model import compute_expected_pick_tilts
    return np.stack([compute_expected_pick_tilts(weights, covariance, value_direction.reshape(-1, 1), pool_size)[:, :, 0]
                     for pool_size in POOL_SIZE_GRID])


def measure_season(format_key: str
                   , season: str
                   , drafts: list[dict]) -> list[dict]:
    """Every realised tilt of this format and season's recorded drafts, with the model's predictions beside it."""
    session = build_league_session(OBJECTIVES[format_key], season, DRAFTER_SETTINGS)[2]
    agent = session.agent
    value_direction = np.asarray(agent.v).reshape(-1)
    for draft in drafts:
        if not np.allclose(draft['value_direction'], value_direction):
            raise ValueError(f'{format_key} {season} {draft["order"]}: the recorded value direction is not the '
                             f'rebuilt session\'s; the drafts were made with a different build.')
        if list(draft['categories']) != list(agent.x_scores.columns):
            raise ValueError(f'{format_key} {season}: the recorded categories {draft["categories"]} are not the '
                             f'X-scores\' {list(agent.x_scores.columns)}.')
    records = [record for draft in drafts for record in measure_realised_tilts(draft, agent.x_scores, value_direction)]
    predictions = predict_tilts(np.stack([record['weights'] for record in records]), agent.L[0], value_direction)
    for index, record in enumerate(records):
        record.update(format=format_key, season=season, predictions=predictions[:, index])
    return records


def summarize_errors(records: list[dict]) -> dict:
    """The pooling-ready summary of these realised tilts: their count, their mean square (the error of predicting no
    tilt) and the mean squared error of the model's prediction at every grid pool size, all per category."""
    realised = np.stack([record['tilt'] for record in records])                       # (N, C)
    predicted = np.stack([record['predictions'] for record in records], axis=1)        # (G, N, C)
    return {'n': len(records), 'realised_mean_square': float((realised ** 2).mean()),
            'mean_squared_errors': [float(error) for error in ((predicted - realised) ** 2).mean(axis=(1, 2))]}


def fit_pool_size(records: list[dict]) -> dict:
    """The grid M closest (least squares over categories and picks) to these realised tilts, with the root mean square
    error per category at that M, at the app's M, and of predicting no tilt at all, and the mean realised and
    predicted (app M) tilt lengths."""
    realised = np.stack([record['tilt'] for record in records])                       # (N, C)
    predicted = np.stack([record['predictions'] for record in records], axis=1)        # (G, N, C)
    errors = np.sqrt(((predicted - realised) ** 2).mean(axis=(1, 2)))                  # (G,)
    best = int(np.argmin(errors))
    app = POOL_SIZE_GRID.index(APP_POOL_SIZE)
    return {'n': len(records), 'effective_pool_size': POOL_SIZE_GRID[best], 'best_error': float(errors[best]),
            'app_error': float(errors[app]), 'no_tilt_error': float(np.sqrt((realised ** 2).mean())),
            'realised_length': float(np.linalg.norm(realised, axis=1).mean()),
            'app_length': float(np.linalg.norm(predicted[app], axis=1).mean())}
