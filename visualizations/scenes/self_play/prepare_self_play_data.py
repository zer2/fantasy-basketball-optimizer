"""Record a real self-play run: what the field looks like as it settles.

Self-play descends every top player at once, each from every seed of its menu, against a field rebuilt every
iteration from each player's current start (HAgent.populate_default_h_scores). This records that descent as the
scene draws it, every iteration: each tracked player's current start's expected category win rates against the field
-- what the build buys, a category at fifty percent being a coin flip -- and the two events the scene marks:

  - switched: the player changed start this iteration, as one of the (at most) _SELF_PLAY_SWITCHES_PER_ITERATION whose
    best start led their current one by the most;
  - in the field: the player is one of the field's seats or its spare -- the top players the field is made of. The
    membership only changes when the field re-ranks its seats (_SELF_PLAY_SEAT_RERANK_INTERVAL); it is read off the
    field itself, so a member outside the tracked rows simply has no row to mark.

The opening is Level 0: everybody solved against a neutral field before self-play starts, with the field seated from
Level 0's ranking. The first iteration's selection is every player's first pick of a start, not a
switch, so it marks nobody.

Nothing here models anything: the recorder only reads the objective calls and start selections the shipped build
makes anyway.

    python visualizations/scenes/self_play/prepare_self_play_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))   # the repository
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))          # visualizations/

from backend.api.routers.sessions import _build_current_settings            # noqa: E402
from backend.math.algorithm_agents import HAgent, _SelfPlayField            # noqa: E402
from backend.services.session_management import build_session               # noqa: E402
from shared.prepare_season_data import build_default_session_request, SEASON, SPORT    # noqa: E402

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent.parent

# How many of the field's players the scene draws. The descent covers the whole top-player universe; a few more rows
# than fit on screen would be recorded and never shown.
TRACKED_PLAYERS = 14


def record_self_play(session) -> tuple[dict, list[dict]]:
    """Re-run the session's self-play with recorders on the agent's objective and the field's start selection and
    seat ranking. Returns Level 0's result and, per iteration, {iteration, players, win rates (players x categories),
    switched, in the field}."""
    agent = session.agent
    steps: list[dict] = []
    level_zero: dict = {}
    original_objective = HAgent.get_objective_and_gradient
    original_update = _SelfPlayField.update_best_starts
    original_rank = _SelfPlayField.rank_seats
    original_pass = HAgent._run_bootstrap_pass
    latest: dict = {}

    def recording_objective(self, *arguments, **keyword_arguments):
        result = original_objective(self, *arguments, **keyword_arguments)
        if self._self_play_field is not None:
            latest.update(iteration=arguments[9], result=result)
        return result

    # A step is recorded where the field adopts each player's start for the iteration just scored (after every
    # iteration but the last; for the last, the serve's selection), so a row shows the start the field actually uses.
    def recording_update(self, scores):
        previous_starts = None if self.incumbent_starts is None else self.incumbent_starts.copy()
        best_rows = original_update(self, scores)
        result = latest['result']
        n_players = len(self.players)
        if previous_starts is None:
            switched = np.zeros(n_players, dtype=bool)
        else:
            switched = self.incumbent_starts != previous_starts
        steps.append({'iteration': latest['iteration'] + 1, 'players': list(self.players),
                      'win_rates': np.asarray(result['CDF-Estimates']).mean(axis=2)[best_rows],
                      'switched': switched, 'in_field': list_field_members(self)})
        return best_rows

    # The re-rank follows the update in the same iteration, so its new seats belong to the step just recorded.
    def recording_rank(self, scores):
        original_rank(self, scores)
        steps[-1]['in_field'] = list_field_members(self)

    def recording_pass(self, empty, n_iterations, cash_remaining_per_team, candidate_subset=None, **keyword_arguments):
        result = original_pass(self, empty, n_iterations, cash_remaining_per_team, candidate_subset,
                               **keyword_arguments)
        if not level_zero:
            level_zero.update(rates=result['Rates'], scores=result['Scores'])
        return result

    HAgent.get_objective_and_gradient = recording_objective
    _SelfPlayField.update_best_starts = recording_update
    _SelfPlayField.rank_seats = recording_rank
    HAgent._run_bootstrap_pass = recording_pass
    try:
        # The build already ran this once; running it again with the recorders on is the only way to see inside it,
        # and it resets its own state at the top, so the second run is not a continuation of the first.
        agent.populate_default_h_scores(session.current_settings['n_iterations'])
    finally:
        HAgent.get_objective_and_gradient = original_objective
        _SelfPlayField.update_best_starts = original_update
        _SelfPlayField.rank_seats = original_rank
        HAgent._run_bootstrap_pass = original_pass
    if not steps:
        raise RuntimeError('Self-play recorded no steps. Opponent modelling is probably off for this session (a '
                           'single neutral pass), so there is no self-play to animate.')
    return level_zero, steps


def list_field_members(field) -> np.ndarray:
    """(players,) True for the field's seats and its spare."""
    members = np.zeros(len(field.players), dtype=bool)
    members[list(field.seated) + [field.spare]] = True
    return members


def build_steps(level_zero: dict, recorded: list[dict], ranking) -> tuple[list[dict], list, list]:
    """Per-iteration win rates and events for the players the scene follows: the top TRACKED_PLAYERS of the served
    board that self-play descends, in that order and fixed once, so a row means the same player in the same place all
    the way through."""
    universe = set(recorded[0]['players'])
    tracked = [player for player in ranking if player in universe][:TRACKED_PLAYERS]
    if len(tracked) < TRACKED_PLAYERS:
        raise RuntimeError(f'Only {len(tracked)} of the ranked players are in the self-play universe, but the scene '
                           f'draws {TRACKED_PLAYERS} rows.')
    # Level 0 solved everybody against a neutral field, so the board is already full before self-play starts. That
    # is the state the scene opens on.
    opening_rates = [level_zero['rates'].loc[player].to_numpy(dtype=float).tolist() for player in tracked]
    steps = []
    for step in recorded:
        position = {player: index for index, player in enumerate(step['players'])}
        rows = [position[player] for player in tracked]
        entry = {
            'iteration': step['iteration'],
            'rates':     np.round(step['win_rates'][rows], 4).tolist(),
            'switched':  [bool(step['switched'][row]) for row in rows],
            'in_field':  [bool(step['in_field'][row]) for row in rows],
        }
        steps.append(entry)
    return steps, tracked, opening_rates


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults...')
    session = build_session(
        current_settings = _build_current_settings(build_default_session_request(SEASON, SPORT)),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )
    print('Re-running self-play with the recorder attached...')
    level_zero, recorded = record_self_play(session)
    ranking = list(session.agent.default_h_scores.index)
    steps, tracked, opening_rates = build_steps(level_zero, recorded, ranking)
    categories = list(session.agent.x_scores.columns)

    print(f'\n{len(steps)} iterations recorded, {len(tracked)} players tracked: '
          f'{sum(sum(step["switched"]) for step in steps)} '
          f'switches')
    changes = [float(np.abs(np.array(later['rates']) - np.array(earlier['rates'])).mean())
               for earlier, later in zip(steps, steps[1:])]
    print(f'mean win-rate change between steps: first {changes[0]:.4f}, last {changes[-1]:.4f}')
    membership = [(step['iteration'], step['in_field']) for step in steps]
    changes_at = [(iteration, sum(a != b for a, b in zip(earlier, later)))
                  for (_, earlier), (iteration, later) in zip(membership, membership[1:]) if earlier != later]
    print(f'top-12 tags: {sum(steps[0]["in_field"])} of the tracked rows in the field at first; tags change at '
          f'{changes_at} (iteration, rows changed)')

    data_path = _VISUALIZATIONS_DIR / 'prepared_data' / 'self_play.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':        SEASON,
        # How many players the field is made of (its seats and spare): the league's own size, for the tag's text.
        'context_size':  session.current_settings['n_drafters'],
        'categories':    categories,
        'player_names':  [session.player_registry[player].name for player in tracked],
        'opening_rates': np.round(np.array(opening_rates, dtype=float), 4).tolist(),
        # Who the field is made of before self-play's first refresh: Level 0's top players.
        'opening_in_field': steps[0]['in_field'],
        'steps':         steps,
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
