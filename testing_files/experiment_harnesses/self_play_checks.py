"""Checks of a session's served board and opponent field that hold whatever process built them.

Self-play produces two things: a field (each modelled opponent seat's player and that player's committed per-pick
tilt) and a served board (every player's build and H-score against that field). These checks read only those -- the
agent's stored board and field state -- so they rerun unchanged when the self-play process changes.

  - measure_best_response_shortfalls: is each top served build its player's best response to the field? The served
    H-score against the best any of several long descents reaches against the same field, from the served build, the
    balanced build and a gentle punt of every category. A shortfall means the served build sits on a lower peak.
  - measure_field_consistency: does the field describe the board it serves? At a consistent field the seats are the
    top players, the builds served to the seated players are the builds the field assumes for them, and, because win
    probabilities are pairwise complementary, the top players average close to 50% in every category against each
    other.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BestResponseCheck:
    player_id: int
    served_h_score: float          # percent
    best_h_score: float            # percent
    best_start: str                # the start whose descent reached best_h_score
    served_win_rates: np.ndarray   # (categories,), 0-1
    best_win_rates: np.ndarray     # (categories,), 0-1

    @property
    def shortfall(self) -> float:
        return self.best_h_score - self.served_h_score


@dataclass
class FieldConsistency:
    top_players: list              # the top served players, top_count of them
    seated_players: list           # the field's seats and spare, as many as top_players
    # (categories,): the seated players' mean served win rate, 0-1. Win probabilities are pairwise complementary only
    # among the players who make up the field, so this is the set the 50% check at C = 1 applies to.
    seated_category_win_rates: np.ndarray
    tilt_gap_by_category: np.ndarray      # (categories,): mean over seats of served tilt minus field tilt (per pick)
    mean_absolute_tilt_gap: float         # mean over seats and categories of |served tilt - field tilt|
    seat_gaps: np.ndarray                 # (seats,): per seated player, the mean over categories of that |gap|


def list_descent_starts(agent) -> list:
    """(label, weights) starts that explore every punt lane: the served build is added per player; these are the
    balanced build and a gentle punt of each category (the app's shared seeds, which already span the lanes)."""
    seeds = agent.list_shared_seeds()
    categories = list(agent.x_scores.columns)
    labels = [f'{category} at 50%' for category in categories] + ['balanced']
    if len(seeds) != len(labels):
        raise ValueError(f'Expected one gentle punt per category plus the balanced build, got {len(seeds)} seeds.')
    return list(zip(labels, seeds))


def solve_from_start(agent, empty_board, drafter, player, start_weights, iterations):
    """One descent of `player` alone against the stored field on the empty board -- the serve's view -- from the
    given weights, through the agent's start-injection hook. Returns (H-score in percent, win rates 0-1)."""
    agent.reset_draft_state()
    agent.initial_category_weights = np.asarray(start_weights, dtype=float) / np.sum(start_weights)
    agent.initial_shares = {code: {base: 1 / len(info['bases']) for base in info['bases']}
                            for code, info in agent.position_structure['flex'].items()}
    try:
        result = agent.get_h_scores(player_assignments=empty_board, drafter=drafter, n_iterations=iterations,
                                    candidate_subset=[player])
    finally:
        agent.clear_initial_weights()
    return 100 * float(result['Scores'].loc[player]), result['Rates'].loc[player].to_numpy(dtype=float)


def measure_best_response_shortfalls(session, top_count, iterations) -> list:
    """For each of the top_count served players: its served H-score and win rates, and the best H-score any long
    descent (iterations) reaches against the same field, from the served build and from every exploring start."""
    agent = session.agent
    teams = session.current_settings['team_names']
    empty_board = {team: [] for team in teams}
    rates = agent._default_result['Rates']
    checks = []
    for player in list(agent.default_h_scores.index[:top_count]):
        starts = [('served build', agent._player_frozen_weights.loc[player].to_numpy(dtype=float))]
        starts += list_descent_starts(agent)
        best_score, best_label, best_rates = -np.inf, None, None
        for label, weights in starts:
            score, win_rates = solve_from_start(agent, empty_board, teams[0], player, weights, iterations)
            if score > best_score:
                best_score, best_label, best_rates = score, label, win_rates
        checks.append(BestResponseCheck(
            player_id=int(player), served_h_score=100 * float(agent.default_h_scores[player]),
            best_h_score=best_score, best_start=best_label,
            served_win_rates=rates.loc[player].to_numpy(dtype=float), best_win_rates=best_rates))
    agent.reset_draft_state()
    return checks


def measure_field_consistency(session, top_count) -> FieldConsistency:
    """The served board against the field it was served on (see the module docstring)."""
    agent = session.agent
    rates = agent._default_result['Rates']
    top_players = list(agent.default_h_scores.index[:top_count])
    seated_players = list(agent._anchor_player_order[:top_count])
    served_tilts = agent._default_result['Future-Diff'] / (agent.n_picks - 1)
    field_tilts = agent._player_committed_future_diffs
    seated_gaps = (served_tilts.loc[seated_players] - field_tilts.loc[seated_players]).to_numpy(dtype=float)
    return FieldConsistency(
        top_players=[int(player) for player in top_players],
        seated_players=[int(player) for player in seated_players],
        seated_category_win_rates=rates.loc[seated_players].to_numpy(dtype=float).mean(axis=0),
        tilt_gap_by_category=seated_gaps.mean(axis=0),
        mean_absolute_tilt_gap=float(np.abs(seated_gaps).mean()),
        seat_gaps=np.abs(seated_gaps).mean(axis=1),
    )


@dataclass
class ServeChanges:
    players: list                  # the players the field is made of (its seats and spare), in seat order
    self_play_ranks: list          # each one's place among them by its self-play H-score (1 = first)
    served_ranks: list             # ... and by its served H-score
    category_changes: np.ndarray   # (players,): mean over categories of |served win rate - self-play win rate|, in %


def measure_serve_changes(session) -> ServeChanges:
    """For the players the field is made of: where self-play left each one (its build's win rates and H-score at the
    end of the joint descent) against what the served board gives it. Self-play's own end state is not stored, so the
    session's populate is run again with a recorder on the field's start selection; it is deterministic, so the served
    board it leaves is the one it had."""
    import pandas as pd
    from backend.math.algorithm_agents import _SelfPlayField
    agent = session.agent
    final = {}
    original_objective = type(agent).get_objective_and_gradient
    original_update = _SelfPlayField.update_best_starts

    def recording_objective(*arguments, **keyword_arguments):
        result = original_objective(agent, *arguments, **keyword_arguments)
        if agent._self_play_field is not None:
            final['result'] = result
        return result

    def recording_update(field, scores):
        rows = original_update(field, scores)
        final.update(rows=rows, players=list(field.players))
        return rows

    agent.get_objective_and_gradient = recording_objective
    _SelfPlayField.update_best_starts = recording_update
    try:
        agent.populate_default_h_scores(session.current_settings['n_iterations'])
    finally:
        del agent.get_objective_and_gradient   # drop the instance shadow, restoring the class method
        _SelfPlayField.update_best_starts = original_update
    if 'rows' not in final:
        raise RuntimeError('Self-play did not run: the session has no opponent field to compare the serve with.')
    rows, result = final['rows'], final['result']
    self_play_rates = pd.DataFrame(np.asarray(result['CDF-Estimates']).mean(axis=2)[rows], index=final['players'],
                                   columns=agent.x_scores.columns)
    self_play_scores = pd.Series(np.asarray(result['Score'], dtype=float)[rows], index=final['players'])
    players = list(agent._anchor_player_order[: agent.n_drafters])
    served_rates = agent._default_result['Rates']
    served_scores = agent.default_h_scores
    by_self_play = sorted(players, key=lambda player: -float(self_play_scores[player]))
    by_served = sorted(players, key=lambda player: -float(served_scores[player]))
    changes = 100 * np.abs(served_rates.loc[players].to_numpy(dtype=float)
                           - self_play_rates.loc[players].to_numpy(dtype=float)).mean(axis=1)
    return ServeChanges(players=players,
                        self_play_ranks=[by_self_play.index(player) + 1 for player in players],
                        served_ranks=[by_served.index(player) + 1 for player in players],
                        category_changes=changes)
