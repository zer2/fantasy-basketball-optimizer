"""
Trade analysis and suggestion engine.

Ported from src/tabs/trading.py, adapted to use the backend Session pattern
(session.H, session.info) instead of Streamlit session state.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from backend.session import Session
from backend.models import (
    TeamHScore, TeamTradeResult, TradeAnalyzeResponse,
    TradeSuggestion, TradeSuggestResponse, ComboParam,
)
from backend.math.position_optimization import check_team_eligibility


# ── Core trade analysis ──────────────────────────────────────────────────────

def analyze_trade(
    session: Session,
    player_assignments: dict[str, list[str]],
    team_1: str,
    team_1_trade: list[str],
    team_2: str,
    team_2_trade: list[str],
    n_iterations: int,
    ignore_position_check: bool = False,
) -> Optional[dict]:
    """Compute pre/post H-scores for both teams after a trade.

    Returns None if the trade violates position structure for either team.
    Otherwise returns {1: {pre: ..., post: ...}, 2: {pre: ..., post: ...}}.
    """
    info = session.info
    H = session.H

    post_trade_team_1 = [p for p in player_assignments[team_1] if p not in team_1_trade] + team_2_trade
    post_trade_team_2 = [p for p in player_assignments[team_2] if p not in team_2_trade] + team_1_trade

    # Check position eligibility for both teams
    if not ignore_position_check:
        pos_cfg = session.H._pos_cfg
        team_1_positions = info['Positions'].loc[post_trade_team_1]
        if not check_team_eligibility(team_1_positions, pos_cfg):
            return None

        team_2_positions = info['Positions'].loc[post_trade_team_2]
        if not check_team_eligibility(team_2_positions, pos_cfg):
            return None

    post_trade_assignments = player_assignments.copy()
    post_trade_assignments[team_1] = post_trade_team_1
    post_trade_assignments[team_2] = post_trade_team_2

    # Pre-trade H-scores
    res_1_pre = next(H.get_h_scores(player_assignments, team_1))
    res_2_pre = next(H.get_h_scores(player_assignments, team_2))

    # Post-trade H-scores (iterate if uneven trade)
    n_player_diff = len(team_1_trade) - len(team_2_trade)

    if n_player_diff > 0:
        generator = H.get_h_scores(post_trade_assignments, team_1)
        for _ in range(n_iterations):
            res_1_post = next(generator)
        res_2_post = next(H.get_h_scores(post_trade_assignments, team_2))
    elif n_player_diff == 0:
        res_1_post = next(H.get_h_scores(post_trade_assignments, team_1))
        res_2_post = next(H.get_h_scores(post_trade_assignments, team_2))
    else:
        res_1_post = next(H.get_h_scores(post_trade_assignments, team_1))
        generator = H.get_h_scores(post_trade_assignments, team_2)
        for _ in range(n_iterations):
            res_2_post = next(generator)

    def _extract(result: dict) -> tuple[float, list[float]]:
        scores = result['Scores']
        rates = result['Rates']
        idx = scores.idxmax()
        return float(scores[idx]), rates.loc[idx].tolist()

    h1_pre, r1_pre = _extract(res_1_pre)
    h1_post, r1_post = _extract(res_1_post)
    h2_pre, r2_pre = _extract(res_2_pre)
    h2_post, r2_post = _extract(res_2_post)

    return {
        1: {
            'pre':  TeamHScore(h_score=h1_pre, rates=r1_pre),
            'post': TeamHScore(h_score=h1_post, rates=r1_post),
        },
        2: {
            'pre':  TeamHScore(h_score=h2_pre, rates=r2_pre),
            'post': TeamHScore(h_score=h2_post, rates=r2_post),
        },
    }


def run_trade_analyze(
    session: Session,
    player_assignments: dict[str, list[str]],
    my_team: str,
    their_team: str,
    my_trade: list[str],
    their_trade: list[str],
    ignore_position_check: bool = False,
) -> TradeAnalyzeResponse:
    """Public entry point for the trade/analyze endpoint."""
    if len(my_trade) == 0 or len(their_trade) == 0:
        return TradeAnalyzeResponse(error='A trade must include at least one player from each team.')

    if abs(len(my_trade) - len(their_trade)) > 6:
        return TradeAnalyzeResponse(error="Too lopsided of a trade!")

    n_iterations = session.current_params.get('n_iterations', 30)
    result = analyze_trade(session, player_assignments, my_team, my_trade, their_team, their_trade, n_iterations, ignore_position_check)

    if result is None:
        return TradeAnalyzeResponse(
            error='Trade is impermissible because it violates position structure for at least one team.',
        )

    return TradeAnalyzeResponse(
        your_team=TeamTradeResult(pre=result[1]['pre'], post=result[1]['post']),
        their_team=TeamTradeResult(pre=result[2]['pre'], post=result[2]['post']),
    )


# ── Trade value estimation ───────────────────────────────────────────────────

def _analyze_trade_value(
    H: object,
    player: str,
    team: str,
    player_assignments: dict[str, list[str]],
) -> float:
    """Estimate how valuable a player is to a particular team.

    Compares H-score with vs. without the player.
    """
    without_player = player_assignments.copy()
    without_player[team] = [p for p in without_player[team] if p != player]

    with_player = player_assignments.copy()
    if player not in with_player[team]:
        with_player[team] = with_player[team] + [player]

    res_without = next(H.get_h_scores(without_player, team, exclusion_list=[player]))
    res_with = next(H.get_h_scores(with_player, team))

    return float(res_with['Scores'].max() - res_without['Scores'].max())


def _identify_trade_candidates(
    H: object,
    my_team: str,
    their_team: str,
    player_assignments: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Identify players from each team that are relatively more valuable to the other.

    Returns (my_candidates, their_candidates).
    """
    my_players = player_assignments[my_team]
    their_players = player_assignments[their_team]

    my_values_to_me = pd.Series(
        [_analyze_trade_value(H, p, my_team, player_assignments) for p in my_players],
        index=my_players,
    )
    my_values_to_me -= my_values_to_me.mean()

    their_values_to_me = pd.Series(
        [_analyze_trade_value(H, p, my_team, player_assignments) for p in their_players],
        index=their_players,
    )
    their_values_to_me -= their_values_to_me.mean()

    my_values_to_them = pd.Series(
        [_analyze_trade_value(H, p, their_team, player_assignments) for p in my_players],
        index=my_players,
    )
    my_values_to_them -= my_values_to_them.mean()

    their_values_to_them = pd.Series(
        [_analyze_trade_value(H, p, their_team, player_assignments) for p in their_players],
        index=their_players,
    )
    their_values_to_them -= their_values_to_them.mean()

    my_differences = my_values_to_me - my_values_to_them
    their_differences = their_values_to_them - their_values_to_me

    my_candidates = [p for p in my_players if my_differences[p] < 0]
    their_candidates = [p for p in their_players if their_differences[p] < 0]

    return my_candidates, their_candidates


# ── Combination generation ───────────────────────────────────────────────────

def _get_combos(
    players_with_weight: list[tuple[str, float]],
    n: int,
) -> list[tuple[list[str], float]]:
    """Generate all n-player combinations with summed general values."""
    combos = list(itertools.combinations(players_with_weight, n))
    return [
        (list(z[0] for z in m), sum(z[1] for z in m))
        for m in combos
    ]


def _get_cross_combos(
    n: int,
    m: int,
    my_players: list[str],
    their_players: list[str],
    general_values: pd.Series,
    replacement_value: float,
    value_threshold: float,
) -> pd.DataFrame:
    """Generate filtered cross-product of (n-from-mine, m-from-theirs) combos.

    Filters by absolute general-value differential within the threshold.
    """
    my_pw = [(p, general_values[p]) for p in my_players if p in general_values.index]
    their_pw = [(p, general_values[p]) for p in their_players if p in general_values.index]

    cross = list(itertools.product(_get_combos(my_pw, n), _get_combos(their_pw, m)))

    if len(cross) == 0:
        return pd.DataFrame(columns=['My Trade', 'My Value', 'Their Trade', 'Their Value'])

    full = pd.DataFrame(cross)
    separated = pd.concat([
        pd.DataFrame(full[0].tolist(), index=full.index),
        pd.DataFrame(full[1].tolist(), index=full.index),
    ], axis=1)
    separated.columns = ['My Trade', 'My Value', 'Their Trade', 'Their Value']

    if n != m:
        separated['My Value'] += replacement_value * (m - n)

    diff = separated['My Value'] - separated['Their Value']
    meets = abs(diff) <= value_threshold / 100  # threshold is in percentage terms

    return separated[meets]


def _make_combo_df(
    session: Session,
    all_combos: pd.DataFrame,
    my_team: str,
    their_team: str,
    player_assignments: dict[str, list[str]],
    ignore_position_check: bool = False,
) -> pd.DataFrame:
    """Evaluate each trade combo and return a DataFrame sorted by Your Score."""
    H = session.H
    rows = []

    for _, row in all_combos.iterrows():
        my_trade = row['My Trade']
        their_trade = row['Their Trade']

        result = analyze_trade(
            session, player_assignments,
            my_team, my_trade, their_team, their_trade,
            n_iterations=1,
            ignore_position_check=ignore_position_check,
        )

        if result is not None:
            your_diff = result[1]['post'].h_score - result[1]['pre'].h_score
            their_diff = result[2]['post'].h_score - result[2]['pre'].h_score
            rows.append({
                'Send': my_trade,
                'Receive': their_trade,
                'Your Score': your_diff,
                'Their Score': their_diff,
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=['Send', 'Receive', 'Your Score', 'Their Score'])

    df = pd.DataFrame(rows)
    return df.sort_values('Your Score', ascending=False)


# ── Suggestion orchestrator ──────────────────────────────────────────────────

def _get_general_values(session: Session) -> pd.Series:
    """Get default H-score values for all players (no assignments)."""
    H = session.H
    cp = session.current_params
    n_drafters = cp['n_drafters']
    n_iterations = cp.get('n_iterations', 30)

    H_clean = H.clear_initial_weights()
    generator = H_clean.get_h_scores(
        player_assignments={f'Team {i+1}': [] for i in range(n_drafters)},
        drafter='Team 1',
    )

    res = None
    for _ in range(max(1, n_iterations)):
        res = next(generator)

    return res['Scores'].sort_values(ascending=False)


def run_trade_suggest(
    session: Session,
    player_assignments: dict[str, list[str]],
    my_team: str,
    their_team: str,
    combo_params: list[ComboParam],
    your_threshold: float,
    their_threshold: float,
    ignore_position_check: bool = False,
) -> TradeSuggestResponse:
    """Public entry point for the trade/suggest endpoint."""
    H = session.H

    # Step 1: identify which players are trade candidates
    my_candidates, their_candidates = _identify_trade_candidates(
        H, my_team, their_team, player_assignments,
    )

    if len(my_candidates) == 0 or len(their_candidates) == 0:
        return TradeSuggestResponse(suggestions=[])

    # Step 2: get general values for filtering
    general_values = _get_general_values(session)

    n_picks = session.current_params['n_picks']
    n_drafters = session.current_params['n_drafters']
    total_players = n_picks * n_drafters
    replacement_value = float(general_values.iloc[min(total_players, len(general_values) - 1)])

    # Step 3: generate all combos across param sets
    all_combo_frames = []
    for cp in combo_params:
        combos = _get_cross_combos(
            cp.n_traded, cp.n_received,
            my_candidates, their_candidates,
            general_values, replacement_value,
            cp.threshold,
        )
        if len(combos) > 0:
            all_combo_frames.append(combos)

    if len(all_combo_frames) == 0:
        return TradeSuggestResponse(suggestions=[])

    all_combos = pd.concat(all_combo_frames, ignore_index=True)

    # Step 4: evaluate each combo
    df = _make_combo_df(session, all_combos, my_team, their_team, player_assignments, ignore_position_check)

    # Step 5: filter by thresholds
    mask = (df['Your Score'] > your_threshold) & (df['Their Score'] > their_threshold)
    df = df[mask]

    suggestions = [
        TradeSuggestion(
            send=row['Send'],
            receive=row['Receive'],
            your_score=round(float(row['Your Score']), 4),
            their_score=round(float(row['Their Score']), 4),
        )
        for _, row in df.iterrows()
    ]

    return TradeSuggestResponse(suggestions=suggestions)
