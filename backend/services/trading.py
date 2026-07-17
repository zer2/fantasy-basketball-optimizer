"""
Trade analysis and suggestion engine.

Ported from src/tabs/trading.py, adapted to use the backend Session pattern
(session.scorer.h_agent, session.scorer.info) instead of Streamlit session state.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from backend.state.session import Session
from backend.models import (
    TeamHScore, TeamTradeResult, TradeAnalyzeResponse,
    TradeSuggestion, TradeSuggestResponse, ComboParam,
)
from backend.math.position_optimization import check_team_eligibility


# ── Core trade analysis ──────────────────────────────────────────────────────

def analyze_trade(
    session: Session
    , player_assignments: dict[str, list[str]]
    , team_1: str
    , team_1_trade: list[str]
    , team_2: str
    , team_2_trade: list[str]
    , n_iterations: int
    , ignore_position_check: bool = False
) -> Optional[dict]:
    """Compute pre/post H-scores for both teams after a trade.

    Returns None if the trade violates position structure for either team.
    Otherwise returns {1: {pre: ..., post: ...}, 2: {pre: ..., post: ...}}.
    """
    info = session.scorer.info
    H = session.scorer.h_agent

    post_trade_team_1 = [p for p in player_assignments[team_1] if p not in team_1_trade] + team_2_trade
    post_trade_team_2 = [p for p in player_assignments[team_2] if p not in team_2_trade] + team_1_trade

    # Check position eligibility for both teams
    if not ignore_position_check:
        pos_cfg = session.scorer.h_agent._pos_cfg
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
    res_1_pre = H.get_h_scores(player_assignments, team_1, n_iterations)
    res_2_pre = H.get_h_scores(player_assignments, team_2, n_iterations)

    # Post-trade H-scores
    res_1_post = H.get_h_scores(post_trade_assignments, team_1, n_iterations)
    res_2_post = H.get_h_scores(post_trade_assignments, team_2, n_iterations)

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
    session: Session
    , player_assignments: dict[str, list[str]]
    , my_team: str
    , their_team: str
    , my_trade: list[str]
    , their_trade: list[str]
    , ignore_position_check: bool = False
) -> TradeAnalyzeResponse:
    """Public entry point for the trade/analyze endpoint."""
    if len(my_trade) == 0 or len(their_trade) == 0:
        return TradeAnalyzeResponse(error='A trade must include at least one player from each team.')

    if abs(len(my_trade) - len(their_trade)) > 6:
        return TradeAnalyzeResponse(error="Too lopsided of a trade!")

    n_iterations = session.current_params['n_iterations']
    result = analyze_trade(session, player_assignments, my_team, my_trade, their_team, their_trade, n_iterations, ignore_position_check)

    if result is None:
        return TradeAnalyzeResponse(
            error='Trade is impermissible because it violates position structure for at least one team.',
        )

    return TradeAnalyzeResponse(
        your_team=TeamTradeResult(pre=result[1]['pre'], post=result[1]['post']),
        their_team=TeamTradeResult(pre=result[2]['pre'], post=result[2]['post']),
    )


# ── Fast trade evaluation ────────────────────────────────────────────────────

def _build_trade_context(
    H: object
    , player_assignments: dict[str, list[str]]
    , my_team: str
    , their_team: str
) -> dict:
    """Pre-compute all quantities that are constant across every trade combo.

    For complete rosters in a k-for-k trade:
      - x_scores_available is identical pre- and post-trade (same pool, just swapped)
      - diff_vars depend only on team sizes (unchanged by equal trades)
      - Pre-trade H-scores are fixed regardless of which combo we evaluate
      - Baseline diff_means change only in the single column corresponding to the
        counterparty team — the delta is pure arithmetic per combo
    """
    team_names = list(player_assignments.keys())
    n_drafters = len(player_assignments)

    # Available player pool is the same for every combo (traded players are assigned
    # to one team or the other regardless of which swap we evaluate).
    players_chosen     = [p for roster in player_assignments.values() for p in roster if p == p]
    x_scores_available = H.x_scores[
        ~H.x_scores.index.isin(players_chosen)
        & H.x_scores.index.isin(H.positions.index)
    ]

    # Pre-compute each team's raw stat sum once.
    team_sums = {
        team: np.array(H.x_scores.loc[[p for p in roster if p == p]].sum(axis=0))
        for team, roster in player_assignments.items()
    }

    def build_baseline_diff_means(drafter: str) -> np.ndarray:
        """Construct the (1, n_categories, n_drafters-1) diff_means array for drafter."""
        drafter_roster = [p for p in player_assignments[drafter] if p == p]
        n_my           = len(drafter_roster)
        # Compare complete teams at their actual size (n_my), not n_my + 1. A k-for-k trade adds no
        # player, so padding opponents up to n_my + 1 would seat a phantom below-replacement player on
        # each of them and inflate the H-score — the same phantom-+1 bug fixed in get_diff_distributions.
        # mean_extra stays as the shortfall model (pads a short-handed opponent), a no-op when full.
        extra_needed   = n_my * n_drafters - len(players_chosen)
        mean_extra     = np.array(x_scores_available.iloc[:extra_needed].mean().fillna(0))

        other_sums = np.vstack([
            (team_sums[team] + max(n_my - len(player_assignments[team]), 0) * mean_extra)
            .reshape(1, H.n_categories, 1)
            for team in team_names if team != drafter
        ]).T  # (1, n_categories, n_drafters-1)

        return (
            team_sums[drafter].reshape(1, H.n_categories, 1)
            - other_sums.reshape(1, H.n_categories, n_drafters - 1)
        )

    baseline_diff_means_my    = build_baseline_diff_means(my_team)
    baseline_diff_means_their = build_baseline_diff_means(their_team)

    # diff_vars depend only on team sizes — constant for equal-size trades.
    diff_vars_my = np.vstack([
        H.get_diff_var(len([p for p in player_assignments[team] if p == p]))
        for team in team_names if team != my_team
    ]).T.reshape(1, H.n_categories, n_drafters - 1)

    diff_vars_their = np.vstack([
        H.get_diff_var(len([p for p in player_assignments[team] if p == p]))
        for team in team_names if team != their_team
    ]).T.reshape(1, H.n_categories, n_drafters - 1)

    # Pre-trade H-scores — constant across all combos, compute once.
    my_players    = [p for p in player_assignments[my_team]    if p == p]
    their_players = [p for p in player_assignments[their_team] if p == p]

    pre_my_h = H.compute_h_score_from_diff_means(
        diff_means = baseline_diff_means_my,
        diff_vars  = diff_vars_my,
    )
    pre_their_h = H.compute_h_score_from_diff_means(
        diff_means = baseline_diff_means_their,
        diff_vars  = diff_vars_their,
    )

    # Pre-slice x_scores for each trading team so _make_combo_df never touches H.x_scores.
    my_x_scores    = H.x_scores.loc[my_players]
    their_x_scores = H.x_scores.loc[their_players]

    # Column indices for the two trading teams in each perspective's diff_means.
    their_col_in_my_view = [t for t in team_names if t != my_team].index(their_team)
    my_col_in_their_view = [t for t in team_names if t != their_team].index(my_team)

    return {
        'my_x_scores':                 my_x_scores,
        'their_x_scores':              their_x_scores,
        'baseline_diff_means_my':      baseline_diff_means_my,
        'baseline_diff_means_their':   baseline_diff_means_their,
        'diff_vars_my':                diff_vars_my,
        'diff_vars_their':             diff_vars_their,
        'pre_my_h':                    pre_my_h,
        'pre_their_h':                 pre_their_h,
        'their_col_in_my_view':        their_col_in_my_view,
        'my_col_in_their_view':        my_col_in_their_view,
    }


# ── Combination generation ───────────────────────────────────────────────────

def _get_combos(
    players_with_weight: list[tuple[str, float]]
    , n: int
) -> list[tuple[list[str], float]]:
    """Generate all n-player combinations with summed general values."""
    combos = list(itertools.combinations(players_with_weight, n))
    return [
        (list(z[0] for z in m), sum(z[1] for z in m))
        for m in combos
    ]


def _get_cross_combos(
    n: int
    , m: int
    , my_players: list[str]
    , their_players: list[str]
    , general_values: pd.Series
    , replacement_value: float
    , value_threshold: float
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
    H: object
    , all_combos: pd.DataFrame
    , my_team: str
    , their_team: str
    , player_assignments: dict[str, list[str]]
    , my_x_numpy: np.ndarray
    , their_x_numpy: np.ndarray
    , my_name_to_index: dict[str, int]
    , their_name_to_index: dict[str, int]
) -> pd.DataFrame:
    """Evaluate every trade combo in one vectorized pass and return sorted by Your Score.

    For each combo, trade_delta = received_sum - sent_sum shifts diff_means on ALL
    opponent columns (my roster improved vs every opponent), with an extra shift on
    the counterparty column (their roster also weakened). Both perspectives are
    computed in a single batched call to compute_h_scores_batched.
    """
    ctx = _build_trade_context(H, player_assignments, my_team, their_team)

    n_combos = len(all_combos)
    if n_combos == 0:
        return pd.DataFrame(columns=['Send', 'Receive', 'Your Score', 'Their Score'])

    # Sum x_scores for each combo's sent / received players: (n_combos, n_categories)
    # Uses numpy integer indexing to avoid per-combo pandas overhead.
    # When all combos have the same size, builds a uniform 2D index array for a
    # fully vectorized sum. When sizes are mixed (asymmetric), falls back to a
    # per-row loop that still uses numpy rather than pandas.
    sent_sizes     = set(len(sent)     for sent     in all_combos['My Trade'])
    received_sizes = set(len(received) for received in all_combos['Their Trade'])

    if len(sent_sizes) == 1 and len(received_sizes) == 1:
        sent_index_array     = np.array([[my_name_to_index[p]    for p in sent]     for sent     in all_combos['My Trade']])
        received_index_array = np.array([[their_name_to_index[p] for p in received] for received in all_combos['Their Trade']])
        sent_sums     = my_x_numpy[sent_index_array].sum(axis=1)
        received_sums = their_x_numpy[received_index_array].sum(axis=1)
    else:
        sent_sums     = np.array([my_x_numpy[   [my_name_to_index[p]    for p in sent]    ].sum(axis=0) for sent     in all_combos['My Trade']])
        received_sums = np.array([their_x_numpy[[their_name_to_index[p] for p in received]].sum(axis=0) for received in all_combos['Their Trade']])
    trade_deltas = received_sums - sent_sums  # (n_combos, n_categories)

    # Build all post diff_means at once: (n_combos, n_categories, n_drafters-1)
    post_diff_means_my    = ctx['baseline_diff_means_my']    + trade_deltas[:, :, np.newaxis]
    post_diff_means_their = ctx['baseline_diff_means_their'] - trade_deltas[:, :, np.newaxis]

    post_diff_means_my   [:, :, ctx['their_col_in_my_view']]    += trade_deltas
    post_diff_means_their[:, :, ctx['my_col_in_their_view']]    -= trade_deltas

    post_my_h_scores    = H.compute_h_scores_batched(post_diff_means_my,    ctx['diff_vars_my'])
    post_their_h_scores = H.compute_h_scores_batched(post_diff_means_their, ctx['diff_vars_their'])

    df = pd.DataFrame({
        'Send':        list(all_combos['My Trade']),
        'Receive':     list(all_combos['Their Trade']),
        'Your Score':  post_my_h_scores    - ctx['pre_my_h'],
        'Their Score': post_their_h_scores - ctx['pre_their_h'],
    })
    return df.sort_values('Your Score', ascending=False)


# ── Suggestion orchestrator ──────────────────────────────────────────────────

def _get_general_values(session: Session) -> pd.Series:
    """Get default H-score values for all players (no assignments).

    Cached on session.scorer.generic_h_scores after the first call — the same field
    used by auction mode — so repeated trade searches within a session pay
    the cost only once.
    """
    if session.scorer.generic_h_scores is not None:
        return session.scorer.generic_h_scores

    H = session.scorer.h_agent
    cp = session.current_params
    n_drafters = cp['n_drafters']
    n_iterations = cp['n_iterations']

    H_clean = H.clear_initial_weights()
    result = H_clean.get_h_scores(
        player_assignments={f'Team {i+1}': [] for i in range(n_drafters)},
        drafter='Team 1',
        n_iterations=n_iterations,
    )
    session.scorer.generic_h_scores = result['Scores'].sort_values(ascending=False)
    return session.scorer.generic_h_scores


def run_trade_suggest(
    session: Session
    , player_assignments: dict[str, list[str]]
    , my_team: str
    , their_team: str
    , combo_params: list[ComboParam]
    , your_threshold: float
    , their_threshold: float
    , ignore_position_check: bool = False
) -> TradeSuggestResponse:
    """Public entry point for the trade/suggest endpoint."""

    # Step 1: get general values for filtering
    general_values = _get_general_values(session)

    n_picks = session.current_params['n_picks']
    n_drafters = session.current_params['n_drafters']
    total_players = n_picks * n_drafters
    replacement_value = float(general_values.iloc[min(total_players, len(general_values) - 1)])

    # Step 2: use all players as candidates (candidate filter disabled)
    my_candidates    = player_assignments[my_team]
    their_candidates = player_assignments[their_team]

    # Step 2b: pre-extract x_scores for the two teams as numpy arrays.
    # Avoids per-combo pandas .loc[].sum() overhead in _make_combo_df.
    my_x_numpy    = session.scorer.h_agent.x_scores.loc[my_candidates].to_numpy()
    their_x_numpy = session.scorer.h_agent.x_scores.loc[their_candidates].to_numpy()
    my_name_to_index    = {name: i for i, name in enumerate(my_candidates)}
    their_name_to_index = {name: i for i, name in enumerate(their_candidates)}

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

    # Step 3: evaluate all combos vectorized
    df = _make_combo_df(
        session.scorer.h_agent
        , all_combos
        , my_team
        , their_team
        , player_assignments
        , my_x_numpy
        , their_x_numpy
        , my_name_to_index
        , their_name_to_index
    )

    # Step 4: filter by score thresholds
    mask = (df['Your Score'] > your_threshold) & (df['Their Score'] > their_threshold)
    df = df[mask]

    # Step 5: position check only on the small set that passed the thresholds
    if not ignore_position_check and len(df) > 0:
        pos_cfg = session.scorer.h_agent._pos_cfg
        valid = []
        for _, row in df.iterrows():
            sent, received = row['Send'], row['Receive']
            post_my_roster    = [p for p in player_assignments[my_team]    if p not in sent]    + received
            post_their_roster = [p for p in player_assignments[their_team] if p not in received] + sent
            if (check_team_eligibility(session.scorer.info['Positions'].loc[post_my_roster],    pos_cfg)
            and check_team_eligibility(session.scorer.info['Positions'].loc[post_their_roster], pos_cfg)):
                valid.append(row)
        df = pd.DataFrame(valid) if valid else df.iloc[:0]

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
