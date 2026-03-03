"""
Runs HAgent.get_h_scores() for a given draft state and converts the result
to the /evaluate response payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from backend.session import get_session
from backend.models import (
    Candidate, GScoreRow, FlexAllocations, FlexRow,
    Roster, RosterAssignment, EvaluateResponse,
)


# ── Public entry point ────────────────────────────────────────────────────────

def run_evaluate(
    session_id: str,
    player_assignments: dict[str, list[str]],
    my_team_id: str,
    exclusion_list: list[str],
    remaining_cash: Optional[dict[str, float]],
    n_iterations: int,
) -> EvaluateResponse:

    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session '{session_id}' not found or expired")
    info       = session.info
    H          = session.H
    categories = session.current_params.get('categories', [])
    cp         = session.current_params

    # Run the generator
    H = H.clear_initial_weights()
    generator = H.get_h_scores(
        player_assignments    = player_assignments,
        drafter               = my_team_id,
        cash_remaining_per_team = remaining_cash,
        exclusion_list        = exclusion_list,
    )

    result = None
    actual_iterations = 0
    for _ in range(max(1, n_iterations)):
        result = next(generator)
        actual_iterations += 1

    if result is None:
        return EvaluateResponse(iteration=0, candidates=[])

    candidates = _build_candidates(result, info, H, categories, player_assignments,
                                   my_team_id, cp)

    return EvaluateResponse(
        iteration  = actual_iterations,
        candidates = candidates,
    )


# ── Build candidate list ──────────────────────────────────────────────────────

def _build_candidates(
    res: dict,
    info: dict,
    H,
    categories: list[str],
    player_assignments: dict[str, list[str]],
    my_team_id: str,
    cp: dict,
) -> list[Candidate]:

    scores         = res['Scores'].sort_values(ascending=False)  # pd.Series player → h_score (0–1), sorted descending
    weights_raw    = res['Weights']          # pd.DataFrame player × cat (raw weights)
    rates          = res['Rates']            # pd.DataFrame player × cat (cdf, 0–1)
    diff_df        = res['Diff']             # pd.DataFrame player × cat (total x-score diff)
    future_diff_df = res['Future-Diff']      # pd.DataFrame or None
    positions_s    = info['Positions']       # pd.Series player → list[str]
    g_scores_all   = info['G-scores']        # pd.DataFrame includes 'Total' col

    fits_roster = res['Rosters'].loc[:, 0] >= 0  # boolean Series

    n_cat = len(categories)
    v_reshaped = H.v.reshape(n_cat)           # (n_cat,) for division
    original_v = np.array(H.original_v)       # (n_cat,)

    # Category weights: normalize by H.v so 100 = neutral
    weights_norm = (weights_raw.values / v_reshaped) * 100   # (n_players, n_cat)

    my_players = [p for p in player_assignments.get(my_team_id, []) if isinstance(p, str)]
    position_structure = H.position_structure
    base_list = position_structure['base_list']
    slot_counts = cp.get('slot_counts', {})

    # Slot names in canonical order
    slot_names = _make_slot_names(slot_counts, position_structure)

    candidates = []
    for rank_idx, player in enumerate(scores.index):
        if not fits_roster.get(player, True):
            continue   # skip players that don't fit the roster

        h_score  = float(scores[player]) * 100
        win_rates_row = rates.loc[player].values * 100

        # category weights row
        cat_weights_row = weights_norm[scores.index.get_loc(player)]

        # G-score rows
        g_rows = _build_g_score_rows(
            player, categories, g_scores_all,
            diff_df, future_diff_df, original_v,
        )

        # Flex allocations
        flex_alloc = _build_flex_allocations(
            player, base_list, position_structure,
            res['Position-Shares'], slot_counts,
        )

        # Roster
        roster_obj = _build_roster(
            player, my_players, positions_s, slot_names, slot_counts,
            position_structure,
        )

        pos_raw = positions_s.get(player, ['?'])
        position_str = ','.join(pos_raw) if isinstance(pos_raw, list) else str(pos_raw)

        candidates.append(Candidate(
            name              = player,
            position          = position_str,
            h_score           = round(h_score, 2),
            h_rank            = rank_idx + 1,
            win_rates         = [round(float(v), 2) for v in win_rates_row],
            category_weights  = [round(float(v), 1) for v in cat_weights_row],
            g_score_rows      = g_rows,
            flex_allocations  = flex_alloc,
            roster            = roster_obj,
        ))

    return candidates


# ── G-score rows ──────────────────────────────────────────────────────────────

def _build_g_score_rows(
    player: str,
    categories: list[str],
    g_scores_all: pd.DataFrame,
    diff_df: pd.DataFrame | None,
    future_diff_df: pd.DataFrame | None,
    original_v: np.ndarray,
) -> list[GScoreRow]:

    n_cat = len(categories)

    # Player's own G-scores (excluding the 'Total' column)
    player_gs = (
        g_scores_all.loc[player, categories].values.astype(float)
        if player in g_scores_all.index
        else np.zeros(n_cat)
    )

    def _row(label, vals, is_total=False):
        v = [round(float(x), 2) for x in vals]
        return GScoreRow(label=label, values=v, total=round(float(sum(vals)), 2), is_total=is_total)

    # Mirrors make_main_df_styled() in src/tabs/candidate_subtabs.py.
    # res['Diff'] is the team diff WITHOUT the candidate player's own stats.
    if diff_df is None or diff_df.empty or player not in diff_df.index:
        return [
            _row('Current diff', np.zeros(n_cat)),
            _row(player,         player_gs),
            _row('Total diff',   player_gs, is_total=True),
        ]

    total_diff            = diff_df.loc[player, categories].values.astype(float) * original_v
    total_diff_plus_player = total_diff + player_gs   # always the Total row

    if future_diff_df is None or player not in future_diff_df.index:
        # No future component: 3-row layout matching original
        return [
            _row('Current diff', total_diff),
            _row(player,         player_gs),
            _row('Total diff',   total_diff_plus_player, is_total=True),
        ]

    future_diff  = future_diff_df.loc[player, categories].values.astype(float) * original_v
    current_diff = total_diff - future_diff

    return [
        _row('Current diff',      current_diff),
        _row(player,              player_gs),
        _row('Future diff',       future_diff),
        _row('Total diff',        total_diff_plus_player, is_total=True),
    ]


# ── Flex allocations ──────────────────────────────────────────────────────────

def _build_flex_allocations(
    player: str,
    base_list: list[str],
    position_structure: dict,
    position_shares: dict,
    slot_counts: dict,
) -> FlexAllocations:

    flex_list = position_structure['flex_list']
    flex_info = position_structure['flex']

    rows: list[FlexRow] = []
    total = {b: 0.0 for b in base_list}

    for flex_type in flex_list:
        n_slots = slot_counts.get(flex_type, 0)
        if n_slots == 0:
            continue

        bases = flex_info[flex_type]['bases']
        shares_df = position_shares.get(flex_type)

        if shares_df is None or player not in shares_df.index:
            row_values: list[float | None] = [None] * len(base_list)
        else:
            row_values = []
            for b in base_list:
                if b in bases:
                    share = float(shares_df.loc[player, b]) * n_slots
                    row_values.append(round(share, 2))
                    total[b] += share
                else:
                    row_values.append(None)

        rows.append(FlexRow(
            label    = f"{flex_type}-{n_slots}",
            values   = row_values,
            is_total = False,
        ))

    # Total row
    total_values = [round(total[b], 2) for b in base_list]
    rows.append(FlexRow(label='Total', values=total_values, is_total=True))

    return FlexAllocations(base_positions=base_list, rows=rows)


# ── Roster assignment ─────────────────────────────────────────────────────────

def _make_slot_names(slot_counts: dict, position_structure: dict) -> list[str]:
    """Return slot IDs in canonical order: base positions then flex."""
    order = position_structure['base_list'] + position_structure['flex_list']
    slots = []
    for pos in order:
        n = slot_counts.get(pos, 0)
        for i in range(1, n + 1):
            slots.append(f"{pos}{i}")
    return slots


def _build_roster(
    candidate: str,
    my_players: list[str],
    positions_s: pd.Series,
    slot_names: list[str],
    slot_counts: dict,
    position_structure: dict,
) -> Roster:
    """Compute optimal slot assignment for my_players + candidate, return Roster."""
    from scipy.optimize import linear_sum_assignment

    n_slots = len(slot_names)
    all_players = my_players + [candidate]
    n_players   = len(all_players)

    if n_players == 0:
        assignments = {s: None for s in slot_names}
        return Roster(slots=slot_names, assignments=assignments)

    # Build eligibility matrix: rows=players, cols=slots
    # value 0 = eligible, -inf = not eligible
    eligibility = np.full((n_players, n_slots), -np.inf)
    base_list   = position_structure['base_list']
    flex_info   = position_structure['flex']

    slot_position_list = _expand_slots(slot_counts, position_structure)  # list of pos per slot index

    for pi, pname in enumerate(all_players):
        pos_raw = positions_s.get(pname)
        if pos_raw is None:
            continue
        player_positions = pos_raw if isinstance(pos_raw, list) else list(pos_raw)

        for si, slot_pos in enumerate(slot_position_list):
            eligible = False
            if slot_pos in base_list:
                eligible = slot_pos in player_positions
            else:
                # flex slot: eligible if any of the flex bases match
                bases = flex_info.get(slot_pos, {}).get('bases', [])
                eligible = any(b in player_positions for b in bases)
            if eligible:
                eligibility[pi, si] = 0

    # Run linear assignment (maximise → use the 0 / -inf matrix)
    try:
        row_ind, col_ind = linear_sum_assignment(eligibility, maximize=True)
    except Exception:
        # Fall back to empty assignment
        assignments = {s: None for s in slot_names}
        return Roster(slots=slot_names, assignments=assignments)

    assignments: dict[str, RosterAssignment | None] = {s: None for s in slot_names}
    for pi, si in zip(row_ind, col_ind):
        if eligibility[pi, si] == 0:
            pname = all_players[pi]
            assignments[slot_names[si]] = RosterAssignment(
                name         = pname,
                is_candidate = (pname == candidate),
            )

    return Roster(slots=slot_names, assignments=assignments)


def _expand_slots(slot_counts: dict, position_structure: dict) -> list[str]:
    """Return a flat list of position types, one entry per slot (in order)."""
    order = position_structure['base_list'] + position_structure['flex_list']
    slots = []
    for pos in order:
        n = slot_counts.get(pos, 0)
        slots.extend([pos] * n)
    return slots
