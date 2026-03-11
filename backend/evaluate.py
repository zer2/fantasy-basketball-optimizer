"""
Runs HAgent.get_h_scores() for a given draft state and converts the result
to the /evaluate response payload.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional

from backend.session import get_session
from backend.models import (
    Candidate, GScoreRow, FlexAllocations, FlexRow,
    Roster, RosterAssignment, AuctionValues, EvaluateResponse,
)
from backend.math.algorithm_helpers import auction_value_adjuster

def _last_name(player_full_name: str) -> str:
    """Return the player's last name, stripping any trailing position suffix.

    For example: "Nikola Jokic (C, PF)" → "Jokic", "LeBron James (PF, SF)" → "James",
    "Nikola Jokic" → "Jokic".
    """
    return ' '.join(player_full_name.split(' (')[0].split(' ')[1:])


# ── Public entry point ────────────────────────────────────────────────────────

def run_evaluate(
    session_id: str,
    player_assignments: dict[str, list[str]],
    my_team_id: str,
    exclusion_list: list[str],
    remaining_cash: Optional[dict[str, float]],
    n_iterations: int,
) -> EvaluateResponse:
    """Drive the HAgent gradient-descent loop and return ranked candidates.

    Retrieves the session, clears warm-start weights so each evaluate call
    starts fresh, then advances the get_h_scores generator for the requested
    number of iterations.  The final yielded result is converted to a
    serialisable EvaluateResponse.

    Args:
        session_id:         Identifies the active session (created by POST /sessions).
        player_assignments: Maps each team name to the list of players already
                            drafted/won by that team.
        my_team_id:         The team name whose perspective the evaluation is from.
        exclusion_list:     Players to exclude from the candidate rankings
                            (e.g. already drafted by the user, injured).
        remaining_cash:     Per-team auction budget remaining; None for draft mode.
        n_iterations:       Number of gradient-descent iterations to run.

    Returns:
        EvaluateResponse containing the iteration count and ranked Candidate list.

    Raises:
        KeyError: If the session does not exist or has expired.
    """
    session = get_session(session_id)
    if session is None:
        raise KeyError(f"Session '{session_id}' not found or expired")

    info           = session.info
    H              = session.H
    categories     = session.current_params['categories']
    current_params = session.current_params

    # Clear warm-start weights so this call is independent of any previous one.
    H = H.clear_initial_weights()
    generator = H.get_h_scores(
        player_assignments      = player_assignments,
        drafter                 = my_team_id,
        cash_remaining_per_team = remaining_cash,
        exclusion_list          = exclusion_list,
    )

    h_score_result    = None
    actual_iterations = 0
    for _ in range(max(1, n_iterations)):
        h_score_result     = next(generator)
        actual_iterations += 1

    if h_score_result is None:
        return EvaluateResponse(iteration=0, candidates=[])

    # ── Generic H-scores cache (auction mode only) ────────────────────────────
    # gnrc_dollar / orig_dollar are anchored to a neutral state: no players taken.
    # On the first call (no players assigned), the current result already represents
    # that neutral state, so we cache it directly — avoiding a redundant second run
    # and guaranteeing Your $ == Gnrc. $ at the start of the auction.
    # On later calls the cached value is reused unchanged.
    # If the session connects mid-auction (players already assigned on first call),
    # run a separate clean evaluation to obtain the neutral baseline.
    if remaining_cash is not None and session.generic_h_scores is None:
        all_assigned = [
            p for team_players in player_assignments.values()
            for p in team_players if isinstance(p, str)
        ]
        if len(all_assigned) == 0:
            # No players taken yet — current scores are the neutral baseline.
            session.generic_h_scores = h_score_result['Scores'].sort_values(ascending=False)
        else:
            # Mid-auction start: run a clean evaluation with all slots empty.
            team_names        = current_params.get('team_names', [my_team_id])
            empty_assignments = {name: [] for name in team_names}
            generic_H         = H.clear_initial_weights()
            generic_gen       = generic_H.get_h_scores(
                player_assignments      = empty_assignments,
                drafter                 = my_team_id,
                cash_remaining_per_team = remaining_cash,
                exclusion_list          = [],
            )
            generic_result = None
            for _ in range(max(1, n_iterations)):
                generic_result = next(generic_gen)
            if generic_result is not None:
                session.generic_h_scores = generic_result['Scores'].sort_values(ascending=False)

    candidates = _build_candidates(
        h_score_result, info, H, categories, player_assignments, my_team_id, current_params,
        remaining_cash,
        generic_h_scores=session.generic_h_scores,
    )

    return EvaluateResponse(
        iteration  = actual_iterations,
        candidates = candidates,
    )


# ── Build candidate list ──────────────────────────────────────────────────────

def _build_candidates(
    h_score_result: dict,
    info: dict,
    H,
    categories: list[str],
    player_assignments: dict[str, list[str]],
    my_team_id: str,
    current_params: dict,
    remaining_cash: Optional[dict[str, float]] = None,
    generic_h_scores: Optional[pd.Series] = None,
) -> list[Candidate]:
    """Convert a raw HAgent result dict into a list of ranked Candidate objects.

    Extracts per-player scores, win rates, category weights, G-score breakdown
    rows, flex-slot allocations, optimal roster assignment, and (in auction mode)
    SAVOR dollar values for each player that physically fits the roster, then
    returns them sorted by H-score.

    Args:
        h_score_result: The final dict yielded by HAgent.get_h_scores().
        info:           Session info dict (Positions, G-scores, etc.).
        H:              The HAgent instance (carries v, original_v, position_structure).
        categories:     Ordered list of scoring category names.
        player_assignments: Current draft/auction state (team → player list).
        my_team_id:     The user's team identifier.
        current_params: Session parameters dict (contains slot_counts, etc.).
        remaining_cash: Per-team auction budget remaining; None in draft mode.

    Returns:
        List of Candidate objects sorted descending by H-score, one per
        eligible player that fits the current roster.
    """
    # Pull named arrays from the result dict.
    # Per-candidate result DataFrames are reindexed to h_scores_sorted order so
    # that positional indexing (.iloc[rank_idx]) is safe in the candidate loop.
    # Full-population lookup tables (player_position_map, player_g_scores) are
    # NOT reindexed — they are used to look up already-drafted players and for
    # auction value calculations across the entire player pool.
    h_scores_sorted          = h_score_result['Scores'].sort_values(ascending=False)
    sorted_index             = h_scores_sorted.index
    category_weights_raw     = h_score_result['Weights'].reindex(sorted_index) if h_score_result['Weights'] is not None else None
    win_rate_cdfs            = h_score_result['Rates'].reindex(sorted_index)
    team_diff_df             = h_score_result['Diff'].reindex(sorted_index) if h_score_result['Diff'] is not None else None
    future_diff_df           = h_score_result['Future-Diff'].reindex(sorted_index) if h_score_result['Future-Diff'] is not None else None
    player_position_map      = info['Positions']           # full-population: used by _build_roster for my_players
    player_g_scores          = info['G-scores']            # full-population: used for auction dollar values

    # res['Rosters'] column 0 encodes whether a valid slot assignment was found.
    # When position data is entirely absent the algorithm yields a single-column
    # DataFrame filled with -1 as a sentinel.  In that case we skip position
    # filtering and return None for all position-derived fields.
    # When position data IS present, any value < 0 indicates an individual
    # candidate cannot be fitted into the current roster and is excluded.
    rosters_col0 = h_score_result['Rosters'].iloc[:, 0]
    no_position_data = bool((rosters_col0 == -1).all())
    player_fits_roster = (
        pd.Series(True, index=sorted_index)
        if no_position_data
        else (rosters_col0 >= 0).reindex(sorted_index)
    )

    n_categories = len(categories)

    # H.v is the normalised weight vector (sums to 1), shaped (n_cat, 1).
    # Reshape to (n_cat,) so it can broadcast against per-player weight rows.
    v_reshaped = H.v.reshape(n_categories)
    original_v = np.array(H.original_v)  # unnormalised weights; see _build_g_score_rows

    # Normalise raw category weights relative to v so that 100 = neutral emphasis.
    # H.v is a scaled version of original_v adjusted to sum to 1 (required by the
    # H-score math).  When the algorithm places no punting emphasis on a category,
    # weights_raw ≈ v, so dividing by v and scaling to 100 gives a neutral baseline of 100.
    category_weights_normalized = None if category_weights_raw is None else \
                                  (category_weights_raw.values / v_reshaped) * 100  # (n_players, n_cat)

    my_players         = [p for p in player_assignments.get(my_team_id, []) if isinstance(p, str)]
    position_structure = H.position_structure
    base_list          = position_structure['base_list']
    slot_counts        = current_params.get('slot_counts', {})
    slot_names         = _make_slot_names(slot_counts, position_structure)

    # ── Auction dollar values (SAVOR) ─────────────────────────────────────────
    # Computed once for all players before the loop; None in draft mode.
    #
    # Three values per player:
    #   your_dollar — SAVOR on H-scores (team-specific, uses remaining cash)
    #   gnrc_dollar — SAVOR on generic H-scores (no players taken), remaining cash/picks
    #   orig_dollar — SAVOR on generic H-scores (no players taken), original full cash/picks
    #
    # n_remaining is the number of players still to be drafted across all teams.
    # The SAVOR function uses this to find the replacement-level player (the
    # n_remaining-th best available player), which anchors the dollar scale.
    player_auction_values: dict[str, AuctionValues] | None = None
    if remaining_cash is not None:
        cash_per_team   = current_params.get('cash_per_team')
        streaming_noise = float(current_params.get('streaming_noise', 10.0))
        total_picks     = H.n_drafters * H.n_picks

        all_players_chosen = [
            p for team_players in player_assignments.values()
            for p in team_players if isinstance(p, str)
        ]
        n_remaining          = total_picks - len(all_players_chosen)
        total_cash_remaining = float(sum(remaining_cash.values()))

        if cash_per_team is not None and n_remaining > 0:
            total_original_cash = float(H.n_drafters * cash_per_team)

            # G-scores for available (undrafted) players, used for G-score dollar values.
            available_in_g = [p for p in h_scores_sorted.index if p in player_g_scores.index]
            g_scores_available = player_g_scores.loc[available_in_g, 'Total']

            # Baseline scores for gnrc/orig: generic run (no players taken) if cached,
            # otherwise fall back to the current team-specific scores.
            baseline_scores = generic_h_scores if generic_h_scores is not None else h_scores_sorted

            try:
                # your_dollar: team-specific H-scores, current state (remaining cash + picks)
                your_dollar_series = auction_value_adjuster(
                    h_scores_sorted, n_remaining, total_cash_remaining, streaming_noise,
                )
                # gnrc_dollar: neutral baseline H-scores, current cash/picks remaining
                gnrc_dollar_series = auction_value_adjuster(
                    baseline_scores, n_remaining, total_cash_remaining, streaming_noise,
                )
                # orig_dollar: neutral baseline H-scores, full original cash/picks
                orig_dollar_series = auction_value_adjuster(
                    baseline_scores, total_picks, total_original_cash, streaming_noise,
                )
                # G-score variants: generic value using G-scores instead of H-scores.
                gnrc_dollar_g_series = auction_value_adjuster(
                    g_scores_available, n_remaining, total_cash_remaining, streaming_noise,
                )
                orig_dollar_g_series = auction_value_adjuster(
                    player_g_scores['Total'], total_picks, total_original_cash, streaming_noise,
                )
                player_auction_values = {
                    p: AuctionValues(
                        your_dollar   = round(float(your_dollar_series.get(p, 0.0)), 2),
                        gnrc_dollar   = round(float(gnrc_dollar_series.get(p, 0.0)), 2),
                        orig_dollar   = round(float(orig_dollar_series.get(p, 0.0)), 2),
                        gnrc_dollar_g = round(float(gnrc_dollar_g_series.get(p, 0.0)), 2),
                        orig_dollar_g = round(float(orig_dollar_g_series.get(p, 0.0)), 2),
                    )
                    for p in h_scores_sorted.index
                }
            except Exception:
                player_auction_values = None  # degrade gracefully on edge cases

    candidates: list[Candidate] = []
    for rank_idx, player in enumerate(sorted_index):
        if not player_fits_roster.iloc[rank_idx]:
            continue  # skip players for whom no valid roster slot exists

        h_score                 = float(h_scores_sorted.iloc[rank_idx]) * 100
        player_win_rates        = win_rate_cdfs.iloc[rank_idx].values * 100
        player_category_weights = None if category_weights_normalized is None else category_weights_normalized[rank_idx]

        g_score_rows = _build_g_score_rows(
            player, categories, player_g_scores,
            team_diff_df, future_diff_df, original_v,
        )

        flex_allocations = (
            None if no_position_data
            else _build_flex_allocations(
                player, base_list, position_structure,
                h_score_result['Position-Shares'], slot_counts,
            )
        )

        roster = (
            None if no_position_data
            else _roster_from_precomputed(
                player, my_players,
                h_score_result['Rosters'].loc[player].values,
                slot_names,
            )
        )

        raw_positions    = player_position_map.get(player, ['?'])
        position_display = ','.join(raw_positions) if isinstance(raw_positions, list) else str(raw_positions)

        candidates.append(Candidate(
            name             = player,
            position         = position_display,
            h_score          = round(h_score, 2),
            h_rank           = rank_idx + 1,
            win_rates        = [round(float(val), 2) for val in player_win_rates],
            category_weights = None if player_category_weights is None else [round(float(val), 1) for val in player_category_weights],
            g_score_rows     = g_score_rows,
            flex_allocations = flex_allocations,
            roster           = roster,
            auction_values   = player_auction_values.get(player) if player_auction_values else None,
        ))

    return candidates


# ── G-score rows ──────────────────────────────────────────────────────────────

def _build_g_score_rows(
    player: str,
    categories: list[str],
    player_g_scores: pd.DataFrame,
    team_diff_df: pd.DataFrame | None,
    future_diff_df: pd.DataFrame | None,
    original_v: np.ndarray,
) -> list[GScoreRow]:
    """Build the G-score breakdown table rows for one candidate player.

    Mirrors make_main_df_styled() in src/tabs/candidate_subtabs.py.

    The table always ends with a 'Total diff' row = current team diff + this
    player's own G-scores.  Depending on whether a future component was
    computed, the layout is either 3 rows (no future) or 4 rows (with future).

    Why the diff decomposition works
    ---------------------------------
    res['Diff'] is stored as:
        expected_future_diff.mean(axis=2) + diff_means.mean(axis=2)
    where diff_means is the current team's x-score differential versus all
    opponents simultaneously (axis=2 of diff_means is the opponent dimension;
    the mean collapses it to a single average-across-opponents value).
    diff_means is the same for every candidate since it depends only on the
    players already drafted.  expected_future_diff is the algorithm's
    projection for the picks remaining, which does vary per candidate.

    Therefore:
        current_diff = res['Diff'] - res['Future-Diff']
                     = (future_diff + diff_means) - future_diff
                     = diff_means        ← same value for every candidate

    Both diff DataFrames store values in x-score space (divided by v during
    computation).  Multiplying by original_v converts them back to G-score
    units so they are comparable to the player_g_scores values.  original_v
    is the exact conversion factor between x-score and G-score space
    (original_v = sqrt(mov/vom) for Rotisserie, sqrt(mov/(mov+vom)) for H2H).
    The algorithm's v is original_v rescaled to sum to 1; original_v is used
    here instead because G-score display is in the original units.

    Note: res['Diff'] does NOT include the candidate player's own G-scores;
    player_g_scores must be added explicitly to form the Total row.

    Args:
        player:          Candidate player name.
        categories:      Ordered list of scoring category names.
        player_g_scores: DataFrame of per-player, per-category G-scores.
        team_diff_df:    res['Diff'] — total expected team x-score differential
                         (future + current), per player row, per category column.
                         None when the dynamic optimiser did not run (e.g., last pick).
        future_diff_df:  res['Future-Diff'] — expected future component only.
                         None when no future picks remain or dynamic=False.
        original_v:      Unnormalised category weight vector; used to convert
                         x-score diff values to G-score units.

    Returns:
        List of GScoreRow objects: either 3 rows (no future) or 4 rows (with future).
    """
    n_categories = len(categories)

    # Retrieve this player's pre-computed G-scores; fall back to zeros if absent.
    player_own_g_scores = (
        player_g_scores.loc[player, categories].values.astype(float)
        if player in player_g_scores.index
        else np.zeros(n_categories)
    )

    def _make_row(label: str, values: np.ndarray, is_total: bool = False) -> GScoreRow:
        """Package a label + numeric array into a GScoreRow, rounding to 2dp."""
        rounded_values = [round(float(x), 2) for x in values]
        return GScoreRow(
            label    = label,
            values   = rounded_values,
            total    = round(float(sum(values)), 2),
            is_total = is_total,
        )

    # Fallback when the diff wasn't computed (last pick or dynamic=False):
    # current team diff is unknown so show zeros; Total = player's own G-scores.
    # Use only the last name to keep the table compact.
    player_last_name = _last_name(player)

    if team_diff_df is None or team_diff_df.empty or player not in team_diff_df.index:
        return [
            _make_row('Current diff',   np.zeros(n_categories)),
            _make_row(player_last_name, player_own_g_scores),
            _make_row('Total diff',     player_own_g_scores, is_total=True),
        ]

    # Convert x-score diff to G-score units by multiplying by original_v.
    # team_diff_df stores (future_diff + current_diff) for this player's row.
    total_diff = team_diff_df.loc[player, categories].values.astype(float) * original_v

    # Total diff always equals team differential + candidate's own contribution.
    # This is the single source of truth for the Total row regardless of whether
    # the future component is present.
    total_diff_with_player = total_diff + player_own_g_scores

    if future_diff_df is None or player not in future_diff_df.index:
        # 3-row layout: dynamic optimiser did not separate current from future.
        # 'Current diff' here is actually total_diff (= diff_means * original_v),
        # which is the same across all candidates.
        return [
            _make_row('Current diff',   total_diff),
            _make_row(player_last_name, player_own_g_scores),
            _make_row('Total diff',     total_diff_with_player, is_total=True),
        ]

    # 4-row layout: separate out the future component so the user can see how
    # much of the expected advantage is 'already there' vs. 'expected from
    # future picks given this player is on the team'.
    future_diff  = future_diff_df.loc[player, categories].values.astype(float) * original_v
    current_diff = total_diff - future_diff  # = diff_means * original_v; same for all candidates

    return [
        _make_row('Current diff',   current_diff),
        _make_row(player_last_name, player_own_g_scores),
        _make_row('Future diff',    future_diff),
        _make_row('Total diff',     total_diff_with_player, is_total=True),
    ]


# ── Flex allocations ──────────────────────────────────────────────────────────

def _build_flex_allocations(
    player: str,
    base_list: list[str],
    position_structure: dict,
    position_shares: dict,
    slot_counts: dict,
) -> FlexAllocations:
    """Build the flex-slot usage table for one candidate player.

    For each flex position type (e.g. 'UTIL') that has at least one slot, one
    row is created showing how many of those slots the candidate is expected to
    occupy in each base-position role (e.g. PG share of UTIL slots).  A Total
    row sums across all flex types.

    Position shares are fractions (0–1) representing the probability the player
    occupies a given base-position role within a flex slot.  Multiplying by the
    slot count gives an expected number of slots used.

    Args:
        player:             Candidate player name.
        base_list:          Ordered list of base position codes (e.g. ['PG','SG','SF','PF','C']).
        position_structure: Dict with 'flex_list' and 'flex' keys describing which
                            base positions each flex type can accommodate.
        position_shares:    Dict mapping flex type → DataFrame of per-player base-position
                            share fractions (output of the position share optimiser).
        slot_counts:        Dict mapping position code → number of roster slots.

    Returns:
        FlexAllocations with one row per active flex type plus a Total row.
    """
    flex_position_types   = position_structure['flex_list']
    flex_position_details = position_structure['flex']

    flex_rows: list[FlexRow]        = []
    totals_by_base: dict[str, float] = {base_pos: 0.0 for base_pos in base_list}

    for flex_type in flex_position_types:
        n_flex_slots = slot_counts.get(flex_type, 0)
        if n_flex_slots == 0:
            continue  # this flex type has no slots in the current league settings

        eligible_bases   = flex_position_details[flex_type]['bases']
        position_share_df = position_shares.get(flex_type)

        if position_share_df is None or player not in position_share_df.index:
            # No share data for this player / flex type; mark every column as ineligible.
            base_position_values: list[float | None] = [None] * len(base_list)
        else:
            base_position_values = []
            for base_pos in base_list:
                if base_pos in eligible_bases:
                    # Convert share fraction to expected slot usage across all flex slots.
                    expected_slot_usage = float(position_share_df.loc[player, base_pos]) * n_flex_slots
                    base_position_values.append(round(expected_slot_usage, 2))
                    totals_by_base[base_pos] += expected_slot_usage
                else:
                    # This base position is ineligible for this flex type.
                    base_position_values.append(None)

        flex_rows.append(FlexRow(
            label    = f"{flex_type}-{n_flex_slots}",
            values   = base_position_values,
            is_total = False,
        ))

    # Append a Total row summing expected slot usage across all flex types.
    total_row_values = [round(totals_by_base[base_pos], 2) for base_pos in base_list]
    flex_rows.append(FlexRow(label='Total', values=total_row_values, is_total=True))

    return FlexAllocations(base_positions=base_list, rows=flex_rows)


# ── Roster assignment ─────────────────────────────────────────────────────────

def _roster_from_precomputed(
    candidate: str,
    my_players: list[str],
    rosters_row: np.ndarray,
    slot_names: list[str],
) -> Roster:
    """Build Roster display from the pre-computed slot assignments in h_score_result['Rosters'].

    rosters_row[j] is the slot index assigned to player j in the ordering
    [team_so_far[0], ..., team_so_far[-1], candidate, future_player[0], ...].
    Columns beyond len(my_players) + 1 are future hypothetical players and are ignored.
    """
    n_team_so_far = len(my_players)
    assignments: dict[str, RosterAssignment | None] = {slot: None for slot in slot_names}

    for i, player_name in enumerate(my_players):
        slot_idx = int(rosters_row[i])
        if 0 <= slot_idx < len(slot_names):
            assignments[slot_names[slot_idx]] = RosterAssignment(
                name=_last_name(player_name),
                is_candidate=False,
            )

    if n_team_so_far < len(rosters_row):
        candidate_slot_idx = int(rosters_row[n_team_so_far])
        if 0 <= candidate_slot_idx < len(slot_names):
            assignments[slot_names[candidate_slot_idx]] = RosterAssignment(
                name=_last_name(candidate),
                is_candidate=True,
            )

    return Roster(slots=slot_names, assignments=assignments)


def _make_slot_names(slot_counts: dict, position_structure: dict) -> list[str]:
    """Return slot IDs in canonical order: base positions then flex.

    Each position type contributes as many slot IDs as it has slots, numbered
    from 1.  For example, two PG slots and one UTIL slot yield
    ['PG1', 'PG2', 'UTIL1'].

    Args:
        slot_counts:        Dict mapping position code → number of roster slots.
        position_structure: Dict with 'base_list' and 'flex_list' for ordering.

    Returns:
        Flat list of slot ID strings in canonical order.
    """
    position_order = position_structure['base_list'] + position_structure['flex_list']
    slot_list: list[str] = []
    for position_type in position_order:
        slot_count = slot_counts.get(position_type, 0)
        for i in range(1, slot_count + 1):
            slot_list.append(f"{position_type}{i}")
    return slot_list



