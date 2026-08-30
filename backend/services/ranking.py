"""
Runs HAgent.get_h_scores() for a given draft state and converts the result
to the /evaluate response payload.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from backend.state.session import Session
from backend.models import (
    Candidate, GScoreRow, FlexAllocations, FlexRow,
    Roster, RosterAssignment, AuctionValues, EvaluateResponse,
)
from backend.math.algorithm_helpers import auction_value_adjuster
from backend.player_identity import FULL_ROSTER_SCORE_PLAYER_ID
from backend.infra.server_timing import record_phase

# The engine's internal index for the one result row of a full-roster evaluate
# (algorithm_agents.get_h_scores, n_players_selected == n_picks branch).
_FULL_ROSTER_RESULT_INDEX = ''


class UnknownRosterPlayersError(ValueError):
    """A rostered player is not in the current player pool.

    Happens when a data-source change alters player identities (or removes players)
    after a board was built against the previous pool. Surfaced as a 400 so the
    user gets an actionable message instead of a KeyError-turned-500.
    """


# ── Public entry point ────────────────────────────────────────────────────────

def rank_candidates(
    session: Session,
    player_assignments: dict[str, list[int]],
    my_team_id: str,
    exclusion_list: list[int],
    remaining_cash: Optional[dict[str, float]],
    candidate_offset: int = 0,
    candidate_limit: Optional[int] = None,
) -> EvaluateResponse:
    """Drive the HAgent gradient-descent loop and return ranked candidates.

    Runs the agent's get_h_scores solve for the requested board and converts the
    result into a serialisable EvaluateResponse. Warm-start state is managed by
    the agent itself, primed from the neutral baseline built at session creation.

    Args:
        session:            The active session (fetched by the caller). n_iterations
                            is read from its current_params.
        player_assignments: Maps each team name to the list of player ids already
                            drafted/won by that team.
        my_team_id:         The team name whose perspective the evaluation is from.
        exclusion_list:     Player ids to exclude from the candidate rankings
                            (e.g. already drafted by the user, injured).
        remaining_cash:     Per-team auction budget remaining; None for draft mode.

    Returns:
        EvaluateResponse containing the iteration count and ranked Candidate list.
    """
    info           = session.agent.info
    h_agent              = session.agent
    current_params = session.current_params
    categories     = current_params['categories']
    n_iterations   = current_params['n_iterations']
    player_registry = session.player_registry

    # Every rostered player must exist in the pool under exactly the identity the board
    # holds. A stale board (identities changed by a data-source switch) would otherwise
    # crash on a KeyError deep inside the H-score math. Unknown ids resolve to names via
    # the registry where possible, so the message stays actionable.
    rostered_players = [
        player_id for team_players in player_assignments.values()
        for player_id in team_players
    ]
    missing_players = sorted({p for p in rostered_players if p not in h_agent.x_scores.index})
    if missing_players:
        missing_display = [
            player_registry[p].name if p in player_registry else str(p)
            for p in missing_players
        ]
        raise UnknownRosterPlayersError(
            'These rostered players are not in the current player pool: '
            + ', '.join(missing_display)
            + '. The data-source change altered the pool; clear the board or restore the previous sources.'
        )

    # ── Candidate batching (draft/waiver only) ────────────────────────────────
    # Slice the available pool by the agent's default (neutral-board) ranking so the top-ranked players
    # are scored (and can paint) first. Auction always evaluates the whole pool — its dollar values anchor
    # on the full-distribution replacement level.
    is_auction = remaining_cash is not None
    if candidate_limit is not None and not is_auction:
        unavailable = {p for team in player_assignments.values() for p in team}
        unavailable |= set(exclusion_list)
        available_ranked = [p for p in session.agent.default_h_scores.index if p not in unavailable]
        candidate_subset = available_ranked[candidate_offset : candidate_offset + candidate_limit]
        has_more         = len(available_ranked) > candidate_offset + candidate_limit
        total_candidates = len(available_ranked)   # falls back to the scored count below when not batching
        if len(candidate_subset) == 0:
            return EvaluateResponse(iteration=0, candidates=[], has_more=False,
                                    total_candidates=total_candidates)
    else:
        candidate_subset = None
        has_more         = False
        total_candidates = None

    with record_phase('hscores'):
        h_score_result = h_agent.get_h_scores(
            player_assignments      = player_assignments,
            drafter                 = my_team_id,
            n_iterations            = n_iterations,
            cash_remaining_per_team = remaining_cash,
            exclusion_list          = exclusion_list,
            candidate_subset        = candidate_subset,
            # Global rank of this batch's first candidate, so the throttle's exact-solve tiers stay
            # global: batches past the first fall outside them and exact-solve nobody. Zero when we
            # score the whole pool (no subset), where the tiers apply normally from the top.
            candidate_offset        = candidate_offset if candidate_subset is not None else 0,
        )
    actual_iterations = max(1, n_iterations)

    if h_score_result is None:
        return EvaluateResponse(iteration=0, candidates=[])

    with record_phase('build_candidates'):
        candidates = _build_candidates(
            h_score_result, info, h_agent, categories, player_assignments, my_team_id, current_params,
            player_registry,
            remaining_cash,
            generic_h_scores=session.agent.default_h_scores,
        )

    return EvaluateResponse(
        iteration        = actual_iterations,
        candidates       = candidates,
        has_more         = has_more,
        total_candidates = total_candidates if total_candidates is not None else len(candidates),
    )


# ── Build candidate list ──────────────────────────────────────────────────────

def _build_candidates(
    h_score_result: dict,
    info: dict,
    h_agent,
    categories: list[str],
    player_assignments: dict[str, list[int]],
    my_team_id: str,
    current_params: dict,
    player_registry: dict,
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
        h_agent:              The HAgent instance (carries v, original_v, position_structure).
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
    scores_series            = h_score_result['Scores']

    # A full roster has nothing to rank: the engine scores the finished team once, under
    # its internal sentinel index. Surface that as a single team-score row — it is not a
    # player (no registry entry), and clients read only its h_score / win_rates.
    if list(scores_series.index) == [_FULL_ROSTER_RESULT_INDEX]:
        team_win_rates = h_score_result['Rates'].iloc[0]
        return [Candidate(
            player_id    = FULL_ROSTER_SCORE_PLAYER_ID,
            h_score      = round(float(scores_series.iloc[0]) * 100, 2),
            h_rank       = 1,
            win_rates    = [round(float(rate) * 100, 2) for rate in team_win_rates.values],
            g_score_rows = [],
        )]

    h_scores_sorted          = scores_series.sort_values(ascending=False)
    sorted_index             = h_scores_sorted.index
    # Every frame in h_score_result carries the same candidate index in the same order, so resolve the
    # sort to integer positions once and reorder each frame positionally with .iloc[order] — a single
    # hash pass instead of a separate hash-aligned .reindex() per frame. (player_g_scores below is the
    # exception: it carries the full-population index and needs a genuine cross-index reindex.)
    order                    = scores_series.index.get_indexer(sorted_index)
    category_weights_raw     = h_score_result['Weights'].iloc[order] if h_score_result['Weights'] is not None else None
    win_rate_cdfs            = h_score_result['Rates'].iloc[order]
    team_diff_df             = h_score_result['Diff'].iloc[order] if h_score_result['Diff'] is not None else None
    future_diff_df           = h_score_result['Future-Diff'].iloc[order] if h_score_result['Future-Diff'] is not None else None
    opponent_tilt_df         = (h_score_result.get('Opponent-Future-Tilt').iloc[order]
                                if h_score_result.get('Opponent-Future-Tilt') is not None else None)
    player_g_scores          = info['G-scores']            # full-population: used for auction dollar values

    # res['Rosters'] column 0 encodes whether a valid slot assignment was found.
    # When position data is entirely absent the algorithm yields a single-column
    # DataFrame filled with -1 as a sentinel.  In that case we skip position
    # filtering and return None for all position-derived fields.
    # When position data IS present, any value < 0 indicates an individual
    # candidate cannot be fitted into the current roster and is excluded.
    rosters_sorted = h_score_result['Rosters'].iloc[order]
    rosters_col0 = rosters_sorted.iloc[:, 0]
    has_position_data = not bool((rosters_col0 == -1).all())
    player_fits_roster = (
        (rosters_col0 >= 0)                                 # already in sorted order
        if has_position_data
        else pd.Series(True, index=sorted_index)
    )

    n_categories = len(categories)

    # h_agent.v is the normalised weight vector (sums to 1), shaped (n_cat, 1).
    # Reshape to (n_cat,) so it can broadcast against per-player weight rows.
    v_reshaped = h_agent.v.reshape(n_categories)
    original_v = np.array(h_agent.original_v)  # unnormalised weights; see _build_g_score_rows

    # Normalise raw category weights relative to v so that 100 = neutral emphasis.
    # h_agent.v is a scaled version of original_v adjusted to sum to 1 (required by the
    # H-score math).  When the algorithm places no punting emphasis on a category,
    # weights_raw ≈ v, so dividing by v and scaling to 100 gives a neutral baseline of 100.
    category_weights_normalized = None if category_weights_raw is None else \
                                  (category_weights_raw.values / v_reshaped) * 100  # (n_players, n_cat)

    # Vectorise the per-candidate scaling + rounding the loop used to do element-by-element: h-scores
    # and win-rates become percentages (× 100) rounded to 2 dp; weights are already × 100, round to 1.
    h_scores_scaled  = np.round(h_scores_sorted.values * 100, 2)          # (n_players,)
    win_rates_scaled = np.round(win_rate_cdfs.values   * 100, 2)          # (n_players, n_cat)
    category_weights_scaled = (
        None if category_weights_normalized is None else np.round(category_weights_normalized, 1)
    )

    my_players         = list(player_assignments.get(my_team_id, []))
    position_structure = h_agent.position_structure
    base_list          = position_structure['base_list']
    slot_counts        = current_params['slot_counts']
    slot_names         = _make_slot_names(slot_counts, position_structure)

    # ── Auction dollar values (SAVOR) ─────────────────────────────────────────
    # Computed once for all players before the loop; None in draft mode.
    #
    # Three values per player:
    #   your_dollar — SAVOR on H-scores (team-specific, uses remaining cash)
    #   original_dollar — SAVOR on generic H-scores (no players taken), original full cash/picks
    #   generic_dollar — original_dollar restricted to the AVAILABLE players and rescaled so it sums to the
    #                 current remaining cash. Not an independent SAVOR run: re-running SAVOR on the
    #                 baseline mid-auction lets already-drafted players absorb part of the remaining
    #                 cash, so the available players' values no longer exhaust the budget (and Diff
    #                 drifts systematically positive). Every dollar column must sum to its budget:
    #                 your$/gnrc$ to the remaining cash over available players, orig$ to the original
    #                 cash over the full pool.
    #
    # n_remaining is the number of players still to be drafted across all teams.
    # The SAVOR function uses this to find the replacement-level player (the
    # n_remaining-th best available player), which anchors the dollar scale.
    player_auction_values: dict[str, AuctionValues] | None = None
    if remaining_cash is not None:
        # The evaluate route enforces that remaining_cash and cash_per_team are set together, so an
        # auction evaluate always has cash_per_team here (no defensive fallback needed).
        cash_per_team   = current_params['cash_per_team']
        streaming_noise = float(current_params.get('streaming_noise', 10.0)) #ZR: What is this dumb fallback? What does claude.md say about this?
        total_picks     = h_agent.n_drafters * h_agent.n_picks

        all_players_chosen = [
            p for team_players in player_assignments.values()
            for p in team_players
        ]
        n_remaining          = total_picks - len(all_players_chosen)

        #ZR: Why are these float calls needed? How would these be anything but floats or ints, which would also be fine I think?
        total_cash_remaining = float(sum(remaining_cash.values()))

        if n_remaining > 0:
            total_original_cash = float(h_agent.n_drafters * cash_per_team)

            # G-scores for available (undrafted) players, used for G-score dollar values.
            available_in_g = [p for p in h_scores_sorted.index if p in player_g_scores.index]
            g_scores_available = player_g_scores.loc[available_in_g, 'Total']

            # Baseline scores for gnrc/orig: generic run (no players taken) if cached,
            # otherwise fall back to the current team-specific scores.
            baseline_scores = generic_h_scores if generic_h_scores is not None else h_scores_sorted

            # No defensive guard here: build_agent rejects any league whose pool can't fill every
            # roster, so the replacement-level index inside auction_value_adjuster always resolves.
            # your_dollar: team-specific H-scores, current state (remaining cash + picks)
            your_dollar_series = auction_value_adjuster(
                h_scores_sorted, n_remaining, total_cash_remaining, streaming_noise,
            )
            # original_dollar: neutral baseline H-scores, full original cash/picks
            original_dollar_series = auction_value_adjuster(
                baseline_scores, total_picks, total_original_cash, streaming_noise,
            )
            # generic_dollar: original_dollar over the available players, rescaled to the remaining cash
            # (see the column definitions above — the sum over available players must equal the
            # remaining cash exactly, like your_dollar).
            orig_available     = original_dollar_series.reindex(h_scores_sorted.index).fillna(0.0)
            generic_dollar_series = orig_available * (total_cash_remaining / orig_available.sum())
            # G-score variants, same construction from the G-score original values.
            original_dollar_g_score_series = auction_value_adjuster(
                player_g_scores['Total'], total_picks, total_original_cash, streaming_noise,
            )
            orig_g_available     = original_dollar_g_score_series.reindex(g_scores_available.index).fillna(0.0)
            generic_dollar_g_score_series = orig_g_available * (total_cash_remaining / orig_g_available.sum())
            player_auction_values = {
                p: AuctionValues(
                    your_dollar   = round(float(your_dollar_series.get(p, 0.0)), 2),
                    generic_dollar   = round(float(generic_dollar_series.get(p, 0.0)), 2),
                    original_dollar   = round(float(original_dollar_series.get(p, 0.0)), 2),
                    generic_dollar_g_score = round(float(generic_dollar_g_score_series.get(p, 0.0)), 2),
                    original_dollar_g_score = round(float(original_dollar_g_score_series.get(p, 0.0)), 2),
                )
                for p in h_scores_sorted.index
            }

    # ── Vectorised expand-view precompute ─────────────────────────────────────
    # Every arithmetic input to the three expand-view tables (G-score rows, flex
    # allocations, roster assignments) is computed across ALL candidates at once with
    # numpy below, then the batch builders only assemble Pydantic models — no per-player
    # arithmetic, rounding, or label lookup remains inside any loop.

    # Per-candidate own G-scores, sorted order, shape (n_players, n_cat).
    g_scores_for_candidates = (
        player_g_scores.reindex(sorted_index)[categories].fillna(0.0).values.astype(float)
    )
    # Team / future differential matrices in x-score space, or None when the dynamic
    # optimiser did not produce them (last pick, dynamic=False). None selects the
    # zero-diff G-score layout, exactly as the old per-row None sentinel did.
    team_diff_matrix = (
        team_diff_df[categories].values
        if team_diff_df is not None and not team_diff_df.empty
        else None
    )
    future_diff_matrix = (
        future_diff_df[categories].values
        if future_diff_df is not None
        else None
    )
    # The opponents' expected future tilts (x-score space), subtracted from the future row at
    # display time — res['Future-Diff'] itself stays raw because the opponent model feeds on it.
    opponent_tilt_matrix = (
        opponent_tilt_df[categories].values
        if opponent_tilt_df is not None and future_diff_matrix is not None
        else None
    )
    rosters_rows = rosters_sorted.values

    # Candidate last names, read once for the whole pool from the registry (they label each
    # candidate's own G-score row).
    candidate_last_names = [player_registry[p].last_name for p in sorted_index]

    # Pre-extract position share arrays for each flex type, aligned to sorted_index.
    # Each entry is (numpy_array, base_to_col) where base_to_col maps base position
    # name → column index in the array, allowing direct positional lookup per player.
    position_shares_arrays = (
        {
            flex_type: (
                share_df.iloc[order].values,
                {base: i for i, base in enumerate(share_df.columns)},
            ) if share_df is not None else None
            for flex_type, share_df in h_score_result['Position-Shares'].items()
        }
        if has_position_data else None
    )

    n_players = len(sorted_index)

    # Batch-build the three expand-view tables for every candidate rank at once.
    g_score_rows_by_rank = _build_g_score_rows(
        candidate_last_names
        , g_scores_for_candidates
        , team_diff_matrix
        , future_diff_matrix
        , original_v
        , opponent_tilt_matrix
    )
    # How many of each flex type's slots are still open for future picks, per candidate. The shares
    # describe how those *remaining* slots would be filled, so the display multiplies by this — not by
    # the full slot count — otherwise the table sums to the league total even when the drafter has
    # already filled flex spots with real players. Read straight from the roster slot assignments.
    remaining_flex_by_rank = (
        _remaining_flex_slots(rosters_rows, len(my_players), slot_counts, position_structure)
        if has_position_data else {}
    )
    flex_allocations_by_rank = (
        _build_flex_allocations(
            n_players
            , base_list
            , position_structure
            , position_shares_arrays
            , slot_counts
            , remaining_flex_by_rank
        )
        if has_position_data else None
    )
    roster_by_rank = (
        _build_roster_assignments(
            list(sorted_index)
            , my_players
            , rosters_rows
            , slot_names
        )
        if has_position_data else None
    )

    # Everything that doesn't need the expand-view builders is precomputed here in
    # sorted order and indexed by rank in the loop. Whole-array .tolist() crosses the
    # numpy→Python boundary once instead of once per row; the position display (a dict
    # lookup + string join, which is not numpy-vectorisable) is built out of the loop too.
    h_scores_list          = h_scores_scaled.tolist()
    win_rates_lists        = win_rates_scaled.tolist()
    category_weights_lists = None if category_weights_scaled is None else category_weights_scaled.tolist()

    # The remaining loop constructs one Candidate per fitting player, indexing into the
    # pre-built tables — no arithmetic left. rank_idx counts every sorted player (including
    # skipped non-fitters) so h_rank stays the player's true H-score rank.
    candidates: list[Candidate] = []
    for rank_idx, (player, fits) in enumerate(zip(sorted_index, player_fits_roster.values)):
        if not fits:
            continue  # skip players for whom no valid roster slot exists

        candidates.append(Candidate(
            player_id        = int(player),
            h_score          = h_scores_list[rank_idx],   # already × 100 and rounded to 2 dp
            h_rank           = rank_idx + 1,
            win_rates        = win_rates_lists[rank_idx],   # already × 100 and rounded to 2 dp
            category_weights = None if category_weights_lists is None else category_weights_lists[rank_idx],
            g_score_rows     = g_score_rows_by_rank[rank_idx],
            flex_allocations = None if flex_allocations_by_rank is None else flex_allocations_by_rank[rank_idx],
            roster           = None if roster_by_rank is None else roster_by_rank[rank_idx],
            auction_values   = player_auction_values.get(player) if player_auction_values else None,
        ))

    return candidates


# ── G-score rows ──────────────────────────────────────────────────────────────

def _build_g_score_rows(
    candidate_last_names: list[str]
    , own_g_scores: np.ndarray
    , team_diff_matrix: np.ndarray | None
    , future_diff_matrix: np.ndarray | None
    , original_v: np.ndarray
    , opponent_tilt_matrix: np.ndarray | None = None
) -> list[list[GScoreRow]]:
    """Build the G-score breakdown table rows for every candidate at once.

    Mirrors make_main_df_styled() in src/tabs/candidate_subtabs.py.

    The table always ends with a 'Total diff' row = current team diff + this
    player's own G-scores.  Depending on whether a future component was
    computed, the layout is either 3 rows (no future) or 4 rows (with future).
    The layout is uniform across candidates — team_diff_matrix / future_diff_matrix
    are present-or-absent for the whole result, never per player — so it is chosen
    once and applied to every rank via vectorised numpy arithmetic.

    Why the diff decomposition works
    ---------------------------------
    res['Diff'] is stored as:
        expected_future_diff.mean(axis=2) + diff_means.mean(axis=2)
    where diff_means is the current team's x-score differential versus all
    opponents simultaneously (axis=2 of diff_means is the opponent dimension;
    the mean collapses it to a single average-across-opponents value).
    expected_future_diff is the algorithm's projection for the picks
    remaining, which varies per candidate.

    res['Future-Diff'] is the drafter's own RAW projected future tilt — the
    opponent model feeds on it, so it is not netted at the source. The
    opponents' expected future tilts arrive separately (opponent_tilt_matrix,
    from res['Opponent-Future-Tilt']; diff_means carries them with opposite
    sign) and are subtracted HERE to form the displayed Future row:

        Future diff  = future_diff - opponent tilt   (net future differential)
        Current diff = Total diff  - Future diff
                     = diff_means + mean opponent future tilt
    i.e. the board as it stands, with every seat's expected FUTURE behaviour
    — the drafter's and the opponents' alike — attributed to the Future row.

    With the opponent model OFF the tilts are zero and diff_means depends
    only on the players already drafted, so Current diff is identical for
    every candidate. With the model ON it is candidate-dependent for the
    predicted-anchor candidates: self-exclusion swaps the candidate's own
    predicted team out of its field for the spare anchor's team (you never
    play against your own team), so a stronger anchor shows a less negative
    Current diff, while every non-anchor candidate shares one identical field.

    Both diff DataFrames store values in x-score space (divided by v during
    computation).  Multiplying by original_v converts them back to G-score
    units so they are comparable to the player_g_scores values.  original_v
    is the exact conversion factor between x-score and G-score space
    (original_v = sqrt(mov/vom) for Rotisserie, sqrt(mov/(mov+vom)) for H2H).
    The algorithm's v is original_v rescaled to sum to 1; original_v is used
    here instead because G-score display is in the original units.

    Note: res['Diff'] does NOT include the candidate player's own G-scores;
    own_g_scores must be added explicitly to form the Total row.

    Args:
        candidate_last_names: Per-rank last name used for the player's own row label.
        categories:           Ordered list of scoring category names.
        own_g_scores:         G-score matrix for all candidates, shape (n_players, n_cat).
                              Zeros where a player was absent from the G-scores table.
        team_diff_matrix:     res['Diff'] matrix, shape (n_players, n_cat), or None when
                              the dynamic optimiser did not run (e.g., last pick).
        future_diff_matrix:   res['Future-Diff'] matrix, shape (n_players, n_cat), or None
                              when no future picks remain or dynamic=False.
        original_v:           Unnormalised category weight vector; used to convert
                              x-score diff values to G-score units.

    Returns:
        One list of GScoreRow objects per candidate rank: either 3 rows (no future)
        or 4 rows (with future).
    """
    n_players, n_categories = own_g_scores.shape

    # Player's own G-score row, rounded once for the whole pool.
    own_values = np.round(own_g_scores, 2).tolist()
    own_totals = np.round(own_g_scores.sum(axis=1), 2).tolist()

    if team_diff_matrix is None:
        # Fallback when the diff wasn't computed (last pick or dynamic=False):
        # current team diff is unknown so show zeros; Total = player's own G-scores.
        zero_values = [0.0] * n_categories
        return [
            [
                GScoreRow(label='Current diff', values=zero_values, total=0.0, is_total=False),
                GScoreRow(label=candidate_last_names[i], values=own_values[i], total=own_totals[i], is_total=False),
                GScoreRow(label='Total diff', values=own_values[i], total=own_totals[i], is_total=True),
            ]
            for i in range(n_players)
        ]

    # Convert x-score diff to G-score units by multiplying by original_v (broadcast
    # over all rows). team_diff_matrix stores (future_diff + current_diff) per player.
    total_diff = team_diff_matrix.astype(float) * original_v
    # Total diff always equals team differential + candidate's own contribution — the
    # single source of truth for the Total row regardless of the future component.
    total_diff_with_player = total_diff + own_g_scores
    total_values = np.round(total_diff_with_player, 2).tolist()
    total_totals = np.round(total_diff_with_player.sum(axis=1), 2).tolist()

    if future_diff_matrix is None:
        # 3-row layout: dynamic optimiser did not separate current from future.
        # 'Current diff' here is actually total_diff (= diff_means * original_v),
        # which is the same across all candidates.
        current_values = np.round(total_diff, 2).tolist()
        current_totals = np.round(total_diff.sum(axis=1), 2).tolist()
        return [
            [
                GScoreRow(label='Current diff', values=current_values[i], total=current_totals[i], is_total=False),
                GScoreRow(label=candidate_last_names[i], values=own_values[i], total=own_totals[i], is_total=False),
                GScoreRow(label='Total diff', values=total_values[i], total=total_totals[i], is_total=True),
            ]
            for i in range(n_players)
        ]

    # 4-row layout: separate out the future component so the user can see how much of
    # the expected advantage is 'already there' vs. 'expected from future picks given
    # this player is on the team'.
    future_diff = future_diff_matrix.astype(float) * original_v
    if opponent_tilt_matrix is not None:
        # Net out the opponents' expected future tilts (see the docstring): the future row
        # becomes the net future differential, and Current diff below correspondingly sheds
        # behaviour that hasn't happened yet. Total diff is unaffected.
        future_diff = future_diff - opponent_tilt_matrix.astype(float) * original_v
    current_diff = total_diff - future_diff  # the board as it stands (all future tilts in Future diff)
    current_values = np.round(current_diff, 2).tolist()
    current_totals = np.round(current_diff.sum(axis=1), 2).tolist()
    future_values  = np.round(future_diff, 2).tolist()
    future_totals  = np.round(future_diff.sum(axis=1), 2).tolist()
    return [
        [
            GScoreRow(label='Current diff', values=current_values[i], total=current_totals[i], is_total=False),
            GScoreRow(label=candidate_last_names[i], values=own_values[i], total=own_totals[i], is_total=False),
            GScoreRow(label='Future diff', values=future_values[i], total=future_totals[i], is_total=False),
            GScoreRow(label='Total diff', values=total_values[i], total=total_totals[i], is_total=True),
        ]
        for i in range(n_players)
    ]


# ── Flex allocations ──────────────────────────────────────────────────────────

def _remaining_flex_slots(
    rosters_rows: np.ndarray
    , n_team_so_far: int
    , slot_counts: dict
    , position_structure: dict
) -> dict[str, np.ndarray]:
    """Per-candidate count of each flex type's slots still open for future picks.

    A flex slot is "taken" when the roster solve seats a current player or the candidate in it — those
    occupy the first n_team_so_far + 1 columns of each candidate's slot-assignment row. The remaining
    slots of that flex type are what the position shares describe filling, so the flex-allocation
    display scales the shares by this count rather than the full league slot count.

    Args:
        rosters_rows:       Slot-index matrix, shape (n_candidates, n_columns); rosters_rows[rank, j]
                            is the slot index assigned to player j in order [team_so_far..., candidate, future...].
        n_team_so_far:      Number of players already on the drafter's team.
        slot_counts:        Dict mapping position code → number of roster slots.
        position_structure: Dict with 'base_list' and 'flex_list'.

    Returns:
        Dict flex type → per-candidate array (length n_candidates) of remaining open slots.
    """
    flex_types     = position_structure['flex_list']
    position_order = position_structure['base_list'] + flex_types
    # Slot index → position code, in the same canonical order _make_slot_names uses.
    slot_type_by_index = [
        position_code
        for position_code in position_order
        for _ in range(slot_counts.get(position_code, 0))
    ]
    filled = rosters_rows[:, :n_team_so_far + 1].astype(int)   # slots taken by current players + candidate
    remaining: dict[str, np.ndarray] = {}
    for flex_type in flex_types:
        type_slot_indices = {i for i, code in enumerate(slot_type_by_index) if code == flex_type}
        taken = np.array([sum(1 for slot in row if slot in type_slot_indices) for row in filled])
        remaining[flex_type] = np.maximum(slot_counts.get(flex_type, 0) - taken, 0)
    return remaining


def _build_flex_allocations(
    n_players: int
    , base_list: list[str]
    , position_structure: dict
    , position_shares_arrays: dict
    , slot_counts: dict
    , remaining_flex_by_rank: dict[str, np.ndarray]
) -> list[FlexAllocations]:
    """Build the flex-slot usage table for every candidate at once.

    For each flex position type (e.g. 'UTIL') that has at least one slot, one
    row is created showing how many of those slots the candidate is expected to
    occupy in each base-position role (e.g. PG share of UTIL slots).  A Total
    row sums across all flex types.

    Position shares are fractions (0–1) representing the probability the player
    occupies a given base-position role within a flex slot.  Multiplying by the
    slot count gives an expected number of slots used.  All of this arithmetic is
    vectorised across candidates: each active flex type contributes a
    (n_players, n_bases) expected-usage matrix, the Total column sums those
    matrices, and only the final FlexRow/FlexAllocations assembly is per-candidate.

    Args:
        n_players:              Number of candidates (rows in the share arrays).
        base_list:              Ordered list of base position codes (e.g. ['PG','SG','SF','PF','C']).
        position_structure:     Dict with 'flex_list' and 'flex' keys describing which
                                base positions each flex type can accommodate.
        position_shares_arrays: Dict mapping flex type → (numpy_array, base_to_col) or None.
                                numpy_array has shape (n_candidates, n_bases); base_to_col
                                maps base position name → column index.
        slot_counts:            Dict mapping position code → number of roster slots.

    Returns:
        One FlexAllocations per candidate rank, each with one row per active flex
        type plus a Total row.
    """
    flex_position_types   = position_structure['flex_list']
    flex_position_details = position_structure['flex']
    n_bases               = len(base_list)
    base_column_index     = {base_pos: i for i, base_pos in enumerate(base_list)}

    # totals_matrix accumulates expected slot usage across all flex types, shape
    # (n_players, n_bases). Bases never eligible in any flex type stay 0.0.
    totals_matrix = np.zeros((n_players, n_bases))
    # Per active flex type: (label, value_lists) where value_lists is a per-rank list of
    # rounded base values (None for ineligible/absent), or None to mark an all-None row.
    flex_row_specs: list[tuple[str, list[list[float | None]] | None]] = []

    for flex_type in flex_position_types:
        n_flex_slots = slot_counts.get(flex_type, 0)
        if n_flex_slots == 0:
            continue  # this flex type has no slots in the current league settings

        label                = f"{flex_type}-{n_flex_slots}"
        eligible_bases       = flex_position_details[flex_type]['bases']
        share_array_and_cols = position_shares_arrays.get(flex_type)
        # Per-candidate count of this flex type's still-open slots (falls back to the full count).
        remaining_counts     = remaining_flex_by_rank.get(flex_type)
        if remaining_counts is None:
            remaining_counts = np.full(n_players, n_flex_slots)

        if share_array_and_cols is None:
            # No share data for this flex type; every column is ineligible for all ranks.
            flex_row_specs.append((label, None))
            continue

        share_array, base_to_col = share_array_and_cols
        # Expected slot usage per (rank, base); NaN marks bases ineligible for this flex.
        usage = np.full((n_players, n_bases), np.nan)
        for base_pos in base_list:
            if base_pos in eligible_bases:
                # Convert share fraction to expected usage across this candidate's *remaining* flex slots.
                usage[:, base_column_index[base_pos]] = share_array[:, base_to_col[base_pos]] * remaining_counts

        totals_matrix += np.nan_to_num(usage)   # NaN (ineligible) contributes nothing
        # Round once, then swap NaN → None so ineligible columns serialise as null.
        value_lists = [
            [None if value != value else value for value in row]
            for row in np.round(usage, 2).tolist()
        ]
        flex_row_specs.append((label, value_lists))

    total_row_lists = np.round(totals_matrix, 2).tolist()

    result: list[FlexAllocations] = []
    for rank_idx in range(n_players):
        flex_rows: list[FlexRow] = [
            FlexRow(
                label    = label,
                values   = [None] * n_bases if value_lists is None else value_lists[rank_idx],
                is_total = False,
            )
            for label, value_lists in flex_row_specs
        ]
        flex_rows.append(FlexRow(label='Total', values=total_row_lists[rank_idx], is_total=True))
        result.append(FlexAllocations(base_positions=base_list, rows=flex_rows))

    return result


# ── Roster assignment ─────────────────────────────────────────────────────────

def _build_roster_assignments(
    candidate_player_ids: list[int]
    , my_players: list[int]
    , rosters_rows: np.ndarray
    , slot_names: list[str]
) -> list[Roster]:
    """Build the Roster display for every candidate from the pre-computed slot assignments.

    rosters_rows[rank, j] is the slot index assigned to player j in the ordering
    [team_so_far[0], ..., team_so_far[-1], candidate, future_player[0], ...].
    Columns beyond len(my_players) + 1 are future hypothetical players and are ignored.

    The already-drafted players' RosterAssignment objects depend only on their identity
    and are identical across candidates, so they are built once and shared; the slot each
    one lands in still varies per candidate (the optimiser may repack the roster around
    the candidate) and is read straight from rosters_rows. Only the candidate's own
    assignment is unique per rank. Slot indices are converted to Python ints in one pass.

    Args:
        candidate_player_ids: Per-rank candidate player id for the candidate's own slot.
        my_players:           Player ids already on the user's team, in roster-column order.
        rosters_rows:         Slot-index matrix, shape (n_players, n_columns).
        slot_names:           Ordered slot IDs (e.g. ['PG1', 'PG2', 'UTIL1']).

    Returns:
        One Roster per candidate rank.
    """
    n_slots       = len(slot_names)
    n_team_so_far = len(my_players)
    n_columns     = rosters_rows.shape[1]

    # Shared assignments for the already-drafted players (built once).
    drafted_assignments = [
        RosterAssignment(player_id=int(player_id), is_candidate=False)
        for player_id in my_players
    ]
    # Cross the numpy→Python boundary once for all slot indices.
    rosters_int = rosters_rows.astype(int).tolist()

    result: list[Roster] = []
    for rank_idx, roster_row in enumerate(rosters_int):
        assignments: dict[str, RosterAssignment | None] = {slot: None for slot in slot_names}

        for team_position, slot_idx in enumerate(roster_row[:n_team_so_far]):
            if 0 <= slot_idx < n_slots:
                assignments[slot_names[slot_idx]] = drafted_assignments[team_position]

        if n_team_so_far < n_columns:
            candidate_slot_idx = roster_row[n_team_so_far]
            if 0 <= candidate_slot_idx < n_slots:
                assignments[slot_names[candidate_slot_idx]] = RosterAssignment(
                    player_id=int(candidate_player_ids[rank_idx]),
                    is_candidate=True,
                )

        result.append(Roster(slots=slot_names, assignments=assignments))

    return result


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



