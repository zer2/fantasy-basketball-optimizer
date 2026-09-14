"""Build everything the roster-slot assignment scene needs, so rendering never touches Snowflake.

Run once (or whenever the season, the defaults or the chosen players change):

    python visualizations/prepare_assignment_data.py

Writes one file into this directory:
  data/assignment_<season>.json    the thirteen roster slots, five drafted players and their
                                   eligibility, the full reward matrix, the optimal assignment
                                   the shipped solver returns, and a naive assignment to beat

Every number in the scene comes from the app's own code. The eligibility rows are built by
`get_player_rows`, the future-pick rewards by `get_future_player_rows`, and the assignment by
`optimize_positions_all_players` -- the same three functions the H-score descent calls on every
iteration. Nothing here restates the position structure or scores a slot by hand, because the
whole claim the scene makes is about what the real optimiser does.

WHAT THE SCENE IS ABOUT. A player already on your roster scores 0 in every slot he is eligible
for and -inf everywhere else, so he contributes nothing wherever he is placed. All the value in
the matrix sits in the rows for the picks you have NOT made yet, and those rows are identical:
a future pick is worth whatever the slot it lands in is worth. Because every slot is used exactly
once, maximising the total is the same as making your drafted players occupy the CHEAPEST slots
they are eligible for -- they are not being placed to be useful, they are being placed to get out
of the way. That is counterintuitive enough to be worth thirty seconds of animation.

Headshots come from `prepare_season_data.py`, which must have been run first; this script only
checks that the five chosen players have one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Import the app itself rather than reimplementing its numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.api.routers.sessions import _build_current_settings            # noqa: E402
from backend.math.position_config import PositionConfig                     # noqa: E402
from backend.math.position_optimization import (                            # noqa: E402
    get_future_player_rows, get_player_rows, optimize_positions_all_players,
)
from backend.services.session_management import build_session               # noqa: E402
from prepare_season_data import build_default_session_request               # noqa: E402


SEASON = '2025-26'
SPORT = 'NBA'

# Five drafted players: ONE PER ROUND, taken in G-score order from the pool this season's default
# settings produce, so the team is one a real draft could hand you rather than five stars nobody
# could have had together. Their eligibilities are also what makes the problem bite -- two players
# who can ONLY play centre take both C slots between them, which forces the third centre off the
# position he actually plays, and the three-position player is the one the naive assignment
# misplaces. A centre-heavy front court is not a contrivance either: it is what a team punting Free
# Throw % ends up with. Ordered as the solver consumes them: the first four are the team so far,
# the fifth fills the candidate row.
DRAFTED_PLAYER_IDS = [
    1626157,    # Karl-Anthony Towns  C, PF     -- round 1 (rank 11); wants a C slot, cannot have one
    1642270,    # Donovan Clingan     C         -- round 2 (rank 19); centre-only
    1631105,    # Jalen Duren         C         -- round 3 (rank 25); centre-only, so both C slots go
    1631095,    # Jabari Smith Jr.    C, PF, SF -- round 4 (rank 46); the flexible one
    1628368,    # De'Aaron Fox        PG, SG    -- round 5 (rank 52); fills out the back court
]

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent
_HEADSHOT_DIR = _VISUALIZATIONS_DIR / 'assets' / 'headshots'


def build_slot_labels(position_config: PositionConfig) -> tuple[list[str], list[str]]:
    """The thirteen roster slots in the order the optimiser lays them out.

    Returns (position codes, display labels). Base positions first, each repeated as many times
    as the league has slots for it, then the flex positions -- the layout `position_ranges`
    already encodes, read back out of it rather than restated, so a league with a different
    slot table produces a different picture without this script changing.
    """
    structure = position_config.position_structure
    ordered_codes = structure['base_list'] + structure['flex_list']

    slot_codes: list[str] = []
    for position_code in ordered_codes:
        slot_codes += [position_code] * position_config.position_numbers[position_code]

    # Every slot carries its own position's label, duplicates included: two C slots really are
    # interchangeable, and numbering them would invent a distinction the optimiser does not have.
    display_labels = [code.upper() for code in slot_codes]
    return slot_codes, display_labels


def build_uniform_position_shares(position_config: PositionConfig, candidate_count: int) -> dict:
    """The flex-slot shares a cold-start descent begins from: every base position equally likely.

    `optimize_positions_all_players` needs these to split flex slots back into base positions for
    its caller. This scene never shows that split -- it shows the assignment -- but the shares are
    a required argument, so they are built the way the agent builds them on a cold start rather
    than with a value invented here.
    """
    return {
        position_code: pd.DataFrame({
            base_position: [1 / len(position_info['bases'])] * candidate_count
            for base_position in position_info['bases']
        })
        for position_code, position_info in position_config.position_structure['flex'].items()
    }


def solve_with_forced_placement(
    reward_matrix: np.ndarray
    , forced_row: int
    , forced_column: int
) -> np.ndarray:
    """The best assignment available once one player has been pinned to one slot.

    This is how the scene gets a wrong answer worth showing: not a permutation made up to look
    bad, but the best the optimiser could do after a plausible human decision has been taken for
    it. Everything except the pinned row is still solved by the same scipy call the app makes.
    """
    free_rows = [row for row in range(reward_matrix.shape[0]) if row != forced_row]
    free_columns = [column for column in range(reward_matrix.shape[1]) if column != forced_column]

    solved_rows, solved_columns = linear_sum_assignment(
        reward_matrix[np.ix_(free_rows, free_columns)], maximize=True)

    assignment = np.empty(reward_matrix.shape[0], dtype=int)
    assignment[forced_row] = forced_column
    for free_row_index, free_column_index in zip(solved_rows, solved_columns):
        assignment[free_rows[free_row_index]] = free_columns[free_column_index]
    return assignment


def total_of_assignment(reward_matrix: np.ndarray, assignment: np.ndarray) -> float:
    """What an assignment is worth: the reward in every chosen cell, added up.

    Finite by construction for any assignment that respects eligibility, and the drafted rows all
    contribute exactly zero -- so this total is entirely the value left to the future picks.
    """
    chosen = reward_matrix[np.arange(len(assignment)), assignment]
    if not np.isfinite(chosen).all():
        raise SystemExit('An assignment placed a player in a slot he is not eligible for; the '
                         'solver and the reward matrix disagree.')
    return float(chosen.sum())


def as_json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
    """JSON has no -inf. Ineligible cells travel as null and the scene draws them dark."""
    return [[None if not np.isfinite(value) else round(float(value), 6) for value in row]
            for row in matrix]


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults (this pulls from Snowflake)...')
    request = build_default_session_request(SEASON, SPORT)
    session = build_session(
        current_settings = _build_current_settings(request),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )

    agent = session.agent
    position_config = agent.position_config
    slot_codes, slot_labels = build_slot_labels(position_config)
    slot_count = len(slot_codes)

    registry = session.player_registry
    eligible_positions = [list(agent.positions.get(player_id)) for player_id in DRAFTED_PLAYER_IDS]
    player_names = [registry[player_id].name for player_id in DRAFTED_PLAYER_IDS]

    missing_headshots = [
        name for player_id, name in zip(DRAFTED_PLAYER_IDS, player_names)
        if not (_HEADSHOT_DIR / f'{player_id}.png').exists()
    ]
    if missing_headshots:
        # Named rather than counted: the scene cannot draw a face it does not have, and the fix is
        # either to run prepare_season_data.py or to choose a different player.
        raise SystemExit(f'No headshot for: {", ".join(missing_headshots)}. Run '
                         f'`python visualizations/prepare_season_data.py` first, or choose '
                         f'players from the pool it writes.')

    # ── The two halves of the matrix, both from the shipped builders ────────────────────
    drafted_rows = get_player_rows(eligible_positions, position_config)

    # Neutral category weights: the vector every descent starts from, so the slot values on screen
    # are the ones a first pick is scored against rather than a punt-specific tilt.
    neutral_weights = agent.get_starting_category_weights().reshape(1, -1)
    position_rewards = agent.get_position_priorities_from_category_weights(neutral_weights)
    future_pick_row = get_future_player_rows(position_rewards, position_config)[0]

    future_pick_count = slot_count - len(DRAFTED_PLAYER_IDS)
    reward_matrix = np.concatenate(
        [drafted_rows, np.tile(future_pick_row, (future_pick_count, 1))], axis=0)

    # ── The real solve ──────────────────────────────────────────────────────────────────
    # The app never solves a bare matrix: it solves "this team, plus this candidate, plus the picks
    # still to come", which is exactly the thirteen rows above with the fifth drafted player in the
    # candidate seat. Calling it this way rather than reaching for scipy keeps the assignment on
    # screen the one the descent actually works with.
    rosters, _, _ = optimize_positions_all_players(
        candidate_player_array = drafted_rows[-1:],
        position_rewards       = position_rewards,
        team_so_far_array      = drafted_rows[:-1],
        position_shares        = build_uniform_position_shares(position_config, 1),
        position_config        = position_config,
    )
    optimal_assignment = np.asarray(rosters[0], dtype=int)
    if sorted(optimal_assignment.tolist()) != list(range(slot_count)):
        raise SystemExit('The solver returned something that is not a permutation of the slots; '
                         'the chosen five probably cannot all be fitted onto one roster.')
    optimal_total = total_of_assignment(reward_matrix, optimal_assignment)

    # ── The assignment a person would guess ─────────────────────────────────────────────
    # "The player who can play anywhere goes in the slot that takes anyone." It is the most
    # natural wrong answer, and it is wrong for the reason the scene is about: a Util slot is the
    # single most valuable thing on the board to a pick you have not made yet, and spending it on
    # a player who is worth zero wherever he stands throws that value away.
    eligibility_counts = np.isfinite(drafted_rows).sum(axis=1)
    most_flexible_row = int(np.argmax(eligibility_counts))
    first_utility_slot = position_config.position_ranges['Util']['start']
    naive_assignment = solve_with_forced_placement(
        reward_matrix, most_flexible_row, first_utility_slot)
    naive_total = total_of_assignment(reward_matrix, naive_assignment)

    if naive_total >= optimal_total:
        raise SystemExit('The naive assignment is no worse than the optimum, so there is nothing '
                         'for the scene to show. Pick players whose flex slot actually costs '
                         'something.')

    print(f'Slots: {" ".join(slot_labels)}')
    for name, positions, column in zip(player_names, eligible_positions, optimal_assignment):
        print(f'  {name:<26} {"/".join(positions):<14} -> {slot_labels[column]}')
    print(f'Future-pick value by slot: '
          f'{" ".join(f"{value:+.3f}" for value in future_pick_row)}')
    print(f'Optimal total {optimal_total:.4f}; '
          f'flex-slot guess {naive_total:.4f} '
          f'(costs {optimal_total - naive_total:.4f})')

    data_path = _VISUALIZATIONS_DIR / 'data' / f'assignment_{SEASON.replace("-", "_")}.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':             SEASON,
        'slot_codes':         slot_codes,
        'slot_labels':        slot_labels,
        'drafted_players': [
            {
                'player_id':          int(player_id),
                'name':               name,
                'eligible_positions': positions,
                'eligible_slots':     [bool(np.isfinite(value)) for value in row],
            }
            for player_id, name, positions, row
            in zip(DRAFTED_PLAYER_IDS, player_names, eligible_positions, drafted_rows)
        ],
        'future_pick_count':  future_pick_count,
        'future_pick_row':    [round(float(value), 6) for value in future_pick_row],
        'reward_matrix':      as_json_matrix(reward_matrix),
        'optimal_assignment': optimal_assignment.tolist(),
        'optimal_total':      round(optimal_total, 6),
        'naive_assignment':   naive_assignment.tolist(),
        'naive_total':        round(naive_total, 6),
    }, indent=1), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
