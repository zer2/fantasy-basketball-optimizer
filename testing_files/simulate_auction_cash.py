# testing_files/simulate_auction_cash.py
"""
Simulates complete auctions with every team on autodraft and reports how much cash each team leaves
unspent, and when it spends it.

Each nomination calls the same backend service the browser does (nominate_auction_player): the
nominator's choice, the opening bid, every team's valuation and every team's maximum allowable bid.
The bidding loop below MIRRORS advanceAutobidders in frontend/data_entry/auction_autodraft.ts and the
nomination rotation MIRRORS advanceNominatorIndex in frontend/data_entry/auction_state.ts: nominations
rotate in board order skipping full teams; bidding opens at $1; teams take turns in board order from the
nominator's left, each raising by $1 while the next bid is within both its valuation and its maximum
allowable bid, otherwise passing for good; the turn returning to the high bidder ends it. (The browser
settles the autodrafters' bidding first, in settleAutodrafterBidding, so manual drafters join at that price;
with every team an autodrafter, as here, that is the whole auction and the result is identical.) Change
these together with the browser code.

Run from the repository root, with the project's environment (it reads the player data from the
database the app uses):

    .venv\Scripts\python testing_files\simulate_auction_cash.py --label baseline
    .venv\Scripts\python testing_files\simulate_auction_cash.py --label smoke --max-nominations 5

Writes testing_files/auction_cash_runs/<label>_<timestamp>/:
    summary.json      unspent cash per team (and max / median / total), each final team's H-score
                      against the field and the field average, run settings, commit, timings
    spend_curve.csv   cumulative dollars spent per team after every nomination
    nominations.csv   one row per nomination: nominator, player, winner, price, the winner's and the
                      runner-up's valuations, seconds taken
"""

import argparse
import csv
import datetime
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

# The app requires these before backend.main is imported (see testing_files/conftest.py).
os.environ.setdefault('SESSION_SECRET_KEY', 'auction-simulation-session-secret')
os.environ.setdefault('RATE_LIMITS_ENABLED', 'false')

_TESTING_DIRECTORY = Path(__file__).resolve().parent
_REPOSITORY_ROOT   = _TESTING_DIRECTORY.parent
sys.path.insert(0, str(_REPOSITORY_ROOT))
sys.path.insert(0, str(_TESTING_DIRECTORY))

from benchmark_helpers import client, _build_session_request                          # noqa: E402
from backend.state.session import get_session                                         # noqa: E402
from backend.services.ranking import rank_candidates                                  # noqa: E402
from backend.services.auction_autodraft import nominate_auction_player                # noqa: E402

_RUNS_DIRECTORY = _TESTING_DIRECTORY / 'auction_cash_runs'


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulate all-autodrafter auctions and report unspent cash.')
    parser.add_argument('--label', required=True, help='Name for this run, e.g. baseline or change-1.')
    parser.add_argument('--objective', default='Each Category',
                        help='An OBJECTIVE_PRESETS name from benchmark_helpers.py (default: Each Category).')
    parser.add_argument('--cash', type=int, default=200, help='Budget per team (default: 200).')
    parser.add_argument('--drafters', type=int, default=None, help='Number of teams (default: parameters.yaml).')
    parser.add_argument('--max-nominations', type=int, default=None,
                        help='Stop after this many nominations (a quick check that the script runs).')
    return parser.parse_args()


def read_current_commit() -> str:
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                          cwd=_REPOSITORY_ROOT).stdout.strip()


def build_auction_session(
    objective: str
    , cash: int
    , drafters: int | None
):
    request = _build_session_request(objective=objective, cash_per_team=cash, n_drafters=drafters)
    response = client.post('/sessions', json=request)
    if response.status_code != 201:
        raise RuntimeError(f'Session build failed ({response.status_code}): {response.text}')
    return get_session(response.json()['session_id'])


def advance_nominator(
    nominator_index: int
    , player_assignments: dict[str, list[int]]
    , teams: list[str]
    , total_roster_picks: int
) -> int:
    """The next team in board order with an empty slot (advanceNominatorIndex)."""
    for step in range(1, len(teams) + 1):
        candidate = (nominator_index + step) % len(teams)
        if len(player_assignments[teams[candidate]]) < total_roster_picks:
            return candidate
    return nominator_index


def run_bidding(
    nominator_index: int
    , opening_bid: int
    , valuations: dict[str, float]
    , max_allowed_bids: dict[str, int]
    , teams: list[str]
) -> tuple[int, int]:
    """The browser's round-robin bidding with every team an autodrafter (advanceAutobidders).
    Returns (winner index, price)."""
    team_count = len(teams)
    active_bidders = {index for index, team in enumerate(teams) if max_allowed_bids[team] >= opening_bid}
    active_bidders.add(nominator_index)
    current_bid = opening_bid
    high_bidder = nominator_index
    turn = (nominator_index + 1) % team_count
    while True:
        if len(active_bidders) <= 1:
            return high_bidder, current_bid
        if turn not in active_bidders:
            turn = (turn + 1) % team_count
            continue
        if turn == high_bidder:
            return high_bidder, current_bid
        team = teams[turn]
        next_bid = current_bid + 1
        if next_bid <= valuations[team] and next_bid <= max_allowed_bids[team]:
            current_bid = next_bid
            high_bidder = turn
        else:
            active_bidders.discard(turn)
        turn = (turn + 1) % team_count


def score_final_teams(
    session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , teams: list[str]
) -> dict[str, float]:
    """Each finished team's H-score against the field (the full-roster evaluate's single row)."""
    scores = {}
    for team in teams:
        result = rank_candidates(session=session, player_assignments=player_assignments, my_team_id=team,
                                 exclusion_list=[], remaining_cash=remaining_cash)
        if len(result.candidates) != 1:
            raise RuntimeError(f'{team}: expected one full-roster row, got {len(result.candidates)}.')
        scores[team] = result.candidates[0].h_score
    return scores


def main() -> None:
    arguments = parse_arguments()
    started = time.perf_counter()
    session = build_auction_session(arguments.objective, arguments.cash, arguments.drafters)
    session_seconds = time.perf_counter() - started

    team_count         = session.current_settings['n_drafters']
    total_roster_picks = session.current_settings['n_picks']
    teams              = [f'Team {index + 1}' for index in range(team_count)]
    player_assignments = {team: [] for team in teams}
    remaining_cash     = {team: float(arguments.cash) for team in teams}
    total_nominations  = team_count * total_roster_picks
    if arguments.max_nominations is not None:
        total_nominations = min(total_nominations, arguments.max_nominations)

    print(f'Session built in {session_seconds:.1f}s: {team_count} teams, {total_roster_picks} picks, '
          f'${arguments.cash}, {arguments.objective}. Running {total_nominations} nominations.', flush=True)

    nomination_rows, spend_rows = [], []
    nominator_index = 0
    for nomination_number in range(1, total_nominations + 1):
        nomination_started = time.perf_counter()
        with session.lock:
            nomination = nominate_auction_player(
                session             = session,
                player_assignments  = player_assignments,
                remaining_cash      = remaining_cash,
                nominator_id        = teams[nominator_index],
                nominated_player_id = None,
                opening_bid         = None,
                valuation_team_ids  = teams,
            )
        winner_index, price = run_bidding(nominator_index, nomination.opening_bid, nomination.valuations,
                                          nomination.max_allowed_bids, teams)
        winner = teams[winner_index]
        player_assignments[winner].append(nomination.nominated_player_id)
        remaining_cash[winner] -= price

        ranked_valuations = sorted(nomination.valuations.values(), reverse=True)
        seconds = time.perf_counter() - nomination_started
        nomination_rows.append({
            'nomination':        nomination_number,
            'nominator':         teams[nominator_index],
            'player_id':         nomination.nominated_player_id,
            'player':            nomination.nominated_player_name,
            'winner':            winner,
            'price':             price,
            'winner_valuation':  nomination.valuations[winner],
            'runner_up_valuation': ranked_valuations[1] if len(ranked_valuations) > 1 else 0.0,
            'seconds':           round(seconds, 2),
        })
        spend_rows.append({'nomination': nomination_number,
                           **{team: round(arguments.cash - remaining_cash[team], 2) for team in teams}})
        print(f'{nomination_number:>4}/{total_nominations}  {nomination.nominated_player_name:<28} '
              f'-> {winner:<8} ${price:<4} ({seconds:.1f}s)', flush=True)

        nominator_index = advance_nominator(nominator_index, player_assignments, teams, total_roster_picks)

    auction_complete = all(len(roster) == total_roster_picks for roster in player_assignments.values())
    final_scores = score_final_teams(session, player_assignments, remaining_cash, teams) if auction_complete else {}
    unspent = [remaining_cash[team] for team in teams]

    summary = {
        'label':            arguments.label,
        'commit':           read_current_commit(),
        'finished_at':      datetime.datetime.now().isoformat(timespec='seconds'),
        'settings':         {'objective': arguments.objective, 'cash': arguments.cash, 'teams': team_count,
                             'picks': total_roster_picks, 'nominations_run': total_nominations},
        'auction_complete': auction_complete,
        'unspent_cash':     {team: remaining_cash[team] for team in teams},
        'unspent_max':      max(unspent),
        'unspent_median':   statistics.median(unspent),
        'unspent_total':    sum(unspent),
        'final_h_scores':   final_scores,
        'final_h_score_average': statistics.mean(final_scores.values()) if final_scores else None,
        'seconds_session':  round(session_seconds, 1),
        'seconds_total':    round(time.perf_counter() - started, 1),
        'seconds_per_nomination_median': statistics.median(row['seconds'] for row in nomination_rows),
    }

    run_directory = _RUNS_DIRECTORY / f'{arguments.label}_{datetime.datetime.now():%Y%m%d_%H%M%S}'
    run_directory.mkdir(parents=True)
    (run_directory / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    for file_name, rows in (('nominations.csv', nomination_rows), ('spend_curve.csv', spend_rows)):
        with open(run_directory / file_name, 'w', newline='', encoding='utf-8') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f'\nUnspent cash: max ${summary["unspent_max"]:g}, median ${summary["unspent_median"]:g}, '
          f'total ${summary["unspent_total"]:g}')
    for team in teams:
        score = f'  H-score {final_scores[team]:.1f}' if final_scores else ''
        print(f'  {team:<8} ${remaining_cash[team]:>6g} unspent{score}')
    if final_scores:
        print(f'Field average H-score: {summary["final_h_score_average"]:.2f} (expected 50)')
    print(f'Results: {run_directory}')


if __name__ == '__main__':
    main()
