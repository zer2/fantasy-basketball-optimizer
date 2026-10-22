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
    .venv\Scripts\python testing_files\simulate_auction_cash.py --label 2023 --season 2023-24

The truth is one season's actual stats (--season; default the benchmark season). Each autodrafter bids from its own
expectation: those actuals plus its own noise (services/auction_autodraft.read_team_x_scores). Every finished team is
scored twice: on the actuals ("actual"), and through its own expectation ("own view"), which is how good it believes
its team is. A team buys the players its noise overrates, so its own view is almost always the higher of the two.

Writes testing_files/auction_cash_runs/<label>_<timestamp>/:
    summary.json      per team: spent and unspent cash, the money paid extra for packing inefficiency, the
                      actual and own-view H-scores against the field; their totals, the field average, run
                      settings, commit, timings
    spend_curve.csv   cumulative dollars spent per team after every nomination
    nominations.csv   one row per nomination: nominator, player, winner, price, the price the same bidding reaches
                      without the cash share, the winner's and the runner-up's valuations, seconds taken

Paid extra for packing inefficiency: "Your $" is scaled as if the league's cash were spent perfectly, which no team
manages, so a team left holding more cash than its open slots can use gets a share of it added to its best options
(auction_pricing). The extra is each sale's price above what the same bidding reaches with every team's share taken
out of its valuation.
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

from experiment_harnesses.auction_simulation import (  # noqa: E402
    build_auction_session, name_auction_teams, run_auction, score_auction,
)

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
    parser.add_argument('--cash-rule', default='simple_scaling', choices=('simple_scaling', 'smooth'),
                        help="Every team's cash-share rule (backend/services/auction_pricing.py; default simple_scaling).")
    parser.add_argument('--season', default=None,
                        help="The season whose actual stats are the truth, e.g. 2023-24 (default: the benchmark season).")
    return parser.parse_args()


def read_current_commit() -> str:
    return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                          cwd=_REPOSITORY_ROOT).stdout.strip()


def main() -> None:
    arguments = parse_arguments()
    started = time.perf_counter()
    session = build_auction_session(arguments.objective, arguments.cash, arguments.drafters, arguments.season, None)
    session_seconds = time.perf_counter() - started

    team_count = session.current_settings['n_drafters']
    total_nominations = team_count * session.current_settings['n_picks']
    if arguments.max_nominations is not None:
        total_nominations = min(total_nominations, arguments.max_nominations)
    print(f'Session built in {session_seconds:.1f}s: {team_count} teams, {session.current_settings["n_picks"]} picks, '
          f'${arguments.cash}, {arguments.objective}. Running {total_nominations} nominations.', flush=True)

    def print_sale(number, total, row):
        print(f'{number:>4}/{total}  {row["player"]:<28} -> {row["winner"]:<8} ${row["price"]:<4} ({row["seconds"]:.1f}s)',
              flush=True)

    teams = name_auction_teams(session)
    run = run_auction(session, arguments.cash, arguments.max_nominations, print_sale,
                      {team: arguments.cash_rule for team in teams})
    teams, remaining_cash = run.teams, run.remaining_cash
    nomination_rows, spend_rows = run.nomination_rows, run.spend_rows
    auction_complete = run.complete
    actual_scores, own_view_scores = score_auction(session, run) if run.complete else ({}, {})
    unspent = [remaining_cash[team] for team in teams]
    excess_by_team = {team: sum(row['paid_extra_for_packing'] for row in nomination_rows if row['winner'] == team)
                      for team in teams}
    last_round = nomination_rows[-team_count:]

    summary = {
        'label':            arguments.label,
        'commit':           read_current_commit(),
        'finished_at':      datetime.datetime.now().isoformat(timespec='seconds'),
        'settings':         {'objective': arguments.objective, 'cash': arguments.cash, 'teams': team_count,
                             'picks': session.current_settings['n_picks'], 'nominations_run': total_nominations},
        'auction_complete': auction_complete,
        'unspent_cash':     {team: remaining_cash[team] for team in teams},
        'unspent_max':      max(unspent),
        'unspent_median':   statistics.median(unspent),
        'unspent_total':    sum(unspent),
        'paid_extra_for_packing': excess_by_team,
        'paid_extra_for_packing_total': sum(excess_by_team.values()),
        'paid_extra_for_packing_last_round': sum(row['paid_extra_for_packing'] for row in last_round),
        'season':           arguments.season or 'the benchmark season',
        'actual_h_scores':  actual_scores,
        'actual_h_score_average': statistics.mean(actual_scores.values()) if actual_scores else None,
        'own_view_h_scores': own_view_scores,
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

    print(f'\n{"Team":<8} {"actual H":>9} {"own view":>9} {"overrated":>10} {"spent":>7} {"unspent":>8} '
          f'{"paid extra (packing)":>21}')
    for team in teams:
        scores = (f'{actual_scores[team]:>9.1f} {own_view_scores[team]:>9.1f} '
                  f'{own_view_scores[team] - actual_scores[team]:>+10.1f}') if actual_scores else f'{"":>30}'
        print(f'{team:<8} {scores} ${arguments.cash - remaining_cash[team]:>6g} ${remaining_cash[team]:>7g} '
              f'${excess_by_team[team]:>20}')
    if actual_scores:
        print(f'{"Average":<8} {summary["actual_h_score_average"]:>9.2f} {statistics.mean(own_view_scores.values()):>9.2f}')
    print(f'\nUnspent cash: total ${summary["unspent_total"]:g} (max ${summary["unspent_max"]:g}). '
          f'Paid extra for packing inefficiency (price above the same bidding without the cash share): '
          f'${summary["paid_extra_for_packing_total"]:g}, ${summary["paid_extra_for_packing_last_round"]:g} of it in '
          f'the last round.')
    print(f'Results: {run_directory}')


if __name__ == '__main__':
    main()
