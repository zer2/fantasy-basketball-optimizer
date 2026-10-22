"""An all-autodraft auction, as the browser runs it, and its scores (testing_files/simulate_auction_cash.py and the
experiment report's Auction cash-share rules section: run_rule_matches_in_parallel).

Each nomination calls the backend service the browser does (nominate_auction_player): the nominator's choice, the opening
bid, every team's valuation and every team's maximum allowable bid. The bidding (run_bidding) MIRRORS advanceAutobidders
in frontend/data_entry/auction_autodraft.ts and the nomination rotation (advance_nominator) MIRRORS
advanceNominatorIndex in frontend/data_entry/auction_state.ts; change these together with the browser code.

The truth is one season's actual stats. Each autodrafter bids from its own expectation, the actuals plus its own noise
(services/auction_autodraft.read_team_x_scores). A finished team is scored on the actuals and through its own
expectation, its "own view": what each drafter is trying to maximise (the user, 2026-10-21).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates
from backend.services.auction_autodraft import nominate_auction_player, read_team_x_scores


@dataclass
class AuctionRun:
    teams: list[str]
    player_assignments: dict[str, list[int]]
    remaining_cash: dict[str, float]
    nomination_rows: list[dict]     # one per sale: who, what, price, valuations, the price the bidding would have
                                    # reached without the cash share and the extra paid over it, seconds
    spend_rows: list[dict]          # cumulative spend per team after every sale
    complete: bool


def build_auction_session(
    objective: str
    , cash: int
    , drafters: int | None
    , season: str | None
    , cash_share_softness: float | None
):
    """An auction session on one season's actual stats, or on the benchmark request's season when it is None. beth = 0,
    as in every simulation: the season's stats are the truth here, and the app's Bayesian self-doubt would score
    every finished team below 50% against a field of its peers (49.84% at the default 3.0, 2026-10-21). SAVOR is off
    (streaming noise S = 0): it moves dollar values toward streamable players' replacement value, but the finished
    teams are scored by H-score, which has no streaming, so a drafter banking on it would be judged on a value it
    never priced (the user, 2026-10-21)."""
    request = _build_session_request(objective=objective, cash_per_team=cash, n_drafters=drafters)
    request['model_settings']['beth'] = 0
    request['model_settings']['streaming_noise'] = 0
    request['model_settings']['replacement_threshold'] = 0
    if cash_share_softness is not None:
        request['model_settings']['cash_share_softness'] = cash_share_softness
    if season is not None:
        request['data_source'] = {'type': 'historical', 'season': season}
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


def score_final_teams_in_own_view(
    session
    , player_assignments: dict[str, list[int]]
    , remaining_cash: dict[str, float]
    , teams: list[str]
) -> dict[str, float]:
    """Each finished team's H-score as it sees it: through its own noisy X-scores (read_team_x_scores), the view it
    bid from. It bought the players it overrates, so this should almost always be above its standard score."""
    scores = {}
    for team in teams:
        with session.agent.substituting_x_scores(read_team_x_scores(session, team)):
            scores.update({team: score_final_teams(session, player_assignments, remaining_cash, [team])[team]})
    return scores


# A category a team expects to win less often than this is one it punts (the experiments' soft and hard punt lines).
SOFT_PUNT_RATE = 40.0
HARD_PUNT_RATE = 15.0


@dataclass
class TeamBuild:
    """A finished team's build as it sees it (its own view): the categories it punts, how much of its players' value
    lies in them, and its misfits, players with at least half their value in a punted category."""
    punted: list[str]                  # expected win rate under SOFT_PUNT_RATE, '!' marking under HARD_PUNT_RATE
    punted_category_share: float       # punted categories over all categories: the value share if players ignored it
    punted_value_share: float          # positive X-score value of its players lying in punted categories, over all
    misfits: list[str]


def read_team_builds(
    session
    , run: AuctionRun
    , short_names: dict[str, str]
) -> dict[str, TeamBuild]:
    """Each finished team's build in its own view: its category win rates through its own X-scores, and its players'
    positive X-score value (also its own view) split between the categories it punts and the rest."""
    categories = session.current_settings['categories']
    builds = {}
    for team in run.teams:
        own_x_scores = read_team_x_scores(session, team)
        with session.agent.substituting_x_scores(own_x_scores):
            result = rank_candidates(session=session, player_assignments=run.player_assignments, my_team_id=team,
                                     exclusion_list=[], remaining_cash=run.remaining_cash)
        rates = result.candidates[0].win_rates
        punted_mask = np.array([rate < SOFT_PUNT_RATE for rate in rates])
        punted = [short_names.get(category, category) + ('!' if rate < HARD_PUNT_RATE else '')
                  for category, rate in zip(categories, rates) if rate < SOFT_PUNT_RATE]
        players = [player for player in run.player_assignments[team] if player in own_x_scores.index]
        positive_values = np.clip(own_x_scores.loc[players, categories].to_numpy(), 0, None)   # (players, categories)
        player_totals = positive_values.sum(axis=1)
        player_punted = positive_values[:, punted_mask].sum(axis=1)
        misfits = [session.player_registry[player].name for player, total, in_punted
                   in zip(players, player_totals, player_punted) if total > 0 and in_punted / total >= 0.5]
        builds[team] = TeamBuild(
            punted                = punted,
            punted_category_share = float(punted_mask.mean()),
            punted_value_share    = float(player_punted.sum() / player_totals.sum()) if player_totals.sum() > 0 else 0.0,
            misfits               = misfits,
        )
    return builds


def name_auction_teams(session) -> list[str]:
    """The simulated auction's teams, in board order (the browser's default labels)."""
    return [f'Team {index + 1}' for index in range(session.current_settings['n_drafters'])]


def run_auction(
    session
    , cash_per_team: int
    , max_nominations: Optional[int]
    , report_sale: Optional[Callable[[int, int, dict], None]]
    , cash_share_rules: dict[str, str]
) -> AuctionRun:
    """Runs the auction to the end (or for max_nominations sales), each team pricing with its cash-share rule
    (auction_pricing.CASH_SHARE_RULES; cash_share_rules names every team). report_sale(number, total, row), when given,
    is called after every sale."""
    team_count         = session.current_settings['n_drafters']
    total_roster_picks = session.current_settings['n_picks']
    teams              = name_auction_teams(session)
    if set(cash_share_rules) != set(teams):
        raise ValueError(f'cash_share_rules names {sorted(cash_share_rules)}, not the teams {teams}.')
    session.auction_cash_share_rules = dict(cash_share_rules)
    player_assignments = {team: [] for team in teams}
    remaining_cash     = {team: float(cash_per_team) for team in teams}
    total_nominations  = team_count * total_roster_picks
    if max_nominations is not None:
        total_nominations = min(total_nominations, max_nominations)

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
        # The same bidding without the unusable-cash shares (auction_pricing). The share exists because "Your $" is
        # scaled as if the league's cash were spent perfectly, which no team manages; what it adds to a price is what
        # that packing inefficiency costs.
        base_valuations = {team: nomination.valuations[team] - nomination.cash_shares[team] for team in teams}
        _, price_without_cash_share = run_bidding(nominator_index, nomination.opening_bid, base_valuations,
                                                  nomination.max_allowed_bids, teams)
        winner = teams[winner_index]
        player_assignments[winner].append(nomination.nominated_player_id)
        remaining_cash[winner] -= price

        ranked_valuations = sorted(nomination.valuations.values(), reverse=True)
        row = {
            'nomination':        nomination_number,
            'nominator':         teams[nominator_index],
            'player_id':         nomination.nominated_player_id,
            'player':            nomination.nominated_player_name,
            'winner':            winner,
            'price':             price,
            'winner_valuation':  nomination.valuations[winner],
            'runner_up_valuation': ranked_valuations[1] if len(ranked_valuations) > 1 else 0.0,
            'price_without_cash_share': price_without_cash_share,
            # Paid extra because of packing inefficiency: the price above what the bidding reaches without the share.
            'paid_extra_for_packing': max(0, price - price_without_cash_share),
            'seconds':           round(time.perf_counter() - nomination_started, 2),
        }
        nomination_rows.append(row)
        spend_rows.append({'nomination': nomination_number,
                           **{team: round(cash_per_team - remaining_cash[team], 2) for team in teams}})
        if report_sale is not None:
            report_sale(nomination_number, total_nominations, row)
        nominator_index = advance_nominator(nominator_index, player_assignments, teams, total_roster_picks)

    return AuctionRun(
        teams              = teams,
        player_assignments = player_assignments,
        remaining_cash     = remaining_cash,
        nomination_rows    = nomination_rows,
        spend_rows         = spend_rows,
        complete           = all(len(roster) == total_roster_picks for roster in player_assignments.values()),
    )


def score_auction(
    session
    , run: AuctionRun
) -> tuple[dict[str, float], dict[str, float]]:
    """(actual H-scores, own-view H-scores) of a finished auction's teams."""
    if not run.complete:
        raise ValueError('Only a finished auction can be scored: every roster must be full.')
    with session.lock:
        return (score_final_teams(session, run.player_assignments, run.remaining_cash, run.teams),
                score_final_teams_in_own_view(session, run.player_assignments, run.remaining_cash, run.teams))


# ── Smooth rule against simple scaling ────────────────────────────────────────

def run_rule_match(
    season: str
    , smooth_rule_softness: float
    , noise_seed: int
    , smooth_parity: int
    , short_category_names: dict[str, str]
) -> dict:
    """One complete auction between the two cash-share rules: the teams whose board index has smooth_parity (0 or 1)
    price with the smooth rule at this softness, the others with simple scaling, every team's view of the players
    drawn from noise_seed. Run with smooth_parity 0 and 1 on the same seed, the two auctions make a pair in which each
    rule drafts from every seat. Returns a JSON-able record: per team its rule, actual and own-view H-score, cash
    left, paid extra (the price above what the bidding reaches without any cash share), cumulative spend after each
    round (one sale per team) and its build in its own view (read_team_builds)."""
    cash = 200
    session = build_auction_session('Each Category', cash, None, season, smooth_rule_softness)
    session.auction_noise_seed = noise_seed
    teams = name_auction_teams(session)
    rules = {team: ('smooth' if index % 2 == smooth_parity else 'simple_scaling') for index, team in enumerate(teams)}
    run = run_auction(session, cash, None, None, rules)
    actual, own_view = score_auction(session, run)
    with session.lock:
        builds = read_team_builds(session, run, short_category_names)
    team_count = len(teams)
    round_ends = [run.spend_rows[sale - 1] for sale in range(team_count, len(run.spend_rows) + 1, team_count)]
    return {
        'season': season, 'smooth_rule_softness': smooth_rule_softness, 'noise_seed': noise_seed,
        'smooth_parity': smooth_parity,
        'teams': {team: {
            'rule':               rules[team],
            'actual':             actual[team],
            'own_view':           own_view[team],
            'cash_left':          run.remaining_cash[team],
            'paid_extra':         sum(row['paid_extra_for_packing'] for row in run.nomination_rows
                                      if row['winner'] == team),
            'spent_after_round':  [round_end[team] for round_end in round_ends],
            'punted':             builds[team].punted,
            'punted_value_share': builds[team].punted_value_share,
            'punted_category_share': builds[team].punted_category_share,
            'misfits':            builds[team].misfits,
        } for team in teams},
    }


def _run_rule_match_reporting_failure(job):
    import traceback
    logging.disable(logging.CRITICAL)
    try:
        return 'measured', run_rule_match(*job)
    except Exception:
        return 'failed', job, traceback.format_exc()


def compute_code_fingerprint() -> str:
    """A hash of everything an auction's outcome depends on -- the backend, parameters.yaml and this harness -- so a
    resumed run never mixes auctions played by different code."""
    repository = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted([*repository.glob('backend/**/*.py'), repository / 'parameters.yaml', Path(__file__).resolve()]):
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def identify_rule_match(record: dict) -> tuple:
    """The (season, softness, seed, seating) an auction record answers."""
    return (record['season'], record['smooth_rule_softness'], record['noise_seed'], record['smooth_parity'])


def run_rule_matches_in_parallel(
    seasons: list[str]
    , softnesses: list[float]
    , noise_seeds: list[int]
    , short_category_names: dict[str, str]
    , n_workers: int
    , record_path: Path
) -> tuple[list[dict], list[dict]]:
    """run_rule_match for every season, softness and noise seed, both seatings of each, one auction per job across
    n_workers processes. Each finished auction is appended to record_path (JSON lines) as it lands, stamped with the
    code fingerprint, so a run that stops loses nothing: a rerun on the same code reuses those records and plays only
    the rest, printing its progress as it goes. Records from other code are left in the file but never reused.
    Returns the records for every requested auction and the failed auctions, each {job, traceback}."""
    import multiprocessing
    fingerprint = compute_code_fingerprint()
    kept = {}
    if record_path.exists():
        with open(record_path, encoding='utf-8') as stored:
            for record in map(json.loads, stored):
                if record['code_fingerprint'] == fingerprint:
                    kept[identify_rule_match(record)] = record
    wanted = [(season, softness, seed, parity)
              for season in seasons for softness in softnesses for seed in noise_seeds for parity in (0, 1)]
    jobs = [(*key, short_category_names) for key in wanted if key not in kept]
    print(f'{len(wanted) - len(jobs)} of {len(wanted)} auctions already recorded at code {fingerprint}; '
          f'playing {len(jobs)}', flush=True)
    failures = []
    if jobs:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with multiprocessing.get_context('spawn').Pool(min(n_workers, len(jobs))) as pool, \
                open(record_path, 'a', encoding='utf-8') as stored:
            for finished, outcome in enumerate(pool.imap_unordered(_run_rule_match_reporting_failure, jobs), 1):
                if outcome[0] == 'measured':
                    record = {**outcome[1], 'code_fingerprint': fingerprint}
                    stored.write(json.dumps(record) + '\n')
                    stored.flush()
                    kept[identify_rule_match(record)] = record
                else:
                    failures.append({'job': outcome[1][:4], 'traceback': outcome[2]})
                minutes = (time.perf_counter() - started) / 60
                print(f'{finished}/{len(jobs)} auctions played ({len(failures)} failed), {minutes:.0f} min',
                      flush=True)
    return [kept[key] for key in wanted if key in kept], failures
