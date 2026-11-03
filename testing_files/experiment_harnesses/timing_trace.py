"""Where the app's time goes: every request a draft and an auction make, through the app's own routes, broken down by
the functions they spend it in.

A CallTracer wraps the app's main functions (TRACED_FUNCTIONS) for the length of a measurement and records, for every
call path (request > rank_candidates > get_h_scores > ...), the time spent and the number of calls. The wrappers are
the only change: the routes, the services and the math run as the browser makes them run.

What is measured, per format, on one season's historical stats:
  - Session build: a cold build (the agent and player-pool caches cleared; the season's data already in memory, so
    Snowflake's network time is not counted), BUILD_REPEATS times.
  - Draft: a whole 12-team snake draft. Every pick evaluates the picking team's first batch of 100 candidates (what an
    autodrafter asks for, and the wait before a person's board appears); the first team's picks also evaluate every
    batch (its whole board). Each pick takes the first batch's top candidate.
  - Auction: a whole autodraft auction ($200 a team): every nomination (the nominator's choice and every team's
    valuation), and after each round of sales the first team's full evaluate (its auction table).
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager

from .league_sessions import prepare_import_path

OBJECTIVES = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
BUILD_REPEATS = 3
CANDIDATE_BATCH_SIZE = 100          # frontend/api/draft_and_auction_session.ts
CASH_PER_TEAM = 200


def list_traced_functions():
    """(owner, attribute) of every function traced: each patched where its callers look it up."""
    prepare_import_path()
    import backend.api.routers.ranking as ranking_routes
    import backend.math.algorithm_agents as agents
    import backend.services.auction_autodraft as autodraft
    import backend.services.build_agent as build
    import backend.services.ranking as ranking
    HAgent = agents.HAgent
    return [
        (build, 'load_player_pool'), (build, 'remove_injured_players'), (build, 'apply_upsilon_adjustment'),
        (build, 'build_scoring_info'), (build, 'process_player_data'), (build, 'build_session_agent'),
        (HAgent, '__init__'), (HAgent, 'populate_default_h_scores'), (HAgent, '_run_bootstrap_pass'),
        (HAgent, 'get_h_scores'), (HAgent, 'refresh_stale_team_states'), (HAgent, 'infer_opponent_category_weights'),
        (HAgent, 'get_diff_distributions'), (HAgent, 'compute_opponent_totals'), (HAgent, '_select_starting_weights'),
        (HAgent, 'perform_iterations'), (HAgent, 'get_objective_and_gradient'), (HAgent, 'apply_volume_awareness'),
        (HAgent, 'get_objective_and_pdf_weights'),
        (agents, 'optimize_positions_all_players'), (agents, 'compute_expected_pick_tilts_and_jacobian'),
        (ranking_routes, 'rank_candidates'), (ranking, '_build_candidates'),
        (ranking_routes, 'nominate_auction_player'), (autodraft, 'select_autodrafter_nominee'),
        (autodraft, 'value_player_for_team'), (autodraft, 'read_team_h_scores'), (autodraft, 'solve_team_h_scores'),
        (autodraft, 'compute_team_dollars'),
    ]


class CallTracer:
    """Records the time of every traced call by its call path. `span(name)` opens a path's root by hand (a request).
    durations: {path tuple: [seconds, calls]}."""

    def __init__(self, functions):
        self.functions = functions
        self.stack: list[str] = []
        self.durations: dict[tuple, list] = {}
        self.originals: list = []

    def _record(self, label, call):
        self.stack.append(label)
        path = tuple(self.stack)
        started = time.perf_counter()
        try:
            return call()
        finally:
            entry = self.durations.setdefault(path, [0.0, 0])
            entry[0] += time.perf_counter() - started
            entry[1] += 1
            self.stack.pop()

    @contextmanager
    def span(self, name):
        self.stack.append(name)
        path = tuple(self.stack)
        started = time.perf_counter()
        try:
            yield
        finally:
            entry = self.durations.setdefault(path, [0.0, 0])
            entry[0] += time.perf_counter() - started
            entry[1] += 1
            self.stack.pop()

    def take(self) -> dict:
        """The durations recorded since the last take, and a fresh start."""
        durations, self.durations = self.durations, {}
        return durations

    def __enter__(self):
        for owner, attribute in self.functions:
            original = owner.__dict__[attribute] if isinstance(owner, type) else getattr(owner, attribute)
            self.originals.append((owner, attribute, original))
            label = attribute.strip('_') if attribute != '__init__' else f'{owner.__name__} init'

            def make_wrapper(function, label):
                @functools.wraps(function)
                def traced(*arguments, **keyword_arguments):
                    return self._record(label, lambda: function(*arguments, **keyword_arguments))
                return traced
            setattr(owner, attribute, make_wrapper(original, label))
        return self

    def __exit__(self, *exception):
        for owner, attribute, original in reversed(self.originals):
            setattr(owner, attribute, original)
        self.originals.clear()


def build_request(format_key: str
                  , season: str
                  , is_auction: bool) -> dict:
    prepare_import_path()
    import benchmark_helpers
    request = benchmark_helpers._build_session_request(objective=OBJECTIVES[format_key],
                                                       cash_per_team=CASH_PER_TEAM if is_auction else None)
    request['data_source']['season'] = season
    return request


def time_session_builds(tracer, format_key, season, is_auction) -> list[dict]:
    """BUILD_REPEATS cold builds through POST /sessions. Returns each build's durations and the last session's id."""
    import benchmark_helpers
    from backend.services.build_agent import clear_agent_cache, clear_v0_cache
    request = build_request(format_key, season, is_auction)
    benchmark_helpers.client.post('/sessions', json=request)          # the season's data into memory
    builds, session_id = [], None
    for _ in range(BUILD_REPEATS):
        clear_agent_cache()
        clear_v0_cache()
        tracer.take()
        with tracer.span('POST /sessions'):
            response = benchmark_helpers.client.post('/sessions', json=request)
        if response.status_code != 201:
            raise RuntimeError(f'Session build failed: {response.text[:300]}')
        session_id = response.json()['session_id']
        builds.append(tracer.take())
    return builds, session_id


def time_draft(format_key: str
               , season: str) -> dict:
    """Session builds and a whole draft. Returns {'build': [durations], 'pick': [(round, durations)], 'board':
    [(round, durations)]}."""
    import benchmark_helpers
    from backend.state.session import get_session
    with CallTracer(list_traced_functions()) as tracer:
        builds, session_id = time_session_builds(tracer, format_key, season, False)
        session = get_session(session_id)
        teams = session.current_settings['team_names']
        n_picks = session.current_settings['n_picks']
        assignments = {team: [] for team in teams}
        picks, boards = [], []
        for round_index in range(n_picks):
            order = teams if round_index % 2 == 0 else teams[::-1]
            for team in order:
                request = {'player_assignments': assignments, 'my_team_id': team, 'exclusion_list': [],
                           'candidate_offset': 0, 'candidate_limit': CANDIDATE_BATCH_SIZE}
                tracer.take()
                with tracer.span('POST /evaluate (first 100)'):
                    response = benchmark_helpers.client.post(f'/sessions/{session_id}/evaluate', json=request)
                picks.append((round_index + 1, tracer.take()))
                if response.status_code != 200:
                    raise RuntimeError(f'Evaluate failed: {response.text[:300]}')
                choice = response.json()['candidates'][0]['player_id']
                if team == teams[0]:
                    offset = 0
                    with tracer.span('POST /evaluate (whole board)'):
                        while True:
                            whole = benchmark_helpers.client.post(
                                f'/sessions/{session_id}/evaluate',
                                json={**request, 'candidate_offset': offset}).json()
                            if not whole['has_more']:
                                break
                            offset += CANDIDATE_BATCH_SIZE
                    boards.append((round_index + 1, tracer.take()))
                assignments[team] = assignments[team] + [choice]
    return {'build': builds, 'pick': picks, 'board': boards}


def time_auction(format_key: str
                 , season: str) -> dict:
    """Session builds and a whole autodraft auction. Returns {'build': [durations], 'nomination': [(round,
    durations)], 'table': [(round, durations)]}."""
    import benchmark_helpers
    from backend.state.session import get_session
    from .auction_simulation import advance_nominator, name_auction_teams, run_bidding
    with CallTracer(list_traced_functions()) as tracer:
        builds, session_id = time_session_builds(tracer, format_key, season, True)
        session = get_session(session_id)
        teams = name_auction_teams(session)
        total_roster_picks = session.current_settings['n_picks']
        assignments = {team: [] for team in teams}
        remaining_cash = {team: float(CASH_PER_TEAM) for team in teams}
        nominations, tables = [], []
        nominator_index = 0
        for sale in range(len(teams) * total_roster_picks):
            request = {'player_assignments': assignments, 'remaining_cash': remaining_cash,
                       'nominator_id': teams[nominator_index], 'nominated_player_id': None, 'opening_bid': None,
                       'valuation_team_ids': teams}
            tracer.take()
            with tracer.span('POST /auction-autodraft/nominate'):
                response = benchmark_helpers.client.post(f'/sessions/{session_id}/auction-autodraft/nominate',
                                                         json=request)
            sale_round = sale // len(teams) + 1
            nominations.append((sale_round, tracer.take()))
            if response.status_code != 200:
                raise RuntimeError(f'Nomination failed: {response.text[:300]}')
            nomination = response.json()
            winner_index, price = run_bidding(nominator_index, nomination['opening_bid'], nomination['valuations'],
                                              nomination['max_allowed_bids'], teams)
            winner = teams[winner_index]
            assignments = {**assignments, winner: assignments[winner] + [nomination['nominated_player_id']]}
            remaining_cash = {**remaining_cash, winner: remaining_cash[winner] - price}
            nominator_index = advance_nominator(nominator_index, assignments, teams, total_roster_picks)
            if (sale + 1) % len(teams) == 0 and len(assignments[teams[0]]) < total_roster_picks:
                with tracer.span('POST /evaluate (auction table)'):
                    table = benchmark_helpers.client.post(
                        f'/sessions/{session_id}/evaluate',
                        json={'player_assignments': assignments, 'my_team_id': teams[0], 'exclusion_list': [],
                              'remaining_cash': remaining_cash})
                tables.append((sale_round, tracer.take()))
                if table.status_code != 200:
                    raise RuntimeError(f'Auction evaluate failed: {table.text[:300]}')
    return {'build': builds, 'nomination': nominations, 'table': tables}
