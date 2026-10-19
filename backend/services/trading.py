"""
Trade analysis and suggestion engine.

Ported from the original Streamlit trading tab, adapted to use the backend Session
pattern (session.agent, session.agent.info) instead of Streamlit session state.
"""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import pandas as pd

from backend.math.algorithm_agents import HAgent
from backend.state.session import Session
from backend.models import (
    TeamHScore, TeamTradeResult, TradeAnalyzeResponse,
    TradeSuggestion, TradeSuggestResponse, ComboParam,
    PointsTradeAnalyzeResponse, PointsTradeSuggestResponse,
    PointsTradeSuggestionModel, LeagueTradeImpactModel,
)
from backend.math.position_optimization import check_team_eligibility
from backend.services.build_agent import count_unscorable_players_as_replacement, list_unscorable_player_ids


# ── Core trade analysis ──────────────────────────────────────────────────────

def _run_points_trade_analyze(
    session: Session
    , player_assignments: dict[str, list[int]] | dict[str, list[str]]
    , my_team: str
    , their_team: str
    , my_trade: list[int] | list[str]
    , their_trade: list[int] | list[str]
    , ignore_position_check: bool = False
) -> PointsTradeAnalyzeResponse:
    from backend.points.trade import analyze_points_trade
    from backend.points.schedule import ScheduleCalendar

    if len(my_trade) == 0 or len(their_trade) == 0:
        return PointsTradeAnalyzeResponse(
            scoring_system='points',
            team_a_pre_fp=0.0,
            team_a_post_fp=0.0,
            team_a_delta_fp=0.0,
            team_a_net_delta_fp=0.0,
            team_a_streamer_boost=0.0,
            team_b_pre_fp=0.0,
            team_b_post_fp=0.0,
            team_b_delta_fp=0.0,
            team_b_net_delta_fp=0.0,
            team_b_streamer_boost=0.0,
            is_legal=False,
            rejection_reason='A trade must include at least one player from each team.',
            error='A trade must include at least one player from each team.',
            team_a_opened_spots=0,
            team_b_opened_spots=0,
        )

    team_a_sends = [int(p) for p in my_trade if p is not None]
    team_b_sends = [int(p) for p in their_trade if p is not None]
    team_a_roster = [int(p) for p in player_assignments.get(my_team, []) if p is not None]
    team_b_roster = [int(p) for p in player_assignments.get(their_team, []) if p is not None]

    # Include league rosters for league impact if provided
    all_team_rosters = {
        str(t): [int(p) for p in r if p is not None]
        for t, r in player_assignments.items()
    } if len(player_assignments) > 2 else None

    # Resolve schedule
    schedule = getattr(session, 'schedule_calendar', None) or getattr(session.agent, 'schedule', None)
    if schedule is None:
        schedule = ScheduleCalendar()

    # Resolve evaluation window
    window = (
        session.current_settings.get('evaluation_window')
        or session.current_settings.get('window')
        or session.current_settings.get('evaluation_dates')
    )
    dates = schedule.normalize_window(window)

    # Resolve per-game FP and position eligibility
    per_game_fp = getattr(session.agent, 'per_game_fp', {}) or {}
    player_positions = getattr(session.agent, 'position_eligibility', {}) or {}
    if not player_positions and session.player_registry:
        player_positions = {pid: list(ident.positions) for pid, ident in session.player_registry.items()}

    # Resolve slot configurations
    active_slots = {}
    bench_slots = 4
    ir_slots = 3
    if hasattr(session.agent, 'config') and session.agent.config:
        active_slots = dict(session.agent.config.active_slots)
        bench_slots = getattr(session.agent.config, 'bench_slots', 4)
        ir_slots = getattr(session.agent.config, 'ir_slots', 3)
    else:
        active_slots = session.current_settings.get('slot_counts', {})
        bench_slots = session.current_settings.get('bench_slots', 4)
        ir_slots = session.current_settings.get('ir_slots', 3)

    roster_size_limit = getattr(session.agent, 'total_roster_spots', None) or session.current_settings.get('n_picks', 13)
    waiver_baseline_fp = session.current_settings.get('waiver_baseline_fp_per_game', 20.0)
    streamer_multiplier = session.current_settings.get('streamer_multiplier')
    streamer_games_per_week = session.current_settings.get('streamer_games_per_week', 6.5)
    position_caps = session.current_settings.get('position_caps')

    result = analyze_points_trade(
        team_a_roster=team_a_roster,
        team_b_roster=team_b_roster,
        team_a_sends=team_a_sends,
        team_b_sends=team_b_sends,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=per_game_fp,
        active_slots=active_slots,
        roster_size_limit=roster_size_limit,
        waiver_baseline_fp_per_game=waiver_baseline_fp,
        all_team_rosters=all_team_rosters,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        position_caps=position_caps,
        streamer_multiplier=streamer_multiplier,
        streamer_games_per_week=streamer_games_per_week,
        raise_on_illegal=False,
        team_a_id=str(my_team),
        team_b_id=str(their_team),
        ignore_position_check=ignore_position_check,
    )

    league_impact_dto = None
    if result.league_impact:
        league_impact_dto = LeagueTradeImpactModel(**result.league_impact.to_dict())

    return PointsTradeAnalyzeResponse(
        scoring_system='points',
        team_a_pre_fp=result.team_a_pre_fp,
        team_a_post_fp=result.team_a_post_fp,
        team_a_delta_fp=result.team_a_delta_fp,
        team_a_net_delta_fp=result.team_a_net_delta_fp,
        team_a_streamer_boost=result.team_a_streamer_boost,
        team_b_pre_fp=result.team_b_pre_fp,
        team_b_post_fp=result.team_b_post_fp,
        team_b_delta_fp=result.team_b_delta_fp,
        team_b_net_delta_fp=result.team_b_net_delta_fp,
        team_b_streamer_boost=result.team_b_streamer_boost,
        is_legal=result.is_legal,
        rejection_reason=result.rejection_reason,
        error=result.rejection_reason if not result.is_legal else None,
        team_a_opened_spots=result.team_a_opened_spots,
        team_b_opened_spots=result.team_b_opened_spots,
        league_impact=league_impact_dto,
        your_net_delta_fp=result.team_a_net_delta_fp,
        their_net_delta_fp=result.team_b_net_delta_fp,
    )


def run_trade_analyze(
    session: Session
    , player_assignments: dict[str, list[int]]
    , my_team: str
    , their_team: str
    , my_trade: list[int]
    , their_trade: list[int]
    , position_check: bool = True
) -> TradeAnalyzeResponse | PointsTradeAnalyzeResponse:
    """Public entry point for the trade/analyze endpoint."""
    if session.current_settings.get('scoring_system') == 'points':
        return _run_points_trade_analyze(
            session=session,
            player_assignments=player_assignments,
            my_team=my_team,
            their_team=their_team,
            my_trade=my_trade,
            their_trade=their_trade,
            ignore_position_check=not position_check,
        )

    if len(my_trade) == 0 or len(their_trade) == 0:
        return TradeAnalyzeResponse(error='A trade must include at least one player from each team.')

    if abs(len(my_trade) - len(their_trade)) > 6:
        return TradeAnalyzeResponse(error="Too lopsided of a trade!")

    unscorable_in_trade = set(my_trade + their_trade) & set(list_unscorable_player_ids(session))
    if unscorable_in_trade:
        names = ', '.join(session.player_registry[player_id].name for player_id in sorted(unscorable_in_trade))
        return TradeAnalyzeResponse(
            error=f'No projection covers every category for {names}, so a trade involving them cannot be scored.')
    player_assignments = count_unscorable_players_as_replacement(session, player_assignments)

    n_iterations = session.current_settings['n_iterations']
    result = analyze_trade(session, player_assignments, my_team, my_trade, their_team, their_trade, n_iterations, position_check)

    if result is None:
        return TradeAnalyzeResponse(
            error='Trade is impermissible because it violates position structure for at least one team.',
        )

    return TradeAnalyzeResponse(
        your_team=TeamTradeResult(pre=result[1]['pre'], post=result[1]['post']),
        their_team=TeamTradeResult(pre=result[2]['pre'], post=result[2]['post']),
    )


def analyze_trade(
    session: Session
    , player_assignments: dict[str, list[int]]
    , team_1: str
    , team_1_trade: list[int]
    , team_2: str
    , team_2_trade: list[int]
    , n_iterations: int
    , position_check: bool = True
) -> Optional[dict]:
    """Compute pre/post H-scores for both teams after a trade.

    Returns None if the trade violates position structure for either team.
    Otherwise returns {1: {pre: ..., post: ...}, 2: {pre: ..., post: ...}}.
    """
    info = session.agent.info
    h_agent = session.agent

    post_trade_team_1 = [p for p in player_assignments[team_1] if p not in team_1_trade] + team_2_trade
    post_trade_team_2 = [p for p in player_assignments[team_2] if p not in team_2_trade] + team_1_trade

    # Check position eligibility for both teams
    if position_check:
        position_config = session.agent.position_config
        team_1_positions = info['Positions'].loc[post_trade_team_1]
        if not check_team_eligibility(team_1_positions, position_config):
            return None

        team_2_positions = info['Positions'].loc[post_trade_team_2]
        if not check_team_eligibility(team_2_positions, position_config):
            return None

    post_trade_assignments = player_assignments.copy()
    post_trade_assignments[team_1] = post_trade_team_1
    post_trade_assignments[team_2] = post_trade_team_2

    # Pre-trade H-scores
    res_1_pre = h_agent.get_h_scores(player_assignments, team_1, n_iterations)
    res_2_pre = h_agent.get_h_scores(player_assignments, team_2, n_iterations)

    # Post-trade H-scores
    res_1_post = h_agent.get_h_scores(post_trade_assignments, team_1, n_iterations)
    res_2_post = h_agent.get_h_scores(post_trade_assignments, team_2, n_iterations)

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


def _run_points_trade_suggest(
    session: Session
    , player_assignments: dict[str, list[int]] | dict[str, list[str]]
    , my_team: str
    , their_team: str
    , combo_params: list[ComboParam]
    , your_threshold: float
    , their_threshold: float
    , ignore_position_check: bool = False
) -> PointsTradeSuggestResponse:
    from backend.points.trade import search_win_win_points_trades
    from backend.points.schedule import ScheduleCalendar

    all_rosters = {
        str(t): [int(p) for p in r if p is not None]
        for t, r in player_assignments.items()
    }
    if their_team and their_team in player_assignments and their_team not in ("All Teams", "all", "", my_team):
        all_rosters = {
            str(my_team): all_rosters.get(str(my_team), []),
            str(their_team): all_rosters.get(str(their_team), []),
        }

    trade_formats = []
    if combo_params:
        for cp in combo_params:
            trade_formats.append(f"{cp.n_traded}v{cp.n_received}")
    if not trade_formats:
        trade_formats = ["1v1", "2v1", "2v2"]

    # Resolve schedule
    schedule = getattr(session, 'schedule_calendar', None) or getattr(session.agent, 'schedule', None)
    if schedule is None:
        schedule = ScheduleCalendar()

    # Resolve evaluation window
    window = (
        session.current_settings.get('evaluation_window')
        or session.current_settings.get('window')
        or session.current_settings.get('evaluation_dates')
    )
    dates = schedule.normalize_window(window)

    # Resolve per-game FP and position eligibility
    per_game_fp = getattr(session.agent, 'per_game_fp', {}) or {}
    player_positions = getattr(session.agent, 'position_eligibility', {}) or {}
    if not player_positions and session.player_registry:
        player_positions = {pid: list(ident.positions) for pid, ident in session.player_registry.items()}

    # Resolve slot configurations
    active_slots = {}
    bench_slots = 4
    ir_slots = 3
    if hasattr(session.agent, 'config') and session.agent.config:
        active_slots = dict(session.agent.config.active_slots)
        bench_slots = getattr(session.agent.config, 'bench_slots', 4)
        ir_slots = getattr(session.agent.config, 'ir_slots', 3)
    else:
        active_slots = session.current_settings.get('slot_counts', {})
        bench_slots = session.current_settings.get('bench_slots', 4)
        ir_slots = session.current_settings.get('ir_slots', 3)

    roster_size_limit = getattr(session.agent, 'total_roster_spots', None) or session.current_settings.get('n_picks', 13)
    waiver_baseline_fp = session.current_settings.get('waiver_baseline_fp_per_game', 20.0)
    streamer_multiplier = session.current_settings.get('streamer_multiplier')
    streamer_games_per_week = session.current_settings.get('streamer_games_per_week', 6.5)
    position_caps = session.current_settings.get('position_caps')

    suggestions = search_win_win_points_trades(
        user_team_id=str(my_team),
        all_team_rosters=all_rosters,
        schedule=schedule,
        evaluation_window=dates,
        player_positions=player_positions,
        player_fp_per_game=per_game_fp,
        active_slots=active_slots,
        roster_size_limit=roster_size_limit,
        min_user_gain=float(your_threshold),
        min_partner_gain=float(their_threshold),
        trade_formats=trade_formats,
        waiver_baseline_fp_per_game=waiver_baseline_fp,
        bench_slots=bench_slots,
        ir_slots=ir_slots,
        position_caps=position_caps,
        streamer_multiplier=streamer_multiplier,
        streamer_games_per_week=streamer_games_per_week,
        max_suggestions=50,
        ignore_position_check=ignore_position_check,
    )

    suggestion_models = []
    for s in suggestions:
        an_dto = None
        if s.analysis:
            li_dto = LeagueTradeImpactModel(**s.analysis.league_impact.to_dict()) if s.analysis.league_impact else None
            an_dto = PointsTradeAnalyzeResponse(
                scoring_system='points',
                team_a_pre_fp=s.analysis.team_a_pre_fp,
                team_a_post_fp=s.analysis.team_a_post_fp,
                team_a_delta_fp=s.analysis.team_a_delta_fp,
                team_a_net_delta_fp=s.analysis.team_a_net_delta_fp,
                team_a_streamer_boost=s.analysis.team_a_streamer_boost,
                team_b_pre_fp=s.analysis.team_b_pre_fp,
                team_b_post_fp=s.analysis.team_b_post_fp,
                team_b_delta_fp=s.analysis.team_b_delta_fp,
                team_b_net_delta_fp=s.analysis.team_b_net_delta_fp,
                team_b_streamer_boost=s.analysis.team_b_streamer_boost,
                is_legal=s.analysis.is_legal,
                rejection_reason=s.analysis.rejection_reason,
                error=s.analysis.rejection_reason if not s.analysis.is_legal else None,
                team_a_opened_spots=s.analysis.team_a_opened_spots,
                team_b_opened_spots=s.analysis.team_b_opened_spots,
                league_impact=li_dto,
                your_net_delta_fp=s.analysis.team_a_net_delta_fp,
                their_net_delta_fp=s.analysis.team_b_net_delta_fp,
            )
        suggestion_models.append(
            PointsTradeSuggestionModel(
                partner_team_id=s.partner_team_id,
                user_sends=s.user_sends,
                partner_sends=s.partner_sends,
                user_gain=s.user_gain,
                partner_gain=s.partner_gain,
                total_gain=s.total_gain,
                trade_format=s.trade_format,
                send=s.send,
                receive=s.receive,
                user_delta_fp=s.user_delta_fp,
                partner_delta_fp=s.partner_delta_fp,
                net_delta_fp=s.user_delta_fp if s.user_delta_fp is not None else s.user_gain,
                analysis=an_dto,
            )
        )

    return PointsTradeSuggestResponse(
        scoring_system='points',
        suggestions=suggestion_models,
    )


def run_trade_suggest(
    session: Session
    , player_assignments: dict[str, list[int]]
    , my_team: str
    , their_team: str
    , combo_params: list[ComboParam]
    , your_threshold: float
    , their_threshold: float
    , position_check: bool = True
) -> TradeSuggestResponse | PointsTradeSuggestResponse:
    """Public entry point for the trade/suggest endpoint."""
    if session.current_settings.get('scoring_system') == 'points':
        return _run_points_trade_suggest(
            session=session,
            player_assignments=player_assignments,
            my_team=my_team,
            their_team=their_team,
            combo_params=combo_params,
            your_threshold=your_threshold,
            their_threshold=their_threshold,
            ignore_position_check=not position_check,
        )

    # A rostered player the league's categories cannot score is replacement level, not a trade piece.
    player_assignments = count_unscorable_players_as_replacement(session, player_assignments)
    # Step 1: the neutral-board H-score ranking, computed once at build time
    # (populate_default_h_scores) and shared with auction anchoring and the throttle.
    general_values = session.agent.default_h_scores

    n_picks = session.current_settings['n_picks']
    n_drafters = session.current_settings['n_drafters']
    total_players = n_picks * n_drafters
    replacement_value = float(general_values.iloc[min(total_players, len(general_values) - 1)])

    # Step 2: use all players as candidates (candidate filter disabled)
    my_candidates    = player_assignments[my_team]
    their_candidates = player_assignments[their_team]

    # Step 2b: pre-extract x_scores for the two teams as numpy arrays.
    # Avoids per-combo pandas .loc[].sum() overhead in _make_combo_df.
    my_x_numpy    = session.agent.x_scores.loc[my_candidates].to_numpy()
    their_x_numpy = session.agent.x_scores.loc[their_candidates].to_numpy()
    my_name_to_index    = {name: i for i, name in enumerate(my_candidates)}
    their_name_to_index = {name: i for i, name in enumerate(their_candidates)}

    # Step 3: generate all combos across param sets
    all_combo_frames = []
    for cp in combo_params:
        combos = _get_cross_combos(
            cp.n_traded, cp.n_received,
            my_candidates, their_candidates,
            general_values, replacement_value,
            cp.threshold, cp.searched_threshold,
        )
        if len(combos) > 0:
            all_combo_frames.append(combos)

    if len(all_combo_frames) == 0:
        return TradeSuggestResponse(suggestions=[])

    all_combos = pd.concat(all_combo_frames, ignore_index=True)

    # Step 3: evaluate all combos vectorized
    df = _make_combo_df(
        session.agent
        , get_trade_context(session, player_assignments, my_team, their_team)
        , all_combos
        , my_candidates
        , their_candidates
        , my_x_numpy
        , their_x_numpy
        , my_name_to_index
        , their_name_to_index
    )

    # Step 4: filter by score thresholds
    mask = (df['Your Score'] > your_threshold) & (df['Their Score'] > their_threshold)
    df = df[mask]

    # Step 5: position check only on the set that passed the thresholds. A roster's eligibility depends only on its
    # players' position eligibilities, so it is decided once per distinct mix: one linear assignment per trade was
    # 41 of the 54 s a 3-for-3 search at the -2% floor took (2026-10-13), and many trades share a mix.
    sends, receives = list(df['Send']), list(df['Receive'])
    your_scores, their_scores = df['Your Score'].to_numpy(), df['Their Score'].to_numpy()
    if position_check and len(df) > 0:
        position_config = session.agent.position_config
        # Each player's eligible positions (a list, e.g. ['PG', 'SG']) as a sorted tuple, so a roster's mix can key
        # the memo.
        position_by_player = {player: tuple(sorted(eligible))
                              for player, eligible in session.agent.info['Positions'].items()}
        eligibility_by_mix: dict[tuple, bool] = {}

        def is_eligible_roster(roster: list) -> bool:
            mix = tuple(sorted(position_by_player[p] for p in roster))
            if mix not in eligibility_by_mix:
                eligibility_by_mix[mix] = check_team_eligibility(list(mix), position_config)
            return eligibility_by_mix[mix]

        my_roster, their_roster = player_assignments[my_team], player_assignments[their_team]
        keep = [is_eligible_roster([p for p in my_roster if p not in sent] + received)
                and is_eligible_roster([p for p in their_roster if p not in received] + sent)
                for sent, received in zip(sends, receives)]
    else:
        keep = [True] * len(sends)

    suggestions = [
        TradeSuggestion(
            their_team=their_team,
            send=[int(player_id) for player_id in sent],
            receive=[int(player_id) for player_id in received],
            your_score=round(float(your_score), 4),
            their_score=round(float(their_score), 4),
        )
        for sent, received, your_score, their_score, kept in zip(sends, receives, your_scores, their_scores, keep)
        if kept
    ]

    return TradeSuggestResponse(suggestions=suggestions)


def run_trade_suggest_across_teams(
    session: Session
    , player_assignments: dict[str, list[int]]
    , my_team: str
    , their_teams: list[str]
    , combo_params: list[ComboParam]
    , your_threshold: float
    , their_threshold: float
    , position_check: bool
) -> TradeSuggestResponse:
    """run_trade_suggest for every team in their_teams, the suggestions ranked together by your score (highest first)."""
    if session.current_settings.get('scoring_system') == 'points':
        from backend.models import PointsTradeSuggestResponse
        all_suggs = []
        for their_team in their_teams:
            resp = run_trade_suggest(
                session, player_assignments, my_team, their_team, combo_params, your_threshold, their_threshold,
                position_check)
            if hasattr(resp, 'suggestions'):
                all_suggs.extend(resp.suggestions)
        all_suggs.sort(key=lambda s: -(s.net_delta_fp if getattr(s, 'net_delta_fp', None) is not None else (s.user_delta_fp or s.user_gain or 0.0)))
        return PointsTradeSuggestResponse(suggestions=all_suggs)

    suggestions = [
        suggestion
        for their_team in their_teams
        for suggestion in run_trade_suggest(
            session, player_assignments, my_team, their_team, combo_params, your_threshold, their_threshold,
            position_check).suggestions
    ]
    suggestions.sort(key=lambda suggestion: -suggestion.your_score)
    return TradeSuggestResponse(suggestions=suggestions)


# ── Combination generation ───────────────────────────────────────────────────

def _get_cross_combos(
    n: int
    , m: int
    , my_players: list[str]
    , their_players: list[str]
    , general_values: pd.Series
    , replacement_value: float
    , value_threshold: Optional[float]
    , searched_value_threshold: Optional[float]
) -> pd.DataFrame:
    """Generate filtered cross-product of (n-from-mine, m-from-theirs) combos.

    Keeps the combos whose absolute general-value differential is within value_threshold (every one when None) and
    beyond searched_value_threshold (every one when None): an earlier batch searched within it. Both in percentage points.
    Rows come in the order itertools.product would give them (mine outer, theirs inner). The gap of every pairing is one
    broadcast subtraction, and only the pairings that meet the window become rows: a 3-for-3 search builds ~27,000
    pairings per team, and making each a Python tuple and then a DataFrame row was most of its bookkeeping (2026-10-13).
    """
    my_pool    = [p for p in my_players    if p in general_values.index]
    their_pool = [p for p in their_players if p in general_values.index]
    my_combos    = _list_index_combinations(len(my_pool), n)       # (A, n) positions in my_pool
    their_combos = _list_index_combinations(len(their_pool), m)    # (B, m)

    if len(my_combos) == 0 or len(their_combos) == 0:
        return pd.DataFrame(columns=['My Trade', 'My Value', 'Their Trade', 'Their Value'])

    # Summed one player at a time, left to right, as Python's sum over a combination would: identical floats.
    my_values    = _sum_columns_in_order(general_values.loc[my_pool].to_numpy()[my_combos])
    their_values = _sum_columns_in_order(general_values.loc[their_pool].to_numpy()[their_combos])
    if n != m:
        my_values = my_values + replacement_value * (m - n)

    gap = np.abs(my_values[:, np.newaxis] - their_values[np.newaxis, :])   # (A, B)
    meets = np.ones(gap.shape, dtype=bool)
    if value_threshold is not None:
        meets &= gap <= value_threshold / 100  # thresholds are in percentage terms
    if searched_value_threshold is not None:
        meets &= gap > searched_value_threshold / 100
    my_rows, their_rows = np.nonzero(meets)

    return pd.DataFrame({
        'My Trade':    [[my_pool[i]    for i in my_combos[row]]    for row in my_rows],
        'My Value':    my_values[my_rows],
        'Their Trade': [[their_pool[i] for i in their_combos[row]] for row in their_rows],
        'Their Value': their_values[their_rows],
    })


def _list_index_combinations(
    pool_size: int
    , combination_size: int
) -> np.ndarray:
    """Every combination_size-subset of range(pool_size), one per row, in itertools.combinations order."""
    return np.array(list(itertools.combinations(range(pool_size), combination_size)), dtype=int) \
        .reshape(-1, combination_size)


def _sum_columns_in_order(
    values: np.ndarray
) -> np.ndarray:
    """Row sums of a (rows, k) array, added column by column from the left."""
    total = values[:, 0].copy()
    for column in range(1, values.shape[1]):
        total = total + values[:, column]
    return total


def _make_combo_df(
    h_agent: HAgent
    , ctx: dict
    , all_combos: pd.DataFrame
    , my_roster: list[int]
    , their_roster: list[int]
    , my_x_numpy: np.ndarray
    , their_x_numpy: np.ndarray
    , my_name_to_index: dict[str, int]
    , their_name_to_index: dict[str, int]
) -> pd.DataFrame:
    """Evaluate every trade combo in one vectorized pass and return sorted by Your Score.

    For each combo, trade_delta = received_sum - sent_sum shifts diff_means on ALL
    opponent columns (my roster improved vs every opponent), with an extra shift on
    the counterparty column (their roster also weakened). Both perspectives are
    computed in a single batched call to compute_h_scores_batched. ctx is the pair's trade context (get_trade_context).
    my_roster and their_roster are the rosters the X-score rows were taken from, in the same order: the name-to-index
    maps point into them, and a roster can hold one id twice (every unscorable player counts as the replacement player).
    """

    n_combos = len(all_combos)
    if n_combos == 0:
        return pd.DataFrame(columns=['Send', 'Receive', 'Your Score', 'Their Score'])

    # Sum x_scores for each combo's sent / received players: (n_combos, n_categories)
    # Uses numpy integer indexing to avoid per-combo pandas overhead.
    # When all combos have the same size, builds a uniform 2D index array for a
    # fully vectorized sum. When sizes are mixed (asymmetric), falls back to a
    # per-row loop that still uses numpy rather than pandas.
    # The same row indices serve every per-player quantity summed over the traded players: the X-scores here and,
    # below, the percentage categories' volume units (which once ran a Python loop per combo, 2026-10-13).
    sent_sizes     = set(len(sent)     for sent     in all_combos['My Trade'])
    received_sizes = set(len(received) for received in all_combos['Their Trade'])
    sent_index_lists     = [[my_name_to_index[p]    for p in sent]     for sent     in all_combos['My Trade']]
    received_index_lists = [[their_name_to_index[p] for p in received] for received in all_combos['Their Trade']]

    if len(sent_sizes) == 1 and len(received_sizes) == 1:
        sent_index_array     = np.array(sent_index_lists)
        received_index_array = np.array(received_index_lists)

        def sum_traded_rows(my_rows: np.ndarray, their_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Per combo, the sent players' rows summed and the received players' rows summed."""
            return my_rows[sent_index_array].sum(axis=1), their_rows[received_index_array].sum(axis=1)
    else:
        def sum_traded_rows(my_rows: np.ndarray, their_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            """Per combo, the sent players' rows summed and the received players' rows summed."""
            return (np.array([my_rows[index].sum(axis=0)    for index in sent_index_lists]),
                    np.array([their_rows[index].sum(axis=0) for index in received_index_lists]))

    sent_sums, received_sums = sum_traded_rows(my_x_numpy, their_x_numpy)
    trade_deltas = received_sums - sent_sums  # (n_combos, n_categories)

    # Build all post diff_means at once: (n_combos, n_categories, n_drafters-1)
    post_diff_means_my    = ctx['baseline_diff_means_my']    + trade_deltas[:, :, np.newaxis]
    post_diff_means_their = ctx['baseline_diff_means_their'] - trade_deltas[:, :, np.newaxis]

    post_diff_means_my   [:, :, ctx['their_col_in_my_view']]    += trade_deltas
    post_diff_means_their[:, :, ctx['my_col_in_their_view']]    -= trade_deltas

    post_volume_my = post_volume_their = None
    if ctx['volume_my'] is not None:
        # The players exchanged move each side's percentage totals and season volume too.
        index = h_agent.volume_category_indices
        my_units_numpy    = h_agent.player_volume_units.loc[my_roster].to_numpy()
        their_units_numpy = h_agent.player_volume_units.loc[their_roster].to_numpy()
        sent_units, received_units = sum_traded_rows(my_units_numpy, their_units_numpy)
        unit_deltas = received_units - sent_units                  # (n_combos, R): my side gains these
        x_deltas    = trade_deltas[:, index]

        def shift_volume_context(context, x_shift, unit_shift, counterparty_column, counterparty_shift):
            opponent_units = np.repeat(context['opponent_units'], n_combos, axis=0)
            opponent_units[:, :, counterparty_column] += counterparty_shift
            return {'my_x_total':     context['my_x_total'] + x_shift,
                    'my_units':       context['my_units'] + unit_shift,
                    'opponent_units': opponent_units}

        post_volume_my    = shift_volume_context(ctx['volume_my'],    x_deltas,  unit_deltas,
                                                 ctx['their_col_in_my_view'], -unit_deltas)
        post_volume_their = shift_volume_context(ctx['volume_their'], -x_deltas, -unit_deltas,
                                                 ctx['my_col_in_their_view'],  unit_deltas)

    post_my_h_scores    = h_agent.compute_h_scores_batched(post_diff_means_my,    ctx['diff_vars_my'],
                                                           volume_context=post_volume_my)
    post_their_h_scores = h_agent.compute_h_scores_batched(post_diff_means_their, ctx['diff_vars_their'],
                                                           volume_context=post_volume_their)

    df = pd.DataFrame({
        'Send':        list(all_combos['My Trade']),
        'Receive':     list(all_combos['Their Trade']),
        'Your Score':  post_my_h_scores    - ctx['pre_my_h'],
        'Their Score': post_their_h_scores - ctx['pre_their_h'],
    })
    return df.sort_values('Your Score', ascending=False)


# ── Fast trade evaluation ────────────────────────────────────────────────────

_TRADE_CONTEXT_CACHE_MAX = 64   # a league's worth of counterparties for a few boards


def get_trade_context(
    session: Session
    , player_assignments: dict[str, list[int]]
    , my_team: str
    , their_team: str
) -> dict:
    """The trade context for a team pair (_build_trade_context), kept on the session. A trade search asks for the same
    pair once per value-gap band, and rebuilding it each time was a quarter of the search (2026-10-13 profile).
    Keyed by both teams and every roster in order (the order sets the opponent columns); an entry built from another
    agent than the session's current one is rebuilt. The context is only read, never modified, by the scoring."""
    key = (my_team, their_team, tuple((team, tuple(roster)) for team, roster in player_assignments.items()))
    cache = session.trade_context_cache
    cached = cache.get(key)
    if cached is not None and cached[0] is session.agent:
        cache.move_to_end(key)
        return cached[1]
    context = _build_trade_context(session.agent, player_assignments, my_team, their_team)
    cache[key] = (session.agent, context)
    cache.move_to_end(key)
    while len(cache) > _TRADE_CONTEXT_CACHE_MAX:
        cache.popitem(last=False)
    return context

def _build_trade_context(
    h_agent: HAgent
    , player_assignments: dict[str, list[int]]
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
    x_scores_available = h_agent.x_scores[
        ~h_agent.x_scores.index.isin(players_chosen)
        & h_agent.x_scores.index.isin(h_agent.positions.index)
    ]

    # Pre-compute each team's raw stat sum once.
    team_sums = {
        team: np.array(h_agent.x_scores.loc[[p for p in roster if p == p]].sum(axis=0))
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
            .reshape(1, h_agent.n_categories, 1)
            for team in team_names if team != drafter
        ]).T  # (1, n_categories, n_drafters-1)

        return (
            team_sums[drafter].reshape(1, h_agent.n_categories, 1)
            - other_sums.reshape(1, h_agent.n_categories, n_drafters - 1)
        )

    baseline_diff_means_my    = build_baseline_diff_means(my_team)
    baseline_diff_means_their = build_baseline_diff_means(their_team)

    # Percentage categories are scored on rates (see HAgent.apply_volume_awareness): each perspective needs
    # its own absolute percentage totals and every team's season volume, unfilled slots at the generic level.
    volume_my = volume_their = None
    if h_agent.volume_categories:
        generic_units, generic_x = h_agent.compute_generic_volume_levels(x_scores_available, players_chosen)
        index = h_agent.volume_category_indices

        def season_units(roster):
            roster = [p for p in roster if p == p]
            return (h_agent.player_volume_units.loc[roster].sum().to_numpy()
                    + max(h_agent.n_picks - len(roster), 0) * generic_units)

        team_units = {team: season_units(roster) for team, roster in player_assignments.items()}

        def build_volume_context(drafter: str) -> dict:
            roster = [p for p in player_assignments[drafter] if p == p]
            others = [team for team in team_names if team != drafter]
            return {
                'my_x_total':     (team_sums[drafter][index]
                                   + max(h_agent.n_picks - len(roster), 0) * generic_x).reshape(1, -1),
                'my_units':       team_units[drafter].reshape(1, -1),
                'opponent_units': np.stack([team_units[team] for team in others], axis=1)
                                  .reshape(1, len(index), len(others)),
            }

        volume_my, volume_their = build_volume_context(my_team), build_volume_context(their_team)

    # diff_vars depend only on team sizes — constant for equal-size trades.
    diff_vars_my = np.vstack([
        h_agent.get_diff_var(len([p for p in player_assignments[team] if p == p]))
        for team in team_names if team != my_team
    ]).T.reshape(1, h_agent.n_categories, n_drafters - 1)

    diff_vars_their = np.vstack([
        h_agent.get_diff_var(len([p for p in player_assignments[team] if p == p]))
        for team in team_names if team != their_team
    ]).T.reshape(1, h_agent.n_categories, n_drafters - 1)

    # Pre-trade H-scores — constant across all combos, compute once.
    my_players    = [p for p in player_assignments[my_team]    if p == p]
    their_players = [p for p in player_assignments[their_team] if p == p]

    pre_my_h = h_agent.compute_h_score_from_diff_means(
        diff_means     = baseline_diff_means_my,
        diff_vars      = diff_vars_my,
        volume_context = volume_my,
    )
    pre_their_h = h_agent.compute_h_score_from_diff_means(
        diff_means     = baseline_diff_means_their,
        diff_vars      = diff_vars_their,
        volume_context = volume_their,
    )

    # Column indices for the two trading teams in each perspective's diff_means.
    their_col_in_my_view = [t for t in team_names if t != my_team].index(their_team)
    my_col_in_their_view = [t for t in team_names if t != their_team].index(my_team)

    return {
        'baseline_diff_means_my':      baseline_diff_means_my,
        'baseline_diff_means_their':   baseline_diff_means_their,
        'diff_vars_my':                diff_vars_my,
        'diff_vars_their':             diff_vars_their,
        'pre_my_h':                    pre_my_h,
        'pre_their_h':                 pre_their_h,
        'their_col_in_my_view':        their_col_in_my_view,
        'my_col_in_their_view':        my_col_in_their_view,
        'volume_my':                   volume_my,
        'volume_their':                volume_their,
    }
