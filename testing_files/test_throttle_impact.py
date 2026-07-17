# testing_files/test_throttle_impact.py
# Guards the position-optimiser *throttle* (an approximation that re-solves roster positions only
# for the top candidates most iterations). Unlike the regular benchmarks — which assert the exact
# algorithm with tight tolerances to catch any unintended drift — these compare the throttled result
# against the un-throttled ('exact') result and bound how far the approximation may move things.
# Higher, deliberately looser tolerances; the point is "close enough", not "identical".

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

_HSCORE_TOL = 0.05   # h-score percentage points (same bar as the regular tests)
_DOLLAR_TOL = 0.25   # auction dollar values are estimates — a looser, cents-level bar
# The throttle re-solves the top 30 candidates every iteration, so that tier — the one users
# actually draft/bid from — stays effectively exact; we validate it tightly. Candidates past 30 are
# the intentionally-approximated tier (only refreshed every 5th iter), where drift grows (~0.09
# h-score / ~$0.28) by design, so they're out of scope for these tight bars.
_TOP_N      = 30


def _evaluate(session, mode, **kwargs):
    """Run an evaluate forcing a specific throttle schedule; return {name: candidate}, [name order]."""
    session.scorer.h_agent._position_mode_override = mode
    res = rank_candidates(session=session, **kwargs)
    return {c.name: c for c in res.candidates}, [c.name for c in res.candidates]


def test_throttle_draft_close_to_exact():
    """The tiered draft throttle must not move the top-N h-score ranking or scores meaningfully."""
    session = get_session(client.post('/sessions', json=_build_session_request()).json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    top_eight  = list(session.scorer.info['G-scores'].sort_values('Total', ascending=False).head(8).index)
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    player_assignments['Team 1'] = top_eight[:4]
    player_assignments['Team 2'] = top_eight[4:]
    kwargs = dict(player_assignments=player_assignments, my_team_id='Team 1',
                  exclusion_list=top_eight[:4], remaining_cash=None)

    def run(mode):
        session.scorer.h_agent._position_mode_override = mode
        session.scorer.generic_h_scores = None
        # Neutral pass first so generic_h_scores — the ranking the throttle prioritises by — is built.
        rank_candidates(session=session,
                     player_assignments={f'Team {i + 1}': [] for i in range(n_drafters)},
                     my_team_id='Team 1', exclusion_list=[], remaining_cash=None)
        return _evaluate(session, mode, **kwargs)

    exact_by, exact_order = run('exact')
    thr_by,   thr_order   = run('tiered')

    assert thr_order[:_TOP_N] == exact_order[:_TOP_N], 'throttle changed the top-30 draft ordering'
    for name in exact_order[:_TOP_N]:
        delta = abs(thr_by[name].h_score - exact_by[name].h_score)
        assert delta <= _HSCORE_TOL, f'{name}: h-score moved {delta:.3f} (> {_HSCORE_TOL})'


def test_throttle_auction_close_to_exact():
    """The light auction throttle must keep the top-N ordering, h-scores, and dollar values close."""
    req = _build_session_request(scoring_format='Head to Head: Each Category', cash_per_team=200)
    session = get_session(client.post('/sessions', json=req).json()['session_id'])
    teams = [f'Drafter {i + 1}' for i in range(session.current_params['n_drafters'])]
    full_cash = {t: 200.0 for t in teams}

    player_assignments = {t: [] for t in teams}
    player_assignments['Drafter 1'] = ['Giannis Antetokounmpo (C,PF)']
    player_assignments['Drafter 2'] = ['Nikola Jokic (C)']
    remaining_cash = {t: 200.0 for t in teams}
    remaining_cash['Drafter 1'] = 150.0
    remaining_cash['Drafter 2'] = 150.0

    def run(mode):
        session.scorer.h_agent._position_mode_override = mode
        session.scorer.generic_h_scores = None
        # Neutral pass first so generic_h_scores is built under the same schedule.
        rank_candidates(session=session, player_assignments={t: [] for t in teams},
                     my_team_id='Drafter 1', exclusion_list=[], remaining_cash=full_cash)
        return _evaluate(session, mode, player_assignments=player_assignments, my_team_id='Drafter 1',
                         exclusion_list=['Giannis Antetokounmpo (C,PF)'], remaining_cash=remaining_cash)

    exact_by, exact_order = run('exact')
    thr_by,   thr_order   = run('light')

    assert thr_order[:_TOP_N] == exact_order[:_TOP_N], 'throttle changed the top-30 auction ordering'
    for name in exact_order[:_TOP_N]:
        delta_h = abs(thr_by[name].h_score - exact_by[name].h_score)
        assert delta_h <= _HSCORE_TOL, f'{name}: h-score moved {delta_h:.3f} (> {_HSCORE_TOL})'
        exact_av, thr_av = exact_by[name].auction_values, thr_by[name].auction_values
        if exact_av is None or thr_av is None:
            continue
        for field in ('your_dollar', 'gnrc_dollar', 'orig_dollar'):
            delta_d = abs(getattr(thr_av, field) - getattr(exact_av, field))
            assert delta_d <= _DOLLAR_TOL, f'{name}.{field}: moved ${delta_d:.2f} (> ${_DOLLAR_TOL})'
