# testing_files/test_throttle_impact.py
# Guards the position-optimiser *throttle* (an approximation that re-solves roster positions only
# for the top candidates most iterations). Unlike the regular benchmarks — which assert the exact
# algorithm with tight tolerances to catch any unintended drift — these compare the throttled result
# against the un-throttled ('exact') result and bound how far the approximation may move things.
# Higher, deliberately looser tolerances; the point is "close enough", not "identical".

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

_HSCORE_TOL = 0.05   # auction throttle h-score bar (percentage points)
_DOLLAR_TOL = 0.25   # auction dollar values are estimates — a looser, cents-level bar
# The draft throttle is a touch looser than the auction one: the punt-seed init (see the multi-start
# change in algorithm_agents) lands the mid-tier in a region where the position approximation moves
# scores a little more (up to ~0.2 pp) — still far below the multi-start gain it buys. The top-N tier
# is re-solved every iteration; candidates past N are the intentionally-approximated, out-of-scope tier.
_DRAFT_HSCORE_TOL = 0.25
_TOP_N      = 30


def _evaluate(session, mode, **kwargs):
    """Run an evaluate forcing a specific throttle schedule; return {name: candidate}, [name order]."""
    session.agent._position_mode_override = mode
    # Control the warm-start hysteresis confound: an evaluate stores the drafter's converged build and
    # the NEXT evaluate warm-starts from it (by design, in the app). Here the two arms must differ ONLY
    # by throttle schedule, so reset the in-draft state before each arm — both then warm-start
    # identically (from the frozen per-player table).
    session.agent.reset_draft_state()
    res = rank_candidates(session=session, **kwargs)
    return {c.name: c for c in res.candidates}, [c.name for c in res.candidates]


def test_throttle_draft_close_to_exact():
    """The tiered draft throttle must not move the top-N h-score ranking or scores meaningfully."""
    session = get_session(client.post('/sessions', json=_build_session_request()).json()['session_id'])
    n_drafters = session.current_params['n_drafters']
    top_eight  = list(session.agent.info['G-scores'].sort_values('Total', ascending=False).head(8).index)
    player_assignments = {f'Team {i + 1}': [] for i in range(n_drafters)}
    player_assignments['Team 1'] = top_eight[:4]
    player_assignments['Team 2'] = top_eight[4:]
    kwargs = dict(player_assignments=player_assignments, my_team_id='Team 1',
                  exclusion_list=top_eight[:4], remaining_cash=None)

    # The throttle prioritises by agent.default_h_scores, which is built once at session build — both
    # the exact and throttled passes share it, so we just force each schedule and evaluate.
    exact_by, exact_order = _evaluate(session, 'exact', **kwargs)
    thr_by,   _           = _evaluate(session, 'tiered', **kwargs)

    # Near-tied candidates can swap order under the throttle, so we check score-closeness rather than
    # exact ordering: the throttle must not move any top-N candidate's h-score beyond the tolerance.
    for name in exact_order[:_TOP_N]:
        delta = abs(thr_by[name].h_score - exact_by[name].h_score)
        assert delta <= _DRAFT_HSCORE_TOL, f'{name}: h-score moved {delta:.3f} (> {_DRAFT_HSCORE_TOL})'


def test_throttle_auction_close_to_exact():
    """The light auction throttle must keep the top-N ordering, h-scores, and dollar values close."""
    req = _build_session_request(scoring_format='Head to Head: Each Category', cash_per_team=200)
    session = get_session(client.post('/sessions', json=req).json()['session_id'])
    teams = [f'Drafter {i + 1}' for i in range(session.current_params['n_drafters'])]

    player_assignments = {t: [] for t in teams}
    player_assignments['Drafter 1'] = ['Giannis Antetokounmpo (C,PF)']
    player_assignments['Drafter 2'] = ['Nikola Jokic (C)']
    remaining_cash = {t: 200.0 for t in teams}
    remaining_cash['Drafter 1'] = 150.0
    remaining_cash['Drafter 2'] = 150.0

    def run(mode):
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
