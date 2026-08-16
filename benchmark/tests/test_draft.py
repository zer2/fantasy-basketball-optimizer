# benchmark/tests/test_draft.py
import numpy as np
from benchmark.draft import run_draft, snake_seat_order

def test_snake_order_first_two_rounds():
    # 3 drafters, 2 rounds: 0,1,2 then 2,1,0
    order = snake_seat_order(n_drafters=3, n_starters=2)
    assert order == [0, 1, 2, 2, 1, 0]

class _SeqAgent:
    """Picks the lowest-numbered unused fake player; deterministic for structure tests."""
    def __init__(self, pool): self.pool = pool
    def make_pick(self, player_assignments, seat, rng):
        taken = {p for v in player_assignments.values() for p in v}
        for p in self.pool:
            if p not in taken:
                return p

def test_run_draft_fills_all_rosters():
    # Pool must exceed total picks (12 * 9 = 108); plan's range(100) is too small.
    pool = [f'P{i}' for i in range(150)]
    agents = {s: _SeqAgent(pool) for s in range(12)}
    result = run_draft(agents, n_drafters=12, n_starters=9, rng=np.random.default_rng(0))
    assert len(result) == 12
    assert all(len(r) == 9 for r in result.values())
    # no duplicate players across the whole draft
    allp = [p for v in result.values() for p in v]
    assert len(allp) == len(set(allp)) == 12 * 9
