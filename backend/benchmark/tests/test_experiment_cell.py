# benchmark/tests/test_experiment_cell.py
from backend.benchmark.config import LeagueConfig, ExperimentConfig
from backend.benchmark.data import load_fixture
from backend.benchmark.bootstrap import bootstrap_session
from backend.benchmark.experiment import run_matched_draft

def test_matched_draft_same_field_picks():
    averages, _ = load_fixture('backend/benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    # TINY MCTS budget: the CRN property (identical non-hero picks) is independent of
    # simulation count. ExperimentConfig()'s default mcts_simulations=200 would run
    # two full drafts x 9 hero picks x 200 sims = minutes and trip the workflow watchdog.
    exp = ExperimentConfig(mcts_simulations=6, mcts_top_k=8)
    hero_h, hero_m = run_matched_draft(info, cfg, exp,
                                       field='gscore', fmt=cfg.scoring_format,
                                       temperature=1.0, hero_seat=3, seed=7)
    # Non-hero seats identical across the two runs (common random numbers)
    for seat in hero_h:
        if seat != 3:
            assert hero_h[seat] == hero_m[seat]
    # Hero seat rosters may differ (different strategy)
    assert 3 in hero_h and 3 in hero_m
