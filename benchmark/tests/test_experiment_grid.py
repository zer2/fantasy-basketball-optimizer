# benchmark/tests/test_experiment_grid.py
from dataclasses import replace
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.experiment import run_experiment

def test_grid_smoke_produces_rows():
    small = ExperimentConfig(fields=('gscore',),
                             formats=('Head to Head: Each Category',),
                             temperatures=(1.0,),
                             n_drafts=2, n_season_sims=10,
                             mcts_simulations=8, mcts_top_k=8)
    results = run_experiment(small, LeagueConfig(), 'benchmark/fixtures/nba_2025-26.parquet')
    row = results[('gscore', 'Head to Head: Each Category', 1.0)]
    assert 'hscore_hero' in row and 'mcts_hero' in row
    assert 'EC' in row['hscore_hero'] and 'EC' in row['mcts_hero']
    assert 0.0 <= row['mcts_hero']['EC'] <= 1.0
