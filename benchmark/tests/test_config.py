# benchmark/tests/test_config.py
from benchmark.config import LeagueConfig, ExperimentConfig

def test_league_config_defaults():
    c = LeagueConfig()
    assert c.n_drafters == 12
    assert c.n_starters == 9          # standard 9-cat starters, no bench in the sim
    assert c.scoring_format == 'Head to Head: Each Category'
    # 9-cat: 7 counting + 2 ratio
    assert c.selected_categories == [
        'Field Goal %', 'Free Throw %',
        'Threes', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

def test_experiment_config_grid():
    e = ExperimentConfig()
    assert e.fields == ('gscore', 'hscore')
    assert e.formats == ('Head to Head: Each Category', 'Head to Head: Most Categories')
    assert len(e.temperatures) >= 3
    assert e.seed == 12345
