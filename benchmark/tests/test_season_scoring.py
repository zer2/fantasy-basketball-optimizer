# benchmark/tests/test_season_scoring.py
import numpy as np
from benchmark.evaluate import compare_categories

def test_compare_categories_counts_wins_with_negative_turnovers():
    a = {'Points': 100, 'Turnovers': 5}
    b = {'Points': 90,  'Turnovers': 8}
    # A wins Points (higher) and wins Turnovers (lower is better) => 2-0
    wins_a, wins_b, ties = compare_categories(a, b, ['Points', 'Turnovers'], negative={'Turnovers'})
    assert (wins_a, wins_b, ties) == (2, 0, 0)

def test_compare_categories_tie():
    a = {'Points': 100}; b = {'Points': 100}
    wins_a, wins_b, ties = compare_categories(a, b, ['Points'], negative=set())
    assert (wins_a, wins_b, ties) == (0, 0, 1)
