# testing_files/test_and_benchmark_trading.py
# Trade suggestion speed benchmarks.
# Uses 2024-25 historical season data with default parameters from parameters.yaml.
# Rosters are a 2024-25 snapshot of an H-score snake draft; the live frontend default
# (default_season_rosters.ts) tracks current-season data and is regenerated via
# testing_files/generate_default_season_rosters.py.
#
# The suggest pipeline has three main cost centres:
#   1. _identify_trade_candidates — one get_h_scores call per player on each team
#   2. _get_general_values        — reads the agent's prebuilt default baseline (no evaluate)
#   3. _make_combo_df             — four get_h_scores calls per surviving combo
#
# Covers:
#   - 1-for-1 trades between Drafter 1 and Drafter 2 (most common user scenario)
#   - 2-for-2 trades (more combos, tests scaling)
#   - 1-for-2 and 2-for-1 (asymmetric, exercises replacement-value adjustment)

import cProfile
import io
import pstats
import time
import pytest

from benchmark_helpers import (
    client
    , _SEASON
    , _build_session_request
)
from backend.state.session import get_session
from backend.services.trading import run_trade_suggest
from backend.models import ComboParam

# 2024-25 snapshot of an H-score snake draft used by the trading benchmarks below.
# EC scoring, 12 drafters, 13 picks, snake-drafted by H-score rank.
_DEFAULT_SEASON_ROSTERS: dict[str, list[str]] = {
    'Drafter 1': [
        'Shai Gilgeous-Alexander (PG,SG)',
        'Brook Lopez (C)',
        'Bam Adebayo (C,PF)',
        'Rudy Gobert (C)',
        'Michael Porter Jr. (SF,PF)',
        'Isaiah Hartenstein (C)',
        'Miles Bridges (SF,PF)',
        'Daniel Gafford (C,PF)',
        'Andrew Nembhard (PG,SG)',
        'Buddy Hield (SG,SF)',
        'Mike Conley (PG)',
        'Gary Trent Jr. (PG,SG,SF)',
        'Justin Champagnie (SG,SF)',
    ],
    'Drafter 2': [
        'Nikola Jokic (C)',
        'Pascal Siakam (C,SF,PF)',
        'DeMar DeRozan (SF,PF)',
        'Onyeka Okongwu (C,PF)',
        'Scottie Barnes (PG,SG,SF,PF)',
        'Tyrese Maxey (PG,SG)',
        'Russell Westbrook (PG,SG)',
        'Yves Missi (C)',
        'Malik Monk (PG,SG,SF)',
        'Ty Jerome (PG,SG)',
        'Derrick Jones Jr. (SF,PF)',
        'Kyshawn George (SG,SF)',
        'Andrew Wiggins (SF,PF)',
    ],
    'Drafter 3': [
        'Tyrese Haliburton (PG,SG)',
        'Derrick White (PG,SG)',
        'Jalen Williams (C,SG,SF,PF)',
        'Jamal Murray (PG,SG)',
        "De'Aaron Fox (PG,SG)",
        'Jalen Green (PG,SG)',
        'Cason Wallace (PG,SG)',
        'Naji Marshall (SG,SF,PF)',
        'Norman Powell (SG,SF)',
        'Ja Morant (PG)',
        'RJ Barrett (SG,SF,PF)',
        'Haywood Highsmith (SF,PF)',
        'Paolo Banchero (SF,PF)',
    ],
    'Drafter 4': [
        'Karl-Anthony Towns (C,PF)',
        'Devin Booker (PG,SG)',
        'Chris Paul (PG)',
        'Jalen Duren (C)',
        'Coby White (PG,SG)',
        'Kyrie Irving (PG,SG)',
        'Scotty Pippen Jr. (PG,SG)',
        'Goga Bitadze (C)',
        'P.J. Washington (SF,PF)',
        'Santi Aldama (C,PF)',
        'Brandin Podziemski (PG,SG)',
        "Kel'el Ware (C,PF)",
        'LaMelo Ball (PG,SG)',
    ],
    'Drafter 5': [
        'James Harden (PG,SG)',
        'Cade Cunningham (PG,SG)',
        'Jaren Jackson Jr. (C,PF)',
        'Anthony Davis (C,PF)',
        'Zach LaVine (SG,SF)',
        'Jakob Poeltl (C)',
        'Trey Murphy III (SG,SF,PF)',
        'Kelly Oubre Jr. (SG,SF)',
        'Isaiah Stewart (C,PF)',
        'Keldon Johnson (SG,SF,PF)',
        'Peyton Watson (SF,PF)',
        'Taurean Prince (SG,SF)',
        'Donte DiVincenzo (PG,SG,SF)',
    ],
    'Drafter 6': [
        'Giannis Antetokounmpo (C,PF)',
        'Jarrett Allen (C)',
        'Kevin Durant (SF,PF)',
        'Alperen Sengun (C)',
        'Franz Wagner (SF,PF)',
        'Jalen Brunson (PG)',
        'Jaylen Brown (SG,SF)',
        'Draymond Green (C,PF)',
        'Harrison Barnes (SF,PF)',
        'Dillon Brooks (SG,SF,PF)',
        'Bilal Coulibaly (SG,SF)',
        'Caris LeVert (SG,SF)',
        'Al Horford (C,PF)',
    ],
    'Drafter 7': [
        'Josh Hart (SG,SF,PF)',
        'Evan Mobley (C,PF)',
        'Trae Young (PG)',
        'Desmond Bane (SG,SF)',
        'Luka Doncic (PG,SG)',
        'Deni Avdija (SF,PF)',
        'Tari Eason (SF,PF)',
        'Julius Randle (C,PF)',
        'Quentin Grimes (SG,SF)',
        'Bennedict Mathurin (SG,SF)',
        'Kentavious Caldwell-Pope (SG,SF)',
        'Kristaps Porzingis (C,PF)',
        'Jonathan Isaac (SF,PF)',
    ],
    'Drafter 8': [
        'Jayson Tatum (SF,PF)',
        'Domantas Sabonis (C)',
        'Amen Thompson (PG,SG,SF,PF)',
        'Donovan Mitchell (PG,SG)',
        'Payton Pritchard (PG,SG)',
        'Jimmy Butler (SF,PF)',
        'Nic Claxton (C)',
        'Luguentz Dort (SG,SF)',
        'Spencer Dinwiddie (PG,SG)',
        'Julian Champagnie (SF,PF)',
        'Stephon Castle (PG,SG)',
        'Alex Caruso (SG,SF)',
        'CJ McCollum (PG,SG)',
    ],
    'Drafter 9': [
        'Dyson Daniels (PG,SG,SF)',
        'Nikola Vucevic (C,PF)',
        'Josh Giddey (PG,SG,SF)',
        'Austin Reaves (PG,SG)',
        'Toumani Camara (SG,SF,PF)',
        'Keegan Murray (SF,PF)',
        'Tyus Jones (PG)',
        'Malik Beasley (SG,SF)',
        'Jrue Holiday (PG,SG)',
        'Bub Carrington (PG,SG)',
        'Shaedon Sharpe (SG,SF)',
        'Anthony Black (PG,SG)',
        'Kevin Porter Jr. (PG)',
    ],
    'Drafter 10': [
        'Ivica Zubac (C)',
        'Tyler Herro (PG,SG)',
        'Christian Braun (SG,SF)',
        'Myles Turner (C)',
        'Walker Kessler (C)',
        'Naz Reid (C,PF)',
        'Jonas Valanciunas (C)',
        'Cameron Johnson (SF,PF)',
        'Ausar Thompson (SF,PF)',
        'Klay Thompson (SG,SF)',
        'Fred VanVleet (PG)',
        'Aaron Wiggins (SG,SF)',
        'Bradley Beal (PG,SG,SF)',
    ],
    'Drafter 11': [
        'Stephen Curry (PG,SG)',
        'Victor Wembanyama (C)',
        'O.G. Anunoby (SF,PF)',
        'Damian Lillard (PG)',
        'Tobias Harris (SF,PF)',
        'Keon Ellis (PG,SG,SF)',
        'T.J. McConnell (PG)',
        'Kris Dunn (PG,SG)',
        'Alex Sarr (C,PF)',
        "Royce O'Neale (SF,PF)",
        "De'Andre Hunter (SF,PF)",
        'Davion Mitchell (PG,SG)',
        'Isaiah Joe (PG,SG)',
    ],
    'Drafter 12': [
        'Anthony Edwards (SG,SF)',
        'LeBron James (SF,PF)',
        'Jaden McDaniels (SF,PF)',
        'Darius Garland (PG)',
        'Jordan Poole (PG,SG)',
        'Mikal Bridges (SG,SF,PF)',
        'Devin Vassell (SG,SF)',
        'Anfernee Simons (PG,SG)',
        'Dennis Schroder (PG,SG)',
        'Jalen Johnson (SF,PF)',
        'Zach Edey (C)',
        'Obi Toppin (PF)',
        'Collin Sexton (PG,SG)',
    ],
}

# (label, combo_params, your_threshold, their_threshold)
# threshold on ComboParam is the general-value filter (percentage points);
#   mirrors the frontend default of 3 (±3% of general H-score).
# your_threshold / their_threshold are the post-evaluate H-score diff filters (0–1 scale);
#   mirrors the frontend defaults of 0% and -0.2%, divided by 100.
_SUGGEST_CONFIGS = [
    pytest.param(
        ('1-for-1', [ComboParam(n_traded=1, n_received=1, threshold=3.0)], 0.0, -0.002),
        id='1v1',
    ),
    pytest.param(
        ('2-for-2', [ComboParam(n_traded=2, n_received=2, threshold=3.0)], 0.0, -0.002),
        id='2v2',
    ),
    pytest.param(
        ('1-for-2 + 2-for-1', [
            ComboParam(n_traded=1, n_received=2, threshold=3.0),
            ComboParam(n_traded=2, n_received=1, threshold=3.0),
        ], 0.0, -0.002),
        id='asymmetric',
    ),
    pytest.param(
        ('1-for-1 + 2-for-2', [
            ComboParam(n_traded=1, n_received=1, threshold=3.0),
            ComboParam(n_traded=2, n_received=2, threshold=3.0),
        ], 0.0, -0.002),
        id='1v1+2v2',
    ),
]


@pytest.fixture(scope='module')
def trading_session():
    """Create one EC session shared across all trade suggest benchmarks."""
    session_request = _build_session_request(scoring_format='Head to Head: Each Category')

    start    = time.perf_counter()
    response = client.post('/sessions', json=session_request)
    session_creation_seconds = time.perf_counter() - start

    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    n_drafters = session_request['league']['n_drafters']
    print(f'\n[benchmark] Session creation — EC ({_SEASON}, {n_drafters} teams): {session_creation_seconds:.2f}s')

    return session_id


def _print_profile(profiler: cProfile.Profile, label: str, top_n: int = 20):
    stream = io.StringIO()
    stats  = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(top_n)
    print(f'\n[profile] {label}')
    print(stream.getvalue())


@pytest.mark.parametrize('config', _SUGGEST_CONFIGS)
def test_trade_suggest_speed(trading_session, config):
    """Times run_trade_suggest for Drafter 1 vs Drafter 2 across combo configurations."""
    label, combo_params, your_threshold, their_threshold = config

    session = get_session(trading_session)

    profiler = cProfile.Profile()
    start    = time.perf_counter()
    result   = profiler.runcall(
        run_trade_suggest
        , session              = session
        , player_assignments   = _DEFAULT_SEASON_ROSTERS
        , my_team              = 'Drafter 1'
        , their_team           = 'Drafter 2'
        , combo_params         = combo_params
        , your_threshold       = your_threshold
        , their_threshold      = their_threshold
        , ignore_position_check = False
    )
    elapsed = time.perf_counter() - start

    n_suggestions = len(result.suggestions)
    print(f'\n[benchmark] Trade suggest — {label}: {elapsed:.2f}s  ({n_suggestions} suggestions)')
    _print_profile(profiler, f'trade suggest — {label}')

    # Sanity: result must be a valid response (not an exception)
    assert result is not None
    assert isinstance(result.suggestions, list)
