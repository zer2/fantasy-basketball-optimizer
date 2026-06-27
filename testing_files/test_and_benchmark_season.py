# testing_files/test_and_benchmark_season.py
# Season Mode correctness tests: waiver wire evaluation.
# Uses 2024-25 historical season data with default parameters from parameters.yaml.
# Rosters mirror the pre-computed defaults in frontend/data_entry/season/default_season_rosters.ts.
#
# Covers:
#   - Waiver evaluate: Drafter 1 drops Gary Trent Jr.
# Trade analysis is covered separately in test_trading.py and test_and_benchmark_trading.py.

from benchmark_helpers import (
    client
    , _SCORE_TOL
    , _build_session_request
)
from backend.session import get_session
from backend.evaluate import run_evaluate

# 2024-25 snapshot of an H-score snake draft used by the assertions below.
# (The frontend default — default_season_rosters.ts — tracks current-season data
# and is regenerated via testing_files/generate_default_season_rosters.py.)
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


def test_season_mode_waiver():
    """Season Mode: default H-score snake-draft rosters, EC scoring.
    Waiver evaluate: Drafter 1 drops Gary Trent Jr.
    """
    session_request = _build_session_request(scoring_format='Head to Head: Each Category')
    response        = client.post('/sessions', json=session_request)
    assert response.status_code == 201, f'Session creation failed: {response.text}'
    session_id = response.json()['session_id']

    session      = get_session(session_id)
    n_iterations = session.current_params['n_iterations']

    drafter_1_roster = _DEFAULT_SEASON_ROSTERS['Drafter 1']

    # ── Waiver: Drafter 1 drops Gary Trent Jr. ───────────────────────────────

    gary_name = next(name for name in drafter_1_roster if name.startswith('Gary Trent'))

    waiver_assignments                 = {team: list(roster) for team, roster in _DEFAULT_SEASON_ROSTERS.items()}
    waiver_assignments['Drafter 1']    = [p for p in drafter_1_roster if p != gary_name]

    waiver_result = run_evaluate(
        session            = session
        , player_assignments = waiver_assignments
        , my_team_id         = 'Drafter 1'
        , exclusion_list     = []
        , remaining_cash     = None
    )

    candidates = waiver_result.candidates
    assert len(candidates) > 0

    h_scores = [c.h_score for c in candidates]
    assert h_scores == sorted(h_scores, reverse=True), 'Waiver candidates not sorted by H-score'

    expected_waiver_top = [
        ('Guerschon Yabusele',         52.6),
        ('Mark Williams',              52.5),
        ('Keyonte George',             52.5),
        ('Gary Trent',                 52.5),
    ]
    candidates_by_name = {c.name: c for c in candidates}
    for expected_name, expected_score in expected_waiver_top:
        match = next((name for name in candidates_by_name if name.startswith(expected_name)), None)
        assert match is not None, f'{expected_name} not found in waiver candidates'
        actual_score = candidates_by_name[match].h_score
        assert abs(actual_score - expected_score) <= _SCORE_TOL, (
            f'{match} (waiver): expected {expected_score}, got {actual_score:.1f}'
        )
