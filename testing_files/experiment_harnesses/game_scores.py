"""Single-game G-scores: every game a player played, scored against his season with the G-score's own definitions.

The app's G-score coefficients (Mean of Means, Variance of Means, Mean of Variances) are computed per season from weekly
totals (backend/math/coefficient_exploration.compute_season_coefficients over WEEKLY_NUMBERS_VIEW). A single game needs
them at game scale, so the same function is given one row per player per game instead of per week: the same
representative players (the top 12 x 13 by G-score), the same definitions, with a game as the period. Each game is then
scored by the app's G-score function (calculate_scores_from_coefficients, then the percentage categories' volume
correction), and its value is the sum over the nine standard categories (turnovers counting against it).

Only games a player actually played count (minutes above 0): a did-not-play row is not a performance.
"""

from __future__ import annotations

import pandas as pd

from .league_sessions import prepare_import_path

# The nine standard categories, scored as the app scores them.
GAME_CATEGORIES = ['Field Goal %', 'Free Throw %', 'Threes', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks',
                   'Turnovers']
# The representative league the coefficients are computed for (the app's default: 12 teams of 13).
REPRESENTATIVE_DRAFTERS = 12
REPRESENTATIVE_PICKS = 13


def load_season_games(season: str
                      , sport_params: dict) -> pd.DataFrame:
    """Every game a player played in the season (minutes above 0), indexed by (Player id, game id), in the app's
    column names, with the date and the player's display name (the historical view's, which numbers shared names)."""
    from backend.infra.snowflake_connection import run_query
    games = run_query(f"""
        SELECT COALESCE(L.NBA_PLAYER_ID, A.NBA_PLAYER_ID) AS NBA_PLAYER_ID, A.GAME_ID, C.GAME_DATE, A.MINUTES_PLAYED,
               A.POINTS, A.REBOUNDS, A.OFFENSIVE_REBOUNDS, A.DEFENSIVE_REBOUNDS, A.ASSISTS, A.STEALS, A.BLOCKS, A.THREES,
               A.TURNOVERS, A.FREE_THROW_ATTEMPTS, A.FREE_THROWS_MADE, A.FIELD_GOAL_ATTEMPTS, A.FIELD_GOALS_MADE,
               A.THREE_ATTEMPTS, A.DOUBLE_DOUBLE AS DOUBLE_DOUBLES
        FROM BOX_SCORES_EXPANDED_VIEW AS A
        LEFT JOIN NBA_PLAYER_ID_ALIAS_TABLE AS L ON A.NBA_PLAYER_ID = L.ALIAS_ID
        JOIN NBA_GAME_TABLE AS C ON A.GAME_ID = C.GAME_ID
        JOIN NBA_SEASON_TABLE AS D ON C.SEASON_ID = D.SEASON_ID
        WHERE D.SEASON = '{season}' AND A.MINUTES_PLAYED > 0""")
    names = run_query(f"SELECT NBA_PLAYER_ID, NBA_PLAYER_NAME FROM HISTORICAL_SEASONAL_AVERAGES_VIEW "
                      f"WHERE SEASON = '{season}'").set_index('NBA_PLAYER_ID')['NBA_PLAYER_NAME']
    games = games.rename(columns=sport_params['stat-df-renamer'])
    games['Name'] = games['NBA_PLAYER_ID'].map(names)
    if games['Name'].isna().any():
        missing = sorted(games.loc[games['Name'].isna(), 'NBA_PLAYER_ID'].unique().tolist())
        raise LookupError(f'{season}: box-score players with no row in the historical view: {missing[:10]}')
    games = games.rename(columns={'NBA_PLAYER_ID': 'Player'}).set_index(['Player', 'GAME_ID']).sort_index()
    return games


# The season whose coefficients every game is also scored with, so every era is judged by today's standard.
MODERN_SEASON = '2025-26'


def read_stat_columns(games: pd.DataFrame) -> pd.DataFrame:
    """The games' box-score statistics (minutes left out: the coefficients are over the categories' statistics)."""
    return games.select_dtypes(include='number').drop(columns=['MINUTES_PLAYED'])


def compute_coefficients_for_games(stat_columns: pd.DataFrame) -> pd.DataFrame:
    """G-score coefficients at game scale for these games (compute_season_coefficients over one row per player-game)."""
    from backend.math.coefficient_exploration import compute_season_coefficients
    from backend.services.build_agent import load_all_params
    return compute_season_coefficients(stat_columns.copy(), load_all_params()['NBA'], REPRESENTATIVE_DRAFTERS,
                                       REPRESENTATIVE_PICKS)['coefficients']


def compute_game_coefficients(season: str) -> pd.DataFrame:
    """The season's G-score coefficients at game scale."""
    prepare_import_path()
    from backend.services.build_agent import load_all_params
    return compute_coefficients_for_games(read_stat_columns(load_season_games(season, load_all_params()['NBA'])))


def score_games(stat_columns: pd.DataFrame
                , coefficients: pd.DataFrame) -> pd.DataFrame:
    """Each game's G-score per category under these coefficients (the app's G-score function, then the percentage
    categories' volume correction)."""
    from backend.math.process_player_data import apply_team_volume_correction, calculate_scores_from_coefficients
    from backend.services.build_agent import load_all_params
    sport_params = load_all_params()['NBA']
    counting_stats = sport_params['counting-statistics']
    ratio_stats = list(sport_params['ratio-statistics'])
    rows = stat_columns.copy()
    for ratio_stat, info in sport_params['ratio-statistics'].items():
        rows[ratio_stat] = rows[info['made-statistic']] / rows[info['volume-statistic']]
    scores = calculate_scores_from_coefficients(rows, coefficients, sport_params, 1, 1, counting_stats, ratio_stats,
                                                counting_stats + ratio_stats)
    return apply_team_volume_correction(scores, rows, coefficients, sport_params, ratio_stats, REPRESENTATIVE_PICKS)


def score_season_games(season: str
                       , modern_coefficients: pd.DataFrame) -> pd.DataFrame:
    """Every game of the season with its G-score per category and its total over GAME_CATEGORIES under the season's
    own coefficients ('G-score'), and its total under the modern season's ('Modern G-score')."""
    prepare_import_path()
    from backend.services.build_agent import load_all_params
    games = load_season_games(season, load_all_params()['NBA'])
    stat_columns = read_stat_columns(games)
    scores = score_games(stat_columns, compute_coefficients_for_games(stat_columns))
    modern_scores = score_games(stat_columns, modern_coefficients)
    scored = games[['Name', 'GAME_DATE', 'MINUTES_PLAYED']].join(stat_columns).join(
        scores[GAME_CATEGORIES].add_prefix('G '))
    scored['G-score'] = scores[GAME_CATEGORIES].sum(axis=1)
    scored['Modern G-score'] = modern_scores[GAME_CATEGORIES].sum(axis=1)
    return scored
