"""Market draft values -- Yahoo's average draft position and average auction cost -- blended with H-scoring's own
order and prices, to predict when other drafters will take each player. Used only by the example team (the
hypothetical team in the candidate panel), never by H-scoring or the ranking.

The algorithm's base H-score order often disagrees with how people actually draft, which made example teams look
unrealistic (a player it likes far more than the market showing up as a late pick). Averaging with the market keeps
the league's own format in the prediction while pulling it toward real draft rooms; neither source decides alone.

Yahoo's numbers are for the season being drafted, so they apply only to sessions built from projections; a historical
season has no market data and keeps H-scoring's order alone. The data is YAHOO_ADP_TABLE, loaded from Yahoo's draft
analysis by the data-scripts repo (yahoo_adp_table.py); each run appends a load, and the newest one is used.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.data_retrieval import get_unified_player_table
from backend.infra.snowflake_connection import query

# The weight of the market in each blend; H-scoring gets the rest. An even split, stated up front, not tuned.
MARKET_WEIGHT = 0.5


def load_market_draft_values() -> pd.DataFrame:
    """Yahoo's draft analysis by player id, from the newest load of YAHOO_ADP_TABLE: average_pick and average_cost (NaN
    for a player Yahoo reports no ADP for), and team (his NBA team for the season)."""
    loads = query('YAHOO_ADP_TABLE')
    latest = loads[loads['UPDATE_TIMESTAMP'] == loads['UPDATE_TIMESTAMP'].max()]
    unified = get_unified_player_table()[['MASTER_PLAYER_ID', 'NBA_PLAYER_ID']].dropna()
    frame = latest.merge(unified.astype({'MASTER_PLAYER_ID': 'int64', 'NBA_PLAYER_ID': 'int64'})
                         , on='MASTER_PLAYER_ID')
    frame = frame.rename(columns={'NBA_PLAYER_ID': 'player_id', 'ADP': 'average_pick'
                                  , 'AVERAGE_AUCTION_PRICE': 'average_cost', 'TEAM': 'team'})
    # A player listed twice with the same numbers is one player (the unified table can enter one person under two master
    # ids); listed twice with different numbers, the data is ambiguous and must be fixed rather than guessed at.
    frame = frame[['player_id', 'average_pick', 'average_cost', 'team']].drop_duplicates()
    conflicting = frame['player_id'][frame['player_id'].duplicated()].unique()
    if len(conflicting) > 0:
        raise ValueError(f'Market draft values list players {list(conflicting)} with conflicting numbers.')
    return frame.set_index('player_id')


def uses_market_draft_values(current_settings: dict) -> bool:
    """Whether this session's example teams use market values: only a session built from projections drafts the season
    the market data describes."""
    return current_settings['data_source_type'] == 'projections'


def compute_availability_order(
    ranked_players: list[int]
    , market_values: Optional[pd.DataFrame]
    , n_total_picks: int
) -> list[int]:
    """Players in the order other drafters are predicted to take them. Without market values that is H-scoring's base
    order (ranked_players); with them, each player's predicted pick blends his base H-score rank (1 = first) with his
    average draft position, a player nobody drafts counting as the draft's last pick (n_total_picks)."""
    if market_values is None:
        return list(ranked_players)
    h_rank = pd.Series(range(1, len(ranked_players) + 1), index=ranked_players, dtype=float)
    average_pick = market_values['average_pick'].reindex(h_rank.index).fillna(float(n_total_picks))
    predicted_pick = (1 - MARKET_WEIGHT) * h_rank + MARKET_WEIGHT * average_pick
    # Stable: equal predictions keep H-scoring's order.
    return list(predicted_pick.sort_values(kind='mergesort').index)


def compute_blended_prices(
    generic_prices: pd.Series
    , market_values: Optional[pd.DataFrame]
) -> pd.Series:
    """Predicted auction prices of the available players (generic_prices' index): generic dollars, blended with Yahoo's
    average cost rescaled so it, too, sums to the remaining cash over these players. A player nobody bought costs $0 on
    the market side. Without market values, the generic dollars alone."""
    if market_values is None:
        return generic_prices
    market_cost = market_values['average_cost'].reindex(generic_prices.index).fillna(0.0)
    if market_cost.sum() <= 0:
        raise ValueError('No available player has a market auction cost to blend with.')
    market_cost = market_cost * (generic_prices.sum() / market_cost.sum())
    return (1 - MARKET_WEIGHT) * generic_prices + MARKET_WEIGHT * market_cost
