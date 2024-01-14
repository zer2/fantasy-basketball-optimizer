import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs

def get_data(season):
  season_str = str(season -1) + '-' + str(season -2000)
  pgl_df = pd.concat(
      [
          playergamelogs.PlayerGameLogs(
              season_nullable="2023-24", season_type_nullable=season_type
          ).player_game_logs.get_data_frame()
          for season_type in ["Regular Season"]
      ]
  )
