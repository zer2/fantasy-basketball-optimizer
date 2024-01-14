import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playergamelogs
from streamlit import cache_resource

@cache_resource(ttl = '1d') 
def get_data(season):
  season_str = str(season -1) + '-' + str(season -2000)
  pgl_df = pd.concat(
      [
          playergamelogs.PlayerGameLogs(
              season_nullable=season_str, season_type_nullable=season_type
          ).player_game_logs.get_data_frame()
          for season_type in ["Regular Season"]
      ]
  )
  
  relevant_columns = ['PLAYER_NAME','PTS','REB','AST','STL','BLK','FG3M','TOV','FTA','FTM','FGA','FGM','GAME_DATE']
  pgl_df = pgl_df[relevant_columns]

def process_game_level_data(df):
  agg_df = df.groupy('PLAYER_NAME').mean()
  agg_df.loc[:,'Free Throw %'] = df['FTM']/df['FTA']
  agg_df.loc[:,'Field Goal %'] = df['FGA']/df['FGM']
