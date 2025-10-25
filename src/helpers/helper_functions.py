import numpy as np
import pandas as pd
import streamlit as st
from functools import reduce 
import os 
import uuid

'''Getters 
Getter functions are used for getting various inputs instead of direct access through session state or otherwise
Using getter functions makes it simpler to modulate between getting parameters from session state, integrations, or parameters file 
'''

def using_manual_entry():
  return st.session_state.data_source == 'Enter your own data'

def get_mode():
   return st.session_state.mode

def get_league_type():
  if st.session_state:
    return st.session_state.league
  else:
    return  'NBA'
   
def get_scoring_format():
  if st.session_state:
     return st.session_state.scoring_format
  
def get_omega():
   return st.session_state.omega

def get_gamma():
   return st.session_state.gamma

def get_psi():
   return st.session_state.psi

def get_chi():
   return st.session_state.chi

def get_beth():
   return st.session_state.beth

def get_n_iterations():
   return st.session_state.n_iterations

def get_streaming_noise():
   return st.session_state.streaming_noise

def get_your_differential_threshold():
   return st.session_state.your_differential_threshold / 100

def get_their_differential_threshold():
   return st.session_state.their_differential_threshold / 100

def get_cash_per_team():
   return st.session_state.cash_per_team

def get_styler():
  if st.session_state:
    return st.session_state.styler
  else: 
    return None
  
def get_style_base():
   return st.session_state.base

def get_params():
  if st.session_state:
    return st.session_state.params
  else:
     return None
  
def get_team_names():
   if using_manual_entry():
      return st.session_state.team_names
   else:
      return st.session_state.integration.get_team_names(st.session_state.integration.league_id
                                                          , st.session_state.integration.division_id)
   
def get_n_drafters():
   return len(get_team_names())

def get_rosters_df():
  if using_manual_entry():
    return st.session_state.selections_df
  else:
    return st.session_state.integration.get_rosters_df(st.session_state.integration.league_id)
  
def get_n_picks():
  if using_manual_entry():
    return st.session_state.n_picks
  else:
    return st.session_state.integration.get_n_picks(st.session_state.integration.league_id)
  
def get_n_starters():
    return get_n_picks() - st.session_state.n_bench

def get_n_games():
   if st.session_state:
      return st.session_state.params['n_games']
      
def get_games_per_week():
   if st.session_state:
      return st.session_state.params['n_games_per_week']
   
def get_selections_default():
   if using_manual_entry():
    return st.session_state.selections_default
   else:
    return st.session_state.integration.selections_default
  
@st.cache_data()
def get_selections_default_manual(n_picks, n_drafters):
   return pd.DataFrame(
            {'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)}
            )

def get_selections_df():
   return st.session_state.selections_df

def remove_selections_df():
  #this will trigger selections_df getting replaced by selections_default
  if 'selections_df' in st.session_state:
    del st.session_state.selections_df

def modify_selections_df(row, col, val):
    st.session_state.selections_df.iloc[row, col] = val

def get_player_name_column():
   if 'integration' in st.session_state:
      return st.session_state.integration.get_player_name_column()
   else:
      return 'PLAYER_NAME'
   
def get_selected_players():
  if 'selections_df' in st.session_state:
    return listify(st.session_state.selections_df)
  else: 
    return []
  
@st.cache_data()
def get_max_info(N):

   max_table = pd.read_csv('src/data_retrieval/max_table.csv')

   info = max_table.set_index('N').loc[N]

   return info['EV(X)'],info['VAR(X)'] 
   
def get_conversion_factors():
    coefficient_series = pd.Series(get_params()['coefficients'])
    return coefficient_series.T    

def get_correlations():
   if get_league_type() == 'NBA':
    rho =  pd.read_csv('src/data_retrieval/basketball_correlations.csv').set_index('Category')
   elif get_league_type() == 'MLB':
    rho = pd.read_csv('src/data_retrieval/baseball_correlations.csv').set_index('Category')

   counting_stats = get_counting_statistics()

   #make the aleph adjustment
   rho.loc[counting_stats, counting_stats] = np.clip(rho.loc[counting_stats, counting_stats] + st.session_state.aleph,-1,1)

   negative_stats = st.session_state.params['negative-statistics']
   rho.loc[:,negative_stats] = - rho.loc[:,negative_stats]
   rho.loc[negative_stats,:] = - rho.loc[negative_stats,:]

   rho_values = rho.values
   np.fill_diagonal(rho_values, 1)
   rho = pd.DataFrame(rho_values, index = rho.index, columns = rho.index)

   return rho

def get_draft_position():
    draft_position = st.session_state.draft_position
    row = draft_position['row']
    drafter = draft_position['drafter']
    return row, drafter


### Category getters 

def get_categories():
    #convenience function to get the list of categories used for fantasy basketball
    return get_ratio_statistics() + get_counting_statistics()

def get_selected_volume_statistics():
   return [st.session_state['params']['ratio-statistics'][x]['volume-statistic'] for x in get_selected_ratio_statistics()] 

def get_pitcher_stats():
   if st.session_state:
    return st.session_state.params['pitcher_stats']
   else:
    return ['Wins','Saves','Strikeouts','ERA','WHIP','Innings Pitched','Quality Starts','Holds','Saves and Holds','K/9','K/BB']
    
def get_counting_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return st.session_state['params']['counting-statistics']
    else: 
      if os.environ['SPORT'] == 'NBA':
        return ['Threes','Points','Rebounds','Assists','Steals','Blocks'
                ,'Turnovers','Double Doubles','Off Rebounds','Def Rebounds','Field Goals Made', 'Free Throws Made']
      elif os.environ['SPORT'] == 'MLB':
        return ['Runs','Home Runs', 'RBI', 'Stolen Bases','Doubles','Triples','Hits','Total Bases'
                ,'Wins', 'Strikeouts','Saves', 'Holds','Saves and Holds','Innings Pitched', 'Quality Starts','Losses'] 
          
def get_ratio_statistics():
    #convenience function to get the list of categories used for fantasy basketball
    if st.session_state:
      return list(st.session_state['params']['ratio-statistics'].keys()) 
    else: 
      if os.environ['SPORT'] == 'NBA':
        return ['Field Goal %','Free Throw %','Three %','Assist to TO']
      elif os.environ['SPORT'] == 'MLB':
        return ['Batting Average', 'Slugging %', 'On Base %', 'ERA','WHIP','K/9','K/BB']
    
def get_selected_categories():
    if st.session_state:
       return st.session_state['selected_categories']
    else: 
       return get_selected_counting_statistics() + get_selected_ratio_statistics()
    
def get_selected_counting_statistics():
   if st.session_state:
      return [category for category in get_counting_statistics() if category in st.session_state['selected_categories']]
   else:
      if os.environ['SPORT'] == 'NBA':
        return  ['Threes','Points','Rebounds','Assists','Steals','Blocks','Turnovers']
      elif os.environ['SPORT'] == 'MLB':
        return ['Runs','Home Runs', 'RBI', 'Stolen Bases', 'Wins', 'Saves', 'Strikeouts'] 

def get_selected_ratio_statistics():
   if st.session_state:
      return [category for category in get_ratio_statistics() if category in st.session_state['selected_categories']]
   else:
      if os.environ['SPORT'] == 'NBA':
        return ['Field Goal %','Free Throw %']
      elif os.environ['SPORT'] == 'MLB':
        return ['Batting Average','ERA','WHIP']

### Position getters 

def get_position_numbers():

    if st.session_state:
        res = {}
        for position_code in st.session_state.params['position_structure']['base_list']:  
           res[position_code] = st.session_state['n_' + position_code]

        for position_code in st.session_state.params['position_structure']['flex_list']:  
           res[position_code] = st.session_state['n_' + position_code]

        return res
    else:
       
        return {'Util' :3
                ,'C' : 2
                ,'G' : 2
                ,'PG' : 1
                ,'SG' : 1
                ,'F' : 2
                ,'PF' : 1
                ,'SF' : 1
                }
    
def get_position_numbers_unwound():
   position_numbers = get_position_numbers()
   return reduce(lambda x, y: x + y ,
                  [[position_code + str(i+1) if position_number >1 else position_code for i in range(position_number)] 
                   for position_code, position_number in position_numbers.items()]
                 )
    
def get_position_structure():
    if st.session_state:
       return st.session_state.params['position_structure']
    else:
       return { 'base_list' :
                        ['PG','SG','SF','PF','C']
               ,'base' : {'C' : {'full_str' : 'Centers'}
                         ,'PG' :{'full_str' : 'Point Guards'}
                         ,'SG' : {'full_str' : 'Shooting Guards'}
                         ,'PF' : {'full_str' : 'Power Forwards'}
                         ,'SF' : {'full_str' : 'Small Forwards'}}
               ,'flex_list' : ['G','F','Util']
               ,'flex' : {'Util' : 
                          {'bases' : ['C','PG','SG','PF','SF']
                           ,'full_str' : 'Utilities'
                          }
                        ,'G' : {'bases' : ['PG','SG']
                           ,'full_str' : 'Guards'
                          }
                        ,'F' : {'bases' : ['PF','SF']
                           ,'full_str' : 'Forwards'
                          }
                            }
                 }

def get_position_ranges():

    if st.session_state:

        start = 0
        end = 0
        res = {}

        position_structure = st.session_state.params['position_structure']
        
        all_positions = position_structure['base_list'] + position_structure['flex_list']

        for position_code in all_positions:
           end += st.session_state['n_' + position_code]
           res[position_code] = {'start' : start, 'end': end}
           start = end

        return res 

    else:
       
       #default to the standard position requirements
       center = {'start' : 0, 'end' : 2}
       point_guard = {'start' : 2, 'end' : 3}
       shooting_guard = {'start' : 3, 'end' : 4}
       power_forward = {'start' : 4, 'end' : 5}
       shooting_forward = {'start' : 5, 'end' : 6}
       utility = {'start' : 6, 'end' : 9}
       guard =  {'start' : 9, 'end' : 11}
       forward = {'start' : 11, 'end' : 13}

       return {
                'C' : center
                ,'PG' : point_guard
                ,'SG' : shooting_guard
                ,'PF' : power_forward
                ,'SF' : shooting_forward
                ,'Util' :utility
                ,'G' : guard
                ,'F' : forward

                }
    
def get_position_indices(params):
   
    position_structure = get_position_structure()
    flex_info =  position_structure['flex']
    base_position_list = position_structure['base_list']

    return {position_code : 
                            [i for i, val in enumerate(base_position_list) if val in position_info['bases']]
                                    for position_code, position_info in flex_info.items()
            }

   
'''Setters 
Setters are used for setting parameters that are not just stored as keys of Streamlit widgets 
'''

def set_params(league):
   st.session_state.params = st.session_state.all_params[league]
  
def initialize_selections_df():
  if 'selections_df' not in st.session_state:
      st.session_state.selections_df = get_selections_default()

def set_draft_position(row, drafter):
   st.session_state.draft_position = {'row' : row
                                      ,'drafter' : drafter}


'''
Dataframe storage

Small dataframes that are unique to individual user sessions are stored in session state. 
Each of the dataframes is associated with a randomly generated key, so that the key can be checked
for hashing instead of the entire dataframe 

Dataframes included with this:
player_stats_v0: dataframe of raw stats, before dropping injured players
player_stats_v1: dataframe of raw stats, after dropping injured players but before the upsilon adjustment
player_stats_v2: dataframe after dropping injured players

need to make player_stats_v0 get updated more correctly, and add metadata

add all of the things from info as a dict (they get updated together)

'''
def gen_key():
    return str(uuid.uuid4())

def store_dataset_in_session_state(df, dataset_name, key):
    #check if updating the dataset is necessary. It isn't if the key has not changed
    if not ((dataset_name in st.session_state.data_dictionary) and (st.session_state.data_dictionary == key)):
      st.session_state.data_dictionary[dataset_name] = {'key' : key
                                                        ,'data' : df}
    
def get_data_key(dataset_name):
   return st.session_state.data_dictionary[dataset_name]['key']

def get_data_from_session_state(dataset_name):
   if dataset_name in st.session_state.data_dictionary:
    return st.session_state.data_dictionary[dataset_name]['data']
   else:
    return None


'''
Drafting helpers 
'''
def get_default_draft_seat_index():
    team_names = get_team_names()
    if ('draft_seat' in st.session_state) and (st.session_state.draft_seat in team_names):
        return team_names.index(st.session_state.draft_seat)
    else:
        return 0
    
def move_forward_one_pick(row: int, drafter: int, n: int):
    
    if st.session_state.third_round_reversal:

      #implement the actual third round reversal
      if (row == 1 and drafter == 0):
         return 2, n - 1
      
      #the orders are switched for all future rounds 
      odd_row = (row % 2 == 1)  if row < 2 else (row % 2 == 0) 

    else:
      odd_row = (row % 2 == 1) 
    
    if odd_row:
      if drafter == 0:
        row = row + 1
      else:
        drafter = drafter - 1
    else:
      if (drafter == n - 1):
        row = row + 1
      else:
        drafter = drafter + 1

    return row, drafter 

def move_back_one_pick(row: int, drafter: int, n: int):
    
    if st.session_state.third_round_reversal:

      #reverse the actual third round reversal
      if (row == 2 and drafter == n-1):
         return 1, 0
      
      #the orders are switched for all future rounds 
      odd_row = (row % 2 == 1)  if row < 2 else (row % 2 == 0) 

    else:
      odd_row = (row % 2 == 1) 

    if odd_row:
      if drafter == (n-1):
        row = row - 1
      else:
        drafter = drafter + 1
    else:
      if (drafter == 0):
        row = row - 1
      else:
        drafter = drafter - 1

    return row, drafter 

def run_h_score():
    st.session_state.run_h_score = True

def stop_run_h_score():
    st.session_state.run_h_score = False

def h_score_is_running():
    return st.session_state.run_h_score

'''
Database connector helper functions
''' 

@st.cache_data(ttl = '1d')
def get_data_from_snowflake(table_name
                            , schema = 'FANTASYBASKETBALLOPTIMIZER'):
   
   con = get_snowflake_connection(schema)

   df = con.cursor().execute('SELECT * FROM ' + table_name).fetch_pandas_all()

   return df

@st.cache_resource(ttl = 3600)
def get_snowflake_connection(schema):
      con = st.connection("snowflake", ttl = 3600)
      return con

'''
Miscellaneous 
''' 
def adjust_teams_dict_for_duplicate_names(teams_dict):
    all_names = []
    for k, v in teams_dict.items():
        i = 1
        new_name = v 

        while new_name in all_names:
            i = i + 1
            new_name = v + ' ' + str(i)

        all_names = all_names + [new_name]

        if i != 1:
            teams_dict[k] = new_name

    return teams_dict

def get_L_weights() -> pd.Series:
   #calculate a default weighting for L
   #this assumes that all flex positions are weighted evenly among their bases 
   position_structure = get_position_structure()
   position_numbers = get_position_numbers()
   flex_positions = position_structure['flex_list']
   base_positions = position_structure['base_list']
   n_slots = sum([v for v in position_numbers.values()])

   shares = pd.Series({position_code : position_numbers[position_code]/n_slots for position_code in base_positions})

   for position_code in flex_positions:
      bases = position_structure['flex'][position_code]['bases']

      for base in bases:
         shares[base] += position_numbers[position_code]/(n_slots * len(bases)) 

   return shares 

@st.cache_data(ttl = 3600)
def get_fixed_player_name(player_name : str, info_key : str) -> str:
    
    """Fix player name string to adhere to common standard

    Args:
        player_name: string

    Returns:
        fixed name string
     """
    if isinstance(player_name, pd.Series):
       player_name = player_name.values[0] #fix for weird thing with auctions

    positions = get_data_from_session_state('info')['Positions'].copy()
    positions.index = [x.split(' (')[0] for x in positions.index]

    if player_name in positions.index:
        return player_name + ' (' + ','.join(positions[player_name]) + ')'
    else:
        st.write(player_name)
        return 'RP'

def get_combo_params():
  combo_params_df = st.session_state.combo_params_df
  combo_params_df[['N-traded','N-received']] = \
        combo_params_df[['N-traded','N-received']].astype(int)

  combo_params = tuple(combo_params_df.itertuples(name = None, index = None))

  return combo_params

def listify(x : pd.DataFrame) -> list:
    #get all values from a dataframe into a list. Useful for listing all chosen players 
    #Goes row by row- very important! 

    x = x.values.tolist()
    return [item for row in x for item in row]

def rotate(l, n):
  #rotate list l by n positions 
  return l[-n:] + l[:-n]


def weighted_cov_matrix(df, weights):
    weighted_means = np.average(df, axis=0, weights=weights)
    deviations = df - weighted_means
    weighted_cov = np.dot(weights * deviations.T, deviations) / weights.sum()
    return pd.DataFrame(weighted_cov, columns=df.columns, index=df.columns)

