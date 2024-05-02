import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from src.helper_functions import get_categories
import os
import streamlit as st

def calculate_coefficients(player_means : pd.DataFrame
                     , representative_player_set : list
                     , translation_factors : pd.Series
                     ) -> dict:
    """calculate the coefficients for each category- \mu,\sigma, and \tau, so we can use them for Z-scores and G-scores

    Args:
        player_means: dataframe of fantasy-relevant statistics 
        representative_player_set: list of players to use as sample for coefficients
        translation_factors: series, converts sigma^2 to tau^2
    Returns:
        Dictionary mapping 'Mean of Means' -> (series mapping category to /mu^2 etc.) 

    """

    params = st.session_state.params

    #counting stats
    var_of_means = player_means.loc[representative_player_set,params['counting-statistics']].var(axis = 0)
    mean_of_means = player_means.loc[representative_player_set,params['counting-statistics']].mean(axis = 0)

    #free throw percent
    fta_mean_of_means = player_means.loc[representative_player_set, 'Free Throw Attempts'].mean()
    mean_of_means.loc['Free Throw Attempts'] = fta_mean_of_means
                       
    ftp_agg_average = (player_means.loc[representative_player_set, 'Free Throw %'] * player_means.loc[representative_player_set, 'Free Throw Attempts']).mean()/fta_mean_of_means
    mean_of_means.loc['Free Throw %'] = ftp_agg_average
                       
    ftp_numerator = player_means.loc[representative_player_set, 'Free Throw Attempts']/fta_mean_of_means * (player_means.loc[representative_player_set, 'Free Throw %'] - ftp_agg_average)
    ftp_var_of_means = ftp_numerator.var()
    var_of_means.loc['Free Throw %'] = ftp_var_of_means
    
    #field goal %
    fga_mean_of_means = player_means.loc[representative_player_set, 'Field Goal Attempts'].mean()
    mean_of_means.loc['Field Goal Attempts'] = fga_mean_of_means
                       
    fgp_agg_average = (player_means.loc[representative_player_set, 'Field Goal %'] * player_means.loc[representative_player_set, 'Field Goal Attempts']).mean()/fga_mean_of_means
    mean_of_means.loc['Field Goal %'] = fgp_agg_average
                       
    fgp_numerator = player_means.loc[representative_player_set, 'Field Goal Attempts']/fga_mean_of_means * (player_means.loc[representative_player_set, 'Field Goal %'] - fgp_agg_average)
    fgp_var_of_means = fgp_numerator.var()
    var_of_means.loc['Field Goal %'] = fgp_var_of_means

    #get mean of vars

    translation_factors['Free Throw Attempts'] = 0
    translation_factors['Field Goal Attempts'] = 0

    mean_of_vars = var_of_means * translation_factors

    coefficients = pd.DataFrame({'Mean of Means' : mean_of_means
                                ,'Variance of Means' : var_of_means
                                ,'Mean of Variances' : mean_of_vars
                                }
                               )
    return coefficients

def calculate_coefficients_historical(weekly_df : pd.DataFrame
                     , representative_player_set : list
                     ) -> dict:
    """calculate the coefficients from a real weekly dataset by week

    Args:
        weekly_df: dataframe of fantasy-relevant statistics 
        representative_player_set: list of players to use as sample for coefficients
    Returns:
        Dictionary mapping 'Mean of Means' -> (series mapping category to /mu^2 etc.) 

    """
    params = st.session_state.params

    player_stats = weekly_df.groupby(level = 'Player').agg(['mean','var'])

    #counting stats
    mean_of_vars = player_stats.loc[representative_player_set,(params['counting-statistics'],'var')].mean(axis = 0)
    var_of_means = player_stats.loc[representative_player_set,(params['counting-statistics'],'mean')].var(axis = 0)
    mean_of_means = player_stats.loc[representative_player_set,(params['counting-statistics'],'mean')].mean(axis = 0)

    #free throw percent
    ft_mean_of_means = player_stats.loc[representative_player_set, ('Free Throws Made','mean')].mean()
    fta_mean_of_means = player_stats.loc[representative_player_set, ('Free Throw Attempts','mean')].mean()
    mean_of_means.loc['Free Throw Attempts'] = fta_mean_of_means
    ftp = player_stats.loc[:, ('Free Throws Made','mean')]/player_stats.loc[:, ('Free Throw Attempts','mean')]
    ftp_agg_average = ft_mean_of_means / fta_mean_of_means
    mean_of_means.loc['Free Throw %'] = ftp_agg_average
    ftp_numerator = player_stats.loc[:, ('Free Throw Attempts','mean')]/fta_mean_of_means * (ftp - ftp_agg_average)
    ftp_var_of_means = ftp_numerator.loc[representative_player_set].var()
    var_of_means.loc['Free Throw %'] = ftp_var_of_means
    
    weekly_df.loc[:,'volume_adjusted_ftp'] = (weekly_df['Free Throws Made'] - weekly_df['Free Throw Attempts']*ftp_agg_average)/ \
                                            fta_mean_of_means
    ftp_mean_of_vars = weekly_df['volume_adjusted_ftp'].loc[representative_player_set].groupby('Player').var().mean()
    mean_of_vars.loc['Free Throw %'] = ftp_mean_of_vars

    #field goal percent
    fg_mean_of_means = player_stats.loc[representative_player_set, ('Field Goals Made','mean')].mean()
    fga_mean_of_means = player_stats.loc[representative_player_set, ('Field Goal Attempts','mean')].mean()
    mean_of_means.loc['Field Goal Attempts'] = fga_mean_of_means
    fgp_agg_average = fg_mean_of_means / fga_mean_of_means
    mean_of_means.loc['Field Goal %'] = fgp_agg_average
    fgp = player_stats.loc[:, ('Field Goals Made','mean')]/player_stats.loc[:, ('Field Goal Attempts','mean')]
    fgp_numerator = player_stats.loc[:, ('Field Goal Attempts','mean')]/fga_mean_of_means * (fgp - fgp_agg_average)
    fgp_var_of_means = fgp_numerator.loc[representative_player_set].var()
    var_of_means.loc['Field Goal %'] = fgp_var_of_means
    weekly_df.loc[:,'volume_adjusted_fgp'] = (weekly_df['Field Goals Made'] - weekly_df['Field Goal Attempts']*fgp_agg_average)/ \
                                        fga_mean_of_means
    fgp_mean_of_vars = weekly_df['volume_adjusted_fgp'].loc[representative_player_set].groupby('Player').var().mean()    
    mean_of_vars.loc['Field Goal %'] = fgp_mean_of_vars

    coefficients = pd.DataFrame({'Mean of Means' : mean_of_means.droplevel(level = 1)
                            ,'Variance of Means' : var_of_means.droplevel(level = 1)
                            ,'Mean of Variances' : mean_of_vars.droplevel(level = 1)
                            }
                            )
    return coefficients
  
def calculate_scores_from_coefficients(player_means : pd.DataFrame
                                       ,coefficients :pd.DataFrame
                                       , params : dict
                                       ,alpha_weight : float = 1
                                       ,beta_weight : float = 1
                                       ) -> pd.DataFrame:
    """Calculate scores based on player info and coefficients

    Args:
        player_means: Dataframe of fantasy-relevant statistics 
        coefficients: Dataframe of coefficients- mean of means, var of means, mean of vars
        params: dict of parameters
        alpha_weight: weight for /sigma
        beta_weight: weight for /tau
    Returns:
        Dataframe of scores, by player/category
    """
    params = st.session_state.params

    counting_cat_mean_of_means = coefficients.loc[params['counting-statistics'],'Mean of Means']
    counting_cat_var_of_means = coefficients.loc[params['counting-statistics'],'Variance of Means']
    counting_cat_mean_of_vars = coefficients.loc[params['counting-statistics'],'Mean of Variances']

    counting_cat_denominator = (counting_cat_var_of_means.values*alpha_weight + counting_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_means.loc[:,params['counting-statistics']] - counting_cat_mean_of_means
    main_scores = numerator.divide(counting_cat_denominator)
    main_scores['Turnovers'] = - main_scores['Turnovers']

    #free throws 
    ftp_denominator = (coefficients.loc['Free Throw %','Variance of Means']*alpha_weight + coefficients.loc['Free Throw %','Mean of Variances']*beta_weight)**0.5
    ftp_numerator = player_means.loc[:, 'Free Throw Attempts']/coefficients.loc['Free Throw Attempts','Mean of Means'] * (player_means['Free Throw %'] - coefficients.loc['Free Throw %','Mean of Means'])
    ftp_score = ftp_numerator.divide(ftp_denominator)

    #field goals
    fgp_denominator = (coefficients.loc['Field Goal %','Variance of Means']*alpha_weight + coefficients.loc['Field Goal %','Mean of Variances']*beta_weight)**0.5
    fgp_numerator = player_means.loc[:, 'Field Goal Attempts']/coefficients.loc['Field Goal Attempts','Mean of Means'] * (player_means['Field Goal %']  - coefficients.loc['Field Goal %','Mean of Means'])
    fgp_score = fgp_numerator.divide(fgp_denominator)
    
    res = pd.concat([fgp_score, ftp_score, main_scores],axis = 1)  
    res.columns = get_categories()
    return res.fillna(0)

def games_played_adjustment(scores : pd.DataFrame
                      , replacement_games_rate : pd.Series
                      , n_players : int
                      , v : pd.Series = None) -> pd.DataFrame :
    """Adjust scores based on effective games played. Requires scores to be in same order as replacement_player_value

    Args:
        scores: Dataframe of category-level scores. One row per player, one column per statistic 
        replacement_player_value: Series of rates between 0 and 1. That rate of games are assumed to be filled in by a replacement-level player
        n_players: Number of players that will be relevant to fantasy
        v: series of weights per category. Only applicable if scores don't translate precisely to overall value 

    Returns:
        Dataframe with same dimensions as scores 
    """

    all_stats = st.session_state.params['percentage-statistics'] + st.session_state.params['counting-statistics']
    if v is None:
       v = pd.Series({stat : 1/9 for stat in all_stats})

    totals = scores.dot(v)

    rv = totals.sort_values(ascending = False).iloc[n_players]
    category_level_rv = get_category_level_rv(rv, v)

    replacement_player_value = np.array(category_level_rv.T).reshape(1,-1) * \
                                np.array(replacement_games_rate).reshape(-1,1)
    adjusted_scores =  scores + replacement_player_value

    return adjusted_scores 

def get_category_level_rv(rv : float, v : pd.Series):
   all_stats = st.session_state.params['percentage-statistics'] + st.session_state.params['counting-statistics']
   rv_multiple = rv/(len(all_stats) -2) if  'Turnovers' in all_stats else rv/len(all_stats)
   value_by_category = pd.Series({stat : - rv_multiple/v[stat] if stat == 'Turnovers' 
                                  else rv_multiple/v[stat] for stat in all_stats})
   
   return value_by_category

@st.cache_data(show_spinner = False)
def process_player_data(  _weekly_df : pd.DataFrame
                        , _player_means : pd.DataFrame
                        , conversion_factors :pd.Series
                        , multipliers : pd.Series
                        , upsilon : float
                        , psi : float
                        , nu : float
                        , n_drafters : int
                        , n_picks : int
                        , player_stats_key
                        ) -> dict:
  """Based on player stats and parameters, do all calculations to set up for running algorithms

  Args:
      weekly_df: Dataframe of fantasy-relevant statistics at the weekly level 
      player_means: Dataframe of player means
      conversion_factors: Conversion factors for /sigma^2 to /tau^2. Needed if weekly_df is None
      multipliers: Manual multipliers for categories, based on user inputs
      psi: parameter scaling the influence of no_play rate
      nu: parameter scaling the influence of category means in covariance calculation
      n_drafters: number of drafters
      n_picks: number of picks per drafter
      player_stats_key: key for version number of player stats. Used to check if player stats has changed
  Returns:
      Info dictionary with many pieces of information relevant to the algorithm 
  """
  params = st.session_state.params

  n_players = n_drafters * n_picks

  if _weekly_df is not None:
    coefficients_first_order = calculate_coefficients_historical(_weekly_df
                                                , pd.unique(_weekly_df.index.get_level_values('Player')))
  else:
    coefficients_first_order = calculate_coefficients(_player_means
                                                  , _player_means.index
                                                  , conversion_factors['Conversion Factor'])
        
  z_scores_first_order =  calculate_scores_from_coefficients(_player_means
                                                          , coefficients_first_order
                                                          , params
                                                          , 1
                                                          ,0)
  z_scores_first_order = z_scores_first_order * multipliers.T.values[0]

  first_order_score = z_scores_first_order.sum(axis = 1)
  representative_player_set = first_order_score.sort_values(ascending = False).index[0:n_picks * n_drafters]

  if _weekly_df is not None:
    coefficients = calculate_coefficients_historical(_weekly_df
                                                , representative_player_set)
  else:
    coefficients = calculate_coefficients(_player_means
                                                  , representative_player_set
                                                  , conversion_factors['Conversion Factor'])

  mov = coefficients.loc[get_categories() , 'Mean of Variances']
  vom = coefficients.loc[get_categories() , 'Variance of Means']
  v = np.sqrt(mov/(mov + vom)) #ZR: This doesn't work for Roto. We need to fix that later
  v = v/v.sum()

  g_scores = calculate_scores_from_coefficients(_player_means, coefficients, params, 1,1)
  z_scores =  calculate_scores_from_coefficients(_player_means, coefficients, params,  1,0)
  x_scores =  calculate_scores_from_coefficients(_player_means, coefficients, params, 0,1)

  replacement_games_rate = (1- _player_means['Games Played %']/100) * psi
  g_scores = games_played_adjustment(g_scores, replacement_games_rate,n_players)
  z_scores = games_played_adjustment(z_scores, replacement_games_rate,n_players)
  x_scores = games_played_adjustment(x_scores, replacement_games_rate,n_players, v = v)

  g_scores = g_scores * multipliers.T.values[0]
  z_scores = z_scores * multipliers.T.values[0]
  x_scores = x_scores * multipliers.T.values[0]

  z_scores.insert(loc = 0, column = 'Total', value = z_scores.sum(axis = 1))

  z_scores.sort_values('Total', ascending = False, inplace = True)

  g_scores.insert(loc = 0, column = 'Total', value = g_scores.sum(axis = 1))
  g_scores.sort_values('Total', ascending = False, inplace = True)

  #need to fix this later for Roto
  x_scores = x_scores.loc[g_scores.index]

  positions = _player_means[['Position']]

  cross_player_var =  x_scores[0:n_players].var()
                          
  players_and_positions = pd.merge(x_scores
                           , positions
                           , left_index = True
                           , right_index = True)

  #get position averages, to make sure the covariance matrix measures differences relative to position
  position_means = players_and_positions[0:n_players].explode('Position').groupby('Position').mean()
  position_means = position_means - position_means.mean(axis = 0)
  #players_and_positions.loc[:,'Position'] = [x[0] for x in players_and_positions.loc[:,'Position']]
  joined = pd.merge(players_and_positions, position_means, right_index = True, left_on = 'Position', suffixes = ['_x',''])

  x_category_scores = joined.groupby('Player')[x_scores.columns].mean()
  x_scores_as_diff = (x_scores - nu * x_category_scores)[x_scores.columns]
  
  L = np.array(x_scores_as_diff.loc[x_scores.index[0:n_players]].cov()) 

  info = {'G-scores' : g_scores
          ,'Z-scores' : z_scores
          ,'X-scores' : x_scores
          ,'X-scores-as-diff' : x_scores_as_diff
          , 'Var' : cross_player_var
          , 'Positions' : positions
          , 'Mov' : mov
          , 'Vom' : vom
          , 'L' : L}

  st.session_state.info_key += 1 
  
  return info
