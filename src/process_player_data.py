import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from src.helper_functions import get_selected_counting_statistics, get_selected_ratio_statistics, get_selected_categories\
                                ,get_position_structure, weighted_cov_matrix, increment_info_key, get_counting_statistics\
                                ,get_ratio_statistics
import os
import streamlit as st
import sys

def calculate_coefficients(player_means : pd.DataFrame
                     , representative_player_set : list
                     , translation_factors : pd.Series
                     ) -> dict:
    """calculate the coefficients for each category- \mu,\sigma, and \tau, so we can use them for Z-scores and G-scores

    Args:
        player_means: dataframe of fantasy-relevant statistics 
        representative_player_set: list of players to use as sample for coefficients
        translation_factors: series, converts sigma^2 to tau^2
        categories: list of categories for which to calculate coefficients 
    Returns:
        Dictionary mapping 'Mean of Means' -> (series mapping category to /mu^2 etc.) 

    """

    counting_statistics = get_selected_counting_statistics()
    ratio_statistics = get_selected_ratio_statistics()

    #ZR: We should really have a 'get_position_structure' equivalent function for this 
    params = st.session_state['params']

    #counting stats
    var_of_means = player_means.loc[representative_player_set,counting_statistics].var(axis = 0)
    mean_of_means = player_means.loc[representative_player_set,counting_statistics].mean(axis = 0)

    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():

        if ratio_stat in ratio_statistics:
            volume_statistic = ratio_stat_info['volume-statistic']

            volume_mean_of_means = player_means.loc[representative_player_set, volume_statistic].mean()
            mean_of_means.loc[volume_statistic] = volume_mean_of_means
                            
            agg_average = (player_means.loc[representative_player_set, ratio_stat] * \
                                player_means.loc[representative_player_set, volume_statistic]).mean()/volume_mean_of_means
            mean_of_means.loc[ratio_stat] = agg_average
                            
            numerator = player_means.loc[representative_player_set, volume_statistic]/volume_mean_of_means * \
                                        (player_means.loc[representative_player_set, ratio_stat] - agg_average)
            ratio_var_of_means = numerator.var()
            var_of_means.loc[ratio_stat] = ratio_var_of_means

            if volume_statistic not in translation_factors:
                translation_factors[volume_statistic] = 0

    #get mean of vars
    mean_of_vars = var_of_means * translation_factors

    coefficients = pd.DataFrame({'Mean of Means' : mean_of_means
                                ,'Variance of Means' : var_of_means
                                ,'Mean of Variances' : mean_of_vars
                                }
                               )
    

    return coefficients

def calculate_coefficients_historical(weekly_df : pd.DataFrame
                     , representative_player_set : list
                     , params : dict
                     , coefficient_exploration_mode : bool = False
                     ) -> dict:
    """calculate the coefficients from a real weekly dataset by week

    Args:
        weekly_df: dataframe of fantasy-relevant statistics 
        representative_player_set: list of players to use as sample for coefficients
    Returns:
        Dictionary mapping 'Mean of Means' -> (series mapping category to /mu^2 etc.) 

    """
    if coefficient_exploration_mode:
        counting_statistics = get_counting_statistics()
        ratio_statistics = get_ratio_statistics()
    else:
        counting_statistics = get_selected_counting_statistics()
        ratio_statistics = get_selected_ratio_statistics()
        
    player_stats = weekly_df.groupby(level = 'Player').agg(['mean','var'])

    #counting stats
    mean_of_vars = player_stats.loc[representative_player_set,(counting_statistics,'var')].mean(axis = 0)
    var_of_means = player_stats.loc[representative_player_set,(counting_statistics,'mean')].var(axis = 0)
    mean_of_means = player_stats.loc[representative_player_set,(counting_statistics,'mean')].mean(axis = 0)

    #there should be a better way to do this loop- maybe with something akin to get_position_structure
    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        if ratio_stat in ratio_statistics:

            volume_statistic = ratio_stat_info['volume-statistic']
            made_statistic = ratio_stat_info['made-statistic']

            made_mean_of_means = player_stats.loc[representative_player_set, (made_statistic,'mean')].mean()
            volume_mean_of_means = player_stats.loc[representative_player_set, (volume_statistic,'mean')].mean()

            mean_of_means.loc[volume_statistic] = volume_mean_of_means
            ratio_agg_average = made_mean_of_means / volume_mean_of_means

            mean_of_means.loc[ratio_stat] = ratio_agg_average

            ratio = player_stats.loc[:, (made_statistic,'mean')]/player_stats.loc[:, (volume_statistic,'mean')]
            ratio_numerator = player_stats.loc[:, (volume_statistic,'mean')]/volume_mean_of_means * (ratio - ratio_agg_average)
            ratio_var_of_means = ratio_numerator.loc[representative_player_set].var()
            var_of_means.loc[ratio_stat] = ratio_var_of_means

            weekly_df.loc[:,'volume_adjusted_' + ratio_stat] = (weekly_df[made_statistic] - weekly_df[volume_statistic]*ratio_agg_average)/ \
                                                volume_mean_of_means
            ratio_mean_of_vars = weekly_df['volume_adjusted_' + ratio_stat].loc[representative_player_set].groupby('Player').var().mean()    
            mean_of_vars.loc[ratio_stat] = ratio_mean_of_vars

    coefficients = pd.DataFrame({'Mean of Means' : mean_of_means.droplevel(level = 1)
                            ,'Variance of Means' : var_of_means.droplevel(level = 1)
                            ,'Mean of Variances' : mean_of_vars.droplevel(level = 1)
                            }
                            )
    
    if coefficient_exploration_mode:
       print(coefficients)
       print(coefficients['Mean of Variances']/coefficients['Variance of Means'])
       sys.exit()

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
    counting_statistics = get_selected_counting_statistics()
    ratio_statistics = get_selected_ratio_statistics()

    counting_cat_mean_of_means = coefficients.loc[counting_statistics,'Mean of Means']
    counting_cat_var_of_means = coefficients.loc[counting_statistics,'Variance of Means']
    counting_cat_mean_of_vars = coefficients.loc[counting_statistics,'Mean of Variances']

    counting_cat_denominator = (counting_cat_var_of_means.values*alpha_weight + counting_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_means.loc[:,counting_statistics] - counting_cat_mean_of_means
    main_scores = numerator.divide(counting_cat_denominator)

    ratio_scores = {}
    for ratio_stat, ratio_stat_info in params['ratio-statistics'].items():
        if ratio_stat in ratio_statistics:

            volume_statistic = ratio_stat_info['volume-statistic']

            denominator = (coefficients.loc[ratio_stat,'Variance of Means']*alpha_weight + coefficients.loc[ratio_stat,'Mean of Variances']*beta_weight)**0.5
            numerator = player_means.loc[:, volume_statistic]/coefficients.loc[volume_statistic,'Mean of Means'] * \
                                        (player_means[ratio_stat] - coefficients.loc[ratio_stat,'Mean of Means'])
            ratio_scores[ratio_stat] = numerator.divide(denominator)
        
    res = pd.concat([ratio_scores[ratio_stat] for ratio_stat in ratio_scores.keys() ] + [main_scores],axis = 1)  

    res.columns = ratio_statistics + counting_statistics

    for negative_stat in params['negative-statistics']:
        if negative_stat in res.columns:
            res[negative_stat] = - res[negative_stat]
    return res.fillna(0)[get_selected_categories()]

def games_played_adjustment(scores : pd.DataFrame
                      , replacement_games_rate : pd.Series
                      , representative_player_set : list[str]
                      , params : dict
                      , v : pd.Series = None) -> pd.DataFrame :
    """Applies injury adjustment formula based on effective games played. 
    Requires scores to be in same order as replacement_player_value

    Args:
        scores: Dataframe of category-level scores. One row per player, one column per statistic 
        replacement_player_value: Series of rates between 0 and 1. That rate of games are assumed to be filled in by a replacement-level player
        representative_player_set: 
        v: series of weights per category. Only applicable if scores don't translate precisely to overall value 

    Returns:
        Dataframe with same dimensions as scores 
    """

    all_stats = get_selected_categories()

    if v is None:
       v = pd.Series({stat : 1/9 for stat in all_stats})

    totals = scores.dot(v)

    n_players = len(representative_player_set)
    rv = totals.sort_values(ascending = False).iloc[n_players]
    category_level_rv = get_category_level_rv(rv, v, params)

    replacement_player_value = np.array(category_level_rv.T).reshape(1,-1) * \
                                np.array(replacement_games_rate).reshape(-1,1)
    adjusted_scores =  scores + replacement_player_value

    #we need to re-normalize because we didn't account for other players being able to bring in replacements before
    adjusted_scores = adjusted_scores - adjusted_scores.loc[representative_player_set].mean() 

    return adjusted_scores 

def get_category_level_rv(rv : float, v : pd.Series, params : dict = None):
   all_stats = get_selected_categories()
   rv_multiple = rv/(len(all_stats) -2) if  'Turnovers' in all_stats else rv/len(all_stats)
   value_by_category = pd.Series({stat : - rv_multiple/v[stat] if stat == 'Turnovers' 
                                  else rv_multiple/v[stat] for stat in all_stats})
   
   return value_by_category

@st.cache_data(show_spinner = False, ttl = 3600)
def process_player_data(weekly_df : pd.DataFrame
                        , _player_means : pd.DataFrame
                        , conversion_factors :pd.Series
                        , upsilon : float
                        , psi : float
                        , n_drafters : int
                        , n_picks : int
                        , params : dict
                        , player_stats_key
                        , coefficient_exploration_mode = False
                        ) -> dict:
  """Based on player stats and parameters, do all calculations to set up for running algorithms

  Args:
      weekly_df: Dataframe of fantasy-relevant statistics at the weekly level 
      player_means: Dataframe of player means
      conversion_factors: Conversion factors for /sigma^2 to /tau^2. Needed if weekly_df is None
      psi: parameter scaling the influence of no_play rate
      nu: parameter scaling the influence of category means in covariance calculation
      n_drafters: number of drafters
      n_picks: number of picks per drafter
      player_stats_key: key for version number of player stats. Used to check if player stats has changed
  Returns:
      Info dictionary with many pieces of information relevant to the algorithm 
  """
  n_players = n_drafters * n_picks

  if weekly_df is not None:
    coefficients_first_order = calculate_coefficients_historical(weekly_df
                                                , pd.unique(weekly_df.index.get_level_values('Player'))
                                                , params)
  else:
    coefficients_first_order = calculate_coefficients(_player_means
                                                  , _player_means.index
                                                  , conversion_factors)
                
  g_scores_first_order =  calculate_scores_from_coefficients(_player_means
                                                          , coefficients_first_order
                                                          , params
                                                          , 1
                                                          ,1)
    
  first_order_score = g_scores_first_order.sum(axis = 1)
  representative_player_set = first_order_score.sort_values(ascending = False).index[0:n_picks * n_drafters]

  if weekly_df is not None:
    coefficients = calculate_coefficients_historical(weekly_df
                                                , representative_player_set
                                                , params
                                                , coefficient_exploration_mode)
  else:
    coefficients = calculate_coefficients(_player_means
                                                  , representative_player_set
                                                  , conversion_factors)
    
    
  mov = coefficients.loc[get_selected_categories() , 'Mean of Variances']
  vom = coefficients.loc[get_selected_categories() , 'Variance of Means']
  v = np.sqrt(mov/(mov + vom)) #ZR: This doesn't work for Roto. We need to fix that later
  v = v/v.sum()

  g_scores = calculate_scores_from_coefficients(_player_means, coefficients, params, 1,1)
  z_scores =  calculate_scores_from_coefficients(_player_means, coefficients, params,  1,0)
  x_scores =  calculate_scores_from_coefficients(_player_means, coefficients, params, 0,1)

  replacement_games_rate = (1- _player_means['Games Played %']/100) * psi
  g_scores = games_played_adjustment(g_scores, replacement_games_rate,representative_player_set, params)
  z_scores = games_played_adjustment(z_scores, replacement_games_rate,representative_player_set, params)
  x_scores = games_played_adjustment(x_scores, replacement_games_rate,representative_player_set, params, v = v)

  z_scores.insert(loc = 0, column = 'Total', value = z_scores.sum(axis = 1))

  z_scores.sort_values('Total', ascending = False, inplace = True)

  g_scores.insert(loc = 0, column = 'Total', value = g_scores.sum(axis = 1))
  g_scores.sort_values('Total', ascending = False, inplace = True)

  #need to fix this later for Roto
  x_scores = x_scores.loc[g_scores.index]

  positions = _player_means['Position'].str.split(',')

  cross_player_var =  x_scores[0:n_players].var()
                          
  #get position averages, to make sure the covariance matrix measures differences relative to position
  #we need to weight averages to avoid over-counting the players that can take multiple positions
  # 
   
  try: 
    players_and_positions = pd.merge(x_scores
                        , positions
                        , left_index = True
                        , right_index = True)

    players_and_positions_limited = players_and_positions[0:n_players]
    categories = get_selected_categories()
    players_and_positions_limited[categories] = players_and_positions_limited[categories] \
                                                    .sub(players_and_positions_limited[categories].mean(axis = 0))
    positions_exploded = players_and_positions_limited.explode('Position').reset_index().set_index(['Player','Position'])
    position_mean_weights = 1/positions_exploded.groupby('Player').transform('count')
    position_means_weighted = positions_exploded.mul(position_mean_weights)

    position_means = position_means_weighted.groupby('Position').sum()/position_mean_weights.groupby('Position').sum()
    positions_exploded = positions_exploded.sub(positions_exploded.mean(axis = 0)) #normalize by mean of the category 

    #we should have some logic for position not being available
    #also all of the position rules should be modularized 
    base_position_list = get_position_structure()['base_list']
    position_means = position_means.loc[base_position_list, :] #this is the order we always use for positions
    
    position_means_g = position_means * v
    position_means_g = position_means_g.sub(position_means_g.mean(axis = 1), axis = 0)
    position_means_g = position_means_g.sub(position_means_g.mean(axis = 0), axis = 1) #experimental
    position_means = position_means_g / v

    L_by_position = pd.concat({position : weighted_cov_matrix(positions_exploded.loc[pd.IndexSlice[:,position],:]
                                                                , position_mean_weights.loc[pd.IndexSlice[:,position]
                                                                                            ,position_mean_weights.columns[0]]) 
                                                                for position in base_position_list}
                                )
  except:
    position_means = None        
    L_by_position = np.array([x_scores.cov()])
    
  info = {'G-scores' : g_scores
          ,'Z-scores' : z_scores
          ,'X-scores' : x_scores
          , 'Var' : cross_player_var
          , 'Positions' : positions
          , 'Mov' : mov
          , 'Vom' : vom
          , 'Position-Means' : position_means
          , 'L-by-Position' : L_by_position
          , 'Positions' : positions}

  increment_info_key()
  
  return info
