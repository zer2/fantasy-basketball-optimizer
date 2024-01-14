#1: based on player stats: calculate coefficients and score tables and such

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import os
import streamlit as st

counting_statistics = ['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
percentage_statistics = ['Free Throw %','Field Goal %']
volume_statistics = ['Free Throw Attempts','Field Goal Attempts']

def calculate_coefficients(player_stats
                     , representative_player_set
                     , translation_factors ):
    """calculate the coefficients for each category- \mu,\sigma, and \tau, so we can use them for Z-scores and G-scores """

    #counting stats
    var_of_means = player_stats.loc[representative_player_set,counting_statistics].var(axis = 0)
    mean_of_means = player_stats.loc[representative_player_set,counting_statistics].mean(axis = 0)

    #free throw percent
    fta_mean_of_means = player_stats.loc[representative_player_set, 'Free Throw Attempts'].mean()
    mean_of_means.loc['Free Throw Attempts'] = fta_mean_of_means
                       
    ftp_agg_average = (player_stats.loc[representative_player_set, 'Free Throw %'] * player_stats.loc[representative_player_set, 'Free Throw Attempts']).mean()/fta_mean_of_means
    mean_of_means.loc['Free Throw %'] = ftp_agg_average
                       
    ftp_numerator = player_stats.loc[representative_player_set, 'Free Throw Attempts']/fta_mean_of_means * (player_stats.loc[representative_player_set, 'Free Throw %'] - ftp_agg_average)
    ftp_var_of_means = ftp_numerator.var()
    var_of_means.loc['Free Throw %'] = ftp_var_of_means
    
    #field goal %
    fga_mean_of_means = player_stats.loc[representative_player_set, 'Field Goal Attempts'].mean()
    mean_of_means.loc['Field Goal Attempts'] = fga_mean_of_means
                       
    fgp_agg_average = (player_stats.loc[representative_player_set, 'Field Goal %'] * player_stats.loc[representative_player_set, 'Field Goal Attempts']).mean()/fga_mean_of_means
    mean_of_means.loc['Field Goal %'] = fgp_agg_average
                       
    fgp_numerator = player_stats.loc[representative_player_set, 'Field Goal Attempts']/fga_mean_of_means * (player_stats.loc[representative_player_set, 'Field Goal %'] - fgp_agg_average)
    fgp_var_of_means = fgp_numerator.var()
    var_of_means.loc['Field Goal %'] = fgp_var_of_means

    #get mean of vars

    translation_factors['Free Throw Attempts'] = 0
    translation_factors['Field Goal Attempts'] = 0

    mean_of_vars = var_of_means * translation_factors
    #os.write(1,bytes(str(translation_factors),'utf-8'))
    #os.write(1,bytes(str(mean_of_means),'utf-8'))
    #os.write(1,bytes(str(mean_of_vars),'utf-8'))


    coefficients = pd.DataFrame({'Mean of Means' : mean_of_means
                                ,'Variance of Means' : var_of_means
                                ,'Mean of Variances' : mean_of_vars
                                }
                               )
    return coefficients
  


def calculate_scores_from_coefficients(player_stats
                                       ,coefficients
                                       ,alpha_weight = 1
                                       ,beta_weight = 1):
    """Calculate scores based on player info and coefficients. alpha_weight is for \sigma, beta_weight is for \tau"""
        
    counting_cat_mean_of_means = coefficients.loc[counting_statistics,'Mean of Means']
    counting_cat_var_of_means = coefficients.loc[counting_statistics,'Variance of Means']
    counting_cat_mean_of_vars = coefficients.loc[counting_statistics,'Mean of Variances']

    counting_cat_denominator = (counting_cat_var_of_means.values*alpha_weight + counting_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_stats.loc[:,counting_statistics] - counting_cat_mean_of_means
    main_scores = numerator.divide(counting_cat_denominator)
    main_scores['Turnovers'] = - main_scores['Turnovers']

    #free throws 
    ftp_denominator = (coefficients.loc['Free Throw %','Variance of Means']*alpha_weight + coefficients.loc['Free Throw %','Mean of Variances']*beta_weight)**0.5
    ftp_numerator = player_stats.loc[:, 'Free Throw Attempts']/coefficients.loc['Free Throw Attempts','Mean of Means'] * (player_stats['Free Throw %'] - coefficients.loc['Free Throw %','Mean of Means'])
    ftp_score = ftp_numerator.divide(ftp_denominator)

    #field goals
    fgp_denominator = (coefficients.loc['Field Goal %','Variance of Means']*alpha_weight + coefficients.loc['Field Goal %','Mean of Variances']*beta_weight)**0.5
    fgp_numerator = player_stats.loc[:, 'Field Goal Attempts']/coefficients.loc['Field Goal Attempts','Mean of Means'] * (player_stats['Field Goal %']  - coefficients.loc['Field Goal %','Mean of Means'])
    fgp_score = fgp_numerator.divide(fgp_denominator)
    
    res = pd.concat([main_scores, ftp_score, fgp_score],axis = 1)  
    res.columns = counting_statistics + percentage_statistics 
    return res

@st.cache_data
def process_player_data(player_stats
                        , conversion_factors
                        , psi
                        , nu
                        , n_drafters
                        , n_picks
                        , rotisserie):
  """Based on player stats and parameters, do all calculations to set up for running algorithms """

  n_players = n_drafters * n_picks

  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics].mul(( 1- player_stats['No Play %'] * psi), axis = 0)

  coefficients_first_order = calculate_coefficients(player_stats, player_stats.index, conversion_factors['Conversion Factor'])
  z_scores_first_order =  calculate_scores_from_coefficients(player_stats, coefficients_first_order, 1,0)
  first_order_score = z_scores_first_order.sum(axis = 1)
  representative_player_set = first_order_score.sort_values(ascending = False).index[0:n_picks * n_drafters]

  coefficients = calculate_coefficients(player_stats, representative_player_set, conversion_factors['Conversion Factor'])
                         
  g_scores = calculate_scores_from_coefficients(player_stats, coefficients, 1,1)
  z_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 1,0)
  x_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 0,1)

  #Design the score table based on what we expect other drafters to use. 
  #Z-score for rotisserie, otherwise G-score
  if rotisserie:
    x_scores = x_scores.loc[z_scores.sum(axis = 1).sort_values(ascending = False).index]
  else:
    x_scores = x_scores.loc[g_scores.sum(axis = 1).sort_values(ascending = False).index]

  os.write(1,bytes(str(x_scores),'utf-8'))

  positions = player_stats[['Position']]

  score_table = x_scores.groupby([np.floor(x/n_drafters) for x in range(len(x_scores))]).agg(['mean','var'])
  score_table_smoothed = x_scores.transform(lambda x: savgol_filter(x,10,1))
  weekly_var = 0 if rotisserie else n_picks * 2
  diff_var =  weekly_var + x_scores[0:n_players].var() * n_picks
                          
  players_and_positions = pd.merge(x_scores
                           , positions
                           , left_index = True
                           , right_index = True)

  #get position averages, to make sure the covariance matrix measures differences relative to position
  position_means = players_and_positions[0:n_players].explode('Position').groupby('Position').mean()
  position_means = position_means - position_means.mean(axis = 0)
  #players_and_positions.loc[:,'Position'] = [x[0] for x in players_and_positions.loc[:,'Position']]
  joined = pd.merge(players_and_positions, position_means, right_index = True, left_on = 'Position', suffixes = ['_x',''])

  os.write(1,bytes(str(players_and_positions),'utf-8'))

  x_category_scores = joined.groupby('Player')[x_scores.columns].mean()
  x_scores_as_diff = (x_scores - nu * x_category_scores)[x_scores.columns]
  x_scores_as_diff = x_scores_as_diff.loc[x_scores.index[0:n_players]]
  
  mov = coefficients.loc[counting_statistics + percentage_statistics , 'Mean of Variances']
  vom = coefficients.loc[counting_statistics + percentage_statistics , 'Variance of Means']
  if rotisserie:  #get weights of X to Z 
    v = np.sqrt(mov/vom)  
  else:   #get weights of X to G 
    v = np.sqrt(mov/(mov + vom))

  v = np.array(v/v.sum()).reshape(9,1)
  
  L = np.array(x_scores_as_diff.cov()) 

  z_scores.loc[:,'Total'] = z_scores.sum(axis = 1)
  z_scores.sort_values('Total', ascending = False, inplace = True)

  g_scores.loc[:,'Total'] = g_scores.sum(axis = 1)
  g_scores.sort_values('Total', ascending = False, inplace = True)

  #os.write(1,b'HIII')

  info = {'G-scores' : g_scores
          ,'Z-scores' : z_scores
          ,'X-scores' : x_scores
          , 'Score-table' : score_table
          , 'Score-table-smoothed' : score_table_smoothed
          , 'Diff-var' : diff_var
          , 'Positions' : positions
          , 'v' : v
          , 'L' : L}
  
  return info
