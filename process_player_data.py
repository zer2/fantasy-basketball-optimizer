#1: based on player stats: calculate coefficients and score tables and such

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

counting_statistics = ['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
percentage_statistics = ['Free Throw %','Field Goal %']
volume_statistics = ['Free Throw Attempts','Field Goal Attempts']

def calculate_scores_from_coefficients(player_stats
                                       ,coefficients
                                       ,alpha_weight = 1
                                       ,beta_weight = 1):
    """Calculate scores based on player info and coefficients. alpha_weight is for \sigma, beta_weight is for \tau"""
        
    main_cat_mean_of_means = coefficients.loc[counting_statistics,'Mean of Means']
    main_cat_var_of_means = coefficients.loc[counting_statistics,'Variance of Means']
    main_cat_mean_of_vars = coefficients.loc[counting_statistics,'Mean of Variances']

    main_cat_denominator = (main_cat_var_of_means.values*alpha_weight + main_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_stats.loc[:,counting_statistics] - main_cat_mean_of_means
    main_scores = numerator.divide(main_cat_denominator)
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

def process_player_data(player_stats, coefficients, psi, nu, n_drafters, n_picks):

  n_players = n_drafters * n_picks

  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics].mul(( 1- player_stats['No Play %']/100 * psi), axis = 0)
  player_stats[percentage_statistics] = player_stats[percentage_statistics]/100 #adjust from the display

  g_scores = calculate_scores_from_coefficients(player_stats, coefficients, 1,1)
  z_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 1,0)
  x_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 0,1)

  positions = player_stats[['Position']]

  score_table = x_scores.groupby([np.floor(x/12) for x in range(len(x_scores))]).agg(['mean','var'])
  diff_var = 26 + x_scores[0:n_players].var() * 13
  score_table_smoothed = x_scores.transform(lambda x: savgol_filter(x,10,1))
  players_and_positions = pd.merge(x_scores
                           , positions
                           , left_index = True
                           , right_index = True)
  
  position_means = players_and_positions[0:n_players].explode('Position').groupby('Position').mean()
  position_means = position_means - position_means.mean(axis = 0)

  players_and_positions.loc[:,'Position'] = [x[0] for x in players_and_positions.loc[:,'Position']]

  joined = pd.merge(players_and_positions, position_means, right_index = True, left_on = 'Position', suffixes = ['_x',''])

  x_category_scores = joined.groupby('Player')[x_scores.columns].mean()
  x_scores_as_diff = (x_scores - nu * x_category_scores)[x_scores.columns]
  x_scores_as_diff = x_scores_as_diff.loc[x_scores.index[0:n_players]]
  
  #get weights of X to G 
  v = np.sqrt(coefficients.loc[counting_statistics + percentage_statistics , 'Mean of Variances']/(coefficients.loc[counting_statistics + percentage_statistics , 'Mean of Variances'] + \
                                                                                                   coefficients.loc[counting_statistics + percentage_statistics , 'Variance of Means']))
  v = np.array(v/v.sum()).reshape(9,1)
  
  L = np.array(x_scores_as_diff.cov()) 
  
  return player_stats, z_scores, x_scores, positions, v, L
