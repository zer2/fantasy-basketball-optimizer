#1: based on player stats: calculate coefficients and score tables and such

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

def process_player_data(player_stats, coefficients, psi):

  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics].mul(( 1- player_stats['No Play %'] * psi), axis = 0)
  player_stats[percentage_statistics] = player_stats[percentage_statistics]/100 #adjust from the display

  g_scores = calculate_scores_from_coefficients(player_stats, coefficients, 1,1)
  z_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 1,0)
  x_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 0,1)
  
  return 'Process player data'
