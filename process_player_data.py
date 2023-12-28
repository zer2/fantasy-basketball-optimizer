#1: based on player stats: calculate coefficients and score tables and such

counting_statistics = ['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
percentage_statistics = ['Free Throw %','Field Goal %']
volume_statistics = ['Free Throw Attempts','Field Goal Attempts']

def calculate_scores_from_coefficients(player_stats
                                       ,coefficients
                                       ,alpha_weight = 1
                                       ,beta_weight = 1):
    """Calculate scores based on player info and coefficients. alpha_weight is for \sigma, beta_weight is for \tau"""
        
    main_cat_mean_of_means = coefficients.loc['Mean of Means',counting_statistics]
    main_cat_var_of_means = coefficients.loc['Variance of Means',counting_statistics]
    main_cat_mean_of_vars = coefficients.loc['Mean of Variances',counting_statistics]

    main_cat_denominator = (main_cat_var_of_means.values*alpha_weight + main_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_stats.loc[:,main_categories] - main_cat_mean_of_means
    main_scores = numerator.divide(main_cat_denominator)
    main_scores['Turnovers'] = - main_scores['Turnovers']

    #free throws 
    ftp_denominator = (coefficients.loc['Variance of Means','Free Throw %']*alpha_weight + coefficients.loc['Mean of Variances','Free Throw %']*beta_weight)**0.5
    ftp_numerator = player_stats.loc[:, 'Free Throw Attempts']/coefficients.loc['Mean of Means','Free Throw Attempts'] * (player_stats['Free Throw %'] - coefficients.loc['Mean of Means','Free Throw %'])
    ftp_score = ftp_numerator.divide(ftp_denominator)

    #field goals
    fgp_denominator = (var_of_means['Field Goal %']*alpha_weight + mean_of_vars['Field Goal %']*beta_weight)**0.5
    fgp_numerator = player_stats.loc[:, 'Field Goal Attempts']/coefficients.loc['Mean of Means','Field Goal Attempts'] * (player_stats['Field Goal %']  - coefficients.loc['Mean of Means','Field Goal %'])
    fgp_score = fgp_numerator.divide(fgp_denominator)
    
    res = pd.concat([main_scores, ftp_score, fgp_score],axis = 1)  
    res.columns = counting_statistics + percentage_statistics 
    return res

def process_player_data(player_stats, coefficients, psi):

  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics] * ( 1- player_stats['No Play %'] * psi) 
  player_stats[percentage_statistics] = player_stats[percentage_statistics]/100 #adjust from the display

  g_scores = calculate_scores_from_coefficients(player_stats, coefficients, 1,1)
  z_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 1,0)
  x_scores =  calculate_scores_from_coefficients(player_stats, coefficients, 0,1)
  
  return 'Process player data'
