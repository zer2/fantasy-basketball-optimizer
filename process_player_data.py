#1: based on player stats: calculate coefficients and score tables and such

counting_statistics = ['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
percentage_statistics = ['Free Throw %',''Field Goal %']
volume_statistics = ['Free Throw Attempts',''Field Goal Attempts']

def calculate_scores_from_coefficients(player_stats
                                       ,mean_of_means
                                       ,var_of_means
                                       ,mean_of_vars
                                       ,alpha_weight = 1
                                       ,beta_weight = 1):
    """Calculate scores based on player info and coefficients. alpha_weight is for \sigma, beta_weight is for \tau"""
        
    main_cat_mean_of_means = mean_of_means.loc[counting_statistics]
    main_cat_var_of_means = var_of_means.loc[counting_statistics]
    main_cat_mean_of_vars = mean_of_vars.loc[counting_statistics]

    main_cat_denominator = (main_cat_var_of_means.values*alpha_weight + main_cat_mean_of_vars.values*beta_weight ) ** 0.5
    numerator = player_stats.loc[:,main_categories] - main_cat_mean_of_means
    main_scores = numerator.divide(main_cat_denominator)
    main_scores['Turnovers'] = - main_scores['Turnovers']

    #free throws 
    ftp_denominator = (var_of_means['Free Throw %']*alpha_weight + mean_of_vars['Free Throw %']*beta_weight)**0.5
    ftp_numerator = player_stats.loc[:, 'Free Throw Attempts']/mean_of_means['Free Throw Attempts'] * (player_stats['Free Throw %'] - mean_of_means['Free Throw %'])
    ftp_score = ftp_numerator.divide(ftp_denominator)

    #field goals
    fgp_denominator = (var_of_means['Field Goal %']*alpha_weight + mean_of_vars['Field Goal %']*beta_weight)**0.5
    fgp_numerator = player_stats.loc[:, 'Field Goal Attempts']/mean_of_means['Field Foal Attempts'] * (player_stats['Field Goal %']  - mean_of_means['Field Goal %'])
    fgp_score = fgp_numerator.divide(fgp_denominator)
    
    res = pd.concat([main_scores, ftp_score, fgp_score],axis = 1)  
    res.columns = counting_statistics + percentage_statistics 
    return res

def process_player_data(edited_df):

  adjusted_df = edited_df.copy()
  adjusted_df[counting_statistics + volume_statistics] = adjusted_df[counting_statistics + volume_statistics] * ( 1- adjusted_df['No Play %'] * psi) 

  (adjusted_df[['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']] - means_of_means[counting_statistics]

  g_scores = calculate_scores_from_coefficients(adjusted_df, mean_of_means, var_of_means, mean_of_vars, 1,1)
  z_scores =  calculate_scores_from_coefficients(adjusted_df, mean_of_means, var_of_means, mean_of_vars, 1,0)
  x_scores =  calculate_scores_from_coefficients(adjusted_df, mean_of_means, var_of_means, mean_of_vars, 0,1)
  
  return 'Process player data'
