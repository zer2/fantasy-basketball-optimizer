
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner = False)
def estimate_matchup_result(team_1_x_scores : pd.Series
                            , team_2_x_scores : pd.Series
                            , n_picks : int
                            , scoring_format : str) -> float:
    """Based on X scores, estimates the result of a matchup. Chance that team 1 will beat team 2

    Args:
      team_1_x_scores: Series of x-scores for one team
      team_2_x_scores: Series of x-scores for other team
      n_picks: number of players on each team
      scoring_format: format to use for analysis

    Returns:
      Dictionary with results of the trade
    """

    cdf_estimates = pd.DataFrame(norm.cdf(team_1_x_scores - team_2_x_scores
                                        , scale = np.sqrt(n_picks*2)
                                        )
                            ).T

    cdf_array = np.expand_dims(np.array(cdf_estimates),2)

    if scoring_format == 'Head to Head: Most Categories':
        score = combinatorial_calculation(cdf_array
                                                    , 1 - cdf_array
                        )

    else:
        score = cdf_array.mean() 

    cdf_estimates.columns = get_categories()
    return float(score), cdf_estimates


def analyze_trade(team_1
                  ,team_1_trade : list[str]
                  ,team_2
                  ,team_2_trade : list[str]
                  ,H
                  ,player_stats : pd.DataFrame
                  ,player_assignments : dict[list[str]]
                  ,n_iterations : int
                  ) -> dict:    

    """Compute the results of a potential trade

    Args:
      team_1_other: remaining players, not to be traded from the first team
      team_1_trade: player(s) to be traded from the first team
      team_2_other: remaining players, not to be traded from the first team
      team_2_trade: player(s) to be traded from the second team
      H: H-scoring agent, which can be used to calculate H-score 
      player_stats: DataFrame of player statistics 
      players_chosen: list of all chosen players
      n_iterations: int, number of gradient descent steps

    Returns:
      Dictionary with results of the trade
    """


    post_trade_team_1 = [p for p in player_assignments[team_1] if p not in team_1_trade] + team_2_trade
    post_trade_team_2 = [p for p in player_assignments[team_2] if p not in team_2_trade] + team_1_trade

    post_trade_assignments = player_assignments.copy()

    post_trade_assignments[team_1] = post_trade_team_1
    post_trade_assignments[team_2] = post_trade_team_2

    res_1_1 = next(H.get_h_scores(player_assignments, team_1))
    res_2_2 = next(H.get_h_scores(player_assignments, team_2))
 
    n_player_diff = len(team_1_trade) - len(team_2_trade)

    if n_player_diff > 0:
        generator = H.get_h_scores(post_trade_assignments, team_1)
        for i in range(n_iterations):
            res_1_2  = next(generator)
        
        res_2_1 = next(H.get_h_scores(post_trade_assignments, team_2))

    elif n_player_diff == 0:
        res_1_2 = next(H.get_h_scores(post_trade_assignments, team_1))
        res_2_1 = next(H.get_h_scores(post_trade_assignments, team_2))

    else:
        res_1_2 = next(H.get_h_scores(post_trade_assignments, team_1))

        generator = H.get_h_scores(post_trade_assignments, team_2)
        for i in range(n_iterations):
            res_2_1= next(generator)
    
    #helper function just for this procedure
    def get_full_row(scores, rates):

        idxmax = scores.idxmax()
        score = pd.Series([scores[idxmax]], index = ['H-score'])
        rate = rates.loc[idxmax]

        return pd.concat([score, rate])

    team_1_info = {'pre' : get_full_row(res_1_1['Scores'], res_1_1['Rates'])
                        ,'post' : get_full_row(res_1_2['Scores'], res_1_2['Rates'])}
    team_2_info = {'pre' : get_full_row(res_2_2['Scores'], res_2_2['Rates'])
                        ,'post' : get_full_row(res_2_1['Scores'], res_2_1['Rates'])}
                      
    results_dict = {1 : team_1_info
                    ,2 : team_2_info
                   }

    return results_dict
                
def analyze_trade_value(player : str
                  ,team : str
                  ,H
                  ,player_stats : pd.DataFrame
                  ,player_assignments : dict[list[str]]
                  ) -> float:    

    """Estimate how valuable a player would be to a particular team

    Args:
      player: player to evaluate
      rest_of_team: other player(s) on team
      H: H-scoring agent, which can be used to calculate H-score 
      player_stats: DataFrame of player statistics 
      players_chosen: list of all chosen players

    Returns:
      Float, relative H-score value
    """

    without_player = player_assignments.copy()
    without_player[team] = [p for p in without_player[team] if p != player]

    with_player = player_assignments.copy()
    if player not in with_player[team]:
        with_player[team] = with_player[team] + [player]


    res_without_player= next(H.get_h_scores(without_player,team, exclusion_list = [player]))
    res_with_player = next(H.get_h_scores(with_player, team))

    res = (res_with_player['Scores'].max() - res_without_player['Scores'].max())

    return res