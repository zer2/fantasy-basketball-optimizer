import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from itertools import combinations
from src.helper_functions import combinatorial_calculation, calculate_tipping_points, get_categories
import streamlit as st 
import numexpr as ne

class HAgent():

    def __init__(self
                 , info : dict
                 , omega : float
                 , gamma : float
                 , alpha : float
                 , beta : float
                 , n_picks : int
                 , n_drafters : int
                 , winner_take_all : bool
                 , punting : bool
                    ):
        """Calculates the rank order based on H-score

        Args:
            info: dictionary with info related to player statistics etc. 
            omega: float, parameter as described in the paper
            gamma: float, parameter as described in the paper
            alpha: float, step size parameter for gradient descent 
            beta: float, decay parameter for gradient descent 
            n_picks: int, number of picks each drafter gets 
            n_drafters : int, number of drafters
            winner_take_all: Boolean of whether to optimize for the winner-take-all format
                             If False, optimizes for total categories
            punting: boolean for whether to adjust expectation of future picks by formulating a punting strategy
        Returns:
            None

        """
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.n_picks = n_picks 
        self.n_drafters = n_drafters
        self.winner_take_all = winner_take_all 
        self.x_scores = info['X-scores']
        self.score_table = info['Score-table']
        self.score_table_smoothed = info['Score-table-smoothed']
        self.diff_var = info['Diff-var']
        self.v = info['v']
        self.L = info['L']
        self.punting = punting
      
    def get_h_scores(self
                  , player_assignments : dict[list[str]]
                  , drafter
                  , exclusion_list = []
                  ) -> tuple: 

        """Performs the H-score algorithm

        Args:
            player_assignments : dictionary of form team -> list of players chosen by that team 
            player: which drafter to perform H-scoring for

        Returns:
            String indicating chosen player
        """
        my_players = player_assignments[drafter]
        n_players_selected = len(my_players) 

        players_chosen = [x for v in player_assignments.values() for x in v if x == x]
        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen + exclusion_list)]

        #we want to use the smoothed score table when the expectation for player strength is different depending on how far into the round you are drafting
        #for the last round, it doesn't really matter, because there are no later rounds to balance it out 

        diff_means = self.get_diff_means(player_assignments
                                        , drafter
                                        , x_scores_available
        )
        
        x_scores_available_array = np.expand_dims(np.array(x_scores_available), axis = 2)

        default_weights = self.v.T.reshape(1,9,1)
        initial_weights = ((diff_means + x_scores_available_array)/((default_weights * 500)) + \
                            default_weights).mean(axis = 2)
        
        scores = []
        weights = []

        return self.perform_iterations(initial_weights
                                       ,my_players
                                       , n_players_selected
                                       , diff_means
                                       , x_scores_available_array
                                       , x_scores_available.index)

    def get_diff_means(self
                    , player_assignments : dict[list[str]]
                    , drafter
                    , x_scores_available : pd.DataFrame) -> pd.Series:
        """Calculates diff-means based on the dynamic method 

        Args:
            players_chosen : list of all players chosen, including my_players
            x_scores_available: DataFrame of x-scores, excluding players chosen by any team
            my_players : list of players chosen already by this team

        Returns:
            Series of form {cat : expected value of opposing teams for the cat}
        """
        my_players = player_assignments[drafter]
        x_self_sum = np.array(self.x_scores.loc[my_players].sum(axis = 0))

        #assume that players for the rest of the round will be chosen from the default ordering 
        players_chosen = [x for v in player_assignments.values() for x in v if x == x]
        extra_players_needed = (len(my_players)+1) * self.n_drafters - len(players_chosen) - 1
        mean_extra_players = x_scores_available.iloc[0:extra_players_needed].mean()

        other_team_sums = np.vstack(
            [self.get_opposing_team_stats(players, mean_extra_players, len(my_players)) for team, players \
                                    in player_assignments.items() if team != drafter]
        ).T
        
        diff_means = x_self_sum.reshape(1,9,1) - other_team_sums.reshape(1,9,self.n_drafters - 1)

        return diff_means

    def get_opposing_team_stats(self, players, replacement_value, n_players):
        players = [p for p in players if p == p]

        n_extra_players = n_players + 1 - len(players)

        opposing_team_stats = np.array(self.x_scores.loc[players].sum(axis =0) + \
                                        n_extra_players * replacement_value)

        return opposing_team_stats 

    def get_diff_means_old(self
                    , player_assignments : dict[list[str]]
                    , drafter
                    , x_scores_available : pd.DataFrame) -> pd.Series:
        """Calculates diff-means based on the dynamic method 

        Args:
            players_chosen : list of all players chosen, including my_players
            x_scores_available: DataFrame of x-scores, excluding players chosen by any team
            my_players : list of players chosen already by this team

        Returns:
            Series of form {cat : expected value of opposing teams for the cat}
        """
        my_players = player_assignments[drafter]
        x_self_sum = self.x_scores.loc[my_players].sum(axis = 0)

        players_chosen = [x for v in player_assignments.values() for x in v if x == x]
        sum_so_far = self.x_scores.loc[players_chosen].sum()

        #assume that players for the rest of the round will be chosen from the default ordering 
        extra_players_needed = (len(my_players)+1) * self.n_drafters - len(players_chosen)
        sum_extra_players = x_scores_available.iloc[0:extra_players_needed].sum()

        average_team_so_far = (sum_so_far + sum_extra_players)/self.n_drafters

        diff_means = np.array(x_self_sum - average_team_so_far).reshape(1,9,1)

        #currently diff_means is one-dimensional 
        #should switch diff_means to be two-dimension, but not aligned with x-candidates
        #so it should be of size (1,9,n_opponents)

        return diff_means 

    def perform_iterations(self
                           ,weights : pd.DataFrame
                           ,my_players : list[str]
                           ,n_players_selected : int
                           ,diff_means : pd.Series
                           ,x_scores_available_array : pd.DataFrame
                           ,result_index
                           ) -> tuple:
        """Performs one iteration of H-scoring
         
         Case (1): If n_players_selected < n_picks -1, the Gaussian multivariate assumption is used for future picks and weight is chosen by gradient descent
         Case (2): If n_players_selected = n_picks -1, each candidate player is evaluated with no need for modeling future picks
         Case (3): If n_players_selected = n_picks, a single number is returned for the team's total H-score
         Case (4): If n_players_selected > n_picks, all subsets of possible players are evaluated for the best subset

        Args:
            weights: Starting choice of weights. Relevant for case (1)
            my_players: list of players selected by the current drafter
            n_players_selected: integer, number of players already selected by the current drafter 
                                This is a param in addition to my_players because n_players_selected is already calculated in the parent function
            diff_means: series, difference in mean between already selected players and expected
            x_scores_available: dataframe, X-scores of unselected players
        Yields:
            Ultimate H-scores, weights used to make those H-scores, and approximate win fractions given those weights

        """
        i = 0
        
        while True:

            #case where many players need to be chosen
            if (n_players_selected < self.n_picks - 1) & (self.punting):

                del_full = self.get_del_full(weights)
        
                expected_future_diff_single = self.get_x_mu_simplified_form(weights)

                expected_future_diff = ((12-n_players_selected) * expected_future_diff_single).reshape(-1,9,1)

                x_diff_array = diff_means + x_scores_available_array + expected_future_diff

                pdf_estimates = self.get_pdf(x_diff_array)
                
                cdf_estimates = self.get_cdf(x_diff_array)
        
                if self.winner_take_all:
        
                    tipping_points = calculate_tipping_points(np.array(cdf_estimates))   
        
                    pdf_weights = (tipping_points*pdf_estimates).mean(axis = 2)
                else:
                    pdf_weights = pdf_estimates.mean(axis = 2)

                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)
        
                step_size = self.alpha * (i + 1)**(-self.beta) 
                change_weights = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                weights = weights + change_weights
                weights[weights < 0] = 0
                weights = weights/weights.sum(axis = 1).reshape(-1,1)

                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                 ).mean(axis = 1)
                else:
                    score = cdf_estimates.mean(axis = 2).mean(axis = 1) 

            #case where one more player needs to be chosen
            elif (n_players_selected == (self.n_picks - 1)) | ((not self.punting) & (n_players_selected < (self.n_picks)) ): 

                x_diff_array = diff_means + x_scores_available_array

                cdf_estimates = self.get_cdf(x_diff_array)

                weights = None
                expected_future_diff = None
                
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                 ).mean(axis = 1)
                else:
                    score = cdf_estimates.mean(axis = 2).mean(axis = 1) 

            #case where no new players need to be chosen
            elif (n_players_selected == self.n_picks): 

                cdf_estimates = self.get_cdf(diff_means)

                weights = None
                expected_future_diff = None
                
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                 ).mean(axis = 1)

                else:

                    score = cdf_estimates.mean(axis = 2).mean(axis = 1) 
                
                result_index = ['']

            #case where there are too many players and some need to be removed. n > n_picks
            else: 

                extra_players = n_players_selected - self.n_picks 
                players_to_remove_possibilities = combinations(my_players,extra_players)

                drop_potentials = pd.concat(
                    (self.x_scores.loc[list(players_to_remove)].sum(axis = 0) \
                    for players_to_remove in players_to_remove_possibilities
                    )
                                                       ,axis = 1).T
                drop_potentials_array = np.expand_dims(np.array(drop_potentials), axis = 2)
                diff_means_mod = diff_means - drop_potentials_array

                cdf_estimates = self.get_cdf(diff_means_mod)
                                        
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                 ).mean(axis = 1)
                else:
                    score = cdf_estimates.mean(axis = 2).mean(axis = 1)

                result_index = drop_potentials.index

                weights = None
                expected_future_diff = None

            i = i + 1

            cdf_means = cdf_estimates.mean(axis = 2)

            if expected_future_diff is not None:
                expected_diff_means = expected_future_diff.mean(axis = 2)
            else:
                expected_diff_means = None

            yield {'Scores' : pd.Series(score, index = result_index)
                    ,'Weights' : pd.DataFrame(weights, index = result_index, columns = get_categories())
                    ,'Rates' : pd.DataFrame(cdf_means, index = result_index, columns = get_categories())
                    ,'Diff' : pd.DataFrame(expected_diff_means, index = result_index, columns = get_categories())}

    ### below are functions used for the optimization procedure 
    def get_pdf(self, x_diff_array):

        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]*x_diff_array.shape[2], -1)

        pdf_estimates = norm.pdf(diff_array_reshaped, scale = np.sqrt(self.diff_var))

        pdf_estimates_reshaped = pdf_estimates.reshape(x_diff_array.shape)

        return pdf_estimates_reshaped
    
    def get_cdf(self, x_diff_array):

        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]*x_diff_array.shape[2], -1)

        cdf_estimates = norm.cdf(diff_array_reshaped, scale = np.sqrt(self.diff_var))

        cdf_estimates_reshaped = cdf_estimates.reshape(x_diff_array.shape)

        return cdf_estimates_reshaped

    def get_x_mu_long_form(self,c):
        #uses the pre-simplified formula for x_mu from page 19 of the paper

        factor = (self.v.dot(self.v.T).dot(self.L).dot(c.T)/self.v.T.dot(self.L).dot(self.v)).T

        c_mod = c - factor
        sigma = np.sqrt((c_mod.dot(self.L) * c_mod).sum(axis = 1))
        
        U = np.array([[self.v.reshape(9),c_0.reshape(9)] for c_0 in c])
        b = np.array([[-self.gamma * s,self.omega * s] for s in sigma]).reshape(-1,2,1)
        U_T = np.swapaxes(U, 1, 2)
        
        q = np.einsum('aij, ajk -> aik', U.dot(self.L), U_T)
        inverse_part = np.linalg.inv(q)

        r = np.einsum('ij, ajk -> aik', self.L, U_T)

        x = np.einsum('aij, ajk -> aik', r, inverse_part)

        x_mu = np.einsum('aij, ajk -> aik', x, b)

        return x_mu

    def get_x_mu_simplified_form(self,c):
        last_four_terms = self.get_last_four_terms(c)
        x_mu = np.einsum('ij, ajk -> aik',self.L, last_four_terms)
        return x_mu


    #below functions use the simplified form of X_mu 
    #term 1: L (covariance)
    #term 2: vj^T - jv^T
    #term 3: L (covariance)
    #term 4: -gamma * j - omega * v
    #term 5: sigma / (j^T L j v^T L V - (v^T L j)^2) 

    def get_term_two(self,c):
        #v = self.v.reshape(9,1)

        return - self.v.reshape(-1,9,1) * c.reshape(-1,1,9) + c.reshape(-1,9,1) * self.v.reshape(-1,1,9)

    def get_del_term_two(self,c):
        arr_a = np.zeros((9,9,9))
        for i in range(9):
            arr_a[i,:,i] = self.v.reshape(9,)

        arr_b = np.zeros((9,9,9))
        for i in range(9):
            arr_b[:,i,i] = self.v.reshape(9,)  

        arr_full = arr_a - arr_b

        return arr_full.reshape(1,9,9,9)

    def get_term_four(self,c):
        #v = np.array([1/9] * 9).reshape(9,1)

        return (c * self.gamma).reshape(-1,9,1) + (self.v * self.omega).reshape(1,9,1)

    def get_term_five(self,c):
        return self.get_term_five_a(c)/self.get_term_five_b(c)

    def get_term_five_a(self,c):
        factor =  (self.v.dot(self.v.T).dot(self.L).dot(c.T)/self.v.T.dot(self.L).dot(self.v)).T
        c_mod = c - factor
        return np.sqrt((c_mod.dot(self.L) * c_mod).sum(axis = 1).reshape(-1,1,1))

    def get_term_five_b(self,c):
        return ((c.dot(self.L) * c).sum(axis = 1) * self.v.T.dot(self.L).dot(self.v) - self.v.T.dot(self.L.dot(c.T))**2).reshape(-1,1,1)

    def get_terms_four_five(self,c):
        #is this the right shape
        return self.get_term_four(c) * self.get_term_five(c)

    def get_del_term_four(self,c):
        return (np.identity(9) * self.gamma).reshape(1,9,9)

    def get_del_term_five_a(self,c):
        factor = (self.v.dot(self.v.T).dot(self.L).dot(c.T)/self.v.T.dot(self.L).dot(self.v)).T

        c_mod = c - factor
        top = c_mod.dot(self.L).reshape(-1,1,9)
        bottom = np.sqrt((c_mod.dot(self.L) * c_mod).sum(axis = 1).reshape(-1,1,1))
        side = np.identity(9) - self.v.dot(self.v.T).dot(self.L)/self.v.T.dot(self.L).dot(self.v)
        res = (top/bottom).dot(side)
        return res.reshape(-1,1,9)

    def get_del_term_five_b(self,c):
        term_one = (2 * c.dot(self.L) * self.v.T.dot(self.L).dot(self.v)).reshape(-1,1,9)
        term_two = (2 * self.v.T.dot(self.L.dot(c.T)).T).reshape(-1,1,1)
        term_three = (self.v.T.dot(self.L)).reshape(1,1,9)
        return term_one.reshape(-1,1,9) - (term_two * term_three).reshape(-1,1,9)

    def get_del_term_five(self,c):
        a = self.get_term_five_a(c)
        del_a = self.get_del_term_five_a(c)
        b = self.get_term_five_b(c)
        del_b = self.get_del_term_five_b(c)

        return (del_a * b - a * del_b) / b**2

    def get_del_terms_four_five(self,c):
        return self.get_term_four(c) * self.get_del_term_five(c) + \
                    self.get_del_term_four(c) * self.get_term_five(c)

    def get_last_three_terms(self,c):
        return np.einsum('ij, ajk -> aik',self.L,self.get_terms_four_five(c))

    def get_del_last_three_terms(self,c):
        return np.einsum('ij, ajk -> aik',self.L,self.get_del_terms_four_five(c))

    def get_last_four_terms(self,c):
        term_two = self.get_term_two(c)
        last_three = self.get_last_three_terms(c)
        return np.einsum('aij, ajk -> aik', term_two, last_three)

    def get_del_last_four_terms(self,c):
        comp_i = self.get_del_term_two(c)
        comp_ii = self.get_last_three_terms(c)
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,9))
        term_b = np.einsum('aij, ajk -> aik', self.get_term_two(c), self.get_del_last_three_terms(c))
        return term_a + term_b

    def get_del_full(self,c):
        return np.einsum('ij, ajk -> aik',self.L,self.get_del_last_four_terms(c))

    def make_pick(self
                  , player_assignments : dict[list]
                  , j : int
                  ): 

        generator = self.get_h_scores(player_assignments, j)
        for i in range(30):
            res  = next(generator)

        scores = res['Scores']

        best_player = scores.idxmax()

        return best_player

class SimpleAgent():
    #Comment

    def __init__(self, order):
        self.order = order

    def make_pick(self, player_assignments : dict[list], j : int) -> str:

        players_chosen = [x for v in player_assignments.values() for x in v]

        #ZR: Can this be done more efficiently?
        available_players = [p for p in self.order if not p in players_chosen]
        player = available_players[0]

        return player

def estimate_matchup_result(team_1_x_scores : pd.Series
                            , team_2_x_scores : pd.Series
                            , n_picks : int
                            , scoring_format : str) -> float:
    """Based on X scores, estimates the result of a matchup

    Args:
      team_1_x_scores: Series of x-scores for one team
      team_2_x_scores: Series of x-scores for other team
      n_picks: number of players on each team
      scoring_format: format to use for analysis

    Returns:
      Dictionary with results of the trade
    """

    cdf_estimates = pd.DataFrame(norm.cdf(team_2_x_scores - team_1_x_scores
                                        , scale = np.sqrt(n_picks*2)
                                        )
                            ).T

    if scoring_format == 'Head to Head: Most Categories':
        score = combinatorial_calculation(cdf_estimates
                                                    , 1 - cdf_estimates
                                                    , categories = cdf_estimates.columns
                        )

    else:
        score = cdf_estimates.mean(axis = 1) 

    return float(score)


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


    res_without_player= next(H.get_h_scores(without_player,team, [player]))
    res_with_player = next(H.get_h_scores(with_player, team))

    res = (res_with_player['Scores'].max() - res_without_player['Scores'].max())

    return res