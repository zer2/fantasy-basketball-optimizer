import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from itertools import combinations
from src.helper_functions import combinatorial_calculation, calculate_tipping_points, get_categories

class HAgent():

    def __init__(self
                 , info : dict
                 , omega : float
                 , gamma : float
                 , alpha : float
                 , beta : float
                 , n_picks : int
                 , winner_take_all : bool
                 , punting : bool
                    ):
        """Calculates the rank order based on U-score

        Args:
            info: dictionary with info related to player statistics etc. 
            omega: float, parameter as described in the paper
            gamma: float, parameter as described in the paper
            alpha: float, step size parameter for gradient descent 
            beta: float, decay parameter for gradient descent 
            n_picks: int, number of picks each drafter gets 
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
        self.winner_take_all = winner_take_all 
        self.x_scores = info['X-scores']
        self.score_table = info['Score-table']
        self.score_table_smoothed = info['Score-table-smoothed']
        self.diff_var = info['Diff-var']
        self.v = info['v']
        self.L = info['L']
        self.punting = punting
      
    def get_h_scores(self
                  , my_players : list[str]
                  , players_chosen : list[str]
                  ) -> tuple: 

        """Performs the H-score algorithm

        Args:
            my_players : list of players picked by other teams
            players_chosen : list of all players chosen, including my_players

        Returns:
            String indicating chosen player
        """
        n_players_selected = len(my_players) 

        x_self_sum = self.x_scores.loc[my_players].sum(axis = 0)

        #we want to use the smoothed score table when the expectation for player strength is different depending on how far into the round you are drafting
        #for the last round, it doesn't really matter, because there are no later rounds to balance it out 
        if n_players_selected < (self.n_picks - 1):
            previous_rounds_expected = self.score_table.iloc[0:n_players_selected].sum().loc[(self.x_scores.columns,'mean')].droplevel(1)
            this_round_expected = self.score_table_smoothed.iloc[len(players_chosen)].values
            diff_means = x_self_sum - previous_rounds_expected - this_round_expected
        else:
            previous_rounds_expected = self.score_table.iloc[0:self.n_picks].sum().loc[(self.x_scores.columns,'mean')].droplevel(1)
            diff_means = x_self_sum - previous_rounds_expected 

        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen)]
                      
        initial_weights = np.array((diff_means + x_scores_available)/(self.v.T * 500) + self.v.T)
        
        scores = []
        weights = []

        return self.perform_iterations(initial_weights
                                       ,my_players
                                       , n_players_selected
                                       , diff_means
                                       , x_scores_available)

    def perform_iterations(self
                           ,weights : pd.DataFrame
                           ,my_players : list[str]
                           ,n_players_selected : int
                           ,diff_means : pd.Series
                           ,x_scores_available : pd.DataFrame
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
        
                expected_x = self.get_x_mu(weights)

                expected_future_diff = ((12-n_players_selected) * expected_x).reshape(-1,9)
        
                pdf_estimates = norm.pdf(diff_means + x_scores_available + expected_future_diff
                                          , scale = np.sqrt(self.diff_var))
                
                cdf_estimates = pd.DataFrame(norm.cdf(diff_means + x_scores_available + expected_future_diff
                                          , scale = np.sqrt(self.diff_var))
                                 ,index = x_scores_available.index)
        
                if self.winner_take_all:
        
                    tipping_points = calculate_tipping_points(np.array(cdf_estimates))   
        
                    pdf_weights = (tipping_points*pdf_estimates)
                else:
                    pdf_weights = pdf_estimates
        
                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)
        
                step_size = self.alpha * (i + 1)**(-self.beta) 
                change_weights = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                weights = weights + change_weights
                weights[weights < 0] = 0
                weights = weights/weights.sum(axis = 1).reshape(-1,1)

                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )
                else:
                    score = cdf_estimates.mean(axis = 1) 

            #case where one more player needs to be chosen
            elif (n_players_selected == (self.n_picks - 1)) | (self.punting & (n_players_selected < (self.n_picks - 1)) ): 
                cdf_estimates = pd.DataFrame(norm.cdf(diff_means + x_scores_available
                              , scale = np.sqrt(self.diff_var))
                     ,index = x_scores_available.index)

                weights = None
                
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )
                else:
                    score = cdf_estimates.mean(axis = 1) 

            #case where no new players need to be chosen
            elif n_players_selected == self.n_picks: 
                cdf_estimates = pd.DataFrame(norm.cdf(diff_means
                              , scale = np.sqrt(self.diff_var)
                                                     )
                              , index = diff_means.index
                                            ).T

                weights = None
                
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )

                else:
                    score = cdf_estimates.mean(axis = 1) 

            #case where there are too many players and some need to be removed. n > n_picks
            else: 

                extra_players = n_players_selected - self.n_picks 
                players_to_remove_possibilities = combinations(my_players,extra_players)

                diff_means_mod = diff_means - pd.concat((self.x_scores.loc[list(players_to_remove)].sum(axis = 0) for players_to_remove in players_to_remove_possibilities)
                                                       ,axis = 1).T

                cdf_estimates = pd.DataFrame(norm.cdf(diff_means_mod
                                              , scale = np.sqrt(self.diff_var))
                                     ,index = diff_means_mod.index)
                                        
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )
                else:
                    score = cdf_estimates.mean(axis = 1)

                weights = None

            i = i + 1

            cdf_estimates.columns = get_categories()
    
            yield score, weights, cdf_estimates


    ### below are functions used for the optimization procedure 

    def get_x_mu(self,c):
        #uses the pre-simplified formula for x_mu from page 19 of the paper. Using the simplified form would work just as well

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

        X_mu = np.einsum('aij, ajk -> aik', x, b)

        return X_mu

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
        return np.einsum('ij, ajk -> aik', self.get_term_two(c), self.get_last_three_terms(c))

    def get_del_last_four_terms(self,c):
        comp_i = self.get_del_term_two(c)
        comp_ii = self.get_last_three_terms(c)
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,9))
        term_b = np.einsum('aij, ajk -> aik', self.get_term_two(c), self.get_del_last_three_terms(c))
        return term_a + term_b

    def get_del_full(self,c):
        return np.einsum('ij, ajk -> aik',self.L,self.get_del_last_four_terms(c))

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


def analyze_trade(team_1_other : list[str]
                  ,team_1_trade : list[str]
                  ,team_2_other : list[str]
                  ,team_2_trade : list[str]
                  ,H
                  ,player_stats : pd.DataFrame
                  ,players_chosen : list[str]
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

    score_1_1, _, rate_1_1 = next(H.get_h_scores(team_1_other + team_1_trade, players_chosen))
    score_2_2, _, rate_2_2 = next(H.get_h_scores(team_2_other + team_2_trade, players_chosen))
 
    n_player_diff = len(team_1_trade) - len(team_2_trade)

    if n_player_diff > 0:
        generator = H.get_h_scores(team_1_other + team_2_trade, players_chosen)
        for i in range(n_iterations):
            score_1_2,_,rate_1_2  = next(generator)
        rate_1_2.columns = rate_1_1.columns
        
        score_2_1,_,rate_2_1 = next(H.get_h_scores(team_2_other + team_1_trade, players_chosen))
        rate_2_1.columns = rate_1_1.columns

    elif n_player_diff == 0:
        score_1_2,_,rate_1_2 = next(H.get_h_scores(team_1_other + team_2_trade, players_chosen))
        score_2_1,_,rate_2_1 = next(H.get_h_scores(team_2_other + team_1_trade, players_chosen))

    else:
        score_1_2,_,rate_1_2 = next(H.get_h_scores(team_1_other + team_2_trade, players_chosen))
        rate_1_2.columns = rate_1_1.columns

        generator = H.get_h_scores(team_2_other + team_1_trade, players_chosen)
        for i in range(n_iterations):
            score_2_1,_,rate_2_1 = next(generator)
    
        rate_2_1.columns = rate_1_1.columns

    #helper function just for this procedure
    def get_full_row(scores, rates):

        idxmax = scores.idxmax()
        score = pd.Series([scores[idxmax]], index = ['H-score'])
        rate = rates.loc[idxmax]

        return pd.concat([score, rate])

    team_1_info = {'pre' : get_full_row(score_1_1, rate_1_1)
                        ,'post' : get_full_row(score_1_2, rate_1_2)}
    team_2_info = {'pre' : get_full_row(score_2_2, rate_2_2)
                        ,'post' : get_full_row(score_2_1, rate_2_1)}
                      
    results_dict = {1 : team_1_info
                    ,2 : team_2_info
                   }

    return results_dict
                
def analyze_trade_value(player : list[str]
                  ,rest_of_team : list[str]
                  ,H
                  ,player_stats : pd.DataFrame
                  ,players_chosen : list[str]
                  ) -> float:    

    """Compute the results of a potential trade

    Args:
      team_1_other: remaining players, not to be traded from the first team
      team_1_trade: player(s) to be traded from the first team
      H: H-scoring agent, which can be used to calculate H-score 
      player_stats: DataFrame of player statistics 
      players_chosen: list of all chosen players

    Returns:
      Float, relative H-score value
    """
    score_without_player, _, _ = next(H.get_h_scores(rest_of_team, players_chosen))
    score_with_player, _, _ = next(H.get_h_scores(rest_of_team + [player], players_chosen))

    res = (score_with_player.max() - score_without_player.max())
    return res