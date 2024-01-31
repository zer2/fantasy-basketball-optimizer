import numpy as np
import pandas as pd
from scipy.stats import norm
import os
from itertools import combinations

from src.helper_functions import combinatorial_calculation, calculate_tipping_points

class HAgent():

    def __init__(self
                 , info
                 , omega
                 , gamma
                 , alpha
                 , beta
                 , n_picks
                 , winner_take_all
                 , punting
):
        """Calculates the rank order based on U-score

        Args:
            season_df: dataframe with weekly data for the season
            positions: Series of player -> list of eligible positions
            gamma: float, how much to scale the G-score
            epsilon: float, how much to weigh correlation
            n_players: number of players to use for second-phase standardization
            winner_take_all: Boolean of whether to optimize for the winner-take-all format
                             If False, optimizes for total categories
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
                  , df
                  , my_players
                  , players_chosen
                  ,):

        """Performs the H-score algorithm

        Args:
            player_assignments: dict of format
                   player : team that picked them

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
            #os.write(1,bytes(str(previous_rounds_expected),'utf-8'))
            #os.write(1,bytes(str(this_round_expected),'utf-8'))
            #os.write(1,bytes(str(diff_means),'utf-8'))
            #os.write(1,bytes(str(x_scores_available),'utf-8'))
        else:
            previous_rounds_expected = self.score_table.iloc[0:self.n_picks].sum().loc[(self.x_scores.columns,'mean')].droplevel(1)
            diff_means = x_self_sum - previous_rounds_expected 


        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen)]
                      

        c = np.array((diff_means + x_scores_available)/(self.v.T * 500) + self.v.T)
        
        scores = []
        weights = []

        return self.perform_iterations(c,my_players, n_players_selected, diff_means, x_scores_available)

    def perform_iterations(self
                           ,c
                           ,my_players
                           ,n_players_selected
                           , diff_means
                           , x_scores_available):
        """Performs one iteration of H-scoring. 
         
         Case (1): If n_players_selected < n_picks -1, the Gaussian multivariate assumption is used for future picks and weight is chosen by gradient descent
         Case (2): If n_players_selected = n_picks -1, each candidate player is evaluated with no need for modeling future picks
         Case (3): If n_players_selected = n_picks, a single number is returned for the team's total H-score
         Case (4): If n_players_selected > n_picks, all subsets of possible players are evaluated for the best subset

        Args:
            c: Starting choice of weights. Relevant for case (1)
            my_players: list of players selected
            n_players_selected: integer, number of players already selected
            diff_means: series, difference in mean between already selected players and expected
            x_scores_available: dataframe, X-scores of unselected players
        Returns:
            None

        """
        i = 0
        
        while True:

            #case where many players need to be chosen
            if (n_players_selected < self.n_picks - 1) & (self.punting):
                del_full = self.get_del_full(c)
        
                expected_x = self.get_x_mu(c)

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
                change_c = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                c = c + change_c
                c[c < 0] = 0
                c = c/c.sum(axis = 1).reshape(-1,1)

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

                c = None
                
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
                              , scale = np.sqrt(self.diff_var))
                     ,index = diff_means.index)

                c = None
                
                if self.winner_take_all:
                    score = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )
                else:
                    score = cdf_estimates.mean() 

            #case where there are too many players and some need to be removed 
            else: #n > n_picks 

                extra_players = n_players_selected - self.n_picks 
                players_to_remove_possibilities = combinations(my_players,extra_players)
                best_score = 0

                diff_means_mod = diff_means - pd.concat((self.x_scores.loc[list(players_to_remove)].sum(axis = 0) for players_to_remove in players_to_remove_possibilities)
                                                       ,axis = 1).T

                os.write(1,bytes(str(diff_means_mod), 'utf-8'))

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

                c = None

            i = i + 1
    
            yield score, c, cdf_estimates


    ### below are functions used for the optimization procedure 

    def get_x_mu(self,c):

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

        #inverse_part = np.linalg.inv(U.dot(self.L).dot(U.T))
        #X_mu = self.L.dot(U.T).dot(inverse_part).dot(b)

        X_mu = np.einsum('aij, ajk -> aik', x, b)

        return X_mu

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

def analyze_trade(team_1_other
                  , team_1_trade
                  , team_2_other
                  , team_2_trade
                  , H
                  , player_stats
                  , players_chosen
                  ,n_iterations):    
                      
    _, H_1_1 = next(H.get_h_scores(player_stats, team_1_other + team_1_trade, players_chosen))
    _, H_2_2 = next(H.get_h_scores(player_stats, team_2_other + team_2_trade, players_chosen))

    n_player_diff = len(team_1_trade) - len(team_2_trade)

    if n_player_diff > 0:
        generator = H.get_h_scores(player_stats, team_1_other + team_2_trade, players_chosen)
        for i in range(n_iterations):
            _, H_1_2 = next(generator)
        
        _, H_2_1 = next(H.get_h_scores(player_stats, team_2_other + team_1_trade, players_chosen))
    elif n_player_diff == 0:
        _, H_1_2 = next(H.get_h_scores(player_stats, team_1_other + team_2_trade, players_chosen))

        os.write(1, bytes(str(team_2_other + team_1_trade),'utf-8'))
        _, H_2_1 = next(H.get_h_scores(player_stats, team_2_other + team_1_trade, players_chosen))
    else:
        _, H_1_2 = next(H.get_h_scores(player_stats, team_1_other + team_2_trade, players_chosen))

        generator = H.get_h_scores(player_stats, team_2_other + team_1_trade, players_chosen)
        for i in range(n_iterations):
            _, H_2_1 = next(generator)

    return H_1_1, H_1_2, H_2_1, H_2_2
                
