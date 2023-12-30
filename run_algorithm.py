#2: based on drafting situation: run gradient descent, get results

import numpy as np
import pandas as pd
from scipy.stats import norm
from helper_functions import combinatorial_calculation, calculate_tipping_points

class HAgent():

    def __init__(self
                 , info
                 , omega
                 , gamma
                 , alpha
                 , beta
                 , n_players
                 , winner_take_all
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
        self.n_players = n_players
        self.winner_take_all = winner_take_all 
        self.x_scores = info['X-scores']
        self.score_table = info['Score-table']
        self.score_table_smoothed = info['Score-table-smoothed']
        self.diff_var = info['Diff-var']
        self.v = info['v']
        self.L = info['L']
      
    def get_h_scores(self
                  , df
                  , my_players
                  , players_chosen ):

        """Picks a player based on the D-score algorithm

        Args:
            player_assignments: dict of format
                   player : team that picked them

        Returns:
            String indicating chosen player
        """
        round_n = len(my_players) 

        x_self_sum = self.x_scores.loc[my_players].sum(axis = 0)
        
        
        previous_rounds_expected = self.score_table.iloc[0:round_n].sum().loc[(self.x_scores.columns,'mean')].droplevel(1)
        this_round_expected = self.score_table_smoothed.iloc[len(players_chosen)].values
        diff_means = x_self_sum - previous_rounds_expected - this_round_expected
        
        other_team_variance = self.score_table.loc[0:12,(self.x_scores.columns,'var')].sum().droplevel(1)
        rest_of_team_variance = self.score_table.loc[(round_n + 1):12,(self.x_scores.columns,'var')].sum().droplevel(1)

        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen)]
        
        c = np.array((diff_means + x_scores_available)/(self.v.T * 500) + self.v.T)
        
        scores = []
        weights = []

        return self.perform_iterations(c,round_n, diff_means, x_scores_available)

    def perform_iterations(self,c,round_n, diff_means, x_scores_available):

        while True:

            if round_n < 12:
                del_full = self.get_del_full(c)
        
                expected_x = self.get_x_mu(c)
                expected_future_diff = ((12-round_n) * expected_x).reshape(-1,9)
        
                pdf_estimates = norm.pdf(diff_means + x_scores_available + expected_future_diff
                                          , scale = np.sqrt(self.diff_var))
        
                if self.winner_take_all:
        
                    tipping_points = calculate_tipping_points(cdf_estimates)   
        
                    pdf_weights = (tipping_points*pdf_estimates)
                else:
                    pdf_weights = pdf_estimates
        
                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)
        
                step_size = self.alpha * (i + 1)**(-self.beta) 
                change_c = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                c = c + change_c
                c[c < 0] = 0
                c = c/c.sum(axis = 1).reshape(-1,1)

            else:
                expected_future_diff = 0
    
            cdf_estimates = pd.DataFrame(norm.cdf(diff_means + x_scores_available + expected_future_diff
                                                      , scale = np.sqrt(self.diff_var))
                                             ,index = x_scores_available.index)
    
            cdf_estimates.columns = cdf_estimates.columns
    
            if self.winner_take_all:
                win_sums = combinatorial_calculation(cdf_estimates
                                                              , 1 - cdf_estimates
                                                              , categories = cdf_estimates.columns
                                 )
            else:
                win_sums = cdf_estimates.sum(axis = 1) 
    
            yield c, win_sums
    
    
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
