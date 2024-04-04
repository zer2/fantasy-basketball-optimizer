import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import os
from itertools import combinations
from src.helper_functions import get_categories
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
                 , scoring_format : str
                 , punting : bool
                 , chi : float
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
            scoring_format
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
        self.chi = chi
        
        self.cross_player_var = info['Var']
        self.L = info['L']
        self.punting = punting
        self.scoring_format = scoring_format
        self.var_fudge_factor = 2

        mov = info['Mov']
        vom = info['Vom']
        x_scores = info['X-scores']

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[info['Z-scores'].sum(axis = 1).sort_values(ascending = False).index]
            v = np.sqrt(mov/vom)  

            n_opponents = n_drafters - 1

            #scale is standard deviation of overall "luck"
            player_stat_luck_overall = np.sqrt(self.chi * self.n_picks * 9)

            max_luck_expected =  norm.ppf((self.n_drafters - 1 - 0.375)/(self.n_drafters - 1 + 0.25)) * \
                                    player_stat_luck_overall

            player_stat_luck_per_category = max_luck_expected/9
            max_cdf = norm.cdf(player_stat_luck_per_category, scale = np.sqrt(self.chi * self.n_picks) )

            ev_max_wins = max_cdf * (self.n_drafters-1) * 9

            self.mu_m = ev_max_wins
            self.var_m = max_cdf * (1-max_cdf) * self.n_picks * self.n_drafters

        else:
            self.x_scores = x_scores.loc[info['G-scores'].sum(axis = 1).sort_values(ascending = False).index]
            v = np.sqrt(mov/(mov + vom))

        self.v = np.array(v/v.sum()).reshape(9,1)


    def get_h_scores(self
                  , player_assignments : dict[list[str]]
                  , drafter
                  , cash_remaining_per_team : dict[int] = None
                  , exclusion_list = []
                  ) -> tuple: 

        """Performs the H-score algorithm

        Args:
            player_assignments : dictionary of form team -> list of players chosen by that team 
            player: which drafter to perform H-scoring for

        Returns:
            String indicating chosen player
        """

        my_players = [p for p in player_assignments[drafter] if p ==p]
        n_players_selected = len(my_players) 

        players_chosen = [x for v in player_assignments.values() for x in v if x == x]
        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen + exclusion_list)]
        total_players = self.n_picks * self.n_drafters

        diff_means, diff_vars, n_values = self.get_diff_distributions(player_assignments
                                        , drafter
                                        , x_scores_available
                                        , cash_remaining_per_team
        )
        
        x_scores_available_array = np.expand_dims(np.array(x_scores_available), axis = 2)

        default_weights = self.v.T.reshape(1,9,1)
        initial_weights = ((diff_means + x_scores_available_array)/((default_weights * 1000)) + \
                            default_weights).mean(axis = 2)
        
        scores = []
        weights = []

        return self.perform_iterations(initial_weights
                                       ,my_players
                                       , n_players_selected
                                       , diff_means
                                       , diff_vars
                                       , x_scores_available_array
                                       , x_scores_available.index
                                       , n_values)

    def get_diff_distributions(self
                    , player_assignments : dict[list[str]]
                    , drafter
                    , x_scores_available : pd.DataFrame
                    , cash_remaining_per_team : dict[int] = None) -> pd.Series:
        """Calculates base distributions of expected difference to opponents, before next player is added

        Args:
            player_distributions : list of all players chosen, including my_players
            x_scores_available: DataFrame of x-scores, excluding players chosen by any team
            cash_remaining_per_team: amount of cash left for each team. Only relevant for auctions
        Returns:
            Series of form {cat : expected value of opposing teams for the cat}
        """

        my_players = [p for p in player_assignments[drafter] if p ==p]

        x_self_sum = np.array(self.x_scores.loc[my_players].sum(axis = 0))

        #assume that players for the rest of the round will be chosen from the default ordering 
        players_chosen = [x for v in player_assignments.values() for x in v if x == x]

        if cash_remaining_per_team:

            total_cash_remaining = np.sum([v for k, v in cash_remaining_per_team.items()])

            remaining_players = self.n_drafters * self.n_picks - len(players_chosen)

            #weight by v to get generic v-weighted value
            replacement_value = (x_scores_available.iloc[remaining_players] * self.v.T.reshape(9)).sum()
            remaining_overall_value = ((x_scores_available.iloc[0:remaining_players] * self.v.T).sum(axis = 1) \
                                        - replacement_value).sum()
            value_per_dollar = remaining_overall_value/total_cash_remaining

            #when translating back to x-scores, reverse the basis by dividing by v 
            category_value_per_dollar = value_per_dollar / (self.v * 9) 
            replacement_value_by_category = replacement_value / (self.v * 9) 

            diff_means = np.vstack(
                [self.get_diff_means_auction(x_self_sum.reshape(1,9,1) - \
                                                np.array(self.x_scores.loc[players].sum(axis = 0)).reshape(1,9,1)
                                            , cash_remaining_per_team[drafter] - cash_remaining_per_team[team]
                                            , len(my_players) - len(players)
                                            , category_value_per_dollar
                                            , replacement_value_by_category) for team, players \
                                        in player_assignments.items() if team != drafter]
            ).T

        else: 

            extra_players_needed = (len(my_players)+1) * self.n_drafters - len(players_chosen) - 1
            mean_extra_players = x_scores_available.iloc[0:extra_players_needed].mean().fillna(0)

            other_team_sums = np.vstack(
                [self.get_opposing_team_means(players, mean_extra_players, len(my_players)) for team, players \
                                        in player_assignments.items() if team != drafter]
            ).T

            diff_means = x_self_sum.reshape(1,9,1) - other_team_sums.reshape(1,9,self.n_drafters - 1)

        #make order statistics adjustment for Roto 
        if self.scoring_format == 'Rotisserie':
            total_players = self.n_drafters * self.n_picks 
            remaining_players = total_players - len(players_chosen)
            scale = np.sqrt(self.cross_player_var * \
                            self.n_picks * \
                            remaining_players/total_players
                            )

            n_values = rankdata(diff_means, axis = 2, method = 'ordinal')
            player_variance_adjustment =  norm.ppf((n_values - 0.375)/(self.n_drafters - 1 + 0.25))

            diff_means = diff_means + player_variance_adjustment * scale.values.reshape(1,9,1)
        else:
            n_values = None

        diff_vars = np.vstack(
            [self.get_diff_var(len([p for p in players if p ==p])) for team, players \
                                    in player_assignments.items() if team != drafter]
        ).T
        
        diff_vars = diff_vars.reshape(1,9,self.n_drafters - 1)

        return diff_means, diff_vars, n_values

    def get_opposing_team_means(self
                            , players
                            , replacement_value
                            , n_players):
        players = [p for p in players if p == p]

        n_extra_players = n_players + 1 - len(players)

        opposing_team_stats = np.array(self.x_scores.loc[players].sum(axis =0) + \
                                        n_extra_players * replacement_value)

        return opposing_team_stats 
    
    def get_diff_means_auction(self
                                        , score_diff
                                        , money_diff
                                        , player_diff
                                        , category_value_per_dollar
                                        , replacement_value_by_category):
        
        player_diff_total = ((player_diff-1) * replacement_value_by_category).reshape(1,9,1)
        money_diff_total = (money_diff * category_value_per_dollar).reshape(1,9,1)

        #player diff total is subtracted because the team with more players gets less replacement value
        total_diff = score_diff - player_diff_total + money_diff_total 

        return total_diff

    def get_diff_var(self
                    , n_their_players):
        #diff_var should just include the player-to-player variance. maybe? and 

        if self.scoring_format == 'Rotisserie':
            diff_var = self.n_picks * 2 * self.chi + 0 * self.cross_player_var
        else:
            diff_var = self.n_picks * \
                (2 +  self.cross_player_var * (self.n_picks - n_their_players)/(self.n_picks))
        return diff_var

    def perform_iterations(self
                           ,weights : pd.DataFrame
                           ,my_players : list[str]
                           ,n_players_selected : int
                           ,diff_means : pd.Series
                           ,diff_vars : pd.Series
                           ,x_scores_available_array : pd.DataFrame
                           ,result_index
                           ,n_values
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
            diff_var: total variance expected in the end result for each category
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

                pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
                cdf_estimates = self.get_cdf(x_diff_array, diff_vars)
        
                score, pdf_weights = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , diff_vars
                                        , n_values
                                        , calculate_pdf_weights = True)

                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)
        
                step_size = self.alpha * (i + 1)**(-self.beta) 
                change_weights = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                weights = weights + change_weights
                weights[weights < 0] = 0
                weights = weights/weights.sum(axis = 1).reshape(-1,1)

            #case where one more player needs to be chosen
            elif (n_players_selected == (self.n_picks - 1)) | ((not self.punting) & (n_players_selected < (self.n_picks)) ): 

                x_diff_array = diff_means + x_scores_available_array

                cdf_estimates = self.get_cdf(x_diff_array
                                            , diff_vars)

                weights = None
                expected_future_diff = None
                pdf_estimates = None
                
                score = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , diff_vars
                                        , n_values
                                        , calculate_pdf_weights = False)

            #case where no new players need to be chosen
            elif (n_players_selected == self.n_picks): 

                cdf_estimates = self.get_cdf(diff_means, diff_vars)

                weights = None
                expected_future_diff = None
                pdf_estimates = None
                
                score = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , diff_vars
                                        , n_values
                                        , calculate_pdf_weights = False)
                
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

                cdf_estimates = self.get_cdf(diff_means_mod, diff_vars)
                pdf_estimates = None
                                        
                score = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , diff_vars
                                        , n_values
                                        , calculate_pdf_weights = False)

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
    def get_objective_and_pdf_weights(self
                                        ,cdf_estimates
                                        , pdf_estimates
                                        , diff_vars
                                        , n_values
                                        , calculate_pdf_weights = False):

        if self.scoring_format == 'Head to Head: Most Categories':

            return self.get_objective_and_pdf_weights_mc(
                        cdf_estimates
                        , pdf_estimates
                        , diff_vars
                        , calculate_pdf_weights) 

        elif self.scoring_format == 'Rotisserie':

            return self.get_objective_and_pdf_weights_rotisserie(
                        cdf_estimates
                        , pdf_estimates
                        , n_values
                        , calculate_pdf_weights) 

        else:
            return self.get_objective_and_pdf_weights_ec(
                        cdf_estimates
                        , pdf_estimates
                        , diff_vars
                        , calculate_pdf_weights) 

    def get_objective_and_pdf_weights_mc(self
                                , cdf_estimates
                                , pdf_estimates
                                , diff_vars
                                , calculate_pdf_weights = False):

        objective = combinatorial_calculation(cdf_estimates
                                                , 1 - cdf_estimates
                                                ).mean(axis = 1)

        if calculate_pdf_weights:

            tipping_points = calculate_tipping_points(np.array(cdf_estimates))   

            pdf_weights = (tipping_points*pdf_estimates/np.sqrt(diff_vars)).mean(axis = 2)

            return objective, pdf_weights

        else: 

            return objective


    def get_objective_and_pdf_weights_ec(self
                            , cdf_estimates
                            , pdf_estimates
                            , diff_vars
                            , calculate_pdf_weights = False):

        objective = cdf_estimates.mean(axis = 2).mean(axis = 1) 

        if calculate_pdf_weights:

            pdf_weights = (pdf_estimates/np.sqrt(diff_vars)).mean(axis = 2)

            return objective, pdf_weights

        else:
            return objective


    def get_objective_and_pdf_weights_rotisserie(self
                                , cdf_estimates
                                , pdf_estimates
                                , n_values = None
                                , calculate_pdf_weights = False):

        """Calculates the objective function for Rotisserie, and the gradient if required
         
        Args:
            cdf_estimates : 
            n_values 
            calculate_pdf_weights
        Yields:
            Either a Series for Objective value 
            or 

        """
        n_opponents = self.n_drafters - 1

        drafter_mean = cdf_estimates.sum(axis = (1,2)).reshape(-1,1,1)

        if n_values is None:
            n_values = rankdata(cdf_estimates, axis = 2, method = 'ordinal')

        mu_values = cdf_estimates.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1) 

        variance_contributions = cdf_estimates * \
                                    (2 * n_opponents - 2 * n_values - 2 * mu_values + 1 )
        category_variance = variance_contributions.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1)

        extra_term = mu_values**2

        category_variance = category_variance + extra_term

        drafter_variance = category_variance.sum(axis = 1).reshape(-1,1,1) * self.var_fudge_factor

        total_variance = drafter_variance + self.var_m

        objective = norm.cdf(drafter_mean - self.mu_m
                            , scale = np.sqrt(total_variance)).reshape(-1)

        if calculate_pdf_weights:

            nabla = total_variance + (self.mu_m - drafter_mean) * self.var_fudge_factor * \
                                                    (n_opponents - n_values - mu_values + 0.5) 

            outer_pdf = norm.pdf((drafter_mean - self.mu_m)/np.sqrt(total_variance))

            gradient = nabla * outer_pdf/ (total_variance * np.sqrt(total_variance))

            pdf_weights = (gradient*pdf_estimates).mean(axis = 2)

            return objective, pdf_weights

        else: 

            return objective

    def get_pdf(self, x_diff_array, diff_vars):

        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]
                                                    , x_diff_array.shape[1] * x_diff_array.shape[2])
        diff_vars_reshaped = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])

        pdf_estimates = norm.pdf(diff_array_reshaped, scale = np.sqrt(diff_vars_reshaped))

        pdf_estimates_reshaped = pdf_estimates.reshape(x_diff_array.shape)

        return pdf_estimates_reshaped
    
    def get_cdf(self, x_diff_array, diff_vars):

        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]
                                                    , x_diff_array.shape[1] * x_diff_array.shape[2])
        diff_vars_reshaped = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])

        cdf_estimates = norm.cdf(diff_array_reshaped, scale = np.sqrt(diff_vars_reshaped))

        cdf_estimates_reshaped = cdf_estimates.reshape(x_diff_array.shape)

        return cdf_estimates_reshaped

    def get_gradient_weights_rotisserie(self
                                        , cdf_estimates
                                        , n_values = None):

        n_opponents = self.n_drafters - 1

        drafter_mean = cdf_estimates.sum(axis = (1,2)).reshape(-1,1,1)

        if n_values is None:
            n_values = rankdata(cdf_estimates, axis = 2, method = 'ordinal')

        mu_values = cdf_estimates.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1) 

        variance_contributions = cdf_estimates * \
                                    (2 * n_opponents - 2 * n_values - 2 * mu_values + 1 )
        category_variance = variance_contributions.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1)

        extra_term = mu_values**2

        category_variance = category_variance + extra_term

        drafter_variance = category_variance.sum(axis = 1).reshape(-1,1,1) * self.var_fudge_factor

        total_variance = drafter_variance + self.var_m

        nabla = total_variance + (self.mu_m - drafter_mean) * self.var_fudge_factor * \
                                                 (n_opponents - n_values - mu_values + 0.5) 

        outer_pdf = norm.pdf((drafter_mean - self.mu_m)/np.sqrt(total_variance))

        #return 2 * ((n_opponents - n_values - mu_values + 0.5) - x_factor)
        return nabla * outer_pdf/ (total_variance * np.sqrt(total_variance))

    def objective_function_rotisserie(self
                                    , cdf_estimates
                                    , n_values = None):

        n_opponents = self.n_drafters - 1

        drafter_mean = cdf_estimates.sum(axis = (1,2)).reshape(-1)

        if n_values is None:
            n_values = rankdata(cdf_estimates, axis = 2, method = 'ordinal')
        mu_values = cdf_estimates.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1) 


        variance_contributions = cdf_estimates * \
                                    (2 * n_opponents - 2 * n_values - 2 * mu_values + 1 )
        category_variance = variance_contributions.sum(axis = 2).reshape(-1
                                                            , 9
                                                            , 1)

        #print('N values:')
        #print(n_values)

        #print('Mu Values:')
        #print(mu_values)

        #print('Var contributions:')
        #print(variance_contributions)

        extra_term = mu_values**2

        category_variance = category_variance + extra_term

        #print('Category variance:')
        #print(variance_contributions)

        drafter_variance = category_variance.sum(axis = 1).reshape(-1) * self.var_fudge_factor

        total_variance = drafter_variance + self.var_m

        #return drafter_variance

        objective = norm.cdf(drafter_mean - self.mu_m, scale = np.sqrt(total_variance))

        return objective 

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


### Analysis of H-scoring results 

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


### Helper functions 

def savor_calculation(raw_values_unselected : pd.Series
                    , n_remaining_players : int
                    , remaining_cash : int
                    , noise = 1) -> pd.Series:
    """Calculate SAVOR- Streaming-adjusted value over replacement

    SAVOR estimates the probability that a player will be replaced by a streamer, and adjusts 
    auction value accordingly

    Args:
      raw_values_unselected: raw value by Z-score, G-score, etc. 
      n_remaining_players: number of players left to be picked
      remaining_cash: amount of cash remaining to spend on players, from all teams
      noise: parameter for the SAVOR function. Controls how noisy we expect player performance to be
             and therefore how likely it is a player will be replaced by a streamer

    Returns:
      Series, SAVOR 
    """

    replacement_value = raw_values_unselected.iloc[n_remaining_players]
    value_above_replacement = np.clip(raw_values_unselected - replacement_value,0,None)

    probability_of_non_streaming = norm.cdf(value_above_replacement/noise)
    adjustment_factor = noise/(2 * np.pi)**(0.5) * (1 - np.exp((-value_above_replacement**2)/(2 * noise)))
    adjusted_value = value_above_replacement * probability_of_non_streaming - adjustment_factor

    remaining_value = adjusted_value.iloc[0:n_remaining_players].sum()
    dollar_per_value = remaining_cash/remaining_value

    savor = adjusted_value * dollar_per_value

    return savor 

def combinatorial_calculation(c : np.array
                              , c_comp : np.array
                              , data = 1 #the latest probabilities. Defaults to 1 at start
                              , level : int = 0 #the number of categories that have been worked into the probability
                              , n_false : int = 0 #the number of category losses that have been tracked so far
                             ):
    """This recursive functions enumerates winning probabilities for the Gaussian optimizer

    The function's recursive structure creates a binary tree, where each split is based on whether the next category is 
    won or lost. At the high level it looks like 
    
                                            (start) 
                                    |                   |
                                won rebounds      lost rebounds
                             |          |           |            |
                          won pts    lost pts   won pts     lost pts
                          
    The probabilities of winning scenarios are then added along the tree. This is more efficient than brute force calculation
    of each possibility, because it doesn't repeat multiplication steps for similar scenarios like (won 9) and (won 8 then 
    lost the last 1). Ultimately it is about five times faster than the equivalent with list comprehension
    
    Args:
        c: Array of all category winning probabilities. Dimensions are (player, category, opponent)
        c_comp: 1 - c
        data: probability of the node's scenario. Defaults to 1 because no categories are required at first
        level: the number of categories that have been worked into the probability
        n_false: the number of category losses that have been tracked so far. When it gets high enough 
                 we write off the node; the remaining scenarios do not contribute to winning chances

    Returns:
        DataFrame with probability of winning a majority of categories for each player 

        axis 0: player 
        axis 1: opponent

    """
    if n_false > (c.shape[1] -1)/2: #scenarios where a majority of categories are losses are overall losses
        return 0 
    elif level < c.shape[1] :
        #find the total winning probability of both branches from this point- if we win or lose the current category 
        return combinatorial_calculation(c, c_comp, data * c[:,level,:], level + 1, n_false) + \
                combinatorial_calculation(c, c_comp, data * c_comp[:,level,:], level + 1, n_false + 1)
    else: #a series where all 9 categories has been processed, and n_false <= the cutoff, can be added to the total %
        return data

@st.cache_data()
def get_grid():
    #create a grid representing 126 scenarios where 5 categories are won and 4 are lost

    which = np.array([list(combinations(range(9), 5))] )
    grid = np.zeros((126, 9), dtype="bool")     
    grid[np.arange(126)[None].T, which] = True

    grid = np.expand_dims(grid, axis = 2)

    return grid

def calculate_tipping_points(x : np.array) -> np.array:
    """Calculate the probability of each category being a tipping point, assuming independence

    Args:
        x: Array of shape (n,9,m) representing probabilities of winning each of the 9 categories 

    Returns:
        DataFrame of shape (n,9,m) representing probabilities of each category being a tipping point
        m is number of opponents
    """

    grid = get_grid()

    #copy grid for each row in x 
    grid = np.array([grid] * x.shape[0])

    x = x.reshape(x.shape[0],1,9, x.shape[2])

    #get the probabilities of the scenarios and filter them by which categories they apply to
    #the categories that are won all become tipping points

    first_part = ne.evaluate('grid * x + (1-grid) * (1-x)') \
                                   .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    positive_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    #do the same but for the inverse scenarios, where 5 categories are lost and 4 are won
    #in this case the lost categories become tipping points 
    first_part = ne.evaluate('(1 - grid) * x + grid * (1-x)') \
                                  .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    negative_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    final_probabilities = ne.evaluate('positive_case_probabilities + negative_case_probabilities')

    return final_probabilities