import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import os
from itertools import combinations
from src.helper_functions import check_team_eligibility
from src.algorithm_helpers import combinatorial_calculation, calculate_tipping_points
from src.process_player_data import get_category_level_rv
import streamlit as st 
from src.helper_functions import get_eligibility_row_simplified
from src.position_optimization import optimize_positions_all_players
import datetime

class HAgent():

    def __init__(self
                 , info : dict
                 , omega : float
                 , gamma : float
                 , alpha : float
                 , beta : float
                 , n_picks : int
                 , n_drafters : int
                 , dynamic : bool
                 , scoring_format : str
                 , chi : float
                 , fudge_factor : float = 1
                 , positions : pd.Series = None
                 , collect_info : bool = False
                    ):
        """Initializes an H-score agent, which can calculate H-scores based on given info 

        Args:
            info: dictionary with info related to player statistics etc. 
            omega: float, parameter as described in the paper
            gamma: float, parameter as described in the paper
            alpha: float, step size parameter for gradient descent 
            beta: float, decay parameter for gradient descent 
            n_picks: int, number of picks each drafter gets 
            n_drafters : int, number of drafters
            scoring_format

            positions: Series of format {'LeBron James' -> ['PF', 'C']}
        Returns:
            None

        """
        self.omega = omega
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.n_picks = n_picks 
        self.n_drafters = n_drafters
        self.dynamic = dynamic
        self.chi = chi
        self.positions = positions

        #try changing this later for cvxpy
        positions_float = pd.DataFrame(np.concatenate([get_eligibility_row_simplified(pos) for pos in positions])
                                             , index = positions.index).astype(float)
        self.positions_float = positions_float.div(positions_float.sum(axis = 1), axis = 0)

        self.collect_info = collect_info
        
        self.cross_player_var = info['Var']
        self.L = info['L']
        self.scoring_format = scoring_format
        self.position_means = np.array(info['Position-Means']).reshape(1,-1,9)
        self.L_by_position = info['L-by-Position']
        self.L_by_position = np.array(self.L_by_position).reshape(1,-1,9,9)

        self.fudge_factor = fudge_factor

        mov = info['Mov']
        vom = info['Vom']
        x_scores = info['X-scores']

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[info['Z-scores'].sum(axis = 1).sort_values(ascending = False).index]
            v = np.sqrt(mov/vom)  

            #scale is standard deviation of overall "luck"
            player_stat_luck_overall = np.sqrt(9)

            max_luck_expected =  norm.ppf((self.n_drafters - 1 - 0.375)/(self.n_drafters - 1 + 0.25)) * \
                                    player_stat_luck_overall
            
            player_stat_luck_per_category = max_luck_expected * self.fudge_factor /9

            max_cdf = norm.cdf(player_stat_luck_per_category)

            ev_max_wins = max_cdf * (self.n_drafters-1) * 9

            self.mu_m = ev_max_wins
            self.var_m = max_cdf * (1-max_cdf) * (self.n_drafters-1) * 9

        else:
            self.x_scores = x_scores.loc[info['G-scores'].sum(axis = 1).sort_values(ascending = False).index]

            v = np.sqrt(mov/(mov + vom))

        self.v = np.array(v/v.sum()).reshape(9,1)

        turnover_inverted_v = self.v.copy()
        turnover_inverted_v[-1] = -turnover_inverted_v[-1]
        self.turnover_inverted_v = turnover_inverted_v/turnover_inverted_v.sum()

        self.all_res_list = [] #for tracking decisions made during testing
        self.players = []

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
        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen + exclusion_list) & \
                                                self.x_scores.index.isin(self.positions.index)]
        total_players = self.n_picks * self.n_drafters

        diff_means, diff_vars, n_values = self.get_diff_distributions(player_assignments
                                        , drafter
                                        , x_scores_available
                                        , cash_remaining_per_team
        )
        
        x_scores_available_array = np.expand_dims(np.array(x_scores_available), axis = 2)

        default_weights = self.v.T.reshape(1,9,1)

        if self.scoring_format == 'Rotisserie':
            category_momentum_factor = 10000
        else:
            category_momentum_factor = 1000

        initial_weights = ((diff_means + x_scores_available_array)/((default_weights * category_momentum_factor)) + \
                            default_weights).mean(axis = 2)

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
                    , cash_remaining_per_team : dict = None) -> pd.Series:
        """Calculates base distributions of expected difference to opponents, before next player is added

        Args:
            player_assignments : dictionary of form team -> list of players chosen by that team 
            drafter: name of the current drafter. Can be int or str
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

            category_value_per_dollar = value_per_dollar / (self.turnover_inverted_v * 9) 

            replacement_value_by_category = get_category_level_rv(replacement_value
                                                                  , pd.Series(self.v.reshape(-1)
                                                                            , index = self.x_scores.columns) #used for category names
                                                    )
            replacement_value_by_category = np.array(replacement_value_by_category).reshape(9,1)


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

            extra_players_needed = (len(my_players)+1) * self.n_drafters - len(players_chosen) 
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

            #let's try without this 
            #diff_means = diff_means + player_variance_adjustment * scale.values.reshape(1,9,1)

        else:
            n_values = None

        diff_vars = np.vstack(
            [self.get_diff_var(len([p for p in players if p ==p])) for team, players \
                                    in player_assignments.items() if team != drafter]
        ).T
        
        diff_vars = diff_vars.reshape(1,9,self.n_drafters - 1)

        #ZR: this is a super ugly implementation
        if cash_remaining_per_team:
            self.value_of_money = self.get_value_of_money_auction(
                        diff_means
                        , diff_vars
                        , n_values
                        , category_value_per_dollar
                        , replacement_value_by_category)
        else:
            self.value_of_money = None


        return diff_means, diff_vars, n_values

    def get_opposing_team_means(self
                            , players : list[str]
                            , mean_extra_players : pd.Series
                            , n_players : int):
        """Calculates the expected value of an opposing team's statistics (up to the current player)

        Args:
            players : dictionary of form team -> list of players chosen by that team 
            mean_extra_players: expected values of any potential unchosen player 
            n_players: number of players up to the current player 
                        (generally len(players) + 1 if opponent has picked for the round, len(players) otherwise )
        Returns:
            1x9 array, scores by category
        """
        players = [p for p in players if p == p]

        n_extra_players = n_players + 1 - len(players)

        opposing_team_stats = np.array(self.x_scores.loc[players].sum(axis =0) + \
                                        n_extra_players * mean_extra_players)

        return opposing_team_stats 
    
    def get_diff_means_auction(self
                                , score_diff : pd.Series
                                , money_diff : float
                                , player_diff : float
                                , category_value_per_dollar : pd.Series
                                , replacement_value_by_category : pd.Series):
        """Calculates the expected value of an opposing team's statistics (up to the current player)

        Args:
            score_diff : dictionary of form team -> list of players chosen by that team 
            money_diff: difference in available $ between the current drafter and opponent. Current - opponent
            player_diff: difference in # of players between the current drafter and opponent. Current - opponent
            category_value_per_dollar: estimate of how much value in each category can be earned with $1
            replacement_value_by_category: estimated statistics of a replacement-level player 
        Returns:
            1x9 array, expected difference in scores by category
        """
        
        player_diff_total = ((player_diff-1) * replacement_value_by_category).reshape(1,9,1)
        money_diff_total = (money_diff * category_value_per_dollar).reshape(1,9,1)

        #player diff total is subtracted because the team with more players gets less replacement value
        total_diff = score_diff - player_diff_total + money_diff_total 

        return total_diff
    
    def get_diff_var(self
                    , n_their_players : int) -> float:

        """
        Gets the variance of the differential between the current drafter and an opponent

        Args:
            n_their_players: Number of players on the other team that have already been selected
        Returns:
            Float, expected variance
        """
        #diff_var should just include the player-to-player variance. maybe? and 

        chi = self.chi if self.scoring_format == 'Rotisserie' else 1

        diff_var = self.n_picks * \
            (2 * chi +  self.cross_player_var * (self.n_picks - n_their_players)/(self.n_picks))
        return diff_var
    
    def get_value_of_money_auction(self
                                   , diff_means
                                   , diff_vars
                                   , n_values
                                   , category_value_per_dollar
                                   , replacement_value_by_category
                                   , max_money = 200
                                   , step_size = 0.1):
        
        """Estimates a monetary equivalent of the value of a particular player. Assumes money is spent indiscriminately

        Args:
            diff_means : 
            diff_vars: 
            n_values:
            category_value_per_dollar:
            replacement_value_by_category:
            max_money: maximum amount of money to check 
            step_size: increment by which to check 
        Returns:
            1x9 array, expected difference in scores by category
        """
        x_diff_array = np.concatenate([diff_means + replacement_value_by_category + category_value_per_dollar * x * step_size \
                                       for x in range(int(max_money/step_size))])

        cdf_estimates = self.get_cdf(x_diff_array, diff_vars)

        score = self.get_objective_and_pdf_weights(
                                cdf_estimates
                                , None
                                , n_values
                                , calculate_pdf_weights = False)
        
        money_df = pd.DataFrame({'value' : score}
                                , index = [x * step_size for x in range(int(max_money/step_size))])
                
        return money_df
        
        


    def perform_iterations(self
                           ,weights : pd.DataFrame
                           ,my_players : list[str]
                           ,n_players_selected : int
                           ,diff_means : pd.Series
                           ,diff_vars : pd.Series
                           ,x_scores_available_array : pd.DataFrame
                           ,result_index
                           ,n_values
                           ) -> dict:
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

        starting_team_comp = self.positions_float.loc[self.players].sum(axis = 0)

        team_comps = np.concatenate([starting_team_comp + self.positions_float.loc[[player]] for player in result_index]
                                    , axis = 0)
        
        
        while True:

            #case where many players need to be chosen
            if (n_players_selected < self.n_picks - 1) & (self.dynamic):

                start = datetime.datetime.now()

                position_rewards = self.get_position_priorities_from_weights(weights)

                future_position_array = optimize_positions_all_players(self.positions.loc[result_index]
                                                                       ,position_rewards
                                                                       ,self.positions.loc[self.players])

                L = np.einsum('aijk, bi-> bjk',self.L_by_position ,future_position_array)
                position_mu = np.einsum('aij, bi-> bj',self.position_means ,future_position_array)
                position_mu = np.expand_dims(position_mu, axis = 2)

                del_full = self.get_del_full(weights, L)

                expected_future_diff_single = self.get_x_mu_simplified_form(weights, L) + position_mu
                expected_future_diff = ((self.n_picks-1-n_players_selected) * expected_future_diff_single).reshape(-1,9,1)

                x_diff_array = diff_means + x_scores_available_array + expected_future_diff

                pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
                cdf_estimates = self.get_cdf(x_diff_array, diff_vars)
        
                score, pdf_weights = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , n_values
                                        , calculate_pdf_weights = True)

                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)
        
                step_size = self.alpha * (i + 1)**(-self.beta) 
                change_weights = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)
        
                weights = weights + change_weights

                weights[weights < 0] = 0

                weights = weights/weights.sum(axis = 1).reshape(-1,1)

            #case where one more player needs to be chosen
            elif (n_players_selected == (self.n_picks - 1)) | ((not self.dynamic) & (n_players_selected < (self.n_picks)) ): 

                x_diff_array = diff_means + x_scores_available_array

                cdf_estimates = self.get_cdf(x_diff_array
                                            , diff_vars)

                weights = None
                expected_future_diff = None
                pdf_estimates = None
                
                score = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
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

            if self.value_of_money is not None:
                
                score = [(self.value_of_money['value'] - s).abs().idxmin()/100 for s in score]


            #if i == 25:
            #    print(n_players_selected)
            #    try:
            #        print(future_position_array)
            #    except:
            #        pass

            yield {'Scores' : pd.Series(score, index = result_index)
                    ,'Weights' : pd.DataFrame(weights, index = result_index, columns = self.x_scores.columns)
                    ,'Rates' : pd.DataFrame(cdf_means, index = result_index, columns = self.x_scores.columns)
                    ,'Diff' : pd.DataFrame(expected_diff_means, index = result_index, columns = self.x_scores.columns)}

    ### below are functions used for the optimization procedure 
    def get_position_priorities_from_weights(self, weights):

        res = np.einsum('ij, akj -> ik', weights/self.v.T, self.position_means)
        return res
    
    '''
    def get_default_position_matrix(self, position_priorities):

        top_position_score = position_priorities.max(axis =2)

        top_guard_score = position_priorities[:,:,(1,2)].max(axis =2)
        top_forward_score = position_priorities[:,:,(3,4)].max(axis =2)

        nc = 2 + (3 * (position_priorities[:,:,0] == top_position_score).astype(int))
        npg = 1 + (2 * (position_priorities[:,:,1] == top_guard_score).astype(int)) + \
                                3 * (position_priorities[:,:,1] == top_position_score).astype(int)
        nsg = 1 + (2 * (position_priorities[:,:,2] == top_guard_score).astype(int)) + \
                                3 * (position_priorities[:,:,2] == top_position_score).astype(int)
        npf = 1 + (2 * (position_priorities[:,:,3] == top_forward_score).astype(int)) + \
                                3 * (position_priorities[:,:,3] == top_position_score).astype(int)
        nsf = 1 + (2 * (position_priorities[:,:,4] == top_forward_score).astype(int)) + \
                                3 * (position_priorities[:,:,4] == top_position_score).astype(int)
        
        default_position_matrix = np.concatenate([nc,npg,nsg,npf,nsf], axis = 1)

        return default_position_matrix

    def get_future_position_matrix(self, team_comps, default_position_matrix):

        #should try replacing this with a cvxpy optimization version later 
        
        future_position_matrix = default_position_matrix - team_comps

        future_position_matrix = future_position_matrix/(self.n_picks - len(self.players) - 1)

        return future_position_matrix
    '''

    
    def get_objective_and_pdf_weights(self
                                        ,cdf_estimates : np.array
                                        , pdf_estimates : np.array
                                        , n_values : np.array = None
                                        , calculate_pdf_weights : bool = False):
        """
        Calculate the objective function and optionally pdf weights for the gradient 

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            n_values: order of matchup means. Useful for Toro
            calculate_pdf_weights: True if pdf weights should also be returned, in addition to objective

        Returns:
            Objective or Objective, Gradient 
        """

        if self.scoring_format == 'Head to Head: Most Categories':

            return self.get_objective_and_pdf_weights_mc(
                        cdf_estimates
                        , pdf_estimates
                        , calculate_pdf_weights) 

        elif self.scoring_format == 'Rotisserie':

            return self.get_objective_and_pdf_weights_rotisserie(
                        cdf_estimates
                        , pdf_estimates
                        , n_values
                        , calculate_pdf_weights) 

        elif self.scoring_format == 'Head to Head: Each Category':
            return self.get_objective_and_pdf_weights_ec(
                        cdf_estimates
                        , pdf_estimates
                        , calculate_pdf_weights) 

    def get_objective_and_pdf_weights_mc(self
                                , cdf_estimates : np.array
                                , pdf_estimates : np.array
                                , calculate_pdf_weights : bool = False):
        """
        Calculate the objective function and optionally pdf weights for the gradient, for Most Categories

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            calculate_pdf_weights: True if pdf weights should also be returned, in addition to objective

        Returns:
            Objective or Objective, Gradient 
        """

        objective = combinatorial_calculation(cdf_estimates
                                                , 1 - cdf_estimates
                                                ).mean(axis = 1)

        if calculate_pdf_weights:

            tipping_points = calculate_tipping_points(np.array(cdf_estimates))   

            pdf_weights = (tipping_points*pdf_estimates).mean(axis = 2)

            return objective, pdf_weights

        else: 

            return objective


    def get_objective_and_pdf_weights_ec(self
                            , cdf_estimates : np.array
                            , pdf_estimates : np.array
                            , calculate_pdf_weights : bool = False):

        """
        Calculate the objective function and optionally pdf weights for the gradient, for Each Category

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            calculate_pdf_weights: True if pdf weights should also be returned, in addition to objective

        Returns:
            Objective or Objective, Gradient 
        """
        objective = cdf_estimates.mean(axis = 2).mean(axis = 1) 

        if calculate_pdf_weights:

            pdf_weights = (pdf_estimates).mean(axis = 2)

            return objective, pdf_weights

        else:
            return objective


    def get_objective_and_pdf_weights_rotisserie(self
                                , cdf_estimates : np.array
                                , pdf_estimates : np.array
                                , n_values : np.array = None
                                , calculate_pdf_weights : bool = False
                                , test_mode : bool = False):

        """
        Calculate the objective function and optionally pdf weights for the gradient, for Roto

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            n_values: order of matchup means. Useful for Roto
            calculate_pdf_weights: True if pdf weights should also be returned, in addition to objective

        Returns:
            Objective or Objective, Gradient 
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

        drafter_variance = category_variance.sum(axis = 1).reshape(-1,1,1)

        total_variance = drafter_variance + self.var_m

        objective = norm.cdf(drafter_mean - self.mu_m
                            , scale = np.sqrt(total_variance)).reshape(-1)
        
        if calculate_pdf_weights:

            nabla = total_variance + (self.mu_m - drafter_mean) * (n_opponents - n_values - mu_values + 0.5) 

            outer_pdf = norm.pdf((drafter_mean - self.mu_m)/np.sqrt(total_variance))

            gradient = nabla * outer_pdf/ (total_variance * np.sqrt(total_variance))

            if test_mode:
                return gradient
            else:
                pdf_weights = (gradient*pdf_estimates).mean(axis = 2)

                return objective, pdf_weights

        else: 

            return objective

    def get_pdf(self
                , x_diff_array : np.array
                , diff_vars : np.array) -> np.array:
        """
        Calculate the PDF via arrays of means and variance

        Args:
            x_diff_array: array of category means against opponents by candidate player x category x opponent
            diff_vars: array of differential variances, by category x opponent

        Returns:
            Array of PDFs
        """
        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]
                                                    , x_diff_array.shape[1] * x_diff_array.shape[2])
        diff_vars_reshaped = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])

        pdf_estimates = norm.pdf(diff_array_reshaped, scale = np.sqrt(diff_vars_reshaped))

        pdf_estimates_reshaped = pdf_estimates.reshape(x_diff_array.shape)

        return pdf_estimates_reshaped
    
    def get_cdf(self
            , x_diff_array : np.array
            , diff_vars : np.array) -> np.array:
        """
        Calculate the CDF via arrays of means and variance

        Args:
            x_diff_array: array of category means against opponents by candidate player x category x opponent
            diff_vars: array of differential variances, by category x opponent

        Returns:
            Array of CDFs
        """

        #need to resize
        diff_array_reshaped = x_diff_array.reshape(x_diff_array.shape[0]
                                                    , x_diff_array.shape[1] * x_diff_array.shape[2])
        diff_vars_reshaped = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])

        cdf_estimates = norm.cdf(diff_array_reshaped, scale = np.sqrt(diff_vars_reshaped))

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

    def get_x_mu_simplified_form(self,c, L):
        last_four_terms = self.get_last_four_terms(c,L)
        x_mu = np.einsum('aij, ajk -> aik',L, last_four_terms)
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

    def get_term_five(self,c,L):
        return self.get_term_five_a(c,L)/self.get_term_five_b(c,L)

    def get_term_five_a(self,c, L):


        v_dot_v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.dot(self.v.T), L)
        factor_top = np.einsum('pad, dp -> ap', v_dot_v_T_dot_L, c.T)

        v_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_dot_L_dot_v = np.einsum('pad, dc -> ap', v_dot_L, self.v)

        factor =  (factor_top/v_dot_L_dot_v).T
        
        c_mod = c - factor
        c_mod_dot_L = np.einsum('pc, pcd -> pd', c_mod, L)
        c_mod_dot_L_c_mod = np.einsum('pd, pd -> p', c_mod_dot_L, c_mod)

        res = np.sqrt(c_mod_dot_L_c_mod.reshape(-1,1,1))

        return res

    def get_term_five_b(self,c,L):

        c_dot_L = np.einsum('pc, pcd -> pd', c, L)
        c_dot_L_c = np.einsum('pd, pd -> p', c_dot_L, c)

        v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_T_dot_L_dot_v = np.einsum('pad, dc -> ap', v_T_dot_L, self.v)

        L_dot_c_T = np.einsum('pcd, dp -> cp', L, c.T)
        v_T_dot_L_dot_c = np.einsum('ac, cp -> ap', self.v.T, L_dot_c_T)

        res = (c_dot_L_c * v_T_dot_L_dot_v - v_T_dot_L_dot_c**2).reshape(-1,1,1)

        return res

    def get_terms_four_five(self,c,L):
        #is this the right shape
        return self.get_term_four(c) * self.get_term_five(c,L)

    def get_del_term_four(self,c):
        return (np.identity(9) * self.gamma).reshape(1,9,9)

    def get_del_term_five_a(self,c,L):

        v_dot_v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.dot(self.v.T), L)
        factor_top = np.einsum('pad, dp -> ap', v_dot_v_T_dot_L, c.T)

        v_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_dot_L_dot_v = np.einsum('pad, dj -> jp', v_dot_L, self.v)

        factor =  (factor_top/v_dot_L_dot_v).T

        c_mod = c - factor

        top_og = np.einsum('pc, pcd -> pd', c_mod, L)

        top = top_og.reshape(-1,1,9)
        bottom = np.sqrt((np.einsum('pd, pd -> p',top_og,c_mod)).reshape(-1,1,1))

        side= np.identity(9) - np.einsum('ac, pcd -> pad', self.v.dot(self.v.T), L)/v_dot_L_dot_v.reshape(-1,1,1)
        res = np.einsum('pia, pad -> pid', top/bottom, side)

        return res.reshape(-1,1,9)

    def get_del_term_five_b(self,c,L):

        c_dot_L = np.einsum('pc, pcd -> pd', c, L)

        v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_T_dot_L_dot_v = np.einsum('pad, dj -> paj', v_T_dot_L, self.v)

        L_dot_c_T = np.einsum('pcd, dp -> cp', L, c.T)
        v_T_dot_L_dot_c = np.einsum('ac, cp -> ap', self.v.T, L_dot_c_T)

        term_one = (2 * c_dot_L * v_T_dot_L_dot_v.reshape(-1,1)).reshape(-1,1,9)
        term_two = (2 * v_T_dot_L_dot_c.T).reshape(-1,1,1)
        term_three = v_T_dot_L.reshape(-1,1,9)

        res = term_one.reshape(-1,1,9) - (term_two * term_three).reshape(-1,1,9)

        return res

    def get_del_term_five(self,c,L):
        a = self.get_term_five_a(c,L)
        del_a = self.get_del_term_five_a(c,L)
        b = self.get_term_five_b(c,L)
        del_b = self.get_del_term_five_b(c,L)

        return (del_a * b - a * del_b) / b**2

    def get_del_terms_four_five(self,c,L):
        return self.get_term_four(c) * self.get_del_term_five(c,L) + \
                    self.get_del_term_four(c) * self.get_term_five(c,L)

    def get_last_three_terms(self,c,L):
        return np.einsum('aij, ajk -> aik',L,self.get_terms_four_five(c,L))

    def get_del_last_three_terms(self,c,L):
        return np.einsum('aij, ajk -> aik',L,self.get_del_terms_four_five(c,L))

    def get_last_four_terms(self,c,L):
        term_two = self.get_term_two(c)
        last_three = self.get_last_three_terms(c,L)
        return np.einsum('aij, ajk -> aik', term_two, last_three)

    def get_del_last_four_terms(self,c,L):
        comp_i = self.get_del_term_two(c)
        comp_ii = self.get_last_three_terms(c,L)
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,9))
        term_b = np.einsum('aij, ajk -> aik', self.get_term_two(c), self.get_del_last_three_terms(c,L))
        return term_a + term_b

    def get_del_full(self,c, L):
        return np.einsum('aij, ajk -> aik',L,self.get_del_last_four_terms(c,L))

    def make_pick(self
                  , player_assignments : dict[list]
                  , j : int
                  ): 

        generator = self.get_h_scores(player_assignments, j)

        res_list = []
        for i in range(30):
            res = next(generator)
            res_list = res_list + [res] 

        scores = res['Scores']

        available_players_sorted = scores.sort_values(ascending = False)    

        if self.positions is not None:
            player = choose_eligible_player(self.players, available_players_sorted.index, self.positions)     
        else: 
            player = available_players_sorted.index[0]

        if self.collect_info:

            self.all_res_list = self.all_res_list + [res_list]
            self.players = self.players + [player]

        return player

class SimpleAgent():
    #Comment

    def __init__(self, order, positions = None):
        self.order = order
        self.players = []
        self.positions = positions

    def make_pick(self, player_assignments : dict[list], j : int) -> str:

        players_chosen = [x for v in player_assignments.values() for x in v]

        #ZR: Can this be done more efficiently?
        available_players = [p for p in self.order if not p in players_chosen]

        if self.positions is not None:

            player = choose_eligible_player(self.players, available_players, self.positions)
                    
        else: 
            player = available_players[0]

        self.players = self.players + [player]
        return player



def choose_eligible_player(team, available_players, positions):
    for player in available_players:
        new_team = team+ [player]
        if check_team_eligibility(positions.loc[new_team]):
            return player


