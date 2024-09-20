import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import os
from itertools import combinations
from src.algorithm_helpers import combinatorial_calculation, calculate_tipping_points
from src.process_player_data import get_category_level_rv
import streamlit as st 
from src.helper_functions import get_position_structure, get_position_indices, get_L_weights
from src.position_optimization import optimize_positions_all_players, check_single_player_eligibility, check_all_player_eligibility
import datetime
import scipy

class HAgent():

    def __init__(self
                 , info : dict
                 , omega : float
                 , gamma : float
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
            n_picks: int, number of picks each drafter gets 
            n_drafters : int, number of drafters
            scoring_format

            positions: Series of format {'LeBron James' -> ['PF', 'C']}
        Returns:
            None

        """
        self.omega = omega
        self.gamma = gamma
        self.n_picks = n_picks 
        self.n_drafters = n_drafters
        self.dynamic = dynamic
        self.chi = chi

        #ZR: we really need to fix this later lol. The thing is that the positions table 
        #in snowflake for 2011 is slightly messed up for JR smith and we need to fix it
        if positions is None:
            self.positions = info['Positions']
        else:
            self.positions = positions

        self.collect_info = collect_info
        
        self.cross_player_var = info['Var']
        self.scoring_format = scoring_format

        x_scores = info['X-scores']

        self.n_categories = x_scores.shape[1]

        if info['Position-Means'] is not None:
            self.position_means = np.array(info['Position-Means']).reshape(1,-1,self.n_categories)
        else:
            self.position_means = None

        L_by_position = info['L-by-Position']
        L_by_position = np.array(L_by_position).reshape(1,-1,self.n_categories,self.n_categories)

        L_weights = get_L_weights().values.reshape(1,-1,1,1)
        self.L = (L_by_position * L_weights).sum(axis = 1) #ZR: This should weight by base position options 

        self.fudge_factor = fudge_factor

        mov = info['Mov']
        vom = info['Vom']

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[info['Z-scores'].sum(axis = 1).sort_values(ascending = False).index]
            v = np.sqrt(mov/vom)  

            #scale is standard deviation of overall "luck"
            player_stat_luck_overall = np.sqrt(self.n_categories)

            max_luck_expected =  norm.ppf((self.n_drafters - 1 - 0.375)/(self.n_drafters - 1 + 0.25)) * \
                                    player_stat_luck_overall
            
            player_stat_luck_per_category = max_luck_expected * self.fudge_factor /self.n_categories

            max_cdf = norm.cdf(player_stat_luck_per_category)

            ev_max_wins = max_cdf * (self.n_drafters-1) * self.n_categories

            self.mu_m = ev_max_wins
            self.var_m = max_cdf * (1-max_cdf) * (self.n_drafters-1) * self.n_categories

        else:
            self.x_scores = x_scores.loc[info['G-scores'].sum(axis = 1).sort_values(ascending = False).index]

            v = np.sqrt(mov/(mov + vom))

        self.v = np.array(v/v.sum()).reshape(self.n_categories,1)

        turnover_inverted_v = self.v.copy()
        turnover_inverted_v[-1] = -turnover_inverted_v[-1]
        self.turnover_inverted_v = turnover_inverted_v/turnover_inverted_v.sum()

        self.category_weights = None
        self.utility_shares = None
        self.forward_shares = None
        self.guard_shares = None

        self.position_structure = get_position_structure()
        self.position_indices = get_position_indices(self.position_structure)

        self.initial_category_weights = None

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
        my_players = [p for p in player_assignments[drafter] if p == p]

        self.players = my_players #this is a bit of a hack

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

        default_weights = self.v.T.reshape(1,self.n_categories,1)

        if self.scoring_format == 'Rotisserie':
            category_momentum_factor = 10000
        else:
            category_momentum_factor = 1000

        if self.initial_category_weights is None:

            initial_category_weights = ((diff_means + x_scores_available_array)/((default_weights * category_momentum_factor)) + \
                    default_weights).mean(axis = 2)
            initial_category_weights = initial_category_weights/(initial_category_weights.sum(axis = 1).reshape(-1,1))

                    
            initial_position_shares = {
                        position_code : 
                                    pd.DataFrame(
                                        {base_position_code : [1/len(position_info['bases'])] * len(x_scores_available_array)
                                        for base_position_code in position_info['bases']}
                                                ) 
                        for position_code, position_info in self.position_structure['flex'].items()  
                                        }
                        
        else: 

            initial_category_weights = np.array([self.initial_category_weights] * len(x_scores_available))
            
            initial_position_shares = {
                        position_code : 
                                    pd.DataFrame(
                                        {base_position_code : [self.initial_shares[position_code][base_position_code]] * \
                                                                            len(x_scores_available_array)
                                        for base_position_code in position_info['bases']}
                                                ) 
                        for position_code, position_info in self.position_structure['flex'].items()  
                                        }
                

        return self.perform_iterations(initial_category_weights
                                       ,initial_position_shares
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

        my_players = [p for p in player_assignments[drafter] if p == p]

        print(my_players)

        x_self_sum = np.array(self.x_scores.loc[my_players].sum(axis = 0))

        #assume that players for the rest of the round will be chosen from the default ordering 
        players_chosen = [x for v in player_assignments.values() for x in v if x == x]

        if cash_remaining_per_team:

            total_cash_remaining = np.sum([v for k, v in cash_remaining_per_team.items()])

            remaining_players = self.n_drafters * self.n_picks - len(players_chosen)

            #weight by v to get generic v-weighted value
            replacement_value = (x_scores_available.iloc[remaining_players] * self.v.T.reshape(self.n_categories)).sum()
            remaining_overall_value = ((x_scores_available.iloc[0:remaining_players] * self.v.T).sum(axis = 1) \
                                        - replacement_value).sum()
            value_per_dollar = remaining_overall_value/total_cash_remaining

            #when translating back to x-scores, reverse the basis by dividing by v 

            category_value_per_dollar = value_per_dollar / (self.turnover_inverted_v * self.n_categories) 

            replacement_value_by_category = get_category_level_rv(replacement_value
                                                                  , pd.Series(self.v.reshape(-1)
                                                                            , index = self.x_scores.columns) #used for category names
                                                    )
            replacement_value_by_category = np.array(replacement_value_by_category).reshape(self.n_categories,1)

            diff_means = np.vstack(
                [self.get_diff_means_auction(x_self_sum.reshape(1,self.n_categories,1) - \
                                                np.array(self.x_scores.loc[players].sum(axis = 0)).reshape(1,self.n_categories,1)
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

            diff_means = x_self_sum.reshape(1,self.n_categories,1) - \
                        other_team_sums.reshape(1,self.n_categories,self.n_drafters - 1)

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
            #diff_means = diff_means + player_variance_adjustment * scale.values.reshape(1,self.n_categories,1)

        else:
            n_values = None

        diff_vars = np.vstack(
            [self.get_diff_var(len([p for p in players if p == p])) for team, players \
                                    in player_assignments.items() if team != drafter]
        ).T
        
        diff_vars = diff_vars.reshape(1,self.n_categories,self.n_drafters - 1)

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
            1xself.n_categories array, scores by category
        """
        players = [p for p in players if p == p]

        if n_players == self.n_picks:
            n_extra_players = 0
        else:
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
            1xself.n_categories array, expected difference in scores by category
        """
        
        player_diff_total = ((player_diff-1) * replacement_value_by_category).reshape(1,self.n_categories,1)
        money_diff_total = (money_diff * category_value_per_dollar).reshape(1,self.n_categories,1)

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

        #is cross_player_var just the diagonal entries of L? 

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
            1xself.n_categories array, expected difference in scores by category
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
                           ,category_weights : pd.DataFrame
                           ,position_shares : dict[pd.DataFrame]
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
            category_weights: Starting choice of weights. Relevant for case (1)
            utility_shares: starting fraction of future utility spots expected to be used for each position
            guard_shares: starting fraction of future guard spots expected to be used for each position
            forward_shares: starting fraction of future forward spots expected to be used for each position
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

        optimizers = {'Categories' : AdamOptimizer(learning_rate = 0.001)
                      ,'Shares' : {position_code : AdamOptimizer(learning_rate = 0.01)
                                      for position_code in self.position_structure['flex'].keys()}
        }

        while True:

                            
            category_weights_current = category_weights
            position_shares_current = position_shares

            #case where many players need to be chosen
            if (n_players_selected < self.n_picks - 1) & (self.dynamic):

                res = self.get_objective_and_gradient(category_weights
                                                    ,position_shares
                                                    ,diff_means
                                                    ,diff_vars
                                                    ,x_scores_available_array
                                                    ,result_index
                                                    ,n_players_selected
                                                    ,n_values)
                score = res['Score']
                gradients = res['Gradients']
                cdf_estimates = res['CDF-Estimates']
                expected_future_diff = res['Future-Diffs']   
                rosters = res['Rosters']

                category_gradients_centered = gradients['Categories'] - gradients['Categories'].mean(axis = 1).reshape(-1,1) 
                
                category_updates = optimizers['Categories'].minimize(category_gradients_centered)

                category_weights = category_weights + category_updates
                category_weights[category_weights < 0] = 0
                category_weights = category_weights/category_weights.sum(axis = 1).reshape(-1,1)

                assert np.all(np.abs(category_weights.sum(axis=1).reshape(-1, 1) - 1) < 1e-8)

                weights_df = pd.DataFrame(category_weights, index = result_index, columns = self.x_scores.columns)
                assert np.all(np.abs(weights_df.sum(axis=1) - 1) < 1e-8)

                #update position shares 
                ######
                if self.position_means is not None:

                    share_gradients_centered = \
                            {
                                k : grad - grad.mean(axis = 1).reshape(-1,1) for k, grad in gradients['Shares'].items()
                            }
                    share_updates = \
                        {
                            k : optimizers['Shares'][k].minimize(grad) for k, grad in share_gradients_centered.items()
                        }


                    #update flex shares and ensure that they stay compliant with their definitions    
                    for position_code in self.position_structure['flex'].keys():

                        position_shares[position_code] = position_shares[position_code] + share_updates[position_code]
                        position_shares[position_code] = np.clip(position_shares[position_code], 0, 1)
                        position_shares[position_code] = position_shares[position_code].div(position_shares[position_code].sum(axis = 1), axis = 0)

                    best_player = score.argmax()

                    self.initial_category_weights = category_weights[best_player]/2 + self.v.reshape(self.n_categories)/2
                    self.initial_shares = {position_code : shares.iloc[best_player]/2 + 1/(2 *shares.shape[1])
                                             for position_code, shares in position_shares.items()
                                            }
                                           
            #case where one more player needs to be chosen
            elif (n_players_selected == (self.n_picks - 1)) | ((not self.dynamic) & (n_players_selected < (self.n_picks)) ): 

                x_diff_array = diff_means + x_scores_available_array

                cdf_estimates = self.get_cdf(x_diff_array
                                            , diff_vars)

                category_weights = None
                expected_future_diff = None
                pdf_estimates = None

                team_positions = self.positions.loc[self.players]
                 

                #ZR: This is actually super inefficient. Should be fixed later
                player_eligibilities = check_all_player_eligibility(self.positions.loc[result_index], team_positions)
                rosters = [1 if x else -1 for x in player_eligibilities]
  
                score = self.get_objective_and_pdf_weights(
                                        cdf_estimates
                                        , pdf_estimates
                                        , n_values
                                        , calculate_pdf_weights = False)

            #case where no new players need to be chosen
            elif (n_players_selected == self.n_picks): 

                cdf_estimates = self.get_cdf(diff_means, diff_vars)

                category_weights_current = None
                position_shares_current = None
                expected_future_diff = None
                pdf_estimates = None
                rosters = None
                
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

                category_weights_current = None
                position_shares_current = None
                expected_future_diff = None
                rosters = None

            i = i + 1

            cdf_means = cdf_estimates.mean(axis = 2)

            if expected_future_diff is not None:
                expected_diff_means = expected_future_diff.mean(axis = 2)
            else:
                expected_diff_means = None

            if self.value_of_money is not None:
                
                score = [(self.value_of_money['value'] - s).abs().idxmin()/100 for s in score]

            res = {'Scores' : pd.Series(score, index = result_index)
                    ,'Weights' : pd.DataFrame(category_weights_current, index = result_index, columns = self.x_scores.columns)
                    ,'Rates' : pd.DataFrame(cdf_means, index = result_index, columns = self.x_scores.columns)
                    ,'Diff' : pd.DataFrame(expected_diff_means, index = result_index, columns = self.x_scores.columns)
                    ,'Rosters' : pd.DataFrame(rosters, index = result_index)
                    ,'Position-Shares' : {position_code : 
                                                    pd.DataFrame(position_shares_current[position_code].values
                                                                 , index = result_index
                                                     , columns = position_info['bases']) 
                                                     for position_code, position_info in self.position_structure['flex'].items()
                                            } if position_shares_current is not None else \
                                                {position_code : None for position_code in self.position_structure['flex'].keys() }
                    
                    }

            yield res

    ### below are functions used for the optimization procedure 
    def get_position_priorities_from_category_weights(self, weights):

        res = np.einsum('ij, akj -> ik', weights/self.v.T, self.position_means)
        return res

    def get_objective_and_gradient(self
                                    ,category_weights
                                    ,position_shares
                                    ,diff_means
                                    ,diff_vars
                                    ,x_scores_available_array
                                    ,result_index
                                    ,n_players_selected
                                    ,n_values
                                    ,):
        
            #calculate scores and category-level gradients
            ######

            if self.position_means is not None:

                position_rewards = self.get_position_priorities_from_category_weights(category_weights)

                rosters, future_position_array, flex_shares = optimize_positions_all_players(self.positions.loc[result_index]
                                                                        ,position_rewards
                                                                        ,self.positions.loc[self.players]
                                                                        , position_shares)
                


                position_mu = np.einsum('aij, bi-> bj',self.position_means ,future_position_array)
                position_mu = np.expand_dims(position_mu, axis = 2)
            else:
                position_mu = 0
                rosters = None
                flex_shares = None

            #this causes an issue with gradients. It doesn't change much so we can just keep L constant
            #L = np.einsum('aijk, bi-> bjk',self.L_by_position ,future_position_array)
            L = self.L

            del_full = (self.n_picks-1-n_players_selected) * self.get_del_full(category_weights, L)

            expected_future_diff_single = self.get_x_mu_simplified_form(category_weights, L) + position_mu
            expected_future_diff = ((self.n_picks-1-n_players_selected) * expected_future_diff_single).reshape(-1,self.n_categories,1)


            x_diff_array = diff_means + x_scores_available_array + expected_future_diff

            pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
            cdf_estimates = self.get_cdf(x_diff_array, diff_vars)
    
            score, pdf_weights = self.get_objective_and_pdf_weights(
                                    cdf_estimates
                                    , pdf_estimates
                                    , n_values
                                    , calculate_pdf_weights = True)

            category_gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)


            if self.position_means is not None:
                position_gradient = np.einsum('ai , aki -> ak', pdf_weights, self.position_means) 

                share_gradients =  {position_code : 
                                        position_gradient[:,self.position_indices[position_code]]  * flex_share.reshape(-1,1)
                                        for position_code, flex_share in flex_shares.items()
                                    }

                gradients =   {
                                'Categories' : category_gradient
                                ,'Shares' : share_gradients
                                }

            else:
                gradients =  {
                        'Categories' : category_gradient
                    }

                flex_shares = None

            res = {'Score' : score
                ,'Gradients' : gradients
                ,'CDF-Estimates' : cdf_estimates
                ,'Flex-Shares' : flex_shares
                , 'Future-Diffs' : expected_future_diff
                , 'Rosters' : rosters
            }



            return res 

    
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
                                                            , self.n_categories
                                                            , 1) 

        variance_contributions = cdf_estimates * \
                                    (2 * n_opponents - 2 * n_values - 2 * mu_values + 1 )
        category_variance = variance_contributions.sum(axis = 2).reshape(-1
                                                            , self.n_categories
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

        #the addition of diff_array_reshaped is the experimental Poisson adjustment 
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
        
        U = np.array([[self.v.reshape(self.n_categories),c_0.reshape(self.n_categories)] for c_0 in c])
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

    def get_term_two(self,c, L = None):
        return - self.v.reshape(-1,self.n_categories,1) * c.reshape(-1,1,self.n_categories) + \
                            c.reshape(-1,self.n_categories,1) * self.v.reshape(-1,1,self.n_categories)

    def get_del_term_two(self,c, L = None):
        arr_a = np.zeros((self.n_categories,self.n_categories,self.n_categories))
        for i in range(self.n_categories):
            arr_a[i,:,i] = self.v.reshape(self.n_categories,)

        arr_b = np.zeros((self.n_categories,self.n_categories,self.n_categories))
        for i in range(self.n_categories):
            arr_b[:,i,i] = self.v.reshape(self.n_categories,)  

        arr_full = arr_a - arr_b

        return arr_full.reshape(1,self.n_categories,self.n_categories,self.n_categories)

    def get_term_four(self,c, L = None):
        #v = np.array([1/self.n_categories] * self.n_categories).reshape(self.n_categories,1)

        return (c * self.gamma).reshape(-1,self.n_categories,1) + (self.v * self.omega).reshape(1,self.n_categories,1)

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

    def get_del_term_four(self,c, L = None):
        return (np.identity(self.n_categories) * self.gamma).reshape(1,self.n_categories,self.n_categories)

    def get_del_term_five_a(self,c,L):

        v_dot_v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.dot(self.v.T), L)
        factor_top = np.einsum('pad, dp -> ap', v_dot_v_T_dot_L, c.T)

        v_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_dot_L_dot_v = np.einsum('pad, dj -> jp', v_dot_L, self.v)

        factor =  (factor_top/v_dot_L_dot_v).T

        c_mod = c - factor

        top_og = np.einsum('pc, pcd -> pd', c_mod, L)

        top = top_og.reshape(-1,1,self.n_categories)
        bottom = np.sqrt((np.einsum('pd, pd -> p',top_og,c_mod)).reshape(-1,1,1))

        side= np.identity(self.n_categories) - np.einsum('ac, pcd -> pad', self.v.dot(self.v.T), L)/v_dot_L_dot_v.reshape(-1,1,1)
        res = np.einsum('pia, pad -> pid', top/bottom, side)

        return res.reshape(-1,1,self.n_categories)

    def get_del_term_five_b(self,c,L):

        c_dot_L = np.einsum('pc, pcd -> pd', c, L)

        v_T_dot_L = np.einsum('ac, pcd -> pad', self.v.T, L)
        v_T_dot_L_dot_v = np.einsum('pad, dj -> paj', v_T_dot_L, self.v)

        L_dot_c_T = np.einsum('pcd, dp -> cp', L, c.T)
        v_T_dot_L_dot_c = np.einsum('ac, cp -> ap', self.v.T, L_dot_c_T)

        term_one = (2 * c_dot_L * v_T_dot_L_dot_v.reshape(-1,1)).reshape(-1,1,self.n_categories)
        term_two = (2 * v_T_dot_L_dot_c.T).reshape(-1,1,1)
        term_three = v_T_dot_L.reshape(-1,1,self.n_categories)

        res = term_one.reshape(-1,1,self.n_categories) - (term_two * term_three).reshape(-1,1,self.n_categories)

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
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,self.n_categories))
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
            res_list.append(res)

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

    team_positions = positions.loc[team]
    for player in available_players:        
        if check_single_player_eligibility(positions.loc[player], team_positions):
            return player


class AdamOptimizer:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def minimize(self, gradient):
        if self.m is None:
            self.m = 0
            self.v = 0

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
        
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return update 