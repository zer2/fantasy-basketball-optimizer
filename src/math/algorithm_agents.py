import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
import os
from itertools import combinations
from src.math.algorithm_helpers import combinatorial_calculation, calculate_tipping_points
from src.math.process_player_data import get_category_level_rv
import streamlit as st 
from src.helpers.helper_functions import get_league_type, get_position_structure, get_position_indices, get_L_weights, get_selected_categories, get_rho \
                                            ,get_max_info
from src.math.position_optimization import optimize_positions_all_players, check_single_player_eligibility, check_all_player_eligibility
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
                 , team_names : list = None
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
        if team_names is not None:
            self.team_names = team_names
        else:
            self.team_names = st.session_state.team_names

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
            self.x_scores = x_scores.loc[info['G-scores'].sum(axis = 1).sort_values(ascending = False).index]
            v = np.sqrt(mov/vom)  

            categories = x_scores.columns
            rho = np.array(get_rho().set_index('Category').loc[categories, categories])

            #ZR: This next line is not necessary is it?
            np.fill_diagonal(rho, 1)
            self.rho = np.expand_dims(rho, 0)

            if self.n_drafters <= 21:
                self.max_ev, self.max_var = get_max_info(self.n_drafters - 1)

            else:
                self.max_ev = np.sqrt( 2 * np.log(self.n_drafters - 1))
                self.max_var = 2/(self.n_drafters - 1)

        else:
            self.x_scores = x_scores.loc[info['G-scores'].sum(axis = 1).sort_values(ascending = False).index]

            v = np.sqrt(mov/(mov + vom))
        
        self.original_v = np.array(v)
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

        if get_league_type() == 'MLB':
            cats = x_scores.columns
            self.pitching_stat_indices = [i for i in range(len(cats)) if cats[i] in st.session_state.params['pitcher_stats']]
            self.batting_stat_indices = [i for i in range(len(cats)) if i not in self.pitching_stat_indices]

            self.pitching_L = self.L[:,self.pitching_stat_indices][:,:,self.pitching_stat_indices]
            self.batting_L = self.L[:,self.batting_stat_indices][:,:,self.batting_stat_indices]


            batting_v = v[self.batting_stat_indices]
            pitching_v = v[self.pitching_stat_indices]

            self.batting_v = np.array(batting_v/batting_v.sum()).reshape(-1,1)
            self.pitching_v = np.array(pitching_v/pitching_v.sum()).reshape(-1,1)

            self.average_round_value = info['Average-Round-Value']

            #if you were to transfer one point of G-score from a batter to a pitcher, what would it look like?
            #start by inverting v: this converts one point of G-score into one point of X-score
            #ZR: shouldnt this be self.original_v?
            pitching_preference_vector = 1/self.v
            #normalize so that the scores add up to 1 for both hitters and batters
            pitching_preference_vector[self.pitching_stat_indices] = pitching_preference_vector[self.pitching_stat_indices]/ \
                                                                    pitching_preference_vector[self.pitching_stat_indices].sum()
            pitching_preference_vector[self.batting_stat_indices] = - pitching_preference_vector[self.batting_stat_indices]/ \
                                                                        pitching_preference_vector[self.batting_stat_indices].sum()

            #multiply by two because v naturally adds up to 1
            self.pitching_preference_vector = pitching_preference_vector
            self.pitching_preference_damper = 1

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
        self.n_drafters = len(player_assignments) #ZR: Kind of a hack, but it helps sometimes when the session state gets messed up
        my_players = [p for p in player_assignments[drafter] if p == p]

        self.players = my_players #this is a bit of a hack

        n_players_selected = len(my_players) 

        players_chosen = [x for v in player_assignments.values() for x in v if x == x]

        x_scores_available = self.x_scores[~self.x_scores.index.isin(players_chosen + exclusion_list) & \
                                                self.x_scores.index.isin(self.positions.index)]
        
        total_players = self.n_picks * self.n_drafters

        diff_means, diff_vars, sigma_2_m = self.get_diff_distributions(player_assignments
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
                                       , sigma_2_m)

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
                                                np.array(self.x_scores.loc[player_assignments[team]].sum(axis = 0)).reshape(1,self.n_categories,1)
                                            , cash_remaining_per_team[drafter] - cash_remaining_per_team[team]
                                            , len(my_players) - len(player_assignments[team])
                                            , category_value_per_dollar
                                            , replacement_value_by_category) for team \
                                        in self.team_names if team != drafter]
            ).T

        else: 

            extra_players_needed = (len(my_players)+1) * self.n_drafters - len(players_chosen) 
            mean_extra_players = x_scores_available.iloc[0:extra_players_needed].mean().fillna(0)

            other_team_sums = np.vstack(
                [self.get_opposing_team_means(player_assignments[team], mean_extra_players, len(my_players)) for team \
                                        in self.team_names if team != drafter]
            ).T

            diff_means = x_self_sum.reshape(1,self.n_categories,1) - \
                        other_team_sums.reshape(1,self.n_categories,self.n_drafters - 1)

        diff_vars = np.vstack(
            [self.get_diff_var(len([p for p in player_assignments[team] if p == p])) for team \
                                    in self.team_names if team != drafter]
        ).T

        #make order statistics adjustment for Roto 
        if self.scoring_format == 'Rotisserie':
            sigma_c = (diff_means / np.sqrt(diff_vars))[0,:,:].std(axis = 1, ddof = 1) * np.sqrt(2)

            h_m = self.get_h_m(sigma_c,self.n_drafters)

            sigma_2_m = self.get_sigma_2_m(sigma_c
                            ,  h_m
                            , self.rho
                            , self.n_drafters)
        else:
            sigma_2_m = None

        diff_vars = diff_vars.reshape(1,self.n_categories,self.n_drafters - 1)

        #ZR: this is a super ugly implementation
        if cash_remaining_per_team:
            self.value_of_money = self.get_value_of_money_auction(
                        diff_means
                        , diff_vars
                        , sigma_2_m
                        , category_value_per_dollar
                        , replacement_value_by_category)
        else:
            self.value_of_money = None


        return diff_means, diff_vars, sigma_2_m

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
                                   , sigma_2_m
                                   , category_value_per_dollar
                                   , replacement_value_by_category
                                   , max_money = 200
                                   , step_size = 0.1):
        
        """Estimates a monetary equivalent of the value of a particular player. Assumes money is spent indiscriminately

        Args:
            diff_means : 
            diff_vars: 
            sigma_2_m:
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
                                x_diff_array
                                , diff_vars
                                , cdf_estimates
                                , None
                                , sigma_2_m
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
                           ,sigma_2_m
                           ):
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

        if get_league_type() == 'MLB':
            optimizers['Pitcher Preference'] = AdamOptimizer(learning_rate = 0.05)
            pitching_preference = 0
        else:
            pitching_preference = None

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
                                                    ,sigma_2_m
                                                    ,pitching_preference)
                
                score = res['Score']
                gradients = res['Gradients']
                cdf_estimates = res['CDF-Estimates']
                expected_future_diff = res['Future-Diffs']   
                rosters = res['Rosters']

                category_gradients_centered = gradients['Categories'] - gradients['Categories'].mean(axis = 1).reshape(-1,1) 
                
                category_updates = optimizers['Categories'].minimize(category_gradients_centered)

                category_weights = category_weights + category_updates
                category_weights[category_weights < 0] = 0

                if get_league_type() == 'NBA':
                    category_weights = category_weights/category_weights.sum(axis = 1).reshape(-1,1)
                elif get_league_type() == 'MLB': 
                    batting_weights = category_weights[:,self.batting_stat_indices] 
                    category_weights[:,self.batting_stat_indices] = batting_weights/(2 * batting_weights.sum(axis = 1).reshape(-1,1))

                    pitching_weights = category_weights[:,self.pitching_stat_indices] 
                    category_weights[:,self.pitching_stat_indices] = pitching_weights/(2 * pitching_weights.sum(axis = 1).reshape(-1,1))
                    
                    #make the pitching adjustment 
                    pitching_preference_update = optimizers['Pitcher Preference'].minimize(gradients['Pitcher Preference'])
                    pitching_preference = np.clip(pitching_preference + pitching_preference_update, -0.5, 0.5)

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
                category_weights_current = None
                expected_future_diff = None
                pdf_estimates = None

                team_positions = self.positions.loc[self.players]
                 

                #ZR: This is actually super inefficient. Should be fixed later
                player_eligibilities = check_all_player_eligibility(self.positions.loc[result_index], team_positions)
                rosters = [1 if x else -1 for x in player_eligibilities]
  
                score = self.get_objective_and_pdf_weights(
                                        x_diff_array
                                        , diff_vars
                                        , cdf_estimates
                                        , pdf_estimates
                                        , sigma_2_m
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
                                        diff_means
                                        , diff_vars
                                        , cdf_estimates
                                        , pdf_estimates
                                        , sigma_2_m
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
                                        diff_means
                                        , diff_vars
                                        , cdf_estimates
                                        , pdf_estimates
                                        , sigma_2_m
                                        , calculate_pdf_weights = False)

                result_index = drop_potentials.index

                category_weights_current = None
                position_shares_current = None
                expected_future_diff = None
                rosters = None

            i = i + 1

            cdf_means = cdf_estimates.mean(axis = 2)

            if expected_future_diff is not None:
                expected_diff_means = (expected_future_diff.mean(axis = 2) + diff_means.mean(axis = 2)) / (self.original_v.reshape(1,-1))
            else:
                expected_diff_means = None

            if self.value_of_money is not None:
                
                score = [(self.value_of_money['value'] - s).abs().idxmin()/100 for s in score]

            if expected_future_diff is not None:    
                future_diff_df = pd.DataFrame(np.squeeze(expected_future_diff), index = result_index, columns = self.x_scores.columns)
            else:
                future_diff_df = None

            res = {'Scores' : pd.Series(score, index = result_index)
                    ,'Weights' : pd.DataFrame(category_weights_current, index = result_index, columns = self.x_scores.columns)
                    ,'Rates' : pd.DataFrame(cdf_means, index = result_index, columns = self.x_scores.columns)
                    ,'Diff' : pd.DataFrame(expected_diff_means, index = result_index, columns = self.x_scores.columns)
                    ,'Future-Diff' : future_diff_df
                    ,'Rosters' : pd.DataFrame(rosters, index = result_index)
                    ,'Position-Shares' : {position_code : 
                                                    pd.DataFrame(position_shares_current[position_code].values
                                                                 , index = result_index
                                                     , columns = position_info['bases']) 
                                                     for position_code, position_info in self.position_structure['flex'].items()
                                            } if position_shares_current is not None else \
                                                {position_code : None for position_code in self.position_structure['flex'].keys() }
                    ,'CDFs' : [pd.DataFrame(cdf_estimates[:,:,i]
                                            , index = result_index
                                            , columns = get_selected_categories()) for i in range(self.n_drafters -1)]
                    
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
                                    ,sigma_2_m
                                    ,pitching_preference = None
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


            if st.session_state.league == 'NBA':
                expected_future_diff_single = self.get_x_mu_simplified_form(category_weights, L, self.v) + position_mu
                del_full = (self.n_picks-1-n_players_selected) * self.get_del_full(category_weights, L, self.v)

            elif st.session_state.league == 'MLB':

                #ZR: This works for now, but should be careful about it 
                pitching_share = future_position_array[:, -2:].sum(axis = 1).reshape(-1,1,1)
                batting_share = 1 - pitching_share

                batting_diff = self.get_x_mu_simplified_form(category_weights[:,self.batting_stat_indices]
                                                                            , self.batting_L
                                                                            , self.batting_v) 
                batting_diff_single = batting_diff * batting_share
                
                pitching_diff = self.get_x_mu_simplified_form(category_weights[:,self.pitching_stat_indices]
                                                            , self.pitching_L
                                                            , self.pitching_v) 
                pitching_diff_single = pitching_diff * pitching_share

                #pitching preference adjustment
                convertible_slots = (np.minimum(batting_share, pitching_share) * (self.n_picks-1-n_players_selected)).astype(int)
                total_convertible_value_map = {slots : self.average_round_value[n_players_selected:n_players_selected + slots].sum() + \
                                        self.average_round_value[-slots:].sum() for slots in pd.unique(convertible_slots[:,0,0])}
                total_convertible_value = np.array([total_convertible_value_map[x] for x in convertible_slots[:,0,0]]) * \
                                            self.pitching_preference_damper

                values_converted = (total_convertible_value * pitching_preference).reshape(-1,1,1) * \
                                    self.pitching_preference_vector.reshape(1,-1,1)
                                                
                expected_future_diff_single = np.concatenate([batting_diff_single,pitching_diff_single], axis = 1) \
                                                            + position_mu + values_converted/(self.n_picks-1-n_players_selected)
    

                del_batting = self.get_del_full(category_weights[:,self.batting_stat_indices]
                                                , self.batting_L
                                                , self.batting_v)
                del_pitching = self.get_del_full(category_weights[:,self.pitching_stat_indices]
                                                 , self.pitching_L
                                                 , self.pitching_v)

                del_full = np.zeros(shape = (del_batting.shape[0], self.n_categories, self.n_categories))
                del_full[:,:del_batting.shape[1], :del_batting.shape[1]] = del_batting * batting_share
                del_full[:,del_batting.shape[1]:, del_batting.shape[1]:] = del_pitching * pitching_share
                del_full = del_full * (self.n_picks-1-n_players_selected) 

            expected_future_diff = ((self.n_picks-1-n_players_selected) * expected_future_diff_single).reshape(-1,self.n_categories,1)

            x_diff_array = diff_means + x_scores_available_array + expected_future_diff

            pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
            cdf_estimates = self.get_cdf(x_diff_array, diff_vars)
    
            score, pdf_weights = self.get_objective_and_pdf_weights(
                                    x_diff_array
                                    , diff_vars
                                    , cdf_estimates
                                    , pdf_estimates
                                    , sigma_2_m
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

            if get_league_type() == 'MLB':
                #this is just proportional to the gradient- it is not exact
                gradients['Pitcher Preference'] = np.einsum('ai , ik -> a', pdf_weights, self.pitching_preference_vector) 

            res = {'Score' : score
                ,'Gradients' : gradients
                ,'CDF-Estimates' : cdf_estimates
                ,'Flex-Shares' : flex_shares
                , 'Future-Diffs' : expected_future_diff
                , 'Rosters' : rosters
            }



            return res 

    
    def get_objective_and_pdf_weights(self
                                        ,x_diff_array
                                        ,diff_vars
                                        ,cdf_estimates : np.array
                                        , pdf_estimates : np.array
                                        , sigma_2_m : np.array = None
                                        , calculate_pdf_weights : bool = False):
        """
        Calculate the objective function and optionally pdf weights for the gradient 

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            sigma_2_m: order of matchup means. Useful for Toro
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
                        x_diff_array
                        , diff_vars
                        , cdf_estimates
                        , pdf_estimates
                        , sigma_2_m
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
                                , x_diff_array : np.array
                                , diff_vars : np.array
                                , cdf_estimates : np.array
                                , pdf_estimates : np.array
                                , sigma_2_m : float
                                , calculate_pdf_weights : bool = False
                                , test_mode : bool = False):

        """
        Calculate the objective function and optionally pdf weights for the gradient, for Roto

        Args:
            cdf_estimates: array of CDF at 0 estimates for differentials against opponents
            pdf_estimates: array of PDF at 0 estimates for differentials against opponents
            sigma_2_m: order of matchup means. Useful for Roto
            calculate_pdf_weights: True if pdf weights should also be returned, in addition to objective

        Returns:
            Objective or Objective, Gradient 
        """
        #make sure the PDF and CDF estimates, plus diff_means, are adjusted to the specs of Roto

        diff_means = x_diff_array / np.sqrt(diff_vars) 
        pdf_estimates = norm.pdf(diff_means)
        #CDF estimates dont need to be adjusted
        f = self.get_f(pdf_estimates)
        g = self.get_g(pdf_estimates)
        f00 = f[0][0] * f[0][0]

        h_p = self.get_h_p(f,g)

        sigma_2_l = self.get_sigma_2_l(sigma_2_m, self.n_drafters)
        sigma_2_p = self.get_sigma_2_p(cdf_estimates, h_p, self.rho)

        mu_l = self.get_mu_l(sigma_2_m, self.n_drafters)
        mu_p = self.get_mu_p(cdf_estimates)

        sigma_2_d = self.get_sigma_2_d( sigma_2_p, sigma_2_l, self.n_drafters).reshape(-1,1,1)
        mu_d = self.get_mu_d(mu_p, mu_l, self.n_drafters, self.n_categories).reshape(-1,1,1)

        sigma_d = np.sqrt(sigma_2_d)
        objective = self.get_v(mu_d, sigma_d)
    
        if calculate_pdf_weights:

            del_sigma_2_p = self.get_del_sigma_2_p(diff_means
                            , self.rho
                            , pdf_estimates
                            , cdf_estimates
                            , f)

            del_mu_d = self.get_del_mu_d(self.n_drafters, pdf_estimates)
        
            gradient = self.get_del_v(sigma_d, del_mu_d, mu_d, del_sigma_2_p)

            #remember that del_v is relative to the modified basis for mu. It must be translated back into the regular mu basis
            gradient = gradient * np.sqrt(diff_vars)

            if test_mode:
                return gradient
            else:
                pdf_weights = gradient.sum(axis = 2)

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

    def get_x_mu_long_form(self,c, v):
        #uses the pre-simplified formula for x_mu from page 19 of the paper

        factor = (v.dot(v.T).dot(self.L).dot(c.T)/v.T.dot(self.L).dot(v)).T

        c_mod = c - factor
        sigma = np.sqrt((c_mod.dot(self.L) * c_mod).sum(axis = 1))
        
        U = np.array([[v.reshape(self.n_categories),c_0.reshape(self.n_categories)] for c_0 in c])
        b = np.array([[-self.gamma * s,self.omega * s] for s in sigma]).reshape(-1,2,1)
        U_T = np.swapaxes(U, 1, 2)
        
        q = np.einsum('aij, ajk -> aik', U.dot(self.L), U_T)
        inverse_part = np.linalg.inv(q)

        r = np.einsum('ij, ajk -> aik', self.L, U_T)

        x = np.einsum('aij, ajk -> aik', r, inverse_part)

        x_mu = np.einsum('aij, ajk -> aik', x, b)

        return x_mu

    def get_x_mu_simplified_form(self,c, L, v):
        last_four_terms = self.get_last_four_terms(c,L, v)
        x_mu = np.einsum('aij, ajk -> aik',L, last_four_terms)
        return x_mu

    def clear_initial_weights(self):
        self.initial_category_weights = None
        self.initial_position_shares = None
        return self

    #below functions use the simplified form of X_mu 
    #term 1: L (covariance)
    #term 2: vj^T - jv^T
    #term 3: L (covariance)
    #term 4: -gamma * j - omega * v
    #term 5: sigma / (j^T L j v^T L V - (v^T L j)^2) 

    def get_term_two(self,c, v):
        return - v.reshape(-1,v.shape[0],1) * c.reshape(-1,1,v.shape[0]) + \
                            c.reshape(-1,v.shape[0],1) * v.reshape(-1,1,v.shape[0])

    def get_del_term_two(self,v ):
        arr_a = np.zeros((v.shape[0],v.shape[0],v.shape[0]))
        for i in range(v.shape[0]):
            arr_a[i,:,i] = v.reshape(v.shape[0],)

        arr_b = np.zeros((v.shape[0],v.shape[0],v.shape[0]))
        for i in range(v.shape[0]):
            arr_b[:,i,i] = v.reshape(v.shape[0],)  

        arr_full = arr_a - arr_b

        return arr_full.reshape(1,v.shape[0],v.shape[0],v.shape[0])

    def get_term_four(self,c, v):
        #v = np.array([1/v.shape[0]] * v.shape[0]).reshape(v.shape[0],1)

        return (c * self.gamma).reshape(-1,v.shape[0],1) + (v * self.omega).reshape(1,v.shape[0],1)

    def get_term_five(self,c,L,v):
        return self.get_term_five_a(c,L,v)/self.get_term_five_b(c,L,v)

    def get_term_five_a(self,c, L,v):


        v_dot_v_T_dot_L = np.einsum('ac, pcd -> pad', v.dot(v.T), L)
        factor_top = np.einsum('pad, dp -> ap', v_dot_v_T_dot_L, c.T)

        v_dot_L = np.einsum('ac, pcd -> pad', v.T, L)
        v_dot_L_dot_v = np.einsum('pad, dc -> ap', v_dot_L, v)

        factor =  (factor_top/v_dot_L_dot_v).T
        
        c_mod = c - factor
        c_mod_dot_L = np.einsum('pc, pcd -> pd', c_mod, L)
        c_mod_dot_L_c_mod = np.einsum('pd, pd -> p', c_mod_dot_L, c_mod)

        res = np.sqrt(c_mod_dot_L_c_mod.reshape(-1,1,1))

        return res

    def get_term_five_b(self,c,L,v):

        c_dot_L = np.einsum('pc, pcd -> pd', c, L)
        c_dot_L_c = np.einsum('pd, pd -> p', c_dot_L, c)

        v_T_dot_L = np.einsum('ac, pcd -> pad', v.T, L)
        v_T_dot_L_dot_v = np.einsum('pad, dc -> ap', v_T_dot_L, v)

        L_dot_c_T = np.einsum('pcd, dp -> cp', L, c.T)
        v_T_dot_L_dot_c = np.einsum('ac, cp -> ap', v.T, L_dot_c_T)

        res = (c_dot_L_c * v_T_dot_L_dot_v - v_T_dot_L_dot_c**2).reshape(-1,1,1)

        return res

    def get_terms_four_five(self,c,L, v):
        #is this the right shape
        return self.get_term_four(c, v) * self.get_term_five(c,L,v)

    def get_del_term_four(self,c, v):
        return (np.identity(v.shape[0]) * self.gamma).reshape(1,v.shape[0],v.shape[0])

    def get_del_term_five_a(self,c,L,v):

        v_dot_v_T_dot_L = np.einsum('ac, pcd -> pad', v.dot(v.T), L)
        factor_top = np.einsum('pad, dp -> ap', v_dot_v_T_dot_L, c.T)

        v_dot_L = np.einsum('ac, pcd -> pad', v.T, L)
        v_dot_L_dot_v = np.einsum('pad, dj -> jp', v_dot_L, v)

        factor =  (factor_top/v_dot_L_dot_v).T

        c_mod = c - factor

        top_og = np.einsum('pc, pcd -> pd', c_mod, L)

        top = top_og.reshape(-1,1,v.shape[0])
        bottom = np.sqrt((np.einsum('pd, pd -> p',top_og,c_mod)).reshape(-1,1,1))

        side= np.identity(v.shape[0]) - np.einsum('ac, pcd -> pad', v.dot(v.T), L)/v_dot_L_dot_v.reshape(-1,1,1)
        res = np.einsum('pia, pad -> pid', top/bottom, side)

        return res.reshape(-1,1,v.shape[0])

    def get_del_term_five_b(self,c,L, v):

        c_dot_L = np.einsum('pc, pcd -> pd', c, L)

        v_T_dot_L = np.einsum('ac, pcd -> pad', v.T, L)
        v_T_dot_L_dot_v = np.einsum('pad, dj -> paj', v_T_dot_L, v)

        L_dot_c_T = np.einsum('pcd, dp -> cp', L, c.T)
        v_T_dot_L_dot_c = np.einsum('ac, cp -> ap', v.T, L_dot_c_T)

        term_one = (2 * c_dot_L * v_T_dot_L_dot_v.reshape(-1,1)).reshape(-1,1,v.shape[0])
        term_two = (2 * v_T_dot_L_dot_c.T).reshape(-1,1,1)
        term_three = v_T_dot_L.reshape(-1,1,v.shape[0])

        res = term_one.reshape(-1,1,v.shape[0]) - (term_two * term_three).reshape(-1,1,v.shape[0])

        return res

    def get_del_term_five(self,c,L,v):
        a = self.get_term_five_a(c,L,v)
        del_a = self.get_del_term_five_a(c,L,v)
        b = self.get_term_five_b(c,L,v)
        del_b = self.get_del_term_five_b(c,L,v)

        return (del_a * b - a * del_b) / b**2

    def get_del_terms_four_five(self,c,L,v):
        return self.get_term_four(c, v) * self.get_del_term_five(c,L,v) + \
                    self.get_del_term_four(c, v) * self.get_term_five(c,L,v)

    def get_last_three_terms(self,c,L,v):
        return np.einsum('aij, ajk -> aik',L,self.get_terms_four_five(c,L,v))

    def get_del_last_three_terms(self,c,L,v):
        return np.einsum('aij, ajk -> aik',L,self.get_del_terms_four_five(c,L,v))

    def get_last_four_terms(self,c,L,v):
        term_two = self.get_term_two(c,v)
        last_three = self.get_last_three_terms(c,L,v)
        return np.einsum('aij, ajk -> aik', term_two, last_three)

    def get_del_last_four_terms(self,c,L, v):
        comp_i = self.get_del_term_two(v)
        comp_ii = self.get_last_three_terms(c,L,v)
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,v.shape[0]))
        term_b = np.einsum('aij, ajk -> aik', self.get_term_two(c,v), self.get_del_last_three_terms(c,L,v))
        return term_a + term_b

    def get_del_full(self,c, L,v):
        return np.einsum('aij, ajk -> aik',L,self.get_del_last_four_terms(c,L,v))
    

    #below functions implement the Rotisserie objective 
    #helpers
    def get_f(self, pdfs : np.array) -> np.array:
        #equation 1
        return pdfs.sum(axis = 2)

    def get_g(self, pdfs : np.array) -> np.array:
        #equation 2
        return np.einsum('pao,pbo -> pab', pdfs, pdfs)

    def get_h_p(self
                ,f : np.array
                ,g : np.array) -> np.array :
        g1 = g.copy()
        g1[:,np.arange(g.shape[1]),np.arange(g.shape[2])] = 0

        g2 = g * np.expand_dims(np.identity(self.n_categories),0)

        f_part = np.einsum('pa, pb -> pab', f,f)

        return f_part + g1 - g2

    def get_h_m(self
                ,sigma_c : np.array
                ,n_managers : int ) -> np.array:
        sigmas_mod = (sigma_c**2) + 1

        sigma_matrix = np.sqrt(np.einsum('a, b -> ab', sigmas_mod,sigmas_mod))

        first_version = n_managers/sigma_matrix - (2/sigma_matrix)*np.identity(len(sigma_c))
    
        return (n_managers - 1)/(2 * np.pi) * first_version

    #main functions

    def get_v(self
            ,mu_d
            ,sigma_d) -> np.array:
        #equation 5
        return norm.cdf(mu_d/sigma_d).reshape(-1)

    def get_mu_d(self
                , mu_p : float
                , mu_l : float
                , n_managers : int
                , n_categories : int) -> float:
        #equation 6
        first_component = mu_p*n_managers/(n_managers - 1)
        second_component = n_categories * n_managers/2
        return first_component - second_component - mu_l

    def get_sigma_2_d( self
                    , sigma_2_p : float
                    , sigma_2_l : float
                    , n_managers : int) -> float:
        #equation 7
        first_component = sigma_2_p * n_managers / (n_managers - 1)
        return first_component + sigma_2_l

    def get_mu_p(self
                , cdfs : np.array) -> float:
        #equation 8
        return cdfs.sum(axis = (1,2))

    def get_mu_l(self
                , sigma_2_m : float
                , n_managers : int) -> float :
        #equation 9
        return self.max_ev * np.sqrt(sigma_2_m)

    def get_sigma_2_p(self
                    , cdfs : np.array
                    , h_p : np.array
                    , rho : np.array) -> float:
        #equation 10
        first_component = (cdfs * (1 - cdfs)).sum(axis = (1,2))
        second_component = (rho * h_p).sum(axis = (1,2))/2
        return first_component + second_component
    
    def get_sigma_2_l(self
                    , sigma_2_m : np.array
                    , n_managers : int) -> np.array:
        #equation 11
        return sigma_2_m * self.max_var

    def get_sigma_2_m(self
                    , sigma_c : np.array
                    , h_m : np.array
                    , rho : np.array
                    , n_managers : int) -> float:
        #equation 12
        sigma_squared = sigma_c**2
        component_1 = (n_managers - 1) * np.arccos(sigma_squared/(1 + sigma_squared)).sum()/(2 * np.pi)
        component_2 = (rho * h_m).sum(axis = (1,2))/2
        return component_1 + component_2


    def get_del_sigma_2_p(self
                        , opponent_mu_matrix : np.array
                        , rho : np.array
                        , pdfs : np.array
                        , cdfs : np.array
                        , f : np.array) -> np.array:
        

        rho_ignoring_diag = rho.copy()
        rho_ignoring_diag[:,np.arange(rho_ignoring_diag.shape[1]),np.arange(rho_ignoring_diag.shape[2])] = 0
    
        #ZR: first component is totally wrong
        inside = - pdfs - f.reshape(-1,f.shape[1],1)
        first_part = np.einsum('pab, pbc -> pac', rho_ignoring_diag, inside)

        first_component = (opponent_mu_matrix * pdfs) * (first_part + (pdfs - f.reshape(-1,f.shape[1],1)))
        second_component = pdfs * (1 - 2 * cdfs)

        return first_component + second_component

    def get_del_mu_d(self
                    , n_managers : int
                    , pdfs : np.array) -> np.array:
        return n_managers/(n_managers - 1) * pdfs

    def get_del_v(self
                , sigma_d : np.array
                , del_mu_d : np.array
                , mu_d : np.array
                , del_sigma_2_p : np.array) -> np.array:
        del_v = norm.pdf(mu_d/sigma_d)/(sigma_d**3) * (sigma_d**2 * del_mu_d - mu_d * del_sigma_2_p/2)
        return del_v

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
    
#ZR: This should be a method of the H-score class
@st.cache_data(show_spinner = False, ttl = 3600)
def get_base_h_score(_info : dict
                , omega : float
                , gamma : float
                , n_picks : int
                , n_drafters : int
                , scoring_format : str
                , chi : float
                , player_assignments : dict[list[str]]
                , team : str
                , info_key : int):
  """Calculate your team's H-score

  Args:
    info: dictionary with info related to player statistics etc. 
    omega: float, parameter as described in the paper
    gamma: float, parameter as described in the paper
    n_picks: int, number of picks each drafter gets 
    n_drafters: int, number of drafters
    scoring_format: 
    player_assignments : player assignment dictionary
    team: name of team to evaluate

  Returns:
      None
  """

  H = HAgent(info = _info
    , omega = omega
    , gamma = gamma
    , n_picks = n_picks
    , n_drafters = n_drafters
    , dynamic = False
    , scoring_format = scoring_format
    , chi = chi)

  return next(H.get_h_scores(player_assignments, team))   