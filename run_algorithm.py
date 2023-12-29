#2: based on drafting situation: run gradient descent, get results

def run_algorithm():
  return 'Run algorithm'

#below here experimental

class UAgent():

    def __init__(self
                 , omega
                 , gamma
                 , alpha
                 , beta
                 , n_iterations 
                 , n_players = n_players
                 , winner_take_all
                 , info
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
        self.v = v
        self.L = L
        self.n_iterations = n_iterations
        self.n_players = n_players
        self.winner_take_all = winner_take_all 
        self.x_scores = info['X-scores']
        self.score_table = info['Score-table']
        self.smoothed_score_table = info['Smoothed-score-table']
        self.diff_var = info['Diff-var']
      
    def make_pick(self
                  , df
                  , my_players
                  , all_players_chosen ):

        """Picks a player based on the D-score algorithm

        Args:
            player_assignments: dict of format
                   player : team that picked them

        Returns:
            String indicating chosen player
        """
        round_n = len(self.players) 

        x_self_sum = self.x_scores.loc[self.players].sum(axis = 0)
        
        current_score_table = self.score_table[0:round_n].sum()
        diff_means =  x_self_sum - current_score_table.loc[(self.x_scores.columns,'mean')].droplevel(1)
        
        previous_rounds_expected = self.score_table.iloc[0:round_n].sum().loc[(self.x_scores.columns,'mean')].droplevel(1)
        this_round_expected = self.score_table_smoothed.iloc[len(player_assignments)].values
        diff_means = x_self_sum - previous_rounds_expected - this_round_expected
        
        other_team_variance = self.score_table.loc[0:12,(self.x_scores.columns,'var')].sum().droplevel(1)
        rest_of_team_variance = self.score_table.loc[(round_n + 1):12,(self.x_scores.columns,'var')].sum().droplevel(1)

        top_players = self.x_scores[self.x_scores.index.isin(self.representative_player_set)]
        diff_var = 26 + top_players.var() * 13

        x_scores_available = self.x_scores[~self.x_scores.index.isin(player_assignments.keys())]
        
        c = np.array((diff_means + x_scores_available)/(self.v.T * 500) + self.v.T)
        
        #first_moment = np.zeros(shape = c.shape)
        #second_moment = np.zeros(shape = c.shape)
        
        scores = []
        weights = []

        if round_n < 12:
            for i in range(self.n_iterations):

                del_full = self.get_del_full(c,L)

                expected_x = self.get_x_mu(c,L)
                expected_future_diff = ((12-round_n) * expected_x).reshape(-1,9)

                pdf_estimates = norm.pdf(diff_means + x_scores_available + expected_future_diff
                                          , scale = np.sqrt(diff_var))

                if self.winner_take_all:


                    tipping_points = calculate_tipping_points(cdf_estimates)   

                    pdf_weights = (tipping_points*pdf_estimates)
                else:
                    pdf_weights = pdf_estimates

                gradient = np.einsum('ai , aik -> ak', pdf_weights, del_full)

                #first_moment = self.beta_1 * first_moment + (1- self.beta_1) * gradient 
                #second_moment = self.beta_2 * second_moment + (1- self.beta_2) * gradient**2 

                #first_moment_unbiased = first_moment/(1 - self.beta_1**2)
                #second_moment_unbiased = second_moment/(1 - self.beta_2**2)

                step_size = self.alpha * (i + 1)**(-self.beta) #* first_moment_unbiased/(np.sqrt(second_moment_unbiased) + 1E-8)
                change_c = step_size * gradient/np.linalg.norm(gradient,axis = 1).reshape(-1,1)

                c = c + change_c
                c[c < 0] = 0
                c = c/c.sum(axis = 1).reshape(-1,1)

                cdf_estimates = norm.cdf(diff_means + x_scores_available + expected_future_diff
                          , scale = np.sqrt(diff_var))
                
                scores = scores + [pd.DataFrame(cdf_estimates.mean(axis = 1), index = x_scores_available.index) ]
                weights = weights + [pd.DataFrame(c, index = x_scores_available.index) ]

        else:
            expected_future_diff = 0
            
        win_probabilities = pd.DataFrame(norm.cdf(diff_means + x_scores_available + expected_future_diff
                                                  , scale = np.sqrt(diff_var))
                                         ,index = x_scores_available.index)

        win_probabilities.columns = x_scores_available.columns

        if self.winner_take_all:
            win_sums = combinatorial_calculation(win_probabilities
                                                          , 1 - win_probabilities
                                                          , categories = win_probabilities.columns
                             )
        else:
            win_sums = win_probabilities.sum(axis = 1) #+ optimal_punt_reward

        win_sums.name = 'value'
        players_and_positions = pd.merge(win_sums
                                         , self.positions
                                         , left_index = True
                                         , right_index = True)
        #players_and_positions['pos'] = [list(eval(x)) for x in players_and_positions['pos']]
        players_and_positions = players_and_positions.explode('pos')

        replacement_level_players = players_and_positions[~players_and_positions.index.isin(top_players.index)]

        replacement_level_values = pd.Series({pos : 
                                    replacement_level_players[[pos in x for x in replacement_level_players['pos']]].max().values[0] \
                                     for pos in ['C','PF','PG','SF','SG']}
                                            )

        n_per_position = self.positions.loc[self.players].explode().value_counts()
        adjusted_replacement_level_values = adjust_replacement_level_values(n_per_position,replacement_level_values)
        adjusted_replacement_level_values.name = 'replacement_value'

        joined = pd.merge(players_and_positions
                                     , adjusted_replacement_level_values, right_index = True, left_on = 'pos')
        mined = joined.groupby('player').min()
        adjusted_win_sums = mined['value'] #- mined['replacement_value']

        players_sorted = adjusted_win_sums.sort_values(ascending = False)
                
        player = self.pick_from_order(players_sorted)
        
        self.info[round_n] = {'scores' : scores,'weights' : weights}
        return player
    
    
    def get_x_mu(self,c,L):

        factor = (self.v.dot(self.v.T).dot(L).dot(c.T)/self.v.T.dot(L).dot(self.v)).T

        c_mod = c - factor
        sigma = np.sqrt((c_mod.dot(L) * c_mod).sum(axis = 1))
        U = np.array([[self.v.reshape(9),c_0.reshape(9)] for c_0 in c])
        b = np.array([[-self.gamma * s,self.omega * s] for s in sigma]).reshape(-1,2,1)

        U_T = np.swapaxes(U, 1, 2)

        q = np.einsum('aij, ajk -> aik', U.dot(L), U_T)

        inverse_part = np.linalg.inv(q)

        r = np.einsum('ij, ajk -> aik', L, U_T)

        x = np.einsum('aij, ajk -> aik', r, inverse_part)

        #inverse_part = np.linalg.inv(U.dot(L).dot(U.T))
        #X_mu = L.dot(U.T).dot(inverse_part).dot(b)

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

    def get_term_five(self,c,L):
        return self.get_term_five_a(c,L)/self.get_term_five_b(c,L)

    def get_term_five_a(self,c,L):
        factor =  (self.v.dot(self.v.T).dot(L).dot(c.T)/self.v.T.dot(L).dot(self.v)).T
        c_mod = c - factor
        return np.sqrt((c_mod.dot(L) * c_mod).sum(axis = 1).reshape(-1,1,1))

    def get_term_five_b(self,c,L):
        return ((c.dot(L) * c).sum(axis = 1) * self.v.T.dot(L).dot(self.v) - self.v.T.dot(L.dot(c.T))**2).reshape(-1,1,1)

    def get_terms_four_five(self,c,L):
        #is this the right shape
        return self.get_term_four(c) * self.get_term_five(c,L)

    def get_del_term_four(self,c):
        return (np.identity(9) * self.gamma).reshape(1,9,9)

    def get_del_term_five_a(self,c,L):
        factor = (self.v.dot(self.v.T).dot(L).dot(c.T)/self.v.T.dot(L).dot(self.v)).T

        c_mod = c - factor
        top = c_mod.dot(L).reshape(-1,1,9)
        bottom = np.sqrt((c_mod.dot(L) * c_mod).sum(axis = 1).reshape(-1,1,1))
        side = np.identity(9) - self.v.dot(self.v.T).dot(L)/self.v.T.dot(L).dot(self.v)
        res = (top/bottom).dot(side)
        return res.reshape(-1,1,9)

    def get_del_term_five_b(self,c,L):
        term_one = (2 * c.dot(L) * self.v.T.dot(L).dot(self.v)).reshape(-1,1,9)
        term_two = (2 * self.v.T.dot(L.dot(c.T)).T).reshape(-1,1,1)
        term_three = (self.v.T.dot(L)).reshape(1,1,9)
        return term_one.reshape(-1,1,9) - (term_two * term_three).reshape(-1,1,9)

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
        return np.einsum('ij, ajk -> aik',L,self.get_terms_four_five(c,L))

    def get_del_last_three_terms(self,c,L):
        return np.einsum('ij, ajk -> aik',L,self.get_del_terms_four_five(c,L))

    def get_last_four_terms(self,c,L):
        return np.einsum('ij, ajk -> aik', self.get_term_two(c), self.get_last_three_terms(c,L))

    def get_del_last_four_terms(self,c,L):
        comp_i = self.get_del_term_two(c)
        comp_ii = self.get_last_three_terms(c,L)
        term_a = np.einsum('aijk, aj -> aik', comp_i, comp_ii.reshape(-1,9))
        term_b = np.einsum('aij, ajk -> aik', self.get_term_two(c), self.get_del_last_three_terms(c,L))
        return term_a + term_b

    def get_del_full(self,c,L):
        return np.einsum('ij, ajk -> aik',L,self.get_del_last_four_terms(c,L))
