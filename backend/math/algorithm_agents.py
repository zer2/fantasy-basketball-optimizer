"""
Backend-only copy of src/math/algorithm_agents.py.

Changes vs original:
- HAgent.__init__ takes explicit `sport`, `params`, `slot_counts`, `aleph` params.
- All get_*() / st.session_state calls replaced with self.sport, self._pos_cfg, etc.
- Imports changed to backend.math.position_optimization and backend.math.process_player_data.
- @st.cache_resource removed; build_h_agent is a plain function.
- All pure-math methods (get_pdf, get_term_*, Roto helpers, AdamOptimizer) are identical.
The original src/ file is untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
from itertools import combinations

from backend.math.algorithm_helpers import compute_win_probability, calculate_tipping_points
from backend.math.process_player_data import get_category_level_rv
from backend.math.position_optimization import (
    optimize_positions_all_players,
    check_single_player_eligibility,
    check_all_player_eligibility,
)
from backend.session import PositionConfig, build_position_config


class HAgent:

    def __init__(self,
                 info: dict,
                 omega: float,
                 gamma: float,
                 n_picks: int,
                 n_drafters: int,
                 dynamic: bool,
                 scoring_format: str,
                 # ── explicit context (replaces get_*() calls) ──
                 sport: str,
                 params: dict,
                 slot_counts: dict,
                 aleph: float = 0.0,
                 # ── original optional args ──
                 beth: float = 0,
                 collect_info: bool = False,
                 ):

        self.omega         = omega
        self.gamma         = gamma
        self.n_picks       = n_picks
        self.dynamic       = dynamic
        self.n_drafters    = n_drafters
        self.collect_info  = collect_info
        self.scoring_format = scoring_format

        # ── store explicit context ─────────────────────────────────────────────
        self.sport  = sport
        self.params = params

        # Build position config (replaces all get_position_*() calls)
        self._pos_cfg: PositionConfig = build_position_config(params, slot_counts)

        # ── info dict unpacking ────────────────────────────────────────────────
        self.positions = info['Positions']
        self.w         = info['w']
        x_scores       = info['X-scores']

        self.n_categories = x_scores.shape[1]

        #TODO: clean this up 
        if info['Position-Means'] is not None:
            self.position_means = np.array(info['Position-Means']).reshape(1, -1, self.n_categories)
            
            position_means_df = info['Position-Means']
            position_means_df.loc['NP'] = 0 

            #TODO: This should probably use the average of the position means for a player
            #E.g. if they are a PF/C, this should be the average of the means for PF and C
            #that get subtracted out 
            rel_players = [p for p in x_scores.index if p != 'RP']
            self.pos_avg = pd.DataFrame(
                [position_means_df.loc[self.positions.get(p)[0]]
                for p in rel_players],
                index= rel_players,
                columns=x_scores.columns
            )
        else:
            self.position_means = None
            self.pos_avg = None

        L_by_position = info['L-by-Position']
        L_by_position = np.array(L_by_position).reshape(1, -1, self.n_categories, self.n_categories)

        # ── L_weights (replaces get_L_weights()) ──────────────────────────────
        pn         = self._pos_cfg.position_numbers
        ps         = self._pos_cfg.position_structure
        base_list  = ps['base_list']
        flex_list  = ps['flex_list']
        n_slots    = sum(pn.values())

        lw = pd.Series({p: pn[p] / n_slots for p in base_list})
        for fp in flex_list:
            bases = ps['flex'][fp]['bases']
            for base in bases:
                lw[base] += pn[fp] / (n_slots * len(bases))

        L_weights = lw.values.reshape(1, -1, 1, 1)
        self.L = (L_by_position * L_weights).sum(axis=1)

        mov = info['Mov']
        vom = info['Vom']

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[
                info['G-scores'].sum(axis=1).sort_values(ascending=False).index
            ]
            v = np.sqrt(mov / vom)

            categories = list(x_scores.columns)

            # ── correlations (replaces get_correlations()) ────────────────────
            if sport == 'NBA':
                rho = pd.read_csv('backend/data/basketball_correlations.csv').set_index('Category')
            else:
                rho = pd.read_csv('backend/data/baseball_correlations.csv').set_index('Category')

            counting_stats_all = params['counting-statistics']
            rho.loc[counting_stats_all, counting_stats_all] = np.clip(
                rho.loc[counting_stats_all, counting_stats_all] + aleph, -1, 1
            )
            negative_stats = params['negative-statistics']
            rho.loc[:, negative_stats] = -rho.loc[:, negative_stats]
            rho.loc[negative_stats, :] = -rho.loc[negative_stats, :]
            rho_values = rho.values
            np.fill_diagonal(rho_values, 1)
            rho = pd.DataFrame(rho_values, index=rho.index, columns=rho.index)

            self.rho = np.expand_dims(np.array(rho.loc[categories, categories]), 0)

            # ── max_info (replaces get_max_info()) ────────────────────────────
            if self.n_drafters <= 21:
                max_table = pd.read_csv('backend/data/max_table.csv')
                info_row = max_table.set_index('N').loc[self.n_drafters - 1]
                self.max_ev, self.max_var = float(info_row['EV(X)']), float(info_row['VAR(X)'])
            else:
                self.max_ev  = float(np.sqrt(2 * np.log(self.n_drafters - 1)))
                self.max_var = 2.0 / (self.n_drafters - 1)

        else:
            self.x_scores = x_scores.loc[
                info['G-scores'].sum(axis=1).sort_values(ascending=False).index
            ]
            v = np.sqrt(mov / (mov + vom))

        self.original_v = np.array(v)
        self.v          = np.array(v / v.sum()).reshape(self.n_categories, 1)

        turnover_inverted_v = self.v.copy()
        turnover_inverted_v[-1] = -turnover_inverted_v[-1]
        self.turnover_inverted_v = turnover_inverted_v / turnover_inverted_v.sum()

        self.category_weights  = None
        self.utility_shares    = None
        self.forward_shares    = None
        self.guard_shares      = None

        # ── position structure (replaces get_position_structure()) ────────────
        self.position_structure = self._pos_cfg.position_structure
        self.position_indices   = self._pos_cfg.position_indices

        self.initial_category_weights = None

        # ── MLB-specific setup (replaces get_pitcher_stats() / get_league_type()) ──
        if sport == 'MLB':
            cats = list(x_scores.columns)
            pitcher_stats = params.get('pitcher_stats', [])
            self.pitching_stat_indices = [i for i, c in enumerate(cats) if c in pitcher_stats]
            self.batting_stat_indices  = [i for i in range(len(cats)) if i not in self.pitching_stat_indices]

            self.pitching_L = self.L[:, self.pitching_stat_indices][:, :, self.pitching_stat_indices]
            self.batting_L  = self.L[:, self.batting_stat_indices][:, :, self.batting_stat_indices]

            batting_v  = v[self.batting_stat_indices]
            pitching_v = v[self.pitching_stat_indices]
            self.batting_v  = np.array(batting_v  / batting_v.sum()).reshape(-1, 1)
            self.pitching_v = np.array(pitching_v / pitching_v.sum()).reshape(-1, 1)

            self.average_round_value = info['Average-Round-Value']

            pitching_preference_vector = 1 / self.v
            pitching_preference_vector[self.pitching_stat_indices] = (
                pitching_preference_vector[self.pitching_stat_indices]
                / pitching_preference_vector[self.pitching_stat_indices].sum()
            )
            pitching_preference_vector[self.batting_stat_indices] = (
                -pitching_preference_vector[self.batting_stat_indices]
                / pitching_preference_vector[self.batting_stat_indices].sum()
            )
            self.pitching_preference_vector = pitching_preference_vector
            self.pitching_preference_damper = 1

        self.all_res_list = []
        self.players      = []

        transformation_matrix = (
            np.identity(self.n_categories)
            + np.full(shape=(self.n_categories, self.n_categories),
                      fill_value=beth / self.n_categories ** 2)
        )
        self.transformation_matrix_inverted   = np.linalg.inv(transformation_matrix)
        self.transformation_addition_constant = beth / (2 * self.n_categories)

    # ── public API ─────────────────────────────────────────────────────────────

    def get_h_scores(self,
                     player_assignments: dict,
                     drafter,
                     n_iterations: int,
                     cash_remaining_per_team: dict = None,
                     exclusion_list: list = []) -> dict:

        self.n_drafters = len(player_assignments)
        my_players = [p for p in player_assignments[drafter] if p == p]
        self.players = my_players

        n_players_selected = len(my_players)
        players_chosen     = [x for v in player_assignments.values() for x in v if x == x]

        x_scores_available = self.x_scores[
            ~self.x_scores.index.isin(players_chosen + exclusion_list)
            & self.x_scores.index.isin(self.positions.index)
        ]

        diff_means, diff_vars, sigma_2_m = self.get_diff_distributions(
            player_assignments, drafter, x_scores_available, cash_remaining_per_team
        )

        x_scores_available_array = np.expand_dims(np.array(x_scores_available), axis=2)

        if len(my_players) > 0:
            cdf_original = norm.cdf(
                (diff_means + x_scores_available_array).mean(axis=2),
                scale=np.sqrt(diff_vars.mean(axis=2)),
            )
            cdf_mod = np.einsum(
                'ij,ai -> aj',
                self.transformation_matrix_inverted,
                cdf_original + self.transformation_addition_constant,
            )
            corrected_strength = norm.ppf(cdf_mod, scale=np.sqrt(diff_vars.mean(axis=2)))
            x_scores_available_mod    = corrected_strength - diff_means.mean(axis=2)
            x_scores_available_array  = np.expand_dims(x_scores_available_mod, axis=2)

        default_weights = self.v.T.reshape(1, self.n_categories, 1)
        category_momentum_factor = 10000 if self.scoring_format == 'Rotisserie' else 1000



        if self.initial_category_weights is None:

            #we want to subtract out roughly the player's position means 
            #to account for the fact that they are taking up that slot
            if self.pos_avg is not None:
                pos_avg = self.pos_avg.loc[x_scores_available.index]
                pos_avg_array = np.expand_dims(np.array(pos_avg), axis=2)
            else: 
                pos_avg_array = 0 #not an array, but this will work fine

            #ZR: Below is a hack, to keep things consistent with streamlit. Uncomment it for testing
            #pos_avg_array = 0 

            initial_category_weights = (
                (diff_means + x_scores_available_array - pos_avg_array)
                / (default_weights * category_momentum_factor)
                + default_weights
            ).mean(axis=2)
            initial_category_weights /= initial_category_weights.sum(axis=1).reshape(-1, 1)

            initial_position_shares = {
                pos_code: pd.DataFrame({
                    base: [1 / len(pos_info['bases'])] * len(x_scores_available_array)
                    for base in pos_info['bases']
                })
                for pos_code, pos_info in self.position_structure['flex'].items()
            }
        else:
            initial_category_weights = np.array(
                [self.initial_category_weights] * len(x_scores_available)
            )
            initial_position_shares = {
                pos_code: pd.DataFrame({
                    base: [self.initial_shares[pos_code][base]] * len(x_scores_available_array)
                    for base in pos_info['bases']
                })
                for pos_code, pos_info in self.position_structure['flex'].items()
            }

        return self.perform_iterations(
            initial_category_weights,
            initial_position_shares,
            my_players,
            n_players_selected,
            diff_means,
            diff_vars,
            x_scores_available_array,
            x_scores_available.index,
            sigma_2_m,
            n_iterations,
        )

    def get_diff_distributions(self, player_assignments, drafter,
                               x_scores_available, cash_remaining_per_team=None):
        team_names = list(player_assignments.keys())
        my_players = [p for p in player_assignments[drafter] if p == p]
        x_self_sum = np.array(self.x_scores.loc[my_players].sum(axis=0))
        players_chosen = [x for v in player_assignments.values() for x in v if x == x]

        if cash_remaining_per_team:
            total_cash = sum(cash_remaining_per_team.values())
            remaining_players = self.n_drafters * self.n_picks - len(players_chosen)

            replacement_value = (
                x_scores_available.iloc[remaining_players] * self.v.T.reshape(self.n_categories)
            ).sum()
            remaining_overall_value = (
                (x_scores_available.iloc[:remaining_players] * self.v.T).sum(axis=1)
                - replacement_value
            ).sum()
            value_per_dollar = remaining_overall_value / total_cash
            category_value_per_dollar = value_per_dollar / (self.turnover_inverted_v * self.n_categories)

            replacement_value_by_category = get_category_level_rv(
                replacement_value,
                pd.Series(self.v.reshape(-1), index=self.x_scores.columns),
                list(self.x_scores.columns),
            )
            replacement_value_by_category = np.array(replacement_value_by_category).reshape(self.n_categories, 1)

            diff_means = np.vstack([
                self.get_diff_means_auction(
                    x_self_sum.reshape(1, self.n_categories, 1)
                    - np.array(self.x_scores.loc[player_assignments[team]].sum(axis=0)).reshape(1, self.n_categories, 1),
                    cash_remaining_per_team[drafter] - cash_remaining_per_team[team],
                    len(my_players) - len(player_assignments[team]),
                    category_value_per_dollar,
                    replacement_value_by_category,
                )
                for team in team_names if team != drafter
            ]).T

        else:
            extra_needed     = (len(my_players) + 1) * self.n_drafters - len(players_chosen)
            mean_extra       = x_scores_available.iloc[:extra_needed].mean().fillna(0)
            other_team_sums  = np.vstack([
                self.get_opposing_team_means(player_assignments[team], mean_extra, len(my_players))
                for team in team_names if team != drafter
            ]).T
            diff_means = (
                x_self_sum.reshape(1, self.n_categories, 1)
                - other_team_sums.reshape(1, self.n_categories, self.n_drafters - 1)
            )

        diff_vars = np.vstack([
            self.get_diff_var(len([p for p in player_assignments[team] if p == p]))
            for team in team_names if team != drafter
        ]).T

        if self.scoring_format == 'Rotisserie':
            sigma_c  = (diff_means / np.sqrt(diff_vars))[0, :, :].std(axis=1, ddof=1) * np.sqrt(2)
            h_m      = self.get_h_m(sigma_c, self.n_drafters)
            sigma_2_m = self.get_sigma_2_m(sigma_c, h_m, self.rho, self.n_drafters)
        else:
            sigma_2_m = None

        diff_vars = diff_vars.reshape(1, self.n_categories, self.n_drafters - 1)

        if cash_remaining_per_team:
            self.value_of_money = self.get_value_of_money_auction(
                diff_means, diff_vars, sigma_2_m,
                category_value_per_dollar, replacement_value_by_category,
            )
        else:
            self.value_of_money = None

        return diff_means, diff_vars, sigma_2_m

    def get_opposing_team_means(self, players, mean_extra_players, n_my_players):
        n_extra = max(n_my_players + 1 - len(players), 0)
        player_sum  = np.array(self.x_scores.loc[[p for p in players if p == p]].sum(axis=0))
        extra_sum   = np.array(mean_extra_players) * n_extra
        return (player_sum + extra_sum).reshape(1, self.n_categories, 1)

    def get_diff_means_auction(self, score_diff, money_diff, player_diff,
                                category_value_per_dollar, replacement_value_by_category):
        player_diff_total = ((player_diff - 1) * replacement_value_by_category).reshape(1, self.n_categories, 1)
        money_diff_total  = (money_diff * category_value_per_dollar).reshape(1, self.n_categories, 1)
        return score_diff - player_diff_total + money_diff_total

    def get_diff_var(self, n_their_players):
        return self.n_picks * (2 + self.w * (self.n_picks - n_their_players) / self.n_picks)

    def get_value_of_money_auction(self, diff_means, diff_vars, sigma_2_m,
                                    category_value_per_dollar, replacement_value_by_category,
                                    max_money=200, step_size=0.1):
        x_diff_array = np.concatenate([
            diff_means + replacement_value_by_category + category_value_per_dollar * x * step_size
            for x in range(int(max_money / step_size))
        ])
        cdf_estimates = self.get_cdf(x_diff_array, diff_vars)
        score = self.get_objective_and_pdf_weights(
            x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m, calculate_pdf_weights=False
        )
        return pd.DataFrame({'value': score},
                            index=[x * step_size for x in range(int(max_money / step_size))])

    def perform_iterations(self, category_weights, position_shares, my_players,
                           n_players_selected, diff_means, diff_vars,
                           x_scores_available_array, result_index, sigma_2_m,
                           n_iterations):

        optimizers = {
            'Categories': AdamOptimizer(learning_rate=0.001),
            'Shares': {
                pos_code: AdamOptimizer(learning_rate=0.01)
                for pos_code in self.position_structure['flex'].keys()
            },
        }

        if self.sport == 'MLB':
            optimizers['Pitcher Preference'] = AdamOptimizer(learning_rate=0.05)
            pitching_preference = 0
        else:
            pitching_preference = None

        for _ in range(max(1, n_iterations)):
            category_weights_current  = category_weights
            position_shares_current   = position_shares

            if (n_players_selected < self.n_picks - 1) and self.dynamic:

                gradient_result = self.get_objective_and_gradient(
                    category_weights, position_shares, diff_means, diff_vars,
                    x_scores_available_array, result_index, n_players_selected,
                    sigma_2_m, pitching_preference,
                )

                score               = gradient_result['Score']
                gradients           = gradient_result['Gradients']
                cdf_estimates       = gradient_result['CDF-Estimates']
                expected_future_diff = gradient_result['Future-Diffs']
                rosters             = gradient_result['Rosters']

                cat_grad_centered = gradients['Categories'] - gradients['Categories'].mean(axis=1).reshape(-1, 1)
                cat_updates       = optimizers['Categories'].minimize(cat_grad_centered)
                category_weights  = category_weights + cat_updates
                category_weights[category_weights < 0] = 0

                if self.sport == 'NBA':
                    category_weights = category_weights / category_weights.sum(axis=1).reshape(-1, 1)
                elif self.sport == 'MLB':
                    bw = category_weights[:, self.batting_stat_indices]
                    category_weights[:, self.batting_stat_indices] = bw / (2 * bw.sum(axis=1).reshape(-1, 1))
                    pw = category_weights[:, self.pitching_stat_indices]
                    category_weights[:, self.pitching_stat_indices] = pw / (2 * pw.sum(axis=1).reshape(-1, 1))
                    pp_update = optimizers['Pitcher Preference'].minimize(gradients['Pitcher Preference'])
                    pitching_preference = np.clip(pitching_preference + pp_update, -0.5, 0.5)

                if self.position_means is not None:
                    share_gradients_centered = {
                        k: grad - grad.mean(axis=1).reshape(-1, 1)
                        for k, grad in gradient_result['Gradients']['Shares'].items()
                    }
                    for pos_code, optimizer in optimizers['Shares'].items():
                        share_update = optimizer.minimize(share_gradients_centered[pos_code])
                        position_shares[pos_code] = position_shares[pos_code] + share_update
                        position_shares[pos_code].values[:] = np.clip(position_shares[pos_code].values, 0, 1)
                        position_shares[pos_code] = position_shares[pos_code].div(
                            position_shares[pos_code].sum(axis=1), axis=0
                        )

                # Store best weights for warm-starting
                best_player_idx = np.argmax(score)
                self.initial_category_weights = category_weights[best_player_idx] / 2 + self.v.reshape(self.n_categories) / 2
                self.initial_shares = {
                    pos_code: {
                        base: float(position_shares[pos_code][base].iloc[best_player_idx])
                        for base in self.position_structure['flex'][pos_code]['bases']
                    }
                    for pos_code in self.position_structure['flex'].keys()
                }

            elif (n_players_selected == self.n_picks - 1) or (not self.dynamic and n_players_selected < self.n_picks):
                x_diff_array   = diff_means + x_scores_available_array
                cdf_estimates  = self.get_cdf(x_diff_array, diff_vars)
                score          = self.get_objective_and_pdf_weights(
                    x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m,
                    calculate_pdf_weights=False,
                )
                rosters              = None
                expected_future_diff = None
                category_weights_current = None

            else:
                x_diff_array   = diff_means + x_scores_available_array
                cdf_estimates  = self.get_cdf(x_diff_array, diff_vars)
                score          = self.get_objective_and_pdf_weights(
                    x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m,
                    calculate_pdf_weights=False,
                )
                rosters              = None
                expected_future_diff = None
                category_weights_current = None

        cdf_means = cdf_estimates.mean(axis=2)

        if expected_future_diff is not None:
            expected_diff_means = expected_future_diff.mean(axis=2) + diff_means.mean(axis=2)
        else:
            #ZR: This is a bit of a hack. It just gets diff_means into the right shape
            expected_diff_means = cdf_means * 0 + diff_means.mean(axis=2)

        future_diff_df = (
            pd.DataFrame(expected_future_diff.mean(axis=2),
                         index=result_index, columns=self.x_scores.columns)
            if expected_future_diff is not None else None
        )

        return {
            'Scores':  pd.Series(score, index=result_index),
            'Weights': None if category_weights_current is None else \
                        pd.DataFrame(category_weights_current, index=result_index, columns=self.x_scores.columns),
            'Rates':   pd.DataFrame(cdf_means, index=result_index,
                                    columns=self.x_scores.columns),
            'Diff':    (pd.DataFrame(expected_diff_means, index=result_index,
                                     columns=self.x_scores.columns)
                        if expected_diff_means is not None else None),
            'Future-Diff': future_diff_df,
            'Rosters': pd.DataFrame(rosters, index=result_index) if rosters is not None else
                       pd.DataFrame(np.zeros((len(result_index), 1)) - 1, index=result_index),
            'Position-Shares': (
                {
                    pos_code: pd.DataFrame(
                        position_shares_current[pos_code].values,
                        index=result_index,
                        columns=pos_info['bases'],
                    )
                    for pos_code, pos_info in self.position_structure['flex'].items()
                }
                if position_shares_current is not None else
                {pos_code: None for pos_code in self.position_structure['flex']}
            ),
            'CDFs': [
                pd.DataFrame(cdf_estimates[:, :, i], index=result_index,
                             columns=list(self.x_scores.columns))
                for i in range(self.n_drafters - 1)
            ],
        }

    def get_position_priorities_from_category_weights(self, weights):
        return np.einsum('ij, akj -> ik', weights / self.v.T, self.position_means)

    def get_objective_and_gradient(self, category_weights, position_shares, diff_means,
                                    diff_vars, x_scores_available_array, result_index,
                                    n_players_selected, sigma_2_m, pitching_preference=None):

        if self.position_means is not None:
            position_rewards = self.get_position_priorities_from_category_weights(category_weights)
            rosters, future_position_array, flex_shares = optimize_positions_all_players(
                self.positions.loc[result_index],
                position_rewards,
                self.positions.loc[self.players],
                position_shares,
                self._pos_cfg,
            )
            position_mu = np.einsum('aij,bi -> bj', self.position_means, future_position_array)
            position_mu = np.expand_dims(position_mu, axis=2)
        else:
            position_mu  = 0
            rosters      = None
            flex_shares  = None

        L = self.L

        if self.sport == 'NBA':
            expected_future_diff_single = (
                self.get_x_mu_simplified_form(category_weights, L, self.v) + position_mu
            )
            del_full = (self.n_picks - 1 - n_players_selected) * self.get_del_full(category_weights, L, self.v)

        elif self.sport == 'MLB':
            pitching_share = future_position_array[:, -2:].sum(axis=1).reshape(-1, 1, 1)
            batting_share  = 1 - pitching_share

            batting_diff  = self.get_x_mu_simplified_form(
                category_weights[:, self.batting_stat_indices], self.batting_L, self.batting_v
            )
            pitching_diff = self.get_x_mu_simplified_form(
                category_weights[:, self.pitching_stat_indices], self.pitching_L, self.pitching_v
            )

            convertible_slots = (np.minimum(batting_share, pitching_share)
                                 * (self.n_picks - 1 - n_players_selected)).astype(int)
            total_convertible_value_map = {
                slots: (self.average_round_value[n_players_selected:n_players_selected + slots].sum()
                        + self.average_round_value[-slots:].sum())
                for slots in pd.unique(convertible_slots[:, 0, 0])
            }
            total_convertible_value = (
                np.array([total_convertible_value_map[x] for x in convertible_slots[:, 0, 0]])
                * self.pitching_preference_damper
            )
            values_converted = (
                (total_convertible_value * pitching_preference).reshape(-1, 1, 1)
                * self.pitching_preference_vector.reshape(1, -1, 1)
            )

            expected_future_diff_single = (
                np.concatenate([batting_diff * batting_share, pitching_diff * pitching_share], axis=1)
                + position_mu
                + values_converted / (self.n_picks - 1 - n_players_selected)
            )

            del_batting  = self.get_del_full(category_weights[:, self.batting_stat_indices],
                                              self.batting_L, self.batting_v)
            del_pitching = self.get_del_full(category_weights[:, self.pitching_stat_indices],
                                              self.pitching_L, self.pitching_v)
            del_full = np.zeros((del_batting.shape[0], self.n_categories, self.n_categories))
            del_full[:, :del_batting.shape[1], :del_batting.shape[1]]   = del_batting * batting_share
            del_full[:, del_batting.shape[1]:, del_batting.shape[1]:]   = del_pitching * pitching_share
            del_full *= (self.n_picks - 1 - n_players_selected)

        expected_future_diff = (
            (self.n_picks - 1 - n_players_selected) * expected_future_diff_single
        ).reshape(-1, self.n_categories, 1)

        x_diff_array  = diff_means + x_scores_available_array + expected_future_diff
        pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
        cdf_estimates = self.get_cdf(x_diff_array, diff_vars)

        score, pdf_weights = self.get_objective_and_pdf_weights(
            x_diff_array, diff_vars, cdf_estimates, pdf_estimates, sigma_2_m,
            calculate_pdf_weights=True,
        )

        category_gradient = np.einsum('ai,aik -> ak', pdf_weights, del_full)

        if self.position_means is not None:
            position_gradient = np.einsum('ai,aki -> ak', pdf_weights, self.position_means)
            share_gradients = {
                pos_code: position_gradient[:, self.position_indices[pos_code]] * flex_share.reshape(-1, 1)
                for pos_code, flex_share in flex_shares.items()
            }
            gradients = {'Categories': category_gradient, 'Shares': share_gradients}
        else:
            gradients  = {'Categories': category_gradient}
            flex_shares = None

        if self.sport == 'MLB':
            gradients['Pitcher Preference'] = np.einsum(
                'ai,ik -> a', pdf_weights, self.pitching_preference_vector
            )

        return {
            'Score':         score,
            'Gradients':     gradients,
            'CDF-Estimates': cdf_estimates,
            'Flex-Shares':   flex_shares,
            'Future-Diffs':  expected_future_diff,
            'Rosters':       rosters,
        }

    # ── objective / pdf helpers (unchanged from original) ─────────────────────

    def get_objective_and_pdf_weights(self, x_diff_array, diff_vars, cdf_estimates,
                                       pdf_estimates, sigma_2_m=None,
                                       calculate_pdf_weights=False):
        if self.scoring_format == 'Head to Head: Most Categories':
            return self.get_objective_and_pdf_weights_mc(cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights)
        elif self.scoring_format == 'Rotisserie':
            return self.get_objective_and_pdf_weights_rotisserie(
                x_diff_array, diff_vars, cdf_estimates, pdf_estimates,
                sigma_2_m, calculate_pdf_weights)
        elif self.scoring_format == 'Head to Head: Each Category':
            return self.get_objective_and_pdf_weights_ec(cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights)

    def get_objective_and_pdf_weights_mc(self, cdf_estimates, pdf_estimates,
                                          calculate_pdf_weights=False):
        objective = compute_win_probability(np.array(cdf_estimates)).mean(axis=1)
        if calculate_pdf_weights:
            tipping_points = calculate_tipping_points(np.array(cdf_estimates))
            return objective, (tipping_points * pdf_estimates).mean(axis=2)
        return objective

    def get_objective_and_pdf_weights_ec(self, cdf_estimates, pdf_estimates,
                                          calculate_pdf_weights=False):
        objective = cdf_estimates.mean(axis=2).mean(axis=1)
        if calculate_pdf_weights:
            return objective, pdf_estimates.mean(axis=2)
        return objective

    def get_objective_and_pdf_weights_rotisserie(self, x_diff_array, diff_vars, cdf_estimates,
                                                   pdf_estimates, sigma_2_m,
                                                   calculate_pdf_weights=False, test_mode=False):
        diff_means   = x_diff_array / np.sqrt(diff_vars)
        pdf_estimates = norm.pdf(diff_means)
        f = self.get_f(pdf_estimates)
        g = self.get_g(pdf_estimates)

        h_p      = self.get_h_p(f, g)
        sigma_2_l = self.get_sigma_2_l(sigma_2_m, self.n_drafters)
        sigma_2_p = self.get_sigma_2_p(cdf_estimates, h_p, self.rho)
        mu_l      = self.get_mu_l(sigma_2_m, self.n_drafters)
        mu_p      = self.get_mu_p(cdf_estimates)
        sigma_2_d = self.get_sigma_2_d(sigma_2_p, sigma_2_l, self.n_drafters).reshape(-1, 1, 1)
        mu_d      = self.get_mu_d(mu_p, mu_l, self.n_drafters, self.n_categories).reshape(-1, 1, 1)
        sigma_d   = np.sqrt(sigma_2_d)
        objective = self.get_v(mu_d, sigma_d)

        if calculate_pdf_weights:
            del_sigma_2_p = self.get_del_sigma_2_p(diff_means, self.rho, pdf_estimates, cdf_estimates, f)
            del_mu_d      = self.get_del_mu_d(self.n_drafters, pdf_estimates)
            gradient      = self.get_del_v(sigma_d, del_mu_d, mu_d, del_sigma_2_p)
            gradient      = gradient * np.sqrt(diff_vars)
            if test_mode:
                return gradient
            return objective, gradient.sum(axis=2)
        return objective

    def get_pdf(self, x_diff_array, diff_vars):
        r = x_diff_array.reshape(x_diff_array.shape[0], x_diff_array.shape[1] * x_diff_array.shape[2])
        v = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])
        return norm.pdf(r, scale=np.sqrt(v)).reshape(x_diff_array.shape)

    def get_cdf(self, x_diff_array, diff_vars):
        r = x_diff_array.reshape(x_diff_array.shape[0], x_diff_array.shape[1] * x_diff_array.shape[2])
        v = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])
        return norm.cdf(r, scale=np.sqrt(v)).reshape(x_diff_array.shape)

    def clear_initial_weights(self):
        self.initial_category_weights = None
        self.initial_position_shares  = None
        return self

    # ── simplified-form x_mu helpers (unchanged) ──────────────────────────────

    def get_x_mu_simplified_form(self, c, L, v):
        return np.einsum('aij,ajk -> aik', L, self.get_last_four_terms(c, L, v))

    def get_term_two(self, c, v):
        return (
            -v.reshape(-1, v.shape[0], 1) * c.reshape(-1, 1, v.shape[0])
            + c.reshape(-1, v.shape[0], 1) * v.reshape(-1, 1, v.shape[0])
        )

    def get_del_term_two(self, v):
        s = v.shape[0]
        arr_a = np.zeros((s, s, s))
        for i in range(s):
            arr_a[i, :, i] = v.reshape(s,)
        arr_b = np.zeros((s, s, s))
        for i in range(s):
            arr_b[:, i, i] = v.reshape(s,)
        return (arr_a - arr_b).reshape(1, s, s, s)

    def get_term_four(self, c, v):
        return (c * self.gamma).reshape(-1, v.shape[0], 1) + (v * self.omega).reshape(1, v.shape[0], 1)

    def get_term_five(self, c, L, v):
        return self.get_term_five_a(c, L, v) / self.get_term_five_b(c, L, v)

    def get_term_five_a(self, c, L, v):
        vvL       = np.einsum('ac,pcd -> pad', v.dot(v.T), L)
        factor_top = np.einsum('pad,dp -> ap', vvL, c.T)
        vL        = np.einsum('ac,pcd -> pad', v.T, L)
        vLv       = np.einsum('pad,dc -> ap', vL, v)
        factor    = (factor_top / vLv).T
        c_mod     = c - factor
        cLc       = np.einsum('pd,pd -> p', np.einsum('pc,pcd -> pd', c_mod, L), c_mod)
        return np.sqrt(cLc.reshape(-1, 1, 1))

    def get_term_five_b(self, c, L, v):
        cL  = np.einsum('pc,pcd -> pd', c, L)
        cLc = np.einsum('pd,pd -> p', cL, c)
        vTL = np.einsum('ac,pcd -> pad', v.T, L)
        vTLv = np.einsum('pad,dc -> ap', vTL, v)
        Lct = np.einsum('pcd,dp -> cp', L, c.T)
        vTLc = np.einsum('ac,cp -> ap', v.T, Lct)
        return (cLc * vTLv - vTLc ** 2).reshape(-1, 1, 1)

    def get_terms_four_five(self, c, L, v):
        return self.get_term_four(c, v) * self.get_term_five(c, L, v)

    def get_del_term_four(self, c, v):
        return (np.identity(v.shape[0]) * self.gamma).reshape(1, v.shape[0], v.shape[0])

    def get_del_term_five_a(self, c, L, v):
        vvL       = np.einsum('ac,pcd -> pad', v.dot(v.T), L)
        factor_top = np.einsum('pad,dp -> ap', vvL, c.T)
        vL        = np.einsum('ac,pcd -> pad', v.T, L)
        vLv       = np.einsum('pad,dj -> jp', vL, v)
        factor    = (factor_top / vLv).T
        c_mod     = c - factor
        top_og    = np.einsum('pc,pcd -> pd', c_mod, L)
        top       = top_og.reshape(-1, 1, v.shape[0])
        bottom    = np.sqrt(np.einsum('pd,pd -> p', top_og, c_mod).reshape(-1, 1, 1))
        side      = (np.identity(v.shape[0])
                     - np.einsum('ac,pcd -> pad', v.dot(v.T), L) / vLv.reshape(-1, 1, 1))
        return np.einsum('pia,pad -> pid', top / bottom, side).reshape(-1, 1, v.shape[0])

    def get_del_term_five_b(self, c, L, v):
        cL   = np.einsum('pc,pcd -> pd', c, L)
        vTL  = np.einsum('ac,pcd -> pad', v.T, L)
        vTLv = np.einsum('pad,dj -> paj', vTL, v)
        Lct  = np.einsum('pcd,dp -> cp', L, c.T)
        vTLc = np.einsum('ac,cp -> ap', v.T, Lct)
        t1   = (2 * cL * vTLv.reshape(-1, 1)).reshape(-1, 1, v.shape[0])
        t2   = (2 * vTLc.T).reshape(-1, 1, 1)
        t3   = vTL.reshape(-1, 1, v.shape[0])
        return (t1 - (t2 * t3)).reshape(-1, 1, v.shape[0])

    def get_del_term_five(self, c, L, v):
        a, da = self.get_term_five_a(c, L, v), self.get_del_term_five_a(c, L, v)
        b, db = self.get_term_five_b(c, L, v), self.get_del_term_five_b(c, L, v)
        return (da * b - a * db) / b ** 2

    def get_del_terms_four_five(self, c, L, v):
        return (self.get_term_four(c, v) * self.get_del_term_five(c, L, v)
                + self.get_del_term_four(c, v) * self.get_term_five(c, L, v))

    def get_last_three_terms(self, c, L, v):
        return np.einsum('aij,ajk -> aik', L, self.get_terms_four_five(c, L, v))

    def get_del_last_three_terms(self, c, L, v):
        return np.einsum('aij,ajk -> aik', L, self.get_del_terms_four_five(c, L, v))

    def get_last_four_terms(self, c, L, v):
        return np.einsum('aij,ajk -> aik', self.get_term_two(c, v), self.get_last_three_terms(c, L, v))

    def get_del_last_four_terms(self, c, L, v):
        ci   = self.get_del_term_two(v)
        cii  = self.get_last_three_terms(c, L, v)
        ta   = np.einsum('aijk,aj -> aik', ci, cii.reshape(-1, v.shape[0]))
        tb   = np.einsum('aij,ajk -> aik', self.get_term_two(c, v), self.get_del_last_three_terms(c, L, v))
        return ta + tb

    def get_del_full(self, c, L, v):
        return np.einsum('aij,ajk -> aik', L, self.get_del_last_four_terms(c, L, v))

    # ── Rotisserie helpers (unchanged) ─────────────────────────────────────────

    def get_f(self, pdfs):
        return pdfs.sum(axis=2)

    def get_g(self, pdfs):
        return np.einsum('pao,pbo -> pab', pdfs, pdfs)

    def get_h_p(self, f, g):
        g1 = g.copy()
        g1[:, np.arange(g.shape[1]), np.arange(g.shape[2])] = 0
        g2     = g * np.expand_dims(np.identity(self.n_categories), 0)
        f_part = np.einsum('pa,pb -> pab', f, f)
        return f_part + g1 - g2

    def get_h_m(self, sigma_c, n_managers):
        s_mod = sigma_c ** 2 + 1
        sigma_matrix = np.sqrt(np.einsum('a,b -> ab', s_mod, s_mod))
        first = n_managers / sigma_matrix - (2 / sigma_matrix) * np.identity(len(sigma_c))
        return (n_managers - 1) / (2 * np.pi) * first

    def get_v(self, mu_d, sigma_d):
        return norm.cdf(mu_d / sigma_d).reshape(-1)

    def get_mu_d(self, mu_p, mu_l, n_managers, n_categories):
        return mu_p * n_managers / (n_managers - 1) - n_categories * n_managers / 2 - mu_l

    def get_sigma_2_d(self, sigma_2_p, sigma_2_l, n_managers):
        return sigma_2_p * n_managers / (n_managers - 1) + sigma_2_l

    def get_mu_p(self, cdfs):
        return cdfs.sum(axis=(1, 2))

    def get_mu_l(self, sigma_2_m, n_managers):
        return self.max_ev * np.sqrt(sigma_2_m)

    def get_sigma_2_p(self, cdfs, h_p, rho):
        return (cdfs * (1 - cdfs)).sum(axis=(1, 2)) + (rho * h_p).sum(axis=(1, 2)) / 2

    def get_sigma_2_l(self, sigma_2_m, n_managers):
        return sigma_2_m * self.max_var

    def get_sigma_2_m(self, sigma_c, h_m, rho, n_managers):
        s2 = sigma_c ** 2
        c1 = (n_managers - 1) * np.arccos(s2 / (1 + s2)).sum() / (2 * np.pi)
        c2 = (rho * h_m).sum(axis=(1, 2)) / 2
        return c1 + c2

    def get_del_sigma_2_p(self, opponent_mu_matrix, rho, pdfs, cdfs, f):
        rho2 = rho.copy()
        rho2[:, np.arange(rho2.shape[1]), np.arange(rho2.shape[2])] = 0
        inside  = -pdfs - f.reshape(-1, f.shape[1], 1)
        fp      = np.einsum('pab,pbc -> pac', rho2, inside)
        fc1     = (opponent_mu_matrix * pdfs) * (fp + (pdfs - f.reshape(-1, f.shape[1], 1)))
        fc2     = pdfs * (1 - 2 * cdfs)
        return fc1 + fc2

    def get_del_mu_d(self, n_managers, pdfs):
        return n_managers / (n_managers - 1) * pdfs

    def get_del_v(self, sigma_d, del_mu_d, mu_d, del_sigma_2_p):
        return (norm.pdf(mu_d / sigma_d) / (sigma_d ** 3)
                * (sigma_d ** 2 * del_mu_d - mu_d * del_sigma_2_p / 2))


# ── Adam optimiser (unchanged) ─────────────────────────────────────────────────

class AdamOptimizer:
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.m = self.v_adam = None
        self.t = 0

    def minimize(self, gradient):
        if self.m is None:
            self.m = self.v_adam = 0
        self.t += 1
        self.m      = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v_adam = self.beta2 * self.v_adam + (1 - self.beta2) * (gradient ** 2)
        m_hat = self.m      / (1 - self.beta1 ** self.t)
        v_hat = self.v_adam / (1 - self.beta2 ** self.t)
        return self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ── plain factory function (no @st.cache_resource) ────────────────────────────

def build_h_agent(info, omega, gamma, n_starters, n_drafters, beth,
                  scoring_format, dynamic, sport, params, slot_counts, aleph=0.0):
    return HAgent(
        info           = info,
        omega          = omega,
        gamma          = gamma,
        n_picks        = n_starters,
        n_drafters     = n_drafters,
        dynamic        = dynamic,
        scoring_format = scoring_format,
        sport          = sport,
        params         = params,
        slot_counts    = slot_counts,
        aleph          = aleph,
        beth           = beth,
    )
