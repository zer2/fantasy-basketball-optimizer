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

import os

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

# Direct standard-normal pdf / cdf / ppf. Equivalent to scipy.stats.norm.{pdf,cdf,ppf} (loc is
# always 0 here; scale defaults to 1) but far cheaper — scipy.stats.norm.* routes every call
# through the generic continuous-distribution machinery (argsreduce, _support_mask, place, ...),
# which dominated get_pdf/get_cdf in the gradient loop.
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _normal_pdf(x, scale=1.0):
    z = x / scale
    return _INV_SQRT_2PI / scale * np.exp(-0.5 * z * z)


def _normal_cdf(x, scale=1.0):
    return ndtr(x / scale)


def _normal_ppf(q, scale=1.0):
    return scale * ndtri(q)


def _softmax_rows(logits):
    """Row-wise softmax (each row is a probability distribution over that row's columns). Used to
    parameterise the flex position shares: optimising in this unconstrained logit space keeps the
    shares on the simplex automatically, so the share update needs no clip-to-[0,1] or renormalise
    (those distort the step at the boundary and homogenise the survivors).

    Subtracting the row max is purely numerical and has no effect on the result: softmax is
    shift-invariant, so e^{z − c}/Σe^{z − c} = e^z/Σe^z — the common e^{−c} factor cancels between
    numerator and denominator. Its only job is to keep exp() from overflowing on large logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


# Learning rate for the softmax-logit flex-share optimiser. Larger than the old share-space 0.01
# because a logit must move ~±4 to swing a share across [0, 1]; tuned so shares settle within the
# gradient-iteration budget rather than still drifting at the final iteration.
_SHARES_LEARNING_RATE = 0.3

# Punt-seeding for the cold-start category weights (replaces the old heuristic init). Each seed gently
# down-weights one category to this fraction of neutral; a gentle seed starts near the 30-iteration
# optimum so the short descent can reach it, whereas an aggressive punt starts too far and loses.
_PUNT_SEED_FACTOR = 0.9
# From this many of the drafter's own picks on, the roster has a real shape, so seed a single punt of
# its weakest category. Earlier than that there is no reliable weakness, so multi-start over every punt.
_WEAKNESS_SEED_MIN_ROSTER = 3

# Per-format best-practice configuration, resolved per agent in __init__:
#   Rotisserie:  lowvar seed (tilts weight off the noisy categories), regulariser on, unified stability v
#   H2H (EC/MC): multi-start punt seeding, regulariser on (5x schedule)
# The env vars below override individual fields for A/B testing only; when unset (the production path)
# each format uses its per-format default. Leave them unset in production.
_SEED_MODE_OVERRIDE      = os.environ.get('SEED_MODE')        # multistart | heuristic | neutral | lowvar
_PUNT_REG_OVERRIDE       = os.environ.get('PUNT_REG')         # '0' | '1'
_ROTO_V_UNIFIED_OVERRIDE = os.environ.get('ROTO_V_UNIFIED')   # '0' | '1'
# lowvar seed (Roto): exponent on the seed tilt that down-weights high-v (less stable) categories.
_LOWVAR_TILT = float(os.environ.get('LOWVAR_TILT', '0.5'))
# Robustness regulariser: each iteration soft-thresholds the category weights toward neutral v -- the
# proximal step for an L1 penalty -lambda*||w-v||_1. L1 rewards balance without over-penalising a
# committed punt the way L2 would, and its sparsity pins uncontested categories at v while letting a few
# deviate. The per-pick strength lambda follows a Gaussian (phi) schedule (built per agent from n_picks
# in __init__): it starts at the peak on an empty roster and decays to ~0 by the final pick.
# REG_PEAK tunes the peak; PUNT_REG_SCHEDULE (comma-separated floats) overrides the whole schedule.
# Regularisation strength: the peak of the decay schedule (the lambda on an empty roster). Hardcoded --
# not user-configurable -- since testing settled on this value; REG_PEAK overrides it for sweeps.
_REG_STRENGTH = 0.00005
_REG_PEAK_OVERRIDE = os.environ.get('REG_PEAK')  # None unless set; overrides the hardcoded _REG_STRENGTH
# Optional guard (default 0 = off, clean L1 that may snap onto v): keep weights at least this far per
# component from the singular w==v ray. Only needed if REG_PEAK is raised past ~5e-4, where the small
# empty-board deviations let the shrink reach v and term_five goes singular (EC in particular).
_REG_FLOOR = float(os.environ.get('REG_FLOOR', '0.0'))
# Position/flex-share reg strength as a multiple of the category reg: shares live on a coarser simplex
# (few bases), so they need a firmer pull toward uniform to have a comparable effect.
_POSITION_REG_MULT = float(os.environ.get('POSITION_REG_MULT', '1000'))
# Anti-crowded-punt coupling ("kappa"): a linear penalty in the objective on punting categories the
# FIELD is crowding into. On the empty-board first run, the multi-start seed scan reveals which single
# punts the top _PUNT_POPULARITY_TOP_N players most want (the popularity vector); the penalty then
# discourages joining those popular punts, dispersing the crowd (crowded punts are competed away). It
# acts only in the early multi-start rounds (roster < _WEAKNESS_SEED_MIN_ROSTER). 0 disables it (no
# behaviour change); higher = firmer defection from the crowd. Surfaced as the per-session `kappa`
# parameter (default 0.5); the env KAPPA overrides that per-session value for sweeps.
_KAPPA_OVERRIDE = os.environ.get('KAPPA')   # None unless set
_PUNT_POPULARITY_TOP_N = int(os.environ.get('PUNT_POPULARITY_TOP_N', '40'))
# Gaussian (phi) reg-decay shape: lambda_k = peak*(phi(B k/n) - phi(B))/(phi(0)-phi(B)) -- peak on an
# empty roster, decaying to exactly 0 at the final pick. B sets the concave shoulder (~ first n/B picks)
# before the convex tail; B=4 puts the shoulder near pick 3 and matches the old cosine's total budget.
_REG_SHAPE_B = float(os.environ.get('REG_SHAPE_B', '4'))
_schedule_override = os.environ.get('PUNT_REG_SCHEDULE')
# Correlation-correction refresh interval, mirroring the position-optimiser throttle: the
# correction terms are recomputed on iterations where (iteration+1) % interval == 0 (plus
# the cold start) and reused between — they drift slowly with the category weights, so a
# small staleness buys back most of the correction's per-iteration cost. One-shot scoring
# calls (full team, trades, auction values) always compute fresh.
_MC_CORRELATION_REFRESH_INTERVAL = int(os.environ.get('MC_CORRELATION_REFRESH', '4'))
from itertools import combinations

from backend.math.algorithm_helpers import (
    compute_win_probability,
    calculate_tipping_points,
    calculate_correction_terms,
)
from backend.math.process_player_data import get_category_level_rv
from backend.math.position_optimization import (
    optimize_positions_all_players,
    get_player_rows,
)
from backend.math.position_config import PositionConfig, build_position_config


class HAgent:

    # Position-optimiser throttle. _exact_solve_window is how many top candidates (by global rank)
    # get an exact roster solve every iteration; _exact_solve_refresh_interval is how often (in
    # gradient iterations) that set is re-ranked from the latest H-scores. Re-ranking keeps the exact
    # solves pointed at the players currently projecting into the global top window — which shift with
    # the team's context (e.g. a committed punt) — rather than the static pre-draft ranking. Because
    # the set tracks the contextually-strongest players, the window can be smaller than a static one.
    # Refresh interval 0 disables re-ranking (the set stays on its initial, pre-draft order).
    _exact_solve_window           = 30
    _exact_solve_refresh_interval = 10

    def __init__(self
                 , info: dict
                 , omega: float
                 , gamma: float
                 , n_picks: int
                 , n_drafters: int
                 , dynamic: bool
                 , scoring_format: str
                 # ── explicit context (replaces get_*() calls) ──
                 , sport: str
                 , params: dict
                 , slot_counts: dict
                 , aleph: float = 0.0
                 , kappa: float = 0.3
                 # ── original optional args ──
                 , beth: float = 0
                 , collect_info: bool = False
                 ):

        self.omega         = omega
        self.gamma         = gamma
        self.n_picks       = n_picks
        self.dynamic       = dynamic
        self.n_drafters    = n_drafters
        self.collect_info  = collect_info
        self.scoring_format = scoring_format

        # Per-format config (the env vars above override any field for A/B testing). All formats run the
        # robustness regulariser; they differ only in the cold-start seed -- Rotisserie uses the lowvar
        # tilt (its punts are structural, not a strategic fork), the H2H formats use multi-start punt
        # seeding so early picks avoid over-committing to a punt they may drop.
        is_rotisserie       = scoring_format == 'Rotisserie'
        self.seed_mode      = _SEED_MODE_OVERRIDE or ('lowvar' if is_rotisserie else 'multistart')
        self.regulariser_on = (_PUNT_REG_OVERRIDE == '1') if _PUNT_REG_OVERRIDE is not None else True
        roto_v_unified      = (_ROTO_V_UNIFIED_OVERRIDE == '1') if _ROTO_V_UNIFIED_OVERRIDE is not None else True

        # Gaussian (phi) regulariser schedule built from the draft length: strength _REG_STRENGTH (the
        # hardcoded peak) on an empty roster, decaying to ~0 by the final pick (indexed by roster size),
        # with a concave shoulder set by _REG_SHAPE_B. REG_PEAK overrides the peak; PUNT_REG_SCHEDULE the
        # whole schedule.
        reg_peak = float(_REG_PEAK_OVERRIDE) if _REG_PEAK_OVERRIDE is not None else _REG_STRENGTH
        _phi0    = 1.0 - np.exp(-_REG_SHAPE_B ** 2 / 2)
        self.reg_schedule = ([float(x) for x in _schedule_override.split(',')] if _schedule_override
                             else [reg_peak * (np.exp(-(_REG_SHAPE_B * k / n_picks) ** 2 / 2)
                                               - np.exp(-_REG_SHAPE_B ** 2 / 2)) / _phi0
                                   for k in range(n_picks)])

        # ── store explicit context ─────────────────────────────────────────────
        self.sport  = sport
        self.params = params

        # Retain the processed player data so callers read G-scores / positions off the agent
        # (it is explanation-oriented — see _build_candidates). Consumers use agent.info directly.
        self.info = info

        # Neutral empty-board baseline: ranks players so the position-optimiser throttle prioritises
        # the ones most likely to be picked, and anchors auction dollar values. Populated by
        # populate_default_h_scores at the end of the build. None => "not built yet", which makes
        # get_h_scores run a full exact solve (no throttle).
        self.default_h_scores = None   # sorted pd.Series
        self._default_result  = None   # full empty-board result dict (for the empty-board short-circuit)

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

            # A player's positional baseline is the average of the position means over ALL of their
            # eligible positions (e.g. a PF/C uses the mean of the PF and C means, not just PF). reindex
            # then mean(axis=0) skips any listed position absent from position_means_df.
            rel_players = [p for p in x_scores.index if p != 'RP']
            self.pos_avg = pd.DataFrame(
                [position_means_df.reindex(self.positions.get(p)).mean(axis=0)
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

        # ── differential correlation matrix (replaces get_correlations()) ──────
        # Used by Rotisserie's variance model and by the Most-Categories correlation
        # correction. Sign-flipped for negative statistics so it matches the "good
        # direction" orientation of the differential z-scores.
        if scoring_format in ('Rotisserie', 'Head to Head: Most Categories'):
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
            rho.loc[negative_stats, negative_stats] = 1

            correlation_categories = list(x_scores.columns)
            self.rho = np.expand_dims(
                np.array(rho.loc[correlation_categories, correlation_categories]), 0
            )
        else:
            self.rho = None

        # Most-Categories correlation correction (see docs: correlation-correction note, eq (C4)).
        # On by default for Most-Categories; set MC_CORRELATION=0 to disable — to regenerate the
        # pre-correction goldens, or to A/B the correction's effect on rankings.
        self.mc_correlation_enabled = (
            scoring_format == 'Head to Head: Most Categories'
            and os.environ.get('MC_CORRELATION', '1') == '1'
        )
        # Per-descent cache of correction terms (see get_objective_and_pdf_weights_mc);
        # reset at the start of every perform_iterations run.
        self._correction_cache = None

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[
                info['G-scores'].sum(axis=1).sort_values(ascending=False).index
            ]
            v = np.sqrt(mov / (mov + vom)) if roto_v_unified else np.sqrt(mov / vom)

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
        # Anti-crowded-punt coupling (session parameter; env KAPPA overrides for sweeps).
        self.kappa = float(_KAPPA_OVERRIDE) if _KAPPA_OVERRIDE is not None else kappa
        # Field punt-popularity vector (per category), measured once on the empty-board multi-start scan
        # and reused for the early picks; drives the anti-crowded-punt (kappa) objective penalty. None
        # until measured (and whenever kappa=0), which makes the penalty inert.
        self._punt_popularity = None

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

    def get_h_scores(self
                     , player_assignments: dict
                     , drafter
                     , n_iterations: int
                     , cash_remaining_per_team: dict = None
                     , exclusion_list: list = []
                     , candidate_subset: list = None
                     , candidate_offset: int = 0) -> dict:

        # Empty-board short-circuit: the neutral baseline is exactly an all-empty, no-exclusions,
        # full-pool run, which populate_default_h_scores already computed at build. Reuse it rather
        # than re-solving (this is the draft-start evaluate). Guarded so a filtered empty-board call
        # (exclusions or a candidate subset) still computes.
        if (self._default_result is not None
                and not exclusion_list
                and candidate_subset is None
                and all(len(roster) == 0 for roster in player_assignments.values())):
            return self._default_result

        self.n_drafters = len(player_assignments)
        my_players = [p for p in player_assignments[drafter] if p == p]
        self.players = my_players

        # Position-optimiser throttle schedule for this run. Draft uses the 'tiered' schedule; auction
        # defaults to 'exact' (dollar values anchor on the whole-distribution replacement level, so
        # they're sensitive to the approximated tail). _position_mode_override lets benchmarks/tests
        # force a specific schedule: 'exact' | 'tiered' | 'light'.
        self._position_mode = getattr(self, '_position_mode_override', None) or (
            'light' if cash_remaining_per_team is not None else 'tiered'
        )

        n_players_selected = len(my_players)
        players_chosen     = [x for v in player_assignments.values() for x in v if x == x]

        available_mask = (
            ~self.x_scores.index.isin(players_chosen + exclusion_list)
            & self.x_scores.index.isin(self.positions.index)
        )
        # x_scores_available is the FULL available pool, exactly as it has always meant. The opponent /
        # future-pick model (get_diff_distributions) reads its top players, so it must stay complete.
        x_scores_available = self.x_scores[available_mask]

        # x_scores_batch is the slice we actually score. candidate_subset (draft/waiver batching) narrows
        # it to one batch; without it we score the whole pool. The future-pick model below still sees the
        # full pool either way, so each batch's H-scores match a single full evaluation.
        if candidate_subset is not None:
            x_scores_batch = x_scores_available[x_scores_available.index.isin(candidate_subset)]
        else:
            x_scores_batch = x_scores_available

        # diff_means/diff_vars/sigma_2_m are candidate-independent (shape (1, n_cat, n_drafters-1)),
        # so they broadcast against whatever slice of candidates is being scored.
        diff_means, diff_vars, sigma_2_m = self.get_diff_distributions(
            player_assignments, drafter, x_scores_available, cash_remaining_per_team
        )

        x_scores_batch_array = np.expand_dims(np.array(x_scores_batch), axis=2)

        if len(my_players) > 0:
            cdf_original = _normal_cdf(
                (diff_means + x_scores_batch_array).mean(axis=2),
                scale=np.sqrt(diff_vars.mean(axis=2)),
            )
            cdf_mod = np.einsum(
                'ij,ai -> aj',
                self.transformation_matrix_inverted,
                cdf_original + self.transformation_addition_constant,
            )
            corrected_strength = _normal_ppf(cdf_mod, scale=np.sqrt(diff_vars.mean(axis=2)))
            x_scores_batch_mod    = corrected_strength - diff_means.mean(axis=2)
            x_scores_batch_array  = np.expand_dims(x_scores_batch_mod, axis=2)

        if self.initial_category_weights is None:
            # Cold start: uniform flex shares. The category-weight init is normally left to punt-seeding
            # in perform_iterations (None signals it). SEED_MODE=heuristic restores the old per-candidate
            # heuristic init instead, for A/B testing the seeding against the baseline.
            if self.seed_mode == 'heuristic':
                default_weights = self.v.T.reshape(1, self.n_categories, 1)
                category_momentum_factor = 10000 if self.scoring_format == 'Rotisserie' else 1000
                if self.pos_avg is not None:
                    pos_avg_array = np.expand_dims(np.array(self.pos_avg.loc[x_scores_batch.index]), axis=2)
                else:
                    pos_avg_array = 0
                initial_category_weights = (
                    (diff_means + x_scores_batch_array - pos_avg_array)
                    / (default_weights * category_momentum_factor)
                    + default_weights
                ).mean(axis=2)
                initial_category_weights /= initial_category_weights.sum(axis=1).reshape(-1, 1)
            elif self.seed_mode == 'neutral':
                # Start at neutral v (no punt bias) and let the descent re-balance. Nudge off the exact
                # singular ray -- term_five_b (a Cauchy-Schwarz Gram determinant) vanishes when w is
                # parallel to v, giving a 0/0 -- so a negligible jitter keeps it finite for every format.
                neutral = self.v.reshape(self.n_categories)
                jitter  = 1.0 + 1e-9 * np.where(np.arange(self.n_categories) % 2 == 0, 1.0, -1.0)
                initial_category_weights = np.array(
                    [(neutral * jitter) / (neutral * jitter).sum()] * len(x_scores_batch)
                )
            elif self.seed_mode == 'lowvar':
                # Roto: tilt the initial seed to down-weight the high-v (less stable) categories, since
                # a high v means more within-period noise relative to signal, so the category is less
                # reliably winnable and worth less as an initial lean. The descent then re-balances. The
                # tilt also lands a finite distance off the singular w||v ray, dodging the near-v blow-up.
                neutral = self.v.reshape(self.n_categories)
                tilt    = (neutral / neutral.mean()) ** (-_LOWVAR_TILT)
                seed    = neutral * tilt
                initial_category_weights = np.array([seed / seed.sum()] * len(x_scores_batch))
            else:
                initial_category_weights = None
            initial_position_shares = {
                pos_code: pd.DataFrame({
                    base: [1 / len(pos_info['bases'])] * len(x_scores_batch_array)
                    for base in pos_info['bases']
                })
                for pos_code, pos_info in self.position_structure['flex'].items()
            }
        else:
            initial_category_weights = np.array(
                [self.initial_category_weights] * len(x_scores_batch)
            )
            initial_position_shares = {
                pos_code: pd.DataFrame({
                    base: [self.initial_shares[pos_code][base]] * len(x_scores_batch_array)
                    for base in pos_info['bases']
                })
                for pos_code, pos_info in self.position_structure['flex'].items()
            }

        # Position-optimiser throttle priority: rank candidates by the cached default (first-pick)
        # H-scores so the exact-solve tier covers the players most likely to actually be picked.
        # Missing/uncached players sort last; with no cached ranking at all we pass None, which
        # disables throttling entirely (a full, exact solve every iteration).
        if self.default_h_scores is not None:
            ranked = self.default_h_scores.reindex(x_scores_batch.index).to_numpy()
            candidate_priority = np.argsort(-np.nan_to_num(ranked, nan=-np.inf))
        else:
            candidate_priority = None

        return self.perform_iterations(
            initial_category_weights,
            initial_position_shares,
            my_players,
            n_players_selected,
            diff_means,
            diff_vars,
            x_scores_batch_array,
            x_scores_batch.index,
            sigma_2_m,
            n_iterations,
            candidate_priority,
            candidate_offset,
        )

    def get_diff_distributions(self
                               , player_assignments
                               , drafter
                               , x_scores_available
                               , cash_remaining_per_team=None):
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
            # While the roster still has room a candidate is being added, so the drafter's team grows to
            # len+1 and opponents are modelled at that size (shortfalls padded with the expected next-pick
            # quality). Once the roster is full there is no candidate and no future pick, so opponents are
            # compared at their actual size — without this, scoring a complete team pads every opponent
            # with a phantom below-replacement player, inflating the team's H-score.
            adding_candidate = len(my_players) < self.n_picks
            target_team_size = len(my_players) + (1 if adding_candidate else 0)
            extra_needed     = target_team_size * self.n_drafters - len(players_chosen)
            mean_extra       = x_scores_available.iloc[:extra_needed].mean().fillna(0)
            other_team_sums  = np.vstack([
                self.get_opposing_team_means(player_assignments[team], mean_extra, target_team_size)
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

    def compute_h_score_from_diff_means(self
                                         , diff_means: np.ndarray
                                         , diff_vars: np.ndarray) -> float:
        """Fast H-score for a complete (n_picks) roster given pre-computed diff_means.

        Mirrors the n_players_selected == n_picks branch of perform_iterations:
        no available-player array needed, just score diff_means directly.

        diff_means : (1, n_categories, n_drafters-1)
        diff_vars  : (1, n_categories, n_drafters-1)
        """
        if self.scoring_format == 'Rotisserie':
            sigma_c   = (diff_means / np.sqrt(diff_vars))[0, :, :].std(axis=1, ddof=1) * np.sqrt(2)
            h_m       = self.get_h_m(sigma_c, self.n_drafters)
            sigma_2_m = self.get_sigma_2_m(sigma_c, h_m, self.rho, self.n_drafters)
        else:
            sigma_2_m = None

        cdf_estimates = self.get_cdf(diff_means, diff_vars)
        score = self.get_objective_and_pdf_weights(
            diff_means, diff_vars, cdf_estimates, None, sigma_2_m,
            calculate_pdf_weights=False,
        )
        return float(np.max(score))

    def compute_h_scores_batched(self
                                  , diff_means_batch: np.ndarray
                                  , diff_vars: np.ndarray) -> np.ndarray:
        """Vectorized H-score for a batch of complete-roster diff_means.

        Equivalent to calling compute_h_score_from_diff_means N times but in a
        single pass through get_cdf / get_objective_and_pdf_weights, which are
        already vectorized over the first (batch) dimension.

        diff_means_batch : (N, n_categories, n_drafters-1)
        diff_vars        : (1, n_categories, n_drafters-1)  [broadcasts over N]
        Returns          : (N,) scores
        """
        if self.scoring_format == 'Rotisserie':
            z         = diff_means_batch / np.sqrt(diff_vars)   # (N, n_cats, n_drafters-1)
            sigma_c   = z.std(axis=2, ddof=1) * np.sqrt(2)      # (N, n_cats)
            h_m       = self.get_h_m(sigma_c, self.n_drafters)
            sigma_2_m = self.get_sigma_2_m(sigma_c, h_m, self.rho, self.n_drafters)
        else:
            sigma_2_m = None

        cdf_estimates = self.get_cdf(diff_means_batch, diff_vars)
        return self.get_objective_and_pdf_weights(
            diff_means_batch, diff_vars, cdf_estimates, None, sigma_2_m,
            calculate_pdf_weights=False,
        )

    def get_opposing_team_means(self, players, mean_extra_players, target_team_size):
        n_extra = max(target_team_size - len(players), 0)
        player_sum  = np.array(self.x_scores.loc[[p for p in players if p == p]].sum(axis=0))
        extra_sum   = np.array(mean_extra_players) * n_extra
        return (player_sum + extra_sum).reshape(1, self.n_categories, 1)

    def get_diff_means_auction(self
                                , score_diff
                                , money_diff
                                , player_diff
                                , category_value_per_dollar
                                , replacement_value_by_category):
        player_diff_total = ((player_diff - 1) * replacement_value_by_category).reshape(1, self.n_categories, 1)
        money_diff_total  = (money_diff * category_value_per_dollar).reshape(1, self.n_categories, 1)
        return score_diff - player_diff_total + money_diff_total

    def get_diff_var(self, n_their_players):
        return self.n_picks * (2 + self.w * (self.n_picks - n_their_players) / self.n_picks)

    def get_value_of_money_auction(self
                                    , diff_means
                                    , diff_vars
                                    , sigma_2_m
                                    , category_value_per_dollar
                                    , replacement_value_by_category
                                    , max_money=200
                                    , step_size=0.1):
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

    def _select_starting_weights(self
                                 , position_shares
                                 , diff_means
                                 , diff_vars
                                 , x_scores_batch_array
                                 , candidate_player_array
                                 , team_so_far_array
                                 , n_players_selected
                                 , sigma_2_m
                                 , pitching_preference):
        """Choose the cold-start category-weight init per candidate (replaces the old heuristic).

        Each seed gently punts one category to _PUNT_SEED_FACTOR of neutral. Once the drafter holds
        _WEAKNESS_SEED_MIN_ROSTER+ players the roster has a real shape, so seed the single punt of its
        weakest category (same for every candidate). Earlier -- no reliable weakness -- multi-start:
        score one punt per category and keep, per candidate, the one with the best pre-descent objective.
        """
        neutral      = self.v.reshape(self.n_categories)
        n_candidates = x_scores_batch_array.shape[0]

        def gentle_punt(category_index):
            weights = neutral.copy()
            weights[category_index] *= _PUNT_SEED_FACTOR
            return weights / weights.sum()

        if len(self.players) >= _WEAKNESS_SEED_MIN_ROSTER:
            weakest = int(np.argmin(self.x_scores.loc[self.players].sum(axis=0).to_numpy()))
            return np.array([gentle_punt(weakest)] * n_candidates)

        # Multi-start: score every gentle punt with a clean full solve (throttle off, so the seeds do
        # not share cached rosters) and keep each candidate's best-scoring seed.
        seed_vectors = [gentle_punt(i) for i in range(self.n_categories)]

        def score_seeds():
            saved_priority = self._candidate_priority
            self._candidate_priority     = None
            self._position_rosters_cache = None
            scores = np.vstack([
                self.get_objective_and_gradient(
                    np.array([seed] * n_candidates), position_shares, diff_means, diff_vars,
                    x_scores_batch_array, candidate_player_array, team_so_far_array,
                    n_players_selected, sigma_2_m, 0, pitching_preference,
                    correction_mode='skip',
                )['Score']
                for seed in seed_vectors
            ])
            self._candidate_priority     = saved_priority
            self._position_rosters_cache = None
            return scores

        # Empty-board first run with kappa on: two passes. Pass 1 (penalty inert, popularity=None) reveals
        # which single punt the top _PUNT_POPULARITY_TOP_N players each crowd into; store that popularity
        # so pass 2's seed scan -- and the descent that follows -- defect from the crowd (the penalty lives
        # in get_objective_and_gradient). Later picks reuse the empty-board popularity, so no re-measure.
        if self.kappa > 0.0 and len(self.players) == 0:
            self._punt_popularity = None
            raw_scores = score_seeds()
            best_raw   = np.argmax(raw_scores, axis=0)
            top        = np.argsort(-raw_scores.max(axis=0))[:_PUNT_POPULARITY_TOP_N]
            counts     = np.bincount(best_raw[top], minlength=self.n_categories).astype(float)
            self._punt_popularity = counts / max(len(top), 1)

        seed_scores = score_seeds()
        best = np.argmax(seed_scores, axis=0)
        return np.array([seed_vectors[b] for b in best])

    def perform_iterations(self
                           , category_weights
                           , position_shares
                           , my_players
                           , n_players_selected
                           , diff_means
                           , diff_vars
                           , x_scores_batch_array
                           , result_index
                           , sigma_2_m
                           , n_iterations
                           , candidate_priority=None
                           , candidate_offset=0):

        # Stale correction terms must never leak across boards or candidate batches.
        self._correction_cache = None

        optimizers = {
            'Categories': AdamOptimizer(learning_rate=0.001),
            # Shares are optimised in softmax-logit space (see the update below). A logit must travel
            # ~±4 to move a share across most of [0, 1], versus a direct step in share space, so the
            # logit learning rate is an order of magnitude larger than the old share-space 0.01.
            # One optimiser for the shared per-position logits (every flex slot's softmax reads them).
            'Shares': AdamOptimizer(learning_rate=_SHARES_LEARNING_RATE),
        }

        if self.sport == 'MLB':
            optimizers['Pitcher Preference'] = AdamOptimizer(learning_rate=0.05)
            pitching_preference = 0
        else:
            pitching_preference = None

        category_weights_current  = category_weights
        position_shares_current   = position_shares

        # All flex slots share one logit per position (the position_means columns): each slot's shares are
        # the softmax over its own subset of positions, so e.g. the guard slot's PG:SG ratio is exactly the
        # utility slot's. We optimise these shared per-position logits directly. Seed them from the slot that
        # spans every position (Util), then derive each slot's shares so they start mutually consistent.
        # (softmax is shift-invariant, so log(shares) is a valid pre-image; clip so a zero share is finite.)
        if self.position_means is not None:
            n_positions   = self.position_means.shape[1]
            n_cand        = len(next(iter(position_shares.values())))
            # Seed from the slot spanning every position (Util) so the shared logits carry a consistent
            # warm start; if no single slot spans all positions, start uniform (still correct, just no warm
            # start on the shared vector).
            master_pc     = next((pc for pc in self.position_structure['flex']
                                  if len(self.position_indices[pc]) == n_positions), None)
            master_logits = (np.log(np.clip(position_shares[master_pc].values, 1e-12, None))
                             if master_pc is not None else np.zeros((n_cand, n_positions)))
            for pos_code in self.position_structure['flex']:
                position_shares[pos_code].values[:] = _softmax_rows(
                    master_logits[:, self.position_indices[pos_code]])

        if (n_players_selected < self.n_picks - 1) and self.dynamic:

            # Eligibility rows depend only on which players are eligible for which slots — not on the
            # category weights — so build them once here instead of on every gradient iteration.
            if self.position_means is not None:
                n_total_picks          = sum(self._pos_cfg.position_numbers.values())
                candidate_player_array = get_player_rows(self.positions.loc[result_index], self._pos_cfg)
                team_so_far_array      = (get_player_rows(self.positions.loc[self.players], self._pos_cfg)
                                          if len(self.players) > 0
                                          else np.empty((0, n_total_picks)))
            else:
                candidate_player_array = None
                team_so_far_array      = None

            # Throttled position optimisation reuses the previous iteration's roster assignment for
            # lower-ranked candidates; this cache starts empty each perform_iterations call.
            # candidate_priority ranks candidates for the throttle (None disables it → full solves).
            self._position_rosters_cache = None
            self._candidate_priority     = candidate_priority
            # Global rank of this batch's first candidate. The exact-solve tiers below are global ranks,
            # so a batch that starts past them (offset >= 70) exact-solves nobody: the "top 30 of the
            # 5th batch" are deep-bench players, not worth a Hungarian solve every iteration.
            self._candidate_offset       = candidate_offset

            # Cold start (get_h_scores passes None): choose the starting category weights by punt-seeding
            # instead of the old heuristic. Done here, where the objective machinery is already set up.
            if category_weights is None:
                category_weights = self._select_starting_weights(
                    position_shares, diff_means, diff_vars, x_scores_batch_array,
                    candidate_player_array, team_so_far_array, n_players_selected,
                    sigma_2_m, pitching_preference,
                )

            # Robustness regulariser: soft-threshold the category weights toward neutral v each iteration
            # (proximal step for an L1 penalty -lambda*||w-v||_1), strongest on an empty roster and
            # decaying to 0 by mid-draft (see self.reg_schedule), so early picks stay flexible.
            reg_lambda  = (self.reg_schedule[len(self.players)]
                           if (self.regulariser_on and len(self.players) < len(self.reg_schedule)) else 0.0)
            neutral_row = self.v.reshape(1, self.n_categories)

            for iteration in range(max(1, n_iterations)):
                category_weights_current  = category_weights
                position_shares_current   = position_shares

                gradient_result = self.get_objective_and_gradient(
                    category_weights, position_shares, diff_means, diff_vars,
                    x_scores_batch_array, candidate_player_array, team_so_far_array,
                    n_players_selected, sigma_2_m, iteration, pitching_preference,
                )

                score               = gradient_result['Score']
                gradients           = gradient_result['Gradients']
                cdf_estimates       = gradient_result['CDF-Estimates']
                expected_future_diff = gradient_result['Future-Diffs']
                rosters             = gradient_result['Rosters']

                # Re-rank the exact-solve window from the latest H-scores every N iterations, so it
                # tracks the players currently projecting into the global top window (e.g. those that
                # rise under a committed punt) rather than the static pre-draft ranking. Updating here
                # (after the score is known) applies to the next iteration's position solve. Gated on
                # having a ranking to throttle by; a no-op when refreshing is disabled.
                refresh_interval = self._exact_solve_refresh_interval
                if (self._candidate_priority is not None
                        and refresh_interval
                        and (iteration + 1) % refresh_interval == 0):
                    self._candidate_priority = np.argsort(-np.nan_to_num(score, nan=-np.inf))

                cat_grad_centered = gradients['Categories'] - gradients['Categories'].mean(axis=1).reshape(-1, 1)
                cat_updates       = optimizers['Categories'].minimize(cat_grad_centered)
                category_weights  = category_weights + cat_updates
                if reg_lambda > 0.0:
                    # L1 proximal step: shrink each weight toward neutral v by up to reg_lambda (a linear
                    # penalty, so a committed punt is not over-penalised the way L2's proportional pull
                    # would). _REG_FLOOR stops the shrink short of v, keeping weights off the singular w==v
                    # ray. Renormalised to the simplex just below.
                    deviation        = category_weights - neutral_row
                    shrink           = np.minimum(reg_lambda, np.maximum(np.abs(deviation) - _REG_FLOOR, 0.0))
                    category_weights = category_weights - np.sign(deviation) * shrink
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
                    # All flex slots share one logit per position, so each slot's shares are the softmax
                    # over its own subset of positions (position_indices). Accumulate every slot's logit-
                    # space gradient -- the softmax Jacobian s ⊙ (g − ⟨g, s⟩) applied to its restricted
                    # shares -- into the shared per-position gradient, then take one optimiser step.
                    # Overlapping positions (PG is in both the guard and utility slots) sum their gradients.
                    master_grad = np.zeros_like(master_logits)
                    for pos_code in self.position_structure['flex']:
                        shares     = position_shares[pos_code].values
                        share_grad = gradient_result['Gradients']['Shares'][pos_code]
                        logit_grad = shares * (share_grad - (share_grad * shares).sum(axis=1, keepdims=True))
                        master_grad[:, self.position_indices[pos_code]] += logit_grad
                    master_logits = master_logits + optimizers['Shares'].minimize(master_grad)
                    if reg_lambda > 0.0:
                        # L1 reg toward uniform on the shared per-position shares (mirrors the category
                        # weights x _POSITION_REG_MULT); re-derive the logits so the pull persists.
                        pos_reg       = reg_lambda * _POSITION_REG_MULT
                        master_shares = _softmax_rows(master_logits)
                        uniform       = 1.0 / master_shares.shape[1]
                        dev           = master_shares - uniform
                        shrink        = np.minimum(pos_reg, np.maximum(np.abs(dev) - _REG_FLOOR, 0.0))
                        master_shares = np.clip(master_shares - np.sign(dev) * shrink, 1e-12, None)
                        master_logits = np.log(master_shares / master_shares.sum(axis=1, keepdims=True))
                    # Derive each flex slot's shares from the shared logits (softmax over its own positions).
                    for pos_code in self.position_structure['flex']:
                        position_shares[pos_code].values[:] = _softmax_rows(
                            master_logits[:, self.position_indices[pos_code]])

            # Warm-start the next pick from the converged best candidate. Computed once after the loop
            # (only the final iteration's values are ever kept) and read straight from the numpy arrays,
            # avoiding a per-iteration pandas column-select + .iloc scan of the share DataFrames.
            best_player_idx = int(np.argmax(score))
            self.initial_category_weights = category_weights[best_player_idx] / 2 + self.v.reshape(self.n_categories) / 2
            self.initial_shares = {
                pos_code: {
                    base: float(position_shares[pos_code].values[best_player_idx, col_idx])
                    for col_idx, base in enumerate(self.position_structure['flex'][pos_code]['bases'])
                }
                for pos_code in self.position_structure['flex'].keys()
            }

        elif (n_players_selected == self.n_picks - 1) or (not self.dynamic and n_players_selected < self.n_picks):
            x_diff_array   = diff_means + x_scores_batch_array
            cdf_estimates  = self.get_cdf(x_diff_array, diff_vars)
            score          = self.get_objective_and_pdf_weights(
                x_diff_array, diff_vars, cdf_estimates, None, sigma_2_m,
                calculate_pdf_weights=False,
            )
            rosters              = None
            expected_future_diff = None
            category_weights_current = None

        elif n_players_selected == self.n_picks:
            cdf_estimates  = self.get_cdf(diff_means, diff_vars)
            score          = self.get_objective_and_pdf_weights(
                diff_means, diff_vars, cdf_estimates, None, sigma_2_m,
                calculate_pdf_weights=False,
            )
            result_index             = ['']
            rosters                  = None
            expected_future_diff     = None
            category_weights_current = None
            position_shares_current  = None

        else:
            # n_players_selected > n_picks: find the best subset to drop.
            extra_players = n_players_selected - self.n_picks
            players_to_remove_possibilities = combinations(my_players, extra_players)

            drop_potentials = pd.concat(
                (self.x_scores.loc[list(players_to_remove)].sum(axis=0)
                 for players_to_remove in players_to_remove_possibilities),
                axis=1,
            ).T
            drop_potentials_array = np.expand_dims(np.array(drop_potentials), axis=2)
            diff_means_mod = diff_means - drop_potentials_array

            cdf_estimates  = self.get_cdf(diff_means_mod, diff_vars)
            score          = self.get_objective_and_pdf_weights(
                diff_means_mod, diff_vars, cdf_estimates, None, sigma_2_m,
                calculate_pdf_weights=False,
            )
            result_index             = drop_potentials.index
            rosters                  = None
            expected_future_diff     = None
            category_weights_current = None
            position_shares_current  = None

        cdf_means = cdf_estimates.mean(axis=2)

        if expected_future_diff is not None:
            expected_diff_means = expected_future_diff.mean(axis=2) + diff_means.mean(axis=2)
        else:
            # No future picks to account for: broadcast the single current-team diff
            # to one row per candidate player, matching cdf_means' leading dimension.
            expected_diff_means = np.broadcast_to(diff_means.mean(axis=2), cdf_means.shape)

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

    def _active_candidate_count(self, iteration, n_candidates):
        """How many top candidates to re-solve roster positions for this iteration (the rest reuse
        the previous iteration's assignment). iteration+1 so the final iteration is a full pass."""
        mode = self._position_mode
        if mode == 'exact':
            return n_candidates
        if mode == 'light':                       # top 300 every iteration, everyone every 5th
            return n_candidates if (iteration + 1) % 5 == 0 else min(300, n_candidates)
        # 'tiered': the global top _exact_solve_window candidates get an exact solve every iteration;
        # everyone else is re-solved only on the every-10th full pass (which also seeds the cache on
        # iteration 0 and keeps the final iteration a full solve, so output scores stay consistent).
        # The window is by GLOBAL rank via candidate_offset, so a batch starting past it exact-solves
        # nobody — only players projecting into the ultimate top window ever get an exact solve. Which
        # candidates fill that set is refreshed from the latest H-scores in perform_iterations (see
        # _exact_solve_refresh_interval), so it follows the contextually strongest players.
        offset = self._candidate_offset
        if (iteration + 1) % 10 == 0:
            return n_candidates
        return max(0, min(self._exact_solve_window - offset, n_candidates))

    def get_objective_and_gradient(self
                                    , category_weights
                                    , position_shares
                                    , diff_means
                                    , diff_vars
                                    , x_scores_batch_array
                                    , candidate_player_array
                                    , team_so_far_array
                                    , n_players_selected
                                    , sigma_2_m
                                    , iteration
                                    , pitching_preference=None
                                    , correction_mode='full'):

        if self.position_means is not None:
            position_rewards = self.get_position_priorities_from_category_weights(category_weights)
            # Re-solve positions for the top candidates every iteration, the next tier every 5th, and
            # everyone every 10th; reuse cached assignments otherwise. iteration+1 so the last
            # iteration (i = n_iterations-1) lands on a full pass, keeping the final scores consistent.
            # Only throttle when we have a cached ranking to prioritise by; otherwise solve everyone.
            if self._candidate_priority is None:
                active_count = candidate_player_array.shape[0]
            else:
                active_count = self._active_candidate_count(iteration, candidate_player_array.shape[0])
            rosters, future_position_array, flex_shares = optimize_positions_all_players(
                candidate_player_array,
                position_rewards,
                team_so_far_array,
                position_shares,
                self._pos_cfg,
                active_count=active_count,
                cached_rosters=self._position_rosters_cache,
                priority_order=self._candidate_priority,
            )
            self._position_rosters_cache = rosters
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

        x_diff_array  = diff_means + x_scores_batch_array + expected_future_diff
        pdf_estimates = self.get_pdf(x_diff_array, diff_vars)
        cdf_estimates = self.get_cdf(x_diff_array, diff_vars)

        score, pdf_weights = self.get_objective_and_pdf_weights(
            x_diff_array, diff_vars, cdf_estimates, pdf_estimates, sigma_2_m,
            calculate_pdf_weights=True,
            correction_mode=correction_mode, iteration=iteration,
        )

        category_gradient = np.einsum('ai,aik -> ak', pdf_weights, del_full)

        # Anti-crowded-punt penalty (kappa): in the early multi-start rounds, subtract a linear penalty
        # for punting (w<v) categories the field is crowding into (self._punt_popularity), so both the
        # seed scan and the descent defect from popular punts. Inert until the empty-board scan has
        # measured popularity and whenever kappa=0. A full roster has no meaningful weights, and the
        # roster<3 gate keeps the penalty off later picks -- so the final scored H-score is unaffected.
        if (self.kappa > 0.0 and self._punt_popularity is not None
                and len(self.players) < _WEAKNESS_SEED_MIN_ROSTER):
            neutral    = self.v.reshape(1, -1)
            punt_depth = np.maximum(neutral - category_weights, 0.0)
            score             = score - self.kappa * (punt_depth @ self._punt_popularity)
            category_gradient = category_gradient + self.kappa * self._punt_popularity.reshape(1, -1) * (category_weights < neutral)

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

    def get_objective_and_pdf_weights(self
                                       , x_diff_array
                                       , diff_vars
                                       , cdf_estimates
                                       , pdf_estimates
                                       , sigma_2_m=None
                                       , calculate_pdf_weights=False
                                       , correction_mode='full'
                                       , iteration=None):
        if self.scoring_format == 'Head to Head: Most Categories':
            return self.get_objective_and_pdf_weights_mc(x_diff_array, diff_vars,
                                                          cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights,
                                                          correction_mode, iteration)
        elif self.scoring_format == 'Rotisserie':
            return self.get_objective_and_pdf_weights_rotisserie(
                x_diff_array, diff_vars, cdf_estimates, pdf_estimates,
                sigma_2_m, calculate_pdf_weights)
        elif self.scoring_format == 'Head to Head: Each Category':
            return self.get_objective_and_pdf_weights_ec(cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights)

    def get_objective_and_pdf_weights_mc(self
                                          , x_diff_array
                                          , diff_vars
                                          , cdf_estimates
                                          , pdf_estimates
                                          , calculate_pdf_weights=False
                                          , correction_mode='full'
                                          , iteration=None):
        """correction_mode: 'full' applies the correlation correction (recomputing per the
        refresh throttle when an iteration index is given), 'skip' turns it off — used by
        the multi-start seed scan, whose coarse pre-descent ranking doesn't warrant it."""
        probs = np.array(cdf_estimates)
        win_probability = compute_win_probability(probs)      # (n_players, n_opponents)

        # First-order correlation correction (eq (C4) of the correlation-correction note):
        #   P(win | R) ≈ P_indep + ½ φ(z)ᵀ [(R − I) ∘ B] φ(z)
        # with B the exact leave-two-out bracket matrix, evaluated at complex nodes
        # (see calculate_correction_terms). Terms are cached across descent iterations
        # and refreshed on the position-throttle-style schedule.
        correction_terms = None
        apply_correction = self.mc_correlation_enabled and correction_mode != 'skip'
        if apply_correction:
            z_scores     = x_diff_array / np.sqrt(diff_vars)
            standard_pdf = _normal_pdf(z_scores)              # φ(z), NOT the 1/σ-scaled pdf

            cached = self._correction_cache
            refresh = (
                iteration is None
                or cached is None
                or cached['correction'].shape != win_probability.shape
                or (calculate_pdf_weights and cached['probability_gradient'] is None)
                or (iteration + 1) % _MC_CORRELATION_REFRESH_INTERVAL == 0
            )
            if refresh:
                correction, m_phi, probability_gradient = calculate_correction_terms(
                    probs
                    , self.rho[0] - np.eye(self.n_categories)
                    , standard_pdf
                    , calculate_gradient=calculate_pdf_weights
                )
                correction_terms = {'correction': correction, 'm_phi': m_phi,
                                    'probability_gradient': probability_gradient}
                if iteration is not None:
                    self._correction_cache = correction_terms
            else:
                correction_terms = cached
            win_probability = win_probability + correction_terms['correction']

        objective = win_probability.mean(axis=1)
        if not calculate_pdf_weights:
            return objective

        pdf_weights = calculate_tipping_points(probs) * pdf_estimates
        if correction_terms is not None:
            # Exact correction gradient, two pieces chained through pdf_estimates = φ(z)/σ:
            #   through the densities:       −z_c [Mφ(z)]_c
            #   through B (leave-three-out): ∂(correction)/∂p_c
            pdf_weights = pdf_weights + (
                -z_scores * correction_terms['m_phi'] + correction_terms['probability_gradient']
            ) * pdf_estimates
        return objective, pdf_weights.mean(axis=2)

    def get_objective_and_pdf_weights_ec(self
                                          , cdf_estimates
                                          , pdf_estimates
                                          , calculate_pdf_weights=False):
        objective = cdf_estimates.mean(axis=2).mean(axis=1)
        if calculate_pdf_weights:
            return objective, pdf_estimates.mean(axis=2)
        return objective

    def get_objective_and_pdf_weights_rotisserie(self
                                                   , x_diff_array
                                                   , diff_vars
                                                   , cdf_estimates
                                                   , pdf_estimates
                                                   , sigma_2_m
                                                   , calculate_pdf_weights=False
                                                   , test_mode=False):
        diff_means   = x_diff_array / np.sqrt(diff_vars)
        pdf_estimates = _normal_pdf(diff_means)
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
            del_sigma_2_d = self.get_del_sigma_2_d(diff_means, self.rho, pdf_estimates, cdf_estimates, f, self.n_drafters)
            del_mu_d      = self.get_del_mu_d(self.n_drafters, pdf_estimates)
            gradient      = self.get_del_v(sigma_d, del_mu_d, mu_d, del_sigma_2_d)
            gradient      = gradient * np.sqrt(diff_vars)
            if test_mode:
                return gradient
            return objective, gradient.sum(axis=2)
        return objective

    def get_pdf(self, x_diff_array, diff_vars):
        r = x_diff_array.reshape(x_diff_array.shape[0], x_diff_array.shape[1] * x_diff_array.shape[2])
        v = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])
        return _normal_pdf(r, scale=np.sqrt(v)).reshape(x_diff_array.shape)

    def get_cdf(self, x_diff_array, diff_vars):
        r = x_diff_array.reshape(x_diff_array.shape[0], x_diff_array.shape[1] * x_diff_array.shape[2])
        v = diff_vars.reshape(diff_vars.shape[1] * diff_vars.shape[2])
        return _normal_cdf(r, scale=np.sqrt(v)).reshape(x_diff_array.shape)

    def clear_initial_weights(self):
        self.initial_category_weights = None
        self.initial_position_shares  = None
        return self

    def populate_default_h_scores(self, n_iterations: int, cash_remaining_per_team: dict = None) -> None:
        """Compute and cache the neutral (empty-board) H-scores. Run once at the end of the build so
        the agent is always primed — the throttle has a ranking to prioritise by, auction values have
        their anchor, and the draft-start evaluate can short-circuit to this result. Pass full cash
        only in auction mode. Runs full-exact (default_h_scores is still None here, so no throttling)."""
        empty = {f'Team {i + 1}': [] for i in range(self.n_drafters)}
        self.clear_initial_weights()
        result = self.get_h_scores(
            player_assignments      = empty,
            drafter                 = 'Team 1',
            n_iterations            = n_iterations,
            cash_remaining_per_team = cash_remaining_per_team,
        )
        self.default_h_scores = result['Scores'].sort_values(ascending=False)
        self._default_result  = result

    # ── simplified-form x_mu helpers (unchanged) ──────────────────────────────

    def get_x_mu_simplified_form(self, c, L, v):
        # 'aij,ajk -> aik' is a batched matmul; @ dispatches to BLAS (~5-6x faster than einsum here).
        return L @ self.get_last_four_terms(c, L, v)

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
        vvL       = v.dot(v.T) @ L
        factor_top = np.einsum('pad,dp -> ap', vvL, c.T)
        vL        = v.T @ L
        vLv       = np.einsum('pad,dc -> ap', vL, v)
        factor    = (factor_top / vLv).T
        c_mod     = c - factor
        cLc       = np.einsum('pd,pd -> p', np.einsum('pc,pcd -> pd', c_mod, L), c_mod)
        return np.sqrt(cLc.reshape(-1, 1, 1))

    def get_term_five_b(self, c, L, v):
        cL  = np.einsum('pc,pcd -> pd', c, L)
        cLc = np.einsum('pd,pd -> p', cL, c)
        vTL = v.T @ L
        vTLv = np.einsum('pad,dc -> ap', vTL, v)
        Lct = np.einsum('pcd,dp -> cp', L, c.T)
        vTLc = np.einsum('ac,cp -> ap', v.T, Lct)
        return (cLc * vTLv - vTLc ** 2).reshape(-1, 1, 1)

    def get_terms_four_five(self, c, L, v):
        return self.get_term_four(c, v) * self.get_term_five(c, L, v)

    def get_del_term_four(self, c, v):
        return (np.identity(v.shape[0]) * self.gamma).reshape(1, v.shape[0], v.shape[0])

    def get_del_term_five_a(self, c, L, v):
        vvL       = v.dot(v.T) @ L
        factor_top = np.einsum('pad,dp -> ap', vvL, c.T)
        vL        = v.T @ L
        vLv       = np.einsum('pad,dj -> jp', vL, v)
        factor    = (factor_top / vLv).T
        c_mod     = c - factor
        top_og    = np.einsum('pc,pcd -> pd', c_mod, L)
        top       = top_og.reshape(-1, 1, v.shape[0])
        bottom    = np.sqrt(np.einsum('pd,pd -> p', top_og, c_mod).reshape(-1, 1, 1))
        side      = (np.identity(v.shape[0])
                     - (v.dot(v.T) @ L) / vLv.reshape(-1, 1, 1))
        return np.einsum('pia,pad -> pid', top / bottom, side).reshape(-1, 1, v.shape[0])

    def get_del_term_five_b(self, c, L, v):
        cL   = np.einsum('pc,pcd -> pd', c, L)
        vTL  = v.T @ L
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
        return L @ self.get_terms_four_five(c, L, v)

    def get_del_last_three_terms(self, c, L, v):
        return L @ self.get_del_terms_four_five(c, L, v)

    def get_last_four_terms(self, c, L, v):
        return self.get_term_two(c, v) @ self.get_last_three_terms(c, L, v)

    def get_del_last_four_terms(self, c, L, v):
        ci   = self.get_del_term_two(v)
        cii  = self.get_last_three_terms(c, L, v)
        ta   = np.einsum('aijk,aj -> aik', ci, cii.reshape(-1, v.shape[0]))
        tb   = self.get_term_two(c, v) @ self.get_del_last_three_terms(c, L, v)
        return ta + tb

    def get_del_full(self, c, L, v):
        return L @ self.get_del_last_four_terms(c, L, v)

    # ── Rotisserie helpers (unchanged) ─────────────────────────────────────────

    def get_f(self, pdfs):
        return pdfs.sum(axis=2)

    def get_g(self, pdfs):
        # 'pao,pbo -> pab' is a batched matmul (pdfs @ pdfs^T per row); @ dispatches to BLAS.
        return pdfs @ pdfs.transpose(0, 2, 1)

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
        return _normal_cdf(mu_d / sigma_d).reshape(-1)

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

    #ZR: Equation 33 in the Rotisserie paper is wrong: 
    # it is missing the n_managers / (n_managers -1 ) factor that it should have inherited 
    # from equation 33. It also has a typo, the second sigma_t should be sigma_L
    #The issue with the missing scaling factor is fixed here 
    def get_del_sigma_2_d(self, opponent_mu_matrix, rho, pdfs, cdfs, f, n_managers):
        rho2 = rho.copy()
        rho2[:, np.arange(rho2.shape[1]), np.arange(rho2.shape[2])] = 0
        inside  = -pdfs - f.reshape(-1, f.shape[1], 1)
        fp      = rho2 @ inside   # 'pab,pbc -> pac' batched matmul
        fc1     = (opponent_mu_matrix * pdfs) * (fp + (pdfs - f.reshape(-1, f.shape[1], 1)))
        fc2     = pdfs * (1 - 2 * cdfs)
        return (fc1 + fc2) * n_managers / (n_managers - 1)

    def get_del_mu_d(self, n_managers, pdfs):
        return n_managers / (n_managers - 1) * pdfs

    def get_del_v(self, sigma_d, del_mu_d, mu_d, del_sigma_2_p):
        return (_normal_pdf(mu_d / sigma_d) / (sigma_d ** 3)
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

def build_h_agent(info
                  , omega
                  , gamma
                  , n_starters
                  , n_drafters
                  , beth
                  , scoring_format
                  , dynamic
                  , sport
                  , params
                  , slot_counts
                  , aleph=0.0):
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
