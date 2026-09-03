"""
The H-scoring agent: category-weight optimisation over the drafting objective.

Ported from the original Streamlit implementation (whose src/ tree is retired); the port
replaced st.session_state and get_*() config reads with explicit constructor parameters
and moved caching to the Session layer. The pure-math methods (get_pdf, get_term_*, the
Rotisserie helpers, AdamOptimizer) are unchanged from that original.
"""

from __future__ import annotations

from typing import Optional

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
# There are deliberately no environment overrides in this module: experiments vary the
# session parameters the same way a user could, so every run is reproducible from its
# request alone.
# lowvar seed (Roto): exponent on the seed tilt that down-weights high-v (less stable) categories.
_LOWVAR_TILT = 0.5
# Robustness regulariser: each iteration soft-thresholds the category weights toward neutral v -- the
# proximal step for an L1 penalty -lambda*||w-v||_1. L1 rewards balance without over-penalising a
# committed punt the way L2 would, and its sparsity pins uncontested categories at v while letting a few
# deviate. The per-pick strength lambda follows a Gaussian (phi) schedule (built per agent from n_picks
# in __init__): it starts at the peak on an empty roster and decays to ~0 by the final pick.
# Regularisation strength: the peak of the decay schedule (the lambda on an empty roster). Surfaced as
# the session parameter reg_lambda; this value is what parameters.yaml defaults it to, kept here as the
# figure testing settled on.
_REG_STRENGTH = 0.00005
# reg_lambda reaches the agent in units of REG_LAMBDA_UNIT, so that a sidebar box reads 0.05 rather
# than 0.00005 -- the scale the other model parameters live on. This is the only place the two
# representations meet.
REG_LAMBDA_UNIT = 1e-3
# Optional guard (default 0 = off, clean L1 that may snap onto v): keep weights at least this far per
# component from the singular w==v ray. Only needed if reg_lambda's ceiling is ever raised past the
# ~5e-4 it caps at today (parameters.yaml max 0.5 x REG_LAMBDA_UNIT), where the small empty-board
# deviations let the shrink reach v and term_five goes singular (EC in particular).
_REG_FLOOR = 0.0
# Position/flex-share reg strength as a multiple of the category reg: shares live on a coarser simplex
# (few bases), so they need a firmer pull toward uniform to have a comparable effect.
_POSITION_REG_MULT = 1000.0
# Anti-crowded-punt coupling ("kappa"): a linear penalty in the objective on punting categories the
# FIELD is crowding into. On the empty-board first run, the multi-start seed scan reveals which single
# punts the top _PUNT_POPULARITY_TOP_N players most want (the popularity vector); the penalty then
# discourages joining those popular punts, dispersing the crowd (crowded punts are competed away). It
# acts only in the early multi-start rounds (roster < _WEAKNESS_SEED_MIN_ROSTER). 0 disables it (no
# behaviour change); higher = firmer defection from the crowd. Surfaced as the per-session `kappa`
# parameter (default 0.3).
_PUNT_POPULARITY_TOP_N = 40
# Gaussian (phi) reg-decay shape: lambda_k = peak*(phi(B k/n) - phi(B))/(phi(0)-phi(B)) -- peak on an
# empty roster, decaying to exactly 0 at the final pick. B sets the concave shoulder (~ first n/B picks)
# before the convex tail; B=4 puts the shoulder near pick 3 and matches the old cosine's total budget.
_REG_SHAPE_B = 4.0
# Correlation-correction refresh interval, mirroring the position-optimiser throttle: the
# correction terms are recomputed on iterations where (iteration+1) % interval == 0 (plus
# the cold start) and reused between — they drift slowly with the category weights, so a
# small staleness buys back most of the correction's per-iteration cost. One-shot scoring
# calls (full team, trades, auction values) always compute fresh.
_MC_CORRELATION_REFRESH_INTERVAL = 4
# Opponent modelling: instead of padding every opponent's future picks with a category-neutral
# average, treat each opponent as a rational H-score drafter who punts. We track a per-opponent
# mu_edge (the expected per-pick edge vector implied by their inferred category weights) and add
# picks_left * mu_edge to their totals — mirroring the edge term the objective already adds for OUR
# own remaining picks. Stale opponents are refreshed EAGERLY at the top of every evaluate: a pick makes
# exactly one seat stale, and each refresh replays that seat's most recent pick through the real solver
# as a single-candidate solve (~1/40th of an evaluate), so there is no lazy budget and the field model
# is always current. _OPPONENT_INFERENCE_ITERATIONS bounds that nested solve's descent. Setting the
# opponent_model_confidence session parameter to 0 disables the whole feature and restores
# byte-identical neutral-opponent behaviour.
_OPPONENT_INFERENCE_ITERATIONS = 50
# Build-time fictitious-play bootstrap: the base H-scores are recomputed several times, each pass
# best-responding to the RUNNING AVERAGE of prior passes' builds rather than the latest one. Pure
# best-response oscillates (every seat flips the same way each pass); averaging damps that and lets the
# predicted field converge toward a mixed equilibrium where punts actually spread. 1 pass = neutral only.
# 8 is the measured knee: the punt structure settles by pass 6-9 in all six seasons (see the self-play
# convergence experiment), the 8-pass serve lands within 0.16pp of the 15-pass serve with an identical
# top-40, and dropping to 7 or 6 quintuples that drift. Cost is quadratic in the pass count (pass k
# faces k stacked field snapshots), so 15 -> 8 halves populate time.
_OPPONENT_BOOTSTRAP_PASSES = 8
# Rotisserie keeps the longer run: it uses the EMA field path (window 0), whose fixed-rate smoothing
# stabilises more slowly than the window's 1/t fictitious-play steps — at 8 passes a 2024-25 top-12
# Roto build hard-punts a category (breaking the minimal-punting floor), at 15 none do. Roto passes
# face a single 11-column field, so its cost is linear in the pass count and the longer run is cheap.
_OPPONENT_BOOTSTRAP_PASSES_ROTISSERIE = 15
# Damping for the fixed-point field: each bootstrap pass nudges the committed field a fraction alpha
# toward that pass's best-response (exponential smoothing), instead of a running mean. Smoothing
# converges to a genuine fixed point (field == its own best-response — a committed equilibrium), whereas
# a mean converges to a smeared blend of oscillating responses that erases specific punts.
_OPPONENT_SMOOTHING = 0.3
# Windowed fictitious play for the BOOTSTRAP PASSES ONLY (H2H formats). Each pass best-responds to the
# raw fields of the last K passes stacked as separate opponents (11*K columns; the objective's mean over
# the opponent axis weights them uniformly) instead of the single EMA-blended field above. At a mixed
# equilibrium the true field is a DISTRIBUTION over builds — some seats punt one category, some another
# — and stacking preserves each historical build's specific punts where blending smears them into a
# build nobody would draft. The window also replaces the EMA's damping: one new pass moves at most 1/K
# of the field's mass. The DEFAULT keeps the ENTIRE history (any value >= the pass count): that is true
# fictitious play, whose 1/t step decay is what converges — measured settling in all six seasons,
# whereas a small sliding window (fixed step size) sustains large limit cycles. 0 selects the EMA
# path. Strictly self-play machinery: the serve pass and everything mid-draft face
# the FINAL pass's single field. Rotisserie always uses the EMA path (its standings objective scales
# with opponent count, and its punts are structural anyway).
_OPPONENT_FIELD_WINDOW = 999
# Descent iterations for the field-building bootstrap passes. These only need approximate opponent builds
# (the field is smoothed and never exact anyway), so they run short; the final full-pool serve pass uses
# the session's full n_iterations for accurate base H-scores.
_OPPONENT_PASS_ITERATIONS = 15
# Learning-rate scales for WARM-STARTED category descents, tiered by how far the optimum can plausibly
# have moved since the stored weights were computed. A warm start begins inside an established basin, so
# the descent only needs fine adjustment -- and on value-flat plateaus (near-tied builds), full-size Adam
# steps wander a full step-length per evaluate regardless of how converged the start is, visibly flipping
# the displayed build between evaluates. Small steps pin the plateau AND converge better in
# genuine-change cases (the gain there comes from the board, not from weight travel; measured
# H 52.37 -> 52.51 on the Jokic-joins-my-team case). Three tiers:
#   cold start (no stored weights)              -> full rate: must travel from a generic punt seed.
#   warm, MY roster changed since storage       -> full rate: a real pick decision needs full
#       adaptation -- running this tier at 1/10th measurably cost drafting strength in the all-H-field
#       awareness validation (aware-field gain -0.0015 at 0.1 vs +0.0008 at 1.0, the no-regret
#       equilibrium level), so only the warm INIT and the regulariser exemption apply here.
#   warm, MY roster unchanged since storage     -> 1/100th: only the field moved (opponent picks are
#       mean-invariant for predicted players; variance/pool shifts are tiny), so barely adjust. This
#       static tier is where the display-stability guarantees live.
_WARM_START_LEARNING_RATE_SCALE = 1.0
_WARM_START_STATIC_ROSTER_SCALE = 0.01
from pathlib import Path
from itertools import combinations

# Anchored on this file so the reference CSVs load from any working directory.
_DATA_DIR = Path(__file__).parents[1] / 'data'

from backend.math.algorithm_helpers import (
    compute_win_probability,
    calculate_win_probability_and_tipping_points,
    calculate_correction_terms,
)
from backend.math.process_player_data import get_category_level_rv, scale_tiebreaker_value
from backend.math.position_optimization import (
    optimize_positions_all_players,
    get_player_rows,
)
from backend.math.position_config import PositionConfig, build_position_config
from backend.player_identity import RP_PLAYER_ID


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
                 , most_categories_weight: Optional[float]
                 , tiebreaker_category: Optional[str]
                 # ── explicit context (replaces get_*() calls) ──
                 , sport: str
                 , params: dict
                 , slot_counts: dict
                 , aleph: float = 0.0
                 , kappa: float = 0.3
                 , reg_lambda: float = _REG_STRENGTH / REG_LAMBDA_UNIT
                 , opponent_model_confidence: float = 0.5
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

        # Head to Head is one format with a dial: 0 scores every category on its own (Each
        # Category), 1 scores only whether the majority was taken (Most Categories), and values
        # between blend the two objectives. Rotisserie has no such dial and must be given None —
        # a number there would mean the caller thinks a setting applies that this format ignores,
        # and would split the pipeline cache across values that build identical agents.
        if scoring_format == 'Rotisserie':
            if most_categories_weight is not None:
                raise ValueError('most_categories_weight does not apply to Rotisserie; pass None. '
                                 f'Got {most_categories_weight!r}.')
        elif most_categories_weight is None or not 0.0 <= most_categories_weight <= 1.0:
            raise ValueError('Head to Head needs a most_categories_weight in [0, 1]. '
                             f'Got {most_categories_weight!r}.')
        self.most_categories_weight = most_categories_weight

        # Per-format config. All formats run the
        # robustness regulariser; they differ only in the cold-start seed -- Rotisserie uses the lowvar
        # tilt (its punts are structural, not a strategic fork), the H2H formats use multi-start punt
        # seeding so early picks avoid over-committing to a punt they may drop.
        is_rotisserie       = scoring_format == 'Rotisserie'
        self.seed_mode = 'lowvar' if is_rotisserie else 'multistart'

        # Gaussian (phi) regulariser schedule built from the draft length: strength reg_lambda (the
        # peak) on an empty roster, decaying to ~0 by the final pick (indexed by roster size),
        # with a concave shoulder set by _REG_SHAPE_B.
        reg_peak = reg_lambda * REG_LAMBDA_UNIT
        _phi0    = 1.0 - np.exp(-_REG_SHAPE_B ** 2 / 2)
        self.reg_schedule = [reg_peak * (np.exp(-(_REG_SHAPE_B * k / n_picks) ** 2 / 2)
                                         - np.exp(-_REG_SHAPE_B ** 2 / 2)) / _phi0
                             for k in range(n_picks)]

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
        # Throttle ranking DURING populate, where default_h_scores is deliberately None (that also
        # disarms the empty-board short-circuit): each bootstrap pass stores its scores here so the
        # next pass -- and above all the full-pool serve -- runs its normal throttle schedule instead
        # of exact-solving all ~577 candidates every iteration. Live only inside populate.
        self._populate_pass_scores = None

        # Build position config (replaces all get_position_*() calls)
        self.position_config: PositionConfig = build_position_config(params, slot_counts)

        # ── info dict unpacking ────────────────────────────────────────────────
        self.positions = info['Positions']
        self.w         = info['w']
        x_scores       = info['X-scores']

        self.n_categories = x_scores.shape[1]

        # The category that settles an otherwise tied matchup, by counting for two. Resolved to a
        # column index once here: the objective runs every descent iteration and should not be
        # looking names up. It means something only where a tie can happen — the majority
        # objective, with an even number of categories — and callers pin it to None elsewhere when
        # it cannot apply (see normalize_objective_settings), so anything left over here is a
        # mistake worth surfacing rather than ignoring.
        self.tiebreaker_index = None
        if tiebreaker_category is not None:
            # Widest question first: does this objective score matchups at all? Then whether the
            # category exists, then whether there is a tie for it to break.
            if scoring_format == 'Rotisserie' or most_categories_weight == 0:
                raise ValueError('A tiebreaker only applies to the majority objective; it does '
                                 'nothing under Rotisserie or at most_categories_weight 0.')
            category_names = list(x_scores.columns)
            if tiebreaker_category not in category_names:
                raise ValueError(f'Tiebreaker category {tiebreaker_category!r} is not one of this '
                                 f"session's categories: {category_names}.")
            if self.n_categories % 2 == 1:
                raise ValueError('A tiebreaker needs an even number of categories to break a tie; '
                                 f'this session has {self.n_categories}.')
            self.tiebreaker_index = category_names.index(tiebreaker_category)

        #TODO: clean this up
        if info['Position-Means'] is not None:
            self.position_means = np.array(info['Position-Means']).reshape(1, -1, self.n_categories)
            
            position_means_df = info['Position-Means']
            position_means_df.loc['NP'] = 0

            # A player's positional baseline is the average of the position means over ALL of their
            # eligible positions (e.g. a PF/C uses the mean of the PF and C means, not just PF). reindex
            # then mean(axis=0) skips any listed position absent from position_means_df.
            rel_players = [p for p in x_scores.index if p != RP_PLAYER_ID]
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
        pn         = self.position_config.position_numbers
        ps         = self.position_config.position_structure
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
        if scoring_format == 'Rotisserie' or most_categories_weight > 0:
            if sport == 'NBA':
                rho = pd.read_csv(_DATA_DIR / 'basketball_correlations.csv').set_index('Category')
            else:
                rho = pd.read_csv(_DATA_DIR / 'baseball_correlations.csv').set_index('Category')

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
        # Parked OFF while the opponent model is under development: the correction over-suppresses
        # punting on a contested board (it peaks at win-prob 0.5), which fights the opponent-model
        # work. The machinery it gates stays wired for the correlation-factor branch; re-enabling
        # means flipping this flag (deliberately no runtime switch).
        self.mc_correlation_enabled = False
        if self.mc_correlation_enabled and self.tiebreaker_index is not None:
            # The correction's bracket matrix is derived for an unweighted majority (see
            # _bracket_targets); a category worth two points is outside that derivation, so the
            # combination is refused rather than silently corrected by the wrong quantity.
            raise ValueError('The Most Categories correlation correction (mc_correlation_enabled) does '
                             'not support a tiebreaker category.')
        # Per-descent cache of correction terms (see get_objective_and_pdf_weights_mc);
        # reset at the start of every perform_iterations run.
        self._correction_cache = None

        # Rational-opponent modelling (see get_diff_distributions / refresh_stale_team_states).
        # One session parameter carries both halves of it: how sharply opponents are expected to
        # pursue their predicted punts, and — at zero — whether they are modelled as strategic at
        # all. Zero restores neutral-opponent behaviour byte-for-byte. Rotisserie pins full
        # confidence: there is no equivalent uncertainty about punting there.
        self.opponent_model_confidence = (1.0 if scoring_format == 'Rotisserie'
                                          else float(opponent_model_confidence))
        self.models_opponents = self.opponent_model_confidence > 0.0
        # Per-team inferred build, keyed by team name:
        #   {'roster_key': frozenset(their non-NaN roster), 'category_weights': (n_cat,), 'mu_edge': (n_cat,)}.
        # roster_key self-invalidates as rosters grow (a new roster is a new key).
        self._team_states = {}
        # Round-one prior: mu_edge per top player, harvested from the base-H-score build run. Empty
        # opponents (no picks yet) borrow the archetype of the player they are most likely to draft.
        # An Index of the top-3N players by served H-score: purely the ORDERING seat assignment walks.
        self._anchor_player_order = None
        # Committed per-player Future-Diff, scaled per remaining pick (= result['Future-Diff'] /
        # (n_picks - 1), the mu_edge the maths docs describe), for every candidate. When an opponent
        # drafts player X as its first pick we reuse X's committed row verbatim, so a priced-in pick
        # confirms the predicted field instead of re-inferring a different build.
        self._player_committed_future_diffs = None
        # Per-player self-play weights and flex position shares, frozen at populate (each player's
        # converged build vs the equilibrium field). The two are ALWAYS stored and applied together — a
        # build is (category weights + flex allocation), and warm-starting one half while cold-starting
        # the other would start the descent internally inconsistent. Warm-start thin-roster candidate
        # descents and seed opponent inference.
        self._player_frozen_weights = None
        self._player_frozen_shares  = None
        # Row-positions + weights for a partial warm start (set per evaluate when the frozen table only
        # covers part of the batch, e.g. the serve pass); consumed by perform_iterations' seed step.
        self._partial_warm_start_rows = None
        # Per-row learning-rate multipliers for warm-started candidate rows (None = all cold/full rate).
        self._warm_start_row_rate_scales = None
        # Windowed-fictitious-play field: list of (committed_future_diffs, anchor_player_order) snapshots
        # from the last K bootstrap passes, newest last. ONLY ever set between passes inside
        # populate_default_h_scores (and cleared before the serve) — it must never exist mid-draft, so
        # reset_draft_state deliberately leaves it alone (passes reset state while the window is live).
        self._bootstrap_field_snapshots = None
        # True while a nested opponent-inference solve is running: caps the recursion at one level of
        # best-response (the nested solve reads the stored team states, never refreshes them) and
        # suppresses the post-evaluate team-entry hook (the refresh loop stores the entry itself, keyed
        # by the opponent's FULL roster rather than the replay view's roster-minus-last-pick).
        self._opponent_inference_active = False

        if scoring_format == 'Rotisserie':
            self.x_scores = x_scores.loc[
                info['G-scores'].sum(axis=1).sort_values(ascending=False).index
            ]
            v = np.sqrt(mov / (mov + vom))

            # ── max_info (replaces get_max_info()) ────────────────────────────
            if self.n_drafters <= 21:
                max_table = pd.read_csv(_DATA_DIR / 'max_table.csv')
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

        # A tiebreaker is worth (1 + most_categories_weight) of an ordinary category, and v is what
        # a category is worth per unit of x-score — so the neutral vector itself carries it, and
        # everything reading v (punt depth, the field weights in get_x_mu, the descent's starting
        # point) inherits it. process_player_data applies the same factor to the G-scores it
        # derives; the shared helper keeps the two from drifting.
        if self.tiebreaker_index is not None:
            v = scale_tiebreaker_value(np.asarray(v, dtype=float), self.tiebreaker_index,
                                       self.most_categories_weight)
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
        self.position_structure = self.position_config.position_structure
        self.position_indices   = self.position_config.position_indices

        self.initial_category_weights = None
        # Anti-crowded-punt coupling (the per-session kappa parameter).
        self.kappa = kappa
        # How much weight the predicted punting behavior of other teams carries: every opponent tilt
        # read (committed archetype, inferred build, spare seat — draft and auction alike) is scaled by
        # this factor at field construction, so the bootstrap, the serve, and in-draft evaluates all
        # model the same softened world. 1 = full self-play equilibrium; 0 = category-neutral opponents.
        # Real leaguemates punt less sharply than the equilibrium, and against fully-rational punters
        # the best response stops punting entirely (their punts concede those categories for free), so
        # softening re-opens the drafter's own punt lines. ROTISSERIE PINS FULL CONFIDENCE: its
        # self-play field is what suppresses the paper model's perverse hard punts (everyone shades the
        # volatile categories, so abandoning one is punished) — at half strength that discouragement
        # fails and extreme-profile players fall back into 2%-win-rate FT% punts (measured, 2020-21).
        # Field punt-popularity vector (per category), measured once on the empty-board multi-start scan
        # and reused for the early picks; drives the anti-crowded-punt (kappa) objective penalty. None
        # until measured (and whenever kappa=0), which makes the penalty inert.
        self._punt_popularity = None

        # ── MLB-specific setup (replaces get_pitcher_stats() / get_league_type()) ──
        # MLB is UNSUPPORTED: kept from the Streamlit port, but no current ingestion path
        # produces MLB data (every source is NBA-keyed), so the sport == 'MLB' branches in
        # the backend are unreachable and untested. This is the authoritative note; the
        # other branches carry a one-line pointer.
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

        # Bring every stale seat's stored build up to date BEFORE this call's per-evaluate attributes are
        # set (eager: a pick makes exactly one seat stale and a refresh is a cheap single-candidate
        # solve, so the field is always current). The ordering is load-bearing: each nested inference
        # solve is itself a get_h_scores run that overwrites self.players / self.n_drafters /
        # self._position_mode with ITS view, so the refresh must finish before this evaluate assigns its
        # own. Draft and auction alike; suppressed inside a nested inference solve — one level of
        # best-response.
        if self.models_opponents and not self._opponent_inference_active:
            self.refresh_stale_team_states(player_assignments, drafter, cash_remaining_per_team)

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

        # diff_means/diff_vars/sigma_2_m broadcast against the candidate batch. With opponent modelling on,
        # diff_means is candidate-DEPENDENT (N, n_cat, n_drafters-1) for candidates that are themselves a
        # seated anchor — self-exclusion swaps their own team out of the field (see get_diff_distributions).
        diff_means, diff_vars, sigma_2_m, opponent_future_tilts = self.get_diff_distributions(
            player_assignments, drafter, x_scores_available, cash_remaining_per_team,
            candidate_batch=list(x_scores_batch.index),
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

        # initial_category_weights / initial_shares are a TEST-INJECTION hook only (the multistart_*
        # diagnostics set them to force a specific start); production never writes them. Real warm
        # starts live in _team_states / _player_frozen_weights below.
        if self.initial_category_weights is None:
            # Warm start (opponent model on). Weights track the OVERALL BUILD far more than the marginal
            # candidate, so once the roster has a real shape (>= _WEAKNESS_SEED_MIN_ROSTER, the same
            # threshold the cold start uses) every candidate starts from this drafter's own team-level
            # entry -- the same per-team store the opponents live in, updated by this drafter's previous
            # evaluate (or by inference, if this seat was an opponent until a perspective switch). Below
            # that the candidate still largely IS the build (and the empty-board entry may reflect a
            # candidate that was never actually picked), so each candidate starts from its own frozen
            # self-play weights (the per-player table snapshotted at populate). Both paths start descents
            # inside an established basin, which removes the cold seed-scan's near-tie vacillation (midway
            # stalls that showed up as sudden dollar dips). The stores are absent with the model off and
            # reset before every bootstrap pass, so gate-off behaviour and the populate fixed-point are
            # untouched; uncovered cases fall through to the cold-start seeding below.
            warm_start_weights = None
            warm_start_shares  = None   # flex shares are PAIRED with the weights: same source, same rows
            self._partial_warm_start_rows = None
            self._warm_start_row_rate_scales = None   # per-row learning-rate multipliers (None = all cold)
            team_state = self._team_states.get(drafter) if self.models_opponents else None
            # HYBRID start policy for a drafter's own evaluates: the first few picks (roster below
            # _WEAKNESS_SEED_MIN_ROSTER) COLD multi-start via the punt seed scan — the strategic
            # window where re-balancing is cheap and exploration pays — and from the third pick on,
            # every candidate warm-starts from the drafter's own team entry (its identity). Inference
            # replays use the entry at ANY roster size: the pick being explained is known, so "the
            # Wemby team that added Duren" starts from the Wemby identity, never a fresh Duren-led
            # derivation. Entries are always identity-true (committed anchor build or a replay of
            # actual picks; evaluates never write them). Measured at opponent_model_confidence 0.5 over
            # full 12-seat drafts: early punt-set re-balancing 44% vs 28% warm-only, late-round
            # stability equal to warm-only (21% vs 19%), 12/12 distinct final builds, no herding.
            # CAUTION: the cold window is only herd-safe on a softened field — at rationality 1.0 the
            # equilibrium is knife-edge and cold solves crowd into its one open punt lane.
            team_entry_applies = (n_players_selected >= _WEAKNESS_SEED_MIN_ROSTER
                                  or self._opponent_inference_active)
            if (team_entry_applies and team_state is not None
                    and team_state['category_weights'] is not None):
                # Tier by staleness: an unchanged roster means only the field moved since the entry was
                # stored, so the optimum has barely moved and the descent should barely move either.
                # Inference replays always land here (the eager refresh keys each entry to exactly the
                # replay view's roster), which is what damps in-draft best-response herding: a build
                # moves by one polish step per pick, never a full re-derivation.
                roster_unchanged = team_state['roster_key'] == frozenset(my_players)
                warm_scale = (_WARM_START_STATIC_ROSTER_SCALE if roster_unchanged
                              else _WARM_START_LEARNING_RATE_SCALE)
                warm_start_weights = np.array([team_state['category_weights']] * len(x_scores_batch))
                self._warm_start_row_rate_scales = np.full(len(x_scores_batch), warm_scale)
                if team_state['position_shares'] is not None:
                    warm_start_shares = {
                        pos_code: pd.DataFrame(
                            np.tile(team_state['position_shares'][pos_code], (len(x_scores_batch), 1)),
                            columns=pos_info['bases'],
                        )
                        for pos_code, pos_info in self.position_structure['flex'].items()
                    }
            elif (self._player_frozen_weights is not None
                  and (n_players_selected == 0 or self._opponent_inference_active)):
                # Frozen rows were computed on the EMPTY board, so they warm-start only the
                # unchanged-context cases: empty-roster evaluates (the serve pass and pre-draft
                # browsing, which must reproduce the served builds) and inference replay views.
                # A drafter with 1-2 picks deliberately falls through to the COLD punt seed scan —
                # the hybrid policy's exploration window (see the team-entry comment above).
                warm_scale = (_WARM_START_STATIC_ROSTER_SCALE if n_players_selected == 0
                              else _WARM_START_LEARNING_RATE_SCALE)
                stored_for_batch = self._player_frozen_weights.reindex(x_scores_batch.index)
                covered          = ~stored_for_batch.isna().to_numpy().any(axis=1)
                # Shares come from the same serve result as the weights, so coverage is identical.
                frozen_shares = self._player_frozen_shares
                shares_usable = (frozen_shares is not None
                                 and all(frame is not None for frame in frozen_shares.values()))
                if covered.all():
                    warm_start_weights = stored_for_batch.to_numpy()
                    self._warm_start_row_rate_scales = np.full(len(x_scores_batch), warm_scale)
                    if shares_usable:
                        warm_start_shares = {
                            pos_code: frame.reindex(x_scores_batch.index).reset_index(drop=True)
                            for pos_code, frame in frozen_shares.items()
                        }
                elif covered.any():
                    # Partial coverage (the serve pass: the bootstrap's final iteration pass only covers
                    # the top-3N anchors). Covered rows warm-start; the rest cold-start via the punt seed
                    # scan -- the override is applied right after the scan in perform_iterations. Shares
                    # need no handoff: uniform IS the cold share init, so build the mix directly here.
                    covered_positions = np.flatnonzero(covered)
                    self._partial_warm_start_rows = (
                        covered_positions, stored_for_batch.to_numpy()[covered]
                    )
                    self._warm_start_row_rate_scales = np.where(covered, warm_scale, 1.0)
                    if shares_usable:
                        warm_start_shares = {}
                        for pos_code, pos_info in self.position_structure['flex'].items():
                            mixed = np.full((len(x_scores_batch), len(pos_info['bases'])),
                                            1.0 / len(pos_info['bases']))
                            stored_rows = frozen_shares[pos_code].reindex(x_scores_batch.index).to_numpy()
                            mixed[covered_positions] = stored_rows[covered_positions]
                            warm_start_shares[pos_code] = pd.DataFrame(mixed, columns=pos_info['bases'])

            # Cold start: uniform flex shares. The category-weight init is normally left to punt-seeding
            # in perform_iterations (None signals it). SEED_MODE=heuristic restores the old per-candidate
            # heuristic init instead, for A/B testing the seeding against the baseline.
            if warm_start_weights is not None:
                initial_category_weights = warm_start_weights
            elif self.seed_mode == 'heuristic':
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
                # parallel to v, giving a 0/0. The nudge must comfortably exceed the L1 regulariser's
                # capture radius (reg_lambda ~ 5e-5): the prox shrinks toward EXACT v each iteration and
                # _REG_FLOOR defaults to 0, so a sub-radius jitter gets snapped onto the singularity
                # (2022-23 collapsed every neutral-seeded descent to NaN this way). 1% is ~20x the
                # radius while still being neutral for all practical purposes.
                neutral = self.v.reshape(self.n_categories)
                jitter  = 1.0 + 1e-2 * np.where(np.arange(self.n_categories) % 2 == 0, 1.0, -1.0)
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
            # Flex shares: warm rows carry their stored allocation (paired with the weights above);
            # everything else starts uniform, which is and was the cold share init.
            if warm_start_shares is not None:
                initial_position_shares = warm_start_shares
            else:
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
        # During populate that cache is deliberately None; the full-pool SERVE threads the final
        # field pass's scores instead (players outside them sort last). Only the serve: the subset
        # field passes must stay exact — throttling them (draft's 'tiered' schedule solves 30 of
        # the 36 anchors) perturbs the anchor-field equilibria enough to reshuffle the served
        # top-40 by whole ranks (measured ~9pp score moves). Missing/uncached players sort last;
        # with no ranking at all we pass None, which disables throttling entirely.
        ranking_scores = self.default_h_scores
        if ranking_scores is None and candidate_subset is None:
            ranking_scores = self._populate_pass_scores
        if ranking_scores is not None:
            ranked = ranking_scores.reindex(x_scores_batch.index).to_numpy()
            candidate_priority = np.argsort(-np.nan_to_num(ranked, nan=-np.inf))
        else:
            candidate_priority = None

        result = self.perform_iterations(
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
            opponent_future_tilts=opponent_future_tilts,
        )

        # NOTE: evaluates deliberately write NOTHING to _team_states. The store is a pure function of
        # the rosters, written only by refresh_stale_team_states (committed reuse or replay inference,
        # every seat including the drafter), so the field a team faces never depends on which seats were
        # viewed or how often — switching perspective computes nothing and therefore changes nothing.
        # A post-evaluate hook that stored the drafter's own converged build here was the last source of
        # view-dependence: it replaced replay-derived entries with best-candidate builds, visibly moving
        # other teams' evaluates whenever a seat was viewed.
        return result

    def get_diff_distributions(self
                               , player_assignments
                               , drafter
                               , x_scores_available
                               , cash_remaining_per_team=None
                               , opponent_model='inferred'
                               , candidate_batch=None):
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

            # Rational-opponent modelling (auction), parallel to the draft. An EMPTY seat with a predicted
            # first buy is modelled as having ALREADY bought it: the anchor's real stats are priced into the
            # score diff and its estimated price is deducted from the seat's cash, so the field does not jump
            # when the anchor is actually bought (the draft gets this for free; auctions must also charge the
            # cash). Then slots_left * mu_edge tilts the remaining generic fill -- the zero-sum DIRECTION on
            # top of the replacement/cash MAGNITUDE. Off / neutral recursion => no anchors, no tilt =>
            # byte-identical to the pre-feature auction model.
            model_opponents = self.models_opponents and opponent_model == 'inferred'
            opponent_teams = [team for team in team_names if team != drafter]
            # Windowed bootstrap passes stack one complete field per snapshot along the opponent axis
            # (see the draft branch); outside the windowed bootstrap this is a single None entry = the
            # agent's committed field.
            field_snapshots = ((self._bootstrap_field_snapshots or [None]) if model_opponents else [None])
            field_copies    = len(field_snapshots)

            def seat_diff(team, anchor_map, committed_diffs, anchor_override=None):
                """Auction diff of the drafter vs one opponent seat, punt tilt included. Returns
                (column, tilt): the (1, n_cat, 1) diff and the seat's expected-future-tilt vector
                (n_cat,), tracked separately so the display can attribute it to the future
                differential (zeros when the seat has no modelled tilt).
                anchor_map/committed_diffs carry one field snapshot's seat predictions and tilts;
                anchor_override forces the spare anchor (self-exclusion's whole-seat swap)."""
                roster = [p for p in player_assignments[team] if p == p]
                anchor = anchor_override if anchor_override is not None else anchor_map.get(team)
                if roster:
                    roster_sum = np.array(self.x_scores.loc[roster].sum(axis=0))
                    seat_cash  = cash_remaining_per_team[team]
                    roster_len = len(roster)
                    state      = self._team_states.get(team)
                    mu_edge    = None if (roster_len >= self.n_picks or state is None) else state['mu_edge']
                elif model_opponents and anchor is not None:
                    # Seed the anchor's STATS (so the field does not lurch when it is actually bought) but do
                    # NOT charge for it: on an empty board nobody has spent, so keeping every seat at its real
                    # cash keeps money_diff symmetric (= 0). Real cash differences appear on their own as
                    # players are actually bought. Charging only the opponents (not the drafter) was what
                    # added a flat pro-drafter money block and flattened the field.
                    roster_sum = np.array(self.x_scores.loc[[anchor]].sum(axis=0))
                    seat_cash  = cash_remaining_per_team[team]
                    roster_len = 1
                    mu_edge    = committed_diffs.loc[anchor].to_numpy()
                else:
                    roster_sum = np.zeros(self.n_categories)
                    seat_cash  = cash_remaining_per_team[team]
                    roster_len = 0
                    mu_edge    = None
                base = self.get_diff_means_auction(
                    x_self_sum.reshape(1, self.n_categories, 1) - roster_sum.reshape(1, self.n_categories, 1),
                    cash_remaining_per_team[drafter] - seat_cash,
                    len(my_players) - roster_len,
                    category_value_per_dollar,
                    replacement_value_by_category,
                )
                if mu_edge is not None:
                    # opponent_model_confidence scales how sharply this seat is expected to pursue its punts.
                    tilt = (self.n_picks - roster_len) * self.opponent_model_confidence * mu_edge
                    base = base - tilt.reshape(1, self.n_categories, 1)
                    return base, np.asarray(tilt).reshape(self.n_categories)
                return base, np.zeros(self.n_categories)

            # On an all-empty board every seat is the same formula over a different (anchor, tilt) row,
            # so a snapshot's whole block batches into a few array operations instead of a Python call
            # per seat — the per-seat path costs ~2s of populate through thousands of tiny pandas
            # lookups once the windowed passes stack the field (11 columns on pass one, 176 by the
            # last). Any seat with a real roster (mid-draft) drops back to the per-seat construction.
            all_seats_empty = all(not any(p == p for p in player_assignments[team])
                                  for team in opponent_teams)
            drafter_cash = cash_remaining_per_team[drafter]
            seat_cash_diffs = np.array([drafter_cash - cash_remaining_per_team[team]
                                        for team in opponent_teams])

            field_blocks, tilt_blocks, snapshot_exclusions = [], [], []
            for snapshot in field_snapshots:
                committed_diffs = (self._player_committed_future_diffs if snapshot is None else snapshot[0])
                anchor_order    = (None if snapshot is None else snapshot[1])
                empty_seat_anchor, spare_anchor = (
                    self._assign_empty_seat_anchors(player_assignments, drafter, anchor_order)
                    if model_opponents else ({}, None))
                if all_seats_empty and len(empty_seat_anchor) == len(opponent_teams):
                    # The spare (self-exclusion's replacement seat) batches with the real seats: it is
                    # the same formula, and on an all-empty board every seat shares one drafter-vs-seat
                    # cash diff (no opponent has spent), so the spare's column is seat-independent and
                    # rides along as one extra row that is sliced off below. The uniformity check is a
                    # guard for the impossible-in-practice case of unequal empty-seat cash.
                    seats_share_cash = bool(np.all(seat_cash_diffs == seat_cash_diffs[0]))
                    batch_spare  = spare_anchor is not None and seats_share_cash
                    anchors      = [empty_seat_anchor[team] for team in opponent_teams]
                    batch_names  = anchors + ([spare_anchor] if batch_spare else [])
                    anchor_stats = self.x_scores.loc[batch_names].to_numpy()         # (S(+1), n_cat)
                    anchor_tilts = committed_diffs.loc[batch_names].to_numpy()       # (S(+1), n_cat)
                    batch_cash   = (np.append(seat_cash_diffs, seat_cash_diffs[0])
                                    if batch_spare else seat_cash_diffs)
                    # Mirrors seat_diff's anchored-empty-seat arithmetic exactly (same operation
                    # order, so results are bit-identical): get_diff_means_auction at roster_len=1,
                    # then the confidence-scaled punt tilt.
                    score_diff        = x_self_sum.reshape(1, -1) - anchor_stats
                    player_diff_total = (len(my_players) - 1 - 1) * replacement_value_by_category.reshape(1, -1)
                    money_diff_total  = batch_cash.reshape(-1, 1) * np.asarray(category_value_per_dollar).reshape(1, -1)
                    tilt              = (self.n_picks - 1) * self.opponent_model_confidence * anchor_tilts
                    columns           = score_diff - player_diff_total + money_diff_total - tilt
                    field_blocks.append(columns[:len(anchors)].T.reshape(1, self.n_categories, len(anchors)))
                    tilt_blocks.append(tilt[:len(anchors)].T.reshape(1, self.n_categories, len(anchors)))
                    spare_column = columns[len(anchors)] if batch_spare else None
                    spare_tilt   = tilt[len(anchors)] if batch_spare else None
                else:
                    seat_results = [seat_diff(team, empty_seat_anchor, committed_diffs)
                                    for team in opponent_teams]
                    field_blocks.append(np.concatenate([column for column, _ in seat_results], axis=2))
                    tilt_blocks.append(np.stack([seat_tilt for _, seat_tilt in seat_results], axis=1)
                                       .reshape(1, self.n_categories, len(seat_results)))
                    spare_column = None   # computed lazily via seat_diff in the exclusion loop
                    spare_tilt   = None
                snapshot_exclusions.append((empty_seat_anchor, spare_anchor, committed_diffs,
                                            spare_column, spare_tilt))
            diff_means = np.concatenate(field_blocks, axis=2)                     # (1, n_cat, n_teams*K)
            # Each opponent column's expected-future-tilt vector, tracked in lockstep with
            # diff_means (which carries it with opposite sign) so the display can attribute
            # it to the future differential. Zeros with the opponent model off.
            opponent_tilt_columns = np.concatenate(tilt_blocks, axis=2)           # (1, n_cat, n_teams*K)

            # Self-exclusion: a candidate that is itself an empty seat's predicted first buy is not scored
            # against a copy of itself -- rebuild that seat with the spare anchor (whole seat: stats, cost,
            # tilt). Makes the returned diff_means candidate-dependent (N, n_cat, n_opp). Applied per
            # snapshot block — each snapshot has its own seat map, spare anchor, and tilt table.
            if model_opponents and candidate_batch is not None:
                for snapshot_index, (empty_seat_anchor, spare_anchor, committed_diffs,
                                     spare_column, spare_tilt) \
                        in enumerate(snapshot_exclusions):
                    if spare_anchor is None or not empty_seat_anchor:
                        continue
                    anchor_to_slot = {anchor: opponent_teams.index(team)
                                      for team, anchor in empty_seat_anchor.items()}
                    affected = {i: opponent_teams[anchor_to_slot[p]]
                                for i, p in enumerate(candidate_batch) if p in anchor_to_slot}
                    if not affected:
                        continue
                    if diff_means.shape[0] == 1:
                        diff_means = np.repeat(diff_means, len(candidate_batch), axis=0)  # (N, n_cat, n_opp)
                        opponent_tilt_columns = np.repeat(opponent_tilt_columns, len(candidate_batch), axis=0)
                    slot_offset = snapshot_index * len(opponent_teams)
                    spare_seat_by_team = {}
                    for candidate_index, team in affected.items():
                        if team not in spare_seat_by_team:
                            # Fast path precomputed the spare column (seat-independent when every
                            # empty seat shares one cash diff); otherwise compute it per seat.
                            if spare_column is not None:
                                spare_seat_by_team[team] = (spare_column, spare_tilt)
                            else:
                                spare_base, spare_seat_tilt = seat_diff(
                                    team, empty_seat_anchor, committed_diffs,
                                    anchor_override=spare_anchor)
                                spare_seat_by_team[team] = (spare_base[0, :, 0], spare_seat_tilt)
                        slot = anchor_to_slot[candidate_batch[candidate_index]] + slot_offset
                        column_values, column_tilt = spare_seat_by_team[team]
                        diff_means[candidate_index, :, slot] = column_values
                        opponent_tilt_columns[candidate_index, :, slot] = column_tilt

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

            # Rational-opponent modelling: each opponent's future picks carry their own punt tilt, so we
            # add picks_left * mu_edge to their totals (mu_edge = the expected per-pick edge over a generic
            # pick, mirroring the edge the objective already adds for our own remaining picks). Off, or on an
            # inner inference solve (opponent_model='neutral'), the opponents stay category-neutral and the
            # construction below is byte-identical to the pre-feature model.
            model_opponents = self.models_opponents and opponent_model == 'inferred'
            opponent_teams = [team for team in team_names if team != drafter]
            # Windowed bootstrap passes stack one complete field per snapshot along the opponent axis
            # (n_teams * K columns; the objective's mean over that axis weights snapshots uniformly).
            # Outside the windowed bootstrap this is a single None entry = the agent's committed field.
            field_snapshots = ((self._bootstrap_field_snapshots or [None]) if model_opponents else [None])
            field_copies    = len(field_snapshots)
            if model_opponents:
                # On an all-empty board every seat is anchor stats + generic padding + its punt tilt,
                # so a snapshot's block batches into a few array operations instead of a pandas lookup
                # per seat (the windowed passes stack up to n_teams*K columns — the per-seat path was
                # a measured chunk of populate time). Mixed rosters fall back to compute_opponent_totals.
                all_seats_empty = all(not any(p == p for p in player_assignments[team])
                                      for team in opponent_teams)
                mean_extra_array = np.asarray(mean_extra)
                field_blocks, tilt_blocks, snapshot_exclusions = [], [], []
                for snapshot in field_snapshots:
                    committed_diffs = (self._player_committed_future_diffs if snapshot is None else snapshot[0])
                    anchor_order    = (None if snapshot is None else snapshot[1])
                    if all_seats_empty:
                        empty_slot_player, spare_anchor = self._assign_empty_seat_anchors(
                            player_assignments, drafter, anchor_order)
                    if all_seats_empty and len(empty_slot_player) == len(opponent_teams):
                        # Mirrors team_total's anchored-empty-seat arithmetic exactly (same operation
                        # order as get_opposing_team_means + the tilt, so results are bit-identical).
                        anchors      = [empty_slot_player[team] for team in opponent_teams]
                        anchor_stats = self.x_scores.loc[anchors].to_numpy()          # (S, n_cat)
                        anchor_tilts = committed_diffs.loc[anchors].to_numpy()        # (S, n_cat)
                        extra_sum    = mean_extra_array.reshape(1, -1) * (target_team_size - 1)
                        tilt         = ((self.n_picks - 1) * self.opponent_model_confidence) * anchor_tilts
                        totals_rows  = (anchor_stats + extra_sum) + tilt
                        field_blocks.append(totals_rows.T.reshape(1, self.n_categories, len(anchors)))
                        tilt_blocks.append(tilt.T.reshape(1, self.n_categories, len(anchors)))
                        if spare_anchor is not None:
                            spare_tilt  = (((self.n_picks - 1) * self.opponent_model_confidence)
                                           * committed_diffs.loc[spare_anchor].to_numpy())
                            spare_total = ((self.x_scores.loc[[spare_anchor]].sum(axis=0).to_numpy()
                                            + extra_sum.reshape(-1))
                                           + spare_tilt
                                           ).reshape(1, self.n_categories, 1)
                        else:
                            spare_total = None
                            spare_tilt  = None
                    else:
                        totals, seat_tilts, empty_slot_player, spare_total, spare_tilt = \
                            self.compute_opponent_totals(
                                player_assignments, drafter, mean_extra, target_team_size,
                                field_snapshot=snapshot)
                        field_blocks.append(np.concatenate([totals[team] for team in opponent_teams], axis=2))
                        tilt_blocks.append(np.stack([seat_tilts[team] for team in opponent_teams], axis=1)
                                           .reshape(1, self.n_categories, len(opponent_teams)))
                    snapshot_exclusions.append((empty_slot_player, spare_total, spare_tilt))
                other_team_sums = np.concatenate(field_blocks, axis=2)                # (1, n_cat, n_teams*K)
                # Each opponent column's expected-future-tilt vector, in lockstep with the totals
                # (which carry it additively, so diff_means below carries it with opposite sign).
                opponent_tilt_columns = np.concatenate(tilt_blocks, axis=2)           # (1, n_cat, n_teams*K)
            else:
                snapshot_exclusions = [({}, None, None)]
                other_team_sums = np.concatenate([
                    self.get_opposing_team_means(player_assignments[team], mean_extra, target_team_size)
                    for team in opponent_teams
                ], axis=2)                                                            # (1, n_cat, n_opp)
                opponent_tilt_columns = np.zeros_like(other_team_sums)
            diff_means = x_self_sum.reshape(1, self.n_categories, 1) - other_team_sums  # (1, n_cat, n_opp)

            # Self-exclusion: when the candidate being scored is itself one of the predicted opponent teams,
            # swap that seat for the spare predicted team, so a player is never evaluated against a copy of
            # itself. This makes diff_means candidate-dependent (N, n_cat, n_opp) for the affected
            # candidates. Applied per snapshot block — each snapshot has its own seat map and spare.
            if candidate_batch is not None:
                for snapshot_index, (empty_slot_player, spare_total, spare_tilt) \
                        in enumerate(snapshot_exclusions):
                    if spare_total is None or not empty_slot_player:
                        continue
                    slot_offset = snapshot_index * len(opponent_teams)
                    player_to_slot = {player: opponent_teams.index(team) + slot_offset
                                      for team, player in empty_slot_player.items()}
                    affected = {i: player_to_slot[p]
                                for i, p in enumerate(candidate_batch) if p in player_to_slot}
                    if not affected:
                        continue
                    spare_column = (x_self_sum.reshape(self.n_categories)
                                    - spare_total.reshape(self.n_categories))
                    if diff_means.shape[0] == 1:
                        diff_means = np.repeat(diff_means, len(candidate_batch), axis=0)  # (N, n_cat, n_opp)
                        opponent_tilt_columns = np.repeat(opponent_tilt_columns, len(candidate_batch), axis=0)
                    for candidate_index, slot in affected.items():
                        diff_means[candidate_index, :, slot] = spare_column
                        opponent_tilt_columns[candidate_index, :, slot] = spare_tilt

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

        if field_copies > 1:
            # Windowed bootstrap: repeat the per-team variance block once per stacked field snapshot so
            # every opponent column has its variance (the empty board makes the blocks identical).
            diff_vars = np.tile(diff_vars, (1, field_copies))
        diff_vars = diff_vars.reshape(1, self.n_categories, -1)

        # Mean expected future tilt of the field, aligned to diff_means' candidate rows (self-exclusion
        # swaps affect both identically). diff_means carries this with opposite sign, so the display's
        # 'Current diff' (= Diff - Future-Diff) sheds it once 'Future-Diff' subtracts it — a pure
        # reattribution between rows; the objective sums both terms and is untouched.
        opponent_future_tilts = opponent_tilt_columns.mean(axis=2)               # (1|N, n_cat)

        return diff_means, diff_vars, sigma_2_m, opponent_future_tilts

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

    def _assign_empty_seat_anchors(self, player_assignments, drafter, anchor_player_order=None):
        """Assign each EMPTY opponent seat its predicted first player: the best available committed anchor,
        skipping any already drafted and any claimed by an earlier empty seat. Shared by the draft
        (compute_opponent_totals) and the auction (get_diff_distributions) so the seat->anchor mapping is
        identical in both. Returns (empty_seat_anchor, spare_anchor): empty_seat_anchor maps each empty seat
        to its predicted player (a seat is absent when no anchor is left); spare_anchor is the next unclaimed
        anchor, swapped in by self-exclusion when the candidate being scored is itself a predicted first
        pick. Both come back empty when there is no bootstrap / no anchors remain. anchor_player_order
        substitutes a windowed-bootstrap snapshot's ordering for the agent's committed one."""
        if anchor_player_order is None:
            anchor_player_order = self._anchor_player_order
        players_chosen = {x for roster in player_assignments.values() for x in roster if x == x}
        anchor_players = ([] if anchor_player_order is None
                          else list(anchor_player_order))
        anchor_cursor  = 0

        empty_seat_anchor = {}
        for team, roster_players in player_assignments.items():
            if team == drafter or any(p == p for p in roster_players):   # skip me and any non-empty seat
                continue
            while anchor_cursor < len(anchor_players) and anchor_players[anchor_cursor] in players_chosen:
                anchor_cursor += 1
            if anchor_cursor < len(anchor_players):
                empty_seat_anchor[team] = anchor_players[anchor_cursor]
                anchor_cursor += 1

        spare_anchor = None
        while anchor_cursor < len(anchor_players):
            spare = anchor_players[anchor_cursor]
            anchor_cursor += 1
            if spare not in players_chosen:
                spare_anchor = spare
                break
        return empty_seat_anchor, spare_anchor

    def compute_opponent_totals(self, player_assignments, drafter, mean_extra, target_team_size,
                                field_snapshot=None):
        """Full expected-team vector per opponent, (1, n_categories, 1) each. An opponent that has drafted
        is its real roster + picks_left * mu_edge (its inferred/committed build — a zero-sum punt tilt that
        reshapes future picks without changing net strength). An EMPTY opponent is modelled as if it had
        already drafted its predicted starting player X (see _assign_empty_seat_anchors): roster [X] + X's
        committed future. Pricing X's real stats in now means that when the seat actually drafts X the field
        does not move.

        Returns (totals, seat_tilts, empty_slot_player, spare_total, spare_tilt). seat_tilts maps each
        opponent to its expected-future-tilt vector (n_cat; zeros when untilted) — the tilt is inside
        totals additively, and the display attributes it to the future differential, so it is also
        exposed separately. empty_slot_player maps each empty seat to its predicted player;
        spare_total/spare_tilt are the next predicted team's, swapped in for self-exclusion when the
        candidate being scored is itself one of those predicted players (so nobody drafts a copy of it).
        field_snapshot = (committed_future_diffs, anchor_player_order) reads one windowed-bootstrap
        snapshot's field instead of the agent's committed one (None outside the windowed bootstrap)."""
        if field_snapshot is None:
            committed_future_diffs, anchor_player_order = self._player_committed_future_diffs, None
        else:
            committed_future_diffs, anchor_player_order = field_snapshot
        empty_seat_anchor, spare_anchor = self._assign_empty_seat_anchors(player_assignments, drafter,
                                                                          anchor_player_order)

        def team_total(roster, mu_edge):
            base = self.get_opposing_team_means(roster, mean_extra, target_team_size)
            if mu_edge is None:
                return base, np.zeros(self.n_categories)
            picks_left = self.n_picks - len([p for p in roster if p == p])
            # opponent_model_confidence scales how sharply this seat is expected to pursue its punts.
            tilt = picks_left * self.opponent_model_confidence * mu_edge
            return base + tilt.reshape(1, self.n_categories, 1), np.asarray(tilt).reshape(self.n_categories)

        totals, seat_tilts = {}, {}
        for team in player_assignments:
            if team == drafter:
                continue
            roster = [p for p in player_assignments[team] if p == p]
            if len(roster) >= self.n_picks:
                totals[team], seat_tilts[team] = team_total(roster, None)
            elif roster:
                state = self._team_states.get(team)
                totals[team], seat_tilts[team] = team_total(roster, None if state is None else state['mu_edge'])
            elif team in empty_seat_anchor:
                predicted = empty_seat_anchor[team]
                totals[team], seat_tilts[team] = team_total([predicted],
                                                            committed_future_diffs.loc[predicted].to_numpy())
            else:
                totals[team], seat_tilts[team] = team_total(roster, None)   # no prediction left: neutral padding

        spare_total, spare_tilt = ((None, None) if spare_anchor is None
                                   else team_total([spare_anchor],
                                                   committed_future_diffs.loc[spare_anchor].to_numpy()))
        return totals, seat_tilts, empty_seat_anchor, spare_total, spare_tilt

    def refresh_stale_team_states(self, player_assignments, drafter, cash_remaining_per_team=None):
        """Bring EVERY seat whose stored build no longer matches its roster up to date — including the
        drafter's own seat, so the store stays a pure function of the rosters (any evaluate leaves it in
        the same state, no matter whose perspective ran first). First picks of priced-in players reuse
        their committed build (a table lookup); everything else replays the seat's most recent pick
        through the real solver (infer_opponent_category_weights) — a single-candidate solve, cheap
        enough that refresh is eager and unbudgeted (a pick makes exactly one seat stale). Empty seats
        use round-one archetypes, and full seats have no future picks, so neither is refreshed here.
        cash_remaining_per_team (auction only) makes the inference run auction H-scoring."""
        stale = []
        for team, roster_players in player_assignments.items():
            roster = [p for p in roster_players if p == p]
            if not roster or len(roster) >= self.n_picks:
                continue
            state = self._team_states.get(team)
            if state is None or state['roster_key'] != frozenset(roster):
                stored_size = 0 if state is None else len(state['roster_key'])
                stale.append((len(roster) - stored_size, team, roster))

        needs_inference = []
        for staleness, team, roster in sorted(stale, key=lambda item: item[0], reverse=True):
            committed = (None if self._player_committed_future_diffs is None or len(roster) != 1
                         or roster[0] not in self._player_committed_future_diffs.index
                         else self._player_committed_future_diffs.loc[roster[0]].to_numpy())
            if committed is None:
                needs_inference.append((staleness, team, roster))
                continue
            # First pick of a priced-in player: reuse its committed build verbatim so the field does
            # not move. The entry carries the anchor's FULL identity — tilt, weights, and flex shares
            # all come from the same serve solve — so later inference replays warm-start from the
            # team's identity ("the Wemby team that added Duren") instead of re-deriving the team from
            # its newest pick, which let each seat flip its whole strategy every round.
            anchor_shares = self._player_frozen_shares
            self._team_states[team] = {
                'roster_key'       : frozenset(roster),
                'category_weights' : self._player_frozen_weights.loc[roster[0]].to_numpy(),
                'mu_edge'          : committed,
                'position_shares'  : (
                    {pos_code: frame.loc[roster[0]].to_numpy() for pos_code, frame in anchor_shares.items()}
                    if anchor_shares is not None
                    and all(frame is not None for frame in anchor_shares.values()) else None
                ),
            }

        for _, team, roster in needs_inference:
            category_weights, mu_edge, position_shares = self.infer_opponent_category_weights(
                team, roster, player_assignments, cash_remaining_per_team
            )
            self._team_states[team] = {
                'roster_key'       : frozenset(roster),
                'category_weights' : category_weights,
                'mu_edge'          : mu_edge,
                'position_shares'  : position_shares,
            }

    def infer_opponent_category_weights(self
                                        , opponent_team_name
                                        , opponent_players
                                        , player_assignments
                                        , cash_remaining_per_team=None):
        """Infer an opponent's build by replaying the pick they just made through the REAL solver: their
        roster minus that pick as the team, that pick as the sole candidate — reproducing what they were
        thinking when they made it. The nested solve is a normal get_h_scores run (same seeding, warm
        starts, kappa, and position machinery as any evaluate) against the current stored field;
        _opponent_inference_active caps the recursion at one level of best-response and routes the entry
        write through the refresh loop. The replay warm-starts from this team's entry at the polish
        tier (the entry is keyed to exactly the replay view's roster), so the result is the team's
        identity plus a small per-pick evidence nudge — see the tier comment in get_h_scores.
        Single-candidate solves cost ~1/40th of an evaluate, which is what makes eager refresh viable.
        Returns (category_weights, mu_edge, position_shares); mu_edge is per remaining pick."""
        drafted_player = opponent_players[-1]
        prior_players  = opponent_players[:-1]
        picks_left     = self.n_picks - len(opponent_players)
        assignments_view = {**player_assignments, opponent_team_name: prior_players}

        # No other isolation is needed: every remaining piece of shared per-evaluate state the nested
        # solve touches (self.players, _position_mode, throttle priority/cache/offset, warm-start rows)
        # is assigned fresh by the outer evaluate AFTER the refresh completes — see the refresh call's
        # ordering note in get_h_scores. The replay deliberately keeps the normal warm-start tiers: the
        # eager refresh keeps each entry keyed to exactly this view's roster, so the polish tier applies
        # and the inferred build is the team's identity plus a small per-pick evidence nudge. Running
        # replays at the full warm rate instead let every seat re-derive itself each round and sprint to
        # the field's one open punt lane — a best-response herd (measured: six seats flipping to the
        # same assists punt) that no in-draft mechanism damps.
        self._opponent_inference_active = True
        try:
            result = self.get_h_scores(
                player_assignments      = assignments_view,
                drafter                 = opponent_team_name,
                n_iterations            = _OPPONENT_INFERENCE_ITERATIONS,
                cash_remaining_per_team = cash_remaining_per_team,
                candidate_subset        = [drafted_player],
            )
        finally:
            self._opponent_inference_active = False

        category_weights = result['Weights'].loc[drafted_player].to_numpy()
        # Future-Diff sums the per-pick edge over the picks remaining AFTER the replayed pick joins,
        # which is exactly picks_left — so dividing recovers the per-pick mu_edge the field model
        # multiplies back by picks_left.
        mu_edge = (result['Future-Diff'].loc[drafted_player].to_numpy() / picks_left
                   if picks_left > 0 else np.zeros(self.n_categories))
        shares_frames   = result['Position-Shares']
        position_shares = ({pos_code: frame.loc[drafted_player].to_numpy()
                            for pos_code, frame in shares_frames.items()}
                           if all(frame is not None for frame in shares_frames.values()) else None)
        return category_weights, mu_edge, position_shares

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

    # NOTE: the money->win-probability curve (get_value_of_money_auction, stored as
    # self.value_of_money) was removed 2026-08-11: nothing consumed it -- the service layer converts
    # H-scores to dollars via auction_value_adjuster -- and rebuilding a 2000-price-point tensor
    # against every opponent column on every auction call was a measured ~1.7s of populate time.

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
        neutral      = self.get_starting_category_weights()
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
                           , candidate_offset=0
                           , opponent_future_tilts=None):

        # Stale correction terms must never leak across boards or candidate batches.
        self._correction_cache = None

        # Warm-started rows descend at reduced, staleness-tiered rates (see the warm-start scale
        # constants); cold rows keep the full rate. Per-row rates broadcast through Adam's update, and
        # apply to BOTH optimisers -- weights and flex shares are paired halves of one build, so they
        # must move (and hold still) together.
        warm_start_row_rate_scales = self._warm_start_row_rate_scales
        self._warm_start_row_rate_scales = None
        if warm_start_row_rate_scales is not None:
            rate_column            = warm_start_row_rate_scales.reshape(-1, 1)
            category_learning_rate = 0.001 * rate_column
            shares_learning_rate   = _SHARES_LEARNING_RATE * rate_column
        else:
            category_learning_rate = 0.001
            shares_learning_rate   = _SHARES_LEARNING_RATE

        optimizers = {
            'Categories': AdamOptimizer(learning_rate=category_learning_rate),
            # Shares are optimised in softmax-logit space (see the update below). A logit must travel
            # ~±4 to move a share across most of [0, 1], versus a direct step in share space, so the
            # logit learning rate is an order of magnitude larger than the old share-space 0.01.
            # One optimiser for the shared per-position logits (every flex slot's softmax reads them).
            'Shares': AdamOptimizer(learning_rate=shares_learning_rate),
        }

        # MLB: unsupported and unreachable — see the MLB note in __init__.
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
                n_total_picks          = sum(self.position_config.position_numbers.values())
                candidate_player_array = get_player_rows(self.positions.loc[result_index], self.position_config)
                team_so_far_array      = (get_player_rows(self.positions.loc[self.players], self.position_config)
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
                # Mixed warm start: rows with stored converged weights (set by get_h_scores when the
                # frozen table only partially covers the batch) override their cold seeds.
                if self._partial_warm_start_rows is not None:
                    covered_positions, covered_weights = self._partial_warm_start_rows
                    category_weights = category_weights.copy()
                    category_weights[covered_positions] = covered_weights
                    self._partial_warm_start_rows = None

            # Robustness regulariser: soft-threshold the category weights toward neutral v each iteration
            # (proximal step for an L1 penalty -lambda*||w-v||_1), strongest on an empty roster and
            # decaying to 0 by mid-draft (see self.reg_schedule), so early picks stay flexible.
            reg_lambda  = (self.reg_schedule[len(self.players)]
                           if len(self.players) < len(self.reg_schedule) else 0.0)
            # Warm-started rows skip the regulariser: it is a COLD-start robustness device, and on a warm
            # start it drags an already-converged equilibrium build back toward neutral -- worse, at the
            # reduced warm-start step sizes its pull exceeds the descent step, snapping weights exactly
            # onto the singular w==v ray (0/0 -> NaN gradients). Cold rows keep the full schedule.
            if warm_start_row_rate_scales is not None:
                cold_row_multiplier = (warm_start_row_rate_scales >= 1.0).astype(float).reshape(-1, 1)
            else:
                cold_row_multiplier = 1.0
            category_reg_lambda = reg_lambda * cold_row_multiplier
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
                    # ray. Renormalised to the simplex just below. category_reg_lambda is per-row: zero
                    # for warm-started rows (see above), the full schedule for cold rows.
                    deviation        = category_weights - neutral_row
                    shrink           = np.minimum(category_reg_lambda, np.maximum(np.abs(deviation) - _REG_FLOOR, 0.0))
                    category_weights = category_weights - np.sign(deviation) * shrink
                category_weights[category_weights < 0] = 0

                if self.sport == 'NBA':
                    category_weights = category_weights / category_weights.sum(axis=1).reshape(-1, 1)
                # MLB: unsupported and unreachable — see the MLB note in __init__.
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
                        # Warm rows are exempt, exactly like the category weights (paired build halves).
                        pos_reg       = reg_lambda * _POSITION_REG_MULT * cold_row_multiplier
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

        # 'Future-Diff' stays the drafter's RAW projected future tilt: the opponent model
        # feeds on it (committed per-player builds, mu_edge inference), so it must not be
        # netted here. The opponents' expected future tilts ride along as their own entry;
        # the DISPLAY subtracts them from the future row (see _build_g_score_rows) so the
        # expand view's 'Current diff' sheds behaviour that hasn't happened yet.
        future_diff_df = (
            pd.DataFrame(expected_future_diff.mean(axis=2),
                         index=result_index, columns=self.x_scores.columns)
            if expected_future_diff is not None else None
        )
        opponent_future_tilt_df = (
            pd.DataFrame(np.broadcast_to(opponent_future_tilts,
                                         (len(result_index), self.n_categories)),
                         index=result_index, columns=self.x_scores.columns)
            if expected_future_diff is not None and opponent_future_tilts is not None else None
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
            'Opponent-Future-Tilt': opponent_future_tilt_df,
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
                # One plane per opponent column: n_drafters-1 normally, n_teams*K during windowed passes.
                for i in range(cdf_estimates.shape[2])
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
                self.position_config,
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

        # MLB: unsupported and unreachable — see the MLB note in __init__.
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

        # MLB: unsupported and unreachable — see the MLB note in __init__.
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

    def get_starting_category_weights(self) -> np.ndarray:
        """Where a descent starts: the neutral weight vector.

        Neutral already accounts for a tiebreaker — v is the value of a category per unit of
        x-score, and process_player_data scales the tiebreaker's by (1 + most_categories_weight)
        there, which is the one place that keeps the G-scores, the punt-depth reference and the
        field weights telling the same story. So there is nothing to add on top here, and adding
        it would apply the factor twice.
        """
        return self.v.reshape(self.n_categories).copy()

    def get_objective_and_pdf_weights(self
                                       , x_diff_array
                                       , diff_vars
                                       , cdf_estimates
                                       , pdf_estimates
                                       , sigma_2_m=None
                                       , calculate_pdf_weights=False
                                       , correction_mode='full'
                                       , iteration=None):
        """The objective for this session's format, and (optionally) its gradient with respect
        to the category differentials.

        Head to Head is a spectrum rather than two formats: most_categories_weight is how much
        of the objective is the probability of taking the majority, the rest being the average
        per-category win probability. Both ends are probabilities in [0, 1] and both gradients
        are per-category partials of the same shape, so a weight between them is a convex
        combination of the two. The endpoints are computed alone — at weight 0 there is no
        reason to run the win-count DP, and at 1 no reason to average the categories.
        """
        if self.scoring_format == 'Rotisserie':
            return self.get_objective_and_pdf_weights_rotisserie(
                x_diff_array, diff_vars, cdf_estimates, pdf_estimates,
                sigma_2_m, calculate_pdf_weights)

        weight = self.most_categories_weight
        if weight == 1.0:
            return self.get_objective_and_pdf_weights_mc(x_diff_array, diff_vars,
                                                          cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights,
                                                          correction_mode, iteration)
        if weight == 0.0:
            return self.get_objective_and_pdf_weights_ec(cdf_estimates, pdf_estimates,
                                                          calculate_pdf_weights)

        most_categories = self.get_objective_and_pdf_weights_mc(
            x_diff_array, diff_vars, cdf_estimates, pdf_estimates,
            calculate_pdf_weights, correction_mode, iteration)
        each_category = self.get_objective_and_pdf_weights_ec(
            cdf_estimates, pdf_estimates, calculate_pdf_weights)

        if not calculate_pdf_weights:
            return (1 - weight) * each_category + weight * most_categories
        return ((1 - weight) * each_category[0] + weight * most_categories[0],
                (1 - weight) * each_category[1] + weight * most_categories[1])

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
        if calculate_pdf_weights:
            # Gradient steps need the tipping points too, and both come out of one
            # shared win-count DP build (bit-identical to the standalone functions).
            win_probability, tipping_points = calculate_win_probability_and_tipping_points(
                probs, self.tiebreaker_index)
        else:
            win_probability = compute_win_probability(probs, self.tiebreaker_index)  # (n_players, n_opponents)

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

        pdf_weights = tipping_points * pdf_estimates
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
        """Average per-category win probability, and its exact gradient.

        The consumer contracts only the category axis, so a returned weight must be the
        objective's partial with respect to one category's differential, summed over opponents.
        Averaging over opponents supplies that axis' 1/n_opponents; the division below supplies
        the 1/n_categories that the objective's category mean introduces and the per-category
        partials do not carry. Without it these weights are the gradient of the SUM over
        categories -- n_categories times too large, which is invisible under Adam on its own but
        would silently dominate any blend with Most Categories, whose weights are exact.
        """
        objective = cdf_estimates.mean(axis=2).mean(axis=1)
        if calculate_pdf_weights:
            return objective, pdf_estimates.mean(axis=2) / self.n_categories
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

    def reset_draft_state(self):
        """Clear all in-draft mutable state, returning the agent to its just-populated condition: the
        rolling per-team store (inferred opponents + own-build entries) and any injected start weights.
        Populate artifacts survive -- the frozen player tables, anchor order, default result and punt
        popularity are properties of the BUILD, not of a draft. Call this at the start of every fresh
        draft, simulation, or test so nothing leaks across boards (harnesses that poked private
        attributes for this went silently stale when the attributes were renamed)."""
        self._team_states                = {}
        self._partial_warm_start_rows    = None
        self._warm_start_row_rate_scales = None
        return self.clear_initial_weights()

    def populate_default_h_scores(self, n_iterations: int, cash_remaining_per_team: dict = None) -> None:
        """Compute and cache the base (empty-board) H-scores. Run once at the end of the build so the
        agent is always primed — the throttle has a ranking to prioritise by, auction values have their
        anchor, and the draft-start evaluate can short-circuit to this result. Pass full cash only in
        auction mode.

        With opponent modelling on this is a two-pass committed-anchor build. Pass 1 solves every
        candidate against a neutral field; each candidate's Future-Diff is the committed build of a team
        drafted around THAT player (its per-pick mu_edge), stored per player. Pass 2 serves the base
        H-scores against that committed field. No averaging — the anchors are real per-player builds, so
        (a) player-specific punts survive, and (b) they match the from-scratch inference verbatim, so
        drafting a priced-in player confirms the prediction rather than moving the field. The bootstrap
        runs for auctions too (full cash) — the passes solve in auction mode, so the committed anchors are
        auction-shaped (top-heavy) rather than borrowed from the draft. Off, it is a single neutral pass."""
        empty = {f'Team {i + 1}': [] for i in range(self.n_drafters)}
        # Fresh build: drop any prior cache so the passes recompute (the empty-board short-circuit and the
        # throttle both key off these) instead of short-circuiting to stale results.
        self._team_states              = {}
        self._anchor_player_order   = None
        self._player_committed_future_diffs  = None
        self._player_frozen_weights    = None
        self._player_frozen_shares     = None
        self._default_result           = None
        self.default_h_scores          = None
        self._populate_pass_scores     = None
        self._bootstrap_field_snapshots = None

        bootstrap = (self.models_opponents and self.n_picks > 1)
        if not bootstrap:
            result = self._run_bootstrap_pass(empty, n_iterations, cash_remaining_per_team)
            self.default_h_scores = result['Scores'].sort_values(ascending=False)
            self._default_result  = result
            self._populate_pass_scores = None   # populate-scoped; default_h_scores ranks from here on
            return

        # Damped best-response to a fixed point. Pass 0 solves against a neutral field; each later pass
        # solves against the CURRENT committed field and smooths it a fraction alpha toward that pass's
        # builds. Exponential smoothing (not a running mean) converges to a field that is its own
        # best-response — a committed per-player equilibrium — so player-specific punts survive.
        top_count = 3 * self.n_drafters
        def refresh_field(from_result):
            self._player_committed_future_diffs = committed
            # Just the ordering (an Index): seat assignment consumes only the ranking, never the values.
            self._anchor_player_order = from_result['Scores'].sort_values(ascending=False).index[: top_count]

        # The field only needs the committed builds of the top-3N anchors, so the iteration passes score
        # just those players (~36) instead of the whole pool (~577) — the position solve is the cost and it
        # scales with the candidate count. Only the final serve pass ranks everyone.
        anchor_subset = list(self.x_scores.index[: top_count])

        pass_iters = _OPPONENT_PASS_ITERATIONS
        result    = self._run_bootstrap_pass(empty, pass_iters, cash_remaining_per_team, anchor_subset)  # neutral
        committed = result['Future-Diff'] / (self.n_picks - 1)
        window_size = (0 if self.scoring_format == 'Rotisserie' else _OPPONENT_FIELD_WINDOW)
        bootstrap_passes = (_OPPONENT_BOOTSTRAP_PASSES if self.scoring_format != 'Rotisserie'
                            else _OPPONENT_BOOTSTRAP_PASSES_ROTISSERIE)
        if window_size >= 2:
            # Windowed fictitious play (see _OPPONENT_FIELD_WINDOW): each pass best-responds to the RAW
            # fields of the last K passes stacked as separate opponents — no EMA blending, the window
            # itself provides the damping (one pass moves at most 1/K of the field's mass) while every
            # historical build keeps its specific punts. The snapshots exist only between passes; the
            # serve and everything mid-draft face the FINAL pass's single field, set after the loop.
            snapshots = []
            try:
                for _ in range(bootstrap_passes):
                    committed = result['Future-Diff'] / (self.n_picks - 1)
                    refresh_field(result)   # single-field store tracks the latest pass throughout
                    snapshots.append((committed, self._anchor_player_order))
                    self._bootstrap_field_snapshots = snapshots[-window_size:]
                    result = self._run_bootstrap_pass(empty, pass_iters, cash_remaining_per_team, anchor_subset)
            finally:
                self._bootstrap_field_snapshots = None
            committed = result['Future-Diff'] / (self.n_picks - 1)
        else:
            for _ in range(bootstrap_passes):
                refresh_field(result)
                result    = self._run_bootstrap_pass(empty, pass_iters, cash_remaining_per_team, anchor_subset)
                committed = (1 - _OPPONENT_SMOOTHING) * committed \
                            + _OPPONENT_SMOOTHING * (result['Future-Diff'] / (self.n_picks - 1))

        # Serve the full-pool base H-scores against the converged field (in-draft evaluates use it too).
        # The serve WARM-STARTS from the final iteration pass's weights where they exist -- the top-3N
        # anchors, the players whose displayed builds actually get scrutinised -- so their served builds
        # continue the bootstrap's converged builds instead of re-deriving cold (a cold re-derive lands
        # elsewhere on flat plateaus, and the first in-draft evaluate then visibly "switches" the build).
        # The rest of the pool cold-starts via the punt seed scan and may drift slightly; acceptable.
        refresh_field(result)
        # Anchor rows: warm-start source for the serve (weights and flex shares always travel together).
        self._player_frozen_weights = result['Weights']
        self._player_frozen_shares  = result['Position-Shares']
        # The serve keeps the polish tier for its warm rows IN BOTH PATHS. Under the window the final
        # pass's weights answered the stacked-history field, not the single field the serve scores
        # against — but re-converging them at the full rate against one converged field hands every
        # candidate the SAME best defection and herds the whole board into one punt (measured: 12/12
        # EC anchors punting 3s). The polish tier preserves each anchor's own equilibrium build, which
        # is exactly the diversity a mixed equilibrium consists of; the mild score-at-mixture-weights
        # inconsistency is the same approximation the EMA path has always lived with.
        result = self._run_bootstrap_pass(empty, n_iterations, cash_remaining_per_team,
                                          candidate_subset=None, preserve_frozen_weights=True)
        self.default_h_scores = result['Scores'].sort_values(ascending=False)
        self._default_result  = result
        # Freeze the per-player self-play weights AND flex shares (each player's converged build vs the
        # equilibrium field, sitting alongside the committed mu_edge tilts). FROZEN from here on:
        # empty-roster evaluates warm-start each candidate from its rows, and opponent inference seeds
        # from the row of a newly drafted player. In-draft refresh is at the TEAM level only (_team_states).
        self._player_frozen_weights = result['Weights']
        self._player_frozen_shares  = result['Position-Shares']
        self._populate_pass_scores  = None   # populate-scoped; default_h_scores ranks from here on

    def _run_bootstrap_pass(self, empty, n_iterations, cash_remaining_per_team, candidate_subset=None,
                            preserve_frozen_weights=False):
        """One empty-board base-H-score solve against the current _anchor_player_order field. Clears the
        per-pass warm start and inferred-opponent store first (the empty board has no real opponents, so
        only the committed archetype field differs between passes). candidate_subset narrows which players
        are scored — the field-building passes only need the top-2N anchors. preserve_frozen_weights keeps
        the frozen table so this pass warm-starts from it (the second serve pass, which polishes the cold
        serve's weights to stationary points)."""
        self.reset_draft_state()
        if not preserve_frozen_weights:
            self._player_frozen_weights = None
            self._player_frozen_shares  = None
        result = self.get_h_scores(
            player_assignments      = empty,
            drafter                 = 'Team 1',
            n_iterations            = n_iterations,
            cash_remaining_per_team = cash_remaining_per_team,
            candidate_subset        = candidate_subset,
        )
        # Arm the next run's position-optimiser throttle (see _populate_pass_scores). The neutral
        # first pass has no ranking and solves exact; the serve ranks the full pool by the final
        # window pass's anchor scores, with the unranked rest of the pool sorting last.
        self._populate_pass_scores = result['Scores']
        return result

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

