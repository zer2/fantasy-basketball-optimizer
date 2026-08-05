# Most-Categories Correlation: Exploration & Findings

**Branch:** `correlation-factor` (off `mc-correlation`).
**Status:** parked. This branch holds the entire correlation investigation so the shipped
opponent-model baseline (`mc-correlation`) stays clean of it. The pre-existing `MC_CORRELATION`
correction code (env-gated, **default off**) is untouched in the baseline — only the *exploration*
(oracle experiment, factor model, probit method, debug script, this note) lives here.

Two companion write-ups as artifacts (rendered math + figures):
- The factor idea: https://claude.ai/code/artifact/eb50fd28-d01f-486d-8207-fe2321572ba0
- The 2nd-order collapse and the probit fix: https://claude.ai/code/artifact/b302b750-bbcf-4e6b-8323-61cd146ce887

---

## TL;DR

- The MC objective assumes categories are independent. The `MC_CORRELATION` correction re-couples them
  with a **first-order (pairwise, linear-in-ρ)** term. It's env-gated and off by default.
- Measured against a Monte-Carlo oracle on real builds, first-order roughly **halves** the error of
  ignoring correlation, but leaves ~1pp (worst ~2pp) — decision-relevant given H-scores resolve at ~0.1%.
- A **common-factor** approach (condition on the leading eigenvector of R → conditional independence →
  a 1-D integral) is more accurate. The cheap analytic evaluation is a **probit match** of the
  resulting sigmoid `g(u)`, closed-form, ~one extra DP pass, ~**0.56pp** error (half of first-order,
  sometimes better than 9-node quadrature).
- The naive 2nd-order Taylor of `g(u)` **collapses** (31pp) — not a bug, the leading factor is too
  strong for a parabola. Verified: analytic `g''(0)` == finite-difference to the digit.
- **Unresolved and separate from accuracy:** whether correlation should ship for MC *at all*. Even an
  exact correction *flattens punting* (universal soft-Turnovers hedge, few committed punts), which may
  be strategically wrong for MC, whose measured edge comes from committed punting. The decisive test is
  an H-vs-H win-rate sim (correlation-on drafters vs correlation-off), not yet run.

---

## 1. Background: the correction and its two error sources

The MC win probability is P(win a majority of the 9 categories), where the per-category differentials
are jointly **Gaussian** with correlation matrix R (`self.rho`, from the historical correlation CSV +
aleph + turnover sign-flip). The base objective computes this assuming **independence** (an exact
Poisson-binomial DP). The correction (`calculate_correction_terms`, eq C4) adds a **first-order-in-ρ**
pairwise term: `½ Σ_{c≠d} (R−I)_cd φ(z_c) φ(z_d) B_cd`, with `B_cd` the exact leave-two-out bracket.

Two things it misses:
1. **Pairwise curvature:** the true pairwise covariance `Φ₂(z_c,z_d;ρ) − Φ Φ` is nonlinear in ρ; the
   linear term under-states it. (~10% at 70/70, growing for stronger favorites.)
2. **Higher-order coherence:** a sum over pairs cannot represent a *coherent* multi-category swing
   ("a bad week hits most categories at once"). Variance is fully pairwise, but the *shape* of the
   win-count distribution (bimodal sweep-or-swept) is not.

Both are fully *determined* by R (a Gaussian has no free higher-order moments) — they are consequences
the linearization discards, not missing information.

## 2. Audit findings

- **Rotisserie correlation handling is correct.** Verified term-by-term against the paper
  (arXiv 2501.00933v1): σ²_T (eq 6) ↔ `get_sigma_2_p`; H_T (eq 11) ↔ `get_h_p`; H_M (eq 12) ↔
  `get_h_m` with N=|O|+1; σ²_M (eq 8) ↔ `get_sigma_2_m`; μ_D/σ²_D ↔ `get_mu_d`/`get_sigma_2_d`. The
  **two-team √2 factor** is handled: `sigma_c` is the opponent-strength spread ×√2 (paper's own
  definition), and the differential z-scores normalize by the full two-team variance. Appendix A.1.2's
  three-case covariance split (full ρ same-opponent, ρ/2 cross-opponent) is exactly the
  double-count-then-halve in `h_p`. (Possible paper typo: μ_c,o "divided by 2σ" should be √2·σ; the
  code is right.)
- **aleph=0.2 pushes R off positive-semidefinite** (min eigenvalue −0.043). The correction runs on a
  matrix that isn't strictly a correlation matrix. The oracle experiment projects to the nearest PSD
  matrix (max entry shift 0.029) and uses it for every method.

## 3. The herding behavior (correction ON)

An A/B experiment (`test_mc_correlation_punt_profiles`) showed the correction, as-is, **flattens
punting**: committed hard punts collapse to ~0 and nearly every top build takes a soft **Turnovers**
lean. This is the in-code warning ("over-suppresses punting on a contested board; peaks at win-prob
0.5") confirmed at scale.

- **Mechanism (favorites hate correlated variance):** a committed punt build is a slight favorite
  across ~7 positively-correlated categories; correlation raises its win-count variance, and for a
  favorite that means correlated *losses* — penalized. A balanced/hedged build near the majority
  threshold is rewarded.
- **Why Turnovers specifically:** after the sign-flip TO negatively correlates with the volume stats,
  so a soft TO stance is *insurance* against the weeks the volume categories bunch badly. This benefit
  is **non-zero-sum** (structural, not "opponents are weak there"), so it survives self-play and even
  makes the bootstrap converge *faster*.
- **The 2×2 map** (behavior_model_confidence × aleph), all self-play equilibria of the same engine:

  | | confidence 0.5 | confidence 1.0 |
  |---|---|---|
  | aleph 0.2 | universal soft-TO, no commitment | near-total balance |
  | aleph 0   | committed punts + TO hedge | diverse committed, no herd |

  So aleph is doing much of the flattening; with raw correlations, committed punting survives.

## 4. The Gaussian insight

Under the framework's joint-Gaussian assumption, R fully specifies the joint distribution — there is
**one exact number** (the correlated-Gaussian majority probability). Independent, first-order, and the
factor method are **three approximations of that same number**; they differ only in which features of
the exact answer their truncation preserves. The Monte-Carlo oracle estimates the exact number
directly (sample the correlated Gaussian, count majorities), which is why it is the right referee. The
only assumption that could still be *wrong* is Gaussianity itself (genuine tail dependence beyond
correlation — not addressed here or by the current correction).

## 5. The oracle experiment

`testing_files/test_experiments.py :: test_mc_correlation_vs_oracle`. For each season, the top-12 **EC**
builds (clean, correction-agnostic strength profiles) are reduced to their served per-category win
rates, and the single-matchup majority-win probability is computed several ways vs a 300k-sample MC
oracle. On an empty board all 11 opponents are the identical generic field, so the single-opponent
collapse is exact for these served builds.

Results (percentage points of win probability; near-even regime, oracle ~52–53%):

| Method | mean \|err\| | note |
|---|---|---|
| Independent (base) | ~1.9 | ignores correlation — the size of the problem |
| First-order (current) | ~0.99 | halves the independent error |
| **Probit factor** | **~0.56** | proposed cheap analytic; ~1 extra DP pass |
| GH-9 reference | ~0.57 | same integral, 9-node quadrature (model ceiling) |
| 2nd-order Taylor | ~31 | **collapses** (see §6) |

**Caveat:** EC top-12 are all near-even matchups, the regime where correlation's effect *and* every
method's error are smallest. Favorites/underdogs would stress the gap wider. A stress row is a TODO.

## 6. The factor idea, the collapse, and the fix

**Condition on a common factor.** Write R ≈ I + λλᵀ (off-diag) + E, λ = √(leading eigenvalue)·(leading
eigenvector). Introduce a scalar latent week factor U; conditional on U=u the categories are exactly
independent with `p_c(u) = Φ((z_c + λ_c u)/√(1−λ_c²))`, so `P = ∫ φ(u) g(u) du`, `g(u) = V(p(u))`. The
residual E rides on the existing first-order patch.

**The 2nd-order Taylor collapses.** `E[g] ≈ g(0) + ½g''(0)` gives 31pp error. Not a bug — analytic
`g''(0) = −1.327` matches a finite-difference of `g(u)` exactly. The cause: the leading factor is
enormous (eigenvalue **4.7 of 9**, loadings to **0.96** — the volume/pace factor), so
`b_c = λ_c/√(1−λ_c²)` reaches **3.4** and `g(u)` is a steep sigmoid. For Jokic:
`g(0)=68.2% + term_pairs −9.9 + term_self −56.5 → 1.9%`. A −66pp "correction" means the expansion
parameter isn't small; a parabola can't track the sigmoid (it dives to −1.8, −5.8 within ±2σ).

**The fix: match a sigmoid with a sigmoid.** `g(u) ≈ Φ(α + κu)` ⇒ `E_U[g] = Φ(α/√(1+κ²))`, closed form.
Anchor at u=0: `α = Φ⁻¹(g(0))`, `κ = g'(0)/φ(α)`, `g'(0) = Σ_c V_c β_c` with `β_c = b_c φ(a_c)`. Both
`g(0)` and `g'(0)` come from **one** prefix/suffix DP-table build at the sharpened point (value from
the win-count distribution, slope from the leave-one-out/tipping masses). Result: **~0.56pp**, at one
extra DP pass — comparable to what the current correction already spends, and small next to the
per-iteration roster optimization.

Gauss-Hermite convergence (Jokic-era run): GH-5 1.34, GH-7 0.83, GH-9 0.49, GH-15 0.38 (the rank-1 +
residual model floor, set by ‖E‖=0.65). Probit (0.62 on that run) sits near GH-9 for ~1/9 the cost.

## 7. Open items / next steps

1. **Ship-or-not for MC (the real question):** accuracy is solved; strategy is not. Even an exact
   correction flattens committed punting. Run an **H-vs-H win-rate sim** (correlation-on drafters vs
   correlation-off) across seasons — the only test that says whether the hedged builds actually *win*.
2. **Rank-2 factor:** recover the ~0.2pp the probit leaves vs the GH-15 floor, at the price of a 2-D
   quadrature. Only if the H-vs-H test says correlation is worth shipping.
3. **aleph-PSD:** decide whether to keep aleph in the MC correction's R (it's Roto-calibrated and
   pushes R non-PSD) or use raw correlations for MC.
4. **Favorite/underdog stress row** in the oracle experiment — near-even understates the error.
5. **Probit anchoring:** whether `g(0)`/`g'(0)` can be approximated from marginal-point quantities to
   avoid even the one extra DP pass (trades accuracy — measure first).

## 8. Where the code lives

- Experiment: `testing_files/test_experiments.py :: test_mc_correlation_vs_oracle` (independent,
  first-order, probit factor, GH-9 reference, MC oracle) and `:: test_mc_correlation_punt_profiles`
  (the herding A/B).
- Debug script: `testing_files/mc_correlation_factor_debug.py` (2nd-order vs GH vs oracle; the g''(0)
  verification; the probit match; the g(u) curve data used in the artifact figure).
- Correction internals (pre-existing, baseline): `backend/math/algorithm_helpers.py`
  (`compute_win_probability`, `calculate_correction_terms`, `calculate_tipping_points`,
  `calculate_pair_bracket_matrix`); gate in `backend/math/algorithm_agents.py`
  (`mc_correlation_enabled`, env `MC_CORRELATION`, default off).
