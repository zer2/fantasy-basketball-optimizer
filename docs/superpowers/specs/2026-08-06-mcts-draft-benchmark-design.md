# MCTS vs. H-Score Draft Benchmark — Design

**Date:** 2026-08-06
**Status:** Approved (pending spec review)
**Branch:** `mcts-benchmark`

## Purpose

Decide, on numbers rather than argument, whether adding Monte Carlo Tree Search
(MCTS) lookahead to the fantasy-basketball draft tool beats the current
one-ply H-score point estimate. This is a **decision-support benchmark**, not a
production feature: it produces realized-win-rate comparisons that tell us
whether MCTS is worth pursuing in the live tool at all.

The benchmark must clear a high bar of honesty: the strategy under test (MCTS)
must not be scored by the same objective it optimizes, or it wins by
construction. See [Independent Evaluator](#independent-evaluator).

## Hard Constraint: Additive Only

**Nothing zer2 developed may be modified.** The benchmark is purely additive:

- All new code lives in a separate `benchmark/` package.
- No edits to any file under `src/` or to `app.py`.
- The existing engine (`process_player_data`, `process_game_level_data`,
  `HAgent`, `get_default_h_values`, the `helper_functions` getters) is
  **imported and called read-only**.
- The headless bootstrap sets `st.session_state` at runtime, exactly as
  `app.py` does. That is runtime state, not a source edit, and is permitted.
- If any change appears to require editing one of zer2's files, STOP and flag it
  to the user rather than modifying it.

## Settled Decisions

| Axis | Decision |
|---|---|
| Data source | Historical NBA game logs via `nba_api` (public, no credentials) |
| Season | 2025-26 (complete; full 82-game logs — verified reachable, 582 players, 26,651 player-games) |
| Data handling | Pull once → snapshot to committed fixture → all runs read the fixture (deterministic, fast, avoids `stats.nba.com` IP hangs) |
| Ground truth | Independent Monte-Carlo season sim on **resampled real box scores** — never the H-score objective |
| Field types | **Both**, reported separately: (a) G-score field, (b) H-score field |
| Scoring formats | **Each Category (EC)** and **Most Categories (MC)**. Rotisserie deferred. |
| Agent under test | MCTS with stochastic opponents (true MCTS, Approach 2) |
| MCTS leaf | Reuse `HAgent`'s full-roster (`n_players_selected == n_picks`) win-prob branch |
| Opponent model | **Weighted-softmax** over the static ordering (no external ADP; the model's own ranking plays the ADP role). Same model for the scored field AND for MCTS rollout opponents. |
| Temperature `T` | **Swept** across runs — the headline experimental variable ("how much draft-room chaos before lookahead pays off?") |
| Variance control | Common random numbers: both heroes face identical seeded field draws + season bootstrap per matched draft |

## Architecture

Standalone package, decoupled from the Streamlit UI:

```
benchmark/
  data.py         # nba_api pull → snapshot fixture (season averages + game logs)
  bootstrap.py    # headless: populate st.session_state + build `info` via process_player_data
  agents.py       # HScoreAgent (wraps HAgent), GScoreAgent (static), MCTSAgent (new)
  draft.py        # snake-draft engine: seats, order, transitions (mirrors drafting.py logic)
  evaluate.py     # INDEPENDENT season simulator → realized win-rate (EC + MC)
  experiment.py   # orchestration: rotate hero across seats x fields x formats, aggregate
  fixtures/       # committed snapshot(s), e.g. nba_2025-26.parquet
```

### Data flow (one draft)

```
fixture (averages + logs)
   |
   +- averages --> bootstrap --> info dict --> agents make picks --> final rosters
   |                                                                      |
   +- game logs ----------------> evaluate.py (bootstrap real box scores) +
                                        |
                                        v
                              realized EC/MC win-rate per seat
```

**Core separation:** agents draft on season *averages* (their "projection");
the evaluator scores rosters on resampled real *game logs*. The evaluator never
calls the H-score objective, so no strategy can win by optimizing its own
yardstick.

## Components

### data.py

- Pull 2025-26 player game logs via `nba_api` (`playergamelogs.PlayerGameLogs`,
  `league_id_nullable='00'`), reusing the same endpoint the WNBA path already uses.
- Derive two artifacts from the single pull:
  1. **Season averages** table in the shape `process_player_data`/`player_stats_v2`
     expects (reuse `process_game_level_data` for the conversion).
  2. **Per-game box scores** retained for the evaluator's bootstrap.
- Snapshot both to `fixtures/nba_2025-26.parquet` (or a small set of files).
  Commit the fixture so the benchmark runs anywhere with no network.

### bootstrap.py

- Populate `st.session_state` with the settings the getters read (league=NBA,
  scoring_format, n_drafters, n_starters, omega/gamma/beth, params from
  `parameters.yaml`, selected categories, position structure, etc.), mirroring
  `app.py`'s initialization.
- Inject the historical averages as `player_stats_v2`.
- Call `process_player_data(...)` to build the `info` dict the agents consume.
- Fallback if direct session-state population proves brittle: drive setup
  through Streamlit's `AppTest` harness (already used in `test_algorithms.py`).
- Set `SPORT=NBA` so the `os.environ` getter fallbacks are defined.

### agents.py

Common interface: `make_pick(player_assignments, seat) -> player`.

- **GScoreAgent** — picks the top available player by static total G-score
  (position-eligible). Mirrors `SimpleAgent`.
- **HScoreAgent** — thin wrapper over `HAgent.make_pick` (existing).
- **MCTSAgent** — new; see [MCTS Agent](#mcts-agent).
- **RandomAgent** — picks a random position-eligible available player. Used only
  as a sanity-check floor for the evaluator (success criterion 3), not a field.

Field agents (both G-score and H-score fields) pick via the shared
weighted-softmax opponent model (see [Opponent Model](#opponent-model)) over a
**cheap cached static ordering** to keep runtime bounded; the full H-score
engine is reserved for the hero when hero=H-score, and for MCTS leaf evaluation.

### Opponent Model

The single most important modeling choice: how the 11 non-hero drafters pick.

- **No external ADP.** Average Draft Position (real-world consensus draft-order
  data) is not imported. The role ADP would play — the consensus ordering
  everyone drafts roughly along — is played by the **model's own static
  ranking**: total G-score for the G-score field, cached default H-score
  ordering (`get_default_h_values`) for the H-score field.
- **Weighted-softmax draw**, not uniform-random and not a plain weighted
  average: each opponent samples from its top-K available players with
  `P(player_i) ∝ exp(score_i / T)`. `T→0` = always take #1 (chalk);
  higher `T` = more reaches/falls (chaotic room).
- **Shared model:** the scored benchmark field and the MCTS rollout opponents
  use the *same* weighted-softmax policy. This is deliberate — it keeps MCTS's
  opponent model **correctly specified** (it knows the field's policy family),
  which is the right first question: *does lookahead help when the model is
  right?* If MCTS can't win in its best case, it's dead. Model-misspecification
  (field at a different `T` than MCTS assumes) is a harder, later question.
- **Why not a deterministic field:** a deterministic field has zero
  availability uncertainty, so a good agent could compute exactly who is
  available at every future turn. That erases the very thing stochastic MCTS
  rollouts exist to handle — we would be testing MCTS's headline advantage in
  the one world where it cannot appear. (Against a deterministic field the
  honest tool would be deterministic *expectimax*, not stochastic rollouts;
  since Approach 2 committed to stochastic rollouts, the field must be
  stochastic to match.)
- **Temperature `T` is swept** across the experiment grid — it is the headline
  variable answering "how much draft-room chaos before lookahead beats the
  one-ply estimate?"

### draft.py

- Snake-draft engine: given a list of seat agents, run the full draft producing
  12 final rosters. Snake order and pick transitions mirror the logic in
  `src/tabs/drafting.py` (`move_forward_one_pick`), reimplemented in the
  benchmark package (not imported from the UI module) to keep the UI untouched.
- Position eligibility enforced via the existing
  `check_single_player_eligibility` helper (read-only import).

### MCTS Agent

Exposes the same `(player_assignments, seat) -> player` contract (and a
`player -> value` Series for potential future display-compatibility).

- **State/actions/transition:** fully observable snake draft; root actions =
  hero candidate picks abstracted to **top-K (~15-20)** by cached static
  H-score ordering (`get_default_h_values`); transition = snake order.
- **Selection:** PUCT — `Q(s,a) + c_puct * P(s,a) * sqrt(sum N) / (1 + N(s,a))`,
  prior `P` = softmax over static H-score ordering (H-score is the prior policy).
- **Expansion:** add a child for an untried top-K candidate.
- **Rollout (the "MC"):** play the draft to completion. Opponents (and the hero
  past the tree frontier) pick via the shared weighted-softmax opponent model
  (see [Opponent Model](#opponent-model)) over the cached ordering. Different
  rollouts see different players fall to the hero — this is the availability
  uncertainty one-ply H-score cannot represent, and it matches the policy family
  of the scored field.
- **Backpropagation:** at the terminal full hero roster, evaluate the leaf and
  propagate value up the visited path.
- **Leaf evaluation:** reuse `HAgent`'s `n_players_selected == n_picks` branch,
  which returns a scalar team win-rate under the EC/MC objective. This is a
  *search-time heuristic* to guide MCTS — explicitly distinct from the
  independent evaluator that scores final results.
- **Cost controls:** opponent picks are array lookups (cached ordering), not
  optimization; full analytic objective spent only at terminal leaves. Knobs:
  `K`, `n_simulations`, `c_puct`, rollout `temperature`. In the correctly-
  specified baseline the rollout `temperature` **equals the field `T`** for that
  run (MCTS knows the field's policy) — not an independent free knob. A future
  misspecification study would deliberately decouple them.
- **Anytime:** `search()` can yield improving estimates as budget grows,
  matching the existing `get_h_scores` progressive pattern.
- **Determinism:** seeded RNG.

### Independent Evaluator

`evaluate.py` — the component that makes the benchmark trustworthy.

- **Input:** 12 final rosters + the season's real per-game box scores.
- **Method — bootstrap the fantasy season:**
  1. A "week" samples one real game row per rostered player from that player's
     actual logs; sum each team's category totals for the week. Sampling real
     games (not Gaussian draws) preserves each player's true shape (e.g. real
     FT% volatility, boom/bust nights) — the texture punting exploits.
  2. Injuries/missed games fall out of the data: a player with 50 of 82 games
     is sampled from 50 rows; "did not play" weeks injected proportionally so
     availability is realized, not assumed.
  3. Run the **actual format scoring** over a round-robin schedule:
     - **EC:** each week vs. one opponent; win each category taken; tally
       category-wins over the season.
     - **MC:** same schedule; week result = one win to the majority-of-
       categories taker.
  4. Repeat for many simulated seasons (500-1000) → per-seat expected win-rate
     with confidence interval.
- **Output:** per-seat realized win-rate + CI, for EC and MC.
- **Ratio categories (FG%, FT%):** volume-weighted from summed makes/attempts
  across the week (not naive average of percentages); reuse the repo's ratio
  handling from `process_game_level_data` so category math matches the app.
- **Same evaluator for every strategy** — H-score, G-score, and MCTS rosters
  all scored by identical resampled-real-basketball simulation.

### experiment.py

- Orchestrate the full comparison over the grid
  `(field ∈ {G-score, H-score})` x `(format ∈ {EC, MC})` x
  `(T ∈ swept temperatures)`: run many drafts rotating the **hero** seat across
  all 12 positions and randomizing pick order, with hero ∈ {H-score, MCTS}.
- **Common random numbers:** within a matched draft, both heroes face the
  *same* seeded field draws and the *same* season bootstrap, so the
  MCTS-minus-H-score difference is not swamped by field/evaluator variance.
- Aggregate hero realized win-rate (mean + CI) per condition.
- Headline result: MCTS-hero win-rate minus H-score-hero win-rate, plotted
  **against `T`** per (field, format), with confidence intervals — i.e. the
  curve of lookahead's value vs. draft-room chaos.

## Scope Boundaries (Explicit Non-Goals)

- **No streaming / waivers / trades / mid-season adds.** Rosters frozen at draft
  end. Deliberate — isolates *draft* quality, matching the tool's own framing of
  streaming as an out-of-scope limitation.
- **No Rotisserie** in the first pass (structurally different evaluator).
- **No UI integration.** This benchmark answers "is it worth building"; wiring
  MCTS into the live `make_cand_tab` flow is a separate later project, gated on
  positive results here.
- **No live/production data path.** Fixture-only.

## Success Criteria

1. Benchmark runs end-to-end headlessly with no credentials, off the committed
   fixture, deterministically under a fixed seed.
2. Produces per-(field, format) realized win-rate for H-score-hero and
   MCTS-hero with confidence intervals.
3. Sanity checks pass: H-score-hero beats G-score-hero against the G-score
   field (validates the evaluator can detect known skill differences); a random
   agent underperforms both.
4. Zero modifications to `src/` or `app.py` (verifiable via `git diff`).

## Key Risks

- **Headless execution of the engine.** Getters read `st.session_state` with an
  `os.environ['SPORT']` fallback. Mitigation: populate session state directly;
  fall back to `AppTest` if brittle.
- **Runtime.** H-score field + MCTS rollouts are expensive. Mitigation:
  opponents use cached static ordering; analytic objective only at leaves;
  tunable simulation budget.
- **Evaluator realism vs. circularity.** Resampling real games is the honest
  choice but ignores in-season roster management; accepted as an explicit
  non-goal that isolates draft quality.
