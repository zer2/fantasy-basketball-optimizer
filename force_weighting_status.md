# Force-weighting: where this stands

Branch `force-punting` (from `general-cleanup`). Committed 2026-09-29 after the full battery: the e2e
suite passes 72/72, and the pytest suite fails the same 37 golden tests with and without this change
(all pre-existing; the goldens are regenerated in the commit that follows). The toggle was renamed
"Weight pinning" after the first look in a browser. Companion document:
`force_weighting_plan.md` (the design, written before the work).

---

## What the feature does

A toggle in Model Parameters, "Weight pinning", plus one small box per category in the
candidate table's header. Blank = the algorithm sets that weight as usual. A number = the weight is
fixed there and never updated during the descent.

The number is on the **same "100 = neutral" scale the expand view displays**, so a pin round-trips
exactly: type 40, read 40.0. That was the deliberate choice and it is what makes the feature
self-checking.

Pinned weights apply to **stage-2 solves only**. Every bootstrap / self-play pass and the opponent
model's best-response inference run unconstrained, because a pin is the user's own build and not a
claim about what the other seats are doing.

---

## Verified working

Three test scripts, all passing. They live in the session scratchpad
(`%TEMP%\claude\c--Users-zacha-Projects-FBBO-fantasy-basketball-optimizer\<session>\scratchpad\`),
which a restart may clear — **they are not in the repo and would need rewriting.**

1. `check_forced_projection.py` — the projection arithmetic in isolation. Rows still sum to 1, no
   negatives, every pinned column displays the typed number on every row, and the gate is off during
   bootstrap passes and opponent inference.
2. `check_forced_descent.py` — a real agent (app-built via POST /sessions). The field build still
   varies across candidates (unconstrained, as specified); the user-facing solve holds the pins to
   **exactly 0.00e+00 error across all 577 candidates** through 40 iterations of Adam, the L1 prox
   and the renormalisation; and the pins move the answer materially (mean |ΔH-score| 0.041, ~20
   points of displayed weight in the free categories), so the holds are a real constraint.
3. `check_forced_api.py` — end-to-end over HTTP. Pins round-trip exactly through
   `Candidate.category_weights`; the toggle alone governs the feature (same pins, toggle off, nothing
   held); an all-900 pin set is a **400** naming the categories, not a 500; a pin of `0` is honoured
   as a hard punt while `None` releases the category.

`testing_files/test_and_benchmark_draft.py` shows the **identical** failure set with and without this
patch (16 failed / 7 passed either way), so default behaviour is unchanged. See the goldens note
below — those failures are pre-existing and are not about this work.

---

## The two real bugs found along the way

**1. The empty-board short-circuit.** `get_h_scores` returns `self._default_result` for a completely
empty board — and that result comes out of the bootstrap, which is unconstrained by design. So pins
would have silently done nothing at draft start (exactly when you set them) and then the board would
have jumped to the pinned build on the first pick. Fixed by adding `not self._forcing` to the guard;
a pinned empty board pays for a real solve, and only with the toggle on.

**2. A dead wrapper.** My first draft added `_run_bootstrap_pass_unconstrained` alongside an untouched
`_run_bootstrap_pass`, but all seven call sites use the original name — nothing would have called the
wrapper, and every field-building pass would have run *with* the pins. Fixed by keeping
`_run_bootstrap_pass` as the wrapper the call sites already use and renaming the body to
`_solve_bootstrap_pass`, so the suppression cannot be forgotten at a call site.

---

## Backend changes

**`backend/math/algorithm_agents.py`**
- `_FORCED_MINIMUM_BUDGET = 0.05` and `ForcedWeightsInfeasibleError(ValueError)` at module level.
- `__init__` takes `allow_force_weighting: bool = False`, `forced_category_weights: dict = None` and
  delegates to the setter.
- `set_forced_category_weights(allow, pins)` — public, so a pin needs **no agent rebuild**. Converts
  the typed number through `v`: `raw = typed / 100 * v[category]`. Raises
  `ForcedWeightsInfeasibleError` when the pins leave the free categories under 5% (never silently
  rescales — that would show a weight different from the one typed).
- `_forcing` property — the gate: off during bootstrap/self-play (`_unconstrained_run`) and opponent
  inference (`_opponent_inference_active`).
- `_project_forced_weights(weights)` — holds the pinned columns, renormalises only the free ones to
  what the pins leave over.
- **All five weight-mutation sites in `perform_iterations` masked.** The subtle two: gradient centring
  must average over the FREE columns only (centring *is* the simplex projection, so centring over all
  of them points the projected gradient off the constraint set), and the row renormalisation rescales
  every column including pinned ones, which defeats a naive implementation.
- The cold seeds are projected **before** they are scored in `_select_starting_weights` — the menu is
  chosen by comparing seed objectives, so unprojected seeds pick the seed for the wrong problem.
- One projection before the descent loop, covering both cold seeds and a warm start from
  `_player_frozen_weights` (which comes out of the unconstrained field passes).
- `_run_bootstrap_pass` is now a pass-through wrapper setting `_unconstrained_run`; the body is
  `_solve_bootstrap_pass`. The two external monkey-patchers
  (`testing_files/test_experiments.py`, `visualizations/6_self_play/prepare_self_play_data.py`) both
  wrap the outer name and pass positionally, so they keep working and still get the suppression.

**Transport, decided deliberately**
- The **toggle** is a model setting (`ModelSettings.allow_force_weighting`), and is excluded from
  `_agent_cache_key` alongside `team_names` — no pipeline step reads it, so flipping it is a cache
  hit rather than a ~6s rebuild.
- The **pins** ride on the **evaluate request** (`EvaluateRequest.forced_category_weights`), because
  they only affect the stage-2 solve. Typing in a box therefore costs one re-evaluate (~1s) instead
  of an agent rebuild, and they can never pollute the cache key.
- `rank_candidates` sets the pins around the solve and clears them in a `finally`, so the agent's
  resting state is unpinned and no other consumer of `session.agent` (trading) can inherit a pin.
- `backend/api/routers/ranking.py` adds `ForcedWeightsInfeasibleError` to the 400 branch, so an
  infeasible set surfaces its message instead of the blanket 500.

---

## Frontend changes

- **`frontend/table/force_weights.ts`** (new) — the pin store and the header row builder. The pins
  live in the module and in saved preferences, **not in the inputs**, because `buildTableHeader()`
  throws the header away on every settings change; values left only in the DOM would vanish the first
  time an unrelated sidebar control moved. A change dispatches a `forced-weights-changed` document
  event rather than importing the re-evaluate (which would close a cycle).
- **`model_parameters.ts`** — `FORCE_WEIGHTING_SPEC` and `makeForceWeightingItem()`. Its own builder
  because every `PARAM_SPECS` entry is a numeric input resolved against parameters.yaml min/max, and
  this is the section's first checkbox. Included in "Restore defaults".
- **`player_table.ts`** — appends the pin row as the **second** thead row, so the label row stays
  `tr:first-child` and keeps its rounded-corner rules in styles.css. Column-aligned by construction:
  `isAuction ? 5 : 2` blank leading cells, then one box per category.
- **`main.ts`** — `makeEvaluateOnlyChain()` plus a `forced-weights-changed` listener: a pin change
  needs no session patch, just a re-solve.
- **`client.ts` / `types.ts` / `draft_and_auction_session.ts`** — the wire types, and the pins
  attached to the personalised request only. The base ("generic") evaluate is deliberately left
  unpinned, since it is the neutral reference the board compares against.
- **`styles.css`** — `.forced-weights-row`, `.forced-weights-label`, `.forced-weight-input`
  (spinners suppressed; they cost more width than a category column has to give).

---

## Not done

1. **Never run in a browser.** The backend is verified end-to-end over HTTP, but the toggle, the
   boxes, and the CSS have not been looked at on a real page. This was the next step — start the
   local server and check it.
2. `parameters.yaml` has **no** entry for the toggle. It does not need one (the checkbox resolves no
   default/min/max), but every other model parameter is documented there, so consider adding one for
   consistency.
3. No test in `testing_files/` — the three scripts are scratchpad-only.
4. **Open question, deliberately not guessed:** should a forced build carry into the trade analyser?
   Right now it does not (pins are released after each evaluate). Arguably a user who has pinned a
   punt build wants trades judged against it, but that was outside the ask.
5. The `n_picks = n_active` argument name at `build_agent.py:389` is pre-existing and untouched.

---

## The golden failures are NOT from this work — measured, not assumed

`test_and_benchmark_draft.py` has 16 failing goldens on this branch. They fail identically with the
patch reverted, and — the decisive test — they fail identically at **`c5cb446c` itself**, the very
commit the goldens were regenerated at (2026-09-07), running its own code in a clean worktree:

| format | golden | c5cb446c today | HEAD today |
|---|---|---|---|
| Most Categories | 60.5 | 60.1 | 60.1 |
| Each Category | 53.8 | 53.7 | 53.7 |
| Rotisserie | 13.8 | 13.3 | 13.3 |

So no code changed the numbers — **the data moved under the goldens**. `BOX_SCORE_TABLE_BEFORE_TEAM`
(the 2026-09-26 pre-change snapshot, still in Snowflake) versus `BOX_SCORE_TABLE` now:

| season | rows before | rows added |
|---|---|---|
| 22024 (2024-25) | 31,623 | **+892** |
| 22007 | 29,484 | +24 |

The goldens use 2024-25, so the 892 filled-in games are the cause. `_SCORE_TOL` is 0.06, which is why
a ≤0.5 shift trips every one of them. The visualization commits were checked specifically and are
behaviour-neutral: `e3c9129f` only hoists a `drift = None` initialiser and appends to
`self.self_play_trace`, which is `None` unless a caller opts in, and the app never does.

**Agreed with Zach: the goldens are fine / this is understood.** They will need regenerating
(`REGEN_GOLDENS=1 python -m pytest testing_files/test_and_benchmark_draft.py -q -s` prints them
paste-ready) but that is a separate, explicitly-approved change — not part of force-weighting.

---

## Resuming

```bash
git -C C:/Users/zacha/Projects/FBBO/fantasy-basketball-optimizer status   # branch force-punting, all uncommitted
npx tsc                                                                   # should be silent
```

Then start the local server and exercise the UI: flip "Weight pinning" on in Model
Parameters, confirm the pin row appears in the candidate table, type 40 into Free Throw %, and check
the expand view shows Free Throw % at 40.0 for every player while the other categories still move.
