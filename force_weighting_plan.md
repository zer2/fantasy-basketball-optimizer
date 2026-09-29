# Force-weighting: a plan

Let the user pin some category weights by hand, constraining the optimiser into a chosen build.

Spec as given: a toggle in algorithm parameters reading "allow force-weighting", described as "allow some
category weights to be fixed manually, constraining the algorithm and forcing it into a certain build".
Above the categories on the main tables, a small box per category. Blank means the algorithm sets that
weight as usual; a number means it is fixed and never updated during optimisation.

---

## The one decision that changes the feature: what the number means

The UI already shows category weights on a **"100 = neutral"** scale. `backend/services/ranking.py:252-269`:

```python
category_weights_normalized = (category_weights_raw.values / v_reshaped) * 100
```

So a category displayed at 100 is sitting at its neutral weight `v`, and 140 is 40% above it. Internally
the weights sum to 1 (`algorithm_agents.py:645-646`), and `v` itself sums to 1.

That leaves two readings of "out of 100":

| reading | typing 20 for FT% means | to punt FT% you type |
|---|---|---|
| **A. same scale the UI shows, 100 = neutral** | one fifth of neutral weight | a small number, 0 to 30 |
| B. share of the total, all boxes sum to 100 | a fifth of *all* weight on FT% | still a small number, but 11 is neutral with nine categories |

**Recommend A.** Three reasons. It is the scale the user already reads in the expand view, so a pinned box
and the displayed weight agree by construction, which also makes the feature self-checking: type 40, see 40.
Under B the neutral value depends on how many categories the league uses, so the same typed number means
different things in an 8-category and a 10-category league. And B invites the user to pin values that sum to
more than 100, which has no feasible answer, whereas A's failure mode is milder and easier to explain.

Under A, a pinned box value `p` becomes the raw internal weight `p / 100 * v[category]`.

**Feasibility guard.** The pinned raw weights must sum to less than 1, since the free categories need what
is left. With A this only bites if the user pins nearly every category very high. The guard belongs in the
API layer so the error reaches the UI: reject when `sum(pinned_raw) >= 1 - epsilon` and say which boxes to
lower. Do not silently rescale, which would show the user a number different from the one they typed and
destroy the self-checking property.

---

## Backend: where the pin has to bite

All category-weight maths is in `backend/math/algorithm_agents.py`, class `HAgent`. The optimiser is Adam on
a simplex-projected weight matrix; weights are a `(n_candidates, n_categories)` ndarray inside the descent,
one row per candidate player, column order `self.x_scores.columns`.

### The update site

`HAgent.perform_iterations`, loop at `algorithm_agents.py:1847`, mutation block `1874-1889`. **Five** places
mutate the weights and a pinned column must be excluded from every one of them:

| line | what it does | what a pin needs |
|---|---|---|
| 1874 | centres the gradient: `gradients - gradients.mean(axis=1)` | **mean over FREE columns only** — this is the simplex projection, and with some columns pinned the free weights live on a smaller simplex |
| 1876 | the Adam step, `category_weights + cat_updates` | zero `cat_updates` on pinned columns |
| 1885 | the L1 regulariser prox, pulls toward neutral | zero `shrink` on pinned columns |
| 1886 | negativity clip | harmless but skip it; pinned values are non-negative by construction |
| 1889 | **row renormalisation to sum 1** | renormalise only the free columns, to `1 - sum(pinned)`, then write the pinned targets back |

Line 1889 is the one that silently defeats a naive implementation: even with 1876 and 1885 masked, the
renormalisation rescales every column including the pinned ones, every iteration. And line 1874 is the one
easiest to miss, because it looks like bookkeeping rather than a mutation; leaving it unmasked makes the
projected gradient point off the constraint set, so the free weights would be systematically mis-stepped.

Cleanest shape: build the mask and the target row once in `perform_iterations` before the loop, then a single
helper applied after the step.

```python
# pinned_mask: (n_categories,) bool; pinned_raw: (n_categories,) float, the target weights
free = ~pinned_mask
budget = 1.0 - float(pinned_raw[pinned_mask].sum())
...
category_weights[:, pinned_mask] = pinned_raw[pinned_mask]
scale = category_weights[:, free].sum(axis=1, keepdims=True)
category_weights[:, free] *= budget / np.where(scale > 0, scale, 1.0)
```

### The seeds also have to be feasible

If the descent starts off the constraint set the first iterations are wasted pulling back onto it. Apply the
same projection at every seed builder:

- `_select_starting_weights`, `algorithm_agents.py:1637-1717` — the multistart menu, including `gentle_punt`
  at 1657-1660 and the history rows at 1679-1687
- `get_h_scores` seed modes, `algorithm_agents.py:928-961` — heuristic, neutral, lowvar
- the partial-warm-start override at `1823-1827`, which is the existing precedent for writing into specific
  positions of the weight array and worth copying in style

A pinned category should probably also be dropped from the multistart punt menu: a "punt FT%" seed is
meaningless when FT% is pinned at 140, and it wastes one of the seeds evaluated at 1706-1717.

### What the pin should NOT touch

`infer_opponent_category_weights` (`algorithm_agents.py:1566-1617`) and `self._team_states`
(`1544-1564`) hold the opponent model's beliefs about what *other* teams are doing. A pin is a constraint on
the user's own build, not a claim about opponents, so it must not be applied there. This needs to be explicit
in the code rather than left to chance, because the opponent inference calls the same descent machinery with
`_OPPONENT_INFERENCE_ITERATIONS`.

That raises one design question worth settling deliberately: with `opponent_model_confidence` above 0.5 the
app runs mean-field self-play, where the field's weights are the model's own output fed back. If the user
pins weights, should the simulated field also be pinned? I would say no — the user is one seat, not the
league — but it means the pinned seat and the field are solving different problems, which is a change to what
self-play converges to and should be verified rather than assumed.

---

## Frontend and API

Note first: **`kappa` is gone.** It was removed in `5f2a320e` and survives only in `testing_files/`; the
anti-crowded-punt effect is now emergent from `backend/math/truncated_max_pick_model.py:190-192`. The live
template for a new algorithm parameter is `lambda_c`, and `ModelSettings` currently contains **no boolean**
at all -- `use_opponent_awareness` was replaced by the continuous `opponent_model_confidence` -- so the
toggle is the first of its kind in that model and needs its own small amount of new ground.

### The chain, seven hops, following `lambda_c`

| hop | file | line | what to add |
|---|---|---|---|
| default, min, max | `parameters.yaml` NBA.options | 295-304 is `lambda_c` | an entry for the toggle; the per-category values need no YAML, they have no default |
| spec and control | `frontend/setting_collection/model_parameters.ts` | `PARAM_SPECS` 20-74, `makeParamItem` 155-188 | the toggle spec and a `type: 'toggle'` branch |
| read back out of the DOM | same file, `getModelSettings()` | 191-205 | the toggle, and the pinned values |
| TypeScript type | `frontend/types.ts` | `ModelSettings` 66-86 | `allow_force_weighting: boolean` and `forced_category_weights: Record<string, number>` |
| request and patch | `frontend/api/session.ts` 169, `frontend/main.ts` 322-325 | | a change listener on the table boxes, see below |
| Pydantic | `backend/api/schemas.py` | `ModelSettings` 74-100 | the same two fields; `PatchRequest` needs nothing, it uses `model_dump()` |
| flatten into settings | `backend/api/routers/sessions.py` | `_build_current_settings` 62 | one line each; `_build_patch` 82-83 needs nothing |
| into the agent | `backend/services/build_agent.py` 398, `backend/math/algorithm_agents.py` 371 and 422 | | constructor argument and the mask build |

`_build_patch` using `model_dump()` means the patch path needs no per-field code, but
`_build_current_settings` does. Easy to add one and forget the other, and the failure is silent: the value
works on a fresh session and is ignored on an edit, or the reverse.

### The toggle

`makeParamItem` builds only `<input type="number">` (176-185) in a two-column `.param-grid`. The existing
toggle helper is `makeSidebarToggle(id, rightText, leftText?)` at `frontend/helper_functions.ts:111-134`,
which wraps a real checkbox; the best copy target is the third-round-reversal toggle at
`frontend/setting_collection/league_settings.ts:184-190`, which already pairs it with `pref`/`savePref`.

It does not produce a `.param-item`, so either wrap it or extend `makeParamItem` with a `type: 'toggle'`
variant. Extending `makeParamItem` is the better call: it keeps the ⓘ tooltip mechanism
(`infoBtn.dataset.tooltip = spec.caption`, line 173) so the description you wrote appears the same way every
other parameter's does, and it keeps the grid alignment.

### The per-category boxes

Build them from the same `getSelectedCategories()` the header already calls at `player_table.ts:128`, so the
boxes and the category columns cannot drift apart.

Where: `buildTableHeader()` at `player_table.ts:126-196` creates exactly one header row
(`thead.insertRow()`, line 153). A second row above it via `thead.insertRow(0)` is the natural place, and
there is precedent for a second row in the same `<thead>` at `season_rosters.ts:365-371`. Two constraints:

- the table is `table-layout: fixed` with per-`th` widths and rem floors (`CAT_COL_W_REM`, lines 40-42), so
  the new row must carry the same number of cells and no `colSpan`
- virtualization spacer rows use `columnCount` (line 82) and a mismatch squishes the columns

**The trap.** `buildTableHeader()` starts with `table.innerHTML = ''` (line 141) and is called from six
places (`main.ts:65, 105, 176, 195, 249, 374`). So anything typed into a box is destroyed on every rebuild.
The typed values must live outside the DOM -- in app state, persisted with `savePref` like the sidebar
controls do -- and be re-read when the header is rebuilt. Otherwise the boxes appear to work and silently
clear whenever the table refreshes, which is the kind of bug that survives manual testing.

Second: the sidebar patch is wired to `modelSection`'s change event (`main.ts:322-325`). The boxes are not in
that section, so they need their own listener calling the same `applyModelSettings(3, ...)` chain. Step 3 is
right: it reruns upsilon, scoring info and the agent, and the weights are computed in the agent.

### What the boxes should show when empty, and when disabled

Blank means "not pinned", which is the spec. Worth deciding explicitly: when force-weighting is off, hide the
row rather than disable it, so the table does not lose vertical space to a row that does nothing. And a
pinned box should be visually distinct from an empty one, since the whole point is that the algorithm is
constrained -- the user needs to see at a glance which categories they have taken away from it.


---

## Testing

1. **A pinned weight comes back exactly.** Pin FT% at 40, run, assert the value in the API response is 40 to
   floating-point tolerance. This is the end-to-end test and it works only because reading A makes the input
   and output the same scale.
2. **The free weights still sum correctly.** Assert every row sums to 1 internally, and that the free columns
   sum to `1 - sum(pinned)`.
3. **Pinning every category** leaves the optimiser with nothing to do and must not divide by zero at the
   renormalisation. Either reject it in the API or handle `budget == 0`.
4. **The toggle off is a no-op.** Regenerate the goldens and assert they are byte-identical. This is the test
   that protects the existing behaviour, and per the repo's history with `kappa` and the G-score denominator
   change, it is the one that catches an accidental default change.
5. **A pin that fights the gradient holds.** Pin a category the optimiser would otherwise push to zero, run
   the full iteration count, and assert it has not moved. This is the test that would fail if line 1889 or
   1874 were left unmasked, which is the likeliest bug.
6. **Opponent weights are unaffected.** With a pin set, assert `_team_states` weights differ from the pinned
   row, so the constraint has not leaked into the opponent model.

## Order of work

1. The mask plumbing through the API and into `HAgent`, defaulting to off, with the goldens regenerated and
   verified unchanged. No behaviour change yet.
2. The five mutation sites plus the seed builders, with tests 1, 2, 5.
3. The UI: toggle, then the boxes.
4. The opponent-model question above, deliberately, with a self-play check.
