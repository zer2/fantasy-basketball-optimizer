# Code hygiene audit — 2026-09-25

A sweep for duplication, stale comments and inconsistent organisation, recorded so the cleanup
can be done quickly later rather than discovered again. Nothing here is fixed; nothing here is
urgent. Bugs were watched for but not hunted — see *What this did not cover*.

**Headline: the codebase is in better shape than the size suggests.** No commented-out code
anywhere, essentially no dead code, and almost no duplication. What is worth fixing is small and
mostly clerical.

## A. Duplication

Structural comparison of every function of 6+ lines across 118 tracked Python files found
**two** exact-duplicate groups and **one** near-duplicate pair. That is very low.

1. **Three ways to build a test session request.** `_build_default_session_request` is 95%
   identical between [test_algorithms.py:39](testing_files/test_algorithms.py#L39) and
   [test_app_setup.py:32](testing_files/test_app_setup.py#L32), and
   [benchmark_helpers.py:88](testing_files/benchmark_helpers.py#L88) has a third, parameterised
   `_build_session_request` that already does more than either. The two copies should call the
   benchmark helper. **This is the only duplication worth acting on.**

2. **`_build_empty_board_agent`** is identical in
   [multistart_flex_diagnostic.py:42](testing_files/multistart_flex_diagnostic.py#L42) and
   [multistart_prune_sim.py:41](testing_files/multistart_prune_sim.py#L41). Both are diagnostics
   rather than shipped code, so this matters less.

3. **Not duplication, despite matching:** `get_auction_results` is byte-identical in the ESPN and
   Fantrax integrations because both are `return None` — neither platform supports auctions,
   while Yahoo implements it fully. It could be a base-class default returning None, but two
   two-line stubs is a defensible way to make "not supported" explicit per platform.

`main` (12 modules), `construct` (8) and `fetch_league_shape` (3) share names across modules by
design — entry points, manim scenes, and the platform interface.

## B. Comments that no longer describe anything

24 comments name a `.py` file that does not exist. They split cleanly:

**Deliberate provenance, leave alone.** Roughly half point at the *old Streamlit* source
(`src/platform_integration/espn_integration.py`, `src/tabs/drafting.py`, `get_data.py`) to say
where something was ported from. Those files were never in this repo. The frontend `// Mirrors
X() in src/...` headers are the same thing and are genuinely useful.

**Genuinely stale, worth fixing:**

| where | says | reality |
|---|---|---|
| [z_score.py:14](visualizations/1_z_score/z_score.py#L14) | companion scene in `weekly_differential.py` | no such file |
| [g_score.py:3](visualizations/2_z_versus_g_score/g_score.py#L3) | the first scene is `team_differential.py` | no such file |
| [roster_slots.py:10,73](visualizations/3_roster_slots/roster_slots.py#L10) | numbers come from `prepare_assignment_data.py` | no such file |
| [differential_base.py:5-6](visualizations/shared/differential_base.py#L5) | `team_differential.py` / `weekly_differential.py` | neither exists |
| [narration_notes.md:6,39,64,123](visualizations/narration_notes.md) | `assignment.py`, `team_differential.py` | neither exists |
| [process_player_data.py:405](backend/math/process_player_data.py#L405) | `evaluate.py` | no such file |
| [test_trading.py:4](testing_files/test_trading.py#L4) | `benchmark_trading.py` | it is `test_and_benchmark_trading.py` |

The visualisation ones are all the same root cause: scene files were renamed when the folders
were reorganised, and the cross-references between scenes were not followed.

**Markers:** exactly one real code TODO — [algorithm_agents.py:484](backend/math/algorithm_agents.py#L484)
`#TODO: clean this up`. The other two matches are a doc heading and a note about a Streamlit TODO.

## C. Organisation

1. **`backend/` root is a mixed bag — but not one thing.** Everything lives in a package except
   five modules at the root. `main.py` (142) belongs there. The other four look like one layer
   and are not: three import nothing internal at all, and one does.

   | module | imports internally | what it is |
   |---|---|---|
   | `parameters.py` (34) | nothing | loads and caches `parameters.yaml` |
   | `models.py` (117) | nothing | the DTOs services build and the API declares |
   | `player_identity.py` (130) | nothing | player ids, registry, name→id resolution |
   | `data_retrieval.py` (368) | `infra.snowflake_connection`, `player_identity` | domain reads from Snowflake |

   So "core" is the wrong name for the set. Suggested, in increasing order of doubt:

   - **`parameters.py` → `backend/infra/`.** The clearest move. `infra/secret_config.py` (38) is
     already "load configuration from disk"; this is the same job for a different file.
   - **`models.py` + `player_identity.py` → a new package.** These two are the vocabulary of the
     problem — what a player is, and what we hand back. `backend/domain/` says that; `shared/` is
     accurate but says nothing. Note `models.py` cannot go into `api/` without inverting the
     dependency, since services build these and the API only declares them.
   - **`data_retrieval.py` — leave it** until someone has a view. It is the only one with
     dependencies, and its docstring already states its position deliberately: the generic
     Snowflake connection lives in `infra`, and this is the domain data-access layer above it.

2. **Package sizes are lopsided.** `backend/math/` is 4893 lines across 8 files; every other
   package is under 2200. That is inherent to the subject rather than a mistake.

3. **Leading vs trailing commas.** CLAUDE.md asks for commas at the beginning of each parameter
   line. Measured across `backend/`: **372 leading-comma lines against 258 trailing-comma ones** —
   so the convention is followed about 59% of the time. `backend/api/helpers.py` is the clearest
   mixed case. (The count is approximate; the trailing-comma detector will pick up some ordinary
   call arguments.)

## C2. `services/ranking.py` is misnamed, not overloaded

835 lines, but only **377 are code** (45% — the rest is comment and docstring). Of that code,
`rank_candidates` is 73 lines and the five response builders are 289 — **80% of the file is
shaping the answer**, not ranking it.

That looked like a file doing two jobs. It is not. `rank_candidates` returns `EvaluateResponse`:
the whole payload for `/evaluate`. The route validates, takes the session lock, calls it, and
returns the result verbatim. Driving the agent and building the payload are both steps of one
job — *produce the evaluate response* — and `ranking.py` names only the first step of it.

Git confirms the name drifted rather than the file growing:

    backend/evaluate.py  ->  backend/services/evaluate.py  ->  backend/services/ranking.py

It was renamed away from the name that described it. And the comment at
[process_player_data.py:405](backend/math/process_player_data.py#L405) still says `evaluate.py`,
which section B lists as stale — it is more accurate about the file's job than the filename is.

**Suggestion: rename back to `evaluate.py`** rather than split. That matches the endpoint, the
`EvaluateResponse` it returns, and the vocabulary the rest of the code already uses; and it fixes
one of the stale comments for free. `api/routers/ranking.py` would move with it, since the two
are named as a pair.

If the file is ever split anyway, the seam is obvious — the five `_build_*` functions are a
cluster, and are already flagged elsewhere as the next target for vectorisation.

## D. Dead code — essentially none

Of 23 module-level functions with no reference by name, 20 are FastAPI route handlers and pytest
hooks, reached by decorator rather than by name. The genuine leftovers are two 2-line test
helpers: [test_algorithms.py:610](testing_files/test_algorithms.py#L610) `_check_all_gradients_2`
and [test_experiments.py:197](testing_files/test_experiments.py#L197) `_short_names`.

No commented-out code blocks anywhere in `backend/` or `testing_files/`.

## E. Worth a look, not yet looked at

- **`get_darko_data` read `ESPN_PROJECTION_TABLE` where a `_VIEW` existed** — fixed 2026-09-25 in
  `c34d3ece`. The same shape exists elsewhere: `DARKO_PLAYER_TABLE` has a `DARKO_VIEW` beside it.
  Nothing has checked whether every reader picks the right one.
- **`cold-start-investigation`** has two unmerged commits from August about keeping heavy imports
  off the cold-start path. Unmerged work that sounds like it addressed a real problem.

## What this did not cover

Frontend TypeScript beyond file sizes; correctness of the algorithm; the platform integrations'
behaviour; anything in `site/` (build output) or `visualizations/` for dead code. Bugs were not
systematically hunted — one class of data-layer bug was found and fixed the same day, which is
the only reason to think others of that shape may exist.
