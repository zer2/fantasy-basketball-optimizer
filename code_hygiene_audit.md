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

1. **`backend/` root is a mixed bag.** Everything lives in a package except five modules sitting
   at the root: `data_retrieval.py` (368), `models.py` (117), `player_identity.py` (130),
   `parameters.py` (34) and `main.py` (142). `main.py` belongs at the root; the other four are
   shared domain modules that `backend/math/` reaches *up* to import. It is not a layering
   violation so much as an unnamed layer — they are the things every package depends on and
   nothing depends back on. A `backend/domain/` or `backend/core/` package would say so.

2. **Package sizes are lopsided.** `backend/math/` is 4893 lines across 8 files; every other
   package is under 2200. That is inherent to the subject rather than a mistake.

3. **Leading vs trailing commas.** CLAUDE.md asks for commas at the beginning of each parameter
   line. Measured across `backend/`: **372 leading-comma lines against 258 trailing-comma ones** —
   so the convention is followed about 59% of the time. `backend/api/helpers.py` is the clearest
   mixed case. (The count is approximate; the trailing-comma detector will pick up some ordinary
   call arguments.)

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
