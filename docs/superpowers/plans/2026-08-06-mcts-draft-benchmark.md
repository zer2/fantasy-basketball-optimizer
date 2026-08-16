# MCTS vs. H-Score Draft Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a headless, fixture-driven benchmark that measures whether MCTS lookahead beats the current one-ply H-score at drafting, scored by an independent Monte-Carlo of the real fantasy season.

**Architecture:** A standalone `benchmark/` package that imports zer2's engine **read-only** (no edits to `src/` or `app.py`). It pulls one season of real NBA game logs, derives season-average projections (agents draft on these) and keeps per-game box scores (an independent evaluator resamples these to score final rosters). Opponents — both the scored field and MCTS rollout opponents — use a shared weighted-softmax policy over a static ranking, with temperature `T` swept as the headline variable.

**Tech Stack:** Python 3.14, pandas, numpy, scipy, `nba_api`, streamlit (session-state only, headless), pytest. Reuses `src.math.process_player_data.process_player_data`, `src.data_retrieval.get_data.process_game_level_data`, `src.math.algorithm_agents.HAgent` / `get_default_h_values`, `src.math.position_optimization.check_single_player_eligibility`, and `src.helpers.helper_functions` getters.

**Spec:** `docs/superpowers/specs/2026-08-06-mcts-draft-benchmark-design.md`

---

## Hard Invariant (enforced every task)

**No file under `src/` or `app.py` may be modified.** The benchmark only *imports and calls* that code. The bootstrap sets `st.session_state` at runtime (as `app.py` does) — that is allowed. Every commit must pass:

```bash
git diff --name-only benchmark-base -- src/ app.py | grep -q . && echo "VIOLATION: src/ or app.py modified" || echo "OK: engine untouched"
```

> **Guard base:** this branch was cut from `espn-wnba-integration` (which already contains zer2's ESPN/WNBA edits under `src/`). Comparing against `main` would flag *his* changes as violations. The `benchmark-base` git ref is pinned at this branch's fork point, so the guard measures only *our* changes. Never diff against `main` for this check.

## File Structure

| File | Responsibility |
|---|---|
| `benchmark/__init__.py` | Package marker |
| `benchmark/config.py` | Frozen dataclasses: `LeagueConfig`, `ExperimentConfig`. All knobs (K, n_simulations, c_puct, temperatures, seed). |
| `benchmark/data.py` | Pull nba_api game logs → `(averages_df, gamelogs_df)`; snapshot/load fixture parquet. |
| `benchmark/bootstrap.py` | Populate `st.session_state` headlessly; build the `info` dict via `process_player_data`. |
| `benchmark/opponent_model.py` | `weighted_softmax_pick(ranking, available, temperature, rng, positions, team)` — the shared opponent policy. |
| `benchmark/agents.py` | `RandomAgent`, `GScoreAgent`, `HScoreAgent`, `MCTSAgent`. Common `make_pick(player_assignments, seat, rng)`. |
| `benchmark/draft.py` | Snake-draft engine: `run_draft(agents, config, rng) -> player_assignments`. |
| `benchmark/evaluate.py` | Independent season Monte-Carlo → per-seat EC/MC win-rate + CI. |
| `benchmark/experiment.py` | Orchestrate grid (field × format × T), common random numbers, aggregate, save results. |
| `benchmark/fixtures/` | Committed snapshot `nba_2025-26.parquet` (+ tiny synthetic fixtures for tests). |
| `benchmark/tests/` | pytest suite. |

**All test/run commands are executed from the repo root** with `SPORT=NBA` set (getters' `os.environ` fallback), using the project venv: `SPORT=NBA .venv/bin/python -m pytest ...`.

---

## Phase 0: Scaffolding

### Task 0: Package skeleton + config

**Files:**
- Create: `benchmark/__init__.py`, `benchmark/tests/__init__.py`, `benchmark/config.py`
- Test: `benchmark/tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_config.py
from benchmark.config import LeagueConfig, ExperimentConfig

def test_league_config_defaults():
    c = LeagueConfig()
    assert c.n_drafters == 12
    assert c.n_starters == 9          # standard 9-cat starters, no bench in the sim
    assert c.scoring_format == 'Head to Head: Each Category'
    # 9-cat: 7 counting + 2 ratio
    assert c.selected_categories == [
        'Field Goal %', 'Free Throw %',
        'Threes', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

def test_experiment_config_grid():
    e = ExperimentConfig()
    assert e.fields == ('gscore', 'hscore')
    assert e.formats == ('Head to Head: Each Category', 'Head to Head: Most Categories')
    assert len(e.temperatures) >= 3
    assert e.seed == 12345
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.config'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/__init__.py
# (empty package marker)
```
```python
# benchmark/tests/__init__.py
# (empty package marker)
```
```python
# benchmark/config.py
from dataclasses import dataclass, field

# Ratio categories first, then counting — matches process_player_data output ordering
# (calculate_scores_from_coefficients concatenates ratio_statistics + counting_statistics,
#  then reindexes to get_selected_categories()).
NINE_CAT = ['Field Goal %', 'Free Throw %',
            'Threes', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

@dataclass(frozen=True)
class LeagueConfig:
    league: str = 'NBA'
    season: str = '2025-26'
    n_drafters: int = 12
    n_starters: int = 9
    scoring_format: str = 'Head to Head: Each Category'
    selected_categories: list = field(default_factory=lambda: list(NINE_CAT))
    # H-score engine params — verified against parameters.yaml['NBA'].
    # omega/gamma/n_iterations come from the "Moderate punting" level (the app default
    # punting_default), NOT options.*.default. beth/psi/chi/aleph are options.*.default.
    omega: float = 0.7
    gamma: float = 0.25
    beth: float = 3.0
    psi: float = 0.8
    chi: float = 0.6
    n_iterations: int = 30
    aleph: float = 0.2
    third_round_reversal: bool = False

@dataclass(frozen=True)
class ExperimentConfig:
    fields: tuple = ('gscore', 'hscore')
    formats: tuple = ('Head to Head: Each Category', 'Head to Head: Most Categories')
    temperatures: tuple = (0.0, 0.5, 1.0, 2.0)   # T sweep; 0.0 == chalk/deterministic
    n_drafts: int = 24            # drafts per (field, format, T) cell before aggregation
    n_season_sims: int = 500      # evaluator bootstrap seasons per draft
    seed: int = 12345
    # MCTS knobs
    mcts_top_k: int = 15
    mcts_simulations: int = 200
    c_puct: float = 1.4
```

> **Provenance of engine params (verified):** `parameters.yaml['NBA']['options']` gives `beth.default=3`, `psi.default=0.8`, `chi.default=0.6`, `aleph.default=0.2`. `omega`/`gamma`/`n_iterations` are NOT `options.*.default` — the app reads them from `punting_defaults['Moderate punting']` = `{omega:0.7, gamma:0.25, n_iterations:30}` (the `punting_default`). `LeagueConfig` above uses these exact values.

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/__init__.py benchmark/tests/__init__.py benchmark/config.py benchmark/tests/test_config.py
git commit -m "feat(benchmark): package skeleton and config dataclasses"
```

---

## Phase 1: Data acquisition & fixture

### Task 1: Game-log pull → averages + per-game box scores

**Files:**
- Create: `benchmark/data.py`
- Test: `benchmark/tests/test_data.py`

Behavior: `build_datasets(gamelogs_raw, positions)` transforms a raw `nba_api` PlayerGameLogs frame (columns verified: `PLAYER_NAME, GAME_DATE, MIN, PTS, REB, AST, STL, BLK, TOV, FG3M, FGM, FGA, FTM, FTA, OREB, DREB, ...`) into:
1. `averages_df` — one row per player, in the shape `process_player_data` expects (reuse `process_game_level_data`).
2. `gamelogs_df` — tidy per-game rows keyed by player with the raw category components the evaluator needs (counting stats + made/attempt pairs for ratio stats).

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_data.py
import pandas as pd
from benchmark.data import build_datasets, RENAMER

def _fake_raw():
    # two players, two games each; columns mirror nba_api PlayerGameLogs
    rows = [
        # PLAYER_NAME, GAME_DATE, MIN, PTS, REB, AST, STL, BLK, TOV, FG3M, FGM, FGA, FTM, FTA, OREB, DREB
        ('A Player', '2025-11-01T00:00:00', 30, 20, 10, 5, 1, 1, 3, 2, 8, 15, 2, 4, 3, 7),
        ('A Player', '2025-11-03T00:00:00', 32, 24, 12, 7, 2, 0, 2, 3, 9, 16, 3, 3, 4, 8),
        ('B Player', '2025-11-01T00:00:00', 25, 10, 4, 8, 0, 0, 4, 1, 4, 10, 1, 2, 1, 3),
        ('B Player', '2025-11-03T00:00:00', 28, 14, 5, 9, 1, 1, 1, 0, 5, 11, 4, 5, 2, 3),
    ]
    cols = ['PLAYER_NAME','GAME_DATE','MIN','PTS','REB','AST','STL','BLK','TOV','FG3M','FGM','FGA','FTM','FTA','OREB','DREB']
    return pd.DataFrame(rows, columns=cols)

def test_build_datasets_averages_shape():
    positions = pd.Series({'A Player': 'PG,SG', 'B Player': 'C'}, name='Position')
    averages, gamelogs = build_datasets(_fake_raw(), positions)

    # averages indexed by "Name (Position)" like the app does
    assert any(idx.startswith('A Player (') for idx in averages.index)
    # ratio components present, computed as means of per-game makes/attempts
    assert 'Field Goal Attempts' in averages.columns
    assert 'Free Throw Attempts' in averages.columns
    assert 'Position' in averages.columns
    assert 'Games Played %' in averages.columns

def test_build_datasets_gamelogs_are_per_game():
    positions = pd.Series({'A Player': 'PG,SG', 'B Player': 'C'}, name='Position')
    _, gamelogs = build_datasets(_fake_raw(), positions)
    # one row per player-game (4 rows), keeping made/attempt pairs for ratio evaluation
    assert len(gamelogs) == 4
    for col in ['Points','Rebounds','Assists','Steals','Blocks','Turnovers','Threes',
                'Field Goals Made','Field Goal Attempts','Free Throws Made','Free Throw Attempts']:
        assert col in gamelogs.columns
    assert set(gamelogs['Player'].unique()) == {'A Player', 'B Player'}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/data.py
"""Pull real NBA game logs and derive (a) season-average projections and
(b) per-game box scores for the independent evaluator. Read-only use of the engine."""
import os
import pandas as pd

from src.data_retrieval.get_data import process_game_level_data

# nba_api PlayerGameLogs -> app category names. Matches parameters.yaml current-season-api-renamer,
# extended with the columns the evaluator needs per game.
RENAMER = {
    'PLAYER_NAME': 'Player', 'GAME_DATE': 'Game Date', 'MIN': 'MIN',
    'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists', 'STL': 'Steals',
    'BLK': 'Blocks', 'TOV': 'Turnovers', 'FG3M': 'Threes',
    'FTA': 'Free Throw Attempts', 'FTM': 'Free Throws Made',
    'FGA': 'Field Goal Attempts', 'FGM': 'Field Goals Made',
    'OREB': 'Off Rebounds', 'DREB': 'Def Rebounds',
}

# Per-game columns the evaluator resamples (counting cats + ratio components).
GAMELOG_STAT_COLS = [
    'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers', 'Threes',
    'Off Rebounds', 'Def Rebounds',
    'Field Goals Made', 'Field Goal Attempts', 'Free Throws Made', 'Free Throw Attempts',
]

def _rename(raw: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in RENAMER if c in raw.columns]
    return raw[keep].rename(columns={k: RENAMER[k] for k in keep})

def build_datasets(gamelogs_raw: pd.DataFrame, positions: pd.Series):
    """Return (averages_df, gamelogs_df).

    averages_df: one row per "Name (Position)", shape process_player_data expects.
    gamelogs_df: tidy per-game rows with counting cats and ratio components.
    """
    renamed = _rename(gamelogs_raw).fillna(0)

    # (a) Season averages via the engine's own converter (keeps ratio math consistent).
    metadata = positions.reindex(pd.unique(renamed['Player'])).fillna('NP')
    metadata.name = 'Position'
    averages = process_game_level_data(renamed.copy(), metadata)

    # (b) Per-game box scores for the evaluator (Player column retained, not indexed).
    stat_cols = [c for c in GAMELOG_STAT_COLS if c in renamed.columns]
    gamelogs = renamed[['Player'] + stat_cols].reset_index(drop=True)

    return averages, gamelogs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_data.py -v`
Expected: PASS (2 passed)

> If `process_game_level_data` returns empty (its bare `except` swallows column-name errors), print `renamed.columns` and reconcile against the failing `.mean()`/ratio step — do not edit `get_data.py`; adjust `RENAMER` here.

- [ ] **Step 5: Commit**

```bash
git add benchmark/data.py benchmark/tests/test_data.py
git commit -m "feat(benchmark): transform nba_api game logs into averages + per-game datasets"
```

### Task 2: Fixture pull + load (network, one-time)

**Files:**
- Modify: `benchmark/data.py` (add `pull_and_snapshot`, `load_fixture`)
- Test: `benchmark/tests/test_fixture_roundtrip.py`

- [ ] **Step 1: Write the failing test** (round-trip only; no network in tests)

```python
# benchmark/tests/test_fixture_roundtrip.py
import pandas as pd
from benchmark.data import save_fixture, load_fixture

def test_fixture_roundtrip(tmp_path):
    averages = pd.DataFrame({'Points': [20.0], 'Position': ['C'], 'Games Played %': [1.0]},
                            index=pd.Index(['X (C)'], name='Player'))
    gamelogs = pd.DataFrame({'Player': ['X', 'X'], 'Points': [18, 22]})
    path = tmp_path / 'fix.parquet'
    save_fixture(averages, gamelogs, path)
    a2, g2 = load_fixture(path)
    pd.testing.assert_frame_equal(a2, averages)
    pd.testing.assert_frame_equal(g2, gamelogs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_fixture_roundtrip.py -v`
Expected: FAIL — `ImportError: cannot import name 'save_fixture'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmark/data.py`)

```python
# --- append to benchmark/data.py ---

def save_fixture(averages: pd.DataFrame, gamelogs: pd.DataFrame, path) -> None:
    """Store both frames in one parquet file under distinct keys via a MultiIndex marker."""
    import pyarrow  # noqa: F401  (fail loudly if parquet engine missing)
    averages.to_parquet(str(path).replace('.parquet', '.averages.parquet'))
    gamelogs.to_parquet(str(path).replace('.parquet', '.gamelogs.parquet'))

def load_fixture(path):
    averages = pd.read_parquet(str(path).replace('.parquet', '.averages.parquet'))
    gamelogs = pd.read_parquet(str(path).replace('.parquet', '.gamelogs.parquet'))
    return averages, gamelogs

def pull_and_snapshot(season: str, out_path: str):
    """One-time network pull of a full season. Not run in tests."""
    from nba_api.stats.endpoints import playergamelogs, playerindex
    raw = playergamelogs.PlayerGameLogs(
        league_id_nullable='00', season_nullable=season,
        season_type_nullable='Regular Season', timeout=60).get_data_frames()[0]
    idx = playerindex.PlayerIndex(
        league_id='00', season=season, timeout=60).get_data_frames()[0]
    idx['Player'] = idx['PLAYER_FIRST_NAME'] + ' ' + idx['PLAYER_LAST_NAME']
    positions = idx.drop_duplicates('Player').set_index('Player')['POSITION'].fillna('NP')
    positions.name = 'Position'
    averages, gamelogs = build_datasets(raw, positions)
    save_fixture(averages, gamelogs, out_path)
    return averages, gamelogs
```

> **Position mapping:** the app maps raw `POSITION` letters (G/F/C) to `PG,SG`/`SF,PF`/`C` via `parameters.yaml['rotowire-position-adjuster']` in the WNBA path. For the NBA fixture, apply the same adjuster so eligibility works with `base_list = [PG,SG,SF,PF,C]`. If `playerindex` positions are already comma-form, keep as-is. Verify the fixture has non-`NP` positions for the top ~120 players before committing; if raw positions are single letters, map them here (not in `src/`).

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_fixture_roundtrip.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Build the real fixture (manual, network) and commit it**

```bash
SPORT=NBA .venv/bin/python -c "from benchmark.data import pull_and_snapshot; pull_and_snapshot('2025-26','benchmark/fixtures/nba_2025-26.parquet')"
SPORT=NBA .venv/bin/python -c "from benchmark.data import load_fixture; a,g=load_fixture('benchmark/fixtures/nba_2025-26.parquet'); print('players:',len(a),'game-rows:',len(g)); print(a['Position'].value_counts().head())"
```
Expected: ~500+ players in averages, ~26k game rows, positions populated (not all `NP`).

```bash
git add benchmark/data.py benchmark/tests/test_fixture_roundtrip.py benchmark/fixtures/nba_2025-26.averages.parquet benchmark/fixtures/nba_2025-26.gamelogs.parquet
git commit -m "feat(benchmark): fixture pull/save/load + committed 2025-26 snapshot"
```

---

## Phase 2: Headless bootstrap

### Task 3: Populate session state + build `info`

**Files:**
- Create: `benchmark/bootstrap.py`
- Test: `benchmark/tests/test_bootstrap.py`

Behavior: `bootstrap_session(averages, league_cfg)` populates `st.session_state` with exactly the keys the getters read, injects `averages` as `player_stats_v2`, and calls `process_player_data` to produce and store the `info` dict. Returns `info`.

Session-state keys required (verified against `helper_functions.py`): `data_source='Enter your own data'`, `mode='Draft Mode'`, `league`, `scoring_format`, `omega`, `gamma`, `psi`, `chi`, `beth`, `n_iterations`, `aleph`, `third_round_reversal`, `params` (from `parameters.yaml[league]`), `selected_categories`, `n_picks`, `n_bench=0`, `team_names`, `n_<position>` counts, `data_dictionary`, `all_params`, `styler=None`, `base='light'`.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_bootstrap.py
from benchmark.config import LeagueConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session

def test_bootstrap_builds_info():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    info = bootstrap_session(averages, LeagueConfig())
    for key in ['G-scores', 'X-scores', 'Positions', 'v'] if 'v' in info else ['G-scores', 'X-scores', 'Positions']:
        assert key in info
    # G-scores has a Total column and 9 category columns
    gs = info['G-scores']
    assert 'Total' in gs.columns
    # at least a full league's worth of rankable players
    assert len(gs) >= 12 * 9

def test_bootstrap_getters_resolve():
    import streamlit as st
    from src.helpers.helper_functions import get_selected_categories, get_n_drafters, get_scoring_format
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    bootstrap_session(averages, LeagueConfig())
    assert get_n_drafters() == 12
    assert get_scoring_format() == 'Head to Head: Each Category'
    assert len(get_selected_categories()) == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_bootstrap.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.bootstrap'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/bootstrap.py
"""Populate Streamlit session state headlessly so zer2's engine runs outside the app.
Sets runtime state only — never edits src/."""
import os
import yaml
import streamlit as st

from src.helpers.helper_functions import gen_key, store_dataset_in_session_state
from src.math.process_player_data import process_player_data
from benchmark.config import LeagueConfig

def _load_all_params():
    with open('parameters.yaml', 'r') as f:
        return yaml.safe_load(f)

def bootstrap_session(averages, cfg: LeagueConfig):
    os.environ['SPORT'] = cfg.league
    ss = st.session_state
    all_params = _load_all_params()
    params = all_params[cfg.league]

    ss['all_params'] = all_params
    ss['params'] = params
    ss['data_source'] = 'Enter your own data'
    ss['mode'] = 'Draft Mode'
    ss['league'] = cfg.league
    ss['scoring_format'] = cfg.scoring_format
    ss['omega'] = cfg.omega
    ss['gamma'] = cfg.gamma
    ss['psi'] = cfg.psi
    ss['chi'] = cfg.chi
    ss['beth'] = cfg.beth
    ss['n_iterations'] = cfg.n_iterations
    ss['aleph'] = cfg.aleph
    ss['third_round_reversal'] = cfg.third_round_reversal
    ss['selected_categories'] = list(cfg.selected_categories)
    ss['n_picks'] = cfg.n_starters
    ss['n_bench'] = 0
    ss['team_names'] = ['Drafter ' + str(i + 1) for i in range(cfg.n_drafters)]
    ss['styler'] = None
    ss['base'] = 'light'
    ss['data_dictionary'] = {}

    # Position slot counts the getters read as n_<code>. Standard 9-cat single-slot layout.
    position_counts = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
                       'G': 1, 'F': 1, 'Util': 2}
    for code, n in position_counts.items():
        ss['n_' + code] = n

    # Inject projections as player_stats_v2.
    store_dataset_in_session_state(averages, 'player_stats_v2', gen_key())

    # Build info via the engine (weekly_df=None -> uses averages path).
    info, key = process_player_data(
        None, gen_key(), cfg.psi, cfg.chi, cfg.scoring_format,
        cfg.n_drafters, cfg.n_starters, params, list(cfg.selected_categories))
    store_dataset_in_session_state(info, 'info', key)
    return info
```

> **`n_starters` vs. position slots:** the slot counts must sum to `n_starters` (1+1+1+1+1+1+1+2 = 9). If you change `n_starters`, change `position_counts` to match, or `HAgent`'s position optimizer will assert. Verify `sum(position_counts.values()) == cfg.n_starters` — add an `assert` in `bootstrap_session`.
>
> **If direct session-state population is brittle** (Streamlit sometimes rejects writes outside a script run), fall back to the `AppTest` pattern from `testing_files/test_algorithms.py`: `AppTest.from_file("app.py")`, set widget values, `.run()`, then read `at.session_state.info`. Prefer the direct path; only switch if a test surfaces a `StreamlitAPIException`.

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_bootstrap.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/bootstrap.py benchmark/tests/test_bootstrap.py
git commit -m "feat(benchmark): headless session-state bootstrap and info builder"
```

---

## Phase 3: Opponent model & simple agents

### Task 4: Weighted-softmax opponent policy

**Files:**
- Create: `benchmark/opponent_model.py`
- Test: `benchmark/tests/test_opponent_model.py`

Behavior: `weighted_softmax_pick(ranking, available, temperature, rng, positions, team_players)` selects one player. `ranking` is a Series (player → score, higher better). It restricts to `available`, keeps position-eligible players (via `check_single_player_eligibility`), takes the top-K by score, and samples with `P ∝ exp(score/T)`. `T == 0` → deterministic argmax.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_opponent_model.py
import numpy as np
import pandas as pd
from benchmark.opponent_model import weighted_softmax_pick

def _ranking():
    return pd.Series({'P1': 10.0, 'P2': 9.0, 'P3': 8.0, 'P4': 1.0})

def test_zero_temperature_is_argmax():
    rng = np.random.default_rng(0)
    pick = weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                 temperature=0.0, rng=rng, positions=None, team_players=[])
    assert pick == 'P1'

def test_respects_availability():
    rng = np.random.default_rng(0)
    pick = weighted_softmax_pick(_ranking(), available=['P2','P3','P4'],
                                 temperature=0.0, rng=rng, positions=None, team_players=[])
    assert pick == 'P2'

def test_high_temperature_sometimes_picks_nonmax():
    rng = np.random.default_rng(1)
    picks = {weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                   temperature=5.0, rng=rng, positions=None, team_players=[])
             for _ in range(50)}
    assert len(picks) > 1   # stochastic: not always the same player

def test_top_k_limits_candidates():
    rng = np.random.default_rng(2)
    picks = {weighted_softmax_pick(_ranking(), available=['P1','P2','P3','P4'],
                                   temperature=5.0, rng=rng, positions=None, team_players=[], top_k=2)
             for _ in range(50)}
    assert picks <= {'P1', 'P2'}   # P3/P4 never sampled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_opponent_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.opponent_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/opponent_model.py
"""Shared weighted-softmax opponent policy. Used by the scored field AND MCTS rollouts."""
import numpy as np

from src.math.position_optimization import check_single_player_eligibility

def _eligible(candidates, positions, team_players):
    if positions is None:
        return list(candidates)
    team_positions = positions.loc[[p for p in team_players if p in positions.index]] \
        if team_players else positions.loc[[]]
    return [p for p in candidates
            if p in positions.index and check_single_player_eligibility(positions.loc[p], team_positions)]

def weighted_softmax_pick(ranking, available, temperature, rng, positions, team_players, top_k=15):
    """Return one player from `available`, sampled ∝ exp(score/T) over the top-K eligible."""
    avail = [p for p in ranking.index if p in set(available)]        # ranking order (desc)
    avail = _eligible(avail, positions, team_players)
    if not avail:
        return None
    avail = avail[:top_k]
    scores = ranking.loc[avail].to_numpy(dtype=float)
    if temperature <= 0:
        return avail[int(np.argmax(scores))]
    z = scores / temperature
    z = z - z.max()
    w = np.exp(z)
    w = w / w.sum()
    return str(rng.choice(avail, p=w))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_opponent_model.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/opponent_model.py benchmark/tests/test_opponent_model.py
git commit -m "feat(benchmark): weighted-softmax opponent policy with eligibility + top-K"
```

### Task 5: Random / G-score / H-score agents

**Files:**
- Create: `benchmark/agents.py`
- Test: `benchmark/tests/test_agents.py`

Behavior: all agents implement `make_pick(player_assignments, seat, rng) -> player`. `RandomAgent` picks a uniform eligible available player. `GScoreAgent` uses the weighted-softmax policy over total G-score. `HScoreAgent` uses the softmax policy over a **cached static H-score ordering** (from `get_default_h_values`) — this is the cheap field policy the spec calls for, not live gradient descent per pick.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_agents.py
import numpy as np
import pandas as pd
from benchmark.config import LeagueConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.agents import RandomAgent, GScoreAgent, HScoreAgent

def _setup():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    info = bootstrap_session(averages, LeagueConfig())
    return info

def test_gscore_agent_picks_available_player():
    info = _setup()
    agent = GScoreAgent(info, temperature=0.0)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    # deterministic: highest total G-score
    top = info['G-scores'].sort_values('Total', ascending=False).index[0]
    assert pick == top

def test_agent_never_repicks_taken_player():
    info = _setup()
    agent = GScoreAgent(info, temperature=0.0)
    top = info['G-scores'].sort_values('Total', ascending=False).index[0]
    assignments = {0: [top], 1: [], 2: []}
    for i in range(3, 12):
        assignments[i] = []
    pick = agent.make_pick(assignments, 1, np.random.default_rng(0))
    assert pick != top

def test_hscore_agent_uses_cached_ordering():
    info = _setup()
    agent = HScoreAgent(info, LeagueConfig(), temperature=0.0)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    assert pick in info['G-scores'].index
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_agents.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.agents'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/agents.py
"""Draft agents. All expose make_pick(player_assignments, seat, rng) -> player."""
import numpy as np

from src.math.algorithm_agents import get_default_h_values
from src.helpers.helper_functions import gen_key
from benchmark.opponent_model import weighted_softmax_pick, _eligible

def _all_taken(player_assignments):
    return [p for v in player_assignments.values() for p in v if p == p]

class RandomAgent:
    def __init__(self, info):
        self.positions = info['Positions']
        self.pool = list(info['G-scores'].index)

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = _eligible([p for p in self.pool if p not in taken],
                          self.positions, player_assignments[seat])
        return str(rng.choice(avail)) if avail else None

class GScoreAgent:
    def __init__(self, info, temperature=0.0, top_k=15):
        self.ranking = info['G-scores']['Total'].sort_values(ascending=False)
        self.positions = info['Positions']
        self.temperature = temperature
        self.top_k = top_k

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = [p for p in self.ranking.index if p not in taken]
        return weighted_softmax_pick(self.ranking, avail, self.temperature, rng,
                                     self.positions, player_assignments[seat], self.top_k)

class HScoreAgent:
    """Field/hero agent using a cached static H-score ordering (cheap; no per-pick descent)."""
    def __init__(self, info, cfg, temperature=0.0, top_k=15):
        h = get_default_h_values(
            gen_key(), cfg.omega, cfg.gamma, cfg.n_starters, cfg.n_drafters,
            cfg.n_iterations, cfg.beth, cfg.scoring_format)
        self.ranking = h.set_index('Player')['H-score'].sort_values(ascending=False)
        self.positions = info['Positions']
        self.temperature = temperature
        self.top_k = top_k

    def make_pick(self, player_assignments, seat, rng):
        taken = set(_all_taken(player_assignments))
        avail = [p for p in self.ranking.index if p not in taken]
        return weighted_softmax_pick(self.ranking, avail, self.temperature, rng,
                                     self.positions, player_assignments[seat], self.top_k)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_agents.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/agents.py benchmark/tests/test_agents.py
git commit -m "feat(benchmark): Random, G-score, and cached-H-score field agents"
```

---

## Phase 4: Draft engine

### Task 6: Snake-draft runner

**Files:**
- Create: `benchmark/draft.py`
- Test: `benchmark/tests/test_draft.py`

Behavior: `run_draft(seat_agents, n_drafters, n_starters, rng)` runs a snake draft (no third-round reversal in the benchmark), returning `player_assignments: {seat -> [players in pick order]}`. Snake logic reimplemented locally (not imported from `src/tabs/drafting.py`) to keep the UI untouched.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_draft.py
import numpy as np
from benchmark.draft import run_draft, snake_seat_order

def test_snake_order_first_two_rounds():
    # 3 drafters, 2 rounds: 0,1,2 then 2,1,0
    order = snake_seat_order(n_drafters=3, n_starters=2)
    assert order == [0, 1, 2, 2, 1, 0]

class _SeqAgent:
    """Picks the lowest-numbered unused fake player; deterministic for structure tests."""
    def __init__(self, pool): self.pool = pool
    def make_pick(self, player_assignments, seat, rng):
        taken = {p for v in player_assignments.values() for p in v}
        for p in self.pool:
            if p not in taken:
                return p

def test_run_draft_fills_all_rosters():
    pool = [f'P{i}' for i in range(100)]
    agents = {s: _SeqAgent(pool) for s in range(12)}
    result = run_draft(agents, n_drafters=12, n_starters=9, rng=np.random.default_rng(0))
    assert len(result) == 12
    assert all(len(r) == 9 for r in result.values())
    # no duplicate players across the whole draft
    allp = [p for v in result.values() for p in v]
    assert len(allp) == len(set(allp)) == 12 * 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_draft.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.draft'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/draft.py
"""Snake-draft engine. Snake logic mirrors move_forward_one_pick (no third-round reversal),
reimplemented here so the UI module is never imported."""

def snake_seat_order(n_drafters, n_starters):
    order = []
    for rnd in range(n_starters):
        seats = range(n_drafters) if rnd % 2 == 0 else reversed(range(n_drafters))
        order.extend(seats)
    return order

def run_draft(seat_agents, n_drafters, n_starters, rng):
    player_assignments = {s: [] for s in range(n_drafters)}
    for seat in snake_seat_order(n_drafters, n_starters):
        pick = seat_agents[seat].make_pick(player_assignments, seat, rng)
        if pick is None:
            raise RuntimeError(f'Seat {seat} could not find an eligible pick')
        player_assignments[seat].append(pick)
    return player_assignments
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_draft.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/draft.py benchmark/tests/test_draft.py
git commit -m "feat(benchmark): snake-draft engine"
```

---

## Phase 5: Independent evaluator

### Task 7: Weekly bootstrap + category scoring

**Files:**
- Create: `benchmark/evaluate.py`
- Test: `benchmark/tests/test_evaluate.py`

Behavior: `simulate_week(team_players, gamelogs, rng)` samples one real game row per player, sums counting cats, and volume-weights ratio cats (sum makes / sum attempts across the week). `did-not-play` is realized by including the player's true rate of missed games (sample from all season slots; if the sampled slot is a non-game, contribute zeros).

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_evaluate.py
import numpy as np
import pandas as pd
from benchmark.evaluate import simulate_week, team_week_totals

def _logs():
    return pd.DataFrame({
        'Player': ['A','A','B','B'],
        'Points': [10, 20, 30, 30],
        'Rebounds': [5, 5, 1, 1],
        'Assists': [0,0,0,0], 'Steals':[0,0,0,0], 'Blocks':[0,0,0,0], 'Turnovers':[0,0,0,0], 'Threes':[0,0,0,0],
        'Field Goals Made': [5, 10, 12, 12], 'Field Goal Attempts': [10, 10, 24, 24],
        'Free Throws Made': [0,0,0,0], 'Free Throw Attempts': [0,0,0,0],
    })

def test_ratio_is_volume_weighted():
    # Force A's game0 (5/10) and B's game0 (12/24) => FG% = (5+12)/(10+24) = 17/34 = 0.5
    logs = _logs()
    totals = team_week_totals(['A','B'], logs, rng=np.random.default_rng(0),
                              n_games_per_week=1, categories=['Points','Field Goal %'])
    # with 1 game each, deterministic-ish; assert FG% equals volume-weighted, not mean of (0.5, 0.5)
    assert abs(totals['Field Goal %'] - 0.5) < 1e-9

def test_counting_totals_sum_across_players():
    logs = _logs()
    totals = team_week_totals(['A','B'], logs, rng=np.random.default_rng(0),
                              n_games_per_week=2, categories=['Points'])
    # 2 games each; A in {10,20}, B in {30,30} -> min 10+30+30 ... just assert positive & bounded
    assert totals['Points'] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/evaluate.py
"""Independent Monte-Carlo of the fantasy season on resampled REAL game logs.
Never calls the H-score objective."""
import numpy as np
import pandas as pd

RATIO_COMPONENTS = {
    'Field Goal %': ('Field Goals Made', 'Field Goal Attempts'),
    'Free Throw %': ('Free Throws Made', 'Free Throw Attempts'),
}
NEGATIVE = {'Turnovers'}

def team_week_totals(team_players, gamelogs, rng, n_games_per_week, categories):
    """Sum a team's category totals over one simulated week."""
    by_player = {p: g for p, g in gamelogs.groupby('Player')}
    counting = [c for c in categories if c not in RATIO_COMPONENTS]
    totals = {c: 0.0 for c in counting}
    made = {c: 0.0 for c in categories if c in RATIO_COMPONENTS}
    att = {c: 0.0 for c in categories if c in RATIO_COMPONENTS}

    for p in team_players:
        if p not in by_player:
            continue
        rows = by_player[p]
        idx = rng.integers(0, len(rows), size=n_games_per_week)
        sampled = rows.iloc[idx]
        for c in counting:
            if c in sampled.columns:
                totals[c] += float(sampled[c].sum())
        for c in made:
            m, a = RATIO_COMPONENTS[c]
            made[c] += float(sampled[m].sum())
            att[c] += float(sampled[a].sum())

    for c in made:
        totals[c] = made[c] / att[c] if att[c] > 0 else 0.0
    return totals

def simulate_week(team_players, gamelogs, rng, n_games_per_week=3,
                  categories=None):
    return team_week_totals(team_players, gamelogs, rng, n_games_per_week, categories)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_evaluate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluate.py benchmark/tests/test_evaluate.py
git commit -m "feat(benchmark): weekly bootstrap with volume-weighted ratio categories"
```

### Task 8: Season schedule → per-seat EC/MC win-rate

**Files:**
- Modify: `benchmark/evaluate.py` (add `score_season`, `evaluate_rosters`)
- Test: `benchmark/tests/test_season_scoring.py`

Behavior: `evaluate_rosters(player_assignments, gamelogs, cfg, exp_cfg, rng)` runs `n_season_sims` seasons. Each season: for each week, every team plays a round-robin opponent; compute per-matchup category outcomes. EC = fraction of categories won (averaged over opponents & weeks). MC = fraction of weeks where a team won a majority of categories. Returns `{seat: {'EC': winrate, 'MC': winrate}}` plus CIs.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_season_scoring.py
import numpy as np
from benchmark.evaluate import compare_categories

def test_compare_categories_counts_wins_with_negative_turnovers():
    a = {'Points': 100, 'Turnovers': 5}
    b = {'Points': 90,  'Turnovers': 8}
    # A wins Points (higher) and wins Turnovers (lower is better) => 2-0
    wins_a, wins_b, ties = compare_categories(a, b, ['Points', 'Turnovers'], negative={'Turnovers'})
    assert (wins_a, wins_b, ties) == (2, 0, 0)

def test_compare_categories_tie():
    a = {'Points': 100}; b = {'Points': 100}
    wins_a, wins_b, ties = compare_categories(a, b, ['Points'], negative=set())
    assert (wins_a, wins_b, ties) == (0, 0, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_season_scoring.py -v`
Expected: FAIL — `ImportError: cannot import name 'compare_categories'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmark/evaluate.py`)

```python
# --- append to benchmark/evaluate.py ---

def compare_categories(totals_a, totals_b, categories, negative=NEGATIVE):
    wins_a = wins_b = ties = 0
    for c in categories:
        va, vb = totals_a.get(c, 0.0), totals_b.get(c, 0.0)
        if c in negative:
            va, vb = -va, -vb
        if va > vb:
            wins_a += 1
        elif vb > va:
            wins_b += 1
        else:
            ties += 1
    return wins_a, wins_b, ties

def _round_robin_pairs(n):
    return [(i, j) for i in range(n) for j in range(n) if i < j]

def evaluate_rosters(player_assignments, gamelogs, cfg, exp_cfg, rng, weeks=20):
    """Return {seat: {'EC': rate, 'MC': rate, 'EC_ci': half, 'MC_ci': half}}."""
    seats = list(player_assignments.keys())
    n = len(seats)
    cats = list(cfg.selected_categories)
    gpw = 3  # n_games_per_week (NBA default from parameters.yaml)
    pairs = _round_robin_pairs(n)

    ec = np.zeros((exp_cfg.n_season_sims, n))
    mc = np.zeros((exp_cfg.n_season_sims, n))

    for s in range(exp_cfg.n_season_sims):
        ec_num = np.zeros(n); ec_den = np.zeros(n)
        mc_num = np.zeros(n); mc_den = np.zeros(n)
        for _w in range(weeks):
            week_totals = {seat: simulate_week(player_assignments[seat], gamelogs, rng, gpw, cats)
                           for seat in seats}
            for i, j in pairs:
                wa, wb, t = compare_categories(week_totals[i], week_totals[j], cats)
                # EC: share of categories won (ties = half)
                ec_num[i] += wa + 0.5 * t; ec_den[i] += len(cats)
                ec_num[j] += wb + 0.5 * t; ec_den[j] += len(cats)
                # MC: one matchup win to the category majority
                mc_den[i] += 1; mc_den[j] += 1
                if wa > wb: mc_num[i] += 1
                elif wb > wa: mc_num[j] += 1
                else: mc_num[i] += 0.5; mc_num[j] += 0.5
        ec[s] = ec_num / ec_den
        mc[s] = mc_num / mc_den

    out = {}
    for k, seat in enumerate(seats):
        out[seat] = {
            'EC': float(ec[:, k].mean()),
            'MC': float(mc[:, k].mean()),
            'EC_ci': float(1.96 * ec[:, k].std(ddof=1) / np.sqrt(exp_cfg.n_season_sims)),
            'MC_ci': float(1.96 * mc[:, k].std(ddof=1) / np.sqrt(exp_cfg.n_season_sims)),
        }
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_season_scoring.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/evaluate.py benchmark/tests/test_season_scoring.py
git commit -m "feat(benchmark): round-robin season scoring for EC and MC"
```

---

## Phase 6: MCTS agent

### Task 9: Leaf evaluator (reuse HAgent full-roster branch)

**Files:**
- Create: `benchmark/mcts.py`
- Test: `benchmark/tests/test_mcts_leaf.py`

Behavior: `leaf_value(hagent, player_assignments, seat)` returns the scalar team win-rate from `HAgent`'s `n_players_selected == n_picks` branch — the existing full-roster objective. This is the *search heuristic*, distinct from the independent evaluator.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_mcts_leaf.py
import numpy as np
from benchmark.config import LeagueConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.mcts import leaf_value
from src.math.algorithm_agents import HAgent
from src.helpers.helper_functions import get_data_from_session_state

def _full_rosters(info, cfg):
    pool = list(info['G-scores'].index)
    assignments, k = {}, 0
    for seat in range(cfg.n_drafters):
        assignments[seat] = pool[k:k + cfg.n_starters]; k += cfg.n_starters
    return assignments

def test_leaf_value_is_scalar_probability():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    H = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
               n_drafters=cfg.n_drafters, dynamic=False, beth=cfg.beth,
               scoring_format=cfg.scoring_format)
    assignments = _full_rosters(info, cfg)
    v = leaf_value(H, assignments, 0)
    assert 0.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_mcts_leaf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.mcts'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/mcts.py
"""MCTS draft agent. Reuses HAgent's full-roster objective as the search leaf (read-only)."""
import numpy as np

def leaf_value(hagent, player_assignments, seat):
    """Scalar team win-rate for a COMPLETE roster via HAgent's n_picks branch.
    get_h_scores yields once; with a full roster the score index is [''] and holds the team score."""
    gen = hagent.get_h_scores(player_assignments, seat)
    res = next(gen)
    scores = res['Scores']
    return float(scores.iloc[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_mcts_leaf.py -v`
Expected: PASS (1 passed)

> If `leaf_value` errors, inspect `perform_iterations` case `n_players_selected == self.n_picks` (algorithm_agents.py:624): it sets `result_index = ['']` and returns one score. Confirm the yielded `res['Scores']` is a 1-element Series; index by position, not label.

- [ ] **Step 5: Commit**

```bash
git add benchmark/mcts.py benchmark/tests/test_mcts_leaf.py
git commit -m "feat(benchmark): MCTS leaf evaluator reusing HAgent full-roster objective"
```

### Task 10: PUCT search over candidate picks

**Files:**
- Modify: `benchmark/mcts.py` (add `MCTSAgent`)
- Test: `benchmark/tests/test_mcts_agent.py`

Behavior: `MCTSAgent.make_pick(player_assignments, seat, rng)` runs `n_simulations` PUCT iterations. Root actions = top-K available by cached H-score ordering. Each simulation: choose a root candidate by PUCT (prior = softmax of H-score), roll the rest of the draft to completion using the shared weighted-softmax opponent model (hero included, past the root), evaluate the full hero roster via `leaf_value`, backprop. Returns the most-visited root candidate.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_mcts_agent.py
import numpy as np
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.mcts import MCTSAgent

def test_mcts_returns_eligible_available_pick():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    agent = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    empty = {i: [] for i in range(12)}
    pick = agent.make_pick(empty, 0, np.random.default_rng(0))
    assert pick in info['G-scores'].index

def test_mcts_is_deterministic_under_fixed_seed():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    a1 = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    a2 = MCTSAgent(info, cfg, temperature=1.0, n_simulations=20, top_k=10, c_puct=1.4)
    empty = {i: [] for i in range(12)}
    p1 = a1.make_pick(empty, 0, np.random.default_rng(42))
    p2 = a2.make_pick(empty, 0, np.random.default_rng(42))
    assert p1 == p2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_mcts_agent.py -v`
Expected: FAIL — `ImportError: cannot import name 'MCTSAgent'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmark/mcts.py`)

```python
# --- append to benchmark/mcts.py ---
from copy import deepcopy

from src.math.algorithm_agents import HAgent, get_default_h_values
from src.helpers.helper_functions import gen_key
from benchmark.draft import snake_seat_order
from benchmark.opponent_model import weighted_softmax_pick, _eligible

def _all_taken(pa):
    return [p for v in pa.values() for p in v if p == p]

class MCTSAgent:
    def __init__(self, info, cfg, temperature=1.0, n_simulations=200, top_k=15, c_puct=1.4):
        self.info = info
        self.cfg = cfg
        self.positions = info['Positions']
        self.temperature = temperature
        self.n_simulations = n_simulations
        self.top_k = top_k
        self.c_puct = c_puct
        h = get_default_h_values(gen_key(), cfg.omega, cfg.gamma, cfg.n_starters,
                                 cfg.n_drafters, cfg.n_iterations, cfg.beth, cfg.scoring_format)
        self.ranking = h.set_index('Player')['H-score'].sort_values(ascending=False)
        # One HAgent for leaf scoring (dynamic=False -> single-pass full-roster objective).
        self.hagent = HAgent(info=info, omega=cfg.omega, gamma=cfg.gamma, n_picks=cfg.n_starters,
                             n_drafters=cfg.n_drafters, dynamic=False, beth=cfg.beth,
                             scoring_format=cfg.scoring_format)

    def _candidates(self, player_assignments, seat):
        taken = set(_all_taken(player_assignments))
        avail = [p for p in self.ranking.index if p not in taken]
        avail = _eligible(avail, self.positions, player_assignments[seat])
        return avail[:self.top_k]

    def _priors(self, candidates):
        s = self.ranking.loc[candidates].to_numpy(dtype=float)
        t = max(self.temperature, 1e-6)
        z = s / t; z = z - z.max(); w = np.exp(z)
        return w / w.sum()

    def _rollout(self, player_assignments, seat, first_pick, rng):
        """Play the draft to completion from `seat` taking `first_pick`, then softmax for all."""
        pa = deepcopy(player_assignments)
        pa[seat] = pa[seat] + [first_pick]
        # Remaining picks in snake order after the current one.
        order = snake_seat_order(self.cfg.n_drafters, self.cfg.n_starters)
        # advance past picks already made (count of taken) + this one
        made = len(_all_taken(player_assignments)) + 1
        for s in order[made:]:
            pick = weighted_softmax_pick(self.ranking, [p for p in self.ranking.index
                                                        if p not in set(_all_taken(pa))],
                                         self.temperature, rng, self.positions, pa[s], self.top_k)
            if pick is not None:
                pa[s] = pa[s] + [pick]
        return leaf_value(self.hagent, pa, seat)

    def make_pick(self, player_assignments, seat, rng):
        candidates = self._candidates(player_assignments, seat)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        priors = self._priors(candidates)
        N = np.zeros(len(candidates))
        W = np.zeros(len(candidates))
        for _ in range(self.n_simulations):
            total = N.sum()
            Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
            u = self.c_puct * priors * np.sqrt(total + 1) / (1 + N)
            a = int(np.argmax(Q + u))
            value = self._rollout(player_assignments, seat, candidates[a], rng)
            N[a] += 1; W[a] += value
        return candidates[int(np.argmax(N))]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_mcts_agent.py -v`
Expected: PASS (2 passed)

> **Performance:** with `dynamic=False`, each `leaf_value` is one analytic pass (no gradient descent). If a single `make_pick` at `n_simulations=200` exceeds ~a few seconds, lower `n_simulations` in `ExperimentConfig` — correctness is unaffected, only estimate variance.

- [ ] **Step 5: Commit**

```bash
git add benchmark/mcts.py benchmark/tests/test_mcts_agent.py
git commit -m "feat(benchmark): PUCT MCTS agent with softmax rollouts and H-score priors"
```

---

## Phase 7: Orchestration

### Task 11: One matched draft with common random numbers

**Files:**
- Create: `benchmark/experiment.py`
- Test: `benchmark/tests/test_experiment_cell.py`

Behavior: `run_matched_draft(info, cfg, field, fmt, temperature, hero_seat, seed)` builds a field of the given type, places the hero at `hero_seat`, and runs the draft **twice** — once hero=H-score, once hero=MCTS — using the **same seed** so field draws are identical (common random numbers). Returns the two `player_assignments`.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_experiment_cell.py
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.experiment import run_matched_draft

def test_matched_draft_same_field_picks():
    averages, _ = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    # TINY MCTS budget: the CRN property (identical non-hero picks) is independent of
    # simulation count. ExperimentConfig()'s default mcts_simulations=200 would run
    # two full drafts x 9 hero picks x 200 sims = minutes and trip the workflow watchdog.
    exp = ExperimentConfig(mcts_simulations=6, mcts_top_k=8)
    hero_h, hero_m = run_matched_draft(info, cfg, exp,
                                       field='gscore', fmt=cfg.scoring_format,
                                       temperature=1.0, hero_seat=3, seed=7)
    # Non-hero seats identical across the two runs (common random numbers)
    for seat in hero_h:
        if seat != 3:
            assert hero_h[seat] == hero_m[seat]
    # Hero seat rosters may differ (different strategy)
    assert 3 in hero_h and 3 in hero_m
```

> **Runtime discipline (applies to every MCTS-invoking test):** tests must use a tiny MCTS budget (`mcts_simulations` ≤ ~8, `mcts_top_k` ≤ ~8) so each finishes in seconds. Full production simulation counts belong only in Task 15's real run, never in the test suite. The Task 12 grid smoke test already follows this.

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_experiment_cell.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'benchmark.experiment'`

- [ ] **Step 3: Write minimal implementation**

```python
# benchmark/experiment.py
"""Orchestrate the benchmark grid: field x format x temperature, with common random numbers."""
import numpy as np

from benchmark.agents import GScoreAgent, HScoreAgent
from benchmark.mcts import MCTSAgent
from benchmark.draft import run_draft

def _make_field(info, cfg, field, temperature, exp_cfg):
    if field == 'gscore':
        return lambda: GScoreAgent(info, temperature=temperature, top_k=exp_cfg.mcts_top_k)
    elif field == 'hscore':
        return lambda: HScoreAgent(info, cfg, temperature=temperature, top_k=exp_cfg.mcts_top_k)
    raise ValueError(field)

def run_matched_draft(info, cfg, exp_cfg, field, fmt, temperature, hero_seat, seed):
    field_factory = _make_field(info, cfg, field, temperature, exp_cfg)

    def build(hero_agent):
        agents = {s: field_factory() for s in range(cfg.n_drafters)}
        agents[hero_seat] = hero_agent
        return agents

    hero_h_agent = HScoreAgent(info, cfg, temperature=0.0, top_k=exp_cfg.mcts_top_k)
    hero_m_agent = MCTSAgent(info, cfg, temperature=temperature,
                             n_simulations=exp_cfg.mcts_simulations,
                             top_k=exp_cfg.mcts_top_k, c_puct=exp_cfg.c_puct)

    # Common random numbers: identical rng seed => identical field draws in both runs.
    res_h = run_draft(build(hero_h_agent), cfg.n_drafters, cfg.n_starters, np.random.default_rng(seed))
    res_m = run_draft(build(hero_m_agent), cfg.n_drafters, cfg.n_starters, np.random.default_rng(seed))
    return res_h, res_m
```

> **CRN caveat:** identical field picks require that the hero's own picks not perturb the shared rng stream differently between runs. Because both heroes and all field agents draw from the same `rng`, a hero that consumes a different number of rng draws will desynchronize the field. To guarantee identical field picks, give **each field agent its own per-seat rng** seeded from `(seed, seat)` and reserve the shared draw only for that seat. If the test's "non-hero seats identical" assertion fails, refactor `run_draft` to pass `rng_for_seat(seat)` — a `np.random.default_rng([seed, seat])` — instead of one shared rng. Implement that per-seat rng now if the assertion fails.

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_experiment_cell.py -v`
Expected: PASS (1 passed) — apply the per-seat rng refactor from the caveat if needed.

- [ ] **Step 5: Commit**

```bash
git add benchmark/experiment.py benchmark/tests/test_experiment_cell.py
git commit -m "feat(benchmark): matched-draft runner with common random numbers"
```

### Task 12: Full grid + result aggregation + CLI

**Files:**
- Modify: `benchmark/experiment.py` (add `run_experiment`, `main`)
- Test: `benchmark/tests/test_experiment_grid.py`

Behavior: `run_experiment(exp_cfg, cfg, fixture_path)` loops the grid `(field × format × temperature)`, runs `n_drafts` matched drafts (rotating `hero_seat`), evaluates each with the independent evaluator, and aggregates hero EC/MC win-rate for H-score vs. MCTS with CIs. `main()` runs a small smoke config and writes `benchmark/results/<timestamp>.json` (timestamp passed in, not generated, for reproducibility).

- [ ] **Step 1: Write the failing test** (tiny config for speed)

```python
# benchmark/tests/test_experiment_grid.py
from dataclasses import replace
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.experiment import run_experiment

def test_grid_smoke_produces_rows():
    small = ExperimentConfig(fields=('gscore',),
                             formats=('Head to Head: Each Category',),
                             temperatures=(1.0,),
                             n_drafts=2, n_season_sims=10,
                             mcts_simulations=8, mcts_top_k=8)
    results = run_experiment(small, LeagueConfig(), 'benchmark/fixtures/nba_2025-26.parquet')
    row = results[('gscore', 'Head to Head: Each Category', 1.0)]
    assert 'hscore_hero' in row and 'mcts_hero' in row
    assert 'EC' in row['hscore_hero'] and 'EC' in row['mcts_hero']
    assert 0.0 <= row['mcts_hero']['EC'] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_experiment_grid.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_experiment'`

- [ ] **Step 3: Write minimal implementation** (append to `benchmark/experiment.py`)

```python
# --- append to benchmark/experiment.py ---
import json
import numpy as np
from dataclasses import replace

from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.evaluate import evaluate_rosters

def _aggregate(hero_ec, hero_mc):
    arr_ec = np.array(hero_ec); arr_mc = np.array(hero_mc)
    def stat(a):
        return {'mean': float(a.mean()),
                'ci': float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0}
    return {'EC': stat(arr_ec)['mean'], 'EC_ci': stat(arr_ec)['ci'],
            'MC': stat(arr_mc)['mean'], 'MC_ci': stat(arr_mc)['ci']}

def run_experiment(exp_cfg, cfg, fixture_path):
    averages, gamelogs = load_fixture(fixture_path)
    results = {}
    for fmt in exp_cfg.formats:
        cfg_fmt = replace(cfg, scoring_format=fmt)
        info = bootstrap_session(averages, cfg_fmt)
        for field in exp_cfg.fields:
            for T in exp_cfg.temperatures:
                h_ec, h_mc, m_ec, m_mc = [], [], [], []
                for d in range(exp_cfg.n_drafts):
                    hero_seat = d % cfg.n_drafters
                    seed = exp_cfg.seed + d
                    res_h, res_m = run_matched_draft(info, cfg_fmt, exp_cfg, field, fmt, T, hero_seat, seed)
                    eval_rng = np.random.default_rng([exp_cfg.seed, d])
                    ev_h = evaluate_rosters(res_h, gamelogs, cfg_fmt, exp_cfg, eval_rng)
                    eval_rng = np.random.default_rng([exp_cfg.seed, d])  # same season draws
                    ev_m = evaluate_rosters(res_m, gamelogs, cfg_fmt, exp_cfg, eval_rng)
                    key = 'EC' if fmt.endswith('Each Category') else 'MC'
                    h_ec.append(ev_h[hero_seat]['EC']); h_mc.append(ev_h[hero_seat]['MC'])
                    m_ec.append(ev_m[hero_seat]['EC']); m_mc.append(ev_m[hero_seat]['MC'])
                results[(field, fmt, T)] = {
                    'hscore_hero': _aggregate(h_ec, h_mc),
                    'mcts_hero': _aggregate(m_ec, m_mc),
                    'delta_EC': float(np.mean(m_ec) - np.mean(h_ec)),
                    'delta_MC': float(np.mean(m_mc) - np.mean(h_mc)),
                }
    return results

def main(timestamp='manual'):
    results = run_experiment(ExperimentConfig(), LeagueConfig(),
                             'benchmark/fixtures/nba_2025-26.parquet')
    serializable = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in results.items()}
    import os
    os.makedirs('benchmark/results', exist_ok=True)
    with open(f'benchmark/results/{timestamp}.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    for k, v in serializable.items():
        print(k, 'ΔEC=%+.3f ΔMC=%+.3f' % (v['delta_EC'], v['delta_MC']))
    return results

if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_experiment_grid.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add benchmark/experiment.py benchmark/tests/test_experiment_grid.py
git commit -m "feat(benchmark): full experiment grid, aggregation, and CLI entrypoint"
```

---

## Phase 8: Validation (evaluator trust) & full run

### Task 13: Sanity checks — the evaluator can detect known skill gaps

**Files:**
- Create: `benchmark/tests/test_sanity.py`

These are the spec's success-criterion-3 checks. They validate the evaluator BEFORE we trust MCTS numbers: a skilled agent must beat a random one, and H-score must beat G-score against a G-score field.

- [ ] **Step 1: Write the failing test**

```python
# benchmark/tests/test_sanity.py
import numpy as np
from benchmark.config import LeagueConfig, ExperimentConfig
from benchmark.data import load_fixture
from benchmark.bootstrap import bootstrap_session
from benchmark.agents import RandomAgent, GScoreAgent, HScoreAgent
from benchmark.draft import run_draft
from benchmark.evaluate import evaluate_rosters

def test_gscore_beats_random_field():
    averages, gamelogs = load_fixture('benchmark/fixtures/nba_2025-26.parquet')
    cfg = LeagueConfig()
    info = bootstrap_session(averages, cfg)
    exp = ExperimentConfig(n_season_sims=200)
    # Seat 0 = G-score hero; rest random
    agents = {s: RandomAgent(info) for s in range(cfg.n_drafters)}
    agents[0] = GScoreAgent(info, temperature=0.0)
    result = run_draft(agents, cfg.n_drafters, cfg.n_starters, np.random.default_rng(1))
    ev = evaluate_rosters(result, gamelogs, cfg, exp, np.random.default_rng(2))
    others = np.mean([ev[s]['EC'] for s in range(1, cfg.n_drafters)])
    assert ev[0]['EC'] > others   # skill beats randomness by EC win-rate
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/test_sanity.py -v`
Expected: initially may fail if evaluator wiring has a bug; fix `evaluate.py`/`agents.py` (NOT `src/`) until it PASSES. This test failing is the signal the benchmark is untrustworthy.

- [ ] **Step 3: Commit**

```bash
git add benchmark/tests/test_sanity.py
git commit -m "test(benchmark): sanity checks that the evaluator detects skill gaps"
```

### Task 14: Engine-untouched guard + full suite + README

**Files:**
- Create: `benchmark/README.md`
- Test: `benchmark/tests/test_no_src_edits.py`

- [ ] **Step 1: Write the guard test**

```python
# benchmark/tests/test_no_src_edits.py
import subprocess

def test_src_and_app_untouched():
    # Compare against benchmark-base (this branch's fork point), NOT main:
    # the parent branch already carries zer2's src/ edits; only our changes should be measured.
    out = subprocess.run(['git', 'diff', '--name-only', 'benchmark-base', '--', 'src/', 'app.py'],
                         capture_output=True, text=True).stdout.strip()
    assert out == '', f'Engine files modified: {out}'
```

- [ ] **Step 2: Run the full suite**

Run: `SPORT=NBA .venv/bin/python -m pytest benchmark/tests/ -v`
Expected: all pass, including the guard.

- [ ] **Step 3: Write `benchmark/README.md`**

```markdown
# Draft Strategy Benchmark

Measures whether MCTS lookahead beats one-ply H-score at drafting, scored by an
independent Monte-Carlo of the real 2025-26 NBA fantasy season.

**Additive only:** nothing under `src/` or `app.py` is modified; the engine is
imported read-only. `benchmark/tests/test_no_src_edits.py` enforces this.

## Run
```bash
# rebuild the fixture (network, one-time):
SPORT=NBA .venv/bin/python -c "from benchmark.data import pull_and_snapshot; pull_and_snapshot('2025-26','benchmark/fixtures/nba_2025-26.parquet')"
# full benchmark:
SPORT=NBA .venv/bin/python -m benchmark.experiment
# tests:
SPORT=NBA .venv/bin/python -m pytest benchmark/tests/ -v
```

## Reading results
Each grid cell reports hero EC/MC win-rate for H-score vs. MCTS and the deltas
(`ΔEC`, `ΔMC`) as a function of field type and opponent temperature `T`.
Positive delta = MCTS beats H-score. The headline is the delta-vs-`T` curve.
```

- [ ] **Step 4: Commit**

```bash
git add benchmark/README.md benchmark/tests/test_no_src_edits.py
git commit -m "docs(benchmark): README + engine-untouched guard test"
```

### Task 15: Full experiment run (manual, produces the answer)

- [ ] **Step 1: Run the full grid**

Run: `SPORT=NBA .venv/bin/python -m benchmark.experiment`
Expected: `benchmark/results/<timestamp>.json` written; per-cell `ΔEC`/`ΔMC` printed.

- [ ] **Step 2: Interpret**

- If `ΔEC`/`ΔMC` are ≤ 0 within CI across all `T`: MCTS does not beat H-score here — the honest negative result. Stop; do not pursue UI integration.
- If deltas turn positive as `T` rises: lookahead pays off under draft-room uncertainty — quantify at which `T`, and only then consider a live-tool follow-up (separate project).

- [ ] **Step 3: Commit results**

```bash
git add benchmark/results/
git commit -m "chore(benchmark): first full-grid results"
```

---

## Self-Review Notes (addressed)

- **Spec coverage:** data/fixture (T1–2), headless bootstrap (T3), shared opponent model (T4), agents incl. RandomAgent floor (T5), snake draft (T6), independent evaluator with volume-weighted ratios + EC/MC (T7–8), MCTS reusing HAgent leaf + PUCT + softmax rollouts (T9–10), grid with T-sweep + CRN (T11–12), sanity checks (T13), additive-only guard (T14), full run (T15). All spec sections mapped.
- **Non-goals honored:** no streaming/waivers/Roto/UI/live-data anywhere in tasks.
- **Type consistency:** `make_pick(player_assignments, seat, rng)` uniform across all agents; `leaf_value(hagent, player_assignments, seat)` and `MCTSAgent` share it; `evaluate_rosters` returns `{seat: {'EC','MC','EC_ci','MC_ci'}}` consumed unchanged by `run_experiment`.
- **Known risk flagged inline:** CRN rng-desync (T11 caveat) and headless session-state brittleness (T3 note) each carry a concrete fallback.
