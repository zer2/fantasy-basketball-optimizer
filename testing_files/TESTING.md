# Testing Guide

This project has three tiers of tests. The offline tier runs anywhere with no setup. The other
two need either live API access or credentials, and this guide explains how to run them.

| Test | Tier | Requirements |
|---|---|---|
| `test_wnba_data.py::test_map_wnba_position` | Offline | none |
| `test_wnba_data.py::test_calculate_games_played_pct` | Offline | none |
| `test_espn_wnba.py::test_get_espn_league_class` | Offline | none |
| `test_espn_wnba.py::test_entry_game_abbrev` | Offline | none |
| `test_espn_wnba.py::test_filter_entries_for_league` | Offline | none |
| `test_espn_wnba.py::test_espn_contract_constants` | Offline | none |
| `test_espn_wnba.py::test_supported_leagues` | Offline | none |
| `test_espn_wnba.py::test_nba_league_offers_all_integrations` | Offline | none |
| `test_wnba_data.py::test_wnba_player_index_live` | Live WNBA API | `WNBA_LIVE=1` |
| `test_espn_wnba.py::test_wnba_league_offers_espn_only` | Live WNBA API | `WNBA_LIVE=1` |
| `test_algorithms.py::test_x_mu_gradients` | Snowflake | creds + populated views |
| `test_algorithms.py::test_objective_gradients` | Snowflake | creds + populated views |
| `test_app_setup.py::test_draft_defaults` | Broken | fails regardless of creds (see below) |
| ESPN league sync (WNBA + NBA) | Manual E2E | espn_s2/SWID + a real league |

Note: `test_algorithm_evaluation.py` fails at collection (it imports a `testing.simulation`
module that does not exist in the repo), so it is excluded from the commands below.

## Environment setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

pip will report a resolver conflict: `yfpy==15.0.3` pins `requests==2.31.0` while `nba-api`
requires `requests>=2.32.3`. The warning is safe to ignore — nba_api works with requests
2.31.0 in practice (the live WNBA tests pass with it).

**Always run tests and the app from the repo root.** `app.py` opens `parameters.yaml` with a
cwd-relative path, so running pytest from `testing_files/` makes every AppTest-based test fail
with confusing session-state errors instead of the real `FileNotFoundError`.

The `SPORT` env var only matters for code that imports `src/helpers/helper_functions.py`
outside a Streamlit runtime (the getters fall back to `os.environ['SPORT']` when
session state is empty). Setting `SPORT=NBA` for pytest runs keeps those fallbacks defined.

### Baseline (offline)

```bash
SPORT=NBA .venv/bin/python -m pytest testing_files/test_app_setup.py testing_files/test_algorithms.py testing_files/test_wnba_data.py testing_files/test_espn_wnba.py -q
```

Expected without credentials: **3 failed, 11 passed, 2 skipped**. The failures are
`test_draft_defaults` plus the two Snowflake-gated `test_algorithms` tests; the skips are the
two `WNBA_LIVE`-gated tests.

### Live WNBA API tier

```bash
WNBA_LIVE=1 SPORT=NBA .venv/bin/python -m pytest testing_files/test_wnba_data.py testing_files/test_espn_wnba.py -q
```

Expected: **10 passed**. These hit `stats.nba.com` endpoints with `LeagueID=10` (WNBA) and the
app-level WNBA flow through Streamlit's AppTest; they need internet access but no credentials.
The stats endpoints sometimes hang from datacenter IPs — rerun from a residential connection
if they time out.

## Snowflake-gated tests

The NBA data path reads Snowflake views. Without credentials the app crashes during data load,
`st.session_state.info` is never populated, and both `test_algorithms` tests fail with
`AttributeError: info not found in session_state`.

### Credentials

The connection (`get_snowflake_connection` in `src/helpers/helper_functions.py`) reads exactly
five env vars:

```bash
export SNOWFLAKE_ACCOUNT="<account identifier>"
export SNOWFLAKE_USER="<user>"
export SNOWFLAKE_PASSWORD="<password>"
export SNOWFLAKE_DATABASE="<database>"
export SNOWFLAKE_SCHEMA="FANTASYBASKETBALLOPTIMIZER"
```

Keep these in a file outside the repo (or gitignored) and `source` it. A read-only user with
`SELECT` on the schema is sufficient — the app only ever runs `SELECT * FROM <view>`.

**Schema quirk:** `get_data_from_snowflake(table_name, schema=...)` takes a schema argument,
but it is only used as a cache key — the actual connection always uses `SNOWFLAKE_SCHEMA` from
the environment. So the baseball path's `FANTASYBASEBALLOPTIMIZER` reads silently resolve
against whatever schema the env var names. For the NBA tests, set the env var to the NBA
schema as shown above.

### Smoke-check the credentials

```bash
.venv/bin/python -c "
import os, snowflake.connector
con = snowflake.connector.connect(
    account=os.getenv('SNOWFLAKE_ACCOUNT'), user=os.getenv('SNOWFLAKE_USER'),
    password=os.getenv('SNOWFLAKE_PASSWORD'), database=os.getenv('SNOWFLAKE_DATABASE'),
    schema=os.getenv('SNOWFLAKE_SCHEMA'))
print(con.cursor().execute('SELECT COUNT(*) FROM PLAYER_MAPPING_VIEW').fetchone())
"
```

### What the default test path actually needs

The AppTest runs load the app with default settings (NBA, projections, ESPN weight 0.5 and
DARKO weight 0.5). That path reads these views, which must exist and be populated:

- `ESPN_PROJECTION_VIEW` and `ESPN_PROJECTION_TABLE` — columns per the `espn-renamer` map in
  the NBA block of `parameters.yaml`, plus `ESPN_NAME`, `MINUTES_PLAYED`, `GAMES_PLAYED`,
  `POSITION` (read directly in `src/data_retrieval/get_data.py`)
- `DARKO_VIEW` — columns per the `darko-renamer` map
- `PLAYER_MAPPING_VIEW` — `YAHOO_PLAYER_ID`, `PLAYER_NAME`, and a name column per data source
  (`ESPN_NAME`, `DARKO_NAME`, `HTB_NAME`, `BBM_NAME`), used for cross-source name matching

`HISTORICAL_SEASONAL_AVERAGES_VIEW` is only read when the Historical data option is selected,
and `HTB_PROJECTION_TABLE` only when the Hashtag slider is nonzero — neither blocks the tests.

### Expected results with working credentials

```bash
SPORT=NBA .venv/bin/python -m pytest testing_files/test_app_setup.py testing_files/test_algorithms.py testing_files/test_wnba_data.py testing_files/test_espn_wnba.py -q
```

Expected: **1 failed, 13 passed, 2 skipped**. The two `test_algorithms` failures flip to pass.

`test_draft_defaults` still fails even with credentials: `parameters.yaml` leaves `max:` blank
(None) for `n_drafters`/`n_picks`/`beth`, and streamlit 1.48.1 reports unbounded number inputs
as `9007199254740991.0` rather than the `None` the test asserts. That is a pre-existing
test/streamlit-version mismatch, not an environment problem.

### Gotchas

- Snowflake reads are cached with `@st.cache_data(ttl='1d')` and the connection with
  `@st.cache_resource(ttl=3600)`. Within one long-lived process, view changes may not be
  visible for up to a day. Fresh pytest processes start with cold caches.
- `src/db_scripts/create_db.sql` and `create_views.sql` are DDL only — there are no data
  loaders in this repo, so a fresh Snowflake account gives you empty views. Populating them
  requires the maintainer's ingestion jobs.
- `AVERAGE_NUMBERS_VIEW` in `create_views.sql` hardcodes `COUNT(POINTS)/82` for games-played
  percent; a WNBA equivalent would need `/44` (2025-26 season length) or a per-season count.

## ESPN manual end-to-end (espn_s2 / SWID)

No automated test uses ESPN credentials. Validating the ESPN integration — especially the new
WNBA path — is a manual walkthrough.

**Security first:** `espn_s2` and `SWID` are ESPN session cookies granting full access to your
ESPN fantasy account. The app holds them in Streamlit session state only (in memory, per
browser session — nothing is written to disk), but treat them like passwords: never commit
them, never paste them into files inside the repo, and re-enter them each session.

### Getting the cookies

Log into espn.com in Chrome, then use the
[ESPN Cookie Finder extension](https://chromewebstore.google.com/detail/espn-cookie-finder/oapfffhnckhffnpiophbcmjnpomjkfcj)
(the same link the app shows in its auth dialog). SWID can be pasted with or without braces —
the app strips them.

### WNBA walkthrough

You need an ESPN Fantasy Women's Basketball league on your account (create one at
espn.com/fantasy/womens-basketball if needed).

```bash
.venv/bin/streamlit run app.py
```

1. Sidebar → League Settings → set the sport to **WNBA**.
2. The data source dropdown should offer exactly **Enter your own data** and **Retrieve from
   ESPN** (Yahoo and Fantrax are NBA-only and must not appear).
3. Select **Retrieve from ESPN**. The "Authenticate with ESPN" dialog should open; paste s2
   and SWID.
4. The league dropdown should list **only your WNBA leagues** (entries from ESPN's `wfba`
   game), each labeled with a single year, e.g. `My League (2026 Season)` — not `2025-2026`.
   NBA/football/etc. leagues must not appear. (Leagues from unrecognized ESPN games are kept
   deliberately rather than silently dropped, so an unexpected entry is not automatically a bug.)
5. Select your league. Mode should be **Season Mode** (the only ESPN mode).
6. In the Rosters tab, each fantasy team should appear as a column with its players named like
   `A'ja Wilson (C)` — name plus mapped positions.
7. **`RP` entries:** any roster player whose ESPN name doesn't exactly match a name from the
   WNBA stats API shows as `RP` (replacement player). Scan for `RP` cells — a few may be
   legitimate (players with no game data), but widespread `RP` indicates a name-matching
   problem worth reporting.
8. Exercise the Waiver Wire and Trading tabs: H-scores should compute without errors.

### NBA regression walkthrough

Repeat with the sport set to **NBA** and an ESPN NBA league:

1. Data source dropdown offers all four options: Enter your own data, Yahoo, Fantrax, ESPN.
2. The ESPN league dropdown is **not filtered** — it lists all your fantasy entries, as before.
3. League labels are cross-year (`2025-2026 Season`).
4. Switching the sport back and forth between NBA and WNBA while ESPN is selected must not
   crash — the data source resets to "Enter your own data" when the selected sport doesn't
   support the current integration.

## Adding automated credential-gated tests

Follow the existing gating pattern (from `testing_files/test_wnba_data.py`):

```python
@pytest.mark.skipif(os.environ.get('WNBA_LIVE') != '1', reason = 'set WNBA_LIVE=1 to run live API tests')
```

Proposed conventions for future work: `SNOWFLAKE_LIVE=1` for tests that assert the two
`test_algorithms` tests' preconditions (views reachable, expected columns present), and
`ESPN_LIVE=1` for tests that hit `lm-api-reads.fantasy.espn.com` with cookies from env vars
(`ESPN_S2`, `ESPN_SWID`) — never with hardcoded values.
