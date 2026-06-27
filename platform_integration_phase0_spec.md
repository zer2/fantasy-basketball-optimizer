# Phase 0 Spec — Fantrax Live Integration

Port the Streamlit `src/platform_integration/` (a `PlatformIntegration` ABC + ESPN/Yahoo/Fantrax
subclasses) into the FastAPI backend + TypeScript frontend. Phase 0 wires **Fantrax** end-to-end as
the proving vertical slice: it needs only a league ID (no auth), so it exercises the whole pipeline
without credential/persistence complexity.

Phasing: **Phase 0 Fantrax** → Phase 1 ESPN (Season only) → Phase 2 Yahoo (OAuth + live draft).

## Verified facts this spec rests on
- `fantraxapi 0.2.9` is installed; exposes `FantraxAPI(league_id, session=None)` and a private
  `_request(method, **kwargs)` (plus public `teams`/`roster_info`/`standings`/`positions`). All
  platform deps (`yfpy`, `fantraxapi`, `espn_api`, `yahoo_oauth`, `yahoo_fantasy_api`) are already in
  `requirements.txt`.
- `player_assignments` is `Record<teamName, playerName[]>`, built column-per-team
  (`frontend/data_entry/draft_state.ts:134`) — matches the Streamlit "column per team, row per pick"
  DataFrame.
- `info['Positions']` exists (`backend/math/process_player_data.py:354`), values are position-code
  lists (`player_means['Position'].str.split(',')`).
- Canonical names carry a position suffix: `"Nikola Jokic (C)"`, `"Bam Adebayo (C,PF)"`. Mapping =
  raw platform name → append `(POS,…)` from `info['Positions']`, else `'RP'`.

## End-to-end data flow
```
User picks "Retrieve from Fantrax", enters league_id
        │
        ▼
GET  /platforms/fantrax/divisions?league_id=…      → [{name,id}] (often empty)
POST /platforms/fantrax/connect {league_id,division_id?}
        │   (validates league, fetches team metadata — no session.info needed)
        ▼   returns {team_names, n_drafters, n_picks, available_modes}
Frontend populates league settings, restricts mode selector to available_modes
        │
        ▼
POST /sessions  (existing)  + platform_config{league_id,division_id}
        │   pipeline stores PlatformConfig on the Session
        ▼
"Refresh Analysis" →  GET /sessions/{id}/draft-state
        │   uses session.platform_config + session.info (name mapping)
        ▼   {player_assignments, injured_players, status}
Frontend → existing /evaluate (Draft) or populates roster grid (Season)
```

## Backend

### 1. New package `backend/platform_integration/`
Keeps the class design per CLAUDE.md, but classes are **framework-agnostic** — no `streamlit`, no
`fastapi`, no `st.*`. UI/control-flow that lived in `setup()` moves to the routes + frontend. Layout
(concrete integrations isolated in a subpackage; the dependency-free root + registry above them):

```
backend/platform_integration/
    base.py          # PlatformIntegration ABC + dataclasses (the dependency-free root)
    helpers.py       # deduplicate_team_names, rosters_df_to_player_assignments, build_platform_name_lookup
    registry.py      # get_integration / is_live_platform  (imports the concrete integrations)
    integrations/
        fantrax.py   # FantraxIntegration
```
Import direction is one-way: `integrations/* → base, helpers`; `registry → base, integrations/*`;
`main`/`session` → `base`, `helpers`, `registry`. `registry` stays separate from `base` because
folding it in would create a base ← subclass ← base cycle.

**`base.py` — the ABC (+ `LeagueShape` / `PlatformConfig` / `PlatformSelections` dataclasses):**
```python
class PlatformIntegration(abc.ABC):
    @property
    @abc.abstractmethod
    def available_modes(self) -> list[str]: ...
    @property
    @abc.abstractmethod
    def description_string(self) -> str: ...
    @property
    @abc.abstractmethod
    def player_name_column(self) -> str: ...

    # Data methods — pure, no UI. Credentials passed in (empty dict for Fantrax).
    @abc.abstractmethod
    def list_divisions(self, league_id: str, credentials: dict) -> list[dict]: ...
    @abc.abstractmethod
    def fetch_league_shape(
        self
        , league_id: str
        , division_id: Optional[str]
        , credentials: dict
    ) -> "LeagueShape": ...                       # team_names, n_drafters, n_picks, teams_dict
    @abc.abstractmethod
    def get_draft_results(
        self
        , config: "PlatformConfig"
        , mode: str
        , info: dict
    ) -> tuple[pd.DataFrame, str]: ...             # (cols=teams, rows=picks, NaN=open), status
    @abc.abstractmethod
    def get_auction_results(
        self
        , config: "PlatformConfig"
        , mode: str
        , info: dict
    ) -> Optional[tuple[pd.DataFrame, str]]: ...   # None where unsupported (Fantrax)
```

Dataclasses:
```python
@dataclass
class LeagueShape:
    team_names: list[str]
    n_drafters: int
    n_picks: int
    teams_dict: dict[str, str]      # team_name -> team_id

@dataclass
class PlatformConfig:
    platform: str
    league_id: str
    division_id: Optional[str]
    teams_dict: dict[str, str]
    player_name_column: str

@dataclass
class PlatformSelections:        # returned by get_draft_results / get_auction_results
    player_assignments: dict[str, list[str]]   # {team: [canonical 'Name (POS)', ...]} — the /evaluate shape
    status: str
    injured_players: list[str]   # Season Mode only; returned explicitly, not mutated into global state
    costs: Optional[dict[str, list[float]]] = None   # auction only; costs[team][i] ↔ player_assignments[team][i]
```

The package also contains `registry.py` (`get_integration(platform, credentials=None)` /
`is_live_platform(platform)`). The ABC's `__init__(credentials)` puts credentials on the instance
(Yahoo's `{'auth_dir': ...}`; Fantrax ignores), and a concrete `list_leagues()` (default `[]`) serves
auth platforms. `get_draft_results` returns a `PlatformSelections` rather than the Streamlit
`(df, status)` tuple, so
the injured-player list (a Streamlit `st.session_state` side effect) is returned explicitly. It builds
the assignments **dict directly** (no intermediate DataFrame — the only consumer wants this shape) and
takes a prebuilt `name_lookup` (see §2) rather than `info`.

**`integrations/fantrax.py` — ported method-by-method (drop every `st.*`, `@st.cache_data`):**

| Streamlit method | Port |
|---|---|
| `get_api(league_id)` | `FantraxAPI(league_id)` per call (cheap, stateless) |
| `get_division_dict` | `list_divisions`: `api._request("getStandings", view="SCHEDULE")['displayedLists']['tabs']`, filter out `['All','Combined','Results','Season Stats','Playoffs']` → `[{name,id}]` |
| `get_teams_dict_by_division` | inside `fetch_league_shape`; **dedup fixed** (see deviations) |
| `get_n_picks` | `api._request("getTeamRosterInfo", teamId=<first>)['miscData']['statusTotals']` → `min(sum(max where name!='Inj Res'), 16)` |
| `get_team_info` | `api._request("getTeamRosterInfo", teamId=…)['tables'][0]['rows']` |
| `get_rosters_df` | per team, for rows containing `'scorer'`: map name; `statusId in exclusions` (`('3',)` Season / `()` Draft) → injured set, else roster; build column-per-team DataFrame |
| `get_draft_results` | `return rosters_df, 'Success'` |
| `get_auction_results` | `return None` |

`available_modes = ['Draft Mode', 'Season Mode']`, `player_name_column = 'FANTRAX_PLAYER_NAME'`,
`description_string = 'Retrieve from Fantrax'`.

> Caching: the Streamlit `@st.cache_data(ttl=3600)` is dropped. division/shape are fetched on connect
> (rare); live rosters must **not** be cached (they change during a draft). A short TTL can be added
> later — out of scope for Phase 0.

### 2. Name mapping — `helpers.build_platform_name_lookup` (PLAYER_MAPPING_VIEW-backed)
The Streamlit `get_fixed_player_name` only strips/re-appends the position suffix — it assumes the
platform's spelling already equals the canonical name. That's wrong in general: `FANTRAX_PLAYER_NAME`
exists as a *separate* column precisely because spellings differ, so unmapped players silently become
`'RP'`. Streamlit actually reconciles names a layer down — `get_data.py`'s `map_player_names` re-keys
the whole dataset into the platform's naming via `PLAYER_MAPPING_VIEW`, using `player_name_column`
(which is why every integration declares it). The backend keeps its dataset canonical instead, so we
do the equivalent mapping at roster-read time and **consume `player_name_column`** there:

```python
def build_platform_name_lookup(info, player_name_column, mapping_view) -> dict[str, str]:
    # compose PLAYER_MAPPING_VIEW[player_name_column -> PLAYER_NAME] with
    # info['Positions'][canonical base -> position codes]; build a
    # {platform_name -> 'Name (POS)'} dict once. lookup.get(name, 'RP') = miss fallback.
```

`mapping_view` comes from `data_retrieval.get_player_mapping_view()` (cached `_query`). **Confirmed**
from the view DDL: `PLAYER_MAPPING_VIEW` exposes `PLAYER_NAME` and `FANTRAX_PLAYER_NAME` (alongside
`ESPN_NAME`, `ROTOWIRE_NAME`, `YAHOO_PLAYER_ID`, …). A missing/renamed column would still raise → 502
(fail-noisily). Note for Phase 2: Yahoo's column is `YAHOO_PLAYER_ID` (an id, not a name).

**Built once, not per poll.** The lookup is a function of `info['Positions']` (the player set) and the
platform's `player_name_column` (fixed per session). `info['Positions']` only changes on data/injured
patches, so `main._refresh_platform_name_lookup(session)` rebuilds it and caches it on
`session.platform_name_lookup` right after `run_pipeline` — unconditionally at session creation, and in
the PATCH route only when `from_step <= 2`. It lives in the route layer (not `run_pipeline`) so the
pipeline stays platform-agnostic. `get_draft_results` then just consumes `session.platform_name_lookup`
(`name_lookup.get(raw, 'RP')`), so the integration no longer touches `info` or the mapping view.

### 3. Session additions — `backend/session.py`
Add `platform_config: Optional[PlatformConfig] = None` to the `Session` dataclass. It rides the
existing in-memory store and the eviction sweep. No persistence needed for Fantrax.

### 4. Routes + Pydantic models
The route handlers live **directly in `backend/main.py`** (with the other `@app.*` routes), not in a
separate `platform_routes.py`. A separate routes module would have to import `get_session` and the
`_load_all_params`/`_build_current_params` helpers from `main.py`, while `main.py` would import the
routes module to register them — a circular import, which CLAUDE.md forbids. Instead, all the heavy
logic (integration classes, the `registry`, the DataFrame→assignments converter) lives in the
`backend/platform_integration/` package, which `main.py` imports one-way; the route handlers stay
thin and own all HTTP/error handling, while the integration classes stay HTTP-free. Platform lookup
goes through `backend/platform_integration/registry.py` (`get_integration(platform)` /
`is_live_platform(platform)`).

```
GET  /platforms/fantrax/divisions?league_id=…
     → { divisions: [{name, id}] }

POST /platforms/fantrax/connect   { league_id, division_id? }
     → { team_names, n_drafters, n_picks, available_modes }
       (404/422 on bad league id; no session required)

GET  /sessions/{id}/draft-state
     → { player_assignments: {team: [canonicalName…]}, injured_players: [...], status }
       (uses session.platform_config + session.info; mode from current_params)
```

`backend/models.py` additions:
```python
class PlatformConfigRequest(BaseModel):
    league_id: str
    division_id: Optional[str] = None

# SessionRequest gains:
    platform_config: Optional[PlatformConfigRequest] = None

class DivisionsResponse(BaseModel):
    divisions: list[dict]            # [{name, id}]

class ConnectResponse(BaseModel):
    team_names: list[str]
    n_drafters: int
    n_picks: int
    available_modes: list[str]

class DraftStateResponse(BaseModel):
    player_assignments: dict[str, list[str]]
    injured_players: list[str]
    status: str
```

The pipeline session-creation path reads `platform_config` and stores a `PlatformConfig` on the Session.

### 5. Roster shape — dict, no DataFrame
`get_draft_results` builds `player_assignments` (`{team: [player, ...]}`) directly, which is exactly
the `/evaluate` shape, so there is **no DataFrame and no converter** — the draft-state route returns
`state.player_assignments` as-is. (An earlier draft built a NaN-padded DataFrame and converted it back
to a dict via a `rosters_df_to_player_assignments` helper; both were removed as a pointless round-trip.)

## Frontend
There is **no separate `platform_session.ts`**: routing all three concerns through one new module
would create an import cycle (`league_settings → platform_session → draft_and_auction_session →
league_settings`, since `draft_and_auction_session` already imports `getLeagueSettings`). Same
no-circular-deps rationale as the backend routes. Instead the pieces fold into existing files:

- **`client.ts`** (raw endpoints, consistent with "all fetch lives here"): `fetchDivisions(platform,
  leagueId)`, `connectPlatform(platform, leagueId, divisionId)`, `fetchDraftState(sessionId, mode)`,
  plus the `PlatformConnectResponse` / `DraftStateResponse` types.
- **`league_settings.ts`**: when a live platform is selected, `updateVisibility` reveals a connect cell
  (league-ID input + division dropdown populated via `fetchDivisions` + Connect button + status line).
  Connect calls `connectPlatform`, then sets `n_drafters`/`n_picks` directly (NOT via a change event,
  which would reset the team-name rows) and writes `team_names` into the hidden `#ls-team-names`
  textarea, dispatching its `input` event so the existing seat-selector machinery (`main.ts:139`)
  repopulates. It also restricts the mode selector to `available_modes`. Exports
  `getPlatformConfig()` → `{league_id, division_id}` | null.
- **`draft_and_auction_session.ts`**: holds `livePlayerAssignments` + `setLivePlayerAssignments` /
  `clearLivePlayerAssignments`; `evaluateSeat` uses those instead of the manual board when the platform
  is live; new exported `refreshLiveAnalysis()` = `withSessionRetry(fetchDraftState → set assignments)`
  then `runEvaluate()`. This is where the "Refresh Analysis" orchestration lives (avoids the cycle).
- **`layout.ts` `showLiveLayout()`**: replaces the stub with a **"Refresh Analysis"** button wired to
  `refreshLiveAnalysis()`, and shows the candidate table (`#hscoretable`).
- **`session.ts` `startFreshSession`**: includes `platform_config` from `getPlatformConfig()` in the
  `SessionRequest` when a live platform is connected.
- **`types.ts`**: `SessionRequest.platform_config?: {league_id, division_id?}`.
- Buttons reuse the existing `.section-apply-btn` CSS class (no new CSS), per the styles.css reuse rule.
- The existing `evaluate` flow is reused unchanged — live mode just supplies `player_assignments` from
  the poll instead of the manual board. The `{platform}` route segment is the full platform label,
  URL-encoded.

## Deviations from the Streamlit code (intentional)
1. **Team-name dedup — FIXED for Fantrax.** Streamlit's Fantrax path builds `{name: id}`
   (`fantrax_integration.py:126`) and passes it to `adjust_teams_dict_for_duplicate_names`, which
   dedupes *values* — correct only for the Yahoo orientation `{id: name}`
   (`yahoo_integration.py:293`). For Fantrax this is a no-op (ids are unique) AND the duplicate-name
   collision has already happened in the name-keyed comprehension, silently dropping a team. The port
   dedupes on the **name**, from raw `(name, id)` pairs, before building the dict — applied to BOTH
   the `getFantasyTeams` branch (`:126`) and the division `standings` branch (`:133`, which doesn't
   dedupe at all in Streamlit):
   ```python
   def deduplicate_team_names(team_pairs: list[tuple[str, str]]) -> dict[str, str]:
       """Build {team_name: team_id}, disambiguating duplicate display names with ' 2', ' 3', …
       Dedupes on the NAME — unlike the Streamlit Fantrax path, which keyed by name first and so
       silently dropped duplicate-named teams."""
       teams_dict: dict[str, str] = {}
       used_names: list[str] = []
       for name, team_id in team_pairs:
           unique_name, counter = name, 1
           while unique_name in used_names:
               counter += 1
               unique_name = f"{name} {counter}"
           used_names.append(unique_name)
           teams_dict[unique_name] = team_id
       return teams_dict
   ```
2. **Auction unsupported on Fantrax** — `get_auction_results` returns `None`; `available_modes` omits
   Auction (matches Streamlit).
3. **Caching dropped** — live rosters are never cached; division/shape fetched on demand.
4. **Private `_request`** — works in 0.2.9 but is unofficial; flagged. If it breaks on upgrade, swap
   to public `api.teams` / `api.roster_info(team_id)` / `api.standings`.

## Credentials & construction
No method takes a `credentials` argument (Streamlit didn't either — it stashed creds in
`st.session_state` and the integration read them). The **ABC has no `__init__`**: it constrains
behavior, not construction. Each integration declares its own explicit constructor params (Fantrax
none; Yahoo `auth_dir`), and the registry **spreads the creds bag** into them —
`get_integration(platform, credentials)` does `cls(**(credentials or {}))`. So Fantrax is built with
no args, Yahoo as `YahooIntegration(auth_dir=…)`, and a bag key the constructor doesn't declare raises
`TypeError` (fail-noisily). Routes resolve creds from the persistent store by `client_id` (a
frontend-minted id in `localStorage`, validated to `[A-Za-z0-9_-]{1,64}`) via `credential_store.py`.
The integration only ever *uses* creds; *acquiring* them (OAuth, env, persistence) is route/frontend
orchestration.

## Frontend platform connectors (`frontend/platforms/`)
Each platform's connect/auth UX lives in its own connector module — the frontend counterpart to a
backend integration, paired by the platform label (no shared object; just the label + the HTTP
contract). `connector.ts` defines `PlatformConnector` (`platform`, `element`, `getSelection()`);
`fantrax_connector.ts` (league id + division) and `yahoo_connector.ts` (OAuth → paste code → league
dropdown) implement it; `registry.ts` holds the connector map (`makeConnectors` / `connectorPlatforms`).
`league_settings.ts` is a thin host: it builds every connector, shows the active one, and a single
generic Connect path serves all of them; `getPlatformConfig()` delegates to the active connector. The
platform dropdown is derived from `connectorPlatforms()`, so it can only ever offer platforms with a
frontend connector (a backend-only platform simply never appears).

> No runtime parity check. An earlier version exposed `GET /platforms` (backend registry) and had
> `main.ts` assert at startup that the frontend connectors matched it; that was removed (didn't need to
> run on every load). There is currently **no drift detection** between the backend registry and the
> frontend connectors — if you want it back without per-load cost, a CI test calling a parity helper is
> the right home.

## Phase 2 (Yahoo) — IMPLEMENTED, UNTESTED
`integrations/yahoo.py` is a faithful framework-agnostic port of the Streamlit `YahooIntegration`
(+ `yahoo_helper`), now **fully wired** but **never run against a real Yahoo league/OAuth app**. What's
in place:
- Registered in `registry.py`; `available_modes` includes Auction.
- OAuth (manual code-paste): `build_auth_url` / `exchange_auth_code`, the `/platforms/yahoo/auth-url`
  and `/platforms/yahoo/token` routes, and the per-client token store (`credential_store.yahoo_auth_dir`,
  holding yfpy's `token.json`/`private.json`). Yahoo *app* client id/secret come from
  `YAHOO_CLIENT_ID`/`YAHOO_CLIENT_SECRET` env vars.
- League picking: `list_leagues()` (ABC default `[]`, overridden by Yahoo) + `GET /platforms/{platform}/leagues`.
- Auction: `PlatformSelections.costs` carries per-player cost; the draft-state route dispatches to
  `get_auction_results` in Auction Mode and turns costs into `remaining_cash` (= `cash_per_team` − spent).
- Frontend: `client_id.ts`, Yahoo connect UI in `league_settings.ts` (authenticate → paste code →
  league dropdown), live auction `remaining_cash` plumbed through `draft_and_auction_session.ts`.
- Unit tests cover the pure draft/cost parsing (`_assignments_from_draft`) and `build_auth_url`.

Risks to check first at E2E (also in the file banner): the OAuth handshake + yfpy token refresh have
never run; `YAHOO_PLAYER_ID` dtype must match yfpy's `player_id` (else everyone → `'RP'`); `n_picks`
is hard-coded to 13; exact yfpy method names may differ by version.

## Phase 3 (ESPN) — IMPLEMENTED, UNTESTED
`integrations/espn.py` is a framework-agnostic port of the Streamlit `ESPNIntegration`, fully wired but
**never run against a real ESPN league**. ESPN is **Season-only** (the Streamlit ESPN never did draft).
What's in place:
- Registered in `registry.py`; `player_name_column = 'ESPN_NAME'`; `available_modes = ['Season Mode']`.
- Auth = the `espn_s2` + `SWID` cookies (no OAuth): a `POST /platforms/espn/credentials` route stores
  them per-client via `credential_store` (SWID braces stripped, as Streamlit did); `_credentials_for`
  spreads `{s2, swid}` into `ESPNIntegration(s2=…, swid=…)`.
- League picking: `list_leagues()` via the ESPN fan API (keyed by SWID); the league id is composite
  `"<fan-api id>::<year>"` so `fetch_league_shape` / `get_draft_results` can reconstruct
  `League(league_id=<id>.split(':')[1], year=<year>, espn_s2, swid)`.
- `get_draft_results` returns current rosters (Season); `get_auction_results` → None.
- Frontend: `espn_connector.ts` (paste s2 + SWID → Save → league dropdown) + `client.submitEspnCredentials`.
- Unit tests: composite league-id parse + roster mapping (mocked league).

Risks (also in the file banner): never run against real ESPN; `ESPN_NAME` must match espn_api's
`player.name`; the fan-api id `split(':')[1]` shape is unverified; SWID brace handling for espn_api is
untested.

## Live Season-mode roster fill (Fantrax / Yahoo / ESPN)
The season roster grid is now populated from the platform (this previously blocked ESPN, which is
Season-only). Mechanism: `season_session.ts` caches `livePlatformRosters` and exposes
`refreshSeasonRostersFromPlatform()` (polls `GET /sessions/{id}/draft-state?mode=Season Mode` → cache)
plus `getLivePlatformRosters()` / `clearLivePlatformRosters()`. `renderSeasonRosters` uses the cache as
its **prefill source** when a live platform is selected (blank until loaded), instead of
`DEFAULT_SEASON_ROSTERS`. `main.ts` fires the refresh (then re-`applyLayout`) when the user switches the
mode or platform dropdown *into* (Season Mode ∧ live platform), and clears the cache on leaving — kept
in those two handlers (not `showSeasonLayout`) so it doesn't re-poll on every layout pass.

Untested / deferred: the user must **Connect first** (so team names populate the grid columns matching
the poll's `player_assignments` keys); players that mapped to `'RP'` may not be valid grid options;
`injured_players` from the poll is **not** applied to the exclusion list (left alone for now); and there
is **no manual "Refresh from platform" button** yet (deferred — rosters only re-pull on a mode/platform
switch into season+live).

## Persistence
The per-client token store (`credential_store.py`) persists Yahoo OAuth tokens to disk under
`.platform_credentials/` (gitignored; override with `PLATFORM_CREDENTIAL_DIR`), keyed by `client_id`.

## Testing
- **Unit**: mock `FantraxAPI._request` with captured sample payloads; assert `fetch_league_shape` →
  correct `LeagueShape`, and `get_draft_results` → expected column-per-team DataFrame; assert
  `build_platform_name_lookup` mapping a platform spelling → canonical (and miss → `'RP'`); assert `deduplicate_team_names`
  on duplicate names.
- **Manual E2E**: a real Fantrax league ID through connect → session → draft-state → evaluate, in
  both Draft and Season mode.

## Task order (each independently verifiable)
1. `base.py` (ABC + dataclasses) + `helpers.py` (incl. `build_platform_name_lookup`) + unit tests.
2. `integrations/fantrax.py` port using `helpers` (+ unit test with mocked `_request` and mapping view).
3. `Session.platform_config` + models + `rosters_df_to_player_assignments`.
4. Routes: `divisions`, `connect`, `draft-state`.
5. Session-creation wiring to store `PlatformConfig`.
6. Frontend `platform_session.ts` + `league_settings.ts` connect UI.
7. `showLiveLayout()` + Refresh Analysis poll → evaluate.
8. Manual E2E in Draft + Season.

## To confirm at step 1 (by inspecting a live session, not guessing)
- Exact `info['Positions']` index format (suffixed vs. raw — determines whether the mapping
  strip/re-append is identity).
- A real `getTeamRosterInfo` payload shape from an actual league.
