# FBBO FastAPI Contract

## Architecture

```
HTML/JS frontend
      │
      ├── GET  /config/{sport}                — static sport config (categories, options, positions)
      │
      ├── GET  /seasons                       — available historical seasons
      │
      ├── POST /data/upload                   — upload user-provided projection CSV (once, before session)
      │
      ├── POST /sessions                      — one-time setup per draft; runs full initialization chain
      │
      ├── PATCH /sessions/{id}               — update any parameters mid-draft; backend re-runs from_step
      │
      ├── GET  /sessions/{id}/g-scores        — fetch G-scores for current session
      │
      ├── POST /sessions/{id}/evaluate        — called each round; frontend sends full draft context
      │
      ├── POST /sessions/{id}/trade/analyze   — evaluate a proposed trade between two teams
      │
      ├── POST /sessions/{id}/trade/suggest   — find beneficial trades between two teams
      │
      ├── POST /cache/clear                   — clear the server-side v0 data cache
      │
      └── DELETE /sessions/{id}              — cleanup
```

### Server-side session state

All large datasets live on the server, keyed by `session_id`. The frontend never sends or receives raw player data. Sport-level config (position types, flex eligibility rules, conversion factors, stat definitions) is stored as static server-side config (analogous to `parameters.yaml`) and loaded by sport at session creation. The backend maintains a full state object per session:

```
player_stats_v0   — raw stats, loaded from files/projections at session creation
player_stats_v1   — derived: v0 minus injured_players
player_stats_v2   — derived: v1 with upsilon adjustment applied
info              — derived: process_player_data(v2, psi, chi, scoring_format, n_drafters, n_picks, ...)
HAgent            — derived: build_h_agent(info, omega, gamma, beth, n_picks, n_drafters, scoring_format, ...)
current_params    — snapshot of all params; used to diff PATCH bodies and determine what to re-run
```

Sessions are held in memory and expire after 4 hours of inactivity.

### Re-computation dependency table

When `PATCH /sessions/{id}` is called, the frontend passes `from_step` explicitly. The backend re-runs from that step through step 5:

| Changed field(s)                                          | from_step to pass |
|-----------------------------------------------------------|-------------------|
| `data_source`                                             | 1                 |
| `injured_players`                                         | 2                 |
| `upsilon`                                                 | 3                 |
| `psi`, `chi`, `n_drafters`, `n_picks`, `scoring_format`, `categories`, `slot_counts` | 4 |
| `omega`, `gamma`, `beth`, `n_iterations`, `streaming_noise` | 5              |

Step key:
1. Load player_stats_v0 from data source
2. drop_injured_players → player_stats_v1
3. make_upsilon_adjustment → player_stats_v2
4. process_player_data → info dict (G-scores, X-scores, covariance matrices)
5. build_h_agent → HAgent

---

## `GET /config/{sport}` — Fetch sport config

Returns static configuration for the given sport: available categories, parameter options,
and position structure. Called once on page load to populate the settings UI.

**Response `200`:**
```json
{
  "default_categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"],
  "all_categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"],
  "options": {
    "n_drafters": { "min": 2, "max": 30, "default": 12 },
    "n_picks":    { "min": 1, "max": 25, "default": 13 },
    "omega":      { "min": 0.0, "max": 1.0, "default": 0.85 }
  },
  "positions": {
    "13": {
      "base": { "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 2 },
      "flex": { "G": 2, "F": 2, "Util": 3 }
    }
  },
  "position_structure": {
    "base_list": ["PG", "SG", "SF", "PF", "C"],
    "flex_list": ["G", "F", "Util"]
  },
  "position_names": {
    "PG": "Point Guard", "SG": "Shooting Guard", "SF": "Small Forward",
    "PF": "Power Forward", "C": "Center", "G": "Guard", "F": "Forward", "Util": "Utility"
  }
}
```

**Errors:**
- `400` — unknown sport

---

## `GET /seasons` — List available historical seasons

Returns a list of season strings that can be used as `data_source.season` when creating a session.

**Response `200`:**
```json
{ "seasons": ["2022-23", "2023-24", "2024-25"] }
```

---

## `POST /data/upload` — Upload projection file

Accepts a user-supplied CSV projection file and stores it server-side. Returns a short-lived
`data_id` to reference it when creating a session. Supported file types are third-party
projection sources (HTB = Hashtag Basketball, BBM = Basketball Monster).

Files expire after 2 hours if not referenced by a session.

**Request:** `multipart/form-data`

| Field       | Type   | Required | Description                                  |
|-------------|--------|----------|----------------------------------------------|
| `file`      | file   | yes      | CSV projection file                          |
| `file_type` | string | yes      | One of `"HTB"`, `"BBM"`                     |

**Response `200`:**
```json
{
  "data_id": "f3a9c2",
  "file_type": "HTB",
  "n_players": 312,
  "expires_at": "2025-03-01T15:30:00Z"
}
```

**Errors:**
- `400` — unrecognized file format or missing required columns
- `413` — file too large (limit: 10 MB)

---

## `POST /sessions` — Initialize

Runs the full initialization chain (steps 1–5). Returns once HAgent is ready.
This is the most expensive call; all subsequent calls are faster.

**Request body:**
```json
{
  "league": {
    "sport": "NBA",
    "n_drafters": 10,
    "n_picks": 13,
    "scoring_format": "Head to Head: Most Categories",
    "categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"],
    "cash_per_team": null
  },
  "is_auction": false,
  "slot_counts": {
    "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 2,
    "G": 2, "F": 2, "Util": 3
  },
  "parameters": {
    "omega": 0.85,
    "gamma": 1.0,
    "beth": 0.0,
    "upsilon": 1.0,
    "psi": 0.5,
    "chi": 0.6,
    "aleph": 0.0,
    "n_iterations": 15,
    "streaming_noise": 0.0
  },
  "data_source": {
    "type": "historical",
    "season": "2024-25"
  },
  "injured_players": ["Joel Embiid", "Kawhi Leonard"],
  "my_team_id": "Drafter 1"
}
```

**Field notes:**

`league.sport` — one of `"NBA"`, `"MLB"`. Determines which statistical framework the backend
loads (counting vs ratio stat definitions, negative stats, conversion factors, etc.).

`league.scoring_format` — one of `"Head to Head: Most Categories"`,
`"Head to Head: Each Category"`, `"Rotisserie"`.

`league.cash_per_team` — Auction Mode only; omit or set to `null` for Draft Mode. Only
consulted when the session's `is_auction` is true, so a value left over from an earlier
auction is inert in other modes.

`is_auction` — the session's league type (default `false`). Auction sessions require
`remaining_cash` on every evaluate and non-auction sessions forbid it. Patchable — the
frontend patches it on every mode switch (`true` entering Auction Mode, `false` entering
Draft or Season Mode); a patch that omits it leaves it unchanged.

`slot_counts` — maps each position type to the number of roster slots of that type.
Valid position types (base positions and flex slot categories) are defined server-side per
sport — for NBA: `PG`, `SG`, `SF`, `PF`, `C`, `G`, `F`, `Util`. The backend derives slot
IDs (e.g. `"C2"`, `"Util3"`) from these counts, and also stores flex eligibility rules
(e.g. G slots accept PG/SG) as sport-level config. `n_starters` is derived as
`sum(slot_counts.values())`; bench spots = `n_picks - n_starters`.

`data_source.type` — one of `"projections"`, `"historical"`, or `"csv"`.
- `"projections"`: blends ESPN, DARKO, and optionally custom uploads using `blend_weights`.
  Weights do not need to sum to 1; they are normalized. Omit a source to exclude it entirely.
- `"historical"`: uses stored historical season data. Requires `season` field. `blend_weights`
  and `custom_data_ids` are ignored.
- `"csv"`: uses a single custom-uploaded CSV. Requires a valid `data_id` in `custom_data_ids`.

`data_source.custom_data_ids` — references files uploaded via `POST /data/upload`.
If provided, the corresponding source in `blend_weights` is replaced by the uploaded file.
Set to `null` to use the default server-side source.

`injured_players` — list of player names to drop before coefficient calculation. Optional; defaults to `[]`.

**Response `201`:**
```json
{
  "session_id": "abc123",
  "n_players_loaded": 487,
  "categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"],
  "g_scores": [
    { "name": "Nikola Jokic", "total": 5.90, "values": [1.80, -2.10, -0.90, 1.70, 2.40, 2.20, 0.50, 1.60, -1.30] }
  ],
  "expires_at": "2025-03-01T19:00:00Z"
}
```

`g_scores` — full list of player G-scores at session creation, sorted by the backend.
`values` are per-category G-scores in the same order as `categories`.

**Errors:**
- `400` — invalid field values (e.g. unknown scoring_format, mismatched categories)
- `404` — referenced `data_id` not found or expired
- `422` — slot_counts key is not a valid position type for the selected sport

---

## `PATCH /sessions/{id}` — Update parameters

Updates any subset of session parameters mid-draft. The frontend passes `from_step` to tell
the backend which initialization step to re-run from (through step 5). See the dependency
table above. All fields except `from_step` are optional — only send what changed.

**Request body:**
```json
{
  "from_step": 5,
  "parameters": {
    "omega": 0.95,
    "gamma": 1.2,
    "beth": 0.0,
    "upsilon": 0.9,
    "psi": 0.5,
    "chi": 0.6,
    "aleph": 0.0,
    "n_iterations": 15,
    "streaming_noise": 0.0
  },
  "league": {
    "n_drafters": 10,
    "n_picks": 13,
    "scoring_format": "Head to Head: Most Categories",
    "categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"],
    "cash_per_team": null
  },
  "data_source": {
    "type": "projections",
    "blend_weights": { "espn": 0.5, "darko": 0.5 },
    "custom_data_ids": { "HTB": null, "BBM": null }
  },
  "slot_counts": { "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 2, "G": 2, "F": 2, "Util": 3 },
  "injured_players": ["Joel Embiid", "Kawhi Leonard", "Damian Lillard"]
}
```

Note: `league.sport` cannot be patched — changing the sport requires a new session.

**Response `200`:**
```json
{
  "ok": true,
  "steps_rerun": [5]
}
```

`steps_rerun` lists which initialization steps were re-executed (1–5), for debugging/transparency.

**Errors:**
- `404` — session not found or expired
- `404` — referenced `data_id` not found or expired (if data_source changed)

---

## `GET /sessions/{id}/g-scores` — Fetch G-scores

Returns the current G-scores for all players in the session. Useful after a PATCH to refresh
the display without running a full evaluate.

**Response `200`:**
```json
{
  "g_scores": [
    { "name": "Nikola Jokic", "total": 5.90, "values": [1.80, -2.10, -0.90, 1.70, 2.40, 2.20, 0.50, 1.60, -1.30] }
  ]
}
```

**Errors:**
- `404` — session not found or expired

---

## `POST /sessions/{id}/evaluate` — Run H-scoring

Runs `HAgent.get_h_scores()` for the current draft state and returns the full display
payload for each candidate, sorted by `h_score` descending.

The frontend is responsible for tracking all picks and sending the full draft context
on every call. The session holds no pick state.

**Request body:**
```json
{
  "player_assignments": {
    "Drafter 1": ["Stephen Curry (PG,SG)", "Kevin Durant (SF,PF)"],
    "Drafter 2": ["LeBron James (SF,PF)"],
    "Drafter 3": []
  },
  "remaining_cash": {
    "Drafter 1": 163,
    "Drafter 2": 178,
    "Drafter 3": 200
  },
  "my_team_id": "Drafter 1",
  "exclusion_list": ["Zion Williamson (PF)"]
}
```

`player_assignments` maps every team name to their picks so far, with player names in
`"Name (positions)"` format (e.g. `"Nikola Jokic (C)"`). Required; send empty arrays for
teams with no picks yet.

`remaining_cash` — required in Auction Mode only; maps each team name to the dollar
amount not yet spent. Omit entirely for Draft Mode.

`my_team_id` — the team name of the user running the tool.

`exclusion_list` — optional; removes players from the candidate pool without affecting
coefficient calculations (use `PATCH /sessions/{id}` with `injured_players` for that).

**Response `200`:**
```json
{
  "iteration": 15,
  "candidates": [
    {
      "name": "Nikola Jokic",
      "position": "C",
      "h_score": 53.7,
      "h_rank": 1,
      "win_rates": [66.2, 14.2, 33.9, 66.3, 73.4, 72.3, 59.7, 67.7, 29.7],
      "category_weights": [95, 83, 98, 114, 95, 102, 103, 111, 100],
      "g_score_rows": [
        { "label": "Current diff", "values": [ 0.42, -0.18, -0.31,  0.28,  0.61,  0.54,  0.12,  0.19, -0.38], "total":  1.29, "is_total": false },
        { "label": "Jokic",        "values": [ 1.80, -2.10, -0.90,  1.70,  2.40,  2.20,  0.50,  1.60, -1.30], "total":  5.90, "is_total": false },
        { "label": "Future diff",  "values": [ 0.31, -0.09, -0.22,  0.19,  0.47,  0.38,  0.09,  0.14, -0.27], "total":  1.00, "is_total": false },
        { "label": "Total diff",   "values": [ 2.53, -2.37, -1.43,  2.17,  3.48,  3.12,  0.71,  1.93, -1.95], "total":  8.19, "is_total": true  }
      ],
      "flex_allocations": {
        "base_positions": ["PG", "SG", "SF", "PF", "C"],
        "rows": [
          { "label": "G-1",    "values": [0.65,  0.35,  null,  null,  null], "is_total": false },
          { "label": "F-2",    "values": [null,  null,  1.10,  0.90,  null], "is_total": false },
          { "label": "Util-3", "values": [0.50,  0.40,  0.70,  0.60,  0.80], "is_total": false },
          { "label": "Total",  "values": [1.15,  0.75,  1.80,  1.50,  0.80], "is_total": true  }
        ]
      },
      "roster": {
        "slots": ["PG1", "SG1", "SF1", "PF1", "C1", "C2", "G1", "G2", "F1", "F2", "Util1", "Util2", "Util3"],
        "assignments": {
          "PG1":   { "name": "Curry",  "is_candidate": false },
          "SG1":   null,
          "SF1":   { "name": "Durant", "is_candidate": false },
          "PF1":   null,
          "C1":    { "name": "Jokic",  "is_candidate": true  },
          "C2":    null,
          "G1":    null,
          "G2":    null,
          "F1":    null,
          "F2":    null,
          "Util1": null,
          "Util2": null,
          "Util3": null
        }
      },
      "auction_values": {
        "your_dollar":   53.7,
        "gnrc_dollar":   50.2,
        "orig_dollar":   51.8,
        "gnrc_dollar_g": 48.3,
        "orig_dollar_g": 49.9
      }
    }
  ]
}
```

**Field scales and units:**

| Field                       | Scale            | Notes                                              |
|-----------------------------|------------------|----------------------------------------------------|
| `h_score`                   | 0–100            | Win rate percentage                                |
| `h_rank`                    | 1–N              | Rank among candidates by h_score; 1 = best        |
| `win_rates`                 | 0–100            | Per-category win rate; 50 = average                |
| `category_weights`          | ~100 baseline    | Normalized by H.v; 100 = neutral                  |
| `g_score_rows.values`       | raw G-score diff | Multiplied by H.original_v; length = len(categories) |
| `g_score_rows.total`        | raw G-score diff | Sum across categories                              |
| `flex_allocations.values`   | expected count   | `null` = position ineligible for this flex slot    |
| `roster.assignments`        | —                | `null` slot = empty; `is_candidate: true` = this player |
| `auction_values.your_dollar` | dollars         | SAVOR on H-scores, team-specific cash/picks        |
| `auction_values.gnrc_dollar` | dollars         | SAVOR on H-scores, generic baseline (current cash) |
| `auction_values.orig_dollar` | dollars         | SAVOR on H-scores, full original cash/picks        |
| `auction_values.gnrc_dollar_g` | dollars       | SAVOR on G-scores, generic baseline                |
| `auction_values.orig_dollar_g` | dollars       | SAVOR on G-scores, full original cash/picks        |

`iteration` is the number of algorithm iterations completed. Equals `n_iterations` unless
the algorithm converged early.

`auction_values` is `null` in Draft Mode. `flex_allocations` and `roster` are `null` when
position data is absent.

**Errors:**
- `404` — session not found or expired

---

## `POST /sessions/{id}/trade/analyze` — Analyze a trade

Evaluates the H-score impact of a proposed trade between two teams. Returns pre- and
post-trade H-scores for both teams.

**Request body:**
```json
{
  "player_assignments": {
    "Drafter 1": ["Shai Gilgeous-Alexander (PG,SG)", "Brook Lopez (C)"],
    "Drafter 2": ["Nikola Jokic (C)", "Pascal Siakam (C,SF,PF)"]
  },
  "my_team": "Drafter 1",
  "their_team": "Drafter 2",
  "my_trade": ["Shai Gilgeous-Alexander (PG,SG)"],
  "their_trade": ["Nikola Jokic (C)"],
  "ignore_position_check": false
}
```

`ignore_position_check` — if `true`, skips validation that both teams can legally field
the post-trade roster (used when position data is unavailable).

**Response `200`:**
```json
{
  "your_team": {
    "pre":  { "h_score": 0.5268, "rates": [0.52, 0.48, 0.51, 0.53, 0.55, 0.50, 0.49, 0.54, 0.47] },
    "post": { "h_score": 0.5203, "rates": [0.51, 0.47, 0.50, 0.52, 0.54, 0.49, 0.48, 0.53, 0.46] }
  },
  "their_team": {
    "pre":  { "h_score": 0.5207, "rates": [0.51, 0.49, 0.52, 0.50, 0.53, 0.51, 0.50, 0.52, 0.48] },
    "post": { "h_score": 0.5243, "rates": [0.52, 0.50, 0.53, 0.51, 0.54, 0.52, 0.51, 0.53, 0.49] }
  },
  "error": null
}
```

`h_score` is on the raw 0–1 scale (multiply by 100 for the display percentage).
`rates` are per-category win rates, also on the 0–1 scale.

If the trade is invalid (e.g. position constraints violated), `your_team` and `their_team`
will be `null` and `error` will contain a description.

**Errors:**
- `404` — session not found or expired

---

## `POST /sessions/{id}/trade/suggest` — Suggest trades

Finds all trades between two teams that satisfy the configured differential thresholds.
Returns a list of suggested trades sorted by benefit.

**Request body:**
```json
{
  "player_assignments": {
    "Drafter 1": ["Shai Gilgeous-Alexander (PG,SG)", "Brook Lopez (C)"],
    "Drafter 2": ["Nikola Jokic (C)", "Pascal Siakam (C,SF,PF)"]
  },
  "my_team": "Drafter 1",
  "their_team": "Drafter 2",
  "combo_params": [
    { "n_traded": 1, "n_received": 1, "threshold": 0.0 },
    { "n_traded": 2, "n_received": 2, "threshold": 0.0 }
  ],
  "your_differential_threshold": 0.0,
  "their_differential_threshold": -0.20,
  "ignore_position_check": false
}
```

`combo_params` — list of trade size configurations to search. Each entry specifies the
number of players sent and received, and a minimum net H-score differential for the trade
to be included.

`your_differential_threshold` / `their_differential_threshold` — minimum acceptable
H-score change (0–1 scale) for each side of the trade.

**Response `200`:**
```json
{
  "suggestions": [
    {
      "send":        ["Brook Lopez (C)"],
      "receive":     ["Pascal Siakam (C,SF,PF)"],
      "your_score":  0.531,
      "their_score": 0.518
    }
  ]
}
```

**Errors:**
- `404` — session not found or expired

---

## `POST /cache/clear` — Clear data cache

Clears the server-side v0 data cache (historical and projection data). Useful during
development or after updating data files. Returns `204 No Content`.

---

## `DELETE /sessions/{id}` — End session

Frees all server-side state for this session. No body. Returns `204 No Content`.

**Errors:**
- `404` — session not found or expired

---

## Frontend integration notes

**Naming convention:** All API fields use `snake_case`. The frontend TypeScript layer
should convert to `camelCase` when mapping to the `Player` interface (e.g.
`is_total` → `isTotal`, `is_candidate` → `isCandidate`).

**Ineligible flex slots:** The API uses JSON `null` for ineligible positions in
`flex_allocations.values`. The frontend handles `null` directly at the call site
(see `expand_view.ts`) and never passes ineligible cells to stat_styler functions.

**Candidate ordering:** Candidates in the `/evaluate` response are sorted by `h_score`
descending. `h_rank` is included explicitly for convenience but equals array index + 1.

**Trade H-score scale:** `trade/analyze` returns H-scores on the raw 0–1 scale.
Multiply by 100 to get the display percentage shown in the UI.

**Auction workflow:** In Auction Mode, the frontend must call `/evaluate` once with an
empty board and full cash before assigning any players. This populates
`session.generic_h_scores`, which is the neutral baseline used to compute dollar values.
Subsequent evaluate calls with actual assignments will then produce correct `auction_values`.
