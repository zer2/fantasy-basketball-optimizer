# FBBO FastAPI Contract

## Architecture

```
HTML/JS frontend
      │
      ├── POST /data/upload                — upload user-provided projection CSV (once, before session)
      │
      ├── POST /sessions                   — one-time setup per draft; runs full initialization chain
      │
      ├── PATCH /sessions/{id}             — update any parameters mid-draft; backend re-runs minimally
      │
      ├── POST /sessions/{id}/evaluate     — called each round; frontend sends full draft context
      │
      └── DELETE /sessions/{id}            — cleanup
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

When `PATCH /sessions/{id}` is called, the backend compares the new values against
`current_params` and re-runs only the affected steps:

| Changed field(s)                                          | Steps re-run   |
|-----------------------------------------------------------|----------------|
| `data_source`                                             | 1 → 2 → 3 → 4 → 5 |
| `injured_players`                                         | 2 → 3 → 4 → 5 |
| `upsilon`                                                 | 3 → 4 → 5     |
| `psi`, `chi`, `n_drafters`, `n_picks`, `scoring_format`, `categories`, `slot_counts` | 4 → 5 |
| `omega`, `gamma`, `beth`, `n_iterations`                  | 5 only         |

Step key:
1. Load player_stats_v0 from data source
2. drop_injured_players → player_stats_v1
3. make_upsilon_adjustment → player_stats_v2
4. process_player_data → info dict (G-scores, X-scores, covariance matrices)
5. build_h_agent → HAgent

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
    "scoring_format": "H2H_MC",
    "categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]
  },
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
    "n_iterations": 15
  },
  "data_source": {
    "type": "projections",
    "blend_weights": {
      "espn": 0.4,
      "darko": 0.3,
      "bbm": 0.3
    },
    "custom_data_ids": {
      "HTB": "f3a9c2",
      "BBM": null
    }
  },
  "injured_players": ["Joel Embiid", "Kawhi Leonard"],
  "my_team_id": "team_1"
}
```

**Field notes:**

`league.sport` — one of `"NBA"`, `"MLB"`. Determines which statistical framework the backend
loads (counting vs ratio stat definitions, negative stats, conversion factors, etc.).

`league.scoring_format` — one of `"H2H_MC"` (head-to-head most categories),
`"H2H_EC"` (head-to-head each category), `"Rotisserie"`.

`slot_counts` — maps each position type to the number of roster slots of that type.
Valid position types (base positions and flex slot categories) are defined server-side per
sport — for NBA: `PG`, `SG`, `SF`, `PF`, `C`, `G`, `F`, `Util`. The backend derives slot
IDs (e.g. `"C2"`, `"Util3"`) from these counts, and also stores flex eligibility rules
(e.g. G slots accept PG/SG) as sport-level config. `n_starters` is derived as
`sum(slot_counts.values())`; bench spots = `n_picks - n_starters`.

`data_source.type` — one of `"projections"` or `"historical"`.
- `"projections"`: blends ESPN, DARKO, and optionally custom uploads using `blend_weights`.
  Weights do not need to sum to 1; they are normalized. Omit a source to exclude it entirely.
- `"historical"`: uses stored historical weekly data. `blend_weights` and `custom_data_ids` are ignored.

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
  "expires_at": "2025-03-01T19:00:00Z"
}
```

**Errors:**
- `400` — invalid field values (e.g. unknown scoring_format, mismatched categories)
- `404` — referenced `data_id` not found or expired
- `422` — slot_counts key is not a valid position type for the selected sport

---

## `PATCH /sessions/{id}` — Update parameters

Updates any subset of session parameters mid-draft. The backend diffs the new values
against the stored `current_params` and re-runs only the affected steps (see dependency
table above). All fields are optional — only send what changed.

**Request body** (all fields optional):
```json
{
  "parameters": {
    "omega": 0.95,
    "gamma": 1.2,
    "beth": 0.0,
    "upsilon": 0.9,
    "psi": 0.5,
    "chi": 0.6,
    "aleph": 0.0,
    "n_iterations": 15
  },
  "league": {
    "n_drafters": 10,
    "n_picks": 13,
    "scoring_format": "H2H_MC",
    "categories": ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]
  },
  "data_source": {
    "type": "projections",
    "blend_weights": { "espn": 0.5, "darko": 0.5, "bbm": 0.0 },
    "custom_data_ids": { "HTB": null, "BBM": null }
  },
  "slot_counts": { "PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 2, "G": 2, "F": 2, "Util": 3 },
  "injured_players": ["Joel Embiid", "Kawhi Leonard", "Damian Lillard"]
}
```

Note: `league.sport` cannot be patched — changing the sport requires a new session.
All other fields are patchable. Changing `league` fields (`n_drafters`, `n_picks`,
`scoring_format`, `categories`) re-runs steps 4–5.

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

## `POST /sessions/{id}/evaluate` — Run H-scoring

Runs `HAgent.get_h_scores()` for the current draft state and returns the full display
payload for each candidate, sorted by `h_score` descending.

The frontend is responsible for tracking all picks and sending the full draft context
on every call. The session holds no pick state.

**Request body:**
```json
{
  "player_assignments": {
    "team_1": ["Stephen Curry", "Kevin Durant"],
    "team_2": ["LeBron James"],
    "team_3": []
  },
  "exclusion_list": ["Zion Williamson"]
}
```

`player_assignments` maps every team ID to their picks so far. Required; send empty arrays
for teams with no picks yet.

`exclusion_list` is optional — removes players from the candidate pool without affecting
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
          "G1":    { "name": "Paul",   "is_candidate": false },
          "G2":    null,
          "F1":    null,
          "F2":    null,
          "Util1": null,
          "Util2": null,
          "Util3": null
        }
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

`iteration` is the number of algorithm iterations completed. Equals `n_iterations` unless
the algorithm converged early.

**Errors:**
- `404` — session not found or expired

---

## `DELETE /sessions/{id}` — End session

Frees all server-side state for this session. No body. Returns `204 No Content`.

---

## Frontend integration notes

**Naming convention:** All API fields use `snake_case`. The frontend TypeScript layer
should convert to `camelCase` when mapping to the `Player` interface (e.g.
`is_total` → `isTotal`, `is_candidate` → `isCandidate`).

**Ineligible flex slots:** The API uses JSON `null` for ineligible positions in
`flex_allocations.values`. The frontend sentinel is `-999`. The mapping layer should
convert `null → -999` before passing values to styler functions.

**Candidate ordering:** Candidates in the `/evaluate` response are sorted by `h_score`
descending. `h_rank` is included explicitly for convenience but equals array index + 1.
