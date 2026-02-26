# FBBO FastAPI Contract

## Architecture

```
HTML/JS frontend
      │
      ├── POST /sessions                  — one-time setup per draft
      │
      ├── POST /sessions/{id}/evaluate    — called each round; returns ranked candidates
      │
      ├── POST /sessions/{id}/pick        — records a pick by any team
      │
      └── DELETE /sessions/{id}           — cleanup
```

---

## `POST /sessions` — Initialize

The backend loads player data, computes G-scores and covariance matrices, and initializes `HAgent`. Expensive but only happens once per draft.

**Request body:**
```json
{
  "league": {
    "n_drafters": 10,
    "n_picks": 13,
    "scoring_format": "H2H_MC",
    "categories": ["FG%", "FT%", "3s", "PTS", "REB", "AST", "STL", "BLK", "TO"],
    "platform": "Yahoo"
  },
  "position_structure": {
    "base_list": ["PG", "SG", "SF", "PF", "C"],
    "flex_list": ["G", "F", "Util"],
    "slots": ["PG1", "SG1", "SF1", "PF1", "C1", "C2", "G1", "G2", "F1", "F2", "Util1", "Util2", "Util3"]
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
  "my_team_id": "team_1"
}
```

`scoring_format` is one of `"H2H_MC"`, `"H2H_EC"`, `"Rotisserie"`.

**Response:**
```json
{
  "session_id": "abc123",
  "n_players_loaded": 487,
  "categories": ["FG%", "FT%", "3s", "PTS", "REB", "AST", "STL", "BLK", "TO"]
}
```

---

## `POST /sessions/{id}/evaluate` — Run H-scoring

Runs `HAgent.get_h_scores()` for the current draft state and returns the full display payload for each candidate.

**Request body:**
```json
{
  "player_assignments": {
    "team_1": ["Stephen Curry", "Kevin Durant"],
    "team_2": ["LeBron James"],
    "team_3": []
  },
  "exclusion_list": []
}
```

`player_assignments` maps every team ID to their picks so far. Empty arrays for teams with no picks.
`exclusion_list` is optional — names of players to exclude from candidates (e.g. injured, on waivers).
The full `player_assignments` can also be sent to `/evaluate` to override session state (useful for
what-if analysis, e.g. season mode trade evaluation).

**Response:**
```json
{
  "iteration": 15,
  "candidates": [
    {
      "name": "Nikola Jokic",
      "position": "C",
      "h_score": 53.7,
      "win_rates": [66.2, 14.2, 33.9, 66.3, 73.4, 72.3, 59.7, 67.7, 29.7],
      "category_weights": [95, 83, 98, 114, 95, 102, 103, 111, 100],
      "g_score_rows": [
        { "label": "Current diff",  "values": [0.42, -0.18, -0.31,  0.28,  0.61,  0.54,  0.12,  0.19, -0.38], "total":  1.29, "is_total": false },
        { "label": "Jokic",         "values": [1.80, -2.10, -0.90,  1.70,  2.40,  2.20,  0.50,  1.60, -1.30], "total":  5.90, "is_total": false },
        { "label": "Future diff",   "values": [0.31, -0.09, -0.22,  0.19,  0.47,  0.38,  0.09,  0.14, -0.27], "total":  1.00, "is_total": false },
        { "label": "Total diff",    "values": [2.53, -2.37, -1.43,  2.17,  3.48,  3.12,  0.71,  1.93, -1.95], "total":  8.19, "is_total": true  }
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

**Scales and units:**

| Field                  | Scale          | Notes                                        |
|------------------------|----------------|----------------------------------------------|
| `h_score`              | 0–100          | Win rate percentage                          |
| `win_rates`            | 0–100          | Per-category win rate; 50 = average          |
| `category_weights`     | ~100 baseline  | Normalized by H.v; 100 = neutral             |
| `g_score_rows.values`  | raw G-score    | Already multiplied by H.original_v           |
| `flex_allocations.values` | expected count | `null` = ineligible slot                  |

---

## `POST /sessions/{id}/pick` — Record a pick

Updates the session's internal `player_assignments`. Avoids resending the full assignment dict
on every evaluate call — the frontend just says who got picked.

**Request:**
```json
{
  "team_id": "team_2",
  "player_name": "Nikola Jokic"
}
```

**Response:**
```json
{
  "ok": true,
  "players_remaining": 431
}
```

---

## `DELETE /sessions/{id}` — End session

No body. Returns `204 No Content`.

---

## Frontend mapping note

The frontend currently uses `-999` as a sentinel for ineligible flex cells. The API uses JSON `null`
instead. The frontend mapping layer should convert `null → -999` before passing values to the styler
functions.
