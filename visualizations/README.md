# Visualizations

Eight animations for the docs, one folder each, in the order they appear on the page. Everything
on screen is measured by a prep script against the real algorithm — no scene invents its numbers.

| # | folder | scene file, class | where it goes |
|---|---|---|---|
| 1 | `1_z_score/` | `z_score.py`, `TeamDifferentialFull` | top of the G-score page |
| 2 | `2_z_versus_g_score/` | `g_score.py`, `GScoreFull` | directly after #1 |
| 3 | `3_roster_slots/` | `roster_slots.py`, `RosterSlotAssignment` | H-score page, roster assignment |
| 4 | `4_category_weights/` | `plane_story.py`, `PlaneStory` | H-score page, category weights |
| 5 | `5_punting/` | `punting.py`, `PuntingSearch` | H-score page, punting |
| 6 | `6_self_play/` | `self_play.py`, `SelfPlayLoop` | H-score page, "No model of other managers" |
| 7 | `7_seed_menu/` | `seed_menu.py`, `SeedMenu` | H-score page, multi-starting |
| 8 | `8_gradient_descent/` | `gradient_descent.py`, `GradientDescent` | H-score page, replacing the YouTube embed |

## Narration

**Every animation's spoken track lives in one file: `<folder>/narration.py`.** That is the only
file to edit to change what is said. Each key is one beat, and the beat's animation runs inside
its line — a longer line holds its own beat rather than pushing everything after it out of sync.

Most lines are still placeholders (they start with the word "Placeholder"), written to say what
that beat is for. Rewrite them and re-render; nothing else needs touching.

The voice is gTTS, so the text is read literally — write "sigma", not "σ".

## Rendering

From the repository root, so that `manim.cfg` is picked up (it puts output in
`visualizations/media/`):

```
manim -ql visualizations/6_self_play/self_play.py SelfPlayLoop      # draft, 480p15
manim -qh visualizations/6_self_play/self_play.py SelfPlayLoop      # 1080p60
```

A scene reads a JSON file from `data/`; if it is missing, the scene says which prep script to
run. Renders do **not** notice that their data changed — after re-running a prep, re-render.

## Data and shared code

- `data/` — everything the prep scripts write. Several files are shared (`pool_2025_26.json`,
  `matchup_2025_26.json`, `weekly_2025_26.json` feed #1 and #2).
- `assets/` — headshots (#1, #2, #3) and the baked density images (#4).
- `shared/` — code more than one animation needs: `prepare_season_data.py` (the season pull,
  also imported by the other preps for the app's default session), `differential_base.py`
  (#1 and #2), `weight_surface_base.py` and `prepare_weight_surface_data.py` (#7 and #8: one
  bootstrap run measures both their surfaces, so they share a prep).
- `deprecated/` — see its own README. Nothing there is rendered or imported by the eight.

## Prep scripts

Run from the repository root. Each one talks to Snowflake and builds a real session, so they
take a few minutes.

```
python visualizations/shared/prepare_season_data.py             # 1, 2
python visualizations/3_roster_slots/prepare_roster_slots_data.py
python visualizations/4_category_weights/prepare_plane_story_data.py
python visualizations/6_self_play/prepare_self_play_data.py
python visualizations/shared/prepare_weight_surface_data.py     # 7 and 8 together
```

`5_punting` has no prep: its curves are computed in the scene from the search it describes.
