# Visualizations

Eleven animations for the docs, one folder each under `scenes/`. The first eight are numbered in the order they
appear on the G-score and H-score pages; the three later ones are named for their subject.

| folder | scene file, class | where it goes |
|---|---|---|
| `scenes/1_z_score/` | `z_score.py`, `TeamDifferentialFull` | top of the G-score page |
| `scenes/2_z_versus_g_score/` | `g_score.py`, `GScoreFull` | directly after #1 |
| `scenes/3_roster_slots/` | `roster_slots.py`, `RosterSlotAssignment` | H-score page, roster assignment |
| `scenes/4_category_weights/` | `plane_story.py`, `PlaneStory` | H-score page, category weights |
| `scenes/5_punting/` | `punting.py`, `PuntingSearch` | H-score page, punting |
| `scenes/6_self_play/` | `self_play.py`, `SelfPlayLoop` | H-score page, "No model of other managers" |
| `scenes/7_seed_menu/` | `seed_menu.py`, `SeedMenu` | H-score page, multi-starting |
| `scenes/8_gradient_descent/` | `gradient_descent.py`, `GradientDescent` | H-score page, replacing the YouTube embed |
| `scenes/most_categories/` | `most_categories.py`, `MostCategories` | H-score page, Most Categories |
| `scenes/rotisserie/` | `rotisserie.py`, `Rotisserie` | H-score page, Rotisserie |
| `scenes/savor/` | `savor.py`, `Savor` | auctions page, the SAVOR adjustment |

The eight numbered scenes are measured: everything on screen comes from a prep script run against the real
algorithm. The three later ones started as wireframes -- first drafts asking *does this argument work as a
picture?* -- and their numbers are computed in the scene or chosen to illustrate, which each says in its
docstring:

- `most_categories` -- the majority payoff is a sum over 512 branches, that tree collapses into a ten-bar tally,
  and the tally gives the tipping-point probability that decides what a category is worth.
- `rotisserie` -- you need an aberrant season to win a league, so spread is worth having on its own, and spread
  is largest when categories sit near fifty-fifty.
- `savor` -- a projection has a floor under it, at the replacement level S_R, because a player who ends up below
  it gets dropped for a streamer; the thing you are bidding against is a flyer with the same floor, and
  subtracting the two is the SAVOR formula.

## Sign-off

Signed off means Zach has watched that render and called it done. A scene that is signed off does
not get touched again without asking — including by a change that only means to improve it.

| scene | signed off | length | notes |
|---|---|---|---|
| 1 z-score | **yes**\* | 137s | \*blessed conditional on the last three fixes — percentage-statistic label, one more second on the final frame, rewritten simulation line — all now in and verified |
| 2 g-score | **yes** | 125s | worked examples cut from the two simulation lines |
| 3 roster slots | **yes** | 102s | |
| 4 plane story | **yes** | 47s | |
| 5 punting | **yes** | 83s | |
| 6 self-play | **yes** | 74s | |
| 7 seed menu | **yes** | 27s | surface drawn with `HOLD_DRAWN_ASSIGNMENT`, which takes the notches out; the opening holds the axes for 0.9s and draws the surface under the line |
| 8 gradient descent | **yes** | 44s | |
| most categories | no | | awaiting review |
| rotisserie | **yes** | | |
| savor | **yes** | 126s | re-signed off 2026-10-21 after the line moved from $0 to S_R ('streaming slot ($2)', the app's default) and the narration was rewritten to match |

The eight numbered scenes exist at both 480p15 (drafts) and 1080p60. Every render says what its `narration.py`
says — `shared/check_narration.py` reads each scene's subtitle track back and reports any line the render has
fallen behind on.

## Publishing

Renders are untracked, so a finished video reaches the docs by being copied into `docs/videos/` — that is the
copy MkDocs serves:

```
cp visualizations/render_cache/videos/z_score/1080p60/TeamDifferentialFull.mp4 docs/videos/z-scores.mp4
```

The pages embed them as raw HTML `<video>` tags. MkDocs rewrites relative paths in Markdown
image syntax but NOT inside raw HTML, so those tags say `../videos/<name>.mp4` explicitly, which
is what resolves from a page served at `/gscores/` or `/hscores/`.

Re-rendering a scene does not update the docs. Copy it across again.

## Render times

The full 1080p60 pass of the eight takes about **two hours** on this machine, serially — roughly 11x the
finished run time. Per scene, slowest first: gradient descent 27m, g-score 24m, seed menu 19m,
z-score 17m, self-play 12m, punting 11m, roster slots 3m, plane story 1m. The two surface
scenes cost the most per second of video: a 56x56 shaded mesh is expensive per pixel, and that
is the part that scales with resolution. Render them one at a time — manim-voiceover's cache is
an unlocked read-modify-write file and concurrent renders have corrupted it.

## Narration

**Every animation's spoken track lives in one file: `<scene folder>/narration.py`.** That is the only
file to edit to change what is said. Each key is one beat, and the beat's animation runs inside
its line — a longer line holds its own beat rather than pushing everything after it out of sync.
Beats are cued to phrases in the line (`shared/narration_timing.wait_until_phrase`), so rewording a line can
move or break a cue: check the phrases the scene waits on.

The voice reads the text literally — write "sigma", not "σ". A draft can use `shared/draft_voice.py` (gTTS)
instead of the shipped voice, which is free and keeps drafts out of the paid cache while the script is still
moving; gTTS reads faster and flatter, so its timings are indicative.

## Rendering

From the repository root, so that `manim.cfg` is picked up (it puts output in
`visualizations/render_cache/`):

```
manim -ql visualizations/scenes/6_self_play/self_play.py SelfPlayLoop      # draft, 480p15
manim -qh visualizations/scenes/6_self_play/self_play.py SelfPlayLoop      # 1080p60
manim -ql visualizations/scenes/savor/savor.py Savor
```

A scene reads a JSON file from `prepared_data/`; if it is missing, the scene says which prep script to
run. Renders do **not** notice that their data changed — after re-running a prep, re-render.

## Data and shared code

- `prepared_data/` — everything the prep scripts write. Several files are shared (`pool_2025_26.json`,
  `matchup_2025_26.json`, `weekly_2025_26.json` feed #1 and #2).
- `prepared_assets/` — headshots (#1, #2, #3, savor) and the baked density images (#4).
- `shared/` — code more than one animation needs: `prepare_season_data.py` (the season pull,
  also imported by the other preps for the app's default session), `differential_base.py`
  (#1 and #2), `weight_surface_base.py` and `prepare_weight_surface_data.py` (#7 and #8: one
  bootstrap run measures both their surfaces, so they share a prep), and the narration tools.
- `render_cache/` — Manim's output, including the purchased narration audio. See its own README.

## Prep scripts

Run from the repository root. Each one talks to Snowflake and builds a real session, so they
take a few minutes.

```
python visualizations/shared/prepare_season_data.py                       # 1, 2
python visualizations/scenes/2_z_versus_g_score/prepare_wembanyama_blocks.py
python visualizations/scenes/3_roster_slots/prepare_roster_slots_data.py
python visualizations/scenes/4_category_weights/prepare_plane_story_data.py
python visualizations/scenes/6_self_play/prepare_self_play_data.py
python visualizations/shared/prepare_weight_surface_data.py               # 7 and 8 together
python visualizations/scenes/most_categories/prepare_outcome_table.py
```

`5_punting` has no prep: its curves are computed in the scene from the search it describes. Nor do
`rotisserie` and `savor`, whose curves are closed form.
