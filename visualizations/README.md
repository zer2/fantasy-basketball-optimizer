# Visualizations

Manim scenes that explain the algorithm — how H-scores are built and why they behave the
way they do — plus the videos they render to. Nothing here is imported by the app or
served by the container; it exists to be watched.

## Rendering

Scenes live in `scenes/`, one file per topic. Render a single scene by naming its file and
its class:

```
manim -ql visualizations/scenes/<file>.py <SceneClass>    # 480p15 — fast, for iterating
manim -qh visualizations/scenes/<file>.py <SceneClass>    # 1080p60 — final
manim -ql -p visualizations/scenes/<file>.py <SceneClass> # -p opens the result when done
```

Output lands under `visualizations/media/videos/<file>/<quality>/<SceneClass>.mp4`. The
repository-root `manim.cfg` is what puts it there — run from the root without it and Manim
writes to `./media`, outside this directory and outside the ignore rules that keep its
intermediates untracked.

First run `python visualizations/prepare_season_data.py`, which pulls the season once and
writes `data/` and `assets/headshots/`. The scenes read only those, so a render is offline,
fast, and reproducible — the draws are seeded in the prep script, so re-rendering a scene
gives back the same video rather than a new sample.

`prepare_assignment_data.py` is a second prep script, for the roster-slot scene. It needs the
headshots the first one writes, so run that one first.

`prepare_draft_equivalence_data.py`, `prepare_payoff_formats_data.py`,
`prepare_self_play_data.py` and `prepare_seed_landscape_data.py` stand alone. The first reads a
season; the other three are pure computation — a search, a twelve-seat field, and an objective
surface — and need neither a season pull nor headshots. Each prints what it found, and each one
found something the planning note had not predicted; their docstrings record it.

## What is here

Both differential scenes work in the same unit — points in a week — so their axes and spreads
can be read against each other. They are meant to be watched in order: `team_differential`
assumes every player contributes his average, and `g_score` is what happens when that
assumption is taken away.

| Scene | Shows |
|---|---|
| `team_differential.py` → `TeamDifferentialFull` | Two random teams; a player is worth their **weekly average**. σ = 86. All a Z-score sees. |
| `g_score.py` → `GScoreFull` | The answer, in one scene. **One fixed matchup replayed in different weeks** (σ = 104, and off-centre, because one of those teams really is better), then **both varying** (σ = 137), then why 86 and 104 make 137: the two sources are independent, so their **variances** add. Narrated. |
| `punting.py` → `PuntingSearch` | Nine categories against an opponent at parity. The bell never moves; the threshold does. Abandoning three outright wins 4.625 categories against the 4.500 that perfect balance gets. |
| `assignment.py` → `RosterSlotAssignment` | The thirteen-slot assignment problem. A drafted player scores 0 wherever he is eligible, so the optimiser is not placing him to be useful — it is moving him out of the way of the picks still to come. |
| `draft_equivalence.py` → `DraftEquivalence` | Why random drafting is not a cop-out: assume value is linear in pick order plus a value-neutral tilt R, and a snake draft's pick numbers pair to the same totals for every seat, so the baselines cancel and two teams differ only by their draws of R. Ends on where the assumption strains, and on the reminder that no static ranking can be right. Narrated. |
| `category_gradient.py` → `CategoryGradient` | The marginal value of a category, which the punting scene already draws without labelling: the height of the bell where the bar crosses. Sweeps one bar to show the peak at parity, then reads all nine off at the punt optimum — six contested marginals level, three abandoned ones lower, which is the first-order condition made visible. |
| `payoff_formats.py` → `ThreeFormats` | Each Category, Most Categories and Rotisserie under one search. **Measured, against the plan's prediction: all three punt exactly three categories.** What differs is what it is worth — +9.5%, +2.8%, +1.1% over perfect balance — and the single-category payoff curves say why. |
| `self_play.py` → `SelfPlayLoop` | Twelve seats best-responding to each other. Searching freely against the latest field, every seat herds onto the same three punts and the drift never falls; responding from where it already is, the field settles in a few passes and the punts spread across categories. |
| `seed_landscape.py` → `SeedLandscape` | The objective over a plane through three of its optima. The balanced build is a stationary point — the cold start scores 4.5000 and takes zero steps — while every challenger seed climbs to a corner near 4.6247, and the warm start gets there in a fraction of the steps. |

`differential_base.py` holds everything the differential scenes share: layout, dealing,
the histogram, and the five acts. A scene file sets a handful of class attributes (which data
file, the axis limits, the captions) and supplies `contribution_values`. Keeping one definition
is what makes the cuts comparable by construction — a change to the pacing or the binning
cannot land in one and miss the others, which matters because the whole point of the set is
that their spreads are read against each other. `load_dataset` is what lets `g_score` run two
of those simulations back to back inside a single scene. Each also exposes single-act classes
(`ActOneSingleDraw`, and so on) so one act can be re-rendered while its timing is being tuned
instead of the whole forty seconds.

## Text on screen

`punting.py` and `assignment.py` carry no explanatory prose — they are narrated instead, and
`narration_notes.md` holds what each beat is saying. Both still hold the frame where a line
goes, so the pauses a voice needs are already cut in rather than needing to be added later.

Labels are not narration and stay: category names, slot headers, player names, axis captions,
and the score readouts. A viewer cannot read the picture without them.

## Manim version

Pinned by practice to **Manim Community v0.18.1**, the version installed on this machine.
Manim makes breaking API changes between minor releases, so a scene written against 0.18
will not necessarily run on 0.21 — upgrade deliberately and re-render everything, never
because the CLI's update banner asked.

Requires `ffmpeg` (the encoder) and a LaTeX installation with `dvisvgm` (everything using
`Tex` or `MathTex`). Both are present here: ffmpeg 7.1 and TeX Live 2024 via TinyTeX.

## What is committed

Scene sources, and final rendered videos worth keeping. Manim's intermediates are
gitignored: `partial_movie_files/` holds one clip per animation and dwarfs the finished
video, while `Tex/` and `texts/` are regenerable build products of the LaTeX step.

The whole directory is excluded from the Docker build context and from Cloud Build
uploads (see `.dockerignore` and `.gcloudignore`) — video in a build context costs deploy
time and buys the running service nothing.
