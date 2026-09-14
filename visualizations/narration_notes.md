# Narration notes

The scenes carry no explanatory text on screen. This is what they are saying, beat by beat, so
the wording lives somewhere even though it is not rendered.

Every scene holds the frame where a line goes — in `assignment.py` those pauses are explicit
`_narration_beat()` calls, and elsewhere they are the `wait()` after each movement. The timings
below are the video's own, so a line written to fit its beat will fit the cut.

A caveat on provenance: the lines for **the roster-slot assignment** were on screen as captions
and were overwritten in place when the text came out, before that file was ever committed. They
are reconstructed here from the scene's structure and its code comments rather than recovered
verbatim. Everything else is the wording that was actually rendered.

---

## Punting — `punting.py` → `PuntingSearch` (22s)

Reading the panels, if the narration needs to say it: the bell never moves, because it is the
distribution of how the category comes out and that does not change. The bar is the threshold,
and you win the category when the result lands to its left. Yellow is the half you get for
showing up; green is what effort bought beyond that; red is what abandoning a category gave back.

| Beat | Line |
|---|---|
| Opening, nine bells at parity | One unit of effort in each: nine coin flips. Four and a half categories. |
| Score readout appears | That is what perfect balance is worth. |
| First category drains | Take the effort out of one category and spread it over the other eight. You lose ground there — and gain a little everywhere else. |
| Score ticks up | It went up. |
| Second, third drain | And again. And again. |
| Fourth drains, score falls | The fourth one costs more than it buys. Three is the answer. |
| Rests at three punted | Nine coin flips is 4.500. Giving up three outright is 4.625 — you win more categories by refusing to compete in a third of them. |

Worth saying once somewhere: the punted categories still come in about sixteen percent of the
time, which is why the real algorithm never drives a punted weight all the way to zero either.

---

## Roster-slot assignment — `assignment.py` → `RosterSlotAssignment` (48s)

*Reconstructed — see the caveat above.*

| Beat | Line |
|---|---|
| Slot headers appear | Thirteen roster slots. |
| Drafted rows fill in | Five players already drafted, and the slots each of them can fill. |
| The zeros land | Every one of those cells is a zero. A player you already have is worth the same to you wherever he lines up. |
| Future rows appear | The eight picks you have not made yet are where the value is. |
| Reward numbers revealed | All eight rows are identical — a future pick is worth whatever the slot is worth, because you have not met him yet. |
| Heat shading | Utility is the most valuable slot on the board. Centre is the least. |
| One marker per row | An assignment takes one cell per row… |
| Column violation rejected | …and never two in a column. Every slot gets used exactly once. |
| The naive guess | Jabari Smith can play three positions — so put him somewhere flexible? |
| Its total | That costs you. The utility slot was the most valuable thing on the board, and you just spent it on a player who is worth zero in it. |
| The solve | Move him to small forward, and the utility slot goes back to a pick you have not made. |
| Final total | The five drafted rows scored zero either way. The optimiser was never placing them — it was moving them out of the way. |

The numbers to quote: the optimum leaves **0.246** to the eight remaining picks; the flexible-slot
guess leaves **0.194**. Both C slots go to the two centre-only players, which is what forces
Karl-Anthony Towns off centre and onto power forward.

---

## The Z-score scene — `team_differential.py` → `TeamDifferentialFull`

Your lines, against the beats they belong to. Not yet wired into the scene.

**Act zero — the pool grid, 156 faces, three random pairs of thirteen ringed out of them.**

> Modeling real drafts is complex, but what if we imagined players got chosen randomly? That way
> we could end up with a simple metric.

> For a weekly matchup, we have two teams of 13 players chosen randomly, for a total of 26
> players. That's quite a few, and it means that the central limit theorem comes into play, which
> says that when you add or subtract a bunch of random numbers together, the end result looks
> like a random bell curve.

The second line runs long for act zero alone, and the CLT half of it is really about what the
histogram then does — it may want to land over the first draws in act one or over the montage,
where the bell is actually arriving, rather than over the grid.

**Acts four and five — the curve, then the density at zero.**

> Notice how the height of the bar tells us how important a single additional point is. If we
> move over to the right when the bar is high, that's a lot of scenarios where the extra point
> helped us win. So we can say, the height of the bar is how important the stat is. Fortunately,
> there is a simple expression for how high the bar is at the top.

That lands exactly where act five already goes: the line sets up the question and the animation
answers it, with the expression turning yellow as x becomes zero and the yellow marker rising to
the height being talked about. Worth noting the scene currently marks ±1 standard deviation
during act four — if the narration is about the height at the middle, those markers may be
competing for attention and could come out.

## Z versus G — `z_versus_g.py` → `VarianceQuadrature` (NARRATED)

This one is wired for voice already. Its lines live in the `NARRATION` dict at the top of the
scene file, currently filled with placeholders that state the beat rather than the wording. Edit
that block and nothing else: every animation takes its `run_time` from however long the audio
turns out to be, so rewriting a line retimes the picture instead of desynchronising it.

Watch the lengths. The placeholders are roughly the duration each beat wants, and the scene runs
74 seconds with them in — a finished script that is tighter will pull the whole thing shorter.

A first draft of the wording, if it is useful as a starting point rather than a constraint:

| Key | Draft line |
|---|---|
| `opening` | A Z-score measures a player against the spread of other players. A G-score widens that to include something a Z-score ignores entirely. |
| `cross_player` | Here is the first source of variation: who you happened to draft. Every player performing at exactly their weekly average, so the only thing changing is which players you got. That is a spread of eighty-six points. |
| `week_to_week` | Now hold the draft still and change only the week. The same twenty-six players, a different week of basketball. A hundred and four. |
| `both` | Let both vary and you get a hundred and thirty-seven. That is what a G-score prices, and a Z-score sees only the first of the three. |
| `the_question` | But eighty-six and a hundred and four do not add to a hundred and thirty-seven. |
| `right_angle` | They combine at a right angle, because the two are independent. Which draft you got tells you nothing about which weeks your players then had. |
| `squares` | It is the squares that add. The areas, which are variances. Standard deviations do not add. Variances do. |

One thing the script should probably not claim: the triangle closes to 135 against a measured
137. That is sampling noise at ten thousand draws, not an error, but a line asserting the three
numbers fit exactly would be overstating it.

## The three differential scenes

`team_differential.py` → `TeamDifferentialFull`, `z_versus_g.py` → `FixedMatchupFull`,
`weekly_differential.py` → `WeeklyDifferentialFull`. These keep two on-screen labels that are
not narration and should stay: the axis caption, and the standard-deviation readout in act four.

One line was removed from act five and belongs in the voice instead:

> Both teams are dealt from the same pool, so neither is favoured — the curve is centred on zero.

That matters because the formula on screen is written with μ already zero. Without the sentence,
a viewer who knows the Normal density will wonder where the mean went.

The rest of act five is meant to be watched rather than explained: the expression turns yellow at
the moment x becomes zero, the yellow line rises on the curve at the same instant, and the
exponential vanishes. If anything is said over it, the point is that **the height of the curve at
a dead heat is one over sigma root two pi** — so the tighter the distribution, the more a small
edge moves the odds. That is the whole reason the spread is the number worth knowing.

Note that `FixedMatchupFull` is the exception to μ = 0: that curve sits over a real edge of about
+125, because one of those two teams genuinely is better. Do not narrate it as centred.

---

## The truncated-max scenes

`truncated_max_plane.py`, `truncated_max_reduction.py`, `truncated_max_scalar_core.py` still
carry their own on-screen text, unlike the scenes above. If they are to match, the text comes out
and the lines move here.
