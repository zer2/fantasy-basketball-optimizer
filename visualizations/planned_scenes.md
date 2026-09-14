# Planned scenes

Sketches for scenes not yet built, in the order I'd build them. Each says what it shows, how it
goes, what has to be computed first, and roughly what it costs.

**Status, 2026-09-13: every sketch below has now been built as a first draft.** Three of them
came back with a result that contradicts what the sketch predicted, and in each case the scene
was rebuilt around the measurement rather than the prediction:

- **Three formats** predicted that Most Categories would punt harder and Rotisserie less. It
  does not: all three formats punt exactly three categories. What differs is how much the punt
  is WORTH — Most Categories +9.5% over perfect balance, Each Category +2.8%, Rotisserie +1.1%.
- **Self-play** predicted that best-responding to the running average is what damps the
  oscillation. In the model it is not enough on its own: a field searching globally still
  thrashes when averaged. What settles it is responding LOCALLY, from where each seat already
  is — which is what a warm start does. That may be a property of the toy; it is recorded as
  what the toy did.
- **Cold, multi and warm starts** predicted a cold start climbing to a nearby, worse peak. It
  does worse than that: perfect balance is a stationary point of the objective, so the cold
  start does not climb at all. Zero steps, 4.5000, while every challenger seed reaches 4.6247.

The sketches are kept below as written, because what they got wrong is part of the record.

Regularization was considered and deliberately held. As "an L1 penalty pulls the weights toward
neutral" it is a bar chart being tugged, and the idea is already clear in a sentence of prose.
It only earns an animation as a narrative — two teams drafting the same board, one regularized
and one not, a star falling in round four that the committed team cannot use — and that version
is the most expensive thing on this page by a distance, with an honesty problem the others lack:
in any single draft the unregularized team might simply win, so making the case properly means
showing many drafts, at which point it is a statistics scene rather than a story.

---

## 0. Why random drafting is not a cop-out

**The defence of the Z-score scene.** Its obvious criticism is that nobody drafts at random — and
the answer is not "it is a rough approximation" but that a value-ordered snake draft *reduces to*
random drafting exactly. That is a much stronger claim and it deserves its own piece.

**The model.** Total value is linear in draft pick order: the first pick is worth 4, the second
3.9, the third 3.8. On top of that value, each player gets a category tilt from a random process
R — a redistribution across categories that does **not** change total value. That last condition
is load-bearing rather than decorative: because the tilts are value-neutral, drafting by total
value is exactly drafting in pick order, so the k-th pick really is the player carrying the k-th
slot value and there is nothing circular in saying so.

### The four steps

**One — the ladder.** Five draft slots with their values written out: 4, 3.9, 3.8, 3.7, 3.6. A
straight line through them makes the linearity the visible assumption rather than a buried one.

**Two — value becomes a shape.** Each player's total value spreads into nine category bars, and
beneath them a few draws of R as their own bars. The point to land is that R moves value BETWEEN
categories and never changes the total — worth showing by letting a tilt's bars settle and having
the running total not move at all.

**Three — the board, paired.** The full 12 × 12 snake with pick numbers. Do **not** jump to the
totals: pair each odd round with the round below it and show that every pair closes to the same
number for every seat. Seat 1 gets 1 + 24, seat 12 gets 12 + 13, seat 6 gets 6 + 19 — all 25. The
six pair-sums are 25, 73, 121, 169, 217, 265 for every seat in the league, totalling 870. A viewer
watching pairs close does not have to take the equal totals on trust.

**Four — the cancellation.** Two seats side by side, first and last. Seat 1 is
(R₁ + c₁) + (R₂₄ + c₂₄) + …, seat 12 is (R₁₂ + c₁₂) + (R₁₃ + c₁₃) + …. Strike the c terms against
each other — they are the same 870 on both sides — and what is left is twelve draws of R against
twelve draws of R. Which is the two-random-teams model, arrived at rather than assumed.

**Five — the scaling, and its caveat.** Steps one to four settle the DIFFERENTIAL: baselines
cancel, so what separates two teams is their draws of R. But a Z-score does not divide by R's
spread, it divides by the POOL's spread, which contains the ladder as well. Those line up only if
R's spread in a category is proportional to that category's ladder slope — in which case pool
sigma and R's sigma differ by one constant shared across every category, and a shared constant is
invisible to a score that only compares categories against each other.

Then concede it. In reality some categories carry far more of their spread in the ladder than
others, and the scene should say so rather than hope nobody checks.

### The narration

1. *The random drafting setup can be equated to a more complex and realistic setup. We just need
   to assume that value per category is on average linear in pick order, with some random process
   on top of that, with standard deviations proportional to the linear bonus. We start with a
   baseline level of each category for the pick, then use a random process we will call R to add
   an individual category-level profile.*
2. *This means that team stats can be calculated as several draws of R, plus the baseline stats
   for their picks. If we plot out a full snake draft, we will notice that the total of each
   drafter's pick numbers is the same. That means that their baseline stats will all cancel out,
   leaving each team with only their draws of R.*
3. *The standard deviation of a category for the whole player pool comes from R, plus a small
   amount from the linear adjustment. If R and the linear adjustment are proportional, as we
   assumed, the linear adjustment doesn't matter — the standard deviation of a category is still
   proportional to the standard deviation of R, validating Z-scores.*
4. *In reality this isn't necessarily true: some categories have more inter-round variance than
   others. But we're looking for an approximation here.*

### What the data says about the assumption (measured, 2025-26)

Proportionality predicts that the share of pool variance explained by draft rank is the SAME in
every category — the slope cancels out of the ratio. Measured over the 156 drafted players:

    Field Goal %   0.3%      Rebounds   22.0%
    Free Throw %   7.9%      Assists    23.1%
    Threes        13.3%      Steals     17.4%
    Points        54.3%      Blocks      9.2%
                             Turnovers  25.6%

One number repeated nine times is what the assumption requires; the spread runs 0.3 to 54. Points
is heavily draft-order-driven, so the R left over is proportionally small (sigma_R ~ 0.68 of pool
sigma) where Field Goal % is almost pure R (~0.999). Since importance goes as 1/sigma_R while a
Z-score divides by pool sigma, Z-scores understate Points against Field Goal % by roughly 45%.

Caveat on those figures: rank is by total G-score, which is built from the categories, so part of
the correlation is mechanical and the absolute percentages are inflated. The DISPERSION across
categories is the finding, and mechanical correlation cannot manufacture a 0.3-versus-54 spread.

Two ways to play beat four. Say it in words and move on, or put these numbers on screen as the
concession. The second is braver and probably better — a scene that names the size of its own
approximation is harder to argue with than one that gestures at it — and it doubles as the setup
for a future piece, since "Z-scores misprice categories according to how much of their spread is
draft-order-driven" is a real claim with real numbers behind it.

### Two decisions before building

**Twelve rounds, not thirteen.** The cancellation is exact only for an even number of rounds. At
13 rounds the pick-number totals ladder from 1015 to 1026 and seat 1 keeps a systematic edge the
snake never pays back. Every other scene in the set uses 13 rounds and 156 players, so this one
either uses 12 and says why, or the discrepancy gets noticed.

**Which way the turnover bar points.** "Make turnovers negative" can mean either the raw stat is a
bad thing, or the player is good at avoiding them and so scores positively. The two give opposite
bars and the scene should pick one deliberately.

### Needs and cost

Synthetic throughout — this is a claim about structure, not about 2025-26, so no prep script and
no real data are required. Headshots could carry step one to tie it to its neighbours. Steps one
and four are simple text and algebra; step two is a nine-bar chart; step three is a numbered
12 × 12 grid with a pairing animation. **Two to three days**, and the honest caveat about
linearity belongs in the narration: the argument buys exactness *given* that value is linear in
pick order, which moves the unrealism from "drafting is random" to "value is linear" — a far more
defensible place to put it.

## 1. Category-level gradient

**Extends `punting.py` rather than standing alone.** Cheapest thing here and the highest value
per hour, because the quantity it is about is already drawn on screen without being labelled.

The algorithm's care for a category is the Normal density at the threshold — which is exactly
the height of the bell where the bar crosses it. The punting scene draws both already.

**Beats.** Add a marginal-value readout to each panel, live as the threshold moves. Then hold
the grid still and sweep one category's bar from far left to far right while its readout rises
to a peak at parity and falls away on both sides. Close by putting all nine marginal values side
by side at the punt optimum: the six contested categories share one value, the three abandoned
ones sit lower — which is the first-order condition made visible, and the reason the search
settled where it did.

**Why it matters beyond punting.** At convergence the optimizer's weights are proportional to
these gradients, so each weight reads as the marginal value of its category. That is the
`Jw = 0` property from the weight-model note, and this is the scene that makes it legible.

**Needs.** Nothing new — no prep script, no data. The density at the threshold is already
computed to draw the shading.

**Cost.** An afternoon, inside an existing file.

---

## 2. Three formats, three shapes of payoff

**Replaces two separate ideas (Rotisserie, Most Categories) with one comparison**, because
separately each would repeat the same setup to make a claim a viewer cannot check. The formats
differ only in the shape of the payoff, and that difference is the whole story:

| Format | Payoff | What it rewards |
|---|---|---|
| Each Category | sum of Φ — saturating | being 90% likely somewhere wastes effort |
| Most Categories | a step at winning 5 of 9 | categories past the fifth are worth nothing, so extremes are cheap |
| Rotisserie | rank among twelve — roughly linear in percentile | nothing is ever cheap |

**Beats.** Open on the nine category outcomes from the punting scene, in the same visual
language. Draw the payoff curve for Each Category beneath them and run the punt search: it lands
on three punts. Swap the curve for the Most Categories step and re-run: it punts harder. Swap for
the Rotisserie rank ladder and re-run: it punts less, or not at all. Three runs, one apparatus,
the curve underneath being the only thing that changed.

**Verify before building.** The claim that MC punts harder and Roto punts less comes from the
docs; run the same search under all three payoffs first and check the optima actually move that
way. If they do not, the scene is about whatever they really do.

**Needs.** A payoff-curve abstraction in the punting search, and a Rotisserie model — rank among
twelve rather than a margin over one opponent, which is a genuinely different computation.

**Cost.** Two or three days, mostly in getting the Roto payoff honest.

---

## 3. The self-play loop

**The only scene in the set that is about other people.** The field's strategies are not assumed,
they are discovered, and how they are discovered is a real story with a villain.

From `algorithm_agents.py`: the base H-scores are recomputed over 32 passes, each best-responding
to the **running average** of prior passes rather than to the latest. The comments say why — pure
best-response *oscillates*, every seat flipping to the same punt each pass — and averaging is what
damps it into a mixed equilibrium where punts spread out across archetypes.

**Beats.** Twelve seats, each a compact nine-category weight strip. Pass one: everyone neutral.
Run pure best-response first and let it visibly thrash — the whole field swinging onto one punt,
then off it together. Then restart with the running average and watch the same field settle,
punts spreading rather than herding. Finish on the measured drift falling (0.012 → 0.0006) and
the archetype mixture holding still.

**Needs.** A harness that runs both update rules and records per-pass weights for twelve seats.
The real one is already instrumented — the drift diagnostic is logged — so most of this is
capturing what already runs.

**Risk.** Legibility. Twelve nine-category strips is a lot on screen at once; this needs a
deliberate design pass before any code, or it is soup.

**Cost.** Three or four days.

---

## 4. Cold, multi, and warm starts

**A limitation the docs already admit, with a number behind it.** Gradient descent finds local
optima, and each punt is one — so where you start decides which one you reach.

The seed menu in `algorithm_agents.py` is: one gentle punt per category, plus the balanced build,
plus each candidate's own converged build from earlier passes. And a comment records the payoff:
**challenger seeds win 24% of bootstrap solves.** A non-obvious start finds a better answer about
a quarter of the time, which is what turns this from housekeeping into a claim.

**Beats.** Draw the objective over a two-punt slice of weight space as a landscape with several
peaks. Drop a cold start at neutral and let it climb — it finds the nearest peak, not the best
one. Drop the full seed menu and let every seed climb at once, with the winner taking the board.
Then a warm start: last pass's converged build, already most of the way up. Close on the 24%.

**Needs.** An objective surface evaluated over a 2D slice of weight space (fix seven categories,
vary two), plus descent traces from each seed. All offline, using the shipped objective.

**Cost.** Two days. The landscape is the interesting part and it is straightforward to compute.

---

## Build order

The category gradient first — it is an afternoon and it upgrades a scene that already exists.
Then the three formats, then self-play. That trio covers why punting works, why the format
decides how much, and what happens when every seat reasons that way at once.

"Why random drafting is not a cop-out" sits outside that ordering. It is not the next thing to
build, but it is the first thing to WATCH: it belongs immediately before or after the Z-score
scene, because it answers the objection that scene raises in anyone who is paying attention.
