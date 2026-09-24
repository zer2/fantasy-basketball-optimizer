# Wireframes

First drafts of scenes that are not yet part of the set of eight. A wireframe exists to answer
one question — *does this argument work as a picture?* — before any effort goes into making it
look finished.

Three things follow from that, and they are what separates this folder from the numbered ones:

- **The voice is gTTS, not Alistair.** `shared/draft_voice.py` — free, and it keeps drafts out of
  the paid cache while the script is still moving. gTTS reads faster and flatter, so timings here
  are indicative. Beats anchored with `narration_timing` follow the words either way; fixed
  `run_time`s want a look after the swap.
- **Numbers are stand-ins, not measurements.** Every shipped scene is measured against the real
  algorithm by a prep script. These are not, and say so in their docstrings. A prep script is
  part of promoting one, not part of drafting it.
- **Unfinished beats are marked `PLACEHOLDER`,** with a note on what the finished version needs.
  They exist so the beat can be timed and the scene can be watched end to end.

| folder | scene | what it argues | where it would go |
|---|---|---|---|
| `most_categories/` | `MostCategories` | the majority payoff is a sum over 512 branches, that tree collapses into a ten-bar tally, and the tally gives the tipping-point probability that decides what a category is worth | `hscores.md`, Most Categories |
| `rotisserie/` | `Rotisserie` | you need an aberrant season to win a league, so spread is worth having on its own — and spread is largest when categories sit near fifty-fifty | `hscores.md`, Rotisserie |
| `savor/` | `Savor` | a projection has a floor under it because a bust gets dropped, and the thing you are bidding against is a free player with the same floor; subtracting the two is the SAVOR formula | `auctions.md`, the SAVOR adjustment |

```
manim -ql visualizations/wireframes/most_categories/most_categories.py MostCategories
manim -ql visualizations/wireframes/rotisserie/rotisserie.py Rotisserie
manim -ql visualizations/wireframes/savor/savor.py Savor
```

## Promoting one

1. Settle the script. Everything else is cheaper once the words stop moving.
2. Write a prep script so the numbers are measured rather than chosen.
3. Fill in the `PLACEHOLDER` beats.
4. Swap `DraftVoice()` for `NarrationVoice()` in `construct`, and re-check any fixed `run_time`
   against the slower voice.
5. Move the folder to a numbered one, add it to the sign-off table in the parent README, and
   embed it with a `/// caption` block the way the other eight are.
