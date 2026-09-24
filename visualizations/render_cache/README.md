# render_cache

Everything manim writes while rendering. `manim.cfg` points `media_dir` here, so this directory
is recreated on the next render whether or not it exists.

Nothing in here is the deliverable. **The published videos live in `docs/videos/`**, with
`ffmpeg -movflags +faststart` applied — manim writes the `moov` atom last, which stops a browser
seeking a file it is still streaming.

| directory | what it is | safe to delete? |
|---|---|---|
| `voiceovers/` | **Purchased narration.** ElevenLabs audio, billed per character, plus `cache.json` | **No. Never.** |
| `videos/` | Rendered scenes, and `partial_movie_files/` — the per-animation render cache | Yes, costs re-render time |
| `texts/`, `Tex/` | Font and LaTeX rasterisation cache | Yes, rebuilt automatically |
| `images/` | Single-frame output from `manim -s` | Yes |

## voiceovers/ is money, not cache

Every line of narration in the set was paid for once. `cache.json` is what stops it being paid
for twice: manim-voiceover looks a line up by its exact text, and a hit costs nothing while a
miss is a fresh purchase. Delete this directory and the next render re-buys all of it.

Two things follow. Editing a line by a single word makes it a new line at full price — whitespace
excepted, since the service collapses that before hashing. And the directory name is fixed:
manim-voiceover derives it as `media_dir/voiceovers`, so renaming it silently orphans the cache
and every line is bought again.

`python visualizations/shared/narration_budget.py` reports what a render would spend and what is
already paid for. Run it before rendering anything whose narration has changed.
