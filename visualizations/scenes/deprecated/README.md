# Deprecated scenes

Superseded, kept because they render and because the shape of an earlier attempt is sometimes
worth re-reading. Nothing in the active set imports them; nothing here is maintained.

They still expect to be run from `scenes/`, since they import `differential_base` /
`truncated_max_base` as top-level modules. Rendering one means either copying it back up a
directory or adding `scenes/` to the path.

| File | Superseded by | Why |
|---|---|---|
| `truncated_max_plane.py` | `plane_story.py` | Same idea, but carried matrix algebra and a weight arrow sitting at right angles to the line it controlled. The replacement shows the weights as two numbers and slides the line instead of turning it. |
| `truncated_max_reduction.py` | `plane_story.py` | The (s, u) ellipse and its marginals. Once the pick distribution is something you have watched move, this is a formalisation of it rather than a revelation. |
| `truncated_max_scalar_core.py` | — | The skew-normal, the best-of-M density, and the Gumbel step up to e(rho). Not superseded: retired for being the wrong depth. It explains how the model is COMPUTED rather than what it does, which is a question the weight-model note already answers in writing, and it puts its own approximation error on screen. |
| `truncated_max_base.py` | — | Shared by the three above and nothing else. |

`prepare_truncated_max_data.py` and `data/truncated_max.json` are still in their original
places, because a prep script that resolves paths relative to itself would break if moved
and nothing here can be revived without them. They feed only deprecated scenes.

`team_differential.py` and `weekly_differential.py` were moved here briefly and put back. They
ARE the Z-score and G-score simulations — the two runs the Z-versus-G story is made of — and
`VarianceQuadrature` does not replace them: it draws three static bells from the data files
rather than playing either simulation.
