"""Measure how much of each category's spread is explained by draft order.

The draft-equivalence scene argues that a value-ordered snake draft reduces to random drafting,
and its last beat concedes where the argument strains. That concession has to be measured rather
than asserted, which is what this script does.

The model behind the scene splits a player into a baseline that is linear in pick order plus a
value-neutral tilt R. Everything linear in pick order cancels between drafters, so what actually
separates two teams is R -- and a Z-score, which divides by the POOL's spread rather than R's,
is only right if those two are proportional across categories.

Proportionality makes a sharp prediction: the share of pool variance explained by draft rank must
be IDENTICAL in every category, because the category's own slope cancels out of the ratio. This
script measures that share per category so the scene can show how far the prediction holds.

    python visualizations/prepare_draft_equivalence_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.api.routers.sessions import _build_current_settings          # noqa: E402
from backend.services.session_management import build_session             # noqa: E402
from prepare_season_data import build_default_session_request, SEASON, SPORT   # noqa: E402

_VISUALIZATIONS_DIR = Path(__file__).resolve().parent


def measure_ladder_share(session, pool_size: int) -> list[dict]:
    """Per category: how much of the drafted pool's spread a straight line in rank accounts for.

    The fit is deliberately LINEAR in rank, because linear-in-pick-order is exactly the model's
    assumption about the baseline. What the line explains is the part that cancels between
    drafters; what it leaves is R.
    """
    g_scores = session.agent.info['G-scores']
    drafted = g_scores['Total'].sort_values(ascending=False).head(pool_size).index
    categories = [column for column in g_scores.columns if column != 'Total']

    rank = np.arange(pool_size)
    measured = []
    for category in categories:
        values = g_scores.loc[drafted, category].to_numpy()
        slope, intercept = np.polyfit(rank, values, 1)
        residual = values - (slope * rank + intercept)
        ladder_share = float(1 - np.var(residual) / np.var(values))
        measured.append({
            'category': category,
            'ladder_share': round(ladder_share, 4),
            # What is left once the ladder is taken out, as a fraction of the pool's spread. This
            # is the number the scene plots: square-rooting a variance share is what turns an
            # alarming 181-fold range into a 1.5-fold one, and the spread of THIS is the honest
            # measure of how far the approximation is off.
            'residual_spread_fraction': round(float(np.sqrt(1 - ladder_share)), 4),
            'slope_per_pick': round(float(slope), 5),
        })
    return measured


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults...')
    session = build_session(
        current_settings = _build_current_settings(
            build_default_session_request(SEASON, SPORT)),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )
    pool_size = (session.current_settings['n_drafters']
                 * session.current_settings['n_picks'])
    measured = measure_ladder_share(session, pool_size)

    fractions = [row['residual_spread_fraction'] for row in measured]
    print(f'\n{"category":16} {"ladder share":>13} {"R as share of pool spread":>27}')
    for row in sorted(measured, key=lambda r: r['residual_spread_fraction']):
        print(f'{row["category"]:16} {row["ladder_share"] * 100:12.1f}% '
              f'{row["residual_spread_fraction"]:27.3f}')
    print(f'\nvariance shares span a factor of '
          f'{max(r["ladder_share"] for r in measured) / max(1e-9, min(r["ladder_share"] for r in measured)):.0f}; '
          f'the spreads themselves span only {max(fractions) / min(fractions):.2f}')

    data_path = _VISUALIZATIONS_DIR / 'data' / 'draft_equivalence.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':     SEASON,
        'pool_size':  pool_size,
        'n_drafters': session.current_settings['n_drafters'],
        'categories': measured,
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')


if __name__ == '__main__':
    main()
