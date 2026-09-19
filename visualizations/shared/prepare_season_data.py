"""Build everything the team-differential scene needs, so rendering never touches Snowflake.

Run once (or whenever the season or the pool definition changes):

    python visualizations/prepare_season_data.py

Writes two things into this directory:
  data/pool_<season>.json       the drafted pool, each player's name and per-game Points,
                                plus the precomputed simulation results
  assets/headshots/<id>.png     one circular headshot per pooled player

Keeping the render offline matters because the scene gets re-rendered dozens of times while
its timing is tuned: a Snowflake round trip per render would dominate, and a pool that
shifted underneath a re-render would make two cuts of the same scene incomparable.

The pool is the top N_DRAFTERS x N_PICKS players by G-score -- the players a standard league
actually drafts. G-score rather than Points on purpose: ranking by the same statistic the
scene plots would select on the very quantity being illustrated, truncating the distribution
it is meant to show.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Import the app itself rather than reimplementing its numbers.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.api.routers.sessions import _build_current_settings          # noqa: E402
from backend.api.schemas import (                                         # noqa: E402
    DataSource, LeagueSettings, ModelSettings, SessionRequest,
)
from backend.data_retrieval import (                                      # noqa: E402
    get_specified_historical_stats, get_weekly_box_scores,
)
from backend.parameters import load_all_params                            # noqa: E402
from backend.services.session_management import build_session             # noqa: E402


SEASON = '2025-26'
SPORT = 'NBA'
STATISTIC = 'Points'
# The key each pool entry carries its plotted value under: a player's MEAN WEEKLY total,
# not their per-game average, so both scenes share one unit and one axis.
WEEKLY_AVERAGE_KEY = 'weekly_average'
# Ten thousand rather than a thousand, for the number the scene puts on screen rather than
# for smoothness: across seeds a thousand draws estimate the spread anywhere from 27.8 to
# 31.4 against a true 30.1, so the displayed standard deviation was luck of the seed. Ten
# thousand pins it to about a fifth of a point. The draws cost nothing; the claim matters.
SIMULATION_COUNT = 10000
TEAM_SIZE = 13
RANDOM_SEED = 4242            # fixed so every render of the scene shows the same draws

# Square side of the written headshot. A roster face occupies about 100 pixels of a 1080p
# frame, so this is a modest oversample -- 256 was two and a half times the display size
# and quadrupled what these committed assets cost.
HEADSHOT_PIXELS = 128
_VISUALIZATIONS_DIR = Path(__file__).resolve().parent.parent
_HEADSHOT_SOURCE_DIR = _VISUALIZATIONS_DIR.parent / '.cache' / 'headshots'


def build_default_session_request(season: str, sport: str) -> SessionRequest:
    """A session on `season` at the app's own default settings, which is what fixes the pool.

    Every parameter comes from parameters.yaml rather than being restated here, so the pool
    tracks the app's defaults instead of drifting from them silently.
    """
    all_params = load_all_params()
    sport_params = all_params[sport]
    options = sport_params['options']
    default_of = lambda name: options[name]['default']        # noqa: E731 - one-liner lookup

    n_picks = default_of('n_picks')
    position_slots = options['positions'][n_picks]
    slot_counts = {**position_slots['base'], **position_slots['flex']}

    return SessionRequest(
        league = LeagueSettings(
            sport                  = sport,
            n_drafters             = default_of('n_drafters'),
            n_picks                = n_picks,
            scoring_format         = 'Head to Head',
            most_categories_weight = default_of('most_categories_weight'),
            categories             = sport_params['default-categories'],
        ),
        slot_counts = slot_counts,
        model_settings = ModelSettings(
            pick_pool_size            = default_of('pick_pool_size'),
            beth                      = default_of('beth'),
            upsilon                   = default_of('upsilon'),
            psi                       = default_of('psi'),
            chi                       = default_of('chi'),
            aleph                     = default_of('aleph'),
            lambda_c                  = default_of('lambda_c'),
            lambda_p                  = default_of('lambda_p'),
            opponent_model_confidence = default_of('opponent_model_confidence'),
            n_iterations              = default_of('n_iterations'),
            streaming_noise           = default_of('S'),
        ),
        data_source = DataSource(type='historical', season=season),
    )


def select_drafted_pool(session, season_statistics: pd.DataFrame, statistic: str) -> list[dict]:
    """The players a standard league drafts, best G-score first, with the plotted statistic.

    Membership comes from the session's G-scores; the value carried alongside is the player's
    real per-game average for the season, straight from the source table. G-scores are
    standardised -- a Points G-score of 0.98 means "a standard deviation above the field",
    which is the wrong unit for a scene whose whole point is that team totals are real
    basketball numbers.

    Returns [{player_id, name, <statistic>}] of exactly n_drafters * n_picks players.
    """
    g_scores = session.agent.info['G-scores']
    pool_size = session.current_settings['n_drafters'] * session.current_settings['n_picks']
    drafted_ids = g_scores['Total'].sort_values(ascending=False).head(pool_size).index

    unscored = [int(i) for i in drafted_ids if i not in season_statistics.index]
    if unscored:
        raise SystemExit(f'{len(unscored)} drafted players are absent from the {statistic} '
                         f'table (ids {unscored[:5]}...) -- the pool and the statistics '
                         f'disagree about who played this season.')

    registry = session.player_registry
    return [
        {
            'player_id': int(player_id),
            'name':      registry[int(player_id)].name,
            statistic:   float(season_statistics.loc[player_id, statistic]),
        }
        for player_id in drafted_ids
    ]


def simulate_team_differentials(
    values: np.ndarray
    , simulation_count: int
    , team_size: int
    , seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deal two teams from the pool, repeatedly, and total each side's statistic.

    Returns (rosters, totals): rosters is (simulation_count, 2 * team_size) pool indices, the
    left team first; totals is (simulation_count, 2). Dealing is without replacement across
    BOTH teams, because one player cannot be on both, which is also what makes the two totals
    dependent and narrows the differential relative to independent sampling.
    """
    rng = np.random.default_rng(seed)
    rosters = np.stack([
        rng.choice(len(values), size=2 * team_size, replace=False)
        for _ in range(simulation_count)
    ])
    totals = np.stack([
        values[rosters[:, :team_size]].sum(axis=1),
        values[rosters[:, team_size:]].sum(axis=1),
    ], axis=1)
    return rosters, totals


def collect_weekly_values(
    pool: list[dict]
    , weekly_box_scores: pd.DataFrame
    , statistic: str
) -> list[list[float]]:
    """Each pooled player's real weekly totals for `statistic`, in season order.

    Ragged on purpose: players appear in between fourteen and twenty-five weeks of this season,
    and a player's missing weeks are weeks they did not play. Padding those to zero would put
    games nobody played into the sample, so a week is only ever drawn from the weeks a player
    actually has.
    """
    keyed_by_player_id = weekly_box_scores.reset_index().set_index('NBA_PLAYER_ID')
    weekly_values = []
    for player in pool:
        player_weeks = keyed_by_player_id.loc[[player['player_id']]].sort_values('WEEK')
        weekly_values.append([float(value) for value in player_weeks[statistic]])
    return weekly_values


def simulate_weekly_team_differentials(
    weekly_values: list[list[float]]
    , simulation_count: int
    , team_size: int
    , seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deal two teams, then deal each drafted player one of their own real weeks.

    The averages version gives every player the same number every time, so the only thing that
    varies between simulations is who was drafted. Here a player brings one week they actually
    played, which adds the variation a season average smooths away -- the reason a favourite
    loses a week.

    Returns (rosters, totals) in the same shape as simulate_team_differentials, so both scenes
    read the same structure.
    """
    rng = np.random.default_rng(seed)
    week_counts = np.array([len(weeks) for weeks in weekly_values])
    # Rectangular copy of a ragged list, padded past each player's real week count. The padding
    # is never sampled: draws are taken modulo the player's own count.
    padded = np.zeros((len(weekly_values), week_counts.max()))
    for player_index, weeks in enumerate(weekly_values):
        padded[player_index, :len(weeks)] = weeks

    rosters = np.stack([
        rng.choice(len(weekly_values), size=2 * team_size, replace=False)
        for _ in range(simulation_count)
    ])
    drawn_weeks = (rng.random(rosters.shape) * week_counts[rosters]).astype(int)
    drawn_values = padded[rosters, drawn_weeks]

    totals = np.stack([
        drawn_values[:, :team_size].sum(axis=1),
        drawn_values[:, team_size:].sum(axis=1),
    ], axis=1)
    return rosters, totals


def simulate_one_matchup_repeatedly(
    weekly_values: list[list[float]]
    , simulation_count: int
    , team_size: int
    , seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Deal the teams ONCE, then play that same matchup over and over in different weeks.

    The third of the three simulations, and the one that isolates the term the other two differ
    by. Holding the draft fixed removes cross-player variation entirely, so everything left is
    week-to-week: the same twenty-six players, a different week each time. Its spread is what,
    squared and added to the cross-player spread, makes the full one.

    It is also the only one of the three that is not centred on zero -- one of these two teams is
    genuinely better than the other, and the curve sits over that edge. That is the question
    H-scoring exists to answer, so the asymmetry is the point rather than a defect.

    Returns (rosters, totals) in the shape the scenes read, with the one roster repeated so the
    display code needs no special case.
    """
    rng = np.random.default_rng(seed)
    week_counts = np.array([len(weeks) for weeks in weekly_values])
    padded = np.zeros((len(weekly_values), week_counts.max()))
    for player_index, weeks in enumerate(weekly_values):
        padded[player_index, :len(weeks)] = weeks

    # Which twenty-six players get dealt matters here in a way it does not elsewhere, because
    # this simulation reports THEIR week-to-week spread rather than the pool's. A draft holding
    # unusually streaky players would measure a term that does not square up with the other two
    # scenes, so the matchup is chosen to carry the pool's typical variance -- representative,
    # not cherry-picked for a dramatic result.
    player_variances = np.array([np.var(weeks) for weeks in weekly_values])
    typical_variance_sum = 2 * team_size * player_variances.mean()
    candidates = [rng.choice(len(weekly_values), size=2 * team_size, replace=False)
                  for _ in range(500)]
    one_roster = min(candidates,
                     key=lambda roster: abs(player_variances[roster].sum() - typical_variance_sum))
    rosters = np.tile(one_roster, (simulation_count, 1))
    drawn_weeks = (rng.random(rosters.shape) * week_counts[rosters]).astype(int)
    drawn_values = padded[rosters, drawn_weeks]

    totals = np.stack([
        drawn_values[:, :team_size].sum(axis=1),
        drawn_values[:, team_size:].sum(axis=1),
    ], axis=1)
    return rosters, totals


def write_circular_headshot(source_path: Path, destination_path: Path, side_pixels: int) -> None:
    """Write `source_path` as a square, circularly-masked PNG with a transparent surround.

    Cropped to the square centred on the top of the source image: NBA headshots are wider
    than they are tall and put the face high, so a centre crop would cut the forehead.
    """
    with Image.open(source_path) as image:
        portrait = image.convert('RGBA')
        crop_side = min(portrait.width, portrait.height)
        left = (portrait.width - crop_side) // 2
        portrait = portrait.crop((left, 0, left + crop_side, crop_side))
        portrait = portrait.resize((side_pixels, side_pixels), Image.LANCZOS)

    mask = Image.new('L', (side_pixels, side_pixels), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, side_pixels - 1, side_pixels - 1), fill=255)
    portrait.putalpha(mask)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    portrait.save(destination_path)


def main() -> None:
    print(f'Building a {SEASON} session at the app defaults (this pulls from Snowflake)...')
    request = build_default_session_request(SEASON, SPORT)
    session = build_session(
        current_settings = _build_current_settings(request),
        platform_config  = None,
        csv_bytes        = None,
        uploaded_dfs     = None,
    )

    season_statistics = get_specified_historical_stats(SEASON, load_all_params()[SPORT])
    pool = select_drafted_pool(session, season_statistics, STATISTIC)

    # Both scenes work in WEEKLY units so their axes can be read against each other. A per-game
    # average and a weekly total differ by however many games fell in the week, so plotting one
    # scene in each would make the interesting comparison -- how much wider real weeks are --
    # an artefact of the units rather than a fact about basketball.
    weekly_box_scores = get_weekly_box_scores(SEASON, load_all_params()[SPORT])
    weekly_values = collect_weekly_values(pool, weekly_box_scores, STATISTIC)
    for player, weeks in zip(pool, weekly_values):
        player[WEEKLY_AVERAGE_KEY] = round(float(np.mean(weeks)), 4)

    values = np.array([player[WEEKLY_AVERAGE_KEY] for player in pool])
    print(f'Pool: {len(pool)} players by G-score, weekly {STATISTIC} '
          f'mean {values.mean():.2f}, sd {values.std():.2f}')

    rosters, totals = simulate_team_differentials(
        values, SIMULATION_COUNT, TEAM_SIZE, RANDOM_SEED)
    differentials = totals[:, 0] - totals[:, 1]
    print(f'{SIMULATION_COUNT} simulations on weekly averages: team totals average '
          f'{totals.mean():.1f}, differential sd {differentials.std():.1f}, '
          f'99% within +/-{np.percentile(np.abs(differentials), 99):.1f}')

    data_path = _VISUALIZATIONS_DIR / 'data' / f'pool_{SEASON.replace("-", "_")}.json'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.write_text(json.dumps({
        'season':           SEASON,
        'statistic':        STATISTIC,
        'value_key':        WEEKLY_AVERAGE_KEY,
        'team_size':        TEAM_SIZE,
        'random_seed':      RANDOM_SEED,
        'pool':             pool,
        'simulation_rosters': rosters.tolist(),
        'simulation_totals':  totals.round(4).tolist(),
    }), encoding='utf-8')
    print(f'Wrote {data_path.relative_to(_VISUALIZATIONS_DIR.parent)}')

    # ── The same draft, played out in real weeks rather than in weekly averages ──────
    weekly_rosters, weekly_totals = simulate_weekly_team_differentials(
        weekly_values, SIMULATION_COUNT, TEAM_SIZE, RANDOM_SEED)
    weekly_differentials = weekly_totals[:, 0] - weekly_totals[:, 1]
    week_counts = [len(weeks) for weeks in weekly_values]
    print(f'Weekly: {min(week_counts)}-{max(week_counts)} weeks per player, '
          f'team totals average {weekly_totals.mean():.1f}, '
          f'differential sd {weekly_differentials.std():.1f}, '
          f'99% within +/-{np.percentile(np.abs(weekly_differentials), 99):.1f}')

    weekly_path = _VISUALIZATIONS_DIR / 'data' / f'weekly_{SEASON.replace("-", "_")}.json'
    weekly_path.write_text(json.dumps({
        'season':           SEASON,
        'statistic':        STATISTIC,
        'value_key':        WEEKLY_AVERAGE_KEY,
        'team_size':        TEAM_SIZE,
        'random_seed':      RANDOM_SEED,
        'pool':             pool,
        'weekly_values':    [[round(value, 2) for value in weeks] for weeks in weekly_values],
        'simulation_rosters': weekly_rosters.tolist(),
        'simulation_totals':  weekly_totals.round(4).tolist(),
    }), encoding='utf-8')
    print(f'Wrote {weekly_path.relative_to(_VISUALIZATIONS_DIR.parent)}')

    # ── One fixed matchup, replayed: the week-to-week term on its own ────────────────
    matchup_rosters, matchup_totals = simulate_one_matchup_repeatedly(
        weekly_values, SIMULATION_COUNT, TEAM_SIZE, RANDOM_SEED)
    matchup_differentials = matchup_totals[:, 0] - matchup_totals[:, 1]
    print(f'One fixed matchup replayed: mean edge {matchup_differentials.mean():+.1f}, '
          f'sd {matchup_differentials.std():.1f}')

    # The claim the combined scene is built on, checked here rather than asserted on screen:
    # cross-player and week-to-week variation are independent, so their variances add.
    quadrature = np.hypot(differentials.std(), matchup_differentials.std())
    print(f'  quadrature check: sqrt({differentials.std():.1f}^2 + '
          f'{matchup_differentials.std():.1f}^2) = {quadrature:.1f} '
          f'against a measured {weekly_differentials.std():.1f}')

    matchup_path = _VISUALIZATIONS_DIR / 'data' / f'matchup_{SEASON.replace("-", "_")}.json'
    matchup_path.write_text(json.dumps({
        'season':           SEASON,
        'statistic':        STATISTIC,
        'value_key':        WEEKLY_AVERAGE_KEY,
        'team_size':        TEAM_SIZE,
        'random_seed':      RANDOM_SEED,
        'pool':             pool,
        'simulation_rosters': matchup_rosters.tolist(),
        'simulation_totals':  matchup_totals.round(4).tolist(),
    }), encoding='utf-8')
    print(f'Wrote {matchup_path.relative_to(_VISUALIZATIONS_DIR.parent)}')

    headshot_directory = _VISUALIZATIONS_DIR / 'assets' / 'headshots'
    missing = []
    for player in pool:
        source = _HEADSHOT_SOURCE_DIR / f'{player["player_id"]}.png'
        if not source.exists():
            missing.append(player['name'])
            continue
        write_circular_headshot(source, headshot_directory / source.name, HEADSHOT_PIXELS)
    print(f'Wrote {len(pool) - len(missing)} circular headshots to '
          f'{headshot_directory.relative_to(_VISUALIZATIONS_DIR.parent)}')
    if missing:
        # Named rather than counted: a scene cannot draw a face it does not have, and which
        # player is missing decides whether that matters.
        raise SystemExit(f'No cached headshot for: {", ".join(missing)}. Start the app and '
                         f'load this season once to populate .cache/headshots, then re-run.')


if __name__ == '__main__':
    main()
