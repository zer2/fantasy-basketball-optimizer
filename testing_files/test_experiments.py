# testing_files/test_experiments.py
# EXPERIMENTS — the fourth tier of the test taxonomy:
#   1. backend tests   (deterministic, must be green)
#   2. frontend tests  (deterministic, must be green)
#   3. golden tests    (regenerated whenever the algorithm intentionally changes)
#   4. EXPERIMENTS     (this file: no exact right answer — measurements to LOOK AT so a human can judge
#      that everything still looks right. Assertions here are only loose catastrophic floors — e.g. "the
#      field collapsed to one punt" — not tight expectations.)
#
# TWO MODES:
#   default pytest run       — one season, quick catastrophic-floor smoke; prints measurements but does
#                              NOT touch the stored reports (a quick run can't clobber full data).
#   RUN_EXPERIMENTS=1      — the REPORT GENERATOR: every property across seasons 2020-21..2025-26
#                              (EXPERIMENT_SEASONS to override), full-draft simulations at every seat,
#                              and the per-section results + tabbed experiment_report.html are rewritten.
#
# Results persist per section in experiment_results/<slug>.json, so running a subset of tests with -k in
# report mode regenerates only those sections; the tabbed page is rebuilt from everything stored.
#
# beth = 0 in all SIMULATIONS (the season-sim convention: historical stats are objectively correct, so
# the Bayesian self-doubt would only dilute the measurement); fast display-property tests keep full app
# settings because they measure exactly what the user sees.
#
# All sessions are built through request parameters (opponent_model_confidence, ...) rather than
# environment pins, and every draft loop calls agent.reset_draft_state() so no state leaks across drafts.

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'season_simulation'))

from benchmark_helpers import client, _build_session_request
from experiment_report_page import render_report_page
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

_RUN_SIMS  = bool(os.environ.get('RUN_EXPERIMENTS'))
# 0 = every seat in the league (a full season's worth of draft positions).
_SIM_SEATS = int(os.environ.get('EXPERIMENT_SEATS', '0'))
_ALL_SEASONS = '2020-21,2021-22,2022-23,2023-24,2024-25,2025-26'
_SEASONS = (os.environ.get('EXPERIMENT_SEASONS') or (_ALL_SEASONS if _RUN_SIMS else '2024-25')).split(',')

_FORMATS = {
    'EC':   'Each Category',
    'MC':   'Most Categories',
    'Roto': 'Rotisserie',
}

_SHORT_CATEGORY = {
    'Field Goal %': 'FG%', 'Free Throw %': 'FT%', 'Threes': '3s', 'Points': 'PTS',
    'Rebounds': 'REB', 'Assists': 'AST', 'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TO',
}


# ── measurement report ────────────────────────────────────────────────────────
# The experiment tier is read by a human: every test records structured table rows here; the conftest
# terminal-summary hook renders the tables (console + markdown + one tabbed HTML page).

_REPORT: dict = {}

# What each section MEANS — rendered under its header so the report stands alone. Keys must match the
# titles passed to _record_row().
_SECTION_EXPLANATIONS = {
    'Punt diversity (top-12 anchors)':
        "For each of a season's top 12 players: the build the self-play field predicts a team drafted around "
        'them commits to, named by its punted categories (expected win rate under 40%; ! marks a hard punt, '
        'under 15%). The breakdown counts how many of the 12 share each combination.\n'
        'Healthy: several distinct builds (floor 3). One build shared by everyone would mean the opponent '
        'model had collapsed and was steering every drafter into the same team.',
    'Rotisserie punting (hard <15% / soft 15-40% expected win rate)':
        'Rotisserie scores every standings point, so abandoning a category is mostly irrational there '
        '(head-to-head formats punt hard on purpose). For the top 12 Rotisserie builds of each season:\n'
        '- Hard: categories expected to win under 15% of the time. Want close to 0; the ceiling is 2, since '
        "an extreme star (Giannis's FT%) can rationally give one up.\n"
        '- Soft: 15-40%, a mild lean. A loose ceiling only catches drift: 2 per build in drafts, 5 in '
        'auctions, whose star-plus-cheap-fill teams spread win rates wider.\n'
        '- Weakest: the lowest category expectation among the 12 (amber when it is a hard punt).',
    'Predicted-pick stability (EC; opponent takes its predicted player)':
        'The app evaluates the board, an opponent takes exactly the player the model predicted for them, and '
        'the app evaluates again. That pick was already priced in, so the candidates in both top 15s should '
        'barely move.\n'
        '- Shared shift: the median H-score change common to every compared candidate, the whole board moving as '
        'one. About 0 in drafts and about +0.6 in auctions, where the market reprices once the player is bought. '
        'Harmless (ceiling 5).\n'
        '- Max relative move: the largest single H-score move after removing the shared shift. H-scores are '
        'expected win rates, so both are in percentage points.\n'
        '- Max dweight: the largest change in a displayed build weight (100 = neutral). Limit 5.\n'
        '- Punt flips: builds whose shown strategy changes: a category crosses the 40% punt line and its win rate '
        'moves at least 10 points. Want 0.',
    'Self-play convergence (MC bootstrap)':
        'Before the draft, the app builds its opponent field by self-play: passes that alternately predict '
        "the field and best-respond to it. A cell counts how many of the season's top 12 players punt that "
        'category (expected win rate under 40%). Convergence means the counts settle from cycle to cycle; '
        'oscillation means they keep swinging. This is a damped mixed equilibrium, so small residual motion '
        'is expected.\n'
        'Row 0 is the first pass against a neutral field; serve is the final full-pool pass, the field the '
        'app actually uses.',
    'H-scoring vs a G-score field':
        'Full 12-team snake drafts: one seat drafts by H-score, the other 11 take the best G-score available. '
        "The number is the H seat's expected head-to-head category win rate against that field (50% = "
        'break-even). S1-S12 are draft positions; later ones are genuinely harder. beth = 0, per the '
        'season-sim convention.\n'
        'Floors: above 50.5% in every season and format, above 52% pooled. This is the most basic claim of '
        'the whole system.',
    'Awareness (opponent model on vs off)':
        'The same seat drafts the same draft twice, with the opponent model on and off; the number is on '
        'minus off, in expected category win rate. Every seat drafts with its own agent.\n'
        '- vs an unaware H field: should be positive or about zero. Clearly negative would mean the model '
        'hurts.\n'
        "- vs a G field: the model's predictions are simply wrong there (G drafters never punt), so the "
        'property is harmlessness, about 0.\n'
        'The ALL rows pool every season and seat, with a standard error; a pooled mean is coloured only when it is '
        'more than 2 standard errors from 0.',
    'Warm start (purpose: display stability, not better answers)':
        "The app's one remaining warm start: the drafter's first pick, on an empty roster, continues from the "
        'builds self-play served (every later evaluate multi-starts from the seed scan). After one opponent '
        'pick, the drafter evaluates twice with equal budgets, warm from the served builds and cold from the '
        'seed scan, and this compares the mean H-score of the top 12.\n'
        "The point is display stability (the first pick's strategies match the rankings already shown), not "
        'better answers, so the expectation is parity. Floor: warm no more than 0.25% below cold.',
    'Multi-start seeding':
        "The app seeds every descent from a scan of gentle punts (one per category, keeping each player's "
        "best) rather than from one neutral build. This solves the same players' builds both ways, against "
        "the same field (Each Category, opponent model off), and compares each of the scan's top 12 with "
        'themselves.\n'
        'A positive gain means the scan found a better build for that player; a neutral-seeded descent tends '
        'to stall near the neutral build. Improved = players gaining more than 0.05%. Floor: mean gain at '
        'least −0.1%.',
    'Punt starts reaching the optimum':
        'Is the objective single-peaked? If it had several peaks, an ascent could stop on the wrong one. This '
        'starts ascent from every single and double punt (45 starts per candidate), with the punted weights '
        'at 5%, 1% and exactly 0% of neutral: far from the optimum, from every direction. A miss is a start '
        'ending more than 0.001% below the best peak.\n'
        "Zero misses means every start climbed to the same peak. The candidates are the loss section's, whose "
        'branch and bound confirms that peak is the global one. No regularization (lambda_c = lambda_p = 0).',
    'Loop gain at the optimum':
        "H-scoring's build is self-referential: the weights decide which players the team expects to pick "
        'later, those decide where the team is strong or weak, and that decides how much each category is '
        'worth, which is what the weights should be. At the peak the loop closes (weight vs value share gap '
        '~0).\n'
        'The loop gain is what happens to a small nudge to the weights after one trip around that loop. Below '
        '1 it comes back smaller, so the build returns to the peak: a strict local maximum. At 1 or above it '
        'would run away. The largest Hessian eigenvalue checks the same thing directly (negative at a peak).\n'
        'Same candidates as the loss section. No regularization (lambda_c = lambda_p = 0).',
    'Quasiconcavity along weight segments':
        'A stronger property than one peak: a quasiconcave objective never dips below the lower of its two '
        'ends along any straight segment of weight space, so every ascent finds the peak by construction. '
        'Random segments are drawn three ways: anywhere, between single or double punts (5-100% of neutral), '
        'and within ±30% of the optimum. Dipping is the share of segments that dip.\n'
        'A surface can be single-peaked without being quasiconcave, so dips far from the optimum do not '
        'contradict the punt-start section; dips near the optimum would. Same candidates as the loss section. '
        'No regularization (lambda_c = lambda_p = 0), and a convex regularizer could not create a dip anyway.',
    'Where the app loses to the optimum':
        'The app takes the most promising of its 12 seeds and runs gradient descent on it for 30 iterations. It '
        'could be improved by running gradient descent on all 12 seeds, using more seeds (58), using more '
        'iterations (2,000), or using the exact branch-and-bound algorithm. This breaks down how much not doing '
        'each of those costs: App + the four losses = Optimal. An H-score is an expected win rate, so all are in '
        "%. Each board's players are in their App order; an arrow by Optimal shows how many places one would move "
        "if ranked by optimal values (▲ higher). The cost-to-fix row gives each improvement's cost as a multiple "
        "of the app's work (each includes the ones before it; branch and bound's is its measured time per "
        'candidate).\n'
        'Boards after rounds 1-10 (default-ranking picks), seat Drafter 1, top 8 candidates, Each Category, '
        '2020-21 to 2025-26.\n'
        'Differences from the real app:\n'
        "- The volume scale and the position term are held at the app's solution; the app re-optimises both as it "
        'descends.\n'
        '- No regularization (lambda_c = lambda_p = 0).\n'
        "- The descent is a re-implementation of the app's (Adam, same learning rate and update) on this fixed "
        'problem.\n'
        'Full assumptions: experiment_harnesses/optimum_loss_decomposition.py.',
    'Planner vs H-scoring (paired 6v6 drafts)':
        "Would planning a concrete team beat H-scoring's smooth model of future picks? Pairs of 12-team snake "
        'drafts (2024-25, Each Category, beth = 0): six random seats draft with the planner and six by '
        "H-scoring, then the seats swap. A planner seat plans the rest of the team around each of H-scoring's "
        'top 40 candidates (an exact naive team, then Frank-Wolfe, then swaps) and takes the one whose '
        "planned team scores highest. The number is the planner seats' mean complete-team H-score minus the "
        "H-scoring seats'.\n"
        "- app's example team: the planner as the app ships it (held to H-scoring's slots, flex allocation and win "
        'rates; one open pick per round for availability).\n'
        "- unconstrained planner: the complete-team H-score alone, any eligible slot, and availability from the seat's "
        'actual remaining pick numbers.',
    'Planner vs H-scoring: individual drafts':
        'Every pair behind the planner tab. In a pair the planner and H-scoring seats swap, so every seat drafts once '
        "with each method. S1-S12 are the seats in draft order; each cell is that seat's complete-team H-score "
        'drafting with the planner minus its H-score drafting by H-scoring. The chart averages each seat over the '
        'pairs.',
    'Smooth vs simple scaling':
        'The problem. "Your $" is scaled so that all the players\' values add up to the money in the auction, as if '
        'every team will spend its last dollar on players worth exactly that much. Teams cannot pack their budgets '
        'that well, and late in an auction a team can hold more money than the players it could still draft are '
        'worth: with $40 left, two open slots and the best two available players worth $20 and $10 to it, $10 is '
        'unusable whatever it does.\n'
        "The simple fix is to scale up in those cases: once a team's cash exceeds what its best options are worth, add "
        'the surplus to their values so it bids the money rather than keeping it. By then it is often too late. A '
        'player goes for about what the second-highest bidder can pay, and late in an auction the other teams are '
        'nearly out of money, so a rich team wins its last players cheaply and still finishes with cash left over. '
        'Could the drafter be forewarned instead -- raising its values a little before the surplus arrives, while '
        'there is still competition to spend the money against?\n'
        "The two rules, with r₁ = the team's spendable cash (its cash minus $1 per empty bench slot) over what its best "
        'options (its top players, one per open slot) are worth:\n'
        '- Simple scaling: when r₁ is above 1, the best options are scaled up by r₁, so together they spend the cash '
        'exactly; nothing before that, and no one else.\n'
        '- Smooth, with softness s: r₂ = 1 + s ln(1 + e^((r₁ - 1) / s)), a smoothed max(1, r₁) -- about 1 well below '
        'r₁ = 1, about r₁ well above it, and rising from around r₁ = 1 - 3s. Every player the team can roster gets an '
        'extra (r₂ - 1) v² / v̄, where v is his value and v̄ the value-weighted average value of the best options, so '
        'the best options together are scaled by r₂, bigger players get a bigger share, and there is no cliff after '
        'the last best option.\n'
        "The simulation. Complete auctions of 12 teams, $200 each, 13 players a team, Each Category, on each season's "
        'actual statistics (2020-21 to 2025-26). Every team is an autodrafter: it nominates, bids $1 at a time up to '
        'its own valuation or its maximum allowed bid, and the highest bidder wins at about the next-highest\'s limit. '
        'Each team values players from its own noisy view of them (noise of SD 0.2 on each X-score for the stars, '
        'halving every 156 players), and each seed is a new draw of those views. Two settings differ from the app:\n'
        '- SAVOR is off (S_σ = 0 and S_R = 0). SAVOR reshapes dollar values for streaming, but finished teams are '
        'scored by H-score, which has no streaming, so a drafter relying on it would be judged on a value it never '
        'priced.\n'
        "- beth = 0. The season's actual statistics are the truth here, so the Bayesian self-doubt that shrinks "
        'projections would only blur the comparison (at the default it scores every finished team below 50% against a '
        'field of its peers).\n'
        'Each pair of auctions puts half the teams (alternate seats) on each rule, then swaps them, so each rule drafts '
        "from every seat: 40 pairs per season and softness, 240 per softness in all. The gap is the smooth teams' mean "
        "H-score minus the simple-scaling teams', averaged over the pair, with its standard error across pairs. Own "
        "view -- each drafter's H-score through its own expectation, which is what it maximises -- is the measure that "
        "matters; actual is on the true statistics. Spending by round shows where the smooth teams' money goes; paid "
        "extra is each sale's price above what the same bidding reaches without any cash share.",
    'Cash-share rules: builds (own view)':
        "Every team's build in the head-to-head auctions, as it sees it, by rule: the categories it punts (expected "
        "win rate under 40%; ! under 15%). Value in punted is the share of its players' positive X-score value lying "
        'in those categories; categories punted is what that share would be if players were bought regardless of the '
        'build. A misfit is a player with at least half his value in categories the team punts.',
    'Dollar shape: replacement level and SAVOR':
        'Vanilla SAVOR assumes a player realizes his value whenever he ends up worth a positive amount of money: his '
        'outcome is normal around his dollar value, with standard deviation S (the SAVOR noise), and every outcome '
        'above $0 counts. But a player who ends up worth a dollar might be used as a streaming slot instead of being '
        'kept on the team. So this version raises the cutoff to a replacement threshold: a player only counts outcomes '
        'above it, and is worth his expected value above the threshold minus a flyer\'s, a $0 player who is worth '
        'something only through his chance of clearing it. A threshold of $0 is vanilla SAVOR; with no noise, a player '
        'is worth his dollar value above the threshold.\n'
        'The grid crosses the threshold ($0, the app\'s $2, and $4) with the noise (S = 0, the app\'s 10, and 20), on '
        'the dollar values at the start of an auction (the neutral board; 12 teams x $200, Each Category). Each cell '
        'is the change '
        'in percent of the average player\'s dollars in the top 10%, the middle 80% and the bottom 10% of the drafted '
        'players (by value), against both off, which shows the dollars themselves. The charts show each rank\'s '
        'change.'
}


def _record_row(section, columns, row):
    """Add one structured row to a section's table (creating the table on first use). Every report
    renderer -- console, markdown, HTML tab -- draws real tables from these rows."""
    table = _REPORT.setdefault(section, {'columns': list(columns), 'rows': []})
    assert table['columns'] == list(columns), f'inconsistent columns within section {section!r}'
    table['rows'].append([str(cell) for cell in row])
    print(f'\n[{section}] ' + ' | '.join(f'{name}={cell}' for name, cell in zip(columns, row)))


def _wrap(text, width=96):
    words, lines, current = text.split(), [], ''
    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f'{current} {word}'.strip()
    if current:
        lines.append(current)
    return lines


def _console_table(columns, rows, indent='    '):
    widths = [max(len(str(columns[i])), *(len(row[i]) for row in rows)) if rows else len(str(columns[i]))
              for i in range(len(columns))]
    header = ' | '.join(str(columns[i]).ljust(widths[i]) for i in range(len(columns)))
    rule   = '-+-'.join('-' * widths[i] for i in range(len(columns)))
    body   = [' | '.join(row[i].ljust(widths[i]) for i in range(len(columns))) for row in rows]
    return [indent + header, indent + rule] + [indent + line for line in body]


def render_experiment_report(markdown=False):
    if not _REPORT:
        return ''
    lines = []
    for section, table in _REPORT.items():
        lines.append(f'## {section}' if markdown else section)
        explanation = _SECTION_EXPLANATIONS.get(section)
        if explanation:
            if markdown:
                lines.append(f'*{explanation}*')
                lines.append('')
            else:
                for paragraph in explanation.split('\n'):
                    lines.extend(f'  | {wrapped}' for wrapped in _wrap(paragraph))
        if markdown:
            lines.append('| ' + ' | '.join(table['columns']) + ' |')
            lines.append('|' + '|'.join('---' for _ in table['columns']) + '|')
            lines.extend('| ' + ' | '.join(row) + ' |' for row in table['rows'])
        else:
            lines.extend(_console_table(table['columns'], table['rows']))
        lines.append('')
    return '\n'.join(lines)


def _short_names(categories, indices):
    return '/'.join(_SHORT_CATEGORY.get(categories[i], categories[i]) for i in sorted(indices))


# ── persistent per-section results + one tabbed report page ──────────────────
# Written ONLY in report mode (RUN_EXPERIMENTS=1): each run persists the sections it measured to
# experiment_results/<slug>.json, then rebuilds ONE tabbed experiment_report.html from every stored section,
# fresh and stale alike, each tab labelled with its own generation time.

def _section_slug(title):
    import re
    return re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:60]


def _load_stored_sections(reports_dir):
    """All persisted section results, in the canonical _SECTION_EXPLANATIONS order."""
    import json
    import os as _os
    stored = {}
    if _os.path.isdir(reports_dir):
        for name in _os.listdir(reports_dir):
            if name.endswith('.json'):
                with open(_os.path.join(reports_dir, name), encoding='utf-8') as stored_file:
                    data = json.load(stored_file)
                stored[data['title']] = data
    ordered = [stored.pop(title) for title in list(_SECTION_EXPLANATIONS) if title in stored]
    return ordered + list(stored.values())


def write_experiment_report_files(base_directory):
    """Persist this run's sections, rebuild the tabbed page from everything stored. Returns the page path."""
    import json
    import os as _os
    from datetime import datetime

    reports_dir = _os.path.join(base_directory, 'experiment_results')
    _os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    # Archive: experiment_results/ keeps only the LATEST run of each section (a rerun
    # overwrites — measured cost: a partial rerun clobbered a full run's rows). Every
    # report-mode run therefore also lands in its own timestamped folder under
    # experiment_archive/ (gitignored, local-only), so no measurement is ever lost.
    archive_dir = _os.path.join(base_directory, 'experiment_archive',
                                datetime.now().strftime('%Y-%m-%dT%H%M%S'))
    _os.makedirs(archive_dir, exist_ok=True)

    for section, table in _REPORT.items():
        payload = {
            'title':       section,
            'explanation': _SECTION_EXPLANATIONS.get(section, ''),
            'columns':     table['columns'],
            'rows':        table['rows'],
            'generated':   stamp,
        }
        with open(_os.path.join(archive_dir, f'{_section_slug(section)}.json'), 'w',
                  encoding='utf-8') as archive_file:
            json.dump(payload, archive_file, indent=2)
        with open(_os.path.join(reports_dir, f'{_section_slug(section)}.json'), 'w',
                  encoding='utf-8') as stored_file:
            json.dump(payload, stored_file, indent=1)

    sections = _load_stored_sections(reports_dir)
    # Explanations belong to the code, not the run: a section measured before its explanation was reworded shows the
    # current wording.
    for data in sections:
        data['explanation'] = _SECTION_EXPLANATIONS.get(data['title'], data['explanation'])
    page = render_report_page(
        sections
        , 'Measurements for human judgment: this tier has no exact right answers. '
          'Each tab regenerates when its test runs in report mode (RUN_EXPERIMENTS=1) and keeps its own timestamp. '
          'Green = passes or no loss; red = a breached floor or a loss; amber = a warning short of a breach.'
    )
    page_path = _os.path.join(base_directory, 'experiment_report.html')
    with open(page_path, 'w', encoding='utf-8') as page_file:
        page_file.write(page)

    with open(_os.path.join(base_directory, 'experiment_report.md'), 'w', encoding='utf-8') as md_file:
        for data in sections:
            md_file.write(f'# {data["title"]}\n\n' + data['explanation'].replace('\n', '\n\n') + '\n\n')
            if 'columns' in data:
                md_file.write('| ' + ' | '.join(data['columns']) + ' |\n')
                md_file.write('|' + '|'.join('---' for _ in data['columns']) + '|\n')
                for row in data['rows']:
                    md_file.write('| ' + ' | '.join(row) + ' |\n')
            else:
                for entry in data.get('entries', []):
                    md_file.write(f'- {entry}\n')
            md_file.write(f'\n(generated {data["generated"]})\n\n')
    return page_path


# ── floors ────────────────────────────────────────────────────────────────────
# A breached floor is a finding, not a reason to stop measuring: every season is recorded, and the test fails at the end
# on all of its breaches together.

def _check_floor(breaches, holds, message):
    if not holds:
        breaches.append(message)


def _fail_on_breaches(breaches):
    assert not breaches, 'floors breached:\n  ' + '\n  '.join(breaches)


# ── session helpers ───────────────────────────────────────────────────────────

def _build_session(objective, auction=False, season=None, **parameter_overrides):
    request = _build_session_request(
        objective=objective, cash_per_team=200 if auction else None,
    )
    if season is not None:
        request['data_source']['season'] = season
    request['model_settings'].update(parameter_overrides)
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'session build failed: {response.text}'
    return get_session(response.json()['session_id'])


@pytest.fixture(scope='module')
def sessions():
    """One session per (format, mode, season) at app-true parameters, shared by the fast tests."""
    cache = {}

    def get(format_key, auction=False, season=None, **overrides):
        key = (format_key, auction, season, tuple(sorted(overrides.items())))
        if key not in cache:
            cache[key] = _build_session(_FORMATS[format_key], auction=auction, season=season, **overrides)
        return cache[key]

    return get


# Punt classification, in expected category WIN RATE (the percentages shown in the app):
#   hard punt: < 15% -- the build has abandoned the category
#   soft punt: 15-40% -- deliberately de-emphasised but not abandoned
_HARD_PUNT_RATE = 0.15
_SOFT_PUNT_RATE = 0.40


def _anchor_rate_rows(agent, count=12):
    """(player, per-category expected win rates 0-1) for the top `count` players by served H-score."""
    rates = agent._default_result['Rates']
    top = [p for p in agent.default_h_scores.index if p in rates.index][:count]
    return [(p, rates.loc[p].to_numpy()) for p in top]


def _classify_punts(rate_row):
    """frozenset of (category_index, is_hard) for every punted category of one build."""
    rates = np.asarray(rate_row, dtype=float)
    return frozenset((i, bool(rates[i] < _HARD_PUNT_RATE))
                     for i in np.flatnonzero(rates < _SOFT_PUNT_RATE))


def _punt_label(categories, punt_set):
    """Readable label: hard punts marked '!', e.g. '3s!/FT%/TO' (3s abandoned, FT%+TO soft)."""
    if not punt_set:
        return '(no punts)'
    parts = sorted((_SHORT_CATEGORY.get(categories[i], categories[i]) + ('!' if hard else ''))
                   for i, hard in punt_set)
    return '/'.join(parts)


# ── FAST: punt diversity (MC + EC, draft + auction, per season) ──────────────

@pytest.mark.parametrize('format_key', ['MC', 'EC'])
@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_punt_diversity(sessions, format_key, auction):
    """The predicted field must be DIVERSE: if the top of the pool all converge to one punt, the
    opponent model has collapsed (the original everyone-punts-3s/FT%/TO failure mode)."""
    mode = 'auction' if auction else 'draft'
    breaches = []
    for season in _SEASONS:
        session = sessions(format_key, auction, season)
        agent   = session.agent
        categories = session.current_settings['categories']
        punts    = [_classify_punts(rates) for _, rates in _anchor_rate_rows(agent)]
        distinct = len(set(punts))
        breakdown = ', '.join(f'{_punt_label(categories, punt_set)} x{count}'
                              for punt_set, count in Counter(punts).most_common())
        hard_total = sum(1 for punt_set in punts for _, hard in punt_set if hard)
        soft_total = sum(1 for punt_set in punts for _, hard in punt_set if not hard)
        _record_row('Punt diversity (top-12 anchors)',
                    ['Season', 'Format', 'Mode', 'Distinct builds', 'Hard', 'Soft',
                     'Breakdown (! = hard)'],
                    [season, format_key, mode, distinct, hard_total, soft_total, breakdown])
        _check_floor(breaches, distinct >= 3,
                     f'{format_key} {season}: only {distinct} distinct punt set(s) — field collapsed: {breakdown}')
    _fail_on_breaches(breaches)


# ── FAST: Rotisserie punts stay shallow (draft + auction, per season) ────────

@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_roto_minimal_punting(sessions, auction):
    """Rotisserie rewards every category point, so committed hard punts are (mostly) irrational; the
    self-play field should learn shallow leans, not the H2H-style full punts."""
    mode = 'auction' if auction else 'draft'
    breaches = []
    for season in _SEASONS:
        session = sessions('Roto', auction, season)
        agent   = session.agent
        categories = session.current_settings['categories']
        rows = _anchor_rate_rows(agent)
        hard_count = sum(1 for _, rates in rows for r in rates if r < _HARD_PUNT_RATE)
        soft_count = sum(1 for _, rates in rows for r in rates if _HARD_PUNT_RATE <= r < _SOFT_PUNT_RATE)
        weakest_player, weakest_rates = min(rows, key=lambda pr: float(np.min(pr[1])))
        weakest_idx = int(np.argmin(weakest_rates))
        weakest_cat = _SHORT_CATEGORY.get(categories[weakest_idx], categories[weakest_idx])
        _record_row('Rotisserie punting (hard <15% / soft 15-40% expected win rate)',
                    ['Season', 'Mode', 'Hard', 'Soft', 'Weakest category', 'Weakest rate', 'Player'],
                    [season, mode, hard_count, soft_count, weakest_cat,
                     f'{100 * float(np.min(weakest_rates)):.0f}%',
                     session.player_registry[weakest_player].name])
        # Hard punts are allowed but must stay RARE: an extreme-profile star can rationally
        # commit to one (Giannis's FT% after the team-denominator correction to percentage
        # G-scores — accepted 2026-08-14), but more than a couple among the top anchors
        # would mean full H2H-style punting has leaked into Rotisserie.
        hard_ceiling = 2
        _check_floor(breaches, hard_count <= hard_ceiling,
                     f'{season} {mode}: {hard_count} hard Roto punt(s) among the top anchors '
                     f'(> {hard_ceiling}) — hard punting should be a rare outlier, not common')
        # Soft counts measure roster SHAPE as much as strategy: an auction anchor's expected team is a
        # star plus budget-priced fill, which mechanically spreads category win rates wider than a
        # snake-draft expectation and parks more of them in the 15-40% band without any punt intent.
        # The near-zero hard floor above is the real Roto property; the soft ceilings only catch
        # egregious drift (draft measured <=21/season; auction measured ~47 on the widest season, 2020-21).
        soft_ceiling = (5 if auction else 2) * len(rows)
        _check_floor(breaches, soft_count <= soft_ceiling,
                     f'{season} {mode}: Roto soft-punting excessively ({soft_count})')
    _fail_on_breaches(breaches)


# ── FAST: early-pick stability (draft + auction, per season) ─────────────────

def _snapshot_candidates(result, top_n=15):
    return {
        c.player_id: (c.h_score, np.asarray(c.category_weights, dtype=float),
                      np.asarray(c.win_rates, dtype=float))
        for c in result.candidates[:top_n]
    }


@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_early_pick_stability(sessions, auction):
    """An opponent drafting its PREDICTED first player confirms the field rather than moving it, so the
    displayed values and builds of the remaining candidates must stay (nearly) put relative to each
    other — the anti-lurch guarantee."""
    mode = 'auction' if auction else 'draft'
    breaches = []
    for season in _SEASONS:
        session = sessions('EC', auction, season)
        agent   = session.agent
        teams   = session.current_settings['team_names']
        cash    = {t: 200.0 for t in teams} if auction else None

        agent.reset_draft_state()
        before = _snapshot_candidates(rank_candidates(session, {t: [] for t in teams}, teams[1], [], cash))

        predicted_first = agent._anchor_player_order[0]
        assignments = {t: [] for t in teams}
        assignments[teams[0]] = [predicted_first]
        if auction:
            cash = dict(cash)
            cash[teams[0]] = 130.0
        after = _snapshot_candidates(rank_candidates(session, assignments, teams[1], [], cash))

        common = [name for name in before if name in after]
        _check_floor(breaches, len(common) >= 10,
                     f'{season} {mode}: candidate overlap collapsed across one predicted pick ({len(common)})')
        if not common:
            continue
        raw_deltas  = {n: after[n][0] - before[n][0] for n in common}
        level_shift = float(np.median(list(raw_deltas.values())))
        h_deltas    = {n: abs(d - level_shift) for n, d in raw_deltas.items()}
        w_deltas    = {n: float(np.max(np.abs(after[n][1] - before[n][1]))) for n in common}
        worst_h_name = max(h_deltas, key=h_deltas.get)

        # PUNT-SET stability: H-scores staying put is not enough — a candidate's displayed STRATEGY
        # (which categories its build punts, win rate < 40) flipping across a predicted pick is a lurch
        # even on a value plateau. Threshold dust is not a flip: a category only counts when it crosses
        # 40% AND its win rate moves at least 10 points, i.e. the build genuinely entered or abandoned a punt.
        punt_flips = {}
        for name in common:
            before_rates, after_rates = before[name][2], after[name][2]
            crossed = [(i, before_rates[i], after_rates[i]) for i in range(len(before_rates))
                       if (before_rates[i] < 40.0) != (after_rates[i] < 40.0)
                       and abs(after_rates[i] - before_rates[i]) >= 10.0]
            if crossed:
                punt_flips[name] = crossed
        if punt_flips:
            flip_id, flip_cats = max(punt_flips.items(), key=lambda kv: max(abs(a - b) for _, b, a in kv[1]))
            categories = session.current_settings['categories']
            worst_flip = ', '.join(f'{_SHORT_CATEGORY.get(categories[i], categories[i])} {b:.0f}->{a:.0f}'
                                   for i, b, a in flip_cats)
            flip_text = f'{session.player_registry[flip_id].name}: {worst_flip}'
        else:
            flip_text = '(none)'
        _record_row('Predicted-pick stability (EC; opponent takes its predicted player)',
                    ['Season', 'Mode', 'Shared shift (H-score)', 'Max relative move (H-score)',
                     'Biggest mover', 'Max dweight', 'Punt flips', 'Worst punt flip', 'Players compared',
                     'Every relative move (H-score)'],
                    [season, mode, f'{level_shift:+.2f}', f'{max(h_deltas.values()):.2f}',
                     session.player_registry[worst_h_name].name, f'{max(w_deltas.values()):.1f}',
                     len(punt_flips), flip_text, len(common),
                     ' '.join(f'{delta - level_shift:+.2f}' for delta in raw_deltas.values())])
        _check_floor(breaches, abs(level_shift) <= 5.0,
                     f'{season} {mode}: implausible global re-level ({level_shift:+.2f})')
        _check_floor(breaches, max(w_deltas.values()) <= 5.0,
                     f'{season} {mode}: build lurch {max(w_deltas.values()):.1f}')
        _check_floor(breaches, not punt_flips,
                     f'{season} {mode}: punt builds flipped across a predicted pick: {flip_text}')
    _fail_on_breaches(breaches)


# ── FAST: self-play convergence (per season) ─────────────────────────────────

def _record_self_play_passes(agent, n_iterations):
    """Run the agent's populate, recording every self-play pass as (is_full_pool, rates)."""
    records  = []
    original = agent._run_bootstrap_pass

    def recording_pass(empty, n_iterations, cash, candidate_subset=None,
                       preserve_frozen_weights=False, _original=original, _records=records):
        result = _original(empty, n_iterations, cash, candidate_subset, preserve_frozen_weights)
        _records.append((candidate_subset is None, result['Rates'].copy()))
        return result

    agent._run_bootstrap_pass = recording_pass
    try:
        agent.populate_default_h_scores(n_iterations)
    finally:
        del agent._run_bootstrap_pass   # drop the instance shadow, restoring the class method
    return records


def test_self_play_convergence(sessions):
    """Show the punt structure of every bootstrap pass directly: per pass, how many of the tracked top
    players punt each category. Convergence = the counts stabilise from pass to pass; oscillation =
    counts swinging back and forth. No abstract metrics -- the evolution itself is the report.
    A second run doubles the number of passes, to show how the process would continue past the app's stopping point."""
    from backend.math import algorithm_agents
    app_passes = algorithm_agents._OPPONENT_BOOTSTRAP_PASSES
    breaches = []
    for season in _SEASONS:
        session = sessions('MC', False, season)
        n_iterations = session.current_settings['n_iterations']
        records = _record_self_play_passes(session.agent, n_iterations)
        # The extended run: a fresh session (so the shared one keeps the app's field), populated again with twice the
        # passes. The pass splits use a fixed seed, so its first app_passes passes repeat the app's.
        extended_session = _build_session(_FORMATS['MC'], season=season)
        algorithm_agents._OPPONENT_BOOTSTRAP_PASSES = 2 * app_passes
        try:
            extended_records = _record_self_play_passes(extended_session.agent, n_iterations)
        finally:
            algorithm_agents._OPPONENT_BOOTSTRAP_PASSES = app_passes

        categories = session.current_settings['categories']
        short      = [_SHORT_CATEGORY.get(category, category) for category in categories]
        # Track a FIXED set of players across passes and both runs so the counts are comparable: the top 12 of the
        # anchor subset (generic ranking), present in every pass.
        tracked = list(records[0][1].index[:12])

        for run, run_records in ((f'app ({app_passes} passes)', records),
                                 (f'doubled ({2 * app_passes} passes)', extended_records)):
            for pass_number, (is_full_pool, rates) in enumerate(run_records):
                # Solve-half passes only carry rows for the group they solved, so each pass
                # reports the punts of whichever tracked players it re-solved; the Level-0
                # pass and the serves cover the full tracked set.
                present = rates.index.intersection(tracked)
                if is_full_pool:
                    assert len(present) == len(tracked), (
                        f'{season} {run} pass {pass_number}: full-pool pass missing tracked players')
                tracked_rates = rates.loc[present].to_numpy()
                assert not np.isnan(tracked_rates).any(), f'{season} {run} pass {pass_number}: NaN win rates'
                punt_counts = (tracked_rates < _SOFT_PUNT_RATE).sum(axis=0)
                label = 'serve' if is_full_pool else str(pass_number)
                _record_row('Self-play convergence (MC bootstrap)',
                            ['Season', 'Run', 'Pass'] + short,
                            [season, run, label] + [int(count) for count in punt_counts])

        final_counts = (records[-1][1].reindex(tracked).to_numpy() < _SOFT_PUNT_RATE).sum(axis=0)
        _check_floor(breaches, final_counts.sum() > 0,
                     f'{season}: served field shows no punts at all — implausible')
    _fail_on_breaches(breaches)


# ── SLOW simulation properties ────────────────────────────────────────────────

sims = pytest.mark.skipif(not _RUN_SIMS, reason='full-draft simulation; set RUN_EXPERIMENTS=1')


def _draft_h_seat_in_g_field(h_session, seat, candidate_limit=40):
    """One snake draft: `seat` drafts with the H agent, everyone else picks by G-score ranking.
    Returns the H seat's final-roster H-score."""
    from simulate import _gscore_ranking, _pick_gscore_player, _has_position_data   # season_simulation

    agent        = h_session.agent
    agent.reset_draft_state()
    n_drafters   = h_session.current_settings['n_drafters']
    n_picks      = h_session.current_settings['n_picks']
    n_iterations = h_session.current_settings['n_iterations']
    teams        = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments  = {t: [] for t in teams}
    g_ranking    = _gscore_ranking(h_session)
    has_positions   = _has_position_data(h_session)
    position_config = agent.position_config
    positions_by_player = {player_id: identity.positions
                           for player_id, identity in h_session.player_registry.items()}
    drafted: set    = set()

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)
            team  = teams[index]
            if index == seat:
                result = rank_candidates(h_session, assignments, team, [], None, 0, candidate_limit)
                chosen = result.candidates[0].player_id
            else:
                chosen = _pick_gscore_player(
                    g_ranking, drafted, assignments[team], position_config, has_positions,
                    positions_by_player)
            assignments[team].append(chosen)
            drafted.add(chosen)

    scores = h_session.agent.get_h_scores(assignments, teams[seat], n_iterations)['Scores']
    return float(scores[scores.idxmax()])


def _seat_columns(n_seats):
    return [f'S{i + 1}' for i in range(n_seats)]


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC'])
def test_h_scoring_beats_g_field(format_key):
    """The most basic utility claim: an H-scoring drafter in a league of G-score drafters should win
    more than it loses, in every season."""
    pooled = []
    n_seats = None
    breaches = []
    for season in _SEASONS:
        # beth=0 and opponent_model_confidence=0: the H-vs-G conventions. Historical stats are
        # objectively correct (no shrinkage), and modeling G-drafters as punt-pursuers is a
        # knowably wrong belief — the canonical baseline (testing_files/baselines/
        # h_vs_g_c0_baseline.jsonl) is recorded at C=0, so this must match to stay comparable.
        session = _build_session(_FORMATS[format_key], season=season, beth=0,
                                 opponent_model_confidence=0)
        n_seats = _SIM_SEATS or session.current_settings['n_drafters']
        scores  = [100 * _draft_h_seat_in_g_field(session, seat) for seat in range(n_seats)]
        pooled.extend(scores)
        _record_row('H-scoring vs a G-score field',
                    ['Format', 'Season', 'Mean win rate'] + _seat_columns(n_seats),
                    [format_key, season, f'{np.mean(scores):.1f}'] + [f'{s:.1f}' for s in scores])
        _check_floor(breaches, float(np.mean(scores)) > 50.5,
                     f'{format_key} {season}: H barely beats G (mean {np.mean(scores):.1f})')
    _record_row('H-scoring vs a G-score field',
                ['Format', 'Season', 'Mean win rate'] + _seat_columns(n_seats),
                [format_key, f'ALL ({len(pooled)} drafts)', f'{np.mean(pooled):.1f}'] + ['—'] * n_seats)
    _check_floor(breaches, float(np.mean(pooled)) > 52.0,
                 f'{format_key}: pooled H-vs-G mean low ({np.mean(pooled):.1f})')
    _fail_on_breaches(breaches)


@sims
def test_awareness_vs_unaware_h_field():
    """Awareness against a field of UNAWARE H-drafters. ONE AGENT PER DRAFTER: a shared
    field agent cross-contaminates its seats (an inference from seat A's perspective can overwrite seat
    B's own team entry, so B would warm-start from A's model of B); every seat drafts with its own
    session."""
    from self_play import draft_population

    section = 'Awareness (opponent model on vs off)'
    all_gains = []
    n_seats = None
    for season in _SEASONS:
        reference = _build_session(_FORMATS['EC'], season=season, opponent_model_confidence=0, beth=0)
        n_drafters   = reference.current_settings['n_drafters']
        n_picks      = reference.current_settings['n_picks']
        n_iterations = reference.current_settings['n_iterations']
        n_seats      = _SIM_SEATS or n_drafters

        field_sessions = [reference] + [
            _build_session(_FORMATS['EC'], season=season, opponent_model_confidence=0, beth=0)
            for _ in range(n_drafters - 1)
        ]
        aware_deviator = _build_session(_FORMATS['EC'], season=season,
                                        opponent_model_confidence=0.5, beth=0)

        def deviator_score(session_by_seat, seat):
            assignments = draft_population(session_by_seat, n_drafters, n_picks, 40)
            # One fixed scorer for both arms: a complete roster's H is parameter-clean, so any agent
            # works, but using the same one removes even rounding asymmetry.
            scores = reference.agent.get_h_scores(assignments, f'Drafter {seat + 1}',
                                                  n_iterations)['Scores']
            return float(scores[scores.idxmax()])

        gains = []
        for seat in range(n_seats):
            unaware_field = {i: field_sessions[i] for i in range(n_drafters)}
            aware_arm     = dict(unaware_field)
            aware_arm[seat] = aware_deviator
            gains.append(100 * (deviator_score(aware_arm, seat) - deviator_score(unaware_field, seat)))
        all_gains.extend(gains)
        season_se = float(np.std(gains, ddof=1) / np.sqrt(len(gains)))
        _record_row(section,
                    ['Comparison', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                    ['vs unaware H field', season, f'{np.mean(gains):+.2f}', f'{season_se:.2f}']
                    + [f'{g:+.2f}' for g in gains])

    pooled_mean = float(np.mean(all_gains))
    pooled_se   = float(np.std(all_gains, ddof=1) / np.sqrt(len(all_gains)))
    _record_row(section,
                ['Comparison', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                ['vs unaware H field', f'ALL ({len(all_gains)} drafts)', f'{pooled_mean:+.2f}',
                 f'{pooled_se:.2f}'] + ['—'] * n_seats)
    assert pooled_mean > -1.0, f'awareness materially harmful vs an unaware H field: {pooled_mean:+.2f}pp'


@sims
def test_awareness_not_harmful_vs_g_field():
    """Awareness mispredicts a G field (G drafters do not punt) — that misprediction must be harmless."""
    section = 'Awareness (opponent model on vs off)'
    all_gains = []
    n_seats = None
    for season in _SEASONS:
        aware   = _build_session(_FORMATS['EC'], season=season,
                                 opponent_model_confidence=0.5, beth=0)
        unaware = _build_session(_FORMATS['EC'], season=season,
                                 opponent_model_confidence=0, beth=0)
        n_seats = _SIM_SEATS or aware.current_settings['n_drafters']
        gains = []
        for seat in range(n_seats):
            gains.append(100 * (_draft_h_seat_in_g_field(aware, seat)
                                - _draft_h_seat_in_g_field(unaware, seat)))
        all_gains.extend(gains)
        season_se = float(np.std(gains, ddof=1) / np.sqrt(len(gains)))
        _record_row(section,
                    ['Comparison', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                    ['vs G field', season, f'{np.mean(gains):+.2f}', f'{season_se:.2f}']
                    + [f'{g:+.2f}' for g in gains])

    pooled_mean = float(np.mean(all_gains))
    pooled_se   = float(np.std(all_gains, ddof=1) / np.sqrt(len(all_gains)))
    _record_row(section,
                ['Comparison', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                ['vs G field', f'ALL ({len(all_gains)} drafts)', f'{pooled_mean:+.2f}',
                 f'{pooled_se:.2f}'] + ['—'] * n_seats)
    assert pooled_mean > -1.0, f'awareness is harmful against a G field: {pooled_mean:+.2f}pp'


@sims
def test_warm_start_no_convergence_cost():
    """Warm starting is NOT primarily about finding a better answer -- its purpose is DISPLAY STABILITY:
    keeping each candidate's strategy consistent with the strategy previously shown to the user, instead
    of wandering between near-tied builds on every evaluate (the plateau-lurch problem). On value, the
    honest expectation is parity: this test only guards that the stability does not COST convergence."""
    breaches = []
    for season in _SEASONS:
        session = _build_session(_FORMATS['EC'], season=season)
        agent   = session.agent
        teams   = session.current_settings['team_names']
        assignments = {t: [] for t in teams}
        assignments[teams[0]] = [agent._anchor_player_order[0]]

        agent.reset_draft_state()
        warm = rank_candidates(session, assignments, teams[1], [], None, 0, 40)
        warm_mean = float(np.mean([c.h_score for c in warm.candidates[:12]]))

        # Cold arm: null the frozen tables so the drafter's candidate rows cold-start via the punt
        # seed scan — but do NOT reset the draft state. The team entries built by the warm arm's
        # refresh are FIELD state (a pure function of the rosters), and a paired warm-vs-cold
        # comparison needs the identical field; resetting would also force the refresh to rebuild the
        # opponent's entry via committed reuse, which reads the very frozen table nulled here.
        frozen_weights, frozen_shares = agent._player_frozen_weights, agent._player_frozen_shares
        try:
            agent._player_frozen_weights = None
            agent._player_frozen_shares  = None
            cold = rank_candidates(session, assignments, teams[1], [], None, 0, 40)
        finally:
            agent._player_frozen_weights = frozen_weights
            agent._player_frozen_shares  = frozen_shares
        cold_mean = float(np.mean([c.h_score for c in cold.candidates[:12]]))

        _record_row('Warm start (purpose: display stability, not better answers)',
                    ['Season', 'Warm mean H', 'Cold mean H', 'Warm - cold'],
                    [season, f'{warm_mean:.3f}', f'{cold_mean:.3f}', f'{warm_mean - cold_mean:+.3f}'])
        _check_floor(breaches, warm_mean >= cold_mean - 0.25,
                     f'{season}: warm starts converge worse than cold ({warm_mean:.3f} vs {cold_mean:.3f})')
    _fail_on_breaches(breaches)


@sims
def test_multi_start_seeding_helps():
    """An OPTIMIZER comparison (no competing agents — see the section explanation): solve the same
    players' builds twice, seeded by the punt scan vs a single neutral seed, and compare PAIRED
    per-player objective values over the scan's top 12."""
    breaches = []
    for season in _SEASONS:
        punt_scan = _build_session(_FORMATS['EC'], season=season, opponent_model_confidence=0)
        neutral   = _build_session(_FORMATS['EC'], season=season, opponent_model_confidence=0)
        neutral.agent.seed_mode = 'neutral'
        neutral.agent.reset_draft_state()
        neutral.agent.populate_default_h_scores(neutral.current_settings['n_iterations'])

        scan_scores    = punt_scan.agent.default_h_scores
        neutral_scores = neutral.agent.default_h_scores
        top_players    = [p for p in scan_scores.index[:12] if p in neutral_scores.index]
        # The neutral-seeded arm can COLLAPSE outright: a cold descent from neutral still carries the
        # regulariser, which can snap weights exactly onto the singular w=v ray (0/0 -> NaN objective).
        # The production punt scan never starts at neutral, so this cannot happen in the app -- when the
        # diagnostic arm collapses, that IS the finding (the strongest form of "multi-start helps").
        pairs      = [(p, 100 * float(scan_scores[p] - neutral_scores[p])) for p in top_players]
        collapsed  = sum(1 for _, delta in pairs if np.isnan(delta))
        deltas     = [delta for _, delta in pairs if not np.isnan(delta)]
        improved   = sum(1 for delta in deltas if delta > 0.05)
        valid_players = [p for p, delta in pairs if not np.isnan(delta)]
        mean_gain  = f'{np.mean(deltas):+.2f}' if deltas else 'n/a'
        max_gain   = f'{max(deltas):+.2f}' if deltas else 'n/a'
        neutral_mean = (f'{100 * float(neutral_scores[valid_players].mean()):.1f}'
                        if valid_players else 'NaN (collapsed)')
        _record_row('Multi-start seeding',
                    ['Season', 'Scan mean H', 'Neutral mean H', 'Mean gain (pp)', 'Max gain (pp)',
                     'Improved (of 12)', 'Neutral arm collapsed (NaN)'],
                    [season, f'{100 * float(scan_scores[top_players].mean()):.1f}', neutral_mean,
                     mean_gain, max_gain, improved, collapsed])
        if deltas:
            _check_floor(breaches, float(np.mean(deltas)) >= -0.1,
                         f'{season}: punt seed scan below a neutral seed (mean gain {np.mean(deltas):+.2f}pp)')
    _fail_on_breaches(breaches)


# ── Exact optimum (Each Category) ─────────────────────────────────────────────
# One measurement serves four sections (the loss decomposition below, punt starts, loop gain, quasiconcavity): for each season, the boards after each of _EXACT_ROUNDS snake rounds (picks in
# default-ranking order), my seat Drafter 1, the top EXPERIMENT_EXACT_CANDIDATES candidates of each board, at app
# settings with no regularization anywhere (lambda_c = lambda_p = 0, the self-play field included): the four sections
# study the unregularized objective, and the app is scored on that same problem. experiment_harnesses/exact_optimum measures each board in its own process (EXPERIMENT_WORKERS of them); the
# first of the four tests to run measures everything, and the rest read it. A board that fails is reported by every
# section (with its traceback) after the boards that worked are recorded; it is never measured twice.

_EXACT_ROUNDS = [int(n_rounds) for n_rounds in
                 os.environ.get('EXPERIMENT_EXACT_ROUNDS', '1,2,3,4,5,6,7,8,9,10').split(',')]
_EXACT_CANDIDATES = int(os.environ.get('EXPERIMENT_EXACT_CANDIDATES', '8'))
_EXACT_TOLERANCE = 1e-5
_EXACT_NODE_LIMIT = int(os.environ.get('EXPERIMENT_EXACT_NODE_LIMIT', '3000'))
_EXACT_LONG_ITERATIONS = 500
_EXPERIMENT_WORKERS = int(os.environ.get('EXPERIMENT_WORKERS', str(max(1, (os.cpu_count() or 2) - 2))))
_exact_measurement_run: dict = {}


def _measure_exact_optimum_cases():
    """Every measured case, measured once per run."""
    from experiment_harnesses.exact_optimum import measure_boards_in_parallel
    if not _exact_measurement_run:
        cases, failures = measure_boards_in_parallel(
            _SEASONS, _EXACT_ROUNDS, _EXACT_CANDIDATES, _EXACT_TOLERANCE, _EXACT_NODE_LIMIT, _EXACT_LONG_ITERATIONS,
            _EXPERIMENT_WORKERS)
        _exact_measurement_run.update(cases=cases, failures=failures)
        for failure in failures:
            print(f'\n[exact optimum] board {failure["season"]} after {failure["rounds"]} rounds FAILED:\n'
                  f'{failure["traceback"]}')
    return _exact_measurement_run['cases']


def _list_failed_exact_optimum_boards():
    return [f'{failure["season"]} after {failure["rounds"]} rounds: not measured, '
            f'{failure["traceback"].strip().splitlines()[-1]}' for failure in _exact_measurement_run['failures']]


@sims
def test_punt_starts_reach_the_optimum():
    """Does ascent from every single and double punt -- at 5%, 1% and exactly 0% of neutral -- climb to the peak?"""
    from experiment_harnesses.exact_optimum import PUNT_DEPTHS
    section = 'Punt starts reaching the optimum'
    columns = (['Season', 'Rounds', 'Player']
               + [f'{label} at {depth:.0%}' for depth in PUNT_DEPTHS for label in ('Missed', 'Worst miss (pp)')])
    cases = _measure_exact_optimum_cases()
    for case in cases:
        cells = []
        for depth in PUNT_DEPTHS:
            counts = case['punt_starts'][depth]
            cells += [f'{counts["missed"]}/{counts["starts"]}', f'{100 * counts["largest_shortfall"]:.4f}']
        _record_row(section, columns, [case['season'], case['rounds'], case['player']] + cells)
    totals = []
    for depth in PUNT_DEPTHS:
        missed = sum(case['punt_starts'][depth]['missed'] for case in cases)
        starts = sum(case['punt_starts'][depth]['starts'] for case in cases)
        worst = max(case['punt_starts'][depth]['largest_shortfall'] for case in cases)
        totals += [f'{missed}/{starts}', f'{100 * worst:.4f}']
    _record_row(section, columns, [f'ALL ({len(cases)} candidates)', '—', '—'] + totals)
    _fail_on_breaches(_list_failed_exact_optimum_boards())


@sims
def test_loop_gain_at_the_optimum():
    """At the peak: weights match marginal values, and every loop-gain eigenvalue is below 1 (a strict peak)."""
    section = 'Loop gain at the optimum'
    columns = ['Season', 'Rounds', 'Player', 'Largest loop gain', 'Weight vs value share gap',
               'Largest Hessian eigenvalue']
    cases = _measure_exact_optimum_cases()
    for case in cases:
        gain = case['loop_gain']
        _record_row(section, columns,
                    [case['season'], case['rounds'], case['player'], f'{gain["max_loop_gain"]:.3f}',
                     f'{gain["share_gap"]:.1e}', f'{gain["max_hessian_eigenvalue"]:.2e}'])
    _record_row(section, columns,
                [f'ALL ({len(cases)} candidates)', '—', '—',
                 f'{max(case["loop_gain"]["max_loop_gain"] for case in cases):.3f} max',
                 f'{max(case["loop_gain"]["share_gap"] for case in cases):.1e} max',
                 f'{max(case["loop_gain"]["max_hessian_eigenvalue"] for case in cases):.2e} max'])
    _fail_on_breaches(_list_failed_exact_optimum_boards()
                      + [f'{case["season"]} after {case["rounds"]} rounds, {case["player"]}: loop gain '
                         f'{case["loop_gain"]["max_loop_gain"]:.3f} >= 1, not a strict local maximum'
                         for case in cases if case['loop_gain']['max_loop_gain'] >= 1])


@sims
def test_quasiconcavity_along_weight_segments():
    """How often the objective dips along a straight segment of weight space, pooled over each board's candidates."""
    section = 'Quasiconcavity along weight segments'
    columns = ['Season', 'Rounds', 'Segments', 'Dipping', 'Median dip (pp)', 'Largest dip (pp)']
    measurements_by_board: dict = {}
    for case in _measure_exact_optimum_cases():
        for segments, measured in case['quasiconcavity'].items():
            measurements_by_board.setdefault((case['season'], case['rounds'], segments), []).append(measured)
    for (season, n_rounds, segments), measurements in measurements_by_board.items():
        _record_row(section, columns,
                    [season, n_rounds, segments,
                     f'{100 * np.mean([measured["share_dipping"] for measured in measurements]):.1f}%',
                     f'{100 * np.median([measured["median_dip"] for measured in measurements]):.4f}',
                     f'{100 * max(measured["largest_dip"] for measured in measurements):.4f}'])
    _fail_on_breaches(_list_failed_exact_optimum_boards())


# ── Where the app loses to the optimum ────────────────────────────────────────
# The exact-optimum boards and candidates again, each solved five ways on the same frozen problem
# (experiment_harnesses/optimum_loss_decomposition, whose header lists the assumptions).

# Read left to right, a row adds up: App + the four losses = Optimal.
_LOSS_CAUSE_LABELS = ('Seed choice', 'Seed menu', 'Iterations', 'Remaining gap')
_LOSS_COLUMNS = ['Season', 'Rd', 'Player', 'App'] + list(_LOSS_CAUSE_LABELS) + ['Optimal', 'Total loss']
_SIGNIFICANT_LOSS = 1e-5        # branch and bound's tolerance: a smaller loss is no loss


def compute_improvement_costs(cases, branch_and_bound_seconds):
    """What each improvement costs, as a multiple of the app's own work per candidate. The first three are counted in
    objective-and-gradient evaluations (what the app's time scales with: it batches every candidate together), and each
    includes the ones before it: the app scores its M menu seeds and descends once for I iterations (M + I); descending
    every menu seed is M x I; adding E more seeds is (M + E) x I; continuing the best descent to LONG_ITERATIONS adds
    LONG_ITERATIONS - I. The last is
    exact branch and bound on its own: its median measured time per candidate over the app's measured time per
    candidate."""
    from experiment_harnesses.optimum_loss_decomposition import LONG_ITERATIONS
    menu = np.mean([case['menu_size'] for case in cases])
    extra = np.mean([case['extra_seed_count'] for case in cases])
    iterations = np.mean([case['n_iterations'] for case in cases])
    app_work = menu + iterations
    more_seeds = (menu + extra) * iterations
    exact = float(np.median(branch_and_bound_seconds)) / float(np.median([case['app_seconds_per_candidate']
                                                                           for case in cases]))
    return [menu * iterations / app_work, more_seeds / app_work,
            (more_seeds + LONG_ITERATIONS - iterations) / app_work, exact]


def format_cost(cost):
    """A cost multiple to two significant figures (a timing is not more precise than that): 58497 -> '58,000x'."""
    import math
    rounded = round(cost, -int(math.floor(math.log10(abs(cost)))) + 1) if cost > 0 else 0
    return f'{rounded:,.0f}x'


def format_loss(loss):
    """A loss to 4 places with '%', without the sign of a rounding remnant ('-0.0000%')."""
    text = f'{100 * loss:.4f}%'
    return '0.0000%' if text == '-0.0000%' else text


def build_loss_rows(cases, branch_and_bound_seconds, uncertified_count):
    """The loss section's rows: one per candidate (App, the four losses, Optimal, the total), each board's candidates
    in their App order, then the mean of each, then what fixing each costs. Optimal carries how many places the
    candidate would move on his board if ranked by optimal values instead (an up arrow for higher)."""
    rows = []
    boards: dict = {}
    for case in cases:
        boards.setdefault((case['season'], case['rounds']), []).append(case)
    n_moved = 0
    for board_cases in boards.values():
        app_order = sorted(board_cases, key=lambda case: -case['steps'][0])
        optimal_place = {id(case): place for place, case in enumerate(sorted(board_cases,
                                                                             key=lambda case: -case['steps'][-1]))}
        for app_place, case in enumerate(app_order):
            steps = case['steps']
            move = app_place - optimal_place[id(case)]
            n_moved += move != 0
            arrow = f' ▲{move}' if move > 0 else (f' ▼{-move}' if move < 0 else '')
            rows.append([case['season'], case['rounds'], case['player'], f'{100 * steps[0]:.4f}%']
                        + [format_loss(later - earlier) for earlier, later in zip(steps, steps[1:])]
                        + [f'{100 * steps[-1]:.4f}%{arrow}', format_loss(steps[-1] - steps[0])])
    losses = np.array([np.diff(case['steps']) for case in cases])                 # (cases, 4)
    rows.append([f'Mean of {len(cases)}' + (f' ({uncertified_count} optimum not certified)' if uncertified_count else ''),
                 '—', '—', '—'] + [format_loss(float(losses[:, index].mean())) for index in range(losses.shape[1])]
                + [f'{n_moved} of {len(cases)} move', format_loss(float(losses.sum(axis=1).mean()))])
    rows.append(["Cost to fix (x the app's work)", '—', '—', '—']
                + [format_cost(cost) for cost in compute_improvement_costs(cases, branch_and_bound_seconds)] + ['—'] * 2)
    return rows


@sims
def test_where_the_app_loses_to_the_optimum():
    """The app's shortfall from the certified optimum, split into seed choice, seed menu, iteration budget and the
    rest. Branch and bound certifies each optimum (the shared exact-optimum measurement)."""
    from experiment_harnesses.optimum_loss_decomposition import decompose_boards_in_parallel
    section = 'Where the app loses to the optimum'
    certifications = {(case['season'], case['rounds'], case['player']): case['certification']
                      for case in _measure_exact_optimum_cases()}
    cases = decompose_boards_in_parallel(_SEASONS, _EXACT_ROUNDS, _EXACT_CANDIDATES, _EXPERIMENT_WORKERS)
    uncertified = sum(certification['gap'] > _EXACT_TOLERANCE for certification in certifications.values())
    for row in build_loss_rows(cases, [certification['seconds'] for certification in certifications.values()],
                               uncertified):
        _record_row(section, _LOSS_COLUMNS, row)
    _fail_on_breaches(
        _list_failed_exact_optimum_boards()
        + [f'{key[0]} after {key[1]} rounds, {key[2]}: branch and bound beat gradient ascent by '
           f'{100 * certification["improvement"]:.3f}pp'
           for key, certification in certifications.items() if certification['improvement'] >= 1e-4]
        + [f'{case["season"]} after {case["rounds"]} rounds, {case["player"]}: the long budget beat the certified '
           f'optimum by {100 * (case["steps"][3] - case["steps"][4]):.4f}pp'
           for case in cases if case['steps'][3] - case['steps'][4] > _SIGNIFICANT_LOSS])


# ── The example team's planner as a drafter ───────────────────────────────────

_PLANNER_PAIRS = int(os.environ.get('EXPERIMENT_PLANNER_PAIRS', '24'))


@sims
def test_planner_versus_h_scoring_drafts():
    """Paired 6v6 drafts: the example team's planner choosing picks against H-scoring, anchored and unanchored."""
    from experiment_harnesses.frank_wolfe_drafts import (
        SEASON, VARIANTS, run_paired_drafts_in_parallel, summarize_pairs,
    )
    section = 'Planner vs H-scoring (paired 6v6 drafts)'
    columns = ['Planner', 'Season', 'Pairs', 'Planner minus H (pp)', 'SE (pp)', 'Planner wins', 'Worst (pp)',
               'Best (pp)']
    records_by_variant = {variant: run_paired_drafts_in_parallel(variant, _PLANNER_PAIRS, _EXPERIMENT_WORKERS, 1)
                          for variant in VARIANTS}
    for variant, records in records_by_variant.items():
        summary = summarize_pairs(records)
        _record_row(section, columns,
                    [variant, SEASON, summary['pairs'], f'{summary["mean"]:+.2f}', f'{summary["standard_error"]:.2f}',
                     f'{summary["planner_wins"]}/{summary["pairs"]}', f'{summary["worst"]:+.2f}',
                     f'{summary["best"]:+.2f}'])
    # Every draft on its own: each seat's complete-team H-score, marked P (planner) or H (H-scoring).
    n_seats = len(records_by_variant[VARIANTS[0]][0]['drafts'][0]['seat_scores'])
    draft_columns = ['Planner', 'Pair', 'Draft', 'Planner minus H (pp)'] + _seat_columns(n_seats)
    for variant, records in records_by_variant.items():
        for pair_number, record in enumerate(records, start=1):
            for draft_number, draft in enumerate(record['drafts'], start=1):
                planner_seats = set(draft['planner_seats'])
                _record_row('Planner vs H-scoring: individual drafts', draft_columns,
                            [variant, pair_number, draft_number,
                             f'{draft["planner_mean"] - draft["h_scoring_mean"]:+.2f}']
                            + [f'{"P" if seat in planner_seats else "H"} {score:.2f}'
                               for seat, score in enumerate(draft['seat_scores'])])


# ── Auction cash-share rules ──────────────────────────────────────────────────

RULES_SECTION = 'Smooth vs simple scaling'
RULE_BUILDS_SECTION = 'Cash-share rules: builds (own view)'

# The smooth rule's softnesses measured against simple scaling (auction_pricing.smooth_spending_scale).
_CASH_SHARE_SOFTNESSES = (0.1, 0.2, 0.3)
# Noise seeds per season and softness: each seed is a pair of auctions (the two seatings), every team drawing its own
# view of the players from it (auction_autodraft.read_team_x_scores).
_AUCTION_NOISE_SEEDS = int(os.environ.get('EXPERIMENT_AUCTION_SEEDS', '40'))
_AUCTION_ROUNDS = 13
_RULE_LABELS = {'smooth': 'smooth', 'simple_scaling': 'simple scaling'}


def _summarize_rule_pairs(records):
    """(pairs, own-view gaps, actual gaps) over the complete pairs: per (season, seed) the smooth teams' mean H-score
    minus the simple-scaling teams', averaged over the two seatings."""
    by_pair = {}
    for record in records:
        teams = list(record['teams'].values())
        gaps = [float(np.mean([team[measure] for team in teams if team['rule'] == 'smooth'])
                      - np.mean([team[measure] for team in teams if team['rule'] == 'simple_scaling']))
                for measure in ('own_view', 'actual')]
        by_pair.setdefault((record['season'], record['noise_seed']), {})[record['smooth_parity']] = gaps
    complete = [seatings for seatings in by_pair.values() if len(seatings) == 2]
    own_gaps = np.array([np.mean([seatings[parity][0] for parity in (0, 1)]) for seatings in complete])
    actual_gaps = np.array([np.mean([seatings[parity][1] for parity in (0, 1)]) for seatings in complete])
    return len(complete), own_gaps, actual_gaps


def _format_gap(gaps):
    """The pair gaps' mean and standard error, as text."""
    if len(gaps) < 2:
        raise ValueError(f'A standard error needs at least two pairs; there are {len(gaps)}.')
    return f'{gaps.mean():+.3f}', f'{gaps.std(ddof=1) / np.sqrt(len(gaps)):.3f}'


def _record_rule_rows(softness_label, season_label, records, columns):
    """One row of the rules section: the head-to-head gaps, cash left and paid extra per drafter by rule, and each
    rule's mean spend per drafter after every round."""
    pairs, own_gaps, actual_gaps = _summarize_rule_pairs(records)
    teams_by_rule = {rule: [team for record in records for team in record['teams'].values() if team['rule'] == rule]
                     for rule in _RULE_LABELS}
    means = [f'{np.mean([team[measure] for team in teams_by_rule[rule]]):.2f}'
             for measure in ('cash_left', 'paid_extra') for rule in _RULE_LABELS]
    spending = [f'{np.mean([team["spent_after_round"][round_index] for team in teams_by_rule[rule]]):.1f}'
                for round_index in range(_AUCTION_ROUNDS) for rule in _RULE_LABELS]
    _record_row(RULES_SECTION, columns,
                [softness_label, season_label, pairs, *_format_gap(own_gaps), *_format_gap(actual_gaps)]
                + means + spending)


@sims
def test_cash_share_rules_head_to_head():
    """The smooth cash-share rule against simple scaling, at each softness: per season and noise seed, a pair of
    auctions (smooth on even seats, then on odd), every team's own-view and actual H-score, its cash left, what it paid
    extra and what it had spent by each round; and every team's build, by rule. Measurements only, apart from every
    auction completing."""
    from experiment_harnesses.auction_simulation import run_rule_matches_in_parallel
    # Every auction is kept here as it finishes, so a stopped run resumes rather than restarting (about three hours on
    # 18 workers for the full set).
    record_path = Path(__file__).resolve().parent / 'experiment_results' / 'cash_share_rule_matches.jsonl'
    records, failures = run_rule_matches_in_parallel(_SEASONS, list(_CASH_SHARE_SOFTNESSES),
                                                     list(range(1, _AUCTION_NOISE_SEEDS + 1)), _SHORT_CATEGORY,
                                                     _EXPERIMENT_WORKERS, record_path)
    columns = (['Softness', 'Season', 'Pairs', 'Own view gap', 'Own view SE', 'Actual gap', 'Actual SE']
               + [f'{measure} ({label})' for measure in ('Unspent', 'Paid extra') for label in _RULE_LABELS.values()]
               + [f'Spent after round {round_number} ({label})' for round_number in range(1, _AUCTION_ROUNDS + 1)
                  for label in _RULE_LABELS.values()])
    build_columns = ['Softness', 'Rule', 'Teams', 'Value in punted', 'Categories punted', 'Misfits per team',
                     'Most common punts']
    for softness in _CASH_SHARE_SOFTNESSES:
        softness_records = [record for record in records if record['smooth_rule_softness'] == softness]
        for season in _SEASONS:
            _record_rule_rows(f'{softness:.2f}', season,
                              [record for record in softness_records if record['season'] == season], columns)
        _record_rule_rows(f'{softness:.2f}', f'ALL ({len(_SEASONS)} seasons)', softness_records, columns)
        for rule, rule_label in _RULE_LABELS.items():
            teams = [team for record in softness_records for team in record['teams'].values() if team['rule'] == rule]
            punt_counts = Counter('/'.join(team['punted']) or '(none)' for team in teams)
            _record_row(RULE_BUILDS_SECTION, build_columns,
                        [f'{softness:.2f}', rule_label, len(teams),
                         f'{100 * np.mean([team["punted_value_share"] for team in teams]):.0f}%',
                         f'{100 * np.mean([team["punted_category_share"] for team in teams]):.0f}%',
                         f'{np.mean([len(team["misfits"]) for team in teams]):.2f}',
                         ', '.join(f'{build} ({count})' for build, count in punt_counts.most_common(3))])
    for failure in failures:
        print(f'auction {failure["job"]} failed:\n{failure["traceback"]}')
    assert not failures, f'{len(failures)} of the auctions failed (tracebacks above)'


# ── Auction dollar values: replacement level and SAVOR ────────────────────────

DOLLAR_SHAPE_SECTION = 'Dollar shape: replacement level and SAVOR'

# (label, value) for each axis of the grid: SAVOR's replacement threshold in dollars (algorithm_helpers.
# savor_calculation; 2 is the app's default), and its streaming noise S (0 is off; 10 is the app's default).
_REPLACEMENT_LEVELS = (('off', 0.0), ('medium', 2.0), ('high', 4.0))
_SAVOR_NOISES = (('off', 0.0), ('medium', 10.0), ('high', 20.0))
# The drafted players' tiers, best first: (name, share of the drafted players).
_DOLLAR_TIERS = (('Top 10%', 0.10), ('Middle 80%', 0.80), ('Bottom 10%', 0.10))


def _split_into_dollar_tiers(dollars, total_picks):
    """The drafted players' dollars (best first): the average player's within each tier."""
    top_count = round(total_picks * _DOLLAR_TIERS[0][1])
    bottom_count = round(total_picks * _DOLLAR_TIERS[2][1])
    drafted = dollars.iloc[:total_picks].to_numpy()
    return [float(drafted[:top_count].mean()), float(drafted[top_count:total_picks - bottom_count].mean()),
            float(drafted[total_picks - bottom_count:].mean())]


def test_dollar_shape_by_replacement_level_and_savor(sessions):
    """The start of an auction's dollar values (the neutral board, all the league's cash) under each replacement level
    and SAVOR noise: the average player's dollars in the top 10%, the middle 80% and the bottom 10% of the drafted
    players, and their change in percent against both off.
    Measurements only."""
    from backend.math.algorithm_helpers import auction_value_adjuster
    tier_names = [name for name, _ in _DOLLAR_TIERS]
    columns = None
    tiers_by_cell = {}
    for season in _SEASONS:
        session = sessions('EC', auction=True, season=season)
        agent = session.agent
        total_picks = agent.n_drafters * agent.n_picks
        total_cash = agent.n_drafters * session.current_settings['cash_per_team']
        if columns is None:
            columns = (['Season', 'Replacement level', 'Replacement threshold', 'SAVOR', 'SAVOR noise']
                       + [f'{name} $ per player' for name in tier_names] + [f'{name} change' for name in tier_names]
                       + ['Players over $1'] + [f'Rank {rank} $' for rank in range(1, total_picks + 1)])
        for replacement_label, replacement_threshold in _REPLACEMENT_LEVELS:
            for savor_label, savor_noise in _SAVOR_NOISES:
                dollars = auction_value_adjuster(agent.default_h_scores, total_picks, total_cash, savor_noise,
                                                 replacement_threshold).sort_values(ascending=False)
                tiers = _split_into_dollar_tiers(dollars, total_picks)
                tiers_by_cell.setdefault((replacement_label, savor_label), []).append(tiers)
                baseline = tiers_by_cell[('off', 'off')][-1]
                _record_row(DOLLAR_SHAPE_SECTION, columns,
                            [season, replacement_label, replacement_threshold, savor_label, savor_noise]
                            + [f'{value:.2f}' for value in tiers]
                            + [f'{100 * (value / base - 1):+.1f}%' for value, base in zip(tiers, baseline)]
                            + [int((dollars.iloc[:total_picks] > 1).sum())]
                            + [f'{value:.2f}' for value in dollars.iloc[:total_picks]])
    curve_start = columns.index('Rank 1 $')
    baseline = np.mean(tiers_by_cell[('off', 'off')], axis=0)
    for replacement_label, replacement_threshold in _REPLACEMENT_LEVELS:
        for savor_label, savor_noise in _SAVOR_NOISES:
            rows = [row for row in _REPORT[DOLLAR_SHAPE_SECTION]['rows']
                    if row[1] == replacement_label and row[3] == savor_label]
            tiers = np.mean(tiers_by_cell[(replacement_label, savor_label)], axis=0)
            curve = [f'{np.mean([float(row[index]) for row in rows]):.2f}'
                     for index in range(curve_start, len(columns))]
            players_over_one = np.mean([float(row[columns.index('Players over $1')]) for row in rows])
            _record_row(DOLLAR_SHAPE_SECTION, columns,
                        [f'ALL ({len(_SEASONS)} seasons)', replacement_label, replacement_threshold, savor_label,
                         savor_noise] + [f'{value:.2f}' for value in tiers]
                        + [f'{100 * (value / base - 1):+.1f}%' for value, base in zip(tiers, baseline)]
                        + [f'{players_over_one:.0f}'] + curve)
