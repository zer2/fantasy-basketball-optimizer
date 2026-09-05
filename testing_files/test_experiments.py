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
        'For each of the top 12 players of a season, the build the self-play process predicts a team '
        'drafted around that player would commit to, expressed by its PUNTED categories (expected win '
        'rate under 40%). Hard punts (under 20%, marked with !) are abandoned categories; soft punts '
        '(20-40%) are de-emphasised. The breakdown shows how many of the 12 share each punt combination. '
        'Healthy: several distinct combinations per season; one combination shared by everyone means the '
        'opponent model has collapsed and the app steers every drafter into the same build.',
    'Rotisserie punting (hard <20% / soft 20-40% expected win rate)':
        'Rotisserie scores every standings point, so abandoning a category is (mostly) irrational there. '
        'Hard punt = a top-12 build expecting to win a category under 20% of the time (want ZERO); soft '
        'punt = 20-40% (a mild lean; a few are fine). Also shown: the single weakest category '
        'expectation across the 12 builds. H2H formats, by contrast, punt hard on purpose.',
    'Predicted-pick stability (EC; opponent takes its predicted player)':
        'Evaluate all candidates, let an opponent take exactly the player the model already predicted '
        'for them, evaluate again, and compare the players present in both top-15 lists ("Players '
        'compared"; the picked player drops out). "Shared shift" = the median H change common to ALL of '
        'them -- movement of the whole board as one block. In drafts it is ~0; in auctions it is '
        'legitimately a few points NEGATIVE, because the bought player leaves the purchasable pool and '
        'the whole market reprices. "Max relative move" = the biggest single-candidate movement AFTER '
        'subtracting that shared shift -- movement against the field, which is what lurch means; a '
        'predicted pick is already priced in, so want well under 1. A large shared shift with a tiny '
        'relative move is the GOOD outcome: everything moved together and no standing changed. '
        '"Max dweight" = biggest change in any displayed build weight (display units, 100 = neutral). '
        '"Punt flips" = candidates whose displayed STRATEGY changed: a category crossed the 40% punt '
        'line by more than 5pp between the two evaluates. H stability alone is not enough -- a build '
        'swapping punts on a value plateau is still a lurch -- so the expectation is zero.',
    'Self-play convergence (MC bootstrap)':
        'The pre-draft bootstrap alternates predict-field / best-respond for ~15 passes. Each row shows '
        'ONE PASS: for a fixed set of the top 12 players (tracked across all passes so counts are '
        'comparable), how many of them punt each category in that pass\'s solve (expected win rate '
        'under 40%). Read down a season: convergence = the counts settle from pass to pass; oscillation '
        '= counts swinging back and forth between passes. This is a damped mixed-equilibrium process, '
        'so small residual motion is expected — what matters is that the structure stops changing. The '
        '"serve" row is the final full-pool pass, i.e. the field the app actually uses.',
    'H-scoring vs a G-score field':
        'Full snake drafts: one seat drafts by H-score, the other 11 pick greedily by G-score ranking. '
        'Numbers are the H seat\'s expected head-to-head category win rate against that field in '
        'percentage points (50.0 = break-even; every point above is real edge). Seat columns are draft '
        'positions 1..12 — later positions are genuinely harder. beth=0 '
        'per the season-sim convention. This is the most basic utility claim of the whole system.',
    'Awareness (opponent model on vs off)':
        'Identical drafts by the same seat with the opponent model ON minus OFF; gains are in H-score '
        'percentage points (+1.0 = one extra point of expected category win rate). vs an UNAWARE H '
        'field: expect small positive-to-zero '
        'means — clearly negative would mean the model hurts against naive H-drafters; the ALL row '
        'pools every season and seat, with its standard error. vs a G field: the '
        'model\'s predictions are simply wrong there (G drafters never punt), so the property is '
        'harmlessness, mean ~0. Every seat drafts with its own independent agent.',
    'Warm start (purpose: display stability, not better answers)':
        'THE START-POLICY LADDER, tested here and in "Multi-start seeding": descents can start from '
        '(1) a single neutral seed, (2) the punt seed scan (cold multi-start), or (3) a warm start '
        'from a stored build. The two experiments compare ADJACENT rungs at the stage where each '
        'comparison is meaningful, so they deliberately have different baselines. THIS test compares '
        'rung 3 vs rung 2 at an IN-DRAFT evaluate — the only place stored builds exist to warm-start '
        'from. Same evaluate, equal iteration budget; numbers are mean H (percentage points) of the '
        'top 12 candidates. Warm starting exists to keep displayed strategies CONSISTENT with what '
        'the user was already shown — the expectation here is parity (stability must not cost '
        'convergence); the stability benefit itself is measured by predicted-pick stability. A '
        'three-way test at one stage is not possible: at populate nothing is stored yet (the scan IS '
        'the cold start there), and in-draft a scan-vs-neutral arm would only repeat what the '
        'multi-start section already isolates.',
    'Multi-start seeding':
        'The other half of the start-policy ladder (see "Warm start"): rung 2 vs rung 1, at POPULATE '
        'time, where warm starts cannot exist because nothing has been stored yet. An OPTIMIZER '
        'comparison, not a competitive simulation: the same players, the same modelled field, and the '
        'same objective are solved twice — once seeded by the punt scan (try one gentle punt per '
        'category, keep each candidate\'s best) and once from a single neutral seed. The H value is '
        'the objective the descent attained (that player\'s expected win rate vs the modelled generic '
        'field), so it is NOT a symmetric 50%-by-definition number — no agents compete here. '
        'Comparison is PAIRED per player over the top 12: a positive gain means the scan found a '
        'genuinely better build for that same player; neutral-seeded descents tend to get stuck near '
        'the neutral build, which is exactly what multi-start exists to fix.',
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
                lines.extend(f'  | {wrapped}' for wrapped in _wrap(explanation))
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

_REPORT_PAGE_CSS = """
  :root { --bg:#ffffff; --fg:#26282e; --muted:#6a6f7a; --line:#e4e6ea; --card:#f7f8fa; --accent:#33556e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161c; --fg:#e6e8ec; --muted:#9aa0ab; --line:#2c303a; --card:#1b1e26; --accent:#8fb4d1; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, "Segoe UI", sans-serif; }
  main { max-width: 1080px; margin: 0 auto; padding: 28px 20px 64px; }
  h1 { font-size: 1.35rem; margin: 0 0 2px; }
  .stamp { color: var(--muted); font-size: .85rem; margin-bottom: 18px; }
  nav { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:16px; }
  nav button { background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px;
               padding:6px 11px; font-size:.85rem; cursor:pointer; }
  nav button.active { border-color:var(--accent); color:var(--accent); font-weight:600; }
  section { display:none; }
  section.active { display:block; }
  h2 { font-size: 1.05rem; margin: 4px 0 2px; color: var(--accent); }
  .generated { color: var(--muted); font-size: .8rem; margin-bottom: 10px; }
  .explain { color: var(--muted); font-size: .9rem; margin: 0 0 12px; max-width: 80ch; }
  .tablewrap { background:var(--card); border:1px solid var(--line); border-radius:10px;
               padding: 4px 12px; overflow-x:auto; }
  table { border-collapse: collapse; width:100%; font-size:.88rem; }
  th { text-align:left; color:var(--muted); font-weight:600; padding:8px 14px 6px 4px;
       border-bottom:1px solid var(--line); white-space:nowrap; }
  td { padding:7px 14px 7px 4px; border-bottom:1px solid var(--line); white-space:nowrap;
       font-variant-numeric: tabular-nums; }
  tr:last-child td { border-bottom:none; }
"""


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
    from html import escape

    reports_dir = _os.path.join(base_directory, 'experiment_results')
    _os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    for section, table in _REPORT.items():
        payload = {
            'title':       section,
            'explanation': _SECTION_EXPLANATIONS.get(section, ''),
            'columns':     table['columns'],
            'rows':        table['rows'],
            'generated':   stamp,
        }
        with open(_os.path.join(reports_dir, f'{_section_slug(section)}.json'), 'w',
                  encoding='utf-8') as stored_file:
            json.dump(payload, stored_file, indent=1)

    sections = _load_stored_sections(reports_dir)
    tabs, panels = [], []
    for position, data in enumerate(sections):
        title = data['title']
        short = title.split(' (')[0]
        active = ' class="active"' if position == 0 else ''
        tabs.append(f'<button data-tab="{position}"{active}>{escape(short)}</button>')
        if 'columns' in data:
            head = ''.join(f'<th>{escape(name)}</th>' for name in data['columns'])
            body = '\n'.join(
                '<tr>' + ''.join(f'<td>{escape(cell)}</td>' for cell in row) + '</tr>'
                for row in data['rows']
            )
            measurements = f'<div class="tablewrap"><table><tr>{head}</tr>{body}</table></div>'
        else:   # legacy stored format (pre-table): plain lines
            entries = ''.join(f'<li>{escape(entry)}</li>' for entry in data.get('entries', []))
            measurements = f'<ul>{entries}</ul>'
        panels.append(
            f'<section id="tab-{position}"{active}>\n<h2>{escape(title)}</h2>\n'
            f'<div class="generated">generated {escape(data["generated"])}</div>\n'
            f'<p class="explain">{escape(data["explanation"])}</p>\n{measurements}\n</section>'
        )

    newline = '\n'
    page = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Experiment report</title>\n'
        f'<style>{_REPORT_PAGE_CSS}</style></head><body><main>\n'
        '<h1>Experiment report</h1>\n'
        '<div class="stamp">Seasons 2020-21 to 2025-26 &middot; measurements for human judgment — this '
        'tier has no exact right answers; each tab regenerates independently when its test runs in '
        'report mode (RUN_EXPERIMENTS=1) and keeps its own timestamp.</div>\n'
        f'<nav>{"".join(tabs)}</nav>\n'
        f'{newline.join(panels)}\n'
        '<script>\n'
        'document.querySelectorAll("nav button").forEach(function (button) {\n'
        '  button.addEventListener("click", function () {\n'
        '    document.querySelectorAll("nav button, section").forEach(function (node) {\n'
        '      node.classList.remove("active");\n'
        '    });\n'
        '    button.classList.add("active");\n'
        '    document.getElementById("tab-" + button.dataset.tab).classList.add("active");\n'
        '  });\n'
        '});\n'
        '</script>\n</main></body></html>'
    )
    page_path = _os.path.join(base_directory, 'experiment_report.html')
    with open(page_path, 'w', encoding='utf-8') as page_file:
        page_file.write(page)

    with open(_os.path.join(base_directory, 'experiment_report.md'), 'w', encoding='utf-8') as md_file:
        for data in sections:
            md_file.write(f'# {data["title"]}\n\n_{data["explanation"]}_\n\n')
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
#   hard punt: < 20% -- the build has abandoned the category
#   soft punt: 20-40% -- deliberately de-emphasised but not abandoned
_HARD_PUNT_RATE = 0.20
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
        assert distinct >= 3, (
            f'{format_key} {season}: only {distinct} distinct punt set(s) — field collapsed: {breakdown}'
        )


# ── FAST: Rotisserie punts stay shallow (draft + auction, per season) ────────

@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_roto_minimal_punting(sessions, auction):
    """Rotisserie rewards every category point, so committed hard punts are (mostly) irrational; the
    self-play field should learn shallow leans, not the H2H-style full punts."""
    mode = 'auction' if auction else 'draft'
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
        _record_row('Rotisserie punting (hard <20% / soft 20-40% expected win rate)',
                    ['Season', 'Mode', 'Hard', 'Soft', 'Weakest category', 'Weakest rate', 'Player'],
                    [season, mode, hard_count, soft_count, weakest_cat,
                     f'{100 * float(np.min(weakest_rates)):.0f}%',
                     session.player_registry[weakest_player].name])
        # Hard punts are allowed but must stay RARE: an extreme-profile star can rationally
        # commit to one (Giannis's FT% after the team-denominator correction to percentage
        # G-scores — accepted 2026-08-14), but more than a couple among the top anchors
        # would mean full H2H-style punting has leaked into Rotisserie.
        hard_ceiling = 2
        assert hard_count <= hard_ceiling, (
            f'{season}: {hard_count} hard Roto punt(s) among the top anchors '
            f'(> {hard_ceiling}) — hard punting should be a rare outlier, not common'
        )
        # Soft counts measure roster SHAPE as much as strategy: an auction anchor's expected team is a
        # star plus budget-priced fill, which mechanically spreads category win rates wider than a
        # snake-draft expectation and parks more of them in the 20-40% band without any punt intent.
        # The near-zero hard floor above is the real Roto property; the soft ceilings only catch
        # egregious drift (draft measured <=21/season; auction measured ~47 on the widest season, 2020-21).
        soft_ceiling = (5 if auction else 2) * len(rows)
        assert soft_count <= soft_ceiling, f'{season}: Roto soft-punting excessively ({soft_count})'


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
        assert len(common) >= 10, f'{season}: candidate overlap collapsed across one predicted pick'
        raw_deltas  = {n: after[n][0] - before[n][0] for n in common}
        level_shift = float(np.median(list(raw_deltas.values())))
        h_deltas    = {n: abs(d - level_shift) for n, d in raw_deltas.items()}
        w_deltas    = {n: float(np.max(np.abs(after[n][1] - before[n][1]))) for n in common}
        worst_h_name = max(h_deltas, key=h_deltas.get)

        # PUNT-SET stability: H-scores staying put is not enough — a candidate's displayed STRATEGY
        # (which categories its build punts, win rate < 40) flipping across a predicted pick is a lurch
        # even on a value plateau. Threshold dust is not a flip: a category only counts when it crosses
        # 40% by a real margin (> 5pp move), i.e. the build genuinely entered or abandoned a punt.
        punt_flips = {}
        for name in common:
            before_rates, after_rates = before[name][2], after[name][2]
            crossed = [(i, before_rates[i], after_rates[i]) for i in range(len(before_rates))
                       if (before_rates[i] < 40.0) != (after_rates[i] < 40.0)
                       and abs(after_rates[i] - before_rates[i]) > 5.0]
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
                    ['Season', 'Mode', 'Shared shift (whole board)', 'Max relative move',
                     'Biggest mover', 'Max dweight', 'Punt flips', 'Worst punt flip', 'Players compared'],
                    [season, mode, f'{level_shift:+.2f}', f'{max(h_deltas.values()):.2f}',
                     session.player_registry[worst_h_name].name, f'{max(w_deltas.values()):.1f}',
                     len(punt_flips), flip_text, len(common)])
        # 1e-6: the deltas come from 2-decimal display scores, so a mover sitting exactly ON the floor
        # (0.60) must not fail on float dust.
        assert max(h_deltas.values()) <= 0.6 + 1e-6, f'{season}: lurch {max(h_deltas.values()):.2f}'
        assert abs(level_shift) <= 5.0, f'{season}: implausible global re-level ({level_shift:+.2f})'
        assert max(w_deltas.values()) <= 5.0, f'{season}: build lurch {max(w_deltas.values()):.1f}'
        assert not punt_flips, f'{season} {mode}: punt builds flipped across a predicted pick: {flip_text}'


# ── FAST: self-play convergence (per season) ─────────────────────────────────

def test_self_play_convergence(sessions):
    """Show the punt structure of every bootstrap pass directly: per pass, how many of the tracked top
    players punt each category. Convergence = the counts stabilise from pass to pass; oscillation =
    counts swinging back and forth. No abstract metrics -- the evolution itself is the report."""
    for season in _SEASONS:
        session = sessions('MC', False, season)
        agent   = session.agent

        records  = []
        original = agent._run_bootstrap_pass

        def recording_pass(empty, n_iterations, cash, candidate_subset=None,
                           preserve_frozen_weights=False, _original=original, _records=records):
            result = _original(empty, n_iterations, cash, candidate_subset, preserve_frozen_weights)
            _records.append((candidate_subset is None, result['Rates'].copy()))
            return result

        agent._run_bootstrap_pass = recording_pass
        try:
            agent.populate_default_h_scores(session.current_settings['n_iterations'])
        finally:
            del agent._run_bootstrap_pass   # drop the instance shadow, restoring the class method

        categories = session.current_settings['categories']
        short      = [_SHORT_CATEGORY.get(category, category) for category in categories]
        # Track a FIXED set of players across passes so the counts are comparable: the top 12 of the
        # anchor subset (generic ranking), present in every pass.
        tracked = list(records[0][1].index[:12])

        for pass_number, (is_full_pool, rates) in enumerate(records):
            # Solve-half passes only carry rows for the group they solved, so each pass
            # reports the punts of whichever tracked players it re-solved; the Level-0
            # pass and the serves cover the full tracked set.
            present = rates.index.intersection(tracked)
            if is_full_pool:
                assert len(present) == len(tracked), (
                    f'{season} pass {pass_number}: full-pool pass missing tracked players')
            tracked_rates = rates.loc[present].to_numpy()
            assert not np.isnan(tracked_rates).any(), f'{season} pass {pass_number}: NaN win rates'
            punt_counts = (tracked_rates < _SOFT_PUNT_RATE).sum(axis=0)
            label = 'serve' if is_full_pool else str(pass_number)
            _record_row('Self-play convergence (MC bootstrap)',
                        ['Season', 'Pass'] + short,
                        [season, label] + [int(count) for count in punt_counts])

        final_counts = (records[-1][1].reindex(tracked).to_numpy() < _SOFT_PUNT_RATE).sum(axis=0)
        assert final_counts.sum() > 0, f'{season}: served field shows no punts at all — implausible'


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
    for season in _SEASONS:
        # beth=0, kappa=0: the season-sim conventions (see the header notes).
        session = _build_session(_FORMATS[format_key], season=season, beth=0)
        n_seats = _SIM_SEATS or session.current_settings['n_drafters']
        scores  = [100 * _draft_h_seat_in_g_field(session, seat) for seat in range(n_seats)]
        pooled.extend(scores)
        _record_row('H-scoring vs a G-score field',
                    ['Format', 'Season', 'Mean win rate'] + _seat_columns(n_seats),
                    [format_key, season, f'{np.mean(scores):.1f}'] + [f'{s:.1f}' for s in scores])
        assert float(np.mean(scores)) > 50.5, f'{format_key} {season}: H barely beats G ({scores})'
    _record_row('H-scoring vs a G-score field',
                ['Format', 'Season', 'Mean win rate'] + _seat_columns(n_seats),
                [format_key, f'ALL ({len(pooled)} drafts)', f'{np.mean(pooled):.1f}'] + ['—'] * n_seats)
    assert float(np.mean(pooled)) > 52.0, f'{format_key}: pooled H-vs-G mean low ({np.mean(pooled):.1f})'


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
        assert warm_mean >= cold_mean - 0.25, (
            f'{season}: warm starts converge worse than cold ({warm_mean:.3f} vs {cold_mean:.3f})'
        )


@sims
def test_multi_start_seeding_helps():
    """An OPTIMIZER comparison (no competing agents — see the section explanation): solve the same
    players' builds twice, seeded by the punt scan vs a single neutral seed, and compare PAIRED
    per-player objective values over the scan's top 12."""
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
            assert float(np.mean(deltas)) >= -0.1, (
                f'{season}: punt seed scan below a neutral seed (mean gain {np.mean(deltas):+.2f}pp)'
            )
