# testing_files/test_behavior_properties.py
# BEHAVIOURAL tests — the fourth tier of the test taxonomy:
#   1. backend tests   (deterministic, must be green)
#   2. frontend tests  (deterministic, must be green)
#   3. golden tests    (regenerated whenever the algorithm intentionally changes)
#   4. BEHAVIOURAL     (this file: no exact right answer — measurements to LOOK AT so a human can judge
#      that everything still looks right. Every test PRINTS its measurements; assertions here are only
#      loose catastrophic floors — e.g. "the field collapsed to one punt" — not tight expectations.)
#
# KAPPA POLICY: kappa = 0.3 (the app default, via the fixture) EVERYWHERE, with exactly one exception:
# simulations against a G-score field, where the field does not punt, so there is no crowd for the
# anti-crowded-punt penalty to defect from and kappa would only distort the H-vs-G comparison.
#
# FAST properties (run in the default suite; single sessions, a few evaluates each):
#   - Punt diversity (MC + EC, draft + auction): the predicted field must not collapse into one punt.
#   - Rotisserie punting (draft + auction): hard punts should be rare-to-absent.
#   - Early-pick stability (draft + auction): an opponent taking its PREDICTED player must not lurch
#     the displayed builds or values (the field was priced in; only tiny residuals may move).
#   - Self-play convergence: the bootstrap must settle (no oscillation) — punt structure locks while
#     the EMA damps the per-pass best-response churn.
#
# SLOW simulation properties (full drafts; set RUN_BEHAVIOR_SIMS=1 to run):
#   - H-scoring beats a G-score field (the most basic utility claim; the kappa exception applies).
#   - Awareness vs a field of UNAWARE H-drafters (at app kappa; report the gain — kappa itself already
#     prices punt-crowding, so ~0 is the expected reading here, and materially negative is the red flag).
#   - Awareness is not harmful against a G-score field (kappa exception applies).
#   - Warm starts converge no worse than cold starts on the same iteration budget.
#   - Multi-start seeding beats a single neutral seed.
#
# All sessions are built through the request parameters (use_opponent_awareness, kappa, ...) rather than
# environment pins, and every draft loop calls agent.reset_draft_state() so no state leaks across drafts.

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'season_simulation'))

from benchmark_helpers import client, _build_session_request
from backend.state.session import get_session
from backend.services.ranking import rank_candidates

_RUN_SIMS = bool(os.environ.get('RUN_BEHAVIOR_SIMS'))
# 0 = every seat in the league (the default: one full season's worth of draft positions).
_SIM_SEATS = int(os.environ.get('BEHAVIOR_SIM_SEATS', '0'))

_FORMATS = {
    'EC':   'Head to Head: Each Category',
    'MC':   'Head to Head: Most Categories',
    'Roto': 'Rotisserie',
}

_SHORT_CATEGORY = {
    'Field Goal %': 'FG%', 'Free Throw %': 'FT%', 'Threes': '3s', 'Points': 'PTS',
    'Rebounds': 'REB', 'Assists': 'AST', 'Steals': 'STL', 'Blocks': 'BLK', 'Turnovers': 'TO',
}


# ── measurement report ────────────────────────────────────────────────────────
# The behavioural tier is read by a human: every test records its measurements here, and the conftest
# terminal-summary hook renders the whole picture at the end of the run (and to behavior_report.md).

_REPORT: dict = {}

# What each section MEANS — rendered under its header so the report stands alone. Keys must match the
# titles passed to _record().
_SECTION_EXPLANATIONS = {
    'Punt diversity (top-12 anchors)':
        'For each of the top 12 players, the build the self-play process predicts a team drafted around '
        'that player would commit to, expressed by its PUNTED categories (expected win rate under 40%). '
        'Hard punts (under 20%, marked with !) are abandoned categories; soft punts (20-40%) are '
        'de-emphasised. Shown: how many of the 12 share each punt combination. Healthy: several distinct '
        'combinations (historically 5-8); one combination shared by everyone = the opponent model has '
        'collapsed and the app steers every drafter into the same build.',
    'Rotisserie punting (hard <20% / soft 20-40% expected win rate)':
        'Rotisserie scores every standings point, so abandoning a category is (mostly) irrational there. '
        'Hard punt = a top-12 build expecting to win a category under 20% of the time (want ZERO); soft '
        'punt = 20-40% (a mild lean; a few are fine). Also shown: the single weakest category expectation '
        'across all 12 builds. H2H formats, by contrast, punt hard on purpose.',
    'Predicted-pick stability (EC; opponent takes its predicted player)':
        'Evaluate all candidates, let an opponent take exactly the player the model already predicted '
        'for them, evaluate again. "Level shift" = how much every candidate moved together (drafts: ~0; '
        'auctions: legitimately nonzero, the bought player leaves the market and prices rescale). '
        '"Max residual |dH|" = the biggest movement of any single candidate RELATIVE to the field, in '
        'H-score percentage points — this is the lurch number; a predicted pick should be pre-priced, so '
        'want well under 1. "Max |dweight|" = biggest change in any displayed build weight (display '
        'units, 100 = neutral); want a few points at most.',
    'Self-play convergence (MC bootstrap)':
        'The pre-draft bootstrap alternates predict-field / best-respond for ~15 passes. "Consecutive-'
        'cos" = how little the predicted field\'s punt profile rotates between the final passes (1.0 = '
        'not rotating at all = settled; low or bouncing = oscillation). "Locks by pass N/M" = the pass '
        'after which the field\'s top-3 punt set stopped changing — want it locked well before the end.',
    'H-scoring vs a G-score field (kappa exception applies)':
        'Full snake drafts: one seat drafts by H-score, the other 11 pick greedily by G-score. The '
        'number = the H seat\'s expected head-to-head category win rate against that field (50.0 = '
        'break-even; every point above is real edge). Per-seat values are draft positions 1..N — later '
        'positions are genuinely harder. This is the most basic utility claim of the whole system.',
    'Awareness (opponent model on vs off)':
        'Identical drafts by the same seat with the opponent model ON minus OFF; gain is in H-score '
        'percentage points (+1.0 = one extra point of expected category win rate). vs an UNAWARE H '
        'field: kappa already prices punt-crowding, so expect a small positive-to-zero gain — clearly '
        'negative would mean the model hurts against naive H-drafters. vs a G field: the model\'s '
        'predictions are simply wrong there (G drafters never punt), so the property is harmlessness, '
        'gain ~0.',
    'Warm start (purpose: display stability, not better answers)':
        'Same evaluate with descents warm-started from stored builds vs cold-started from seed scans, '
        'equal iteration budget; numbers are mean H (percentage points) of the top 12 candidates. Warm '
        'starting exists to keep displayed strategies CONSISTENT with what the user was already shown — '
        'the expectation here is parity (stability must not cost convergence), and the stability itself '
        'is measured by predicted-pick stability above.',
    'Multi-start seeding':
        'Base H-scores computed with the punt seed scan (try one gentle punt per category, keep each '
        'candidate\'s best) vs a single neutral seed; numbers are mean H (percentage points) of the top '
        '12. The scan should win: without it, candidates whose true build is a committed punt get stuck '
        'near neutral.',
}


def _record(section, line):
    _REPORT.setdefault(section, []).append(line)
    print(f'\n[{section}] {line}')


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


def render_behavior_report(markdown=False):
    if not _REPORT:
        return ''
    lines = []
    for section, entries in _REPORT.items():
        lines.append(f'## {section}' if markdown else section)
        explanation = _SECTION_EXPLANATIONS.get(section)
        if explanation:
            if markdown:
                lines.append(f'*{explanation}*')
                lines.append('')
            else:
                lines.extend(f'  | {wrapped}' for wrapped in _wrap(explanation))
        for entry in entries:
            lines.append(f'- **{entry}**' if markdown else f'    {entry}')
        lines.append('')
    return '\n'.join(lines)


def _short_names(categories, indices):
    return '/'.join(_SHORT_CATEGORY.get(categories[i], categories[i]) for i in sorted(indices))


# ── persistent per-section results + one tabbed report page ──────────────────
# Each run persists ONLY the sections it measured to behavior_reports/<slug>.json — so running any
# subset of tests (pytest -k ...) regenerates just those results while the rest keep their last
# measurement — then rebuilds ONE tabbed behavior_report.html from every stored section, fresh and
# stale alike, each tab labelled with its own generation time.

_REPORT_PAGE_CSS = """
  :root { --bg:#ffffff; --fg:#26282e; --muted:#6a6f7a; --line:#e4e6ea; --card:#f7f8fa; --accent:#33556e; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161c; --fg:#e6e8ec; --muted:#9aa0ab; --line:#2c303a; --card:#1b1e26; --accent:#8fb4d1; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, "Segoe UI", sans-serif; }
  main { max-width: 940px; margin: 0 auto; padding: 28px 20px 64px; }
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
  .explain { color: var(--muted); font-size: .9rem; margin: 0 0 12px; max-width: 75ch; }
  ul { list-style:none; margin:0; padding:0; background:var(--card); border:1px solid var(--line);
       border-radius:10px; }
  li { font-family: ui-monospace, Consolas, monospace; font-size: .87rem; padding: 8px 12px;
       border-top: 1px solid var(--line); overflow-x:auto; white-space:nowrap; }
  li:first-child { border-top:none; }
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
    ordered = [stored.pop(title) for title in _SECTION_EXPLANATIONS if title in stored]
    return ordered + list(stored.values())


def write_behavior_report_files(base_directory):
    """Persist this run's sections, rebuild the tabbed page from everything stored. Returns the page path."""
    import json
    import os as _os
    from datetime import datetime
    from html import escape

    reports_dir = _os.path.join(base_directory, 'behavior_reports')
    _os.makedirs(reports_dir, exist_ok=True)
    stamp = datetime.now().strftime('%Y-%m-%d %H:%M')

    for section, entries in _REPORT.items():
        payload = {
            'title':       section,
            'explanation': _SECTION_EXPLANATIONS.get(section, ''),
            'entries':     entries,
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
        rows = '\n'.join(f'<li>{escape(entry)}</li>' for entry in data['entries'])
        panels.append(
            f'<section id="tab-{position}"{active}>\n<h2>{escape(title)}</h2>\n'
            f'<div class="generated">generated {escape(data["generated"])}</div>\n'
            f'<p class="explain">{escape(data["explanation"])}</p>\n<ul>{rows}</ul>\n</section>'
        )

    newline = '\n'
    page = (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Behavioural properties report</title>\n'
        f'<style>{_REPORT_PAGE_CSS}</style></head><body><main>\n'
        '<h1>Behavioural properties report</h1>\n'
        '<div class="stamp">Season 2024-25 &middot; measurements for human judgment — this tier has no '
        'exact right answers; each tab regenerates independently when its test runs and keeps its own '
        'timestamp.</div>\n'
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
    page_path = _os.path.join(base_directory, 'behavior_report.html')
    with open(page_path, 'w', encoding='utf-8') as page_file:
        page_file.write(page)

    with open(_os.path.join(base_directory, 'behavior_report.md'), 'w', encoding='utf-8') as md_file:
        for data in sections:
            md_file.write(f'# {data["title"]}\n\n_{data["explanation"]}_\n\n')
            for entry in data['entries']:
                md_file.write(f'- {entry}\n')
            md_file.write(f'\n(generated {data["generated"]})\n\n')
    return page_path


# ── session helpers ───────────────────────────────────────────────────────────

def _build_session(scoring_format, auction=False, **parameter_overrides):
    request = _build_session_request(
        scoring_format=scoring_format, cash_per_team=200 if auction else None,
    )
    request['parameters'].update(parameter_overrides)
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'session build failed: {response.text}'
    return get_session(response.json()['session_id'])


@pytest.fixture(scope='module')
def sessions():
    """One session per (format, mode) at app-true parameters, shared by every fast property test."""
    cache = {}

    def get(format_key, auction=False, **overrides):
        key = (format_key, auction, tuple(sorted(overrides.items())))
        if key not in cache:
            cache[key] = _build_session(_FORMATS[format_key], auction=auction, **overrides)
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


# ── FAST: punt diversity (MC + EC, draft + auction) ──────────────────────────

@pytest.mark.parametrize('format_key', ['MC', 'EC'])
@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_punt_diversity(sessions, format_key, auction):
    """The predicted field must be DIVERSE: if the top of the pool all converge to one punt, the
    opponent model has collapsed (the original everyone-punts-3s/FT%/TO failure mode)."""
    session = sessions(format_key, auction)
    agent   = session.agent
    categories = session.current_params['categories']
    punts   = [_classify_punts(rates) for _, rates in _anchor_rate_rows(agent)]
    distinct = len(set(punts))
    from collections import Counter
    breakdown = ', '.join(f'{_punt_label(categories, punt_set)} x{count}'
                          for punt_set, count in Counter(punts).most_common())
    hard_total = sum(1 for punt_set in punts for _, hard in punt_set if hard)
    soft_total = sum(1 for punt_set in punts for _, hard in punt_set if not hard)
    _record('Punt diversity (top-12 anchors)',
            f'{format_key:4} {"auction" if auction else "draft":7}: {distinct} distinct builds, '
            f'{hard_total} hard / {soft_total} soft punts — {breakdown}')
    assert distinct >= 3, (
        f'{format_key}: the top {len(punts)} anchors share only {distinct} punt set(s) — '
        f'the predicted field has collapsed: {breakdown}'
    )


# ── FAST: Rotisserie punts stay shallow (draft + auction) ────────────────────

@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_roto_minimal_punting(sessions, auction):
    """Rotisserie rewards every category point, so committed hard punts are (mostly) irrational; the
    self-play field should learn shallow leans, not the H2H-style full punts."""
    session = sessions('Roto', auction)
    agent   = session.agent
    categories = session.current_params['categories']
    rows = _anchor_rate_rows(agent)
    hard_count = sum(1 for _, rates in rows for r in rates if r < _HARD_PUNT_RATE)
    soft_count = sum(1 for _, rates in rows for r in rates if _HARD_PUNT_RATE <= r < _SOFT_PUNT_RATE)
    weakest_player, weakest_rates = min(rows, key=lambda pr: float(np.min(pr[1])))
    weakest_idx = int(np.argmin(weakest_rates))
    _record('Rotisserie punting (hard <20% / soft 20-40% expected win rate)',
            f'{"auction" if auction else "draft":7}: {hard_count} hard, {soft_count} soft punts across '
            f'top-{len(rows)} anchors; weakest category '
            f'{_SHORT_CATEGORY.get(categories[weakest_idx], categories[weakest_idx])} at '
            f'{100 * float(np.min(weakest_rates)):.0f}% ({weakest_player.split(" (")[0]})')
    assert hard_count == 0, f'{hard_count} HARD punt(s) in Rotisserie — irrational category abandonment'
    assert soft_count <= 2 * len(rows), f'Roto anchors soft-punting excessively ({soft_count})'


# ── FAST: early-pick stability (draft + auction) ─────────────────────────────

def _snapshot_candidates(result, top_n=15):
    return {
        c.name: (c.h_score, np.asarray(c.category_weights, dtype=float))
        for c in result.candidates[:top_n]
    }


@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
def test_early_pick_stability(sessions, auction):
    """An opponent drafting its PREDICTED first player confirms the field rather than moving it, so the
    displayed values and builds of the remaining candidates must stay (nearly) put — the anti-lurch
    guarantee (means are constructed invariant; only variance/pool residuals and the honest auction cash
    delta may move things)."""
    session = sessions('EC', auction)
    agent   = session.agent
    teams   = session.current_params['team_names']
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
    assert len(common) >= 10, 'candidate overlap collapsed across one predicted pick'
    # In auction mode the purchase legitimately re-levels EVERY candidate the same way (the bought player
    # leaves the purchasable pool, so replacement value and $-per-value rescale globally). Lurch means
    # RELATIVE movement -- one candidate jumping against the rest -- so judge deltas net of the shared
    # level shift (the median delta; ~0 in draft mode, where nothing re-levels).
    raw_deltas  = {n: after[n][0] - before[n][0] for n in common}
    level_shift = float(np.median(list(raw_deltas.values())))
    h_deltas    = {n: abs(d - level_shift) for n, d in raw_deltas.items()}
    w_deltas    = {n: float(np.max(np.abs(after[n][1] - before[n][1]))) for n in common}
    worst_h, worst_w = max(h_deltas.values()), max(w_deltas.values())
    worst_h_name = max(h_deltas, key=h_deltas.get)
    _record('Predicted-pick stability (EC; opponent takes its predicted player)',
            f'{"auction" if auction else "draft":7}: level shift {level_shift:+.2f}, '
            f'max residual |dH| {worst_h:.2f} ({worst_h_name.split(" (")[0]}), '
            f'max |dweight| {worst_w:.1f} over {len(common)} candidates')
    assert worst_h <= 0.6, f'H lurched {worst_h:.2f} relative to the field on a predicted pick: {h_deltas}'
    assert abs(level_shift) <= 5.0, f'implausible global re-level ({level_shift:+.2f}) on one pick'
    assert worst_w <= 5.0, f'displayed build lurched {worst_w:.1f} on a predicted pick: {w_deltas}'


# ── FAST: self-play convergence (no oscillation) ─────────────────────────────

def test_self_play_convergence(sessions):
    """The bootstrap's damped best-response must SETTLE, not oscillate: per-pass best responses may keep
    moving (a mixed equilibrium), but the EMA-committed field's punt structure should rotate less and
    less (consecutive-pass cosine near 1) and its punt set should lock well before the final pass."""
    from backend.math.algorithm_agents import _OPPONENT_SMOOTHING

    session = sessions('MC', False)
    agent   = session.agent

    records  = []
    original = agent._run_bootstrap_pass

    def recording_pass(empty, n_iterations, cash, candidate_subset=None, preserve_frozen_weights=False):
        result = original(empty, n_iterations, cash, candidate_subset, preserve_frozen_weights)
        records.append((candidate_subset is None, result['Future-Diff'].copy()))
        return result

    agent._run_bootstrap_pass = recording_pass
    try:
        agent.populate_default_h_scores(session.current_params['n_iterations'])
    finally:
        del agent._run_bootstrap_pass   # drop the instance shadow, restoring the class method

    anchor_passes = [diff for is_full_pool, diff in records if not is_full_pool]
    index     = anchor_passes[0].index
    responses = [diff.reindex(index).to_numpy() / (agent.n_picks - 1) for diff in anchor_passes]

    committed = responses[0]
    profiles  = [committed.mean(axis=0)]
    punt_sets = [tuple(np.argsort(profiles[0])[:3].tolist())]
    for response in responses[1:]:
        committed = (1 - _OPPONENT_SMOOTHING) * committed + _OPPONENT_SMOOTHING * response
        profiles.append(committed.mean(axis=0))
        punt_sets.append(tuple(np.argsort(profiles[-1])[:3].tolist()))

    def cosine(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    consecutive = [cosine(profiles[i], profiles[i - 1]) for i in range(1, len(profiles))]
    settle_cos  = float(np.mean(consecutive[-5:]))
    lock_pass   = max((i for i in range(1, len(punt_sets)) if punt_sets[i] != punt_sets[i - 1]), default=0)
    categories  = session.current_params['categories']
    _record('Self-play convergence (MC bootstrap)',
            f'{len(responses)} passes: consecutive-cos last-5 {settle_cos:.3f} (1.0 = settled), '
            f'field punt-set {_short_names(categories, punt_sets[-1])} locks by pass '
            f'{lock_pass}/{len(punt_sets) - 1}')
    assert not any(np.isnan(profile).any() for profile in profiles), 'NaN in the committed field'
    assert settle_cos > 0.6, (
        f'the committed field is still rotating hard at the end of the bootstrap ({settle_cos:.3f}) — '
        f'oscillation, not convergence'
    )


# ── SLOW simulation properties ────────────────────────────────────────────────

sims = pytest.mark.skipif(not _RUN_SIMS, reason='full-draft simulation; set RUN_BEHAVIOR_SIMS=1')


def _draft_h_seat_in_g_field(h_session, seat, candidate_limit=40):
    """One snake draft: `seat` drafts with the H agent, everyone else picks by G-score ranking.
    Returns the H seat's final-roster H-score."""
    from simulate import _gscore_ranking, _pick_gscore_player, _has_position_data   # season_simulation

    agent        = h_session.agent
    agent.reset_draft_state()
    n_drafters   = h_session.current_params['n_drafters']
    n_picks      = h_session.current_params['n_picks']
    n_iterations = h_session.current_params['n_iterations']
    teams        = [f'Drafter {i + 1}' for i in range(n_drafters)]
    assignments  = {t: [] for t in teams}
    g_ranking    = _gscore_ranking(h_session)
    has_positions   = _has_position_data(h_session)
    position_config = agent._pos_cfg
    drafted: set    = set()

    for pick_row in range(n_picks):
        for slot in range(n_drafters):
            index = slot if pick_row % 2 == 0 else (n_drafters - 1 - slot)
            team  = teams[index]
            if index == seat:
                result = rank_candidates(h_session, assignments, team, [], None, 0, candidate_limit)
                chosen = result.candidates[0].name
            else:
                chosen = _pick_gscore_player(
                    g_ranking, drafted, assignments[team], position_config, has_positions)
            assignments[team].append(chosen)
            drafted.add(chosen)

    scores = h_session.agent.get_h_scores(assignments, teams[seat], n_iterations)['Scores']
    return float(scores[scores.idxmax()])


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC'])
def test_h_scoring_beats_g_field(format_key):
    """The most basic utility claim: an H-scoring drafter in a league of G-score drafters should win
    more than it loses. kappa=0 here — the G field does not punt, so there is no crowd to defect from."""
    # beth=0, the season-sim convention: historical stats are objectively correct, and the Bayesian
    # self-doubt (beth) only regresses draft-time H toward average, diluting the measurement.
    session = _build_session(_FORMATS[format_key], kappa=0.0, beth=0)
    n_seats = _SIM_SEATS or session.current_params['n_drafters']
    scores  = [_draft_h_seat_in_g_field(session, seat) for seat in range(n_seats)]
    _record('H-scoring vs a G-score field (kappa exception applies)',
            f'{format_key:4}: mean win rate {100 * np.mean(scores):.1f} over {len(scores)} seats — '
            f'per-seat {[round(100 * s, 1) for s in scores]}')
    assert float(np.mean(scores)) > 0.52, f'H-scoring failed to beat a G field: {scores}'


@sims
def test_awareness_vs_unaware_h_field():
    """Awareness against a field of UNAWARE H-drafters, at app kappa (0.3 — the fixture default; no
    override here). Kappa already prices punt-crowding, so the expected reading is roughly ZERO gain —
    awareness's value shows against fields that punt strategically; this measurement guards the other
    side: it must not be materially NEGATIVE (awareness harming you against naive H-drafters).

    ONE AGENT PER DRAFTER: every seat drafts with its own session. A single shared field agent would
    cross-contaminate its seats -- e.g. refresh_stale_team_states run from seat A's perspective can
    overwrite seat B's own team entry with an INFERENCE of B, so B's next pick warm-starts from A's model
    of B instead of B's own last build. Separate agents make each drafter genuinely self-contained."""
    from self_play import draft_population

    reference = _build_session(_FORMATS['EC'], use_opponent_awareness=False, beth=0)
    n_drafters   = reference.current_params['n_drafters']
    n_picks      = reference.current_params['n_picks']
    n_iterations = reference.current_params['n_iterations']

    field_sessions = [reference] + [
        _build_session(_FORMATS['EC'], use_opponent_awareness=False, beth=0)
        for _ in range(n_drafters - 1)
    ]
    aware_deviator = _build_session(_FORMATS['EC'], use_opponent_awareness=True, beth=0)

    def deviator_score(session_by_seat, seat):
        assignments = draft_population(session_by_seat, n_drafters, n_picks, 40)
        # One fixed scorer for both arms: a complete roster's H is parameter-clean, so any agent works,
        # but using the same one removes even rounding asymmetry.
        scores = reference.agent.get_h_scores(assignments, f'Drafter {seat + 1}', n_iterations)['Scores']
        return float(scores[scores.idxmax()])

    gains = []
    for seat in range(_SIM_SEATS or n_drafters):
        unaware_field = {i: field_sessions[i] for i in range(n_drafters)}
        aware_arm     = dict(unaware_field)
        aware_arm[seat] = aware_deviator
        gains.append(deviator_score(aware_arm, seat) - deviator_score(unaware_field, seat))
    _record('Awareness (opponent model on vs off)',
            f'vs UNAWARE H field (app kappa): mean gain {100 * np.mean(gains):+.2f}pp over {len(gains)} seats — '
            f'per-seat {[round(100 * g, 2) for g in gains]}')
    assert float(np.mean(gains)) > -0.01, f'awareness materially harmful vs an unaware H field: {gains}'


@sims
def test_awareness_not_harmful_vs_g_field():
    """Awareness mispredicts a G field (G drafters do not punt) — that misprediction must be harmless."""
    aware   = _build_session(_FORMATS['EC'], kappa=0.0, use_opponent_awareness=True, beth=0)
    unaware = _build_session(_FORMATS['EC'], kappa=0.0, use_opponent_awareness=False, beth=0)
    n_seats = _SIM_SEATS or aware.current_params['n_drafters']
    gains = []
    for seat in range(n_seats):
        gains.append(_draft_h_seat_in_g_field(aware, seat) - _draft_h_seat_in_g_field(unaware, seat))
    _record('Awareness (opponent model on vs off)',
            f'vs G field (kappa exception): mean gain {100 * np.mean(gains):+.2f}pp over {len(gains)} seats '
            f'(want: not materially negative) — per-seat {[round(100 * g, 2) for g in gains]}')
    assert float(np.mean(gains)) > -0.01, f'awareness is harmful against a G field: {gains}'


@sims
def test_warm_start_no_convergence_cost():
    """Warm starting is NOT primarily about finding a better answer -- its purpose is DISPLAY STABILITY:
    keeping each candidate's strategy consistent with the strategy previously shown to the user, instead
    of wandering between near-tied builds on every evaluate (the plateau-lurch problem). On value, the
    honest expectation is parity: this test only guards that the stability does not COST convergence
    (warm must not be materially worse than cold on the same iteration budget). The stability benefit
    itself is asserted by test_early_pick_stability."""
    session = _build_session(_FORMATS['EC'])
    agent   = session.agent
    teams   = session.current_params['team_names']
    assignments = {t: [] for t in teams}
    assignments[teams[0]] = [agent._anchor_player_order[0]]

    agent.reset_draft_state()
    warm = rank_candidates(session, assignments, teams[1], [], None, 0, 40)
    warm_mean = float(np.mean([c.h_score for c in warm.candidates[:12]]))

    frozen_weights, frozen_shares = agent._player_frozen_weights, agent._player_frozen_shares
    try:
        agent._player_frozen_weights = None
        agent._player_frozen_shares  = None
        agent.reset_draft_state()
        cold = rank_candidates(session, assignments, teams[1], [], None, 0, 40)
    finally:
        agent._player_frozen_weights = frozen_weights
        agent._player_frozen_shares  = frozen_shares
    cold_mean = float(np.mean([c.h_score for c in cold.candidates[:12]]))

    _record('Warm start (purpose: display stability, not better answers)',
            f'top-12 mean H on the same iteration budget: warm {warm_mean:.3f} vs cold {cold_mean:.3f} '
            f'(parity expected; the stability benefit is measured by predicted-pick stability above)')
    assert warm_mean >= cold_mean - 0.05, (
        f'warm starts converge worse than cold starts ({warm_mean:.3f} vs {cold_mean:.3f})'
    )


@sims
def test_multi_start_seeding_helps():
    """The punt seed scan must beat a single neutral seed: without multi-start, candidates whose optimum
    is a committed punt get stuck near neutral and score lower."""
    punt_scan = _build_session(_FORMATS['EC'], use_opponent_awareness=False)
    neutral   = _build_session(_FORMATS['EC'], use_opponent_awareness=False)
    neutral.agent.seed_mode = 'neutral'
    neutral.agent.reset_draft_state()
    neutral.agent.populate_default_h_scores(neutral.current_params['n_iterations'])

    scan_mean    = float(100 * punt_scan.agent.default_h_scores.head(12).mean())
    neutral_mean = float(100 * neutral.agent.default_h_scores.head(12).mean())
    _record('Multi-start seeding',
            f'top-12 mean base H: punt seed scan {scan_mean:.1f} vs single neutral seed {neutral_mean:.1f} (percentage points)')
    assert scan_mean >= neutral_mean - 0.1, (
        f'the punt seed scan scores below a single neutral seed ({scan_mean:.3f} vs {neutral_mean:.3f})'
    )
