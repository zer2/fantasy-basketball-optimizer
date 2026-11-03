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
from datetime import datetime
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
# Shortened seasons are left out of every experiment: a player's games played is a share of 82 games (data_retrieval),
# so in these seasons nobody reaches 100% and every player's expected weekly totals are understated (the user,
# 2026-11-01: exclude them rather than model them for now). A season list given in the environment is filtered too.
_SHORTENED_SEASONS = ('1998-99', '2011-12', '2019-20', '2020-21')


def exclude_shortened_seasons(seasons):
    """The seasons, without the shortened ones (_SHORTENED_SEASONS), saying which were left out."""
    excluded = [season for season in seasons if season in _SHORTENED_SEASONS]
    if excluded:
        print(f'(shortened seasons left out: {", ".join(excluded)})')
    return [season for season in seasons if season not in _SHORTENED_SEASONS]


_ALL_SEASONS = '2021-22,2022-23,2023-24,2024-25,2025-26'
_SEASONS = exclude_shortened_seasons(
    (os.environ.get('EXPERIMENT_SEASONS') or (_ALL_SEASONS if _RUN_SIMS else '2024-25')).split(','))

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
# Sections recorded to since they were last written to disk. In report mode every experiment writes its own sections
# the moment it finishes (persist_sections_after_each_experiment), so a long run never holds finished results only in
# memory, and a run that dies keeps everything measured before it died.
_SECTIONS_TO_SAVE: set = set()

# What each section MEANS — rendered under its header so the report stands alone. Keys must match the
# titles passed to _record_row().
_SECTION_EXPLANATIONS = {
    'Punt diversity (top-12 anchors)':
        "Does the predicted field spread across builds? For each of a season's top 12 players: the build the "
        'self-play field predicts a team drafted around them commits to, named by its punted categories (expected '
        'win rate under 40%; ! marks a hard punt, under 15%). The breakdown counts how many of the 12 share each '
        'combination. One build shared by everyone would mean the opponent model had collapsed and was steering '
        'every drafter into the same team.',
    'Rotisserie punting (hard <15% / soft 15-40% expected win rate)':
        'Does Rotisserie stay out of hard punts? Rotisserie scores every standings point, so abandoning a category '
        "is mostly irrational there (head-to-head formats punt hard on purpose), though an extreme star (Giannis's "
        'FT%) can rationally give one up. For the top 12 Rotisserie builds of each season:\n'
        '- Hard: categories expected to win under 15% of the time.\n'
        '- Soft: 15-40%, a mild lean. Auctions spread win rates wider, since their teams are a star plus cheap '
        'fill.\n'
        '- Weakest: the lowest category expectation among the 12.',
    'Draft pick stability':
        'When an opponent drafts exactly the player the model predicted, does the board stay put? The app '
        'evaluates the board, the opponent drafts its predicted first player, and the app evaluates again; the pick '
        'was already priced in. Each Category, Most Categories and Rotisserie.',
    'Auction pick stability':
        'When an opponent buys exactly the player the model predicted, does the board stay put? The app evaluates '
        'the board, the opponent buys its predicted first player for $70, and the app evaluates again; the purchase '
        'was already priced in. Each Category, Most Categories and Rotisserie.',
    'Self-play convergence':
        'Does self-play settle? Self-play descends the top players together while rebuilding the field they face '
        "every iteration. For each block of 10 iterations: how much the top 12 players' category weights move, "
        'and how many of them punt each category.',
    'Self-play optimality':
        "Is each top served build its player's best response to the field it was served on? Each of the top 12 is"
        ' descended again against that same field, for 300 iterations, from several starts; the shortfall is how '
        'far the served build is below the best of them.',
    'Serve stability':
        "Does solving a player again land where self-play put them? For the players that make up self-play's "
        "field, self-play's build is compared with the build the served board finds when it solves them again, "
        'from the seeds, against that field.',
    'H-scoring vs a G-score field':
        'Does H-scoring beat drafters who take the best G-score available? Full 12-team snake drafts: one seat '
        "drafts by H-score, the other 11 by G-score. Head to head shows the H seat's win rate as its H-score "
        'expects it and in real weeks; Rotisserie shows its probability of winning the league.',
    'Awareness (opponent model on vs off)':
        "Does modelling the opponents' punts help? The same seat drafts the same draft twice, with the opponent "
        'model on and off, against two fields:\n'
        '- unaware H drafters, who punt the way the model predicts: expected positive.\n'
        '- G drafters, who never punt, so the predictions are wrong: expected at least somewhat negative.',
    'App loss':
        "How much H-score does the app's answer give up on its actual objective? The reference is the best build "
        "found by the app's own descent from every start of a full seed menu, run for 2,000 iterations: a lower "
        'bound on the optimum, not a certificate. The loss is split by the fix that recovers it: multi-starting, '
        'the full seed menu, and more iterations.',
    'Peakedness':
        "On the app's actual objective, do all starts climb to the same peak? From every start of the full menu, "
        "the app's descent runs 2,000 iterations; end points within 0.01 H-score points of each other count as "
        'one peak.',
    'How single-peaked are H-scores':
        'On the simplified objective, does every seed climb to the optimum? From every seed of two menus the '
        "app's descent runs to the long budget. If every seed reaches the optimum there is one peak, and seeding "
        'only gives the descent a better starting point.',
    'Conditioning around the optimum':
        "Why does the app's descent get close to the optimum in few iterations? Near the peak, each direction's "
        "curvature times Adam's step size in that direction decides what the descent does: between 0.026 and 38 "
        'it spirals in, above 38 it bounces away, below 0.026 it crawls.',
    'Loop gain at the optimum':
        "Is each optimum a stable peak? H-scoring's build is self-referential: the weights decide which players "
        'the team expects to pick later, those decide where it is strong or weak, and that decides what each '
        'category is worth, which is what the weights should be. The loop gain is what a small nudge to the '
        'weights becomes after one trip around that loop: below 1 it shrinks back to the peak.',
    'Where the app loses to the optimum':
        'How much H-score does the app give up against the true optimum? On a simplified objective that branch '
        'and bound solves exactly, the gap is split by the fix that recovers it: multi-starting, the full seed '
        'menu (55), more iterations (2,000), and the exact algorithm. App + the four losses = Optimal.',
    'Draft order: snake vs third-round reversal':
        'How much does draft position matter when every drafter uses H-scoring, and does third-round reversal even '
        'it out? Twelve H-scoring drafters draft each season twice, once as a snake and once with third-round '
        'reversal; every seat\'s H-score and, head to head, its win rate in real weeks.',
    'Z-scores vs G-scores':
        'Do G-scores draft better teams than Z-scores? Z-scores are G-scores without the week-to-week variance. Every '
        'split of a 12-team league is drafted statically, from 1 seat ranking by Z-score and 11 by G-score to 11 and '
        '1; every team is scored by H-score and in real weeks.',
    'Planner vs H-scoring':
        "Would planning a concrete team beat H-scoring's smooth model of future picks? In pairs of 12-team snake "
        'drafts, six seats draft with the planner and six by H-scoring, then the seats swap.',
    'H-scoring vs a G-score field (auction)':
        'Does H-scoring beat G-scores in an auction too? Complete 12-team auctions: one seat is the app\'s H-scoring '
        'autodrafter, the other 11 bid by G-score dollar values. Each Category and Most Categories show the H seat\'s '
        'win rate as its H-score expects it and in real weeks; Rotisserie its probability of winning the league.',
    'Draft timing':
        'Where does a draft\'s time go? Cold session builds and a whole 12-team draft through the app\'s own routes, '
        'every request broken down by the functions it spends its time in, averaged per request.',
    'Draft timing by round':
        'How long each pick takes, by round: the first 100 candidates (what an autodrafter waits for, and the wait '
        'before a person\'s board appears) and the whole board (every batch).',
    'Auction timing':
        'Where does an auction\'s time go? Cold session builds and a whole autodraft auction through the app\'s own '
        'routes, every request broken down by the functions it spends its time in, averaged per request.',
    'Auction timing by round':
        'How long each nomination and each auction-table evaluate takes, by round of sales.',
    'Pool-size calibration':
        'How big a pool do the future-pick model\'s picks behave like? The model prices each future pick as the best, '
        'by the team\'s weights, of a window of M surviving players (the app uses M = 25). For every pick and each of '
        'that team\'s later picks, the realised tilt is the best player still available by the pick\'s weights minus '
        'the best by generic value; the effective M is the M whose predicted tilt is closest to it. Two fields: twelve '
        'H-scoring seats (the draft-order drafts, both orders), and twelve G-score seats, each seat\'s H-scoring '
        'session recording the weights it would pick with.',
    'Category weights by round':
        'How do the category weights move from the first pick to the last? The weights behind every pick of the '
        'draft-order drafts (twelve H-scoring seats, both orders, every season), by round.',
    'Category weight correlations':
        'Which categories do the weights move together? The correlation between each pair of categories\' weights '
        'over every recorded pick, per format, beside the correlation between players\' category scores.',
    'Best player-seasons':
        'The best player-seasons since 2000, by first-pick H-score in Each Category, with each one\'s rank by Half and '
        'Half, Most Categories and Rotisserie H-score and by G-score.',
    'Best player-seasons, pre-2000':
        'The best player-seasons from 1984-85 to 1999-00, by first-pick H-score in Each Category, with each one\'s rank '
        'by Half and Half, Most Categories and Rotisserie H-score and by G-score. Ranked apart from 2000 on: these '
        'seasons have no reliable positions.',
    'Smooth vs simple scaling':
        'Comparing the simple scaling rule for scaling up dollar values when there isn\'t enough value left on the '
        'board, to the smooth version. See "Overbidding for packing efficiency" in the Auction Mode docs for details.\n'
        'These are 6x6 simulations run via autodraft between the two rules. They show that the smooth version more '
        'successfully spends its money efficiently, and creates teams with higher H-scores.',
    'Cash-share rules: builds (own view)':
        "Does bidding up the stars cost a team its build? Each team's build as it sees it, by rule: the categories it "
        'punts (expected win rate under 40%; ! under 15%). Value in punted is the share of its players\' positive '
        'X-score value in those categories; categories punted is that share for players bought regardless of the build. '
        'A misfit is a player with at least half his value in categories the team punts.',
    'Dollar shape: Gini by SAVOR settings':
        'How unequal are the starting board\'s dollar values under each SAVOR setting? The Gini coefficient of the '
        'dollars of the players who would be drafted, for each replacement threshold and noise.',
    'Dollar inequality by season':
        'Has the spread of auction values changed over the years? For each season, the Gini coefficient of the '
        'dollars of the players who would be drafted at the start of an auction, with SAVOR off and at the app\'s '
        'defaults.',
    'Most valuable player-seasons':
        'The most valuable players of every season, by their dollars at the start of an auction at the app\'s '
        'defaults, ranked across all seasons.',
    'Dollar curves by league size':
        'How does the shape of auction values change with the size of the league? Every drafted player\'s dollars at '
        'the start of an auction, by draft rank, for different numbers of teams and roster sizes.',
    'Dollar shape: replacement level and SAVOR':
        "How do SAVOR's noise and its replacement threshold reshape dollar values? On the board at the start of "
        "an auction, each combination of threshold ($0, the app's $2, $4) and noise (0, the app's 10, 20) is "
        'compared with both off.',
}


def _record_row(section, columns, row):
    """Add one structured row to a section's table (creating the table on first use). Every report
    renderer -- console, markdown, HTML tab -- draws real tables from these rows."""
    table = _REPORT.setdefault(section, {'columns': list(columns), 'rows': []})
    assert table['columns'] == list(columns), f'inconsistent columns within section {section!r}'
    table['rows'].append([str(cell) for cell in row])
    _SECTIONS_TO_SAVE.add(section)
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
# Written ONLY in report mode (RUN_EXPERIMENTS=1): each experiment, as soon as it finishes, persists the sections it
# measured to experiment_results/<slug>.json and rebuilds ONE tabbed experiment_report.html from every stored section,
# fresh and stale alike, each tab labelled with its own generation time.

# One archive folder per run, named when the run starts, so every section the run saves lands in the same place.
_RUN_ARCHIVE_NAME = datetime.now().strftime('%Y-%m-%dT%H%M%S')

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


# The columns a page's partial runs are merged by (merge_other_formats); every other page with a Format column merges by
# format.
_MERGE_KEYS = {'Planner vs H-scoring': ('Season',),
               'Best player-seasons': ('Season',),
               'Best player-seasons, pre-2000': ('Season',),
               'Draft timing': ('Format',),
               'Dollar inequality by season': ('Season',),
               'Dollar curves by league size': ('Season',),
               'Most valuable player-seasons': ('Season',),
               'Draft order: snake vs third-round reversal': ('Format', 'Season'),
               'Z-scores vs G-scores': ('Format', 'Season')}


def merge_other_formats(payload, stored_path, stamp, merge_columns):
    """A run that measured only some values of merge_columns (some formats; for some pages, some formats and seasons)
    replaces only those rows: the stored rows of every other value are kept, after this run's, with the time each was
    last generated (generated_by_format, keyed by the values joined with a space). A stored page with different
    columns is replaced whole -- its rows cannot sit in this table."""
    import json
    with open(stored_path, encoding='utf-8') as stored_file:
        stored = json.load(stored_file)
    indices = [payload['columns'].index(name) for name in merge_columns]

    def key_of(row):
        return ' '.join(str(row[index]) for index in indices)

    measured = {key_of(row) for row in payload['rows']}
    generated = {key: stamp for key in measured}
    if stored.get('columns') == payload['columns']:
        kept = [row for row in stored['rows'] if key_of(row) not in measured]
        payload['rows'] = payload['rows'] + kept
        stored_generated = stored.get('generated_by_format') or {key_of(row): stored['generated']
                                                                  for row in stored['rows']}
        generated.update({key: when for key, when in stored_generated.items() if key not in measured})
    payload['generated_by_format'] = generated


# Each page's definitions and settings, shown under its charts; the explanation above them is the page's question and
# how it is run.
_SECTION_DETAILS = {
    'H-scoring vs a G-score field':
        "- S1-S12: the H seat's draft position; later ones are genuinely harder.\n"
        '- Each Category: the expected category win rate; Most Categories: the expected matchup win rate; break-'
        'even 50%. Rotisserie: the probability of winning the league; break-even 1 in 12 (8.3%).\n'
        '- Real win rate: against each opponent, 1,000 matchups in which every player brings one week of that '
        'season drawn at random (a week without games is a zero week); all 13 players count.\n'
        "- Most Categories' real win rate is above its H-score in every season, most likely because the H-score's "
        'matchup model treats the categories as independent and ignores the correlations between them.\n'
        "- Settings: beth = 0 (the season's actual statistics are the truth); psi = 0, so a missed game counts as"
        " nothing, as it does in the real weeks. The H seat's opponent model is off head to head; Rotisserie pins"
        ' it on.',
    'Awareness (opponent model on vs off)':
        'The number is on minus off, in H-score points. Every seat drafts with its own agent. The ALL rows pool '
        'every season and seat, with a standard error; a pooled mean is coloured only when it is more than 2 '
        'standard errors from 0.',
    'Draft order: snake vs third-round reversal':
        '- Snake: forward in the first round, then alternating. Third-round reversal: rounds one and two as a snake, '
        'round three in round two\'s direction again (the last seat of round one picks first in rounds two and '
        'three), then alternating. H-scoring\'s future-pick model does not depend on the order.\n'
        '- Every seat drafts by H-scoring at the app\'s opponent-model confidence (0.5 head to head; Rotisserie pins '
        'it at 1), each with its own agent. Identical drafters make each draft deterministic: one draft per season '
        'and order gives every seat.\n'
        '- H-score: each team\'s complete-team H-score against the actual field (Each Category the expected category '
        'win rate, Most Categories the expected matchup win rate, Rotisserie the probability of winning the league). '
        'Real weeks: the same rate from 1,000 matchups against each opponent in which every player brings one random '
        'week of the season (a week without games is a zero week).\n'
        '- beth = 0 and psi = 0, as in H-scoring vs a G-score field. The charts average each seat over the seasons; '
        'bars are \u00b12 standard errors across seasons.',
    'Z-scores vs G-scores':
        '- Z-scores: the G-score pipeline with the weekly variance term at 0 (each score\'s denominator is the '
        'spread of players\' means alone), on exactly the inputs the session\'s own pipeline received; everything '
        'else -- the percentage-volume correction, the games-played re-centring, the representative players -- is '
        'computed the same way.\n'
        '- Static drafting: every seat takes the best player left on its own ranking whom its roster can still place.\n'
        '- Each split is drafted 24 times per season, the Z seats chosen at random each time, so draft position '
        'averages out. A draft records each group\'s mean.\n'
        '- H-score: each team\'s complete-team H-score against the actual field. Real weeks: as in H-scoring vs a '
        'G-score field. beth = 0 and psi = 0. Each Category and Most Categories only: G-scores are defined differently '
        'for Rotisserie.',
    'Planner vs H-scoring':
        "- A planner seat plans the rest of its team around each of H-scoring's top 40 candidates (an exact naive"
        ' team, then Frank-Wolfe, then swaps) and takes the one whose planned team scores highest.\n'
        "- app's example team: the planner as the app ships it (held to H-scoring's slots, flex allocation and "
        'win rates; one open pick per round for availability).\n'
        '- unconstrained planner: the complete-team H-score alone, any eligible slot, and availability from the '
        "seat's actual remaining pick numbers.\n"
        "- Planner minus H: the planner seats' mean complete-team H-score minus the H-scoring seats', averaged "
        "over the pair's two drafts. S1-S12: that seat drafting with the planner minus the same seat drafting by "
        'H-scoring.\n'
        '- Settings: 2024-25, Each Category, beth = 0.',
    'Self-play convergence':
        "- Mean change: a player's change in category weights from the previous iteration (Euclidean norm, in the"
        " app's display units, 100 = neutral), averaged over the block's iterations and the 12 players. Max "
        'change: the largest single change in the block, and whose. Switching back and forth counts in full.\n'
        "- Drift: each player's weights averaged over the block against its weights averaged over the previous "
        'block, the average over the 12 and the largest. Switching back and forth cancels; only steady drift '
        'shows.\n'
        '- Category columns: how many of the 12 current starts punt that category (expected win rate under 40%) '
        "at the block's end.\n"
        '- The app runs 100 iterations; the doubled run continues the same descent for twice as long. '
        "Rotisserie's self-play has one start per player, so no switching.",
    'Self-play optimality':
        'The starts: the served build, the balanced build and a gentle punt of each category; Best H is the '
        'highest any of them reaches. A shortfall over 0.1 H-score points means the served build stopped on a '
        'lower peak than one a descent can reach. Reads only the served board and the stored field, so it does '
        'not depend on how self-play built them.',
    'Serve stability':
        '- The field: a build for each of its 11 seats and the spare.\n'
        '- Charts: how many places each player moves among those players (ranked by self-play H-score, then by '
        'served H-score), and how far its category win rates move on average (percentage points).\n'
        "- Each Category and Most Categories at the app's opponent-model confidence (C = 1 is in the table), "
        'Rotisserie at 1.\n'
        "- Mean |tilt gap|: the served per-pick category tilt minus the field's, its size averaged over the "
        'seated players and categories. Worst category gap: the category whose gap averaged over the seats is '
        'largest. Top 12 unseated: players in the served top 12 who are not in the field. Per category: the '
        "seated players' average served win rate against each other (at C = 1 a consistent field puts them near "
        '50%).',
    'Draft pick stability':
        '- Charts: each candidate in both top 15s, before and after: how many places its rank moves, and how far '
        'its category win rates move on average (percentage points).\n'
        '- Shared shift: the median H-score change common to every compared candidate, the whole board moving as '
        'one.\n'
        '- Max relative move: the largest single H-score move after removing the shared shift. Both are in '
        'percentage points.\n'
        '- Max dweight: the largest change in a displayed build weight (100 = neutral).\n'
        '- Punt flips: builds whose shown strategy changes: a category crosses the 40% punt line and its win rate'
        ' moves at least 10 points.',
    'Auction pick stability':
        '- Chart: each candidate in both top 15s, before and after: how far its category win rates move on average '
        '(percentage points). In an auction a player\'s place on the board does not matter, so there is no ranking '
        'chart.\n'
        '- Shared shift: the median H-score change common to every compared candidate, the whole board moving as '
        'one (the market reprices once the player is bought).\n'
        '- Max relative move: the largest single H-score move after removing the shared shift. Both are in '
        'percentage points.\n'
        '- Max dweight: the largest change in a displayed build weight (100 = neutral).\n'
        '- Punt flips: builds whose shown strategy changes: a category crosses the 40% punt line and its win rate'
        ' moves at least 10 points.',
    'App loss':
        '- The app picks its start, then descends for 30 iterations. Each Category and Most Categories pick one '
        'of 10 seeds (a gentle punt of each category and the balanced build) by their objective before '
        'descending; Rotisserie starts every candidate from its one low-variance seed, so multi-starting recovers'
        ' nothing there.\n'
        '- Each loss is what one fix recovers, each including the ones before: multi-starting (descend every seed'
        " of the app's menu, keep the best after 30 iterations), the full seed menu (add every single and double "
        'punt at 5% of neutral, and for Rotisserie the 10 shared seeds), and iterations (2,000). App + the three '
        'losses = Best found. All values are H-scores in %.\n'
        '- Neutral start (Each Category and Most Categories; Rotisserie has no seeding): the app\'s descent from '
        'the neutral build (perturbed by 1%, since the neutral build itself is a stationary point) for the same 30 '
        'iterations. Seeding (in the app) is what the app gains over it, so Neutral start + Seeding = App. It is not '
        'part of the loss.\n'
        '- Rotisserie loses more than the head-to-head formats, but its total loss (about 0.08 H-score points) is '
        'small next to its edge over a G-score field (11.9% against a break-even of 8.3%, about 3.6 points; see '
        'H-scoring vs a G-score field). More iterations or multi-starting could recover some of it, but likely '
        'not enough to be worth their computational cost when the app is already this close to the best build '
        'found.\n'
        '- Boards: the first pick (round 0, against the predicted field) and after rounds 1-10 (default-ranking '
        'picks), seat Drafter 1, top 8 candidates, 2021-22 to 2025-26. The cost rows give each fix as a multiple '
        "of the app's work.",
    'Peakedness':
        'Peaks: how many distinct end points the candidate has. Second peak: how far below the best the next one '
        "is. Starts reaching best: how many of the starts end on the best. App's seed: the start the app's own "
        'policy takes (Each Category and Most Categories: its pick before descending; Rotisserie: the low-'
        'variance seed), and whether its long descent reaches the best (if not, the app is on a lower peak '
        'however long it runs). Same boards and candidates as App loss.',
    'Where the app loses to the optimum':
        '- The app takes the most promising of its 10 seeds (a gentle punt of each category and the balanced '
        'build) and descends for 30 iterations. Seeding is the one fix the app already makes: Neutral start is '
        'the same descent from the neutral build (perturbed by 1%, since the neutral build itself is a stationary'
        ' point), and Neutral start + Seeding = App. It is not part of the loss.\n'
        "- All values are H-scores in %. Each board's players are in App order; an arrow by Optimal shows how "
        'many places one would move if ranked by optimal values. The cost row gives each fix as a multiple of the'
        " app's work (each includes the ones before; branch and bound's is its measured time per candidate).\n"
        '- Boards after rounds 1-10 (default-ranking picks), seat Drafter 1, top 8 candidates, Each Category, '
        '2021-22 to 2025-26.\n'
        "- Differences from the real app: the volume scale and the position term are held at the app's solution "
        '(the app re-optimises both as it descends); no regularization (lambda_c = lambda_p = 0); the descent is '
        "a re-implementation of the app's (Adam, same learning rate and update) on this fixed problem. Full "
        'assumptions: experiment_harnesses/optimum_loss_decomposition.py.',
    'How single-peaked are H-scores':
        '- A seed reaches the optimum when the best point its descent scores is within 0.001% of it. Each cell is'
        ' the share of candidates whose start there reaches the optimum. Hovering shows how often the seed is the'
        ' one chosen, by seeding (the best objective before any descent, as the app picks) and by multi-starting '
        "(the best result after the app's iteration budget).\n"
        "- The app's menu: one gentle punt per category (that weight at 50% of neutral) and the balanced build. "
        "The full menu: the app's, plus every single and double punt at 5% of neutral (the diagonal is the single"
        ' punt). Choices are made within each menu.\n'
        "- The candidates are the loss section's; the optimum is its certified one. No regularization (lambda_c ="
        ' lambda_p = 0).',
    'Conditioning around the optimum':
        '- In the spiral-in range a direction shrinks by exactly sqrt(0.9) = 0.949 per iteration whatever its '
        "curvature (Adam's momentum is 0.9): the distance halves every 13 iterations, and the H-score gap, which "
        'goes with the distance squared, every 7.\n'
        '- The curvature is measured once, at the optimum (the Hessian along the simplex, by finite differences);'
        " Adam's step sizes change every iteration. After each of the app's 30 iterations (from the seed its "
        "seeding chooses), each of the 8 directions' step x curvature is computed; the chart shows the minimum, "
        'average and maximum over every direction of every candidate.\n'
        "- The candidates are the loss section's; the optimum is its certified one. No regularization (lambda_c ="
        ' lambda_p = 0).',
    'Loop gain at the optimum':
        'At the peak the loop closes (weight vs value share gap ~0). At 1 or above a nudge would run away. The '
        'largest Hessian eigenvalue checks the same thing directly (negative at a peak). Same candidates as the '
        'loss section. No regularization (lambda_c = lambda_p = 0).',
    'H-scoring vs a G-score field (auction)':
        '- The H seat is the app\'s autodrafter: it nominates its best available player that fits its roster and '
        'values players by its "Your $". A G-score bidder values a player at the auction table\'s G-score "Gnrc. $" '
        '(the league\'s G-score dollars for the players still available, plus its own share of the cash it could not '
        'otherwise use) and nominates its best G-score player that fits its roster. Both open at $1, and the bidding is '
        'the browser\'s.\n'
        '- S1-S12: the H seat\'s board position (nomination order). $200 a team.\n'
        '- Everyone bids on the season\'s actual statistics: the autodrafter\'s own-view noise is off, SAVOR is off, '
        'beth = 0 and psi = 0, and the H seat\'s opponent model is off head to head (Rotisserie pins it on), as in '
        'the draft experiment. Real weeks as there.',
    'Draft timing':
        '- Session build: a cold build (the agent and player-pool caches cleared; the season\'s data already in '
        'memory, so Snowflake\'s network time is not counted), three times.\n'
        '- Pick: every pick evaluates the picking team\'s first batch of 100 candidates, as the browser does; each '
        'pick takes the top candidate. Whole board: every batch of the first team\'s evaluate, at its picks.\n'
        '- 2024-25 historical stats, 12 teams x 13, the app\'s default settings. The chart nests each function under '
        'its caller; a bar\'s width is its average time per request, the same function the same colour everywhere. '
        'Time a function spends outside the traced functions is the gap under it.',
    'Auction timing':
        '- Session build: as in Draft timing. Nomination: the nominator\'s choice and every team\'s valuation (every '
        'team an autodrafter). Auction table: the first team\'s full evaluate after each round of sales.\n'
        '- 2024-25 historical stats, 12 teams x 13, $200 a team, the app\'s default settings.',
    'Pool-size calibration':
        '- Tilt: a category vector in X-scores (the units of the model\'s covariance). Realised: among the players '
        'not yet drafted at the later pick of the recorded draft, the best by the decision\'s weights minus the best by '
        'generic value v (no positions, as in the model). The decision\'s weights are held fixed, as the model holds '
        'them.\n'
        '- Effective M: the grid M with the smallest mean squared error over categories and pairs, pooled as shown. '
        'Grid: 2 to 400.\n'
        '- Explained: 1 - (mean squared error) / (mean square of the realised tilt), the share of the realised tilt\'s '
        'square the prediction removes; 0 is no better than predicting no tilt.\n'
        '- Effective M for all remaining draft picks: by the round of the current pick (the one the weights were made '
        'for), fitting one M to all of that team\'s remaining picks, as the app applies one M to them all. Effective M '
        'for a single future pick: by the round of the future pick the weights are tested at (from current picks in '
        'any earlier round).\n'
        '- H-scoring field: twelve H-scoring seats, opponent-model confidence 0.5 (Rotisserie 1), beth = 0, psi = 0.\n'
        '- G-score field: twelve seats each taking the best G-score player its roster can place (the H vs G '
        'experiment\'s G drafters). The weights are those each seat\'s own H-scoring session puts behind its top '
        'candidate at that pick, at the H vs G conventions (opponent model off, beth = 0, psi = 0); it does not pick.',
    'Category weights by round':
        '- Weights in display units: 100 = neutral (the weight of generic value), lower = de-emphasised.\n'
        '- A team\'s last pick has no weights (no picks follow it), so the rounds run 1 to 12.\n'
        '- Every pick of every seat, both orders, every season of the draft-order drafts.',
    'Category weight correlations':
        '- Weights: every pick with weights (rounds 1 to 12, every seat, both orders, every season), pooled.\n'
        '- Players: the correlation of the covariance the future-pick model uses (L: X-scores of the draftable pool, '
        'each position weighted by its share of the roster slots), averaged over the seasons.\n'
        '- Both in the categories\' scoring direction (a higher turnover score is fewer turnovers).',
    'Best player-seasons':
        '- H-score: each player\'s first-pick H-score on the neutral board of his season (12 teams x 13, the app\'s '
        'default settings but beth = 0, as the season\'s statistics are the truth). Rotisserie\'s is a league-win '
        'probability.\n'
        '- Half and Half (50/50): head to head, the objective half Each Category and half Most Categories.\n'
        '- G-score: the nine-category G-score total (the head-to-head objectives share it).\n'
        '- Ranks are all-time, among every season\'s player-seasons.\n'
        '- Every season from 2000-01 (earlier seasons have no reliable positions), the shortened seasons left out.',
    'Best player-seasons, pre-2000':
        '- H-score: each player\'s first-pick H-score on the neutral board of his season (12 teams x 13, the app\'s '
        'default settings but beth = 0, as the season\'s statistics are the truth). Rotisserie\'s is a league-win '
        'probability.\n'
        '- Half and Half (50/50): head to head, the objective half Each Category and half Most Categories.\n'
        '- G-score: the nine-category G-score total (the head-to-head objectives share it).\n'
        '- Ranks are all-time within these seasons.\n'
        '- Every season from 1984-85 to 1999-00, the shortened 1998-99 left out.\n'
        '- Known data fault: the source merges some same-name players into one (e.g. Eddie Johnson, 1984-85 and '
        '1985-86; Charles Jones; Charles Smith), adding their games together (up to 153), which inflates their values.',
    'Smooth vs simple scaling':
        "Setup: complete auctions of 12 teams, $200 each, 13 players a team, Each Category, on each season's "
        'actual statistics. Each pair of auctions puts alternate seats on each rule, then swaps them. Every '
        'drafter values players through its own noisy view of them; SAVOR off and beth = 0. Own view: each '
        "drafter's H-score through its own expectation, which is what it maximises; actual: on the true "
        "statistics. Paid extra: a sale's price above what the same bidding reaches without any cash share.",
    'Dollar shape: Gini by SAVOR settings':
        '- The Gini coefficient of the drafted players\' dollars (the top 156 of the starting board, 12 teams x $200, '
        'Each Category): 0 when every player costs the same, towards 1 as the dollars concentrate in a few players.\n'
        '- Each cell is averaged over the seasons. The outlined cell is the app\'s default (S_R $2, S_sigma 10).',
    'Dollar inequality by season':
        '- The Gini coefficient of the top 156 players\' dollars on the starting board (12 teams x $200, Each '
        'Category, the app\'s default settings): 0 when every player costs the same, towards 1 as the dollars '
        'concentrate in a few players.\n'
        '- Each season is priced on its own statistics and weekly variances. Shortened seasons are left out.',
    'Dollar curves by league size':
        '- Each Category, $200 a team, the app\'s default settings (SAVOR S_R $2, S_sigma 10). Team counts at the '
        'default roster of 13; roster sizes 13-16 (the sizes the app has slot layouts for) at the default 12 teams.\n'
        '- Each curve averages the seasons. Shortened seasons are left out.',
    'Dollar shape: replacement level and SAVOR':
        "- Vanilla SAVOR treats a player's outcome as normal around his dollar value, with standard deviation S "
        '(the SAVOR noise), and counts every outcome above $0. A player who ends up worth a dollar might be used '
        'as a streaming slot instead, so this version raises the cutoff to a replacement threshold: a player '
        "counts only outcomes above it, and is worth his expected value above the threshold minus a flyer's (a $0"
        ' player worth something only through his chance of clearing it). A threshold of $0 is vanilla SAVOR; '
        'with no noise, a player is worth his dollar value above the threshold.\n'
        "- The board: 12 teams x $200, Each Category. Each cell is the change in percent of the average player's "
        'dollars in the top 10%, the middle 80% and the bottom 10% of the drafted players (by value), against '
        "both off, which shows the dollars themselves. The charts show each rank's change.",
}


def rebuild_report_page(base_directory):
    """Render experiment_report.html from every stored section, with the explanations as the code has them now (a
    section measured before its explanation was reworded shows the current wording). Returns (page path, sections)."""
    import os as _os
    sections = _load_stored_sections(_os.path.join(base_directory, 'experiment_results'))
    for data in sections:
        data['explanation'] = _SECTION_EXPLANATIONS.get(data['title'], data['explanation'])
        data['details'] = _SECTION_DETAILS.get(data['title'], '')
    page = render_report_page(
        sections
        , 'Measurements for human judgment: this tier has no exact right answers. '
          'Each tab regenerates when its test runs in report mode (RUN_EXPERIMENTS=1) and keeps its own timestamp.'
    )
    page_path = _os.path.join(base_directory, 'experiment_report.html')
    with open(page_path, 'w', encoding='utf-8') as page_file:
        page_file.write(page)
    return page_path, sections


def write_experiment_report_files(base_directory):
    """Persist the sections recorded to since they were last saved, rebuild the tabbed page from everything stored.
    Returns the page path."""
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
    archive_dir = _os.path.join(base_directory, 'experiment_archive', _RUN_ARCHIVE_NAME)
    _os.makedirs(archive_dir, exist_ok=True)

    for section in sorted(_SECTIONS_TO_SAVE):
        table = _REPORT[section]
        payload = {
            'title':       section,
            'explanation': _SECTION_EXPLANATIONS.get(section, ''),
            'columns':     table['columns'],
            'rows':        table['rows'],
            'generated':   stamp,
        }
        stored_path = _os.path.join(reports_dir, f'{_section_slug(section)}.json')
        merge_columns = _MERGE_KEYS.get(section, ('Format',))
        if all(name in table['columns'] for name in merge_columns) and _os.path.exists(stored_path):
            merge_other_formats(payload, stored_path, stamp, merge_columns)
        with open(_os.path.join(archive_dir, f'{_section_slug(section)}.json'), 'w',
                  encoding='utf-8') as archive_file:
            json.dump(payload, archive_file, indent=2)
        with open(_os.path.join(reports_dir, f'{_section_slug(section)}.json'), 'w',
                  encoding='utf-8') as stored_file:
            json.dump(payload, stored_file, indent=1)
    _SECTIONS_TO_SAVE.clear()
    page_path, sections = rebuild_report_page(base_directory)

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


@pytest.fixture(autouse=True)
def persist_sections_after_each_experiment():
    """Report mode: write the sections an experiment recorded as soon as it finishes, pass or fail, and rebuild the
    report page, so finished results are on disk while the rest of the run continues."""
    yield
    if _RUN_SIMS and _SECTIONS_TO_SAVE:
        page_path = write_experiment_report_files(os.path.dirname(__file__))
        print(f'\n(saved; report page {page_path})')


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
        _record_row('Rotisserie punting (hard <15% / soft 15-40% expected win rate)',
                    ['Season', 'Mode', 'Hard', 'Soft', 'Weakest category', 'Weakest rate', 'Player'],
                    [season, mode, hard_count, soft_count, weakest_cat,
                     f'{100 * float(np.min(weakest_rates)):.0f}%',
                     session.player_registry[weakest_player].name])


# ── FAST: early-pick stability (draft + auction, per season) ─────────────────

def _snapshot_candidates(result, top_n=15):
    return {
        c.player_id: (c.h_score, np.asarray(c.category_weights, dtype=float),
                      np.asarray(c.win_rates, dtype=float))
        for c in result.candidates[:top_n]
    }


PICK_STABILITY_SECTIONS = {'draft': 'Draft pick stability', 'auction': 'Auction pick stability'}


@pytest.mark.parametrize('auction', [False, True], ids=['draft', 'auction'])
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_early_pick_stability(sessions, format_key, auction):
    """An opponent drafting its PREDICTED first player confirms the field rather than moving it, so the
    displayed values and builds of the remaining candidates must stay (nearly) put relative to each
    other — the anti-lurch guarantee."""
    mode = 'auction' if auction else 'draft'
    breaches = []
    for season in _SEASONS:
        session = sessions(format_key, auction, season)
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
        # What a user would see change: each candidate's place among the compared ones, before and after, and the
        # mean over the categories of how far its win rates move (percentage points).
        place_before = {name: place for place, name in enumerate(sorted(common, key=lambda n: -before[n][0]))}
        place_after = {name: place for place, name in enumerate(sorted(common, key=lambda n: -after[n][0]))}
        rank_moves = [abs(place_before[name] - place_after[name]) for name in common]
        category_changes = [float(np.abs(after[name][2] - before[name][2]).mean()) for name in common]

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
        _record_row(PICK_STABILITY_SECTIONS[mode],
                    ['Season', 'Format', 'Shared shift (H-score)', 'Max relative move (H-score)',
                     'Biggest mover', 'Max dweight', 'Punt flips', 'Worst punt flip', 'Players compared',
                     'Every relative move (H-score)', 'Every rank move', 'Every category change'],
                    [season, format_key, f'{level_shift:+.2f}', f'{max(h_deltas.values()):.2f}',
                     session.player_registry[worst_h_name].name, f'{max(w_deltas.values()):.1f}',
                     len(punt_flips), flip_text, len(common),
                     ' '.join(f'{delta - level_shift:+.2f}' for delta in raw_deltas.values()),
                     ' '.join(str(move) for move in rank_moves),
                     ' '.join(f'{change:.2f}' for change in category_changes)])
    _fail_on_breaches(breaches)


# ── FAST: self-play convergence (per season) ─────────────────────────────────

_SELF_PLAY_CHECKPOINT_EVERY = 10
# The weight-movement table summarises the per-iteration changes over blocks of this many iterations.
_SELF_PLAY_MOVEMENT_BLOCK = 10


def _record_self_play_descent(agent, n_iterations):
    """Run the agent's populate, recording self-play's joint descent. After every iteration, per universe player, the
    category weights of the start the field adopts for it, in the app's display units (each category's share over its
    neutral share, x100); and every _SELF_PLAY_CHECKPOINT_EVERY iterations (and at the last), the expected category win
    rates of that start. Returns ([(iteration, players x categories frame of display weights)],
    [(iteration, players x categories frame of win rates)])."""
    import pandas as pd
    from backend.math import algorithm_agents
    weights_by_iteration, win_rates_at_checkpoints = [], []
    latest = {}
    neutral = agent.v.reshape(-1) / agent.v.sum()
    original_objective = type(agent).get_objective_and_gradient
    original_update = algorithm_agents._SelfPlayField.update_best_starts

    def recording_objective(*arguments, **keyword_arguments):
        result = original_objective(agent, *arguments, **keyword_arguments)
        if agent._self_play_field is not None:
            latest.update(iteration=arguments[9], result=result, weights=np.array(arguments[0], dtype=float))
        return result

    def recording_update(field, scores):
        best_rows = original_update(field, scores)
        iteration, result = latest['iteration'], latest['result']
        weights = latest['weights'][best_rows]
        display = 100 * (weights / weights.sum(axis=1, keepdims=True)) / neutral
        weights_by_iteration.append((iteration + 1, pd.DataFrame(display, index=field.players,
                                                                 columns=agent.x_scores.columns)))
        if ((iteration + 1) % _SELF_PLAY_CHECKPOINT_EVERY == 0
                or iteration == algorithm_agents._SELF_PLAY_ITERATIONS - 1):
            win_rates = np.asarray(result['CDF-Estimates']).mean(axis=2)[best_rows]
            win_rates_at_checkpoints.append((iteration + 1, pd.DataFrame(win_rates, index=field.players,
                                                                         columns=agent.x_scores.columns)))
        return best_rows

    agent.get_objective_and_gradient = recording_objective
    algorithm_agents._SelfPlayField.update_best_starts = recording_update
    try:
        agent.populate_default_h_scores(n_iterations)
    finally:
        del agent.get_objective_and_gradient   # drop the instance shadow, restoring the class method
        algorithm_agents._SelfPlayField.update_best_starts = original_update
    return weights_by_iteration, win_rates_at_checkpoints


def summarize_weight_movement(weights_by_iteration, tracked):
    """Per block of _SELF_PLAY_MOVEMENT_BLOCK iterations (1-10, 11-20, ...: the blocks end on the punt-count
    checkpoints), two views of how far the tracked players' display weights move (Euclidean norm per player):
      - per iteration: the mean over the block's iterations of the mean (over the players) change from the previous
        iteration, the largest single player change in the block, and that player;
      - drift: each player's weights averaged over the block against its weights averaged over the previous block, the
        mean over the players and the largest. Back-and-forth between the same builds has the same average in both
        blocks and cancels; only movement of the average -- steady drift -- shows. None for the first block.
    Returns [(first iteration, last iteration, per-iteration mean, per-iteration max, that player, drift mean, drift
    max)]."""
    weights = {iteration: frame.loc[tracked].to_numpy() for iteration, frame in weights_by_iteration}
    iterations = sorted(weights)
    blocks, previous_average = [], None
    for first in range(iterations[0], iterations[-1] + 1, _SELF_PLAY_MOVEMENT_BLOCK):
        block = [iteration for iteration in iterations if first <= iteration < first + _SELF_PLAY_MOVEMENT_BLOCK]
        changes = np.vstack([np.linalg.norm(weights[iteration] - weights[iteration - 1], axis=1)
                             for iteration in block if iteration - 1 in weights])
        _, worst_player = np.unravel_index(np.argmax(changes), changes.shape)
        block_average = np.mean([weights[iteration] for iteration in block], axis=0)
        drift = (None if previous_average is None
                 else np.linalg.norm(block_average - previous_average, axis=1))
        blocks.append((block[0], block[-1], float(changes.mean(axis=1).mean()), float(changes.max()),
                       tracked[worst_player], None if drift is None else float(drift.mean()),
                       None if drift is None else float(drift.max())))
        previous_average = block_average
    return blocks


@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_self_play_convergence(sessions, format_key):
    """Show self-play's descent directly, for every format, one row per block of 10 iterations: how far the tracked
    top players' category weights move per iteration, how far their block averages drift, and how many of their current
    starts punt each category at the block's end. Convergence = movement and drift shrinking toward zero and the counts
    settling; oscillation = movement that stays large; wandering = drift that stays large. A second run doubles the
    iterations, to show how the process would continue past the app's stopping point."""
    from backend.math import algorithm_agents
    app_iterations = algorithm_agents._SELF_PLAY_ITERATIONS
    for season in _SEASONS:
        session = sessions(format_key, False, season)
        n_iterations = session.current_settings['n_iterations']
        weights, checkpoints = _record_self_play_descent(session.agent, n_iterations)
        # The extended run: a fresh session (so the shared one keeps the app's field), populated again with twice the
        # iterations. Self-play is deterministic, so its first iterations repeat the app's.
        extended_session = _build_session(_FORMATS[format_key], season=season)
        algorithm_agents._SELF_PLAY_ITERATIONS = 2 * app_iterations
        try:
            extended_weights, extended_checkpoints = _record_self_play_descent(extended_session.agent, n_iterations)
        finally:
            algorithm_agents._SELF_PLAY_ITERATIONS = app_iterations

        categories = session.current_settings['categories']
        short      = [_SHORT_CATEGORY.get(category, category) for category in categories]
        # Track a FIXED set of players across iterations and both runs so the measures are comparable: the top 12 of
        # the served board that self-play descends (the universe).
        universe = checkpoints[0][1].index
        tracked = [player for player in session.agent.default_h_scores.index if player in universe][:12]

        for run, run_weights, run_checkpoints in ((f'app ({app_iterations} iterations)', weights, checkpoints),
                                                  (f'doubled ({2 * app_iterations} iterations)', extended_weights,
                                                   extended_checkpoints)):
            win_rates_at = dict(run_checkpoints)
            for (first, last, mean_change, max_change, player, drift_mean,
                 drift_max) in summarize_weight_movement(run_weights, tracked):
                if last not in win_rates_at:
                    raise RuntimeError(f'{season} {run}: no punt-count checkpoint at iteration {last}, the end of a block.')
                tracked_rates = win_rates_at[last].loc[tracked].to_numpy()
                assert not np.isnan(tracked_rates).any(), f'{season} {run} iteration {last}: NaN win rates'
                punt_counts = (tracked_rates < _SOFT_PUNT_RATE).sum(axis=0)
                _record_row('Self-play convergence',
                            ['Season', 'Format', 'Run', 'Iterations', 'Mean change', 'Max change', 'Max player',
                             'Mean drift', 'Max drift'] + short,
                            [season, format_key, run, f'{first}-{last}', f'{mean_change:.2f}', f'{max_change:.1f}',
                             session.player_registry[player].name,
                             '—' if drift_mean is None else f'{drift_mean:.2f}',
                             '—' if drift_max is None else f'{drift_max:.1f}']
                            + [int(count) for count in punt_counts])


# ── FAST: self-play's served board against its own field (per season) ────────
# Both checks read only the served board and the stored field (experiment_harnesses/self_play_checks), so they hold
# whatever process built them.

_BEST_RESPONSE_TOP = 12
_BEST_RESPONSE_ITERATIONS = 300


@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_self_play_optimality(sessions, format_key):
    """Each top served build should be its player's best response to the field it was served on: no long descent
    from any start -- the served build, the balanced build, a gentle punt of each category -- should beat it."""
    from experiment_harnesses.self_play_checks import measure_best_response_shortfalls
    for season in _SEASONS:
        session = sessions(format_key, False, season)
        categories = session.current_settings['categories']
        short = [_SHORT_CATEGORY.get(category, category) for category in categories]
        checks = measure_best_response_shortfalls(session, _BEST_RESPONSE_TOP, _BEST_RESPONSE_ITERATIONS)

        def describe(win_rates):
            punted = [f'{short[i]} {100 * rate:.0f}%' for i, rate in enumerate(win_rates) if rate < _SOFT_PUNT_RATE]
            return ', '.join(punted) or '(no punts)'

        for check in checks:
            _record_row('Self-play optimality',
                        ['Season', 'Format', 'Player', 'Served H', 'Best H', 'Shortfall', 'Best from',
                         'Served punts', 'Best punts'],
                        [season, format_key, session.player_registry[check.player_id].name,
                         f'{check.served_h_score:.2f}', f'{check.best_h_score:.2f}', f'{check.shortfall:.2f}',
                         check.best_start, describe(check.served_win_rates), describe(check.best_win_rates)])


@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_serve_stability(sessions, format_key):
    """The field should describe the board it serves: the seats are the top players, the seated players' served builds
    are the builds the field assumes for them, and the top players average close to 50% per category against each
    other at C = 1 (win probabilities are pairwise complementary; below 1 the field discounts opponents' punts, so the
    averages drift from 50% by design). Recorded at the app's opponent-model confidence, and for head to head also at
    C = 1 (Rotisserie always runs at 1)."""
    from experiment_harnesses.self_play_checks import measure_field_consistency, measure_serve_changes
    confidences = [None] if format_key == 'Roto' else [None, 1.0]
    for season in _SEASONS:
        for confidence in confidences:
            overrides = {} if confidence is None else {'opponent_model_confidence': confidence}
            session = sessions(format_key, False, season, **overrides)
            categories = session.current_settings['categories']
            short = [_SHORT_CATEGORY.get(category, category) for category in categories]
            consistency = measure_field_consistency(session, session.current_settings['n_drafters'])
            unseated = [session.player_registry[player].name for player in consistency.top_players
                        if player not in consistency.seated_players]
            worst = int(np.argmax(np.abs(consistency.tilt_gap_by_category)))
            changes = measure_serve_changes(session)
            _record_row('Serve stability',
                        ['Season', 'Format', 'C', 'Top 12 unseated', 'Mean |tilt gap|', 'Worst category gap']
                        + [f'{name} (field)' for name in short]
                        + ['Every seat gap', 'Every rank move', 'Every category change'],
                        [season, format_key, f'{session.agent.opponent_model_confidence:g}',
                         ', '.join(unseated) or '(none)', f'{consistency.mean_absolute_tilt_gap:.3f}',
                         f'{short[worst]} {consistency.tilt_gap_by_category[worst]:+.3f}']
                        + [f'{100 * rate:.0f}%' for rate in consistency.seated_category_win_rates]
                        + [' '.join(f'{gap:.3f}' for gap in consistency.seat_gaps),
                           ' '.join(str(abs(served - self_play)) for served, self_play
                                    in zip(changes.served_ranks, changes.self_play_ranks)),
                           ' '.join(f'{change:.2f}' for change in changes.category_changes)])


# ── SLOW simulation properties ────────────────────────────────────────────────

sims = pytest.mark.skipif(not _RUN_SIMS, reason='full-draft simulation; set RUN_EXPERIMENTS=1')


def _draft_h_seat_in_g_field(h_session, seat, candidate_limit=40):
    """One snake draft: `seat` drafts with the H agent, everyone else picks by G-score ranking.
    Returns the H seat's final-roster H-score."""
    assignments = _draft_h_seat_rosters(h_session, seat, candidate_limit)
    scores = h_session.agent.get_h_scores(assignments, f'Drafter {seat + 1}',
                                          h_session.current_settings['n_iterations'])['Scores']
    return float(scores[scores.idxmax()])


def _draft_h_seat_rosters(h_session, seat, candidate_limit=40):
    """One snake draft: `seat` drafts with the H agent, everyone else picks by G-score ranking.
    Returns every team's roster, {'Drafter i': [player id, ...]}."""
    from simulate import _gscore_ranking, _pick_gscore_player, _has_position_data   # season_simulation

    agent        = h_session.agent
    agent.reset_draft_state()
    n_drafters   = h_session.current_settings['n_drafters']
    n_picks      = h_session.current_settings['n_picks']
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
    return assignments


AUCTION_H_VERSUS_G_SECTION = 'H-scoring vs a G-score field (auction)'


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_h_scoring_beats_g_field_in_auctions(format_key):
    """The auction counterpart of test_h_scoring_beats_g_field: per season, one auction with the H-scoring autodrafter
    at each board position against eleven G-score bidders; its H-score and (head to head) real-week win rate.
    Measurements only, apart from every auction completing."""
    from experiment_harnesses.auction_h_versus_g import run_h_versus_g_auctions_in_parallel
    n_seats = _SIM_SEATS or 12
    results, failures = run_h_versus_g_auctions_in_parallel(format_key, _SEASONS, n_seats, _EXPERIMENT_WORKERS)
    columns = ['Format', 'Season', 'Mean H-score', 'Mean real win rate'] + _seat_columns(n_seats) + ['Every real win rate']
    pooled, pooled_real = [], []
    for season in _SEASONS:
        by_seat = sorted((result for result in results if result['season'] == season), key=lambda result: result['h_seat'])
        if len(by_seat) != n_seats:
            continue
        scores = [result['h_score'] for result in by_seat]
        real_rates = [result['real_rate'] for result in by_seat if result['real_rate'] is not None]
        pooled.extend(scores)
        pooled_real.extend(real_rates)
        _record_row(AUCTION_H_VERSUS_G_SECTION, columns,
                    [format_key, season, f'{np.mean(scores):.1f}', f'{np.mean(real_rates):.1f}' if real_rates else '—']
                    + [f'{score:.1f}' for score in scores] + [' '.join(f'{rate:.1f}' for rate in real_rates)])
    _record_row(AUCTION_H_VERSUS_G_SECTION, columns,
                [format_key, f'ALL ({len(pooled)} auctions)', f'{np.mean(pooled):.1f}',
                 f'{np.mean(pooled_real):.1f}' if pooled_real else '—'] + ['—'] * n_seats + [''])
    _fail_on_breaches([f'{failure["job"]} failed:\n{failure["traceback"]}' for failure in failures])


# Random real weeks per opponent in the H vs G real-matchup simulation.
_REAL_MATCHUP_SIMULATIONS = 1000


def _seat_columns(n_seats):
    return [f'S{i + 1}' for i in range(n_seats)]


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_h_scoring_beats_g_field(format_key):
    """The most basic utility claim: an H-scoring drafter in a league of G-score drafters should win
    more than it loses, in every season."""
    from experiment_harnesses.real_matchups import load_season_weeks, simulate_real_win_rates
    pooled, pooled_real = [], []
    n_seats = None
    # Head to head, the real counterpart of the H-score: Each Category's expected category win rate, Most Categories'
    # matchup win probability. Rotisserie has none: one real season is one draw of the standings.
    real_measure = {'EC': 'category', 'MC': 'matchup'}.get(format_key)
    columns = (['Format', 'Season', 'Mean H-score', 'Mean real win rate'] + _seat_columns(_SIM_SEATS or 12)
               + ['Every real win rate'])
    for season in _SEASONS:
        # beth=0 and opponent_model_confidence=0: the H-vs-G conventions. Historical stats are
        # objectively correct (no shrinkage), and modeling G-drafters as punt-pursuers is a
        # knowably wrong belief. psi=0 (upsilon 1): a missed game counts as nothing and nobody replaces it, the
        # assumption the real-week simulation reproduces exactly. The canonical baseline (testing_files/baselines/
        # h_vs_g_c0_baseline.jsonl) was recorded at psi 0.8, so it is no longer comparable.
        session = _build_session(_FORMATS[format_key], season=season, beth=0,
                                 opponent_model_confidence=0, psi=0)
        n_seats = _SIM_SEATS or session.current_settings['n_drafters']
        assert len(_seat_columns(n_seats)) == len(columns) - 5, 'the league size changed between seasons'
        season_weeks = load_season_weeks(season) if real_measure is not None else None
        scores, real_rates = [], []
        for seat in range(n_seats):
            assignments = _draft_h_seat_rosters(session, seat)
            team = f'Drafter {seat + 1}'
            h_scores = session.agent.get_h_scores(assignments, team,
                                                  session.current_settings['n_iterations'])['Scores']
            scores.append(100 * float(h_scores[h_scores.idxmax()]))
            if real_measure is not None:
                rates = simulate_real_win_rates(season_weeks, assignments, team, session.current_settings['categories'],
                                                _REAL_MATCHUP_SIMULATIONS, seed=seat)
                real_rates.append(100 * rates[real_measure])
        pooled.extend(scores)
        pooled_real.extend(real_rates)
        _record_row('H-scoring vs a G-score field', columns,
                    [format_key, season, f'{np.mean(scores):.1f}',
                     f'{np.mean(real_rates):.1f}' if real_rates else '—'] + [f'{s:.1f}' for s in scores]
                    + [' '.join(f'{rate:.1f}' for rate in real_rates)])
    _record_row('H-scoring vs a G-score field', columns,
                [format_key, f'ALL ({len(pooled)} drafts)', f'{np.mean(pooled):.1f}',
                 f'{np.mean(pooled_real):.1f}' if pooled_real else '—'] + ['—'] * n_seats + [''])


def measure_awareness_vs_unaware_h_field(format_key):
    """Awareness against a field of UNAWARE H-drafters. ONE AGENT PER DRAFTER: a shared
    field agent cross-contaminates its seats (an inference from seat A's perspective can overwrite seat
    B's own team entry, so B would be modelled from A's view of B); every seat drafts with its own
    session."""
    from self_play import draft_population, list_snake_order

    section = 'Awareness (opponent model on vs off)'
    all_gains = []
    n_seats = None
    for season in _SEASONS:
        reference = _build_session(_FORMATS[format_key], season=season, opponent_model_confidence=0, beth=0)
        n_drafters   = reference.current_settings['n_drafters']
        n_picks      = reference.current_settings['n_picks']
        n_iterations = reference.current_settings['n_iterations']
        n_seats      = _SIM_SEATS or n_drafters

        field_sessions = [reference] + [
            _build_session(_FORMATS[format_key], season=season, opponent_model_confidence=0, beth=0)
            for _ in range(n_drafters - 1)
        ]
        aware_deviator = _build_session(_FORMATS[format_key], season=season,
                                        opponent_model_confidence=0.5, beth=0)

        def deviator_score(session_by_seat, seat):
            assignments = draft_population(session_by_seat, n_drafters, list_snake_order(n_drafters, n_picks), 40)
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
                    ['Comparison', 'Format', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                    ['vs unaware H field', format_key, season, f'{np.mean(gains):+.2f}', f'{season_se:.2f}']
                    + [f'{g:+.2f}' for g in gains])

    pooled_mean = float(np.mean(all_gains))
    pooled_se   = float(np.std(all_gains, ddof=1) / np.sqrt(len(all_gains)))
    _record_row(section,
                ['Comparison', 'Format', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                ['vs unaware H field', format_key, f'ALL ({len(all_gains)} drafts)', f'{pooled_mean:+.2f}',
                 f'{pooled_se:.2f}'] + ['—'] * n_seats)


def measure_awareness_vs_g_field(format_key):
    """Awareness against a G field: the model mispredicts it (G drafters do not punt), so this measures what that
    misprediction costs."""
    section = 'Awareness (opponent model on vs off)'
    all_gains = []
    n_seats = None
    for season in _SEASONS:
        aware   = _build_session(_FORMATS[format_key], season=season,
                                 opponent_model_confidence=0.5, beth=0)
        unaware = _build_session(_FORMATS[format_key], season=season,
                                 opponent_model_confidence=0, beth=0)
        n_seats = _SIM_SEATS or aware.current_settings['n_drafters']
        gains = []
        for seat in range(n_seats):
            gains.append(100 * (_draft_h_seat_in_g_field(aware, seat)
                                - _draft_h_seat_in_g_field(unaware, seat)))
        all_gains.extend(gains)
        season_se = float(np.std(gains, ddof=1) / np.sqrt(len(gains)))
        _record_row(section,
                    ['Comparison', 'Format', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                    ['vs G field', format_key, season, f'{np.mean(gains):+.2f}', f'{season_se:.2f}']
                    + [f'{g:+.2f}' for g in gains])

    pooled_mean = float(np.mean(all_gains))
    pooled_se   = float(np.std(all_gains, ddof=1) / np.sqrt(len(all_gains)))
    _record_row(section,
                ['Comparison', 'Format', 'Season', 'Mean (pp)', 'SE (pp)'] + _seat_columns(n_seats),
                ['vs G field', format_key, f'ALL ({len(all_gains)} drafts)', f'{pooled_mean:+.2f}',
                 f'{pooled_se:.2f}'] + ['—'] * n_seats)


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC'])
def test_awareness(format_key):
    """The opponent model on vs off, against both fields, measured together so the page is saved whole. Rotisserie pins
    the model on, so it has no 'off' to compare."""
    measure_awareness_vs_unaware_h_field(format_key)
    measure_awareness_vs_g_field(format_key)


# ── Exact optimum (Each Category) ─────────────────────────────────────────────
# One measurement serves two sections (the loss decomposition below and loop gain): for each season, the boards after each of _EXACT_ROUNDS snake rounds (picks in
# default-ranking order), my seat Drafter 1, the top EXPERIMENT_EXACT_CANDIDATES candidates of each board, at app
# settings with no regularization anywhere (lambda_c = lambda_p = 0, the self-play field included): the two sections
# study the unregularized objective, and the app is scored on that same problem. experiment_harnesses/exact_optimum measures each board in its own process (EXPERIMENT_WORKERS of them); the
# first of the two tests to run measures everything, and the rest read it. A board that fails is reported by every
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


# ── Where the app loses to the optimum ────────────────────────────────────────
# The exact-optimum boards and candidates again, each solved five ways on the same frozen problem
# (experiment_harnesses/optimum_loss_decomposition, whose header lists the assumptions). The same run records every
# seed of both menus, which the seed-start section reads; the first of the two tests to run measures it.

_loss_decomposition_run: dict = {}


def _decompose_loss_cases():
    """Every candidate's loss decomposition and seed measurements, measured once per run."""
    from experiment_harnesses.optimum_loss_decomposition import decompose_boards_in_parallel
    if not _loss_decomposition_run:
        _loss_decomposition_run['cases'] = decompose_boards_in_parallel(_SEASONS, _EXACT_ROUNDS, _EXACT_CANDIDATES,
                                                                        _EXPERIMENT_WORKERS)
    return _loss_decomposition_run['cases']

# Read left to right, a row adds up: App + the four losses = Optimal.
_LOSS_CAUSE_LABELS = ('Multi-starting', 'Full seed menu', 'Iterations', 'Remaining gap')
# Seeding comes before the app: it is what the app's seeding already buys over a neutral start, not a loss.
_SEEDING_LABEL = 'Seeding (in the app)'
_LOSS_COLUMNS = (['Season', 'Rd', 'Player', 'Neutral start', _SEEDING_LABEL, 'App'] + list(_LOSS_CAUSE_LABELS)
                 + ['Optimal', 'Total loss'])
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
    """The loss section's rows: one per candidate (the neutral start, what seeding gains over it, App, the four losses,
    Optimal, the total), each board's candidates
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
            rows.append([case['season'], case['rounds'], case['player'], f'{100 * case["neutral_start"]:.4f}%',
                         format_loss(steps[0] - case['neutral_start']), f'{100 * steps[0]:.4f}%']
                        + [format_loss(later - earlier) for earlier, later in zip(steps, steps[1:])]
                        + [f'{100 * steps[-1]:.4f}%{arrow}', format_loss(steps[-1] - steps[0])])
    losses = np.array([np.diff(case['steps']) for case in cases])                 # (cases, 4)
    seeding_gains = [case['steps'][0] - case['neutral_start'] for case in cases]
    rows.append([f'Mean of {len(cases)}' + (f' ({uncertified_count} optimum not certified)' if uncertified_count else ''),
                 '—', '—', '—', format_loss(float(np.mean(seeding_gains))), '—']
                + [format_loss(float(losses[:, index].mean())) for index in range(losses.shape[1])]
                + [f'{n_moved} of {len(cases)} move', format_loss(float(losses.sum(axis=1).mean()))])
    rows.append(["Cost to fix (x the app's work)", '—', '—', '—', 'in the app', '—']
                + [format_cost(cost) for cost in compute_improvement_costs(cases, branch_and_bound_seconds)] + ['—'] * 2)
    return rows


@sims
def test_where_the_app_loses_to_the_optimum():
    """The app's shortfall from the certified optimum, split into multi-starting, the full seed menu, iteration budget and the
    rest. Branch and bound certifies each optimum (the shared exact-optimum measurement)."""
    section = 'Where the app loses to the optimum'
    certifications = {(case['season'], case['rounds'], case['player']): case['certification']
                      for case in _measure_exact_optimum_cases()}
    cases = _decompose_loss_cases()
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


_SEED_MENUS = ('App menu', 'Full menu')
# The heatmaps' per-seed totals use the first seven columns; each start that misses the optimum is a 'Miss' row using
# the seed column and the last six.
_SEED_COLUMNS = ['Menu', 'Seed', 'Reaches optimum', 'Chosen by seeding', 'Chosen by multi-start', 'Worst miss (pp)',
                 'Starts', 'Season', 'Rd', 'Player', "In the app's menu", 'Shortfall (pp)', 'Seeds reaching',
                 "App's seed reaches"]
_SEED_TOTAL_PADDING = ['—'] * 7


def label_seed(seed):
    """A seed's cell: a gentle punt 'FG% at 50%', a full-menu punt 'FG%/FT%' (a single punt names one category),
    'balanced', or Rotisserie's 'low-variance'."""
    if seed['kind'] == 'gentle':
        return f'{_SHORT_CATEGORY[seed["categories"][0]]} at 50%'
    if seed['kind'] == 'punt':
        return '/'.join(_SHORT_CATEGORY[category] for category in seed['categories'])
    return {'balanced': 'balanced', 'lowvar': 'low-variance'}[seed['kind']]


def select_menu_seeds(seeds, menu):
    """The seeds in a menu. The full menu's neutral start is left out: it is the balanced build's weights again, and
    a choice between the two identical starts would always fall on the first."""
    if menu == 'App menu':
        return [seed for seed in seeds if seed['app_menu']]
    return [seed for seed in seeds if seed['kind'] != 'neutral']


def summarize_seed_starts(cases, menu):
    """One row per seed cell of a menu, pooled over the candidates, then an ALL row: the share of its starts whose
    descent at the long budget reaches the optimum (within the exact optimum's PEAK_TOLERANCE), the share of candidates whose seeding and whose
    multi-start choice fall on it, the largest shortfall, and the number of starts."""
    from experiment_harnesses.exact_optimum import PEAK_TOLERANCE
    reached, starts, worst = {}, {}, {}
    chosen_by_seeding, chosen_by_multistart = {}, {}
    for case in cases:
        optimum = case['steps'][-1]
        seeds = select_menu_seeds(case['seeds'], menu)
        for seed in seeds:
            label = label_seed(seed)
            shortfall = optimum - seed['long_budget']
            starts[label] = starts.get(label, 0) + 1
            reached[label] = reached.get(label, 0) + (shortfall <= PEAK_TOLERANCE)
            worst[label] = max(worst.get(label, 0.0), shortfall)
        seeding_label = label_seed(max(seeds, key=lambda seed: seed['pre_descent']))
        multistart_label = label_seed(max(seeds, key=lambda seed: seed['app_budget']))
        chosen_by_seeding[seeding_label] = chosen_by_seeding.get(seeding_label, 0) + 1
        chosen_by_multistart[multistart_label] = chosen_by_multistart.get(multistart_label, 0) + 1
    rows = [[menu, label, f'{100 * reached[label] / starts[label]:.1f}%',
             f'{100 * chosen_by_seeding.get(label, 0) / len(cases):.1f}%',
             f'{100 * chosen_by_multistart.get(label, 0) / len(cases):.1f}%',
             f'{100 * max(worst[label], 0.0):.4f}', starts[label]] + _SEED_TOTAL_PADDING for label in starts]
    rows.append([f'ALL ({menu}, {len(cases)} candidates)', '—',
                 f'{100 * sum(reached.values()) / sum(starts.values()):.1f}%', '—', '—',
                 f'{100 * max(max(worst.values()), 0.0):.4f}', sum(starts.values())] + _SEED_TOTAL_PADDING)
    return rows


def summarize_seed_misses(cases):
    """One 'Miss' row per start of the full menu (which holds the app's seeds) whose long descent ends more than
    PEAK_TOLERANCE short of the optimum: the candidate, the seed, whether the app's menu has it, the shortfall, how many
    of the candidate's seeds do reach the optimum, and whether the seed the app's seeding would choose does."""
    from experiment_harnesses.exact_optimum import PEAK_TOLERANCE
    rows = []
    for case in cases:
        optimum = case['steps'][-1]
        seeds = select_menu_seeds(case['seeds'], 'Full menu')
        app_choice = max(select_menu_seeds(case['seeds'], 'App menu'), key=lambda seed: seed['pre_descent'])
        reaching = sum(optimum - seed['long_budget'] <= PEAK_TOLERANCE for seed in seeds)
        for seed in seeds:
            shortfall = optimum - seed['long_budget']
            if shortfall <= PEAK_TOLERANCE:
                continue
            rows.append(['Miss', label_seed(seed), '—', '—', '—', '—', '—', case['season'], case['rounds'],
                         case['player'], 'yes' if seed['app_menu'] else 'no', f'{100 * shortfall:.4f}',
                         f'{reaching} of {len(seeds)}',
                         'yes' if optimum - app_choice['long_budget'] <= PEAK_TOLERANCE else 'no'])
    return rows


@sims
def test_seed_starts_reaching_the_optimum():
    """Which seeds of the app's menu and of the full menu climb to the certified optimum, and which get chosen."""
    section = 'How single-peaked are H-scores'
    cases = _decompose_loss_cases()
    for menu in _SEED_MENUS:
        for row in summarize_seed_starts(cases, menu):
            _record_row(section, _SEED_COLUMNS, row)
    for row in summarize_seed_misses(cases):
        _record_row(section, _SEED_COLUMNS, row)


_CONDITIONING_SECTION = 'Conditioning around the optimum'
_CONDITIONING_COLUMNS = ['Iteration', 'Min', 'Average', 'Max', 'Candidates outside the window', 'Candidates']
_conditioning_run: dict = {}


def collect_optimum_weights_by_board(cases):
    """{(season, rounds): {player: {'categories': ..., 'optimum_weights': ...}}} from the loss decomposition's cases."""
    by_board: dict = {}
    for case in cases:
        by_board.setdefault((case['season'], case['rounds']), {})[case['player']] = {
            'categories': case['categories'], 'optimum_weights': case['optimum_weights']}
    return by_board


def _measure_conditioning_cases():
    """Every candidate's step x curvature over the iterations, measured once per run, at the loss decomposition's
    optima."""
    from experiment_harnesses.optimum_loss_decomposition import measure_conditioning_boards_in_parallel
    if not _conditioning_run:
        _conditioning_run['cases'] = measure_conditioning_boards_in_parallel(
            _SEASONS, _EXACT_ROUNDS, _EXACT_CANDIDATES, _EXPERIMENT_WORKERS,
            collect_optimum_weights_by_board(_decompose_loss_cases()))
    return _conditioning_run['cases']


def summarize_conditioning(conditioning_cases):
    """One row per iteration of the app's own descent (from the seed its seeding chooses): the minimum, average and
    maximum step x curvature over every direction of every candidate, and how many candidates have a direction outside
    UNDERDAMPED_WINDOW. Every seed's descent is in the saved measurement too; the section shows the app's."""
    from experiment_harnesses.optimum_loss_decomposition import UNDERDAMPED_WINDOW
    low, high = UNDERDAMPED_WINDOW
    values = np.stack([case['app_step_scaled'] for case in conditioning_cases], axis=0)   # (candidates, iterations, directions)
    rows = []
    for iteration in range(values.shape[1]):
        at_iteration = values[:, iteration, :]
        outside = int(((at_iteration <= low) | (at_iteration >= high)).any(axis=1).sum())
        rows.append([iteration + 1, f'{at_iteration.min():.4g}', f'{at_iteration.mean():.4g}',
                     f'{at_iteration.max():.4g}', outside, values.shape[0]])
    return rows


@sims
def test_conditioning_around_the_optimum():
    """The step x curvature at the optimum over the app's own descent, every direction of every candidate."""
    for row in summarize_conditioning(_measure_conditioning_cases()):
        _record_row(_CONDITIONING_SECTION, _CONDITIONING_COLUMNS, row)


# ── Gradient descent on the actual objective (Each Category, Most Categories, Rotisserie) ───
# The app's own descent, on the app's own objective, from every start of a wide menu for a long budget; the best build
# found stands in for EC's certified optimum (experiment_harnesses/app_objective_peaks). Boards are measured once per
# format per run, each in its own process.

_APP_OBJECTIVE_ROUNDS = [0] + _EXACT_ROUNDS
_APP_OBJECTIVE_LONG_ITERATIONS = int(os.environ.get('EXPERIMENT_APP_OBJECTIVE_LONG_ITERATIONS', '2000'))
# Long descents from different starts whose best points are within this of each other end on the same peak.
_PEAK_RESOLUTION = 1e-4
_APP_OBJECTIVE_FORMATS = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
_app_objective_runs: dict = {}


def _measure_app_objective_cases(format_key):
    """Every candidate's starts for one format, measured once per run."""
    from experiment_harnesses.app_objective_peaks import measure_boards_in_parallel
    if format_key not in _app_objective_runs:
        cases, failures = measure_boards_in_parallel(
            _APP_OBJECTIVE_FORMATS[format_key], _SEASONS, _APP_OBJECTIVE_ROUNDS, _EXACT_CANDIDATES,
            _APP_OBJECTIVE_LONG_ITERATIONS, _EXPERIMENT_WORKERS)
        _app_objective_runs[format_key] = {'cases': cases, 'failures': failures}
        for failure in failures:
            print(f'\n[{format_key} best found] board {failure["season"]} after {failure["rounds"]} rounds FAILED:\n'
                  f'{failure["traceback"]}')
    return _app_objective_runs[format_key]


def choose_app_seed(case):
    """The start the app's own policy takes: its one seed (Rotisserie), or the app-menu seed with the best objective
    before descending (head to head)."""
    app_menu = [seed for seed in case['seeds'] if seed['app_menu']]
    if len(app_menu) == 1:
        return app_menu[0]
    return max(app_menu, key=lambda seed: seed['pre_descent'])


def count_peaks(values):
    """Distinct peaks among the long descents' best points: sorted from the best down, a value more than
    _PEAK_RESOLUTION below the top of the current peak starts a new one. Returns the peaks' tops, best first."""
    tops = []
    for value in sorted(values, reverse=True):
        if not tops or tops[-1] - value > _PEAK_RESOLUTION:
            tops.append(value)
    return tops


def build_best_found_loss_rows(format_key, cases):
    """One row per candidate (App, the three losses, Best found, the total), each board's candidates in App order,
    then the mean of each, then what each fix costs as a multiple of the app's work, counted in objective evaluations:
    the app scores its M menu seeds and descends once for I iterations (M + I, or I for a single seed with no scan);
    multi-starting is M x I; the full menu (M + E) x I; the long budget adds LONG - I to the best descent."""
    rows = []
    boards: dict = {}
    for case in cases:
        boards.setdefault((case['season'], case['rounds']), []).append(case)
    losses = []
    for board_cases in boards.values():
        for case in sorted(board_cases, key=lambda case: -case['steps'][0]):
            steps = case['steps']
            gaps = [steps[index + 1] - steps[index] for index in range(3)]
            losses.append(gaps)
            seeding = (['—', '—'] if case['neutral_start'] is None
                       else [f'{100 * case["neutral_start"]:.2f}', f'{100 * (steps[0] - case["neutral_start"]):.2f}'])
            rows.append([format_key, case['season'], case['rounds'], case['player']] + seeding
                        + [f'{100 * steps[0]:.2f}']
                        + [f'{100 * gap:.2f}' for gap in gaps] + [f'{100 * steps[3]:.2f}', f'{100 * (steps[3] - steps[0]):.2f}'])
    losses = np.array(losses)
    seeded = [case for case in cases if case['neutral_start'] is not None]
    assert len(seeded) in (0, len(cases)), 'a format either seeds every board or none'
    mean_seeding = (['—', '—'] if not seeded
                    else [f'{100 * np.mean([case["neutral_start"] for case in cases]):.2f}',
                          f'{100 * np.mean([case["steps"][0] - case["neutral_start"] for case in cases]):.3f}'])
    rows.append([format_key, f'Mean of {len(cases)}', '—', '—'] + mean_seeding
                + [f'{100 * np.mean([case["steps"][0] for case in cases]):.2f}']
                + [f'{100 * losses[:, index].mean():.3f}' for index in range(3)]
                + [f'{100 * np.mean([case["steps"][3] for case in cases]):.2f}', f'{100 * losses.sum(axis=1).mean():.3f}'])
    menu = np.mean([case['menu_size'] for case in cases])
    full_menu = np.mean([case['full_menu_size'] for case in cases])
    iterations = np.mean([case['n_iterations'] for case in cases])
    long_iterations = np.mean([case['long_iterations'] for case in cases])
    app_work = iterations if menu == 1 else menu + iterations
    costs = [menu * iterations, full_menu * iterations, full_menu * iterations + long_iterations - iterations]
    rows.append([format_key, "Cost to fix (x the app's work)", '—', '—', '—', '—', '1x']
                + [f'{cost / app_work:.1f}x' for cost in costs]
                + ['—', '—'])
    return rows


def build_peak_rows(format_key, cases):
    """One row per candidate: its peaks, the gap to the second, how many starts reach the best, the app's seed and
    whether that seed's long descent reaches the best; then an ALL row."""
    rows = []
    multiple, app_misses = 0, 0
    for case in cases:
        tops = count_peaks([seed['long_budget'] for seed in case['seeds']])
        best = tops[0]
        reaching = sum(best - seed['long_budget'] <= _PEAK_RESOLUTION for seed in case['seeds'])
        app_seed = choose_app_seed(case)
        app_reaches = best - app_seed['long_budget'] <= _PEAK_RESOLUTION
        multiple += len(tops) > 1
        app_misses += not app_reaches
        rows.append([format_key, case['season'], case['rounds'], case['player'], f'{100 * best:.2f}', len(tops),
                     f'{100 * (best - tops[1]):.2f}' if len(tops) > 1 else '—',
                     f'{reaching} of {len(case["seeds"])}', label_seed(app_seed), 'yes' if app_reaches else 'no',
                     f'{100 * (best - app_seed["long_budget"]):.2f}'])
    rows.append([format_key, f'ALL ({len(cases)} candidates)', '—', '—', '—', f'{multiple} with more than one', '—',
                 '—', '—', f'{app_misses} miss', '—'])
    return rows


_BEST_FOUND_LOSS_COLUMNS = ['Format', 'Season', 'Rd', 'Player', 'Neutral start', 'Seeding (in the app)', 'App', 'Multi-starting', 'Full seed menu', 'Iterations',
                            'Best found', 'Total loss']
_PEAK_COLUMNS = ['Format', 'Season', 'Rd', 'Player', 'Best found', 'Peaks', 'Second peak (below best)', 'Starts reaching best',
                 "App's seed", "App's seed reaches best", "App's seed's peak (below best)"]


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_where_the_app_loses_to_the_best_found(format_key):
    """The app's shortfall from the best build found, split into multi-starting, the full seed menu and iterations."""
    measured = _measure_app_objective_cases(format_key)
    for row in build_best_found_loss_rows(format_key, measured['cases']):
        _record_row('App loss', _BEST_FOUND_LOSS_COLUMNS, row)
    _fail_on_breaches([f'{failure["season"]} after {failure["rounds"]} rounds: the board failed to measure'
                       for failure in measured['failures']])


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_how_many_peaks(format_key):
    """Where every start of the full menu ends, per candidate: how many peaks, and whether the app's seed reaches the
    best one."""
    measured = _measure_app_objective_cases(format_key)
    for row in build_peak_rows(format_key, measured['cases']):
        _record_row('Peakedness', _PEAK_COLUMNS, row)
    _fail_on_breaches([f'{failure["season"]} after {failure["rounds"]} rounds: the board failed to measure'
                       for failure in measured['failures']])


# ── The example team's planner as a drafter ───────────────────────────────────

_PLANNER_PAIRS = int(os.environ.get('EXPERIMENT_PLANNER_PAIRS', '24'))
# The planner's seasons: its own list, since a pair of planner drafts costs far more than the other experiments' drafts.
_PLANNER_SEASONS = exclude_shortened_seasons((os.environ.get('EXPERIMENT_PLANNER_SEASONS') or '2024-25').split(','))


@sims
def test_planner_versus_h_scoring_drafts():
    """Paired 6v6 drafts: the example team's planner choosing picks against H-scoring, anchored and unanchored."""
    from experiment_harnesses.frank_wolfe_drafts import VARIANTS, run_paired_drafts_in_parallel
    for season in _PLANNER_SEASONS:
        records_by_variant = {variant: run_paired_drafts_in_parallel(variant, season, _PLANNER_PAIRS,
                                                                     _EXPERIMENT_WORKERS, 1)
                              for variant in VARIANTS}
        # Every draft: each seat's complete-team H-score, marked P (planner) or H (H-scoring). The report summarises
        # the pairs from these same rows.
        n_seats = len(records_by_variant[VARIANTS[0]][0]['drafts'][0]['seat_scores'])
        draft_columns = ['Planner', 'Season', 'Pair', 'Draft', 'Planner minus H (pp)'] + _seat_columns(n_seats)
        for variant, records in records_by_variant.items():
            for pair_number, record in enumerate(records, start=1):
                for draft_number, draft in enumerate(record['drafts'], start=1):
                    planner_seats = set(draft['planner_seats'])
                    _record_row('Planner vs H-scoring', draft_columns,
                                [variant, season, pair_number, draft_number,
                                 f'{draft["planner_mean"] - draft["h_scoring_mean"]:+.2f}']
                                + [f'{"P" if seat in planner_seats else "H"} {score:.2f}'
                                   for seat, score in enumerate(draft['seat_scores'])])
        # A season of planner drafts is about half an hour: each is saved as it finishes (the page merges by Season),
        # so a stopped run keeps the seasons it completed.
        if _RUN_SIMS:
            write_experiment_report_files(os.path.dirname(__file__))


# ── Draft order: snake vs third-round reversal ────────────────────────────────

# Each job holds twelve sessions (one per drafter), so fewer workers than the other experiments.
_DRAFT_ORDER_WORKERS = int(os.environ.get('EXPERIMENT_DRAFT_ORDER_WORKERS', '6'))


Z_VERSUS_G_SECTION = 'Z-scores vs G-scores'


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_draft_order_bias(format_key):
    """Every season drafted twice by twelve H-scoring seats, as a snake and with third-round reversal: each seat's
    H-score and (head to head) real-week win rate. Measurements only, apart from every draft completing."""
    from experiment_harnesses.draft_order import ORDERS, measure_draft_orders_in_parallel
    results, failures = measure_draft_orders_in_parallel(format_key, _SEASONS, _DRAFT_ORDER_WORKERS)
    for result in sorted(results, key=lambda result: (result['season'], ORDERS.index(result['order']))):
        scores = result['h_scores']
        _record_row('Draft order: snake vs third-round reversal',
                    ['Format', 'Season', 'Order', 'Mean H-score'] + _seat_columns(len(scores))
                    + ['Every real win rate'],
                    [format_key, result['season'], result['order'], f'{np.mean(scores):.2f}']
                    + [f'{score:.2f}' for score in scores]
                    + [' '.join(f'{rate:.2f}' for rate in result['real_rates'] or [])])
    save_draft_picks(results, DRAFT_PICKS_PATH)
    _fail_on_breaches([f'{failure["job"]} failed:\n{failure["traceback"]}' for failure in failures])


# Every pick of the draft-order drafts: the raw data of the pool-size calibration and weight-evolution analyses.
DRAFT_PICKS_PATH = Path(__file__).resolve().parent / 'experiment_results' / 'draft_order_picks.jsonl'


def save_draft_picks(results
                     , path: Path):
    """The drafts' picks, one line per draft (format, season, order, categories, value direction, picks), merged into
    the file at path: a draft measured again replaces its line, every other line is kept."""
    import json
    lines = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            draft = json.loads(line)
            lines[(draft['format'], draft['season'], draft['order'])] = draft
    for result in results:
        lines[(result['format'], result['season'], result['order'])] = {
            key: result[key] for key in ('format', 'season', 'order', 'categories', 'value_direction', 'picks')}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(draft) for draft in lines.values()) + '\n', encoding='utf-8')


# ── Z-scores vs G-scores ──────────────────────────────────────────────────────

# Drafts per split of the seats (Z drafters 1 to 11), per season and format, each on a random seating.
_Z_VERSUS_G_DRAFTS_PER_SPLIT = int(os.environ.get('EXPERIMENT_Z_VERSUS_G_DRAFTS_PER_SPLIT', '24'))


# Head to head only: G-scores are defined differently for Rotisserie (chi scales their weekly variance), which makes a
# Rotisserie comparison with Z-scores confusing (the user, 2026-11-02).
@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC'])
def test_z_versus_g_scores(format_key):
    """Per season, every split of the seats between Z-score and G-score rankings (1 Z v 11 G to 11 v 1), drafted
    statically on random seatings: each group's mean H-score and real-week win rate per draft. Measurements only, apart
    from every season completing."""
    from experiment_harnesses.z_versus_g import measure_drafter_counts_in_parallel
    records, failures = measure_drafter_counts_in_parallel(format_key, _SEASONS, _Z_VERSUS_G_DRAFTS_PER_SPLIT,
                                                           _EXPERIMENT_WORKERS)
    draft_numbers: dict = {}
    for record in records:
        key = (record['season'], record['n_z_drafters'])
        draft_numbers[key] = draft_numbers.get(key, 0) + 1
        _record_row(Z_VERSUS_G_SECTION,
                    ['Format', 'Season', 'Z drafters', 'Draft', 'Z seats', 'G H-score', 'Z H-score', 'G real',
                     'Z real'],
                    [format_key, record['season'], record['n_z_drafters'], draft_numbers[key],
                     ' '.join(str(seat + 1) for seat in record['z_seats']), f'{record["G_h"]:.2f}',
                     f'{record["Z_h"]:.2f}', f'{record["G_real"]:.2f}', f'{record["Z_real"]:.2f}'])
    _fail_on_breaches([f'{failure["job"]} failed:\n{failure["traceback"]}' for failure in failures])


# ── The future-pick model's pool size, and the weights by round ──────────────
# Both read the picks test_draft_order_bias recorded (DRAFT_PICKS_PATH), every format and season there.

CALIBRATION_SECTION = 'Pool-size calibration'
WEIGHTS_SECTION = 'Category weights by round'
WEIGHT_CORRELATIONS_SECTION = 'Category weight correlations'
# Weight histogram bins (display units, 100 = neutral): 0-10, 10-20, ..., the last open above.
WEIGHT_BIN_WIDTH = 10
WEIGHT_BIN_COUNT = 28


# Every pick of the G-score-field drafts, with the weights each seat's H-scoring session would have picked with.
G_FIELD_PICKS_PATH = Path(__file__).resolve().parent / 'experiment_results' / 'g_field_picks.jsonl'
# The fields the calibration measures against, each with the file its drafts are in.
_CALIBRATION_FIELDS = {'H-scoring': DRAFT_PICKS_PATH, 'G-score': G_FIELD_PICKS_PATH}


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_g_field_drafts(format_key):
    """Per season, one draft of twelve G-score seats, recording the weights each seat's H-scoring session would pick
    with at every pick: the G-score field of the pool-size calibration. Saved to G_FIELD_PICKS_PATH; measurements
    only, apart from every draft completing."""
    from experiment_harnesses.pick_model_calibration import draft_g_fields_in_parallel
    drafts, failures = draft_g_fields_in_parallel(format_key, _SEASONS, _DRAFT_ORDER_WORKERS)
    save_draft_picks(drafts, G_FIELD_PICKS_PATH)
    _fail_on_breaches([f'{failure["job"]} failed:\n{failure["traceback"]}' for failure in failures])


@sims
def test_pool_size_calibration():
    """Per field, format, season and point in the draft (the round of the pick the weights were made for, and the
    round of the later pick they are tested at): how close the future-pick model's predicted tilt comes to what the
    same weights would have taken from the recorded draft, at every grid pool size. Measurements only."""
    from experiment_harnesses.pick_model_calibration import (POOL_SIZE_GRID, measure_season, read_draft_picks,
                                                             summarize_errors)
    columns = (['Field', 'Format', 'Season', 'Grouping', 'Round', 'Pairs', 'No-tilt MS']
               + [f'MSE M={pool_size}' for pool_size in POOL_SIZE_GRID])
    for field, path in _CALIBRATION_FIELDS.items():
        drafts = read_draft_picks(path)
        for format_key, season in sorted({(draft['format'], draft['season']) for draft in drafts}):
            records = measure_season(format_key, season, [draft for draft in drafts
                                                          if (draft['format'], draft['season']) == (format_key, season)])
            for grouping, key in (('Decision round', 'decision_round'), ('Later round', 'later_round')):
                for round_number in sorted({record[key] for record in records}):
                    summary = summarize_errors([record for record in records if record[key] == round_number])
                    _record_row(CALIBRATION_SECTION, columns,
                                [field, format_key, season, grouping, round_number, summary['n'],
                                 f'{summary["realised_mean_square"]:.6f}']
                                + [f'{error:.6f}' for error in summary['mean_squared_errors']])


@sims
def test_category_weights_by_round():
    """Per format, round and category: the distribution of the weights behind every recorded pick (a team's last pick
    has none). Measurements only."""
    from experiment_harnesses.pick_model_calibration import read_draft_picks
    drafts = read_draft_picks(DRAFT_PICKS_PATH)
    weights: dict = {}
    for draft in drafts:
        for pick in draft['picks']:
            if pick['weights'] is None:
                continue
            for category, weight in zip(draft['categories'], pick['weights']):
                weights.setdefault((draft['format'], pick['round'], category), []).append(weight)
    columns = ['Format', 'Round', 'Category', 'Weights', 'P10', 'P25', 'Median', 'P75', 'P90', 'Bin counts']
    for (format_key, round_number, category), values in sorted(
            weights.items(), key=lambda item: (list(_FORMATS).index(item[0][0]), item[0][1])):
        values = np.asarray(values)
        bins = np.minimum((values // WEIGHT_BIN_WIDTH).astype(int), WEIGHT_BIN_COUNT - 1)
        counts = np.bincount(bins, minlength=WEIGHT_BIN_COUNT)
        _record_row(WEIGHTS_SECTION, columns,
                    [format_key, round_number, category, len(values)]
                    + [f'{np.percentile(values, percentile):.1f}' for percentile in (10, 25, 50, 75, 90)]
                    + [' '.join(str(count) for count in counts)])
    record_weight_correlations(drafts)


def record_weight_correlations(drafts):
    """Per format: the correlation matrix of the weights over every recorded pick (every round and season pooled), and
    the players' -- the correlation of the covariance L the future-pick model uses (X-scores of the draftable pool),
    averaged over the seasons drafted."""
    from experiment_harnesses.draft_order import DRAFTER_SETTINGS, OBJECTIVES
    from experiment_harnesses.league_sessions import build_league_session
    for format_key in [key for key in _FORMATS if any(draft['format'] == key for draft in drafts)]:
        format_drafts = [draft for draft in drafts if draft['format'] == format_key]
        categories = format_drafts[0]['categories']
        if any(draft['categories'] != categories for draft in format_drafts):
            raise ValueError(f'{format_key}: the drafts do not all have the same categories.')
        weights = np.array([pick['weights'] for draft in format_drafts for pick in draft['picks']
                            if pick['weights'] is not None])
        player_correlations = []
        for season in sorted({draft['season'] for draft in format_drafts}):
            agent = build_league_session(OBJECTIVES[format_key], season, DRAFTER_SETTINGS)[2].agent
            if list(agent.x_scores.columns) != categories:
                raise ValueError(f'{format_key} {season}: the X-scores\' categories are not the drafts\'.')
            covariance = np.asarray(agent.L[0])
            deviations = np.sqrt(np.diag(covariance))
            player_correlations.append(covariance / np.outer(deviations, deviations))
        for matrix_name, matrix in (('Weights', np.corrcoef(weights, rowvar=False)),
                                    ('Players', np.mean(player_correlations, axis=0))):
            for category, correlations in zip(categories, matrix):
                _record_row(WEIGHT_CORRELATIONS_SECTION, ['Format', 'Matrix', 'Category'] + categories,
                            [format_key, matrix_name, category] + [f'{value:.3f}' for value in correlations])


# ── Best player-seasons ───────────────────────────────────────────────────────

BEST_SEASONS_SECTION = 'Best player-seasons'
BEST_SEASONS_PRE_2000_SECTION = 'Best player-seasons, pre-2000'
# Each season records the union of its best players under every ranking, with every ranking's value: enough that the
# all-time ranks of the player-seasons the page shows are exact.
_BEST_PER_SEASON = 100
# The two eras, ranked separately: seasons before 2000-01 have no reliable position data, and positions shape every
# H-score, so a pre-2000 season is comparable only with another. {section: (first season, last season)}.
_BEST_SEASONS_ERAS = {BEST_SEASONS_SECTION: ('2000-01', '9999-99'),
                      BEST_SEASONS_PRE_2000_SECTION: ('1984-85', '1999-00')}
# The objectives ranked by H-score: the three formats, and head to head half Each Category, half Most Categories.
_BEST_SEASONS_OBJECTIVES = ('Each Category', 'Half and Half', 'Most Categories', 'Rotisserie')


@sims
@pytest.mark.parametrize('section', list(_BEST_SEASONS_ERAS))
def test_best_player_seasons(section):
    """Per season of this era: the players among the season's best by first-pick H-score on the neutral board (Each
    Category, Half and Half, Most Categories, Rotisserie) or by G-score (the nine-category G-scores the head-to-head
    objectives share), each with every ranking's value. beth = 0: the season's statistics are the truth. Measurements
    only."""
    from backend.services.build_agent import load_all_params
    n_games = load_all_params()['NBA']['n_games']
    columns = (['Season', 'Player id', 'Player', 'Games']
               + [f'H-score, {objective}' for objective in _BEST_SEASONS_OBJECTIVES] + ['G-score'])
    first, last = _BEST_SEASONS_ERAS[section]
    for season in [season for season in _SEASONS if first <= season <= last]:
        values: dict = {}
        names, games = {}, None
        for objective in _BEST_SEASONS_OBJECTIVES:
            session = _build_session(objective, season=season, beth=0)
            registered = [player for player in session.agent.default_h_scores.index
                          if player in session.player_registry]
            if games is None:
                games = (session.v1_clean['Games Played %'] * n_games).round().astype(int)
            values[f'H-score, {objective}'] = 100 * session.agent.default_h_scores[registered]
            if objective == 'Each Category':
                values['G-score'] = session.agent.info['G-scores']['Total'][registered]
            names.update({player: session.player_registry[player].name for player in registered})
        recorded = set()
        for ranking in values.values():
            recorded |= set(ranking.sort_values(ascending=False).index[:_BEST_PER_SEASON])
        for player in sorted(recorded, key=lambda player: -values['H-score, Each Category'][player]):
            _record_row(section, columns,
                        [season, player, names[player], int(games[player])]
                        + [f'{values[ranking][player]:.2f}' for ranking in columns[4:]])


# ── Timing ────────────────────────────────────────────────────────────────────

DRAFT_TIMING_SECTION = 'Draft timing'
DRAFT_TIMING_BY_ROUND_SECTION = 'Draft timing by round'
AUCTION_TIMING_SECTION = 'Auction timing'
AUCTION_TIMING_BY_ROUND_SECTION = 'Auction timing by round'
# The season the timings are taken on (the backend tests' league).
_TIMING_SEASON = '2024-25'


def average_durations(samples):
    """{path: (mean seconds, mean calls)} per sample over the samples (each {path: [seconds, calls]})."""
    totals: dict = {}
    for sample in samples:
        for path, (seconds, calls) in sample.items():
            entry = totals.setdefault(path, [0.0, 0])
            entry[0] += seconds
            entry[1] += calls
    return {path: (seconds / len(samples), calls / len(samples)) for path, (seconds, calls) in totals.items()}


def record_timing_breakdown(section, format_key, operation, samples):
    for path, (seconds, calls) in sorted(average_durations(samples).items()):
        _record_row(section, ['Format', 'Operation', 'Part', 'Calls per operation', 'Mean ms', 'Samples'],
                    [format_key, operation, ' > '.join(path), f'{calls:.2f}', f'{1000 * seconds:.1f}', len(samples)])


def total_milliseconds(sample):
    """A sample's request time: its one root path."""
    roots = [seconds for path, (seconds, _) in sample.items() if len(path) == 1]
    if len(roots) != 1:
        raise ValueError(f'A timing sample has {len(roots)} root paths, expected one.')
    return 1000 * roots[0]


@sims
@pytest.mark.parametrize('format_key', ['EC', 'MC', 'Roto'])
def test_timing_breakdown(format_key):
    """Where the app's time goes, per format: cold session builds, a whole draft (every pick's first 100 candidates,
    the first team's whole board) and a whole autodraft auction (every nomination, the first team's table after each
    round), each request broken down by the functions it spends its time in. Run on an otherwise idle machine.
    Measurements only."""
    from experiment_harnesses.timing_trace import time_auction, time_draft
    draft = time_draft(format_key, _TIMING_SEASON)
    record_timing_breakdown(DRAFT_TIMING_SECTION, format_key, 'Session build', draft['build'])
    record_timing_breakdown(DRAFT_TIMING_SECTION, format_key, 'Pick: first 100 candidates',
                            [sample for _, sample in draft['pick']])
    record_timing_breakdown(DRAFT_TIMING_SECTION, format_key, "Whole board (the first team's picks)",
                            [sample for _, sample in draft['board']])
    for round_number in sorted({round_number for round_number, _ in draft['pick']}):
        picks = [total_milliseconds(sample) for number, sample in draft['pick'] if number == round_number]
        boards = [total_milliseconds(sample) for number, sample in draft['board'] if number == round_number]
        _record_row(DRAFT_TIMING_BY_ROUND_SECTION, ['Format', 'Round', 'First 100 ms', 'Whole board ms'],
                    [format_key, round_number, f'{np.mean(picks):.0f}', f'{np.mean(boards):.0f}' if boards else '—'])
    auction = time_auction(format_key, _TIMING_SEASON)
    record_timing_breakdown(AUCTION_TIMING_SECTION, format_key, 'Session build', auction['build'])
    record_timing_breakdown(AUCTION_TIMING_SECTION, format_key, 'Nomination (every team\'s valuation)',
                            [sample for _, sample in auction['nomination']])
    record_timing_breakdown(AUCTION_TIMING_SECTION, format_key, "Auction table (the first team's evaluate)",
                            [sample for _, sample in auction['table']])
    for round_number in sorted({round_number for round_number, _ in auction['nomination']}):
        nominations = [total_milliseconds(sample) for number, sample in auction['nomination'] if number == round_number]
        tables = [total_milliseconds(sample) for number, sample in auction['table'] if number == round_number]
        _record_row(AUCTION_TIMING_BY_ROUND_SECTION, ['Format', 'Round', 'Nomination ms', 'Table ms'],
                    [format_key, round_number, f'{np.mean(nominations):.0f}', f'{np.mean(tables):.0f}' if tables else '—'])


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
DOLLAR_GINI_SECTION = 'Dollar shape: Gini by SAVOR settings'
# The Gini heatmap's finer grid around the app's defaults (S_R $2, S_sigma 10; the app allows S_R $0-50 and S_sigma
# 0-200).
_GINI_REPLACEMENT_THRESHOLDS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
_GINI_SAVOR_NOISES = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)


def compute_gini_coefficient(values):
    """The Gini coefficient of non-negative values: the mean absolute difference between every two of them over twice
    their mean (0 every value equal, towards 1 all in one)."""
    values = np.sort(np.asarray(values, dtype=float))
    if (values < 0).any():
        raise ValueError(f'A Gini coefficient needs non-negative values; the smallest is {values[0]}.')
    n = len(values)
    ranks = np.arange(1, n + 1)
    return float(np.sum((2 * ranks - n - 1) * values) / (n * values.sum()))
# The drafted players' tiers, best first: (name, share of the drafted players).
_DOLLAR_TIERS = (('Top 10%', 0.10), ('Middle 80%', 0.80), ('Bottom 10%', 0.10))


def _split_into_dollar_tiers(dollars, total_picks):
    """The drafted players' dollars (best first): the average player's within each tier."""
    top_count = round(total_picks * _DOLLAR_TIERS[0][1])
    bottom_count = round(total_picks * _DOLLAR_TIERS[2][1])
    drafted = dollars.iloc[:total_picks].to_numpy()
    return [float(drafted[:top_count].mean()), float(drafted[top_count:total_picks - bottom_count].mean()),
            float(drafted[total_picks - bottom_count:].mean())]


DOLLAR_GINI_BY_SEASON_SECTION = 'Dollar inequality by season'
# The SAVOR settings the season trend is measured at: (label, replacement threshold S_R, noise S_sigma).
_GINI_TREND_SETTINGS = (('SAVOR off', 0.0, 0.0), ('the app (S_R $2, S_sigma 10)', 2.0, 10.0))
TOP_VALUES_SECTION = 'Most valuable player-seasons'
# Each season's most valuable players recorded for the all-time ranking (the page ranks across seasons).
_TOP_VALUES_PER_SEASON = 10


@sims
def test_dollar_gini_by_season(sessions):
    """Per season, the Gini coefficient of the drafted players' dollars on the starting board (12 teams x $200, Each
    Category), with SAVOR off and at the app's defaults. Measurements only."""
    from backend.math.algorithm_helpers import auction_value_adjuster
    for season in _SEASONS:
        session = sessions('EC', auction=True, season=season)
        agent = session.agent
        total_picks = agent.n_drafters * agent.n_picks
        total_cash = agent.n_drafters * session.current_settings['cash_per_team']
        ginis = []
        for _, replacement_threshold, savor_noise in _GINI_TREND_SETTINGS:
            dollars = auction_value_adjuster(agent.default_h_scores, total_picks, total_cash, savor_noise,
                                             replacement_threshold).sort_values(ascending=False)
            ginis.append(compute_gini_coefficient(dollars.iloc[:total_picks]))
        _record_row(DOLLAR_GINI_BY_SEASON_SECTION, ['Season'] + [f'Gini, {label}' for label, _, _ in _GINI_TREND_SETTINGS],
                    [season] + [f'{gini:.4f}' for gini in ginis])
        # The last setting is the app's: its dollars rank the season's most valuable players.
        for rank, (player_id, value) in enumerate(dollars.iloc[:_TOP_VALUES_PER_SEASON].items(), start=1):
            _record_row(TOP_VALUES_SECTION, ['Season', 'Rank in season', 'Player', 'Dollars'],
                        [season, rank, session.player_registry[player_id].name, f'{value:.1f}'])


DOLLAR_CURVES_SECTION = 'Dollar curves by league size'
# The league sizes the dollar curves compare: (teams, roster size). The team counts at the default roster of 13, and
# the roster sizes parameters.yaml has slot layouts for (13-16) at the default 12 teams.
_DOLLAR_CURVE_LEAGUES = ((8, 13), (10, 13), (12, 13), (14, 13), (16, 13), (12, 14), (12, 15), (12, 16))


def build_league_size_session(n_drafters, n_picks, season):
    """An Each Category auction session ($200 a team, the app's defaults) with this many teams and roster spots, the
    slots parameters.yaml lays out for that roster size."""
    import yaml
    from benchmark_helpers import _PARAMS_PATH
    request = _build_session_request(objective='Each Category', n_drafters=n_drafters, cash_per_team=200)
    with open(_PARAMS_PATH) as params_file:
        positions = yaml.safe_load(params_file)['NBA']['options']['positions'][n_picks]
    request['league']['n_picks'] = n_picks
    request['slot_counts'] = {**positions['base'], **positions['flex']}
    request['data_source']['season'] = season
    response = client.post('/sessions', json=request)
    assert response.status_code == 201, f'session build failed: {response.text}'
    return get_session(response.json()['session_id'])


@sims
def test_dollar_curves_by_league_size():
    """Per season and league size, every drafted player's dollars at the start of an auction, by draft rank, at the
    app's SAVOR settings. Measurements only."""
    from backend.math.algorithm_helpers import auction_value_adjuster
    for season in _SEASONS:
        for n_drafters, n_picks in _DOLLAR_CURVE_LEAGUES:
            session = build_league_size_session(n_drafters, n_picks, season)
            settings = session.current_settings
            total_picks = n_drafters * n_picks
            dollars = auction_value_adjuster(session.agent.default_h_scores, total_picks,
                                             n_drafters * settings['cash_per_team'], settings['streaming_noise'],
                                             settings['replacement_threshold']).sort_values(ascending=False)
            _record_row(DOLLAR_CURVES_SECTION, ['Season', 'Teams', 'Roster size', 'Top player $', 'Every dollar value'],
                        [season, n_drafters, n_picks, f'{dollars.iloc[0]:.1f}',
                         ' '.join(f'{value:.2f}' for value in dollars.iloc[:total_picks])])


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
        # The Gini heatmap: the drafted players' dollars (the top total_picks) on the finer grid.
        for replacement_threshold in _GINI_REPLACEMENT_THRESHOLDS:
            for savor_noise in _GINI_SAVOR_NOISES:
                dollars = auction_value_adjuster(agent.default_h_scores, total_picks, total_cash, savor_noise,
                                                 replacement_threshold).sort_values(ascending=False)
                _record_row(DOLLAR_GINI_SECTION, ['Season', 'Replacement threshold', 'SAVOR noise', 'Gini'],
                            [season, replacement_threshold, savor_noise,
                             f'{compute_gini_coefficient(dollars.iloc[:total_picks]):.4f}'])
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
