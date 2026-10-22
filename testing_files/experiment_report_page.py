"""The experiment report page: one tab per stored section, each with its explanation, a takeaway card (a headline and a
chart drawn from the section's own rows) and its table, colour-coded.

Colour code, the same in every tab: green = passes (or no loss), red = breaches a floor (or a loss, stronger for more),
amber = a warning short of a breach. Values that are expected win rates or differences of them carry '%'.

Each section's view (SECTION_VIEWS) is read from its stored rows alone, so the page can be rebuilt without rerunning
anything. A section with no view is drawn as a plain table.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html import escape
from typing import Callable, Optional

import numpy as np

PAGE_CSS = """
  :root { --bg:#ffffff; --fg:#26282e; --muted:#6a6f7a; --line:#e4e6ea; --card:#f7f8fa; --accent:#33556e;
          --good:#1e8a3a; --bad-rgb:200,30,30; --warn:#a85d00; --heat-rgb:0,114,178;
          --series-0:#0072b2; --series-1:#e69f00; --series-2:#56b4e9; --series-3:#cc79a7; --series-4:#009e73; --series-5:#d55e00; --series-6:#f0e442; --other:#80858e; --no-hard-punt:#d5d8dd; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#14161c; --fg:#e6e8ec; --muted:#9aa0ab; --line:#2c303a; --card:#1b1e26; --accent:#8fb4d1;
            --good:#5ccb7a; --bad-rgb:255,105,105; --warn:#f0b43c; --heat-rgb:86,180,233;
            --series-0:#3d9be0; --series-1:#f0b43c; --series-2:#7cc8f2; --series-3:#de9cc4; --series-4:#3cc79a; --series-5:#f07b3f; --series-6:#e8dc5a; --other:#7a808a; --no-hard-punt:#3a3f48; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, "Segoe UI", sans-serif; }
  main { max-width: 1400px; margin: 0 auto; padding: 28px 20px 64px; }
  h1 { font-size: 1.35rem; margin: 0 0 2px; }
  .stamp { color: var(--muted); font-size: .85rem; margin-bottom: 18px; }
  nav { margin-bottom:18px; padding-bottom:10px; border-bottom:1px solid var(--line); }
  .group-tabs { display:flex; flex-wrap:wrap; gap:2px 18px; border-bottom:1px solid var(--line); margin-bottom:10px; }
  .group-tabs button { background:none; border:none; border-bottom:2px solid transparent; color:var(--muted);
                       padding:6px 2px; font-size:.9rem; font-weight:600; cursor:pointer; margin-bottom:-1px; }
  .group-tabs button.active { color:var(--accent); border-bottom-color:var(--accent); }
  .tab-buttons { display:none; flex-wrap:wrap; gap:6px; }
  .tab-buttons.active { display:flex; }
  .tab-buttons button { background:var(--card); color:var(--fg); border:1px solid var(--line); border-radius:8px;
                        padding:5px 10px; font-size:.82rem; cursor:pointer; }
  .tab-buttons button.active { border-color:var(--accent); color:var(--accent); font-weight:600; }
  section { display:none; }
  section.active { display:block; }
  h2 { font-size: 1.05rem; margin: 4px 0 2px; color: var(--accent); }
  .generated { color: var(--muted); font-size: .8rem; margin-bottom: 10px; }
  .explain { color: var(--muted); font-size: .9rem; margin: 0 0 10px; }
  ul.explain { padding-left: 20px; }
  ul.explain li { margin: 2px 0; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:0 0 12px; }
  .takeaway { margin-bottom:8px; }
  .status { display:inline-block; font-size:.78rem; font-weight:600; border-radius:999px; padding:1px 9px;
            margin-right:8px; border:1px solid currentColor; }
  .status.good { color:var(--good); } .status.bad { color:rgb(var(--bad-rgb)); } .status.warn { color:var(--warn); }
  .chart { width:100%; height:auto; display:block; }
  .chart text { fill:var(--muted); font-size:11px; font-family:system-ui, "Segoe UI", sans-serif; }
  .chart .grid { stroke:var(--line); stroke-width:1; }
  .chart .axis { stroke:var(--muted); stroke-width:1; }
  .chart .reference { stroke:var(--muted); stroke-width:1.2; stroke-dasharray:5 4; }
  .chart .reference.bad { stroke:rgb(var(--bad-rgb)); }
  .chart .reference-label { font-size:10.5px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 18px; margin-top:6px; font-size:.85rem; color:var(--muted); }
  .swatch { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; }
  .multiples { display:grid; grid-template-columns:repeat(auto-fit, minmax(420px, 1fr)); gap:4px 24px; }
  .multiples h3 { font-size:.85rem; font-weight:600; color:var(--muted); margin:6px 0 0; }
  .other { background:var(--other); }
  .builds { display:flex; flex-wrap:wrap; gap:4px 5px; }
  .build { display:inline-flex; align-items:baseline; gap:4px; border:1px solid var(--line); border-left-width:5px;
           border-radius:6px; padding:1px 7px; background:var(--bg); font-size:.82rem; white-space:nowrap; }
  .build b { color:var(--fg); }
  .build .soft { color:var(--muted); }
  .build .count { color:var(--muted); font-size:.75rem; margin-left:2px; }
  .bar-track { display:flex; gap:2px; height:22px; border-radius:6px; overflow:hidden; }
  .bar-segment { flex-grow:0; flex-shrink:0; min-width:3px; }
  .note { color:var(--muted); font-size:.85rem; margin:0 0 8px; }
  .tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:8px; margin:8px 0 12px; }
  .tile { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:8px 11px; }
  .tile-value { font-size:1.4rem; font-weight:650; line-height:1.25; color:var(--fg);
                font-variant-numeric:tabular-nums; }
  .tile-value.good { color:var(--good); } .tile-value.bad { color:rgb(var(--bad-rgb)); }
  .tile-value.warn { color:var(--warn); }
  .tile-label { color:var(--muted); font-size:.8rem; line-height:1.35; }
  .card h3 { font-size:.92rem; font-weight:600; color:var(--fg); margin:16px 0 2px; }
  .tablewrap { background:var(--card); border:1px solid var(--line); border-radius:10px;
               padding: 4px 12px; overflow-x:auto; }
  table { border-collapse: collapse; width:100%; font-size:.88rem; }
  th { text-align:left; color:var(--muted); font-weight:600; padding:8px 8px 6px 4px; vertical-align:bottom;
       border-bottom:1px solid var(--line); }
  td { padding:6px 8px 6px 4px; border-bottom:1px solid var(--line); white-space:nowrap;
       font-variant-numeric: tabular-nums; }
  td.wrap { white-space:normal; min-width:240px; }
  tr:last-child td { border-bottom:none; }
  tr.summary td { white-space:normal; font-weight:600; border-bottom:2px solid var(--line); }
"""
for _index in range(7):
    PAGE_CSS += f"  .series-{_index} {{ background:var(--series-{_index}); }}\n"

GOOD = ' style="color:var(--good)"'
WARN = ' style="color:var(--warn);font-weight:600"'
BAD = ' style="color:rgb(var(--bad-rgb));font-weight:600"'


def style_bad(intensity: float = 1.0) -> str:
    """Red text, stronger (and bold in the top half) for a larger problem."""
    weight = ';font-weight:600' if intensity > 0.5 else ''
    return f' style="color:rgba(var(--bad-rgb),{0.45 + 0.55 * min(max(intensity, 0.0), 1.0):.2f}){weight}"'


def style_heat(intensity: float) -> str:
    """A blue background for a count, deeper for more (sequential, one hue)."""
    return f' style="background:rgba(var(--heat-rgb),{0.06 + 0.5 * min(max(intensity, 0.0), 1.0):.2f})"'


def read_leading_number(cell) -> Optional[float]:
    """The first number in a cell ('0.0049 mean (49%)' -> 0.0049; '+1.23' -> 1.23), or None."""
    match = re.search(r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][-+]?\d+)?', str(cell))
    return float(match.group().replace(',', '')) if match else None


def is_summary_row(row) -> bool:
    """A pooled row: its first or second cell names it (ALL ..., Mean of ..., Cost to fix ...)."""
    return any(str(cell).startswith(('ALL', 'Mean of', 'Cost to fix')) for cell in row[:2])


def format_percent(cell, signed: bool = False, places: Optional[int] = None) -> str:
    """'52.8' -> '52.8%', keeping the cell's own precision unless places is given; non-numbers pass through."""
    text = str(cell)
    if text in ('—', '') or text.endswith('%') or read_leading_number(text) is None:
        return text
    number = read_leading_number(text)
    digits = places if places is not None else (len(text.split('.')[-1].split(' ')[0]) if '.' in text else 0)
    shown = f'{number:+.{digits}f}' if signed else f'{number:.{digits}f}'
    return shown + '%' + (text[len(text.split(' ')[0]):] if ' ' in text else '')


# ── Charts (inline SVG) ───────────────────────────────────────────────────────

@dataclass
class Series:
    name: str
    values: list                              # floats, or None for a gap
    errors: Optional[list] = None             # half-widths of an error bar per point


@dataclass
class ReferenceLine:
    value: float
    label: str
    is_floor: bool = False                    # drawn red: crossing it is a breach


def compute_nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    span = high - low if high > low else 1.0
    raw = span / count
    magnitude = 10 ** np.floor(np.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=raw)
    first = np.ceil(low / step) * step
    return [round(value, 10) for value in np.arange(first, high + step * 0.5, step) if value <= high + 1e-12]


def render_chart(
    x_labels: list[str]
    , series: list[Series]
    , y_title: str
    , kind: str = 'line'
    , references: tuple = ()
    , y_floor: Optional[float] = None
    , y_ceiling: Optional[float] = None
    , value_suffix: str = ''
    , value_places: int = 2
    , height: int = 240
    , background_series: tuple = ()
    , width: int = 1100
    , show_points: bool = True
) -> str:
    """A chart over categorical x positions: 'line' (lines with points), 'dots', 'bars' (grouped) or 'stacked' (one bar
    per position, the series stacked bottom first). Reference lines are dashed (red for a floor). A legend is drawn
    below when there is more than one series. width is the drawing's own width, so a chart set beside another (in
    .multiples) keeps legible text by being drawn narrow rather than scaled down; show_points False draws a line
    chart's lines alone, for curves too dense for a marker at every point."""
    left, right, top, bottom = 58, (120 if width > 700 else 70), 14, 46
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for one in list(series) + list(background_series)
              for value, error in zip(one.values, one.errors or [0] * len(one.values))
              for value in ((value - error, value + error) if value is not None else ()) if value is not None]
    values += [reference.value for reference in references]
    if kind == 'stacked':
        values = [sum(one.values[index] or 0.0 for one in series) for index in range(len(x_labels))] + [0.0]
    low = min(values) if values else 0.0
    high = max(values) if values else 1.0
    if kind in ('bars', 'stacked'):
        low, high = min(low, 0.0), max(high, 0.0)
    if y_floor is not None:
        low = min(low, y_floor)
    if y_ceiling is not None:
        high = max(high, y_ceiling)
    padding = (high - low) * 0.08 or 1.0
    low, high = (low - padding if kind not in ('bars', 'stacked') or low < 0 else low), high + padding
    ticks = compute_nice_ticks(low, high)
    low, high = min(low, ticks[0]), max(high, ticks[-1])

    def y_of(value):
        return top + plot_height * (1 - (value - low) / (high - low))

    n = len(x_labels)
    slot = plot_width / max(n, 1)

    def x_of(index):
        return left + slot * (index + 0.5)

    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_title)}">']
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_width}" y1="{y_of(tick):.1f}" y2="{y_of(tick):.1f}"/>'
                     f'<text x="{left - 6}" y="{y_of(tick) + 3.5:.1f}" text-anchor="end">'
                     f'{tick + 0.0:g}{escape(value_suffix)}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" '
                 f'y2="{top + plot_height}"/>')
    parts.append(f'<text x="14" y="{top + plot_height / 2:.0f}" transform="rotate(-90 14 {top + plot_height / 2:.0f})" '
                 f'text-anchor="middle">{escape(y_title)}</text>')
    label_step = max(1, int(np.ceil(n / (22 * width / 1100))))
    for index, label in enumerate(x_labels):
        if index % label_step == 0:
            parts.append(f'<text x="{x_of(index):.1f}" y="{top + plot_height + 16}" text-anchor="middle">'
                         f'{escape(str(label))}</text>')
    for reference in references:
        y = y_of(reference.value)
        css = 'reference bad' if reference.is_floor else 'reference'
        parts.append(f'<line class="{css}" x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text class="reference-label" x="{left + plot_width + 6}" y="{y + 3.5:.1f}">'
                     f'{escape(reference.label)}</text>')
    for one in background_series:
        path = ' '.join(f'{x_of(index):.1f},{y_of(value):.1f}' for index, value in enumerate(one.values)
                        if value is not None)
        parts.append(f'<polyline points="{path}" style="fill:none;stroke:var(--muted);stroke-opacity:.45;'
                     f'stroke-width:1.5"><title>{escape(one.name)}</title></polyline>')
    group_width = slot * 0.72
    bar_width = group_width / max(len(series), 1)
    stack_tops = [0.0] * n
    for series_index, one in enumerate(series):
        color = f'var(--series-{series_index % 6})'
        points = []
        for index, value in enumerate(one.values):
            if value is None:
                points.append(None)
                continue
            tip = escape(f'{one.name} · {x_labels[index]}: {value:.{value_places}f}{value_suffix}')
            if kind in ('bars', 'stacked') and value == 0:
                continue
            if kind == 'stacked':
                x = x_of(index) - group_width / 2
                y0, y1 = y_of(stack_tops[index] + value), y_of(stack_tops[index])
                parts.append(f'<rect x="{x:.1f}" y="{y0:.1f}" width="{group_width:.1f}" height="{max(y1 - y0, 1):.1f}" '
                             f'style="fill:{color};stroke:var(--card);stroke-width:2"><title>{tip}</title></rect>')
                stack_tops[index] += value
                continue
            if kind == 'bars':
                x = x_of(index) - group_width / 2 + bar_width * series_index
                y0, y1 = sorted((y_of(0.0), y_of(value)))
                parts.append(f'<rect x="{x + 1:.1f}" y="{y0:.1f}" width="{max(bar_width - 2, 1):.1f}" '
                             f'height="{max(y1 - y0, 1):.1f}" rx="2" style="fill:{color}"><title>{tip}</title></rect>')
                center = x + bar_width / 2
            else:
                offset = (series_index - (len(series) - 1) / 2) * min(6.0, slot * 0.15) if kind == 'dots' else 0.0
                center = x_of(index) + offset
                points.append((center, y_of(value)))
            if one.errors is not None and one.errors[index]:
                error = one.errors[index]
                parts.append(f'<line x1="{center:.1f}" x2="{center:.1f}" y1="{y_of(value - error):.1f}" '
                             f'y2="{y_of(value + error):.1f}" style="stroke:{color};stroke-width:1.5"/>')
            if kind != 'bars' and (show_points or kind != 'line'):
                parts.append(f'<circle cx="{center:.1f}" cy="{y_of(value):.1f}" r="{3.2 if n > 30 else 4}" '
                             f'style="fill:{color};stroke:var(--card);stroke-width:1.5"><title>{tip}</title></circle>')
        if kind == 'line':
            runs, run = [], []
            for point in points:
                if point is None:
                    if run:
                        runs.append(run)
                    run = []
                else:
                    run.append(point)
            if run:
                runs.append(run)
            for run in runs:
                if len(run) > 1:
                    path = ' '.join(f'{x:.1f},{y:.1f}' for x, y in run)
                    parts.insert(1, f'<polyline points="{path}" style="fill:none;stroke:{color};stroke-width:2"/>')
    parts.append('</svg>')
    legend = ''
    if len(series) > 1:
        legend = '<div class="legend">' + ''.join(
            f'<span><span class="swatch series-{index % 6}"></span>{escape(one.name)}</span>'
            for index, one in enumerate(series)) + '</div>'
    return ''.join(parts) + legend


def render_stat_tiles(tiles: list[tuple[str, str, str]]) -> str:
    """A row of headline numbers, each (value, what it is, status): status is '' or good / warn / bad, which colours
    the value. Labels are text tokens, never the series colours."""
    return '<div class="tiles">' + ''.join(
        f'<div class="tile"><div class="tile-value {status}">{escape(value)}</div>'
        f'<div class="tile-label">{escape(label)}</div></div>' for value, label, status in tiles) + '</div>'


def render_takeaway(status: str, status_text: str, text: str) -> str:
    """The card's headline: a status pill (good / warn / bad) and one sentence."""
    return f'<div class="takeaway"><span class="status {status}">{escape(status_text)}</span>{text}</div>'


# ── Section views ─────────────────────────────────────────────────────────────

@dataclass
class SectionView:
    card: Callable[[list, list], str]                                    # (columns, rows) -> the takeaway card's HTML
    cell_styles: Optional[Callable[[list, list], list]] = None           # (columns, rows) -> per-row, per-cell styles
    formatters: dict = field(default_factory=dict)                       # column -> cell -> displayed text
    row_filter: Optional[Callable[[list, list], tuple]] = None           # (columns, rows) -> (rows, note)
    sort_rows: Optional[Callable[[list, list], list]] = None             # (columns, rows) -> rows
    wrap_columns: tuple = ()
    derive_table: Optional[Callable[[list, list], tuple]] = None         # (columns, rows) -> the table's (columns, rows)
    html_formatters: dict = field(default_factory=dict)                  # column -> cell -> the cell's HTML


def make_rule_styles(rules: dict) -> Callable:
    """cell_styles from per-column rules: rule(value, row, columns) -> a style string or ''."""
    def compute(columns, rows):
        styles = []
        for row in rows:
            row_styles = [''] * len(row)
            for name, rule in rules.items():
                if name in columns:
                    index = columns.index(name)
                    value = read_leading_number(row[index])
                    if value is not None:
                        row_styles[index] = rule(value, row, columns)
            styles.append(row_styles)
        return styles
    return compute


def candidate_rows(rows):
    return [row for row in rows if not is_summary_row(row)]


def column_values(columns, rows, name):
    return [read_leading_number(row[columns.index(name)]) for row in rows]


def find_summary(rows, label_start, column_index=0):
    return next((row for row in rows if str(row[column_index]).startswith(label_start)), None)


# H-scoring vs a G-score field
def card_h_versus_g(columns, rows):
    pooled = {row[0]: read_leading_number(row[2]) for row in rows if str(row[1]).startswith('ALL')}
    seasons = sorted({row[1] for row in candidate_rows(rows)})
    by_format = {format_key: {row[1]: read_leading_number(row[2]) for row in candidate_rows(rows) if row[0] == format_key}
                 for format_key in ('EC', 'MC')}
    below = sum(value <= 50.5 for values in by_format.values() for value in values.values())
    status = ('good', 'passes') if below == 0 and all(value > 52 for value in pooled.values()) else ('bad', 'breach')
    text = (f'H-scoring wins <b>{pooled.get("EC", float("nan")):.1f}%</b> of category matchups in Each Category and '
            f'<b>{pooled.get("MC", float("nan")):.1f}%</b> in Most Categories against a G-score field (50% = break-even). '
            f'{below} of {len(seasons) * 2} season-formats fall at or below the 50.5% floor.')
    seat_columns = [name for name in columns if re.fullmatch(r'S\d+', name)]
    series = []
    for format_key in ('EC', 'MC'):
        by_seat = np.array([[read_leading_number(row[columns.index(name)]) for name in seat_columns]
                            for row in candidate_rows(rows) if row[0] == format_key])
        standard_errors = by_seat.std(axis=0, ddof=1) / np.sqrt(len(by_seat))
        series.append(Series(format_key, list(by_seat.mean(axis=0)), list(2 * standard_errors)))
    chart = render_chart([name[1:] for name in seat_columns], series, 'Win rate',
                         references=(ReferenceLine(50.0, 'parity 50%'),), value_suffix='%', value_places=1)
    note = (f'<div class="note">By draft position, averaged over the {len(seasons)} seasons; bars are ±2 standard '
            f'errors across seasons.</div>')
    return render_takeaway(*status, text) + note + chart


def rule_season_win_rate(value, row, columns):
    if is_summary_row(row):
        return GOOD if value > 52 else BAD
    return GOOD if value > 50.5 else BAD


def rule_seat_win_rate(value, row, columns):
    return style_bad(1.0) if value < 50 else ''


# Awareness
def card_awareness(columns, rows):
    pooled = {row[0]: (read_leading_number(row[2]), read_leading_number(row[3]))
              for row in rows if str(row[1]).startswith('ALL')}
    seasons = sorted({row[1] for row in candidate_rows(rows)})
    comparisons = [comparison for comparison in ('vs unaware H field', 'vs G field') if comparison in pooled]
    series = []
    for comparison in comparisons:
        values = {row[1]: (read_leading_number(row[2]), read_leading_number(row[3]))
                  for row in candidate_rows(rows) if row[0] == comparison}
        series.append(Series(comparison, [values[s][0] if s in values else None for s in seasons] + [pooled[comparison][0]],
                             [2 * values[s][1] if s in values else 0 for s in seasons] + [2 * pooled[comparison][1]]))
    harmful = any(mean < -2 * error for mean, error in pooled.values())
    unaware = pooled.get('vs unaware H field', (float('nan'), float('nan')))
    g_field = pooled.get('vs G field', (float('nan'), float('nan')))
    text = (f'The opponent model adds <b>{unaware[0]:+.2f}% ± {unaware[1]:.2f}</b> against H-drafters without it, and '
            f'<b>{g_field[0]:+.2f}% ± {g_field[1]:.2f}</b> against G-score drafters, where its predictions are wrong. '
            f'± is one standard error; the bars are ±2.')
    chart = render_chart([season[2:] for season in seasons] + ['ALL'], series, 'On minus off',
                         kind='dots', references=(ReferenceLine(0.0, 'no effect'),),
                         value_suffix='%', value_places=2)
    status = ('bad', 'hurts') if harmful else ('good', 'no harm')
    return render_takeaway(*status, text) + chart


def rule_pooled_significance(value, row, columns):
    """Only the pooled means, and only when more than 2 standard errors from 0: green above, red below."""
    if not is_summary_row(row):
        return ''
    error = read_leading_number(row[columns.index('SE (pp)')])
    if value > 2 * error:
        return GOOD
    return BAD if value < -2 * error else ''


def rule_signed_gain(floor):
    def rule(value, row, columns):
        if value < floor:
            return BAD
        return GOOD if value > 0 else ''
    return rule


# Punt diversity
# Kinds of build: the exact set of categories a build HARD punts (marked '!'). Every chart colours a kind the same way,
# in pairs of one hue (ColorBrewer Paired): a single hard punt dark, the two-punt combination containing it light. A build
# with no hard punt is its own kind; one whose hard punts are none of these is Other.
BUILD_KINDS = [
    ('FT%', {'FT%'}, '#1f78b4'),
    ('3s + FT%', {'3s', 'FT%'}, '#a6cee3'),
    ('3s', {'3s'}, '#33a02c'),
    ('3s + PTS', {'3s', 'PTS'}, '#b2df8a'),
    ('AST', {'AST'}, '#e31a1c'),
    ('AST + PTS', {'AST', 'PTS'}, '#fb9a99'),
    ('BLK', {'BLK'}, '#ff7f00'),
    ('BLK + FG%', {'BLK', 'FG%'}, '#fdbf6f'),
    ('FG%', {'FG%'}, '#6a3d9a'),
    ('FG% + TO', {'FG%', 'TO'}, '#cab2d6'),
    ('TO', {'TO'}, '#b15928'),
    ('PTS', {'PTS'}, '#e8d84a'),
]
NO_HARD_PUNT = len(BUILD_KINDS)
OTHER_BUILD = len(BUILD_KINDS) + 1


def classify_build(combination: str) -> int:
    """The kind of a punt combination ('3s/FT%!/TO' -> FT%): a BUILD_KINDS index, NO_HARD_PUNT or OTHER_BUILD."""
    hard_punts = {category[:-1] for category in combination.split('/') if category.endswith('!')}
    if not hard_punts:
        return NO_HARD_PUNT
    return next((index for index, (_, punted, _) in enumerate(BUILD_KINDS) if punted == hard_punts), OTHER_BUILD)


def name_build_kind(kind: int) -> str:
    return {NO_HARD_PUNT: 'No hard punt', OTHER_BUILD: 'Other'}.get(kind) or BUILD_KINDS[kind][0]


def fill_build_kind(kind: int) -> str:
    return {NO_HARD_PUNT: 'var(--no-hard-punt)', OTHER_BUILD: 'var(--other)'}.get(kind) or BUILD_KINDS[kind][2]


def read_breakdown(breakdown: str) -> list[tuple[str, int]]:
    """'3s/FT%/TO x2, FT%/TO x1' -> [('3s/FT%/TO', 2), ('FT%/TO', 1)]."""
    parts = []
    for part in breakdown.split(', '):
        match = re.fullmatch(r'(.+) x(\d+)', part)
        if match is None:
            raise ValueError(f'Unreadable breakdown entry {part!r}')
        parts.append((match.group(1), int(match.group(2))))
    return parts


def render_stacked_bars(
    x_labels: list[str]
    , stacks: list[list[tuple[int, float, str]]]
    , y_title: str
    , total: float
    , height: int = 210
) -> str:
    """Stacked bars, one per x position; each stack is [(kind index, value, tooltip)], bottom first, coloured by kind
    (fill_build_kind). Segments are separated by a 2px gap in the card colour."""
    width, left, right, top, bottom = 520, 40, 8, 10, 26
    plot_width, plot_height = width - left - right, height - top - bottom
    slot = plot_width / len(x_labels)
    bar_width = slot * 0.62
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_title)}">']
    for tick in compute_nice_ticks(0, total, 4):
        y = top + plot_height * (1 - tick / total)
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{left - 6}" y="{y + 3.5:.1f}" text-anchor="end">{tick + 0.0:g}</text>')
    for index, (label, stack) in enumerate(zip(x_labels, stacks)):
        x = left + slot * index + (slot - bar_width) / 2
        running = 0.0
        for kind, value, tip in stack:
            y_top = top + plot_height * (1 - (running + value) / total)
            segment_height = plot_height * value / total
            fill = fill_build_kind(kind)
            parts.append(f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_width:.1f}" height="{segment_height:.1f}" '
                         f'style="fill:{fill};stroke:var(--card);stroke-width:2"><title>{escape(tip)}</title></rect>')
            running += value
        parts.append(f'<text x="{x + bar_width / 2:.1f}" y="{top + plot_height + 16}" text-anchor="middle">'
                     f'{escape(label)}</text>')
    parts.append(f'<line class="axis" x1="{left}" x2="{left + plot_width}" y1="{top + plot_height}" '
                 f'y2="{top + plot_height}"/></svg>')
    return ''.join(parts)


def count_distinct_builds(breakdown: str, hard_only: bool) -> int:
    """Distinct builds in one season's breakdown: by all punts, or by the hard punts alone (no hard punt is a build)."""
    if not hard_only:
        return len(read_breakdown(breakdown))
    return len({frozenset(category for category in combination.split('/') if category.endswith('!'))
                for combination, _ in read_breakdown(breakdown)})


PUNT_DIVERSITY_COLUMNS = ['Season', 'Format', 'Mode', 'Builds (all punts)', 'Builds (hard punts)', 'Hard', 'Soft',
                          'Builds (bold = hard punt)']


def derive_punt_diversity_table(columns, rows):
    """The stored table with the distinct-build count by hard punts alone beside the count by all punts."""
    breakdown_index = columns.index('Breakdown (! = hard)')
    derived = [[row[columns.index('Season')], row[columns.index('Format')], row[columns.index('Mode')],
                row[columns.index('Distinct builds')], str(count_distinct_builds(row[breakdown_index], True)),
                row[columns.index('Hard')], row[columns.index('Soft')], row[breakdown_index]] for row in rows]
    return PUNT_DIVERSITY_COLUMNS, derived


def render_build_chips(breakdown: str) -> str:
    """One chip per build, most common first: a bar in its hard-punt colour, hard punts bold, soft punts muted."""
    chips = []
    for combination, count in sorted(read_breakdown(breakdown), key=lambda item: (-item[1], classify_build(item[0]))):
        categories = combination.split('/')
        hard = [category[:-1] for category in categories if category.endswith('!')]
        soft = [category for category in categories if not category.endswith('!')]
        label = ' '.join([f'<b>{escape(category)}</b>' for category in hard]
                         + [f'<span class="soft">{escape(category)}</span>' for category in soft])
        times = f'<span class="count">×{count}</span>' if count > 1 else ''
        chips.append(f'<span class="build" style="border-left-color:{fill_build_kind(classify_build(combination))}">'
                     f'{label}{times}</span>')
    return f'<div class="builds">{"".join(chips)}</div>'


def card_punt_diversity(columns, rows):
    seasons = sorted({row[0] for row in rows})
    breakdown_index = columns.index('Breakdown (! = hard)')
    panels = []
    for format_key in ('MC', 'EC'):
        for mode in ('draft', 'auction'):
            by_season = {row[0]: row for row in rows if row[1] == format_key and row[2] == mode}
            if not by_season:
                continue
            stacks = []
            for season in seasons:
                if season not in by_season:
                    stacks.append([])
                    continue
                by_kind: dict[int, list[tuple[str, int]]] = {}
                for combination, count in read_breakdown(by_season[season][breakdown_index]):
                    by_kind.setdefault(classify_build(combination), []).append((combination, count))
                stacks.append([(kind, float(sum(count for _, count in members)),
                                f'{season} · {name_build_kind(kind)}: '
                                + ', '.join(f'{combination} x{count}' for combination, count in members))
                               for kind, members in sorted(by_kind.items())])
            name = {'MC': 'Most Categories', 'EC': 'Each Category'}[format_key]
            panels.append(f'<div><h3>{name}, {mode}</h3>'
                          + render_stacked_bars([season[2:] for season in seasons], stacks, 'Builds', 12.0) + '</div>')
    present = {classify_build(combination) for row in rows for combination, _ in read_breakdown(row[breakdown_index])}
    shown = [kind for kind in range(len(BUILD_KINDS)) if kind in present] + [NO_HARD_PUNT] + (
        [OTHER_BUILD] if OTHER_BUILD in present else [])
    legend = '<div class="legend">' + ''.join(
        f'<span><span class="swatch" style="background:{fill_build_kind(kind)}"></span>{escape(name_build_kind(kind))}'
        f'</span>' for kind in shown) + '</div>'
    all_counts = [read_leading_number(row[columns.index('Distinct builds')]) for row in rows]
    hard_counts = [count_distinct_builds(row[breakdown_index], True) for row in rows]
    fewest = min(all_counts)
    text = (f'In a season, the top-12 builds use <b>{fewest:.0f}-{max(all_counts):.0f}</b> distinct punt combinations '
            f'counting soft punts, and <b>{min(hard_counts)}-{max(hard_counts)}</b> counting hard punts alone (floor 3 '
            f'on the first: fewer would mean the field has collapsed onto one build).')
    note = ("<div class=\"note\">Each bar is a season's 12 builds, coloured by exactly what they hard punt (under 15%); "
            'soft punts are ignored here but listed in the table. Light shades are two hard punts, paired with the '
            'single punt of the same hue. Hover a segment for its exact combinations.</div>')
    return (render_takeaway('good' if fewest >= 3 else 'bad', 'passes' if fewest >= 3 else 'breach', text) + note
            + legend + f'<div class="multiples">{"".join(panels)}</div>')


# Rotisserie punting
def card_rotisserie(columns, rows):
    most = max(read_leading_number(row[2]) for row in rows)
    weakest = min(rows, key=lambda row: read_leading_number(row[5]))
    text = (f'At most <b>{most:.0f}</b> hard punt{"s" if most != 1 else ""} among the top-12 Rotisserie builds in any '
            f'season (ceiling 2). The weakest category anywhere: {escape(weakest[4])} at {escape(weakest[5])} '
            f'({escape(weakest[6])}, {escape(weakest[0])} {escape(weakest[1])}).')
    return render_takeaway('good' if most <= 2 else 'bad', 'passes' if most <= 2 else 'breach', text)


def rule_hard_punts(value, row, columns):
    return GOOD if value <= 2 else BAD


def rule_soft_punts(value, row, columns):
    ceiling = (5 if row[1] == 'auction' else 2) * 12
    return '' if value <= ceiling else BAD


def rule_weakest_rate(value, row, columns):
    """A hard punt is allowed (the ceiling counts them), so the weakest rate only warns."""
    return WARN if value < 15 else ''


# Predicted-pick stability
def find_column(columns, prefix):
    """The one column whose name starts with prefix (stored tables from before a rename carry the older name)."""
    matches = [index for index, name in enumerate(columns) if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f'{len(matches)} columns start with {prefix!r}: {columns}')
    return matches[0]


def render_strip_chart(
    x_labels: list[str]
    , series: list[tuple[str, list[list[float]]]]
    , y_title: str
    , value_suffix: str = ''
    , height: int = 240
) -> str:
    """Every value as its own dot: per x position, one column of dots per series (side by side), each dot's tooltip its
    value. series: [(name, values per x position)]."""
    width, left, right, top, bottom = 1100, 58, 120, 14, 30
    plot_width, plot_height = width - left - right, height - top - bottom
    values = [value for _, columns in series for column in columns for value in column] + [0.0]
    low, high = min(values), max(values)
    padding = (high - low) * 0.08 or 0.1
    low, high = low - padding, high + padding
    ticks = compute_nice_ticks(low, high)
    low, high = min(low, ticks[0]), max(high, ticks[-1])

    def y_of(value):
        return top + plot_height * (1 - (value - low) / (high - low))

    slot = plot_width / len(x_labels)
    parts = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(y_title)}">']
    for tick in ticks:
        parts.append(f'<line class="grid" x1="{left}" x2="{left + plot_width}" y1="{y_of(tick):.1f}" y2="{y_of(tick):.1f}"/>'
                     f'<text x="{left - 6}" y="{y_of(tick) + 3.5:.1f}" text-anchor="end">{tick + 0.0:g}'
                     f'{escape(value_suffix)}</text>')
    parts.append(f'<line class="reference" x1="{left}" x2="{left + plot_width}" y1="{y_of(0):.1f}" y2="{y_of(0):.1f}"/>')
    parts.append(f'<text x="14" y="{top + plot_height / 2:.0f}" transform="rotate(-90 14 {top + plot_height / 2:.0f})" '
                 f'text-anchor="middle">{escape(y_title)}</text>')
    for index, label in enumerate(x_labels):
        center = left + slot * (index + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top + plot_height + 18}" text-anchor="middle">{escape(label)}</text>')
        for series_index, (name, columns) in enumerate(series):
            x = center + (series_index - (len(series) - 1) / 2) * slot * 0.28
            for value in columns[index]:
                parts.append(f'<circle cx="{x:.1f}" cy="{y_of(value):.1f}" r="3.5" style="fill:var(--series-{series_index});'
                             f'fill-opacity:.7;stroke:var(--card);stroke-width:1"><title>{escape(name)} · {escape(label)}: '
                             f'{value:+.2f}{escape(value_suffix)}</title></circle>')
    parts.append('</svg>')
    legend = '<div class="legend">' + ''.join(
        f'<span><span class="swatch series-{index}"></span>{escape(name)}</span>'
        for index, (name, _) in enumerate(series)) + '</div>'
    return ''.join(parts) + legend


def find_column(columns, prefix):
    """The one column whose name starts with prefix (stored tables from before a rename carry the older name)."""
    matches = [index for index, name in enumerate(columns) if name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f'{len(matches)} columns start with {prefix!r}: {columns}')
    return matches[0]


def card_stability(columns, rows):
    seasons = sorted({row[0] for row in rows})
    move_index = find_column(columns, 'Max relative move')
    flips_index, worst_flip_index = columns.index('Punt flips'), columns.index('Worst punt flip')
    largest = max(read_leading_number(row[move_index]) for row in rows)
    flips = sum(read_leading_number(row[flips_index]) for row in rows)
    text = (f'After an opponent takes the player the model predicted, no candidate\'s H-score moves more than '
            f'<b>{largest:.2f}%</b> against the field (an H-score is an expected win rate, so that is in percentage '
            f'points), and <b>{flips:.0f}</b> build{"s" if flips != 1 else ""} change their punts (want 0).'
            + ('' if flips == 0 else ' Flips: ' + '; '.join(escape(f'{row[0]} {row[1]}: {row[worst_flip_index]}')
                                                           for row in rows if read_leading_number(row[flips_index]))
               + '.'))
    status = render_takeaway('good' if flips == 0 else 'bad', 'passes' if flips == 0 else 'punt flips', text)
    if 'Every relative move (H-score)' not in columns:
        return status
    every_index = columns.index('Every relative move (H-score)')
    series = []
    for mode in ('draft', 'auction'):
        by_season = {row[0]: [float(value) for value in row[every_index].split()] for row in rows if row[1] == mode}
        if by_season:
            series.append((mode, [by_season.get(season, []) for season in seasons]))
    note = ("<div class=\"note\">Every compared candidate's H-score move against the field (its change minus the "
            'shared shift), one dot each.</div>')
    return status + note + render_strip_chart([season[2:] for season in seasons], series, 'H-score move', '%')


def derive_stability_table(columns, rows):
    """The table without the per-candidate moves (they are the chart)."""
    if 'Every relative move (H-score)' not in columns:
        return columns, rows
    every_index = columns.index('Every relative move (H-score)')
    return ([name for index, name in enumerate(columns) if index != every_index],
            [[cell for index, cell in enumerate(row) if index != every_index] for row in rows])


# Self-play convergence
def group_self_play_cycles(columns, rows, run):
    """For one run (rows tagged with that Run; a table without a Run column is one run): per season, the neutral pass,
    then one row per cycle (two consecutive passes, which between them re-solve every tracked player exactly once, so
    cycle counts are comparable), then the serve. Returns (rows, cycle changes by season), the changes being how many
    punts differ from the previous cycle, summed over categories."""
    pass_index = columns.index('Pass')
    run_rows = [row for row in rows if 'Run' not in columns or row[columns.index('Run')] == run]
    grouped, changes = [], {}
    for season in sorted({row[0] for row in run_rows}):
        season_rows = [row for row in run_rows if row[0] == season]
        passes = [row for row in season_rows if row[pass_index] not in ('0', 'serve')]
        if len(passes) % 2:
            raise ValueError(f'{season} {run}: {len(passes)} self-play passes, not whole cycles of two')
        grouped += [row for row in season_rows if row[pass_index] == '0']
        cycles = []
        for first, second in zip(passes[0::2], passes[1::2]):
            counts = np.array([read_leading_number(a) + read_leading_number(b)
                               for a, b in zip(first[pass_index + 1:], second[pass_index + 1:])])
            cycles.append(counts)
            grouped.append(first[:pass_index] + [f'cycle {len(cycles)} (passes {first[pass_index]}-{second[pass_index]})']
                           + [f'{count:.0f}' for count in counts])
        grouped += [row for row in season_rows if row[pass_index] == 'serve']
        changes[season] = [float(np.abs(later - earlier).sum()) for earlier, later in zip(cycles, cycles[1:])]
    return grouped, changes


def list_self_play_runs(columns, rows):
    """The runs in the table, the app's first; a table without a Run column is the app's run alone."""
    if 'Run' not in columns:
        return ['app']
    return list(dict.fromkeys(row[columns.index('Run')] for row in rows))


def card_self_play(columns, rows):
    runs = list_self_play_runs(columns, rows)
    _, changes = group_self_play_cycles(columns, rows, runs[0])
    extended = group_self_play_cycles(columns, rows, runs[1])[1] if len(runs) > 1 else {}
    longest = max(len(values) for values in list(changes.values()) + list(extended.values()))
    early = np.mean([np.mean(values[:3]) for values in changes.values()])
    late = np.mean([np.mean(values[-5:]) for values in changes.values()])
    text = (f'From one cycle to the next, <b>{early:.0f}</b> punts change on average in the first three cycles '
            f'(summed over the top 12 players and every category) and <b>{late:.0f}</b> in the app\'s last five: the '
            f'field settles, with small residual motion, as a damped mixed equilibrium should.')
    if extended:
        extended_late = np.mean([np.mean(values[-5:]) for values in extended.values()])
        text += (f' Run for twice as many passes, the last five cycles change <b>{extended_late:.0f}</b> punts on '
                 f'average.')
    status = ('good', 'settles') if late < early / 2 else ('warn', 'not settling')
    series = [Series(season, values + [None] * (longest - len(values)))
              for season, values in sorted(changes.items())]
    chart = render_chart([str(index + 2) for index in range(longest)], series, 'Punts changed', kind='line',
                         value_places=0, y_floor=0.0,
                         background_series=[Series(f'{season} doubled', values + [None] * (longest - len(values)))
                                            for season, values in sorted(extended.items())])
    note = 'Punts changed from the previous cycle, by cycle and season.'
    if extended:
        note += ' Grey: the same seasons with the pass count doubled (its first cycles repeat the app\'s).'
    return render_takeaway(*status, text) + f'<div class="note">{note}</div>' + chart


def filter_self_play(columns, rows):
    runs = list_self_play_runs(columns, rows)
    grouped, _ = group_self_play_cycles(columns, rows, runs[0])
    return grouped, ('Each pass re-solves a random half of the players, so passes are paired into cycles that '
                     're-solve all 12 once; a cell counts the top 12 players punting that category. The table shows '
                     "the app's run.")


def styles_self_play(columns, rows):
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        for index in range(columns.index('Pass') + 1, len(row)):
            value = read_leading_number(row[index])
            if value:
                row_styles[index] = style_heat(value / 12)
        styles.append(row_styles)
    return styles


# Warm start
def card_warm_start(columns, rows):
    seasons = [row[0] for row in rows]
    differences = [read_leading_number(row[3]) for row in rows]
    ok = all(value >= -0.25 for value in differences)
    text = (f'Continuing the first pick from the self-play builds instead of the seed scan changes the top-12 mean '
            f'H-score by between <b>{min(differences):+.3f}%</b> and <b>{max(differences):+.3f}%</b> (floor −0.25%): '
            f'parity, as intended.')
    chart = render_chart([season[2:] for season in seasons], [Series('warm minus cold', differences)],
                         'Warm minus cold', kind='dots',
                         references=(ReferenceLine(0.0, 'parity'), ReferenceLine(-0.25, 'floor −0.25%', True)),
                         value_suffix='%', value_places=3)
    return render_takeaway('good' if ok else 'bad', 'passes' if ok else 'breach', text) + chart


# Multi-start seeding
def card_multi_start(columns, rows):
    seasons = [row[0] for row in rows]
    gains = [read_leading_number(row[3]) for row in rows]
    improved = [read_leading_number(row[5]) for row in rows]
    ok = all(value >= -0.1 for value in gains)
    text = (f'Seeding from the punt scan instead of a single neutral seed raises the top-12 mean H-score by '
            f'<b>{np.mean(gains):+.2f}%</b> on average ({min(gains):+.2f}% to {max(gains):+.2f}% by season), and '
            f'improves {np.mean(improved):.0f} of the 12 players in a typical season (floor −0.1%).')
    chart = render_chart([season[2:] for season in seasons], [Series('mean gain', gains)], 'Scan minus neutral',
                         kind='bars', references=(ReferenceLine(-0.1, 'floor −0.1%', True),),
                         value_suffix='%', value_places=2)
    return render_takeaway('good' if ok else 'bad', 'passes' if ok else 'breach', text) + chart


# Punt starts reaching the optimum
def card_punt_starts(columns, rows):
    summary = find_summary(rows, 'ALL')
    starts = [str(summary[columns.index(name)]) for name in ('Missed at 5%', 'Missed at 1%', 'Missed at 0%')]
    shares = [100 * read_leading_number(text.split('/')[0]) / read_leading_number(text.split('/')[1]) for text in starts]
    candidates = candidate_rows(rows)
    with_miss = [row for row in candidates if any(read_leading_number(row[columns.index(name)].split('/')[0]) > 0
                                                  for name in ('Missed at 5%', 'Missed at 1%', 'Missed at 0%'))]
    rounds = sorted({int(row[1]) for row in candidates})
    per_round = [sum(int(row[1]) == n_rounds for row in with_miss) / max(sum(int(row[1]) == n_rounds
                                                                              for row in candidates), 1) * 100
                 for n_rounds in rounds]
    text = (f'<b>{np.mean(shares):.2f}%</b> of punt starts end on a lower peak; <b>{len(with_miss)} of '
            f'{len(candidates)}</b> candidates have one. Most candidates are single-peaked, and multi-starting covers '
            f'the rest.')
    chart = render_chart([str(n_rounds) for n_rounds in rounds], [Series('candidates with a lower peak', per_round)],
                         'Candidates with a miss', kind='bars', value_suffix='%', value_places=0)
    status = ('good', 'mostly single-peaked') if np.mean(shares) < 5 else ('warn', 'many peaks')
    return render_takeaway(*status, text) + '<div class="note">By draft round (players on the board).</div>' + chart


def filter_punt_starts(columns, rows):
    names = ('Missed at 5%', 'Missed at 1%', 'Missed at 0%')
    kept = [row for row in rows if is_summary_row(row)
            or any(read_leading_number(row[columns.index(name)].split('/')[0]) > 0 for name in names)]
    omitted = len(rows) - len(kept)
    return kept, f'{omitted} candidates where every start reached the optimum are not listed.'


def rule_missed(value, row, columns):
    return GOOD if value == 0 else style_bad(0.6)


def rule_shortfall(value, row, columns):
    if value == 0:
        return GOOD
    return style_bad(min(value / 0.5, 1.0))


# Loop gain
def card_loop_gain(columns, rows):
    gains = [value for value in column_values(columns, candidate_rows(rows), 'Largest loop gain') if value is not None]
    edges = np.arange(0.0, 1.2, 0.1)
    counts, _ = np.histogram(gains, bins=edges)
    ok = max(gains) < 1
    text = (f'The largest loop gain at any of the {len(gains)} optima is <b>{max(gains):.3f}</b>: every one is below 1, '
            f'so a nudge to the weights shrinks on its way round and each optimum is a true peak.')
    chart = render_chart([f'{low:.1f}–{low + 0.1:.1f}' for low in edges[:-1]], [Series('optima', list(counts))],
                         'Optima', kind='bars', value_places=0)
    return (render_takeaway('good' if ok else 'bad', 'passes' if ok else 'breach', text)
            + '<div class="note">How many optima have each largest loop gain (1 is the limit).</div>' + chart)


def rule_loop_gain(value, row, columns):
    return BAD if value >= 1 else (WARN if value >= 0.8 else GOOD)


def sort_by_loop_gain(columns, rows):
    index = columns.index('Largest loop gain')
    return sorted(rows, key=lambda row: -(read_leading_number(row[index]) or 0))


# Quasiconcavity
def card_quasiconcavity(columns, rows):
    rounds = sorted({int(row[1]) for row in rows})
    kinds = ['near the optimum', 'between punts', 'anywhere']
    series = []
    for kind in kinds:
        values = []
        for n_rounds in rounds:
            matching = [read_leading_number(row[3]) for row in rows if row[2] == kind and int(row[1]) == n_rounds]
            values.append(float(np.mean(matching)) if matching else None)
        series.append(Series(kind, values))
    overall = {kind: np.mean([read_leading_number(row[3]) for row in rows if row[2] == kind]) for kind in kinds}
    text = (f'Straight segments that dip below both ends: <b>{overall["near the optimum"]:.2f}%</b> near the optimum, '
            f'{overall["between punts"]:.1f}% between punts, {overall["anywhere"]:.1f}% anywhere. The surface is not '
            f'quasiconcave far out, but near the peak it almost never dips.')
    status = ('good', 'concave near the peak') if overall['near the optimum'] < 1 else ('warn', 'dips near the peak')
    chart = render_chart([str(n_rounds) for n_rounds in rounds], series, 'Segments dipping', kind='line',
                         value_suffix='%', value_places=2, y_floor=0.0)
    return render_takeaway(*status, text) + '<div class="note">By draft round, mean over seasons.</div>' + chart


def rule_dipping(value, row, columns):
    """Only dips near the optimum would contradict anything; far-out dips are expected and stay uncoloured."""
    if row[columns.index('Segments')] != 'near the optimum':
        return ''
    return GOOD if value == 0 else style_bad(min(value / 1.0, 1.0))


# Planner vs H-scoring
def card_planner(columns, rows):
    names = [row[0] for row in rows]
    means = [read_leading_number(row[3]) for row in rows]
    errors = [2 * read_leading_number(row[4]) for row in rows]
    wins = [str(row[5]) for row in rows]
    text = ('Planning concrete teams loses to H-scoring\'s smooth model in every pair: '
            + '; '.join(f'{escape(name)} <b>{mean:+.2f}%</b> (wins {escape(win)})'
                        for name, mean, win in zip(names, means, wins)) + '. Bars are ±2 standard errors.')
    chart = render_chart(names, [Series('planner minus H-scoring', means, errors)], 'Planner minus H', kind='dots',
                         references=(ReferenceLine(0.0, 'even'),), value_suffix='%', value_places=2, height=200)
    return render_takeaway('good', 'H-scoring wins', text) + chart


def rule_planner(value, row, columns):
    """No floor here: a negative number is H-scoring winning, the expected result. Only a planner win is marked."""
    return WARN if value > 0 else ''


# Planner vs H-scoring: individual drafts
def compute_seat_differences(columns, rows):
    """Per planner variant, per seat: the seat's planner score minus its H-scoring score within each pair (each seat
    drafts once with each method in a pair). Returns {variant: (S, pairs) array}."""
    seat_columns = [name for name in columns if re.fullmatch(r'S\d+', name)]
    differences = {}
    for variant in dict.fromkeys(row[0] for row in rows):
        pairs = {}
        for row in rows:
            if row[0] == variant:
                pairs.setdefault(row[1], []).append(row)
        per_pair = []
        for drafts in pairs.values():
            if len(drafts) != 2:
                raise ValueError(f'{variant} pair {drafts[0][1]}: {len(drafts)} drafts, expected 2')
            seat_difference = []
            for name in seat_columns:
                cells = {draft[columns.index(name)].split(' ')[0]: read_leading_number(draft[columns.index(name)])
                         for draft in drafts}
                seat_difference.append(cells['P'] - cells['H'])
            per_pair.append(seat_difference)
        differences[variant] = np.array(per_pair).T
    return differences


def card_planner_drafts(columns, rows):
    differences = compute_seat_differences(columns, rows)
    seat_count = next(iter(differences.values())).shape[0]
    series = [Series(variant, list(values.mean(axis=1)),
                     list(2 * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])))
              for variant, values in differences.items()]
    worst = {variant: int(np.argmin(values.mean(axis=1))) + 1 for variant, values in differences.items()}
    best = {variant: int(np.argmax(values.mean(axis=1))) + 1 for variant, values in differences.items()}
    text = ('Each seat against itself, planner minus H-scoring: '
            + '; '.join(f'{escape(variant)} from <b>{values.mean(axis=1).min():+.2f}%</b> (seat {worst[variant]}) to '
                        f'<b>{values.mean(axis=1).max():+.2f}%</b> (seat {best[variant]})'
                        for variant, values in differences.items()) + '. Bars are ±2 standard errors over the pairs.')
    winning = any((values.mean(axis=1) > 0).any() for values in differences.values())
    status = ('warn', 'planner wins somewhere') if winning else ('good', 'H-scoring wins every seat')
    chart = render_chart([str(seat + 1) for seat in range(seat_count)], series, 'Planner minus H', kind='dots',
                         references=(ReferenceLine(0.0, 'even'),), value_suffix='%', value_places=2)
    return render_takeaway(*status, text) + '<div class="note">By draft position.</div>' + chart


def derive_planner_pair_table(columns, rows):
    """One row per pair: the pair's planner-minus-H-scoring mean, and per seat that seat's planner score minus its
    H-scoring score (each seat drafts once with each method in a pair)."""
    seat_columns = [name for name in columns if re.fullmatch(r'S\d+', name)]
    differences = compute_seat_differences(columns, rows)
    difference_index = columns.index('Planner minus H (pp)')
    pair_rows = []
    for variant in dict.fromkeys(row[0] for row in rows):
        pairs = list(dict.fromkeys(row[1] for row in rows if row[0] == variant))
        for position, pair in enumerate(pairs):
            drafts = [row for row in rows if row[0] == variant and row[1] == pair]
            pair_mean = float(np.mean([read_leading_number(draft[difference_index]) for draft in drafts]))
            pair_rows.append([variant, pair, f'{pair_mean:+.2f}']
                             + [f'{differences[variant][seat, position]:+.2f}' for seat in range(len(seat_columns))])
    return ['Planner', 'Pair', 'Planner minus H (pp)'] + seat_columns, pair_rows


def styles_planner_pairs(columns, rows):
    """A diverging tint per difference: blue where H-scoring won, orange where the planner won, deeper for more
    (full depth at 3 points)."""
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        for index in range(columns.index('Planner minus H (pp)'), len(row)):
            value = read_leading_number(row[index])
            depth = 0.08 + 0.42 * min(abs(value) / 3.0, 1.0)
            color = 'var(--heat-rgb)' if value < 0 else '230,159,0'
            row_styles[index] = f' style="background:rgba({color},{depth:.2f})"'
        styles.append(row_styles)
    return styles


# Auction cash-share rules
RULE_LABELS = ('smooth', 'simple scaling')
AUCTION_ROUNDS = 13


def read_rule_summaries(columns, rows):
    """{softness: its ALL row} in the rules section."""
    return {row[0]: row for row in rows if is_summary_row(row)}


def card_rules_head_to_head(columns, rows):
    """The answer first -- the best softness's own-view gain, how many seasons it wins, the money left unspent -- then
    the evidence: the gain season by season, where the smooth drafters' money went round by round, and what each rule
    left at the end."""
    summaries = read_rule_summaries(columns, rows)
    softnesses = list(summaries)

    def read(softness, name):
        return read_leading_number(summaries[softness][columns.index(name)])

    best = max(softnesses, key=lambda softness: read(softness, 'Own view gap'))
    worst = min(softnesses, key=lambda softness: read(softness, 'Own view gap'))
    best_gap, best_error = read(best, 'Own view gap'), read(best, 'Own view SE')
    worst_gap, worst_error = read(worst, 'Own view gap'), read(worst, 'Own view SE')
    seasons = sorted({row[1] for row in candidate_rows(rows)})
    by_season = {(row[0], row[1]): row for row in candidate_rows(rows)}
    seasons_ahead = sum(1 for season in seasons
                        if read_leading_number(by_season[(best, season)][columns.index('Own view gap')]) > 0)

    def judge(gap, error):
        return 'good' if gap - 2 * error > 0 else 'bad' if gap + 2 * error < 0 else 'warn'

    tiles = render_stat_tiles([
        (f'{best_gap:+.2f}', f'own-view H-score points per team for the smooth rule at s = {best} '
                             f'(±{2 * best_error:.2f})', judge(best_gap, best_error)),
        (f'{seasons_ahead} of {len(seasons)}', f'seasons in which s = {best} comes out ahead', ''),
        (f'${read(best, "Unspent (smooth)"):.2f} vs ${read(best, "Unspent (simple scaling)"):.2f}',
         f'left unspent per team at the end: smooth (s = {best}) vs simple scaling', ''),
        (f'{worst_gap:+.2f}', f'at s = {worst}: it spends too much too early', judge(worst_gap, worst_error)),
    ])
    clear = best_gap - 2 * best_error > 0
    status = ('good', 'smooth ahead') if clear else ('warn', 'no clear winner')
    text = (f'Spending surplus cash smoothly, from a little before it is surplus, beats waiting for the surplus: at '
            f'<b>s = {escape(best)}</b> the smooth drafters finish ahead of the simple-scaling drafters in their own '
            f'view in {seasons_ahead} of {len(seasons)} seasons and leave a fraction of the money on the table. Softer '
            f'is not better without limit: at s = {escape(worst)} they overpay early and lose.')

    groups = seasons + ['All']

    def read_season(softness, season, name):
        row = summaries[softness] if season == 'All' else by_season[(softness, season)]
        return read_leading_number(row[columns.index(name)])

    gap_chart = render_chart(
        [season[2:] if season != 'All' else season for season in groups],
        [Series(f's = {softness}', [read_season(softness, season, 'Own view gap') for season in groups],
                [2 * read_season(softness, season, 'Own view SE') for season in groups])
         for softness in softnesses],
        'Own-view H-score points, smooth - simple', kind='bars', references=(ReferenceLine(0.0, 'no difference'),),
        value_places=2, height=250)

    def spending_gap(softness, round_number):
        return (read(softness, f'Spent after round {round_number} (smooth)')
                - read(softness, f'Spent after round {round_number} (simple scaling)'))

    rounds = list(range(1, AUCTION_ROUNDS + 1))
    spending_chart = render_chart(
        [str(round_number) for round_number in rounds],
        [Series(f's = {softness}', [spending_gap(softness, round_number) for round_number in rounds])
         for softness in softnesses],
        '$ spent so far, smooth - simple', references=(ReferenceLine(0.0, 'even'),), value_places=1, height=280,
        width=540)
    unspent_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(label, [read(softness, f'Unspent ({label})') for softness in softnesses]) for label in RULE_LABELS],
        '$ left per team', kind='bars', value_places=2, height=280, width=540)
    return (render_takeaway(*status, text) + tiles
            + '<h3>Does it win every season?</h3><div class="note">Own view, each season and pooled; whiskers are ±2 '
              'standard errors across 40 pairs of auctions per season.</div>' + gap_chart
            + '<div class="multiples"><div><h3>Where the money goes</h3><div class="note">How much more the smooth '
              'drafters had spent by the end of each round (one sale per team).</div>' + spending_chart + '</div>'
            + '<div><h3>Money left at the end</h3><div class="note">Per team, in the same auctions.</div>'
            + unspent_chart + '</div></div>')


def derive_rules_table(columns, rows):
    """The rules table as shown: each gap with its standard error, and cash left and paid extra by rule (the
    by-round spending is in the chart)."""
    shown = ['Softness', 'Season', 'Pairs', 'Own view gap ± SE', 'Actual gap ± SE', 'Unspent (smooth)',
             'Unspent (simple scaling)', 'Paid extra (smooth)', 'Paid extra (simple scaling)']

    def cell(row, name):
        if name.endswith(' ± SE'):
            gap_name = name.removesuffix(' ± SE')
            return f'{row[columns.index(gap_name)]} ± {row[columns.index(gap_name.replace("gap", "SE"))]}'
        return row[columns.index(name)]

    return shown, [[cell(row, name) for name in shown] for row in rows]


def styles_rules_table(columns, rows):
    """A gap in green when it is more than two standard errors above zero, red when more than two below."""
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        for name in ('Own view gap ± SE', 'Actual gap ± SE'):
            index = columns.index(name)
            gap, error = (float(part) for part in str(row[index]).split(' ± '))
            if gap - 2 * error > 0:
                row_styles[index] = GOOD
            elif gap + 2 * error < 0:
                row_styles[index] = BAD
        styles.append(row_styles)
    return styles


def card_rule_builds(columns, rows):
    """Whether bidding up stars costs a coherent build: each rule's misfits and value in punted categories, side by
    side at each softness."""
    softnesses = sorted({row[0] for row in rows})
    by_key = {(row[0], row[1]): row for row in rows}

    def read(softness, rule, name):
        return read_leading_number(by_key[(softness, rule)][columns.index(name)])

    no_cost = [softness for softness in softnesses
               if read(softness, 'smooth', 'Misfits per team') <= read(softness, 'simple scaling', 'Misfits per team')
               and read(softness, 'smooth', 'Value in punted') <= read(softness, 'simple scaling', 'Value in punted')]
    tiles = render_stat_tiles([
        (f'{read(softness, "smooth", "Misfits per team"):.2f} vs {read(softness, "simple scaling", "Misfits per team"):.2f}',
         f'misfits per team at s = {softness}, smooth vs simple scaling',
         'good' if softness in no_cost else 'warn')
        for softness in softnesses])
    text = ('Bidding up the stars does not cost the smooth drafters their builds: they put the same share of their '
            'value into the categories they punt as the simple-scaling drafters, and carry about as many misfits'
            + (f' (no more at s = {", ".join(no_cost)}).' if no_cost else '.'))
    misfit_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(rule, [read(softness, rule, 'Misfits per team') for softness in softnesses]) for rule in RULE_LABELS],
        'Misfits per team', kind='bars', value_places=2, height=260, width=540)
    punted_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(rule, [read(softness, rule, 'Value in punted') for softness in softnesses]) for rule in RULE_LABELS],
        'Value in punted categories', kind='bars', value_suffix='%', value_places=0, height=260, width=540)
    return (render_takeaway('good' if no_cost else 'warn', 'builds hold' if no_cost else 'builds slip', text) + tiles
            + '<div class="multiples"><div><h3>Misfits</h3><div class="note">Players with at least half their value '
              'in categories the team punts.</div>' + misfit_chart + '</div>'
            + '<div><h3>Value in punted categories</h3><div class="note">Share of the team\'s positive value; a '
              'build bought regardless of its punts would put about 17% there.</div>' + punted_chart + '</div></div>')


# Auction dollar values
DOLLAR_LEVELS = ('off', 'medium', 'high')


def read_dollar_shape_summaries(columns, rows):
    """{(replacement level, SAVOR level): its ALL row}."""
    return {(row[1], row[3]): row for row in rows if is_summary_row(row)}


DOLLAR_TIERS = ('Top 10%', 'Middle 80%', 'Bottom 10%')


# The grid's medium settings are the app's defaults (test_experiments._REPLACEMENT_LEVELS and _SAVOR_NOISES).
APP_DOLLAR_CELL = ('medium', 'medium')


def card_dollar_shape(columns, rows):
    """What the app's setting does to the dollar curve, then how each adjustment adds to it: each rank's change in
    percent, against neither adjustment."""
    summaries = read_dollar_shape_summaries(columns, rows)

    def read(replacement, savor, name):
        return read_leading_number(summaries[(replacement, savor)][columns.index(name)])

    def describe(replacement, savor):
        return ' / '.join(f'{read(replacement, savor, f"{tier} change"):+.0f}%' for tier in DOLLAR_TIERS)

    app_replacement, app_savor = APP_DOLLAR_CELL
    threshold = read(app_replacement, app_savor, 'Replacement threshold')
    noise = read(app_replacement, app_savor, 'SAVOR noise')
    tiles = render_stat_tiles([
        (f'{read(app_replacement, app_savor, f"{tier} change"):+.0f}%',
         f'{tier.lower()} of the drafted players, at the app\'s setting (S = {noise:g}, S_R = ${threshold:g})',
         'good' if tier == DOLLAR_TIERS[0] else '')
        for tier in DOLLAR_TIERS])
    text = (f'SAVOR moves money from the bottom of the draft to the top, and the replacement threshold adds to it. At '
            f'the app\'s noise (S = {noise:g}), vanilla SAVOR moves the top 10% / middle 80% / bottom 10% by '
            f'<b>{describe("off", app_savor)}</b>; with the app\'s threshold of ${threshold:g}, '
            f'<b>{describe(app_replacement, app_savor)}</b>. With no noise a threshold simply cuts everyone below it to '
            f'zero (the first column of the grid); noise softens that, since a cheap player may still clear it.')
    rank_columns = [name for name in columns if name.startswith('Rank ')]
    ranks = [name.split()[1] for name in rank_columns]

    def percent_change(replacement, savor):
        """Each rank's dollars in this cell, in percent against neither adjustment."""
        return [100 * (read(replacement, savor, name) / read('off', 'off', name) - 1) for name in rank_columns]

    cells = [('off', app_savor, f'vanilla SAVOR (S = {noise:g})'),
             (app_replacement, app_savor, f'the app: S = {noise:g}, S_R = ${threshold:g}'),
             ('high', app_savor, f'S = {noise:g}, S_R = ${read("high", app_savor, "Replacement threshold"):g}'),
             (app_replacement, 'high', f'S = {read(app_replacement, "high", "SAVOR noise"):g}, S_R = ${threshold:g}')]
    shape_chart = render_chart(
        ranks, [Series(label, percent_change(replacement, savor)) for replacement, savor, label in cells],
        '% change in dollars', references=(ReferenceLine(0.0, 'neither'),), value_suffix='%', value_places=0,
        height=280, show_points=False)
    return (render_takeaway('good', 'shape', text) + tiles
            + '<h3>Who gains and who pays</h3><div class="note">Each drafted rank\'s dollars at the start of an '
              'auction, in percent against neither adjustment, mean over the six seasons. The top ranks gain what the '
              'bottom ones lose.</div>' + shape_chart)


def styles_dollar_shape_table(columns, rows):
    """Each cell shaded by how much its top 10% gains (deepest at the largest gain in the grid); the app's cell bold."""
    gains = {}
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row[1:], 1):
            if cell.endswith('%'):
                gains[(row_index, column_index)] = read_leading_number(cell)
    largest = max(gains.values())
    styles = []
    for row_index, row in enumerate(rows):
        row_styles = [''] * len(row)
        for column_index in range(1, len(row)):
            gain = gains.get((row_index, column_index))
            shade = style_heat(gain / largest) if gain is not None else ''
            if row[0].startswith(APP_DOLLAR_CELL[0]) and columns[column_index].startswith(f'SAVOR {APP_DOLLAR_CELL[1]}'):
                shade = shade[:-1] + ';font-weight:700"' if shade else ' style="font-weight:700"'
            row_styles[column_index] = shade
        styles.append(row_styles)
    return styles


def derive_dollar_shape_table(columns, rows):
    """The 3x3 grid over the seasons: replacement level down, SAVOR across. Each cell is the average player's dollars in
    the top 10% / middle 80% / bottom 10% of the drafted players: with both off in dollars, elsewhere their change
    against it in percent."""
    summaries = read_dollar_shape_summaries(columns, rows)

    def cell(replacement, savor):
        row = summaries[(replacement, savor)]
        if (replacement, savor) == ('off', 'off'):
            return ' / '.join(f'${float(row[columns.index(f"{tier} $ per player")]):.2f}' for tier in DOLLAR_TIERS)
        return ' / '.join(f'{read_leading_number(row[columns.index(f"{tier} change")]):+.0f}%'
                          for tier in DOLLAR_TIERS)

    def label(level, name, prefix):
        row = next(row for key, row in summaries.items() if key[0 if prefix == 'replacement' else 1] == level)
        value = float(row[columns.index(name)])
        app = ', the app' if level == (APP_DOLLAR_CELL[0] if prefix == 'replacement' else APP_DOLLAR_CELL[1]) else ''
        return f'{level} (${value:g}{app})' if prefix == 'replacement' else f'SAVOR {level} (S = {value:g}{app})'

    shown = ['Replacement level'] + [label(level, 'SAVOR noise', 'savor') for level in DOLLAR_LEVELS]
    return shown, [[label(replacement, 'Replacement threshold', 'replacement')]
                   + [cell(replacement, savor) for savor in DOLLAR_LEVELS] for replacement in DOLLAR_LEVELS]


# Where the app loses to the optimum
LOSS_COLUMNS = ['Seed choice', 'Seed menu', 'Iterations', 'Remaining gap', 'Total loss']


def card_loss(columns, rows):
    summary = find_summary(rows, ('ALL', 'Mean of'))
    cost_row = find_summary(rows, 'Cost to fix')
    candidates = candidate_rows(rows)
    total = read_leading_number(summary[columns.index('Total loss')])
    loss_sum = sum(read_leading_number(row[columns.index('Total loss')]) for row in candidates)
    segments, legend = [], []
    for position, cause in enumerate(LOSS_COLUMNS[:-1]):
        cause_sum = sum(read_leading_number(row[columns.index(cause)]) for row in candidates)
        share = round(100 * cause_sum / loss_sum) if loss_sum else 0
        segments.append(f'<div class="bar-segment series-{position}" style="flex-basis:{share:g}%" '
                        f'title="{escape(cause)}: {share:g}% of the loss"></div>')
        cost = f' &middot; fix: {escape(cost_row[columns.index(cause)])} the work' if cost_row else ''
        legend.append(f'<span><span class="swatch series-{position}"></span>{escape(cause)}{cost}</span>')
    average_app = np.mean([read_leading_number(row[columns.index('App')]) for row in candidates])
    headline = (f'<div class="takeaway">Average total loss: <b>{total:.4f}%</b>, off an average H-score of '
                f'{average_app:.2f}% &middot; {escape(summary[0])}</div>')
    return headline + f'<div class="bar-track">{"".join(segments)}</div><div class="legend">{"".join(legend)}</div>'


def styles_loss(columns, rows):
    """Green for no loss at all; red strengthening with the loss (in the summary row, with its share of the total),
    bold in the top half of the scale, which tops out at the 95th percentile of the losses; a gain stays uncoloured."""
    indices = [columns.index(name) for name in LOSS_COLUMNS if name in columns]
    total_index = columns.index('Total loss')
    losses = [read_leading_number(row[index]) for row in rows if not is_summary_row(row) for index in indices]
    losses = [loss for loss in losses if loss is not None and loss > 0]
    scale_top = float(np.percentile(losses, 95)) if losses else 1.0
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        for index in indices:
            loss = read_leading_number(row[index])
            if loss is None:
                continue
            if is_summary_row(row):
                total = read_leading_number(row[total_index])
                if not total or index == total_index or row[0].startswith('Cost to fix'):
                    continue
                row_styles[index] = style_bad(loss / total)
            elif loss == 0:
                row_styles[index] = GOOD
            elif loss > 0:
                row_styles[index] = style_bad(min(loss / scale_top, 1.0))
        styles.append(row_styles)
    return styles


SECTION_VIEWS: dict[str, SectionView] = {
    'Smooth vs simple scaling': SectionView(card=card_rules_head_to_head, derive_table=derive_rules_table,
                                            cell_styles=styles_rules_table),
    'Cash-share rules: builds (own view)': SectionView(card=card_rule_builds, wrap_columns=('Most common punts',)),
    'Dollar shape: replacement level and SAVOR': SectionView(card=card_dollar_shape,
                                                             derive_table=derive_dollar_shape_table,
                                                             cell_styles=styles_dollar_shape_table),
    'H-scoring vs a G-score field': SectionView(
        card=card_h_versus_g,
        cell_styles=make_rule_styles({'Mean win rate': rule_season_win_rate,
                                      **{f'S{seat}': rule_seat_win_rate for seat in range(1, 13)}}),
        formatters={'Mean win rate': format_percent, **{f'S{seat}': format_percent for seat in range(1, 13)}},
    ),
    'Awareness (opponent model on vs off)': SectionView(
        card=card_awareness,
        cell_styles=make_rule_styles({'Mean (pp)': rule_pooled_significance}),
        formatters={'Mean (pp)': lambda cell: format_percent(cell, signed=True), 'SE (pp)': format_percent,
                    **{f'S{seat}': (lambda cell: format_percent(cell, signed=True)) for seat in range(1, 13)}},
    ),
    'Punt diversity (top-12 anchors)': SectionView(
        card=card_punt_diversity,
        cell_styles=make_rule_styles({'Builds (all punts)': lambda value, row, columns: GOOD if value >= 3 else BAD}),
        wrap_columns=('Builds (bold = hard punt)',),
        derive_table=derive_punt_diversity_table,
        html_formatters={'Builds (bold = hard punt)': render_build_chips},
    ),
    'Rotisserie punting (hard <15% / soft 15-40% expected win rate)': SectionView(
        card=card_rotisserie,
        cell_styles=make_rule_styles({'Hard': rule_hard_punts, 'Soft': rule_soft_punts,
                                      'Weakest rate': rule_weakest_rate}),
    ),
    'Predicted-pick stability (EC; opponent takes its predicted player)': SectionView(
        card=card_stability,
        cell_styles=make_rule_styles({
            'Max dweight': lambda value, row, columns: GOOD if value <= 5 else BAD,
            'Punt flips': lambda value, row, columns: GOOD if value == 0 else BAD,
            'Shared shift (H-score)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
            'Shared shift (whole board)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
        }),
        formatters={'Shared shift (H-score)': lambda cell: format_percent(cell, signed=True),
                    'Max relative move (H-score)': format_percent},
        derive_table=derive_stability_table,
    ),
    'Self-play convergence (MC bootstrap)': SectionView(card=card_self_play, cell_styles=styles_self_play,
                                                        row_filter=filter_self_play),
    'Warm start (purpose: display stability, not better answers)': SectionView(
        card=card_warm_start,
        cell_styles=make_rule_styles({'Warm - cold': lambda value, row, columns: GOOD if value >= -0.25 else BAD}),
        formatters={'Warm mean H': format_percent, 'Cold mean H': format_percent,
                    'Warm - cold': lambda cell: format_percent(cell, signed=True)},
    ),
    'Multi-start seeding': SectionView(
        card=card_multi_start,
        cell_styles=make_rule_styles({'Mean gain (pp)': rule_signed_gain(-0.1), 'Max gain (pp)': rule_signed_gain(-0.1),
                                      'Neutral arm collapsed (NaN)': lambda value, row, columns: '' if value == 0 else WARN}),
        formatters={'Scan mean H': format_percent, 'Neutral mean H': format_percent,
                    'Mean gain (pp)': lambda cell: format_percent(cell, signed=True),
                    'Max gain (pp)': lambda cell: format_percent(cell, signed=True)},
    ),
    'Punt starts reaching the optimum': SectionView(
        card=card_punt_starts,
        cell_styles=make_rule_styles({name: rule for depth in ('5%', '1%', '0%')
                                      for name, rule in ((f'Missed at {depth}', rule_missed),
                                                         (f'Worst miss (pp) at {depth}', rule_shortfall))}),
        formatters={f'Worst miss (pp) at {depth}': format_percent for depth in ('5%', '1%', '0%')},
        row_filter=filter_punt_starts,
    ),
    'Loop gain at the optimum': SectionView(
        card=card_loop_gain,
        cell_styles=make_rule_styles({'Largest loop gain': rule_loop_gain,
                                      'Largest Hessian eigenvalue':
                                          lambda value, row, columns: GOOD if value < 0 else BAD}),
        sort_rows=sort_by_loop_gain,
    ),
    'Quasiconcavity along weight segments': SectionView(
        card=card_quasiconcavity,
        cell_styles=make_rule_styles({'Dipping': rule_dipping}),
        formatters={'Median dip (pp)': format_percent, 'Largest dip (pp)': format_percent},
    ),
    'Where the app loses to the optimum': SectionView(card=card_loss, cell_styles=styles_loss),
    'Planner vs H-scoring: individual drafts': SectionView(
        card=card_planner_drafts,
        cell_styles=styles_planner_pairs,
        derive_table=derive_planner_pair_table,
        formatters={'Planner minus H (pp)': lambda cell: format_percent(cell, signed=True),
                    **{f'S{seat}': (lambda cell: format_percent(cell, signed=True)) for seat in range(1, 13)}},
        row_filter=lambda columns, rows: (rows, 'Each cell is that seat drafting with the planner minus the same seat '
                                                'drafting by H-scoring, in the same pair. Blue: H-scoring better; '
                                                'orange: the planner better.'),
    ),
    'Planner vs H-scoring (paired 6v6 drafts)': SectionView(
        card=card_planner,
        cell_styles=make_rule_styles({'Planner minus H (pp)': rule_planner, 'Worst (pp)': rule_planner,
                                      'Best (pp)': rule_planner}),
        formatters={'Planner minus H (pp)': lambda cell: format_percent(cell, signed=True), 'SE (pp)': format_percent,
                    'Worst (pp)': lambda cell: format_percent(cell, signed=True),
                    'Best (pp)': lambda cell: format_percent(cell, signed=True)},
    ),
}

# Column headers shown without the '(pp)' unit: the cells carry '%'.
HEADER_RENAMES = {'Mean (pp)': 'Mean', 'SE (pp)': 'SE', 'Mean gain (pp)': 'Mean gain', 'Max gain (pp)': 'Max gain',
                  'Planner minus H (pp)': 'Planner minus H', 'Worst (pp)': 'Worst', 'Best (pp)': 'Best',
                  'Median dip (pp)': 'Median dip', 'Largest dip (pp)': 'Largest dip',
                  'Worst miss (pp) at 5%': 'Worst miss at 5%', 'Worst miss (pp) at 1%': 'Worst miss at 1%',
                  'Worst miss (pp) at 0%': 'Worst miss at 0%'}


# ── The page ──────────────────────────────────────────────────────────────────

def render_explanation_html(explanation: str) -> str:
    """Each line a paragraph, and runs of lines starting '- ' a list."""
    parts, items = [], []
    for line in explanation.split('\n'):
        if line.startswith('- '):
            items.append(f'<li>{escape(line[2:])}</li>')
            continue
        if items:
            parts.append(f'<ul class="explain">{"".join(items)}</ul>')
            items = []
        if line.strip():
            parts.append(f'<p class="explain">{escape(line)}</p>')
    if items:
        parts.append(f'<ul class="explain">{"".join(items)}</ul>')
    return '\n'.join(parts)


def render_table(columns: list, rows: list, view: Optional[SectionView]) -> str:
    """Summary rows first, then the rest (in the view's order); cells formatted and coloured by the view."""
    summary_rows = [row for row in rows if is_summary_row(row)]
    other_rows = [row for row in rows if not is_summary_row(row)]
    if view is not None and view.sort_rows is not None:
        other_rows = view.sort_rows(columns, other_rows)
    ordered = summary_rows + other_rows
    styles = (view.cell_styles(columns, ordered) if view is not None and view.cell_styles is not None
              else [[''] * len(row) for row in ordered])
    formatters = view.formatters if view is not None else {}
    html_formatters = view.html_formatters if view is not None else {}
    wrap_columns = view.wrap_columns if view is not None else ()
    head = ''.join(f'<th>{escape(HEADER_RENAMES.get(name, name))}</th>' for name in columns)
    body = []
    for row, row_styles in zip(ordered, styles):
        cells = []
        for name, cell, style in zip(columns, row, row_styles):
            css = ' class="wrap"' if name in wrap_columns else ''
            if name in html_formatters:
                cells.append(f'<td{css}{style}>{html_formatters[name](cell)}</td>')
                continue
            text = formatters[name](cell) if name in formatters else str(cell)
            cells.append(f'<td{css}{style}>{escape(text)}</td>')
        body.append(('<tr class="summary">' if is_summary_row(row) else '<tr>') + ''.join(cells) + '</tr>')
    return f'<div class="tablewrap"><table><tr>{head}</tr>{"".join(body)}</table></div>'


def render_section_body(data: dict) -> str:
    """The takeaway card and the table for one stored section."""
    if 'columns' not in data:
        entries = ''.join(f'<li>{escape(entry)}</li>' for entry in data.get('entries', []))
        return f'<ul>{entries}</ul>'
    view = SECTION_VIEWS.get(data['title'])
    columns, rows = data['columns'], data['rows']
    card = f'<div class="card">{view.card(columns, rows)}</div>' if view is not None else ''
    if view is not None and view.derive_table is not None:
        columns, rows = view.derive_table(columns, rows)
    note = ''
    if view is not None and view.row_filter is not None:
        rows, note_text = view.row_filter(columns, rows)
        note = f'<div class="note">{escape(note_text)}</div>'
    return card + note + render_table(columns, rows, view)


def render_section_panel(data: dict, position: int) -> str:
    """One tab's panel: title, generation time, explanation, takeaway card, table."""
    active = ' class="active"' if position == 0 else ''
    return (f'<section id="tab-{position}"{active}>\n<h2>{escape(data["title"])}</h2>\n'
            f'<div class="generated">generated {escape(data["generated"])}</div>\n'
            f'{render_explanation_html(data["explanation"])}\n{render_section_body(data)}\n</section>')


# The tab groups, in page order; a stored section in none of them is listed last under 'Other'.
SECTION_GROUPS = {
    'Draft performance simulations': [
        'H-scoring vs a G-score field',
        'Awareness (opponent model on vs off)',
        'Planner vs H-scoring (paired 6v6 drafts)',
        'Planner vs H-scoring: individual drafts',
    ],
    'Algorithm behavior': [
        'Self-play convergence (MC bootstrap)',
        'Punt diversity (top-12 anchors)',
        'Rotisserie punting (hard <15% / soft 15-40% expected win rate)',
        'Predicted-pick stability (EC; opponent takes its predicted player)',
        'Warm start (purpose: display stability, not better answers)',
    ],
    'Gradient descent optimality': [
        'Where the app loses to the optimum',
        'Multi-start seeding',
        'Punt starts reaching the optimum',
        'Loop gain at the optimum',
        'Quasiconcavity along weight segments',
    ],
    'Auctions': [
        'Smooth vs simple scaling',
        'Cash-share rules: builds (own view)',
        'Dollar shape: replacement level and SAVOR',
    ],
}


def group_sections(sections: list[dict]) -> list[tuple[str, list[dict]]]:
    """The stored sections under their group names, in SECTION_GROUPS order."""
    by_title = {data['title']: data for data in sections}
    groups = [(name, [by_title.pop(title) for title in titles if title in by_title])
              for name, titles in SECTION_GROUPS.items()]
    groups.append(('Other', list(by_title.values())))
    return [(name, members) for name, members in groups if members]


# The page's tab switching: a group tab shows its row of section tabs and the section last open in it.
PAGE_SCRIPT = """
function showSection(button) {
  document.querySelectorAll('.tab-buttons button, section').forEach(function (node) {
    node.classList.remove('active');
  });
  button.classList.add('active');
  button.parentElement.dataset.lastTab = button.dataset.tab;
  document.getElementById('tab-' + button.dataset.tab).classList.add('active');
}
document.querySelectorAll('.tab-buttons button').forEach(function (button) {
  button.addEventListener('click', function () { showSection(button); });
});
document.querySelectorAll('.group-tabs button').forEach(function (button) {
  button.addEventListener('click', function () {
    document.querySelectorAll('.group-tabs button, .tab-buttons').forEach(function (node) {
      node.classList.remove('active');
    });
    button.classList.add('active');
    var row = document.querySelector('.tab-buttons[data-group="' + button.dataset.group + '"]');
    row.classList.add('active');
    var last = row.dataset.lastTab;
    showSection(last ? row.querySelector('button[data-tab="' + last + '"]') : row.querySelector('button'));
  });
});
"""


def render_report_page(sections: list[dict], subtitle: str) -> str:
    """The whole page: a row of group tabs, under it the chosen group's section tabs, one panel per section."""
    group_buttons, section_rows, panels, position = [], [], [], 0
    for group_index, (group_name, members) in enumerate(group_sections(sections)):
        group_active = ' class="active"' if group_index == 0 else ''
        group_buttons.append(f'<button data-group="{group_index}"{group_active}>{escape(group_name)}</button>')
        buttons = []
        for data in members:
            active = ' class="active"' if position == 0 else ''
            buttons.append(f'<button data-tab="{position}"{active}>{escape(data["title"].split(" (")[0])}</button>')
            panels.append(render_section_panel(data, position))
            position += 1
        row_class = 'tab-buttons active' if group_index == 0 else 'tab-buttons'
        section_rows.append(f'<div class="{row_class}" data-group="{group_index}">{"".join(buttons)}</div>')
    newline = '\n'
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>Experiment report</title>\n'
        f'<style>{PAGE_CSS}</style></head><body><main>\n'
        '<h1>Experiment report</h1>\n'
        f'<div class="stamp">{subtitle}</div>\n'
        f'<nav><div class="group-tabs">{"".join(group_buttons)}</div>{"".join(section_rows)}</nav>\n'
        f'{newline.join(panels)}\n'
        f'<script>{PAGE_SCRIPT}</script>\n</main></body></html>'
    )
