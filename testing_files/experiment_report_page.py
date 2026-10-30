"""The experiment report page: one tab per stored section, each with its explanation, a takeaway card (a headline and a
chart drawn from the section's own rows) and its table, colour-coded.

Colour code, the same in every tab: green = good, red = bad (a loss is deeper red the larger it is), amber = worth a
look. The cards report numbers, not verdicts. Values that are expected win rates or differences of them carry '%'.

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
  .seeding { background:repeating-linear-gradient(135deg, var(--other) 0 4px, transparent 4px 7px);
             outline:1px solid var(--other); outline-offset:-1px; }
  .builds { display:flex; flex-wrap:wrap; gap:4px 5px; }
  .build { display:inline-flex; align-items:baseline; gap:4px; border:1px solid var(--line); border-left-width:5px;
           border-radius:6px; padding:1px 7px; background:var(--bg); font-size:.82rem; white-space:nowrap; }
  .build b { color:var(--fg); }
  .build .soft { color:var(--muted); }
  .build .count { color:var(--muted); font-size:.75rem; margin-left:2px; }
  h2.part { margin-top:36px; }
  .details { color:var(--muted); font-size:.85rem; margin:0 0 12px; }
  .details p.explain, .details ul.explain { font-size:.85rem; color:var(--muted); margin:2px 0; }
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
  table.seed-grid { width:auto; margin:4px 0 10px; }
  table.seed-grid th { text-align:center; padding:4px 6px; border-bottom:none; font-size:.8rem; }
  table.seed-grid td { text-align:center; min-width:58px; padding:5px 6px; border:1px solid var(--line);
                       cursor:default; }
  table.seed-grid td.empty { border:none; }
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
    , *
    , x_title: str
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
    , uses_log_scale: bool = False
) -> str:
    """A chart over categorical x positions: 'line' (lines with points), 'dots', 'bars' (grouped) or 'stacked' (one bar
    per position, the series stacked bottom first). Reference lines are dashed (red for a floor). A legend is drawn
    below when there is more than one series. width is the drawing's own width, so a chart set beside another (in
    .multiples) keeps legible text by being drawn narrow rather than scaled down; show_points False draws a line
    chart's lines alone, for curves too dense for a marker at every point. uses_log_scale draws the y axis in powers of
    ten (line or dots charts of positive values), for data spanning several orders of magnitude; y_floor and
    y_ceiling then only widen it to the decades they fall in."""
    left, right, top, bottom = 58, (120 if width > 700 else 70), 14, 58
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
    if uses_log_scale:
        if kind not in ('line', 'dots') or low <= 0:
            raise ValueError(f'A log-scale chart needs a line or dots chart of positive values (kind {kind!r}, '
                             f'smallest value {low}).')
        low_exponent, high_exponent = int(np.floor(np.log10(low))), int(np.ceil(np.log10(high)))
        high_exponent = max(high_exponent, low_exponent + 1)
        ticks = [10.0 ** exponent for exponent in range(low_exponent, high_exponent + 1)]

        def y_of(value):
            return top + plot_height * (1 - (np.log10(value) - low_exponent) / (high_exponent - low_exponent))
    else:
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
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height + 38}" text-anchor="middle">'
                 f'{escape(x_title)}</text>')
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


def render_takeaway(text: str) -> str:
    """The card's headline: the numbers that answer the experiment's question, without a verdict."""
    return f'<div class="takeaway">{text}</div>'


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
# Break-even H-score per format: head to head, a 50% expected category win rate; Rotisserie, the league-win
# probability of one team in 12.
H_VERSUS_G_BREAK_EVEN = {'EC': 50.0, 'MC': 50.0, 'Roto': 100 / 12}


def card_h_versus_g(columns, rows):
    """Per format, the H seat's pooled H-score against its break-even, then by draft position: one chart for the
    head-to-head formats (expected category win rate) and one for Rotisserie (league-win probability)."""
    format_index, value_index = columns.index('Format'), columns.index('Mean H-score')
    pooled = {row[format_index]: read_leading_number(row[value_index]) for row in rows
              if str(row[columns.index('Season')]).startswith('ALL')}
    seasons = sorted({row[columns.index('Season')] for row in candidate_rows(rows)})
    names = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
    has_real = 'Mean real win rate' in columns
    pooled_real = {row[format_index]: read_leading_number(row[columns.index('Mean real win rate')]) for row in rows
                   if has_real and str(row[columns.index('Season')]).startswith('ALL')}

    def describe_format(format_key, value):
        real = pooled_real.get(format_key)
        real_text = f', <b>{real:.1f}%</b> in real weeks' if real is not None else ''
        return (f'{names[format_key]}: <b>{value:.1f}%</b> H-score{real_text} (break-even '
                f'{H_VERSUS_G_BREAK_EVEN[format_key]:.1f}%)')

    sentences = [describe_format(format_key, value) for format_key, value in pooled.items()]
    text = 'Against a G-score field, the H-scoring seat scores ' + '; '.join(sentences) + '.'
    seat_columns = [name for name in columns if re.fullmatch(r'S\d+', name)]

    def summarize_by_seat(by_seat, label):
        standard_errors = by_seat.std(axis=0, ddof=1) / np.sqrt(len(by_seat))
        return Series(label, list(by_seat.mean(axis=0)), list(2 * standard_errors))

    def chart_formats(format_keys, y_title, reference_label):
        series = []
        for format_key in format_keys:
            format_rows = [row for row in candidate_rows(rows) if row[format_index] == format_key]
            by_seat = np.array([[read_leading_number(row[columns.index(name)]) for name in seat_columns]
                                for row in format_rows])
            if format_key in pooled_real:
                series.append(summarize_by_seat(by_seat, f'{format_key} H-score'))
                real_by_seat = np.array([[float(value) for value in row[columns.index('Every real win rate')].split()]
                                         for row in format_rows])
                series.append(summarize_by_seat(real_by_seat, f'{format_key} real weeks'))
            else:
                series.append(summarize_by_seat(by_seat, format_key))
        return render_chart([name[1:] for name in seat_columns], series, y_title, x_title='Seat (draft position)',
                            references=(ReferenceLine(H_VERSUS_G_BREAK_EVEN[format_keys[0]], reference_label),),
                            value_suffix='%', value_places=1)

    head_to_head = [format_key for format_key in ('EC', 'MC') if format_key in pooled]
    charts = ''
    if head_to_head:
        charts += ('<div class="note">Head to head by draft position: Each Category\'s category win rate, Most '
                   'Categories\' matchup win rate, as the H-score expects it and in real weeks.</div>'
                   + chart_formats(head_to_head, 'Win rate', 'break-even 50%'))
    if 'Roto' in pooled:
        charts += ('<div class="note">Rotisserie: probability of winning the league by draft position.</div>'
                   + chart_formats(['Roto'], 'League-win probability', 'break-even 8.3%'))
    note = (f'<div class="note">Averaged over the {len(seasons)} seasons; bars are ±2 standard errors across '
            f'seasons.</div>')
    return render_takeaway(text) + note + charts


def rule_season_win_rate(value, row, columns):
    """Green above the format's break-even, red at or below it."""
    return GOOD if value > H_VERSUS_G_BREAK_EVEN[row[columns.index('Format')]] else BAD


def rule_seat_win_rate(value, row, columns):
    return style_bad(1.0) if value < H_VERSUS_G_BREAK_EVEN[row[columns.index('Format')]] else ''


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
    unaware = pooled.get('vs unaware H field', (float('nan'), float('nan')))
    g_field = pooled.get('vs G field', (float('nan'), float('nan')))
    text = (f'The opponent model adds <b>{unaware[0]:+.2f}% ± {unaware[1]:.2f}</b> against H-drafters without it, and '
            f'<b>{g_field[0]:+.2f}% ± {g_field[1]:.2f}</b> against G-score drafters, where its predictions are wrong. '
            f'± is one standard error; the bars are ±2.')
    chart = render_chart([season[2:] for season in seasons] + ['ALL'], series, 'On minus off', x_title='Season',
                         kind='dots', references=(ReferenceLine(0.0, 'no effect'),),
                         value_suffix='%', value_places=2)
    return render_takeaway(text) + chart


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
    return (render_takeaway(text) + note
            + legend + f'<div class="multiples">{"".join(panels)}</div>')


# Rotisserie punting
def card_rotisserie(columns, rows):
    most = max(read_leading_number(row[2]) for row in rows)
    weakest = min(rows, key=lambda row: read_leading_number(row[5]))
    text = (f'At most <b>{most:.0f}</b> hard punt{"s" if most != 1 else ""} among the top-12 Rotisserie builds in any '
            f'season. The weakest category anywhere: {escape(weakest[4])} at {escape(weakest[5])} '
            f'({escape(weakest[6])}, {escape(weakest[0])} {escape(weakest[1])}).')
    return render_takeaway(text)


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
    , *
    , x_title: str
    , value_suffix: str = ''
    , height: int = 240
) -> str:
    """Every value as its own dot: per x position, one column of dots per series (side by side), each dot's tooltip its
    value. series: [(name, values per x position)]."""
    width, left, right, top, bottom = 1100, 58, 120, 14, 50
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
    parts.append(f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height + 40}" text-anchor="middle">'
                 f'{escape(x_title)}</text>')
    parts.append('</svg>')
    legend = '<div class="legend">' + ''.join(
        f'<span><span class="swatch series-{index}"></span>{escape(name)}</span>'
        for index, (name, _) in enumerate(series)) + '</div>'
    return ''.join(parts) + legend


def collect_player_values(columns, rows, column, group_of):
    """{group: every per-player value in `column` (space-separated in each row)} over the rows, grouped by
    group_of(row)."""
    collected = {}
    for row in rows:
        collected.setdefault(group_of(row), []).extend(float(value) for value in row[columns.index(column)].split())
    return collected


def render_stability_charts(columns, rows, group_of, what, include_rank_moves):
    """The stability charts: how many places each player moves (when include_rank_moves), and how far its category
    win rates move on average (percentage points), each a histogram grouped by group_of(row). `what` names the
    comparison. Results saved before the per-player columns existed say so instead of drawing."""
    if 'Every rank move' not in columns:
        return ('<div class="takeaway">These results predate the per-player rank and win-rate columns; rerun the '
                'experiment to draw the charts.</div>')
    moves = collect_player_values(columns, rows, 'Every rank move', group_of)
    changes = collect_player_values(columns, rows, 'Every category change', group_of)
    largest_move = int(max(max(values) for values in moves.values()))
    move_chart = render_chart([str(move) for move in range(largest_move + 1)],
                              [Series(group, [sum(int(value) == move for value in values)
                                              for move in range(largest_move + 1)])
                               for group, values in moves.items()],
                              'Players', x_title=f'Places moved ({what})', kind='bars', value_places=0)
    edges = (0.0, 1.0, 2.0, 5.0, 10.0)
    labels = [f'{low:g}–{high:g}' for low, high in zip(edges, edges[1:])] + [f'{edges[-1]:g}+']
    change_chart = render_chart(labels,
                                [Series(group, [sum(low <= value < high for value in values)
                                                for low, high in zip(edges, edges[1:])]
                                        + [sum(value >= edges[-1] for value in values)])
                                 for group, values in changes.items()],
                                'Players', x_title=f'Average category win-rate change, points ({what})',
                                kind='bars', value_places=0)
    def describe_group(group):
        places = (f'{sum(value == 0 for value in moves[group])} of {len(moves[group])} players keep their place, and '
                  if include_rank_moves else '')
        return (f'{escape(group)}: {places}the average category win rate moves <b>{np.median(changes[group]):.1f}</b> '
                f'points for the median player (largest {max(changes[group]):.1f}).')

    sentences = [describe_group(group) for group in changes]
    note_moves = ("Ranking: how many places each player's position moves. 0 is the same place; 1 is two neighbours "
                  'swapping.')
    note_changes = ("Builds: the mean over the categories of how far each player's expected category win rate moves, "
                    'in percentage points. A refinement moves it a point or two; a different punt moves it by ten or '
                    'more.')
    ranking = f'<div class="note">{note_moves}</div>' + move_chart if include_rank_moves else ''
    return (f'<div class="takeaway">{" ".join(sentences)}</div>' + ranking
            + f'<div class="note">{note_changes}</div>' + change_chart)


def card_draft_pick_stability(columns, rows):
    """Per format, how the board moves when an opponent drafts exactly the player the model predicted for them."""
    return render_stability_charts(columns, rows, lambda row: row[columns.index('Format')],
                                   'before vs after the predicted pick', include_rank_moves=True)


def card_auction_pick_stability(columns, rows):
    """Per format, how the builds move when an opponent buys exactly the player the model predicted for them. No
    ranking chart: in an auction a player's place on the board does not matter."""
    return render_stability_charts(columns, rows, lambda row: row[columns.index('Format')],
                                   'before vs after the predicted purchase', include_rank_moves=False)


def derive_table_without_lists(columns, rows):
    """The table without the per-player lists (they are the charts)."""
    kept = [index for index, name in enumerate(columns) if not name.startswith('Every ')]
    return [columns[index] for index in kept], [[row[index] for index in kept] for row in rows]


# Self-play convergence
def list_self_play_runs(columns, rows):
    """The runs in the table, the app's first."""
    return list(dict.fromkeys(row[columns.index('Run')] for row in rows))


def list_punt_columns(columns):
    """The per-category punt-count columns: every column after 'Max drift'."""
    return columns[columns.index('Max drift') + 1:]


def track_convergence(columns, rows, run, measure):
    """For one run: per (format, season), one value per block, in order -- 'Mean change' or 'Mean drift' (None
    where a block has none), or 'Punt changes': how many punt counts changed since the previous block's end, summed
    over the categories (None for the first block). Returns ({(format, season): [value]}, [block labels])."""
    format_index, run_index = columns.index('Format'), columns.index('Run')
    punt_indices = [columns.index(name) for name in list_punt_columns(columns)]
    values, labels, previous_counts = {}, [], {}
    for row in rows:
        if row[run_index] != run:
            continue
        key = (row[format_index], row[0])
        if measure == 'Punt changes':
            counts = np.array([read_leading_number(row[index]) for index in punt_indices])
            value = None if key not in previous_counts else float(np.abs(counts - previous_counts[key]).sum())
            previous_counts[key] = counts
        else:
            value = read_leading_number(row[columns.index(measure)])
        values.setdefault(key, []).append(value)
        labels.append(row[columns.index('Iterations')])
    return values, list(dict.fromkeys(labels))


def average_blocks_by_format(values_by_key):
    """{(format, season): [value or None]} -> {format: [mean over seasons, or None]}."""
    averaged = {}
    for format_key in dict.fromkeys(key[0] for key in values_by_key):
        lists = [values for key, values in values_by_key.items() if key[0] == format_key]
        longest = max(len(values) for values in lists)
        averaged[format_key] = []
        for position in range(longest):
            present = [values[position] for values in lists if position < len(values) and values[position] is not None]
            averaged[format_key].append(float(np.mean(present)) if present else None)
    return averaged


def card_self_play(columns, rows):
    """One chart per format, from the longest run (its first blocks are the app's run): the per-iteration movement,
    the block-to-block drift and the punt counts changed, each averaged over the seasons. Numbers only, no verdict."""
    runs = list_self_play_runs(columns, rows)
    longest_run = runs[-1]
    app_blocks = len(track_convergence(columns, rows, runs[0], 'Mean change')[1])
    changes, labels = track_convergence(columns, rows, longest_run, 'Mean change')
    drifts, _ = track_convergence(columns, rows, longest_run, 'Mean drift')
    punt_changes, _ = track_convergence(columns, rows, longest_run, 'Punt changes')
    changes, drifts = average_blocks_by_format(changes), average_blocks_by_format(drifts)
    punt_changes = average_blocks_by_format(punt_changes)
    sentences, charts = [], []
    for format_key in changes:
        sentence = (f"{escape(format_key)}: a top player's weights move <b>{changes[format_key][0]:.1f}</b> per "
                    f"iteration in the first 10 iterations and <b>{changes[format_key][app_blocks - 1]:.1f}</b> in the "
                    f"app's last 10; the block average drifts <b>{drifts[format_key][1]:.1f}</b> between the first two "
                    f"blocks and <b>{drifts[format_key][app_blocks - 1]:.1f}</b> into the app's last")
        if len(runs) > 1:
            sentence += f', <b>{drifts[format_key][-1]:.1f}</b> at the end of the longer run'
        sentences.append(sentence + '.')
        measures = (('per-iteration movement', changes[format_key]), ('drift', drifts[format_key]),
                    ('punt counts changed', punt_changes[format_key]))
        # In colour up to where the app stops; the longer run's continuation in grey, joined at the app's last block.
        within_app = [Series(name, values[:app_blocks] + [None] * (len(values) - app_blocks))
                      for name, values in measures]
        beyond_app = [Series(f'{name}, beyond the app', [None] * (app_blocks - 1) + values[app_blocks - 1:])
                      for name, values in measures] if len(runs) > 1 else []
        charts.append(f'<div class="note"><b>{escape(format_key)}</b></div>' + render_chart(
            labels, within_app, 'Change', x_title='Iterations', kind='line', value_places=1, y_floor=0.0, height=220,
            background_series=beyond_app))
    note = ("Per-iteration movement: the average over each block of 10 iterations of a top-12 player's change in "
            'category weights from the previous iteration (Euclidean norm, display units); switching back and forth '
            "counts in full. Drift: each player's weights averaged over a block against the previous block's average; "
            'back-and-forth cancels and only movement of the average shows. Punt counts changed: how many of the '
            "table's punt counts (top 12 players under 40% in a category) changed since the previous block's end. All "
            'averaged over the seasons.' + (f" The charts run to {longest_run.split('(')[-1].rstrip(')')}; the app "
                                             f'stops after {app_blocks} blocks, and the rest is grey.'
                                             if len(runs) > 1 else ''))
    return f'<div class="takeaway">{" ".join(sentences)}</div><div class="note">{note}</div>' + ''.join(charts)


def filter_self_play(columns, rows):
    runs = list_self_play_runs(columns, rows)
    return ([row for row in rows if row[columns.index('Run')] == runs[0]],
            "One row per block of 10 iterations. The category columns count the top 12 players whose current start "
            "punts that category at the block's end. The table shows the app's run.")


def styles_self_play(columns, rows):
    styles = []
    first_punt_column = columns.index('Max drift') + 1
    for row in rows:
        row_styles = [''] * len(row)
        for index in range(first_punt_column, len(row)):
            value = read_leading_number(row[index])
            if value:
                row_styles[index] = style_heat(value / 12)
        styles.append(row_styles)
    return styles


# Self-play best responses
def card_best_responses(columns, rows):
    """The numbers only: how far the top served builds fall below the best a long descent reaches, no verdict."""
    shortfalls = [read_leading_number(row[columns.index('Shortfall')]) for row in rows]
    worst = max(rows, key=lambda row: read_leading_number(row[columns.index('Shortfall')]))
    text = (f'Of {len(rows)} top served builds, the mean shortfall from the best a long descent reaches against the '
            f'same field is <b>{np.mean(shortfalls):.3f}</b> H-score points, and <b>{sum(s > 0.1 for s in shortfalls)}</b> '
            f'fall more than 0.1 below it. The largest: {escape(worst[columns.index("Player")])} ({escape(worst[0])}, '
            f'{escape(worst[1])}) at {escape(worst[columns.index("Shortfall")])}.')
    edges = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    labels = [f'{low:g}–{high:g}' for low, high in zip(edges, edges[1:])] + [f'{edges[-1]:g}+']
    series = []
    for format_key in dict.fromkeys(row[columns.index('Format')] for row in rows):
        values = np.array([read_leading_number(row[columns.index('Shortfall')]) for row in rows
                           if row[columns.index('Format')] == format_key])
        counts = [int(((values >= low) & (values < high)).sum()) for low, high in zip(edges, edges[1:])]
        series.append(Series(format_key, counts + [int((values >= edges[-1]).sum())]))
    chart = render_chart(labels, series, 'Top served builds', x_title='Shortfall from the best response (H-score points)',
                         kind='bars', value_places=0)
    note = ('How many of the top served builds fall short of the best a long descent reaches by each amount '
            '(H-score points), by format.')
    ranked = rank_by_served_and_best(columns, rows)
    moves = {}
    for (format_key, _), places in ranked.items():
        moves.setdefault(format_key, []).extend(abs(served - best) for served, best in places.values())
    largest_move = max(max(values) for values in moves.values())
    move_labels = [str(move) for move in range(largest_move + 1)]
    move_series = [Series(format_key, [sum(value == move for value in values) for move in range(largest_move + 1)])
                   for format_key, values in moves.items()]
    move_chart = render_chart(move_labels, move_series, 'Top-12 players', x_title='Places moved if served optimally',
                              kind='bars', value_places=0)
    moved = '; '.join(f'{escape(format_key)} {sum(value > 0 for value in values)} of {len(values)}'
                      for format_key, values in moves.items())
    move_note = ("If every top-12 player were served its best response instead, how many places would it move within "
                 f"its season's top 12? Players who would move: {moved}. Players outside the top 12 have no best "
                 'response measured, so a move into the top 12 cannot show.')
    return (f'<div class="takeaway">{text}</div><div class="note">{note}</div>' + chart
            + f'<div class="note">{move_note}</div>' + move_chart)


def rank_by_served_and_best(columns, rows):
    """Per (format, season): each player's place in the top 12 by served H-score and by best-response H-score (1 =
    first). Returns {(format, season): {player: (served place, best place)}}."""
    places = {}
    for key in dict.fromkeys((row[columns.index('Format')], row[0]) for row in rows):
        group = [row for row in rows if (row[columns.index('Format')], row[0]) == key]
        by_served = sorted(group, key=lambda row: -read_leading_number(row[columns.index('Served H')]))
        by_best = sorted(group, key=lambda row: -read_leading_number(row[columns.index('Best H')]))
        best_place = {row[columns.index('Player')]: place for place, row in enumerate(by_best, start=1)}
        places[key] = {row[columns.index('Player')]: (place, best_place[row[columns.index('Player')]])
                       for place, row in enumerate(by_served, start=1)}
    return places


def derive_optimality_table(columns, rows):
    """The rows with each player's place in its season's top 12, served and if optimal, after its name."""
    places = rank_by_served_and_best(columns, rows)
    player_index = columns.index('Player')
    derived = []
    for row in rows:
        served, best = places[(row[columns.index('Format')], row[0])][row[player_index]]
        derived.append(row[:player_index + 1] + [str(served), str(best)] + row[player_index + 1:])
    return columns[:player_index + 1] + ['Served rank', 'Rank if optimal'] + columns[player_index + 1:], derived


# Serve stability
def is_app_confidence(row, columns):
    """Rows at the app's own confidence: head to head below C = 1, Rotisserie (always at 1)."""
    return row[columns.index('Format')] == 'Roto' or read_leading_number(row[columns.index('C')]) < 1


def card_serve_stability(columns, rows):
    """Serve stability, numbers only: per format at the app's own confidence, how the field's players move between
    self-play's builds and the served ones."""
    app_rows = [row for row in rows if is_app_confidence(row, columns)]
    return render_stability_charts(columns, app_rows, lambda row: row[columns.index('Format')],
                                   'self-play vs served', include_rank_moves=True)


# How single-peaked are H-scores
SEED_GRID_CATEGORIES = ('FG%', 'FT%', '3s', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO')
# The share of the app's menu's starts reaching the optimum above which the headline reads them as all on one hill.
SEED_ONE_HILL_SHARE = 99.0


SEED_MISS_COLUMNS = ['Season', 'Rd', 'Player', 'Seeds that miss', 'Shortfall(s)', "App's seed reaches"]
# A shortfall this large (in percentage points) is a real second peak; below it, starts stop on a nearly flat ridge by
# the optimum. The measured shortfalls sit far either side: 0.16 and up, or under 0.006.
SECOND_PEAK_SHORTFALL = 0.1


def derive_seed_miss_table(columns, rows):
    """The table under the heatmaps: one row per candidate with a start that misses the optimum, its missing seeds in
    one cell (the app's seeds are the 'at 50%' ones and 'balanced'), largest shortfall first, under a count."""
    by_candidate: dict = {}
    for row in rows:
        if row[0] != 'Miss':
            continue
        key = tuple(row[columns.index(name)] for name in ('Season', 'Rd', 'Player'))
        candidate = by_candidate.setdefault(key, {'seeds': [], 'shortfalls': [],
                                                  'app_reaches': row[columns.index("App's seed reaches")]})
        candidate['seeds'].append(row[columns.index('Seed')])
        candidate['shortfalls'].append(float(row[columns.index('Shortfall (pp)')]))
    table = []
    for (season, n_rounds, player), candidate in sorted(by_candidate.items(), key=lambda item: -max(item[1]['shortfalls'])):
        low, high = min(candidate['shortfalls']), max(candidate['shortfalls'])
        shortfall = f'{high:.4f}%' if f'{low:.4f}' == f'{high:.4f}' else f'{low:.4f}% to {high:.4f}%'
        table.append([season, n_rounds, player, ', '.join(candidate['seeds']), shortfall, candidate['app_reaches']])
    starts = read_leading_number(find_summary(rows, 'ALL (Full menu')[columns.index('Starts')])
    n_misses = sum(len(candidate['seeds']) for candidate in by_candidate.values())
    summary = [f'ALL: {n_misses} of {starts:,.0f} starts miss, in {len(by_candidate)} candidates'] + ['—'] * 5
    return SEED_MISS_COLUMNS, [summary] + table


def describe_seed_misses(columns, rows):
    """The note above the miss table (derive_seed_miss_table's rows): how many candidates have a second peak, how many of
    those peaks are significantly lower, and whether the app's chosen seed ever lands on one."""
    candidates = [row for row in rows if not is_summary_row(row)]

    def largest_shortfall(row):
        return float(re.findall(r'[\d.]+', str(row[columns.index('Shortfall(s)')]))[-1])
    significant = [row for row in candidates if largest_shortfall(row) >= SECOND_PEAK_SHORTFALL]
    chosen_misses = [row for row in candidates if row[columns.index("App's seed reaches")] == 'no']
    text = (f'Of the {len(candidates)} candidates where some starts end on another peak, the other peak is '
            f'significantly lower (by {SECOND_PEAK_SHORTFALL}% or more) in only {len(significant)}.')
    significant_chosen = [row for row in significant if row in chosen_misses]
    if not significant_chosen:
        text += (f' In all {len(significant)}, the seed the app\'s seeding chooses reaches the true optimum: the app '
                 f'never lands on a significantly lower peak.')
    else:
        text += (f' In {len(significant_chosen)} of them, the seed the app\'s seeding chooses lands on the lower '
                 f'peak.')
    insignificant_chosen = [row for row in chosen_misses if row not in significant]
    if insignificant_chosen:
        names = ', '.join(f'{row[columns.index("Player")]} ({row[columns.index("Season")]}, round '
                          f'{row[columns.index("Rd")]}), {largest_shortfall(row):.4f}% below'
                          for row in insignificant_chosen)
        occasion = 'time' if len(insignificant_chosen) == 1 else 'times'
        text += (f' The only {occasion} the app\'s chosen seed ends on another peak, across all candidates, that peak '
                 f'is barely lower: {names}.')
    text += (" The app's seeds are the 'at 50%' ones and 'balanced'; the rest are the full menu's punts at 5%, 55 seeds "
             "in all. App's seed reaches: whether the seed the app's seeding would choose reaches the optimum.")
    return rows, text


def read_seed_cells(columns, rows, menu):
    """A menu's seed cells by label: their reach, choice and shortfall cells as stored."""
    names = ('Reaches optimum', 'Chosen by seeding', 'Chosen by multi-start', 'Worst miss (pp)', 'Starts')
    return {row[1]: dict(zip(names, (row[columns.index(name)] for name in names)))
            for row in candidate_rows(rows) if row[0] == menu}


def render_seed_cell(label, cell, shown_text):
    """One heatmap cell: its reach rate, red the more of its starts miss (full red at 10% missing), and the choice
    rates on hover."""
    if cell is None:
        return '<td class="empty"></td>'
    reach = read_leading_number(cell['Reaches optimum'])
    missing = min((100 - reach) / 10, 1.0)
    background = f' style="background:rgba(var(--bad-rgb),{0.55 * missing:.2f})"' if missing > 0 else ''
    hover = (f'{label}: {cell["Reaches optimum"]} of {cell["Starts"]} starts reach the optimum; chosen by seeding '
             f'{cell["Chosen by seeding"]}, by multi-starting {cell["Chosen by multi-start"]}; worst miss '
             f'{format_percent(cell["Worst miss (pp)"])}')
    return f'<td{background} title="{escape(hover)}">{escape(shown_text)}</td>'


def render_seed_strip(cells, labels, heads):
    """One row of cells under their headings."""
    head = ''.join(f'<th>{escape(text)}</th>' for text in heads)
    body = ''.join(render_seed_cell(label, cells.get(label), str(cells[label]['Reaches optimum']) if label in cells
                                    else '') for label in labels)
    return f'<table class="seed-grid"><tr>{head}</tr><tr>{body}</tr></table>'


def render_punt_grid(cells):
    """The full menu's single and double punts at 5%: row i, column j is the double punt of i and j, the diagonal the
    single punt; the lower triangle repeats the upper, so it is left blank."""
    head = '<th></th>' + ''.join(f'<th>{name}</th>' for name in SEED_GRID_CATEGORIES)
    body = []
    for first_index, first in enumerate(SEED_GRID_CATEGORIES):
        row_cells = []
        for second_index, second in enumerate(SEED_GRID_CATEGORIES):
            if second_index < first_index:
                row_cells.append('<td class="empty"></td>')
                continue
            label = first if first == second else f'{first}/{second}'
            row_cells.append(render_seed_cell(f'punt {label} at 5%', cells.get(label),
                                              str(cells[label]['Reaches optimum']) if label in cells else ''))
        body.append(f'<tr><th>{first}</th>{"".join(row_cells)}</tr>')
    return f'<table class="seed-grid"><tr>{head}</tr>{"".join(body)}</table>'


def card_seed_starts(columns, rows):
    app_cells = read_seed_cells(columns, rows, 'App menu')
    full_cells = read_seed_cells(columns, rows, 'Full menu')
    app_all = read_leading_number(find_summary(rows, 'ALL (App menu')[columns.index('Reaches optimum')])
    full_all = read_leading_number(find_summary(rows, 'ALL (Full menu')[columns.index('Reaches optimum')])
    reached = (f'<b>{app_all:.2f}%</b> of starts from the app\'s menu and <b>{full_all:.2f}%</b> from the full menu '
               f'climb to the certified optimum given the long budget.')
    # Nearly every start on one hill: the seed menu cannot be choosing the hill, so seeding's gain is a closer start.
    if app_all >= SEED_ONE_HILL_SHARE:
        conclusion = (' Choosing a good starting seed is rarely necessary to eventually find the true optimum. Its '
                      'consistent benefit comes instead from starting closer to the peak, which gets the descent closer '
                      'in fewer iterations.')
    else:
        conclusion = (' Some of the app\'s seeds climb a different hill, so for those candidates seeding also decides '
                      'which optimum the app reaches.')
    text = (reached + conclusion + ' Each cell is its seed\'s share; red marks starts that end on another hill. Hover '
            'for how often the seed is the one chosen.')
    app_labels = [f'{name} at 50%' for name in SEED_GRID_CATEGORIES] + ['balanced']
    app_heads = list(SEED_GRID_CATEGORIES) + ['balanced']
    return (render_takeaway(text)
            + '<h3>The app\'s menu</h3><div class="note">Gentle punts (that weight at 50% of neutral) and the balanced '
              'build.</div>' + render_seed_strip(app_cells, app_labels, app_heads)
            + '<h3>The full menu</h3><div class="note">The app\'s seeds above, plus single (diagonal) and double punts '
              'at 5% of neutral. A seed reaches the optimum as often in either menu (its descent is the same); only '
              'how often it is chosen differs, which the hover and the table give for every seed.</div>'
            + render_punt_grid(full_cells))


# Conditioning around the optimum
# The window inside which every direction spirals in at sqrt(beta1) (optimum_loss_decomposition.UNDERDAMPED_WINDOW,
# for Adam's beta1 = 0.9); repeated here so the page can be rebuilt without importing the harness.
UNDERDAMPED_LOW = (1 - np.sqrt(0.9)) ** 2 / (1 - 0.9)
UNDERDAMPED_HIGH = (1 + np.sqrt(0.9)) ** 2 / (1 - 0.9)


def card_conditioning(columns, rows):
    """One chart over the app's iterations: the minimum, average and maximum step x curvature over every direction of
    every candidate's own descent, on a log axis with both edges of the window."""
    iterations = [str(row[columns.index('Iteration')]) for row in rows]

    def column(name):
        return [float(row[columns.index(name)]) for row in rows]

    outside = [int(row[columns.index('Candidates outside the window')]) for row in rows]
    candidates = int(rows[0][columns.index('Candidates')])
    inside_from = next((index + 1 for index in range(len(outside)) if not any(outside[index:])), None)
    window = f'{UNDERDAMPED_LOW:.3g}-{UNDERDAMPED_HIGH:.3g}'
    if inside_from is not None:
        text = (f'For all {candidates} candidates, every direction\'s step x curvature is inside the {window} window '
                f'from iteration <b>{inside_from}</b> of {len(iterations)} on: from there every direction closes in at '
                f'sqrt(0.9) = 0.949 per iteration.')
    else:
        text = (f'{outside[-1]} of {candidates} candidates still have a direction outside the {window} window after '
                f'{len(iterations)} iterations.')
    if outside[0]:
        text += (f' In the first iterations Adam\'s step sizes come from its first few gradients alone, and some '
                 f'directions sit far above 38 (up to {column("Max")[0]:,.0f} at iteration 1).')
    # A log axis: the first iterations run to the hundreds of thousands, and the window's edges are 0.026 and 38.
    chart = render_chart(iterations, [Series(name, column(name)) for name in ('Max', 'Average', 'Min')],
                         'Step x curvature (log)', x_title='Iteration', kind='line',
                         references=(ReferenceLine(UNDERDAMPED_HIGH, 'stability edge', True),
                                     ReferenceLine(UNDERDAMPED_LOW, 'slow edge', True)),
                         value_places=3, show_points=False, uses_log_scale=True)
    return (render_takeaway(text) + chart
            + '<div class="note">Over every direction of every candidate\'s own descent: every direction lies between '
              'the Max and Min lines.</div>')


# Loop gain
def card_loop_gain(columns, rows):
    gains = [value for value in column_values(columns, candidate_rows(rows), 'Largest loop gain') if value is not None]
    edges = np.arange(0.0, 1.2, 0.1)
    counts, _ = np.histogram(gains, bins=edges)
    text = (f'The largest loop gain at any of the {len(gains)} optima is <b>{max(gains):.3f}</b>: every one is below 1, '
            f'so a nudge to the weights shrinks on its way round and each optimum is a true peak.')
    chart = render_chart([f'{low:.1f}–{low + 0.1:.1f}' for low in edges[:-1]], [Series('optima', list(counts))],
                         'Optima', x_title='Largest loop gain', kind='bars', value_places=0)
    return (render_takeaway(text)
            + '<div class="note">How many optima have each largest loop gain (1 is the limit).</div>' + chart)


def rule_loop_gain(value, row, columns):
    return BAD if value >= 1 else (WARN if value >= 0.8 else GOOD)


def sort_by_loop_gain(columns, rows):
    index = columns.index('Largest loop gain')
    return sorted(rows, key=lambda row: -(read_leading_number(row[index]) or 0))


# Planner vs H-scoring
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


def summarize_planner_pairs(columns, rows):
    """Per planner variant, from the individual drafts: each pair's planner-minus-H-scoring (the mean of its two
    drafts), and the drafts. Returns {variant: {'pairs': array of pair differences, 'drafts': array of draft
    differences}}."""
    difference_index = columns.index('Planner minus H (pp)')
    summary = {}
    for variant in dict.fromkeys(row[0] for row in rows):
        pairs = {}
        for row in rows:
            if row[0] == variant:
                pairs.setdefault(row[1], []).append(read_leading_number(row[difference_index]))
        summary[variant] = {'pairs': np.array([np.mean(drafts) for drafts in pairs.values()]),
                            'drafts': np.array([value for drafts in pairs.values() for value in drafts])}
    return summary


def card_planner_drafts(columns, rows):
    """The whole experiment on one card: per variant the mean pair difference with its standard error and how many
    pairs and drafts the planner won; then each seat against itself, by draft position. Numbers only."""
    summary = summarize_planner_pairs(columns, rows)
    differences = compute_seat_differences(columns, rows)
    seat_count = next(iter(differences.values())).shape[0]
    sentences = []
    for variant, values in summary.items():
        pairs, drafts = values['pairs'], values['drafts']
        seat_means = differences[variant].mean(axis=1)
        sentences.append(
            f'{escape(variant)}: <b>{pairs.mean():+.2f}%</b> per pair (standard error {pairs.std(ddof=1) / np.sqrt(len(pairs)):.2f}, '
            f'pairs from {pairs.min():+.2f} to {pairs.max():+.2f}); the planner wins <b>{int((pairs > 0).sum())} of '
            f'{len(pairs)}</b> pairs and {int((drafts > 0).sum())} of {len(drafts)} single drafts; by seat, from '
            f'{seat_means.min():+.2f}% (seat {int(np.argmin(seat_means)) + 1}) to {seat_means.max():+.2f}% (seat '
            f'{int(np.argmax(seat_means)) + 1}).')
    series = [Series(variant, list(values.mean(axis=1)),
                     list(2 * values.std(axis=1, ddof=1) / np.sqrt(values.shape[1])))
              for variant, values in differences.items()]
    chart = render_chart([str(seat + 1) for seat in range(seat_count)], series, 'Planner minus H',
                         x_title='Seat (draft position)', kind='dots',
                         references=(ReferenceLine(0.0, 'even'),), value_suffix='%', value_places=2)
    note = ("A pair is two drafts in which the planner and H-scoring seats swap; its number is the planner seats' mean "
            "complete-team H-score minus the H-scoring seats', averaged over the two drafts. The chart: each dot is one "
            "draft position (seat 1 picks first), averaged over the pairs: that seat drafting with the planner minus "
            'the same seat drafting with H-scoring; bars are ±2 standard errors. One colour per planner variant.')
    return f'<div class="takeaway">{" ".join(sentences)}</div><div class="note">{note}</div>' + chart


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
    """A diverging tint per difference: purple where H-scoring won, green where the planner won, deeper for more
    (full depth at 3 points). Its own pair of hues, so it cannot be read as the chart's planner-variant colours."""
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        for index in range(columns.index('Planner minus H (pp)'), len(row)):
            value = read_leading_number(row[index])
            depth = 0.08 + 0.42 * min(abs(value) / 3.0, 1.0)
            color = '117,112,179' if value < 0 else '27,158,119'
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
        (f'{worst_gap:+.2f}', f'own-view H-score points per team at s = {worst} (±{2 * worst_error:.2f})',
         judge(worst_gap, worst_error)),
    ])

    groups = seasons + ['All']

    def read_season(softness, season, name):
        row = summaries[softness] if season == 'All' else by_season[(softness, season)]
        return read_leading_number(row[columns.index(name)])

    gap_chart = render_chart(
        [season[2:] if season != 'All' else season for season in groups],
        [Series(f's = {softness}', [read_season(softness, season, 'Own view gap') for season in groups],
                [2 * read_season(softness, season, 'Own view SE') for season in groups])
         for softness in softnesses],
        'Own-view H-score points, smooth - simple', x_title='Season', kind='bars', references=(ReferenceLine(0.0, 'no difference'),),
        value_places=2, height=250)

    def spending_gap(softness, round_number):
        return (read(softness, f'Spent after round {round_number} (smooth)')
                - read(softness, f'Spent after round {round_number} (simple scaling)'))

    rounds = list(range(1, AUCTION_ROUNDS + 1))
    spending_chart = render_chart(
        [str(round_number) for round_number in rounds],
        [Series(f's = {softness}', [spending_gap(softness, round_number) for round_number in rounds])
         for softness in softnesses],
        '$ spent so far, smooth - simple', x_title='Auction round', references=(ReferenceLine(0.0, 'even'),), value_places=1, height=280,
        width=540)
    unspent_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(label, [read(softness, f'Unspent ({label})') for softness in softnesses]) for label in RULE_LABELS],
        '$ left per team', x_title='Cash-share softness', kind='bars', value_places=2, height=280, width=540)
    return (tiles
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
    misfit_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(rule, [read(softness, rule, 'Misfits per team') for softness in softnesses]) for rule in RULE_LABELS],
        'Misfits per team', x_title='Cash-share softness', kind='bars', value_places=2, height=260, width=540)
    punted_chart = render_chart(
        [f's = {softness}' for softness in softnesses],
        [Series(rule, [read(softness, rule, 'Value in punted') for softness in softnesses]) for rule in RULE_LABELS],
        'Value in punted categories', x_title='Cash-share softness', kind='bars', value_suffix='%', value_places=0, height=260, width=540)
    return (tiles
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
        '% change in dollars', x_title='Rank of the drafted player (by value)', references=(ReferenceLine(0.0, 'neither'),), value_suffix='%', value_places=0,
        height=280, show_points=False)
    return (render_takeaway(text) + tiles
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
LOSS_COLUMNS = ['Multi-starting', 'Full seed menu', 'Iterations', 'Remaining gap', 'Total loss']


SEEDING_COLUMN = 'Seeding (in the app)'


def card_loss(columns, rows):
    """The headline, and one bar from a neutral start to the optimum: what the app's seeding already recovers (muted,
    first), then the four losses the app still has, each a share of the whole."""
    summary = find_summary(rows, ('ALL', 'Mean of'))
    cost_row = find_summary(rows, 'Cost to fix')
    candidates = candidate_rows(rows)
    total = read_leading_number(summary[columns.index('Total loss')])
    loss_sums = {cause: sum(read_leading_number(row[columns.index(cause)]) for row in candidates)
                 for cause in LOSS_COLUMNS[:-1]}
    seeding_sum = max(sum(read_leading_number(row[columns.index(SEEDING_COLUMN)]) for row in candidates), 0.0)
    whole = seeding_sum + sum(loss_sums.values())
    seeding_share = round(100 * seeding_sum / whole) if whole else 0
    segments = [f'<div class="bar-segment seeding" style="flex-basis:{seeding_share:g}%" '
                f'title="Seeding: {seeding_share:g}% of the gap from a neutral start, already recovered by the app">'
                f'</div>']
    legend = ['<span><span class="swatch seeding"></span>Seeding &middot; already in the app</span>']
    for position, (cause, cause_sum) in enumerate(loss_sums.items()):
        share = round(100 * cause_sum / whole) if whole else 0
        segments.append(f'<div class="bar-segment series-{position}" style="flex-basis:{share:g}%" '
                        f'title="{escape(cause)}: {share:g}% of the gap from a neutral start"></div>')
        cost = f' &middot; fix: {escape(cost_row[columns.index(cause)])} the work' if cost_row else ''
        legend.append(f'<span><span class="swatch series-{position}"></span>{escape(cause)}{cost}</span>')
    average_app = np.mean([read_leading_number(row[columns.index('App')]) for row in candidates])
    average_seeding = read_leading_number(summary[columns.index(SEEDING_COLUMN)])
    headline = (f'<div class="takeaway">Average total loss: <b>{total:.4f}%</b>, off an average H-score of '
                f'{average_app:.2f}% &middot; {escape(summary[0])}. Seeding, which the app already does, gains '
                f'{average_seeding:.4f}% over a neutral start.</div>')
    return headline + f'<div class="bar-track">{"".join(segments)}</div><div class="legend">{"".join(legend)}</div>'


APP_LOSS_COLUMNS = ['Multi-starting', 'Full seed menu', 'Iterations']


def card_app_loss(columns, rows):
    """Per format, one bar: the format's total loss split among the three fixes that recover it,
    each a share of the whole (the exact-solution page's bar; App loss has no neutral start, so no seeding segment).
    Every bar is its own format's gap from a neutral start to the best found (the formats' H-scores are on different
    scales: Rotisserie's is a league-win probability): first what the app's seeding already recovers, then the three
    losses; its label gives the total loss. Rotisserie has no seeding, so its bar is its loss alone."""
    format_index = columns.index('Format')
    names = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
    bars = []
    for format_key in [key for key in names if any(row[format_index] == key for row in rows)]:
        format_rows = [row for row in rows if row[format_index] == format_key]
        summary = find_summary(format_rows, 'Mean of', column_index=1)
        cost_row = find_summary(format_rows, 'Cost to fix', column_index=1)
        candidates = candidate_rows(format_rows)
        total = read_leading_number(summary[columns.index('Total loss')])
        average_app = read_leading_number(summary[columns.index('App')])
        loss_sums = {cause: sum(read_leading_number(row[columns.index(cause)]) for row in candidates)
                     for cause in APP_LOSS_COLUMNS}
        # Rotisserie has no seeding (its one low-variance seed): no seeding segment, and its bar is its loss alone.
        average_seeding = read_leading_number(summary[columns.index(SEEDING_COLUMN)])
        if average_seeding is None:
            whole = sum(loss_sums.values())
            segments, legend = [], []
            span = 'the bar is the whole loss; Rotisserie has no seeding'
        else:
            seeding_sum = max(sum(read_leading_number(row[columns.index(SEEDING_COLUMN)]) for row in candidates), 0.0)
            whole = seeding_sum + sum(loss_sums.values())
            seeding_share = round(100 * seeding_sum / whole) if whole else 0
            segments = [f'<div class="bar-segment seeding" style="flex-basis:{seeding_share:g}%" '
                        f'title="Seeding: {seeding_share:g}% of the gap from a neutral start, already recovered by the '
                        f'app"></div>']
            legend = [f'<span><span class="swatch seeding"></span>Seeding {seeding_share:g}% &middot; already in the '
                      f'app (+{average_seeding:.3f} over a neutral start)</span>']
            span = 'the bar is the gap from a neutral start to the best found'
        for position, (cause, cause_sum) in enumerate(loss_sums.items()):
            share = round(100 * cause_sum / whole) if whole else 0
            segments.append(f'<div class="bar-segment series-{position}" style="flex-basis:{share:g}%" '
                            f'title="{escape(cause)}: {share:g}% of the bar"></div>')
            legend.append(f'<span><span class="swatch series-{position}"></span>{escape(cause)} {share:g}% '
                          f'&middot; fix: {escape(cost_row[columns.index(cause)])} the work</span>')
        bars.append((names[format_key], total, average_app, span, segments, legend))
    return ''.join(
        f'<h3>{name}: total loss <b>{total:.3f}%</b> <span class="note">(of an average H-score of '
        f'{average_app:.2f}%; {span})</span></h3>'
        f'<div class="bar-track">{"".join(segments)}</div>'
        f'<div class="legend">{"".join(legend)}</div>' for name, total, average_app, span, segments, legend in bars)


# How far below the best the app's seed's peak ends, in H-score points: the bins of the Peakedness chart for the seeds
# that miss the best (whether a seed reaches it is the stored yes/no, judged before rounding).
PEAK_SHORTFALL_EDGES = (0.05, 0.1, 0.5)
# Candidates with this many peaks or more share the last bar of the peaks chart.
PEAK_COUNT_CAP = 10


def card_peakedness(columns, rows):
    """Per format: how many candidates have more than one peak and how often the app's seed ends below the best, then
    two histograms -- the number of peaks per candidate, and how far below the best the app's seed's peak is."""
    format_index = columns.index('Format')
    names = {'EC': 'Each Category', 'MC': 'Most Categories', 'Roto': 'Rotisserie'}
    formats = [key for key in names if any(row[format_index] == key for row in rows)]
    by_format = {key: candidate_rows([row for row in rows if row[format_index] == key]) for key in formats}
    peaks = {key: [int(row[columns.index('Peaks')]) for row in by_format[key]] for key in formats}
    misses = {key: [read_leading_number(row[columns.index("App's seed's peak (below best)")])
                    for row in by_format[key] if row[columns.index("App's seed reaches best")] == 'no']
              for key in formats}

    def describe_format(key):
        count = len(peaks[key])
        if misses[key]:
            miss_text = (f"the app's seed ends below the best for {len(misses[key])} (median "
                         f'{np.median(misses[key]):.2f}, largest {max(misses[key]):.2f} points)')
        else:
            miss_text = "the app's seed reaches the best for every candidate"
        return (f'{names[key]}: {sum(value > 1 for value in peaks[key])} of {count} candidates have more than one '
                f'peak; {miss_text}.')

    peak_labels = [str(count) for count in range(1, PEAK_COUNT_CAP)] + [f'{PEAK_COUNT_CAP}+']
    peak_chart = render_chart(peak_labels,
                              [Series(key, [sum(value == count for value in peaks[key])
                                            for count in range(1, PEAK_COUNT_CAP)]
                                      + [sum(value >= PEAK_COUNT_CAP for value in peaks[key])]) for key in formats],
                              'Candidates', x_title="Peaks (distinct end points of the full menu's starts)",
                              kind='bars', value_places=0)
    edges = PEAK_SHORTFALL_EDGES
    labels = (['reaches best', f'under {edges[0]:g}'] + [f'{low:g}–{high:g}' for low, high in zip(edges, edges[1:])]
              + [f'{edges[-1]:g}+'])

    def bin_shortfalls(key):
        values = misses[key]
        return ([len(peaks[key]) - len(values), sum(value < edges[0] for value in values)]
                + [sum(low <= value < high for value in values) for low, high in zip(edges, edges[1:])]
                + [sum(value >= edges[-1] for value in values)])

    shortfall_chart = render_chart(labels, [Series(key, bin_shortfalls(key)) for key in formats],
                                   'Candidates', x_title="How far below the best the app's seed ends after 2,000 "
                                                         'iterations, H-score points', kind='bars', value_places=0)
    return (render_takeaway(' '.join(describe_format(key) for key in formats))
            + '<div class="note">How many peaks each candidate has.</div>' + peak_chart
            + "<div class=\"note\">Where the app's own seed ends: on the best peak, or how far below it.</div>"
            + shortfall_chart)


def styles_loss(columns, rows):
    """Green for no loss at all; red strengthening with the loss (in the summary row, with its share of the total),
    bold in the top half of the scale, which tops out at the 95th percentile of the losses; a gain stays uncoloured."""
    indices = [columns.index(name) for name in LOSS_COLUMNS if name in columns]
    total_index = columns.index('Total loss')
    seeding_index = columns.index(SEEDING_COLUMN) if SEEDING_COLUMN in columns else None
    losses = [read_leading_number(row[index]) for row in rows if not is_summary_row(row) for index in indices]
    losses = [loss for loss in losses if loss is not None and loss > 0]
    scale_top = float(np.percentile(losses, 95)) if losses else 1.0
    styles = []
    for row in rows:
        row_styles = [''] * len(row)
        # Seeding is a gain the app already has: green when it gains, uncoloured otherwise.
        if seeding_index is not None and (read_leading_number(row[seeding_index]) or 0) > 0:
            row_styles[seeding_index] = GOOD
        for index in indices:
            loss = read_leading_number(row[index])
            if loss is None:
                continue
            if is_summary_row(row):
                total = read_leading_number(row[total_index])
                if not total or index == total_index or any(str(cell).startswith('Cost to fix')
                                                            for cell in row[:2]):
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
        cell_styles=make_rule_styles({'Mean H-score': rule_season_win_rate,
                                      **{f'S{seat}': rule_seat_win_rate for seat in range(1, 13)}}),
        formatters={'Mean H-score': format_percent, 'Mean real win rate': format_percent,
                    **{f'S{seat}': format_percent for seat in range(1, 13)}},
        derive_table=derive_table_without_lists,
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
    'Draft pick stability': SectionView(
        card=card_draft_pick_stability,
        cell_styles=make_rule_styles({
            'Max dweight': lambda value, row, columns: GOOD if value <= 5 else BAD,
            'Punt flips': lambda value, row, columns: GOOD if value == 0 else BAD,
            'Shared shift (H-score)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
            'Shared shift (whole board)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
        }),
        formatters={'Shared shift (H-score)': lambda cell: format_percent(cell, signed=True),
                    'Max relative move (H-score)': format_percent},
        derive_table=derive_table_without_lists,
    ),
    'Auction pick stability': SectionView(
        card=card_auction_pick_stability,
        cell_styles=make_rule_styles({
            'Max dweight': lambda value, row, columns: GOOD if value <= 5 else BAD,
            'Punt flips': lambda value, row, columns: GOOD if value == 0 else BAD,
            'Shared shift (H-score)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
            'Shared shift (whole board)': lambda value, row, columns: '' if abs(value) <= 5 else BAD,
        }),
        formatters={'Shared shift (H-score)': lambda cell: format_percent(cell, signed=True),
                    'Max relative move (H-score)': format_percent},
        derive_table=derive_table_without_lists,
    ),
    'Self-play convergence': SectionView(card=card_self_play, cell_styles=styles_self_play, row_filter=filter_self_play),
    'Self-play optimality': SectionView(
        card=card_best_responses,
        derive_table=derive_optimality_table,
    ),
    'Serve stability': SectionView(card=card_serve_stability, derive_table=derive_table_without_lists,
                                   wrap_columns=('Top 12 unseated',)),
    'How single-peaked are H-scores': SectionView(
        card=card_seed_starts,
        derive_table=derive_seed_miss_table,
        # Dark for a real second peak, light for a start that stops on the flat ridge by the optimum (judged by the
        # largest shortfall in the cell, a range's upper end).
        cell_styles=make_rule_styles({'Shortfall(s)': lambda value, row, columns: style_bad(
            1.0 if float(re.findall(r'[\d.]+', str(row[columns.index('Shortfall(s)')]))[-1]) >= SECOND_PEAK_SHORTFALL
            else 0.2)}),
        wrap_columns=('Seeds that miss',),
        row_filter=describe_seed_misses,
    ),
    'Conditioning around the optimum': SectionView(card=card_conditioning),
    'Loop gain at the optimum': SectionView(
        card=card_loop_gain,
        cell_styles=make_rule_styles({'Largest loop gain': rule_loop_gain,
                                      'Largest Hessian eigenvalue':
                                          lambda value, row, columns: GOOD if value < 0 else BAD}),
        sort_rows=sort_by_loop_gain,
    ),
    'Where the app loses to the optimum': SectionView(card=card_loss, cell_styles=styles_loss),
    'App loss': SectionView(card=card_app_loss, cell_styles=styles_loss),
    'Peakedness': SectionView(card=card_peakedness),
    'Planner vs H-scoring': SectionView(
        card=card_planner_drafts,
        cell_styles=styles_planner_pairs,
        derive_table=derive_planner_pair_table,
        formatters={'Planner minus H (pp)': lambda cell: format_percent(cell, signed=True),
                    **{f'S{seat}': (lambda cell: format_percent(cell, signed=True)) for seat in range(1, 13)}},
        row_filter=lambda columns, rows: (rows, 'Each cell is that seat drafting with the planner minus the same seat '
                                                'drafting by H-scoring, in the same pair. Purple: H-scoring better; '
                                                'green: the planner better.'),
    ),
}

# Column headers shown without the '(pp)' unit: the cells carry '%'.
HEADER_RENAMES = {'Mean (pp)': 'Mean', 'SE (pp)': 'SE', 'Mean gain (pp)': 'Mean gain', 'Max gain (pp)': 'Max gain',
                  'Planner minus H (pp)': 'Planner minus H', 'Worst (pp)': 'Worst', 'Best (pp)': 'Best',
                  'Median dip (pp)': 'Median dip', 'Largest dip (pp)': 'Largest dip',
                  'Worst miss (pp)': 'Worst miss', 'Shortfall (pp)': 'Shortfall'}


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
    details = (f'<div class="details"><b>Details</b>{render_explanation_html(data["details"])}</div>'
               if data.get('details') else '')
    return card + details + note + render_table(columns, rows, view)


def describe_generation(data: dict) -> str:
    """When the page's results were generated: one time, or one per format when its formats were run separately."""
    by_format = data.get('generated_by_format')
    if not by_format or len(set(by_format.values())) == 1:
        return f'generated {escape(data["generated"])}'
    return 'generated ' + ', '.join(f'{escape(format_key)} {escape(when)}' for format_key, when in by_format.items())


def render_section_panel(data: dict, position: int, parts: list[dict]) -> str:
    """One tab's panel: title, generation time, explanation, takeaway card, table; then each part (another stored
    section of the same experiment) under its own heading, with its explanation, card and table."""
    active = ' class="active"' if position == 0 else ''
    rendered_parts = ''.join(f'\n<h2 class="part">{escape(PART_HEADINGS[part["title"]])}</h2>\n'
                             f'{render_explanation_html(part["explanation"])}\n{render_section_body(part)}'
                             for part in parts)
    return (f'<section id="tab-{position}"{active}>\n<h2>{escape(data["title"])}</h2>\n'
            f'<div class="generated">{describe_generation(data)}</div>\n'
            f'{render_explanation_html(data["explanation"])}\n{render_section_body(data)}{rendered_parts}\n</section>')


# Stored sections that are parts of another section's page, not tabs of their own: {page: [part, ...]}. An experiment
# that records two tables still has one page.
SECTION_PARTS = {
    'Smooth vs simple scaling': ['Cash-share rules: builds (own view)'],
}
PART_HEADINGS = {'Cash-share rules: builds (own view)': 'Builds'}


# The tab groups, in page order; a stored section in none of them is listed last under 'Other'.
SECTION_GROUPS = {
    # Does H-scoring win drafts?
    'Draft performance': [
        'H-scoring vs a G-score field',
        'Awareness (opponent model on vs off)',
        'Planner vs H-scoring',
    ],
    # Does self-play find an equilibrium?
    'Self-play': [
        'Self-play convergence',
        'Self-play optimality',
    ],
    # Does the board move when nothing has really changed?
    'Stability': [
        'Serve stability',
        'Draft pick stability',
        'Auction pick stability',
    ],
    # Are the builds the app predicts and serves sensible?
    'Served builds': [
        'Punt diversity (top-12 anchors)',
        'Rotisserie punting (hard <15% / soft 15-40% expected win rate)',
    ],
    # How far is the app's answer from the best build, on the app's actual objective?
    'Gradient descent optimality': [
        'App loss',
        'Peakedness',
    ],
    # The same questions on a simplified objective, against a certified optimum.
    'Simplified form with exact solution': [
        'Where the app loses to the optimum',
        'How single-peaked are H-scores',
        'Conditioning around the optimum',
        'Loop gain at the optimum',
    ],
    'Auctions': [
        'Smooth vs simple scaling',
        'Dollar shape: replacement level and SAVOR',
    ],
}


def group_sections(sections: list[dict]) -> list[tuple[str, list[dict]]]:
    """The stored sections under their group names, in SECTION_GROUPS order."""
    by_title = {data['title']: data for data in sections}
    for part_titles in SECTION_PARTS.values():
        for title in part_titles:
            by_title.pop(title, None)
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
    by_title = {data['title']: data for data in sections}
    for group_index, (group_name, members) in enumerate(group_sections(sections)):
        group_active = ' class="active"' if group_index == 0 else ''
        group_buttons.append(f'<button data-group="{group_index}"{group_active}>{escape(group_name)}</button>')
        buttons = []
        for data in members:
            active = ' class="active"' if position == 0 else ''
            buttons.append(f'<button data-tab="{position}"{active}>{escape(data["title"].split(" (")[0])}</button>')
            parts = [by_title[title] for title in SECTION_PARTS.get(data['title'], []) if title in by_title]
            panels.append(render_section_panel(data, position, parts))
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
