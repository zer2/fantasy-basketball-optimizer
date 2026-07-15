# testing_files/season_simulation/render.py
# Part 2 of the season-simulation harness: read the JSON produced by simulate.py and build an HTML
# report. Nothing is simulated here.
#
# Layout: one section per scoring format (EC / MC / Roto). Each section is a table whose columns are
# the seats (Drafter 1..N = "the H-score drafter sitting here"), and whose rows are a (season,
# category) MultiIndex — an "Overall" H-score row plus one row per category, per season. Cells are
# coloured with the exact same stat styler the website uses (ported below). Every team cell is
# clickable: it lazily fetches that (season, format)'s JSON and renders the team's detail — its
# roster, final H-score, and the per-pick candidate tables the website would have shown.
#
# Run:  python -m testing_files.season_simulation.render
# Then: cd testing_files/season_simulation/output && python -m http.server   # open index.html

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE / 'output' / 'data'
_OUT_HTML = _HERE / 'output' / 'index.html'

_FORMAT_ORDER = ['EC', 'MC', 'Roto']
_FORMAT_TITLES = {
    'EC':   'Head to Head — Each Category',
    'MC':   'Head to Head — Most Categories',
    'Roto': 'Rotisserie',
}


# ── Stat styler, ported verbatim from frontend/styles/styler_functions.ts ──────────────────────────

def _blend_from_white(target: tuple[int, int, int], intensity: float, cap: int) -> tuple[int, int, int]:
    t = min(intensity, cap) / cap * 0.7   # LIGHT_BLEND_MAX
    return tuple(round(255 + t * (c - 255)) for c in target)


def _light_primary(value: float, multiplier: float, middle: float) -> tuple[int, int, int]:
    raw = (value - middle) * multiplier
    target = (70, 160, 100) if raw > 0 else (205, 80, 80)
    return _blend_from_white(target, abs(raw), 110)


def _dark_primary(value: float, multiplier: float, middle: float) -> tuple[int, int, int]:
    raw = (value - middle) * multiplier
    intensity = min(round(abs(raw)), 110)
    return (
        55 if raw > 0 else 55 + intensity,
        55 + intensity if raw > 0 else 55,
        70 + round(intensity * 0.7),
    )


def _pick_text(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    return 'black' if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else 'white'


def stat_style(value: float, multiplier: float, middle: float) -> str:
    """CSS `style` string matching stat_styler_primary (dual light-dark output)."""
    light = _light_primary(value, multiplier, middle)
    dark  = _dark_primary(value, multiplier, middle)
    return (f'color:light-dark({_pick_text(light)},{_pick_text(dark)});'
            f'background-color:light-dark(rgb{light},rgb{dark});')


# ── Per-format cell math (value shown + styler params), mirroring the frontend tables ──────────────

def _cell_value_and_style(format_key: str, rate: float, n_drafters: int) -> tuple[str, str]:
    """(display text, css) for a per-category cell. `rate` is the stored 0-100 value."""
    if format_key == 'Roto':
        roto_middle = (n_drafters - 1) / 2 + 1
        roto_value  = 1 + (rate / 100) * (n_drafters - 1)
        return f'{roto_value:.2f}', stat_style(roto_value, 3 * (n_drafters - 1), roto_middle)
    return f'{rate:.1f}', stat_style(rate, 3, 50)   # EC / MC win rate


def _overall_value_and_style(format_key: str, h_score: float, n_drafters: int) -> tuple[str, str]:
    """(display text, css) for the Overall H-score row. `h_score` is 0-100."""
    if format_key == 'Roto':
        # Roto overall H-scores are small (~8%); colour on their own modest scale around a neutral 8.
        return f'{h_score:.1f}', stat_style(h_score, 6, 8)
    return f'{h_score:.1f}', stat_style(h_score, 2, 50)   # EC / MC probability


# ── HTML assembly ──────────────────────────────────────────────────────────────────────────────────

def _load_records() -> dict[str, list[dict]]:
    """Group the per-(season, format) JSON files by format key, sorted by season descending."""
    by_format: dict[str, list[dict]] = {key: [] for key in _FORMAT_ORDER}
    for path in sorted(_DATA_DIR.glob('*.json')):
        record = json.loads(path.read_text(encoding='utf-8'))
        by_format.setdefault(record['format_key'], []).append(record)
    for records in by_format.values():
        records.sort(key=lambda r: r['season'], reverse=True)
    return by_format


def _render_format_table(format_key: str, records: list[dict]) -> str:
    if not records:
        return f'<h2>{_FORMAT_TITLES[format_key]}</h2><p class="empty">No data.</p>'

    n_drafters = records[0]['n_drafters']
    categories = records[0]['categories']
    seats_present = sorted({seat['hscore_seat'] for r in records for seat in r['seats']})

    head = ''.join(f'<th>Drafter {seat + 1}</th>' for seat in seats_present)
    rows_html: list[str] = []

    for record in records:
        season = record['season']
        by_seat = {seat['hscore_seat']: seat for seat in record['seats']}
        row_labels = ['H-score'] + list(categories)
        flag = ' <span class="noposition" title="No position data for this season">✦</span>' if not record['has_position_data'] else ''

        for row_i, label in enumerate(row_labels):
            cells: list[str] = []
            for seat in seats_present:
                seat_data = by_seat.get(seat)
                if seat_data is None:
                    cells.append('<td class="missing"></td>')
                    continue
                if row_i == 0:
                    text, css = _overall_value_and_style(format_key, seat_data['team_h_score'], n_drafters)
                else:
                    text, css = _cell_value_and_style(format_key, seat_data['team_rates'][row_i - 1], n_drafters)
                cells.append(
                    f'<td class="stat" style="{css}" '
                    f'data-season="{season}" data-format="{format_key}" data-seat="{seat}" '
                    f'title="Drafter {seat + 1}, {season} — click for team detail">{text}</td>'
                )
            season_cell = (f'<th class="season" rowspan="{len(row_labels)}">{season}{flag}</th>'
                           if row_i == 0 else '')
            rows_html.append(f'<tr>{season_cell}<th class="rowlabel">{label}</th>{"".join(cells)}</tr>')

    return (
        f'<h2 id="{format_key}">{_FORMAT_TITLES[format_key]}</h2>'
        f'<table class="report"><thead><tr>'
        f'<th class="season">Season</th><th class="rowlabel"></th>{head}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody></table>'
    )


def _page(body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Season Simulation — H-score Results</title>
<style>{_CSS}</style>
</head><body>
<header>
  <h1>Season Simulation — H-score vs G-score field</h1>
  <p>One seat drafts by H-score, the other eleven by G-score order. Each cell is that H-score drafter's
     result at that seat, measured in H-score terms. Click any cell for the team's detail.
     <span class="noposition">✦</span> = season has no position data.</p>
</header>
{body}
<div id="overlay" class="overlay hidden" onclick="if(event.target===this)closeDetail()">
  <div id="panel" class="panel"><button class="close" onclick="closeDetail()">✕</button>
  <div id="panel-body"></div></div>
</div>
<script>{_JS}</script>
</body></html>"""


def main() -> None:
    by_format = _load_records()
    sections = ''.join(_render_format_table(key, by_format.get(key, [])) for key in _FORMAT_ORDER)
    _OUT_HTML.write_text(_page(sections), encoding='utf-8')
    total_cells = sum(len(r['seats']) for records in by_format.values() for r in records)
    print(f'Wrote {_OUT_HTML}  ({total_cells} team cells across '
          f'{sum(len(v) for v in by_format.values())} season-formats)')
    print(f'View: cd {_OUT_HTML.parent} && python -m http.server   # then open index.html')


# ── Static assets (CSS + the lazy-detail JS, which re-implements the styler client-side) ────────────

_CSS = """
:root { color-scheme: light dark; font-family: 'Source Sans 3', system-ui, sans-serif; }
body { margin: 0 auto; max-width: 1400px; padding: 16px 24px 80px; }
header p { color: light-dark(#555,#aaa); max-width: 900px; }
.noposition { color: light-dark(#b26a00,#e0a34a); }
h2 { margin-top: 40px; border-bottom: 1px solid light-dark(#ddd,#3d3d55); padding-bottom: 4px; }
table.report { border-collapse: collapse; font-size: 13px; }
table.report th, table.report td { border: 1px solid light-dark(#e5e5e5,#2a2c3a); padding: 3px 7px; text-align: right; }
th.season { position: sticky; left: 0; background: light-dark(#fff,#0e1117); text-align: left; font-weight: 700; vertical-align: top; }
th.rowlabel { text-align: left; color: light-dark(#555,#aaa); font-weight: 400; white-space: nowrap; }
thead th { background: light-dark(#f5f5f5,#161922); position: sticky; top: 0; }
td.stat { cursor: pointer; font-variant-numeric: tabular-nums; }
td.stat:hover { outline: 2px solid light-dark(#333,#ccc); outline-offset: -2px; }
td.missing, td.empty { background: light-dark(#fafafa,#111); }
.overlay { position: fixed; inset: 0; background: rgba(0,0,0,.55); display: flex; align-items: flex-start; justify-content: center; padding: 40px; overflow: auto; }
.overlay.hidden { display: none; }
.panel { background: light-dark(#fff,#0e1117); border: 1px solid light-dark(#ddd,#3d3d55); border-radius: 8px; padding: 20px 24px; max-width: 1100px; width: 100%; position: relative; }
.panel .close { position: absolute; top: 10px; right: 12px; border: none; background: none; font-size: 20px; cursor: pointer; color: inherit; }
.panel h3 { margin: 0 0 4px; } .panel h4 { margin: 18px 0 6px; }
.pick { margin: 14px 0; border-top: 1px solid light-dark(#eee,#2a2c3a); padding-top: 8px; }
.detail-table { border-collapse: collapse; font-size: 12px; margin-top: 4px; }
.detail-table th, .detail-table td { border: 1px solid light-dark(#eee,#2a2c3a); padding: 2px 6px; text-align: right; }
.detail-table td.name, .detail-table th.name { text-align: left; white-space: nowrap; }
.picked { font-weight: 700; }
"""

# The client-side detail renderer. Re-implements the stat styler (kept byte-identical to the Python
# port above and to styler_functions.ts) so the lazily-rendered per-pick tables colour like the app.
_JS = r"""
const CACHE = {};
function blendFromWhite(t, intensity, cap){ const k=Math.min(intensity,cap)/cap*0.7;
  return [Math.round(255+k*(t[0]-255)),Math.round(255+k*(t[1]-255)),Math.round(255+k*(t[2]-255))]; }
function lightPrimary(v,m,mid){ const raw=(v-mid)*m; const t=raw>0?[70,160,100]:[205,80,80]; return blendFromWhite(t,Math.abs(raw),110); }
function darkPrimary(v,m,mid){ const raw=(v-mid)*m; const i=Math.min(Math.round(Math.abs(raw)),110);
  return [raw>0?55:55+i, raw>0?55+i:55, 70+Math.round(i*0.7)]; }
function txt(c){ return (c[0]*0.299+c[1]*0.587+c[2]*0.114)>150?'black':'white'; }
function statStyle(v,m,mid){ const l=lightPrimary(v,m,mid), d=darkPrimary(v,m,mid);
  return `color:light-dark(${txt(l)},${txt(d)});background-color:light-dark(rgb(${l}),rgb(${d}));`; }

function catCell(formatKey, rate, nd){
  if(formatKey==='Roto'){ const mid=(nd-1)/2+1, rv=1+(rate/100)*(nd-1); return [rv.toFixed(2), statStyle(rv,3*(nd-1),mid)]; }
  return [rate.toFixed(1), statStyle(rate,3,50)];
}

async function loadData(season, formatKey){
  const key = season+'__'+formatKey;
  if(!CACHE[key]) CACHE[key] = await (await fetch('data/'+key+'.json')).json();
  return CACHE[key];
}

async function openDetail(season, formatKey, seat){
  const rec = await loadData(season, formatKey);
  const seatData = rec.seats.find(s => s.hscore_seat === seat);
  const cats = rec.categories, nd = rec.n_drafters;
  let html = `<h3>Drafter ${seat+1} — ${season} · ${formatKey}</h3>`;
  html += `<p><b>Final team H-score:</b> ${seatData.team_h_score}%</p>`;
  html += `<h4>Roster</h4><ol>` + seatData.roster.map(p=>`<li>${p}</li>`).join('') + `</ol>`;
  html += `<h4>How each pick was made (H-score candidate tables)</h4>`;
  for(const pick of seatData.picks){
    html += `<div class="pick"><b>Round ${pick.round}</b> → <span class="picked">${pick.picked}</span>`;
    html += `<table class="detail-table"><thead><tr><th class="name">Candidate</th><th>H</th>` +
            cats.map(c=>`<th>${c.split(' ')[0]}</th>`).join('') + `</tr></thead><tbody>`;
    for(const c of pick.candidates.slice(0, 12)){
      const cls = c.name===pick.picked ? ' class="picked"' : '';
      const [ov, ost] = formatKey==='Roto' ? [c.h_score.toFixed(1), statStyle(c.h_score,6,8)] : [c.h_score.toFixed(1), statStyle(c.h_score,2,50)];
      html += `<tr${cls}><td class="name">${c.name}</td><td style="${ost}">${ov}</td>` +
        c.win_rates.map(r=>{ const [t,s]=catCell(formatKey,r,nd); return `<td style="${s}">${t}</td>`; }).join('') + `</tr>`;
    }
    html += `</tbody></table></div>`;
  }
  document.getElementById('panel-body').innerHTML = html;
  document.getElementById('overlay').classList.remove('hidden');
}
function closeDetail(){ document.getElementById('overlay').classList.add('hidden'); }
document.addEventListener('click', e => {
  const td = e.target.closest('td.stat'); if(!td) return;
  openDetail(td.dataset.season, td.dataset.format, parseInt(td.dataset.seat));
});
document.addEventListener('keydown', e => { if(e.key==='Escape') closeDetail(); });
"""


if __name__ == '__main__':
    main()
