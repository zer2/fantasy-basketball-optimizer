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
from statistics import mean

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

# Single colour multiplier for the overall H-score cells across all three formats, so the mapping is
# consistent (a given delta from neutral is the same colour everywhere). Set high enough that EC's
# tight spread (~50-57%) shows a clear gradient; wider-spread MC/Roto simply saturate sooner. One knob.
_OVERALL_MULTIPLIER = 8


def _overall_style(format_key: str, h_score: float, n_drafters: int) -> tuple[str, str]:
    """(display text, css) for a team's overall H-score (0-100). The neutral midpoint is the average
    outcome: 50% for EC/MC, and 1/N (≈ 8.3% for 12 drafters) for Rotisserie."""
    middle = 100 / n_drafters if format_key == 'Roto' else 50
    return f'{h_score:.1f}', stat_style(h_score, _OVERALL_MULTIPLIER, middle)


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
    """One row per season of overall team H-scores (columns = seats), a right Avg column, and an
    'Average' summary row across seasons at the top. Per-category detail lives only in the drill-down."""
    if not records:
        return f'<h2 id="{format_key}">{_FORMAT_TITLES[format_key]}</h2><p class="empty">No data.</p>'

    n_drafters    = records[0]['n_drafters']
    seats_present = sorted({seat['hscore_seat'] for r in records for seat in r['seats']})

    def overall_cell(h_score: float, extra: str) -> str:
        text, css = _overall_style(format_key, h_score, n_drafters)
        return f'<td style="{css}"{extra}>{text}</td>'

    def avg_cell(values: list[float], css_class: str) -> str:
        return (overall_cell(mean(values), f' class="{css_class}"') if values
                else f'<td class="{css_class} missing"></td>')

    across: dict[int, list[float]] = {seat: [] for seat in seats_present}
    body_rows: list[str] = []
    for record in sorted(records, key=lambda r: r['season'], reverse=True):
        by_seat = {seat['hscore_seat']: seat for seat in record['seats']}
        cells, present = [], []
        for seat in seats_present:
            seat_data = by_seat.get(seat)
            if seat_data is None:
                cells.append('<td class="missing"></td>')
                continue
            h_score = seat_data['team_h_score']
            present.append(h_score)
            across[seat].append(h_score)
            cells.append(overall_cell(h_score,
                f' class="stat" data-season="{record["season"]}" data-format="{format_key}"'
                f' data-seat="{seat}" title="Drafter {seat + 1}, {record["season"]} — click for detail"'))
        flag = (' <span class="noposition" title="No position data for this season">✦</span>'
                if not record['has_position_data'] else '')
        body_rows.append(f'<tr><th class="season">{record["season"]}{flag}</th>'
                         f'{"".join(cells)}{avg_cell(present, "avg avgcol")}</tr>')

    summary_cells, summary_present = [], []
    for seat in seats_present:
        if across[seat]:
            seat_mean = mean(across[seat])
            summary_present.append(seat_mean)
            summary_cells.append(overall_cell(seat_mean, ' class="avg"'))
        else:
            summary_cells.append('<td class="avg missing"></td>')
    summary_row = (f'<tr><th class="season">Average</th>'
                   f'{"".join(summary_cells)}{avg_cell(summary_present, "avg avgcol")}</tr>')

    head = ''.join(f'<th>Drafter {seat + 1}</th>' for seat in seats_present)
    return (
        f'<h2 id="{format_key}">{_FORMAT_TITLES[format_key]}</h2>'
        f'<table class="report"><thead><tr>'
        f'<th class="season">Season</th>{head}<th class="avgh">Avg</th></tr></thead>'
        f'<tbody class="summary">{summary_row}</tbody><tbody>{"".join(body_rows)}</tbody></table>'
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
th.season { position: sticky; left: 0; background: light-dark(#fff,#0e1117); text-align: left; font-weight: 700; white-space: nowrap; }
thead th { background: light-dark(#f5f5f5,#161922); position: sticky; top: 0; }
td.stat, td.avg { font-variant-numeric: tabular-nums; }
td.stat { cursor: pointer; }
td.stat:hover { outline: 2px solid light-dark(#333,#ccc); outline-offset: -2px; }
td.missing, td.empty { background: light-dark(#fafafa,#111); }
/* Top "Average across seasons" summary block. */
tbody.summary { border-bottom: 2px solid light-dark(#bbb,#555); }
tbody.summary td, tbody.summary th.season { font-weight: 600; }
/* Right "Avg across seats" column. */
td.avgcol, th.avgh { border-left: 2px solid light-dark(#bbb,#555); font-weight: 600; }
/* Per-pick drill-down tables in the detail panel. */
.pick > details > summary { cursor: pointer; user-select: none; }
.ptable { border-collapse: collapse; font-size: 11px; margin: 4px 0 8px; }
.ptable th, .ptable td { border: 1px solid light-dark(#eee,#2a2c3a); padding: 1px 5px; text-align: right; font-variant-numeric: tabular-nums; }
.ptable th.name, .ptable td.name { text-align: left; white-space: nowrap; }
.ptable td.ineligible { background: light-dark(#f3f3f3,#181a22); }
.ptable td.cand { font-weight: 700; }
.ptable caption { text-align: left; color: light-dark(#666,#999); font-size: 11px; padding: 2px 0; }
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
// Stat styler ported from frontend/styles/styler_functions.ts (primary + tertiary), so the lazily
// rendered per-pick tables colour exactly like the website's candidate table and expand view.
function blendFromWhite(t, intensity, cap){ const k=Math.min(intensity,cap)/cap*0.7;
  return [Math.round(255+k*(t[0]-255)),Math.round(255+k*(t[1]-255)),Math.round(255+k*(t[2]-255))]; }
function txt(c){ return (c[0]*0.299+c[1]*0.587+c[2]*0.114)>150?'black':'white'; }
function dual(l,d){ return `color:light-dark(${txt(l)},${txt(d)});background-color:light-dark(rgb(${l}),rgb(${d}));`; }
function lightPrimary(v,m,mid){ const raw=(v-mid)*m; return blendFromWhite(raw>0?[70,160,100]:[205,80,80],Math.abs(raw),110); }
function darkPrimary(v,m,mid){ const raw=(v-mid)*m, i=Math.min(Math.round(Math.abs(raw)),110);
  return [raw>0?55:55+i, raw>0?55+i:55, 70+Math.round(i*0.7)]; }
function statStyle(v,m,mid){ return dual(lightPrimary(v,m,mid), darkPrimary(v,m,mid)); }
function lightTertiary(v,m,mid){ return blendFromWhite([120,150,215],Math.abs((v-mid)*m),100); }
function darkTertiary(v,m,mid){ const raw=Math.round((v-mid)*m), i=Math.min(Math.abs(raw),130);
  return [raw>0?28+Math.round(i/6):28-Math.round(i/60), raw>0?34+Math.round(i/2):34-Math.round(i/20), raw>0?46+Math.round(i*0.7):46-Math.round(i/10)]; }
function statTertiary(v,m,mid){ return dual(lightTertiary(v,m,mid), darkTertiary(v,m,mid)); }

function catCell(formatKey, rate, nd){
  if(formatKey==='Roto'){ const mid=(nd-1)/2+1, rv=1+(rate/100)*(nd-1); return [rv.toFixed(2), statStyle(rv,3*(nd-1),mid)]; }
  return [rate.toFixed(1), statStyle(rate,3,50)];
}
const OVERALL_MULT = 8;   // keep in sync with _OVERALL_MULTIPLIER in render.py
function overallCell(formatKey, h, nd){ const mid = formatKey==='Roto' ? 100/nd : 50; return [h.toFixed(1), statStyle(h, OVERALL_MULT, mid)]; }

function candidateTableHTML(pick, cats, formatKey, nd){
  let h = `<table class="ptable"><caption>Candidate ranking (top ${Math.min(pick.candidates.length,15)})</caption>`+
          `<thead><tr><th class="name">Candidate</th><th>H</th>`+cats.map(c=>`<th>${c.split(' ')[0]}</th>`).join('')+`</tr></thead><tbody>`;
  for(const c of pick.candidates.slice(0,15)){
    const [ov,ost] = overallCell(formatKey, c.h_score, nd);
    const cls = c.name===pick.picked ? ' class="cand"' : '';
    h += `<tr${cls}><td class="name">${c.name}</td><td style="${ost}">${ov}</td>`+
      c.win_rates.map(r=>{ const [t,s]=catCell(formatKey,r,nd); return `<td style="${s}">${t}</td>`; }).join('')+`</tr>`;
  }
  return h+`</tbody></table>`;
}
function gscoreTableHTML(detail, cats){
  if(!detail.g_score_rows) return '';
  let h = `<table class="ptable"><caption>G-score expectations (difference vs. other teams)</caption>`+
          `<thead><tr><th class="name"></th><th>Total</th>`+cats.map(c=>`<th>${c.split(' ')[0]}</th>`).join('')+`</tr></thead><tbody>`;
  for(const r of detail.g_score_rows){
    h += `<tr><th class="name">${r.label}</th><td>${r.total.toFixed(2)}</td>`+
      r.values.map(v=>`<td style="${statStyle(v,60,0)}">${v.toFixed(2)}</td>`).join('')+`</tr>`;
  }
  return h+`</tbody></table>`;
}
function flexTableHTML(detail){
  const f = detail.flex_allocations; if(!f) return '';
  let h = `<table class="ptable"><caption>Position allocations for future flex picks</caption>`+
          `<thead><tr><th class="name"></th>`+f.base_positions.map(p=>`<th>${p}</th>`).join('')+`</tr></thead><tbody>`;
  for(const r of f.rows){
    h += `<tr><th class="name">${r.label}</th>`+
      r.values.map(v=> v===null ? `<td class="ineligible"></td>` : `<td style="${statTertiary(v,50,0)}">${v.toFixed(2)}</td>`).join('')+`</tr>`;
  }
  return h+`</tbody></table>`;
}
function rosterTableHTML(roster){
  if(!roster) return '';
  const types=[], byType={};
  for(const slot of roster.slots){ const t=slot.replace(/\d+$/,''); if(!byType[t]){byType[t]=[];types.push(t);} byType[t].push(slot); }
  const maxDepth = Math.max(...types.map(t=>byType[t].length));
  let h = `<table class="ptable"><caption>Roster assignments</caption><thead><tr><th class="name"></th>`+
          types.map(t=>`<th>${t}</th>`).join('')+`</tr></thead><tbody>`;
  for(let d=0; d<maxDepth; d++){
    h += `<tr><th class="name">Slot ${d+1}</th>`;
    for(const t of types){ const slot=byType[t][d];
      if(slot===undefined){ h+=`<td class="ineligible"></td>`; continue; }
      const a = roster.assignments[slot];
      h += a ? `<td class="name${a.is_candidate?' cand':''}">${a.name}</td>` : `<td></td>`;
    }
    h += `</tr>`;
  }
  return h+`</tbody></table>`;
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
  let html = `<h3>Drafter ${seat+1} — ${season} · ${formatKey}`+(rec.has_position_data?'':' · no position data')+`</h3>`;
  html += `<p><b>Final team H-score:</b> ${seatData.team_h_score}%</p>`;
  html += `<h4>Roster</h4><ol>` + seatData.roster.map(p=>`<li>${p}</li>`).join('') + `</ol>`;
  html += `<h4>How each pick was made — the tables the website would have shown</h4>`;
  for(const pick of seatData.picks){
    html += `<div class="pick"><details><summary><b>Round ${pick.round}</b> &rarr; <span class="picked">${pick.picked}</span></summary>`+
      candidateTableHTML(pick, cats, formatKey, nd)+
      gscoreTableHTML(pick.picked_detail, cats)+
      flexTableHTML(pick.picked_detail)+
      rosterTableHTML(pick.picked_detail.roster)+
      `</details></div>`;
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
