// table/player_table.ts
// Renders the H-score candidate table: headers, player rows, expand buttons.
// Reads current player and category state from app_state.ts.
//
// VIRTUALIZED. The full candidate list lives in `candidateRows` (data + a pre-built, detached row pair
// each), but only the rows inside the current scroll window are attached to the DOM. Two spacer rows —
// one above, one below — stand in for the off-window rows' height (the bottom spacer also reserves height
// for candidates still streaming in), so the scrollbar and scroll position match the full list while the
// DOM holds ~60 cells instead of thousands. A rAF-throttled scroll listener re-syncs the window.
//
// Why: the table is one big <table>. Expanding/collapsing a detail row changes the height of a row and
// forces the browser to REPAINT every row below it (table rows can't be composited or skipped via
// content-visibility). With hundreds of rows that was ~400ms of paint per toggle; with only the visible
// window in the DOM, a toggle repaints a handful of rows.
//
// The table is still built incrementally so draft/waiver evaluations can stream in batches (top-ranked
// first): resetTable() clears it, addBatch() merges a sorted batch into `candidateRows` (data only), and
// renderWindow() syncs the DOM. buildTable() is the single-shot entry point (auction).

import { stat_styler_primary, stat_styler_secondary, H_MULTIPLIER } from '../styles/styler_functions.js'
import { buildExpandPanel } from './expand_view.js'
import { PlayerResult } from '../types.js'
import { getScoringFormat, getSelectedCategories } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getShortCategoryNames } from '../app_state.js'
import { isMobileViewport } from '../helper_functions.js'
import { buildFullPlayerDisplayHtml } from '../player_display.js'

const table = document.getElementById('hscoretable') as HTMLTableElement

// ── Column widths ────────────────────────────────────────────────────────────
// Floor values in rem (1rem = root font-size, see styles.css). Below these the
// table is unreadable; above them the browser (table-layout: fixed) distributes
// the extra width to category columns. Using rem so widths shrink alongside
// the smaller mobile root font.

const PLAYER_COL_W_REM = 5
const SCORE_COL_W_REM  = 1.5
const CAT_COL_W_REM    = 3      // desired minimum width per category column

function computeContentMinWidth(): string {
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'
    const nScoreCols = isAuction ? 4 : 1
    const categories = getSelectedCategories()
    return (PLAYER_COL_W_REM + nScoreCols * SCORE_COL_W_REM + categories.length * CAT_COL_W_REM) + 'rem'
}

// ── Virtual render state ───────────────────────────────────────────────────────
// A candidate: its player data, its two pre-built (detached) DOM rows, and its expand state. `expanded`
// is whether its detail panel is open; `expandedHeight` is the panel's measured pixel height (0 when
// collapsed), used to keep the spacer maths correct when the expanded row scrolls out of the window.
interface CandidateRow {
    player:         PlayerResult
    displayRow:     HTMLTableRowElement
    expandRow:      HTMLTableRowElement
    expanded:       boolean
    expandedHeight: number
}

// The virtual "dataframe": candidate rows in descending H-score order (mirrors what the DOM would show).
let candidateRows: CandidateRow[] = []
let tbodyEl: HTMLTableSectionElement | null = null
let totalCandidates = 0     // final count (>= candidateRows.length); reserves scroll height while streaming
let rowHeight = 0           // measured height of one collapsed display row (uniform)

// Window / scroll bookkeeping.
const WINDOW_BUFFER = 8     // extra rows rendered above & below the viewport so fast scrolls don't flash blank
let scrollWired = false
let scrollScheduled = false
let renderedStart = -1
let renderedEnd = -1
let windowDirty = false     // force a re-render even if the [start,end) window is unchanged (expand/batch/reserve)
let topSpacer: HTMLTableRowElement | null = null
let bottomSpacer: HTMLTableRowElement | null = null
// Spacer rows must span exactly the real column count. A colSpan larger than the number of columns the
// data rows define makes the browser lay the table out with that many columns under table-layout:fixed,
// squishing the real columns. Set from renderCtx (player col + score col(s) + one per category).
let columnCount = 1
// Id of a candidate to visually mark (the waiver drop-player highlight), or null. Applied by
// renderWindow to whichever windowed row matches, so it survives scrolling under virtualization.
let highlightedPlayerId: number | null = null

// Per-render context, captured once in resetTable so every addBatch renders rows consistently.
interface RenderContext {
    categories: string[]
    isAuction:  boolean
    isRoto:     boolean
    rotoData:   { nDrafters: number; rotoMiddle: number } | null
    decimals:   number
}
let renderCtx: RenderContext | null = null

/** Resets all virtual-render state (kept in one place so every entry point clears it consistently). */
function resetVirtualState(): void {
    candidateRows   = []
    tbodyEl         = null
    totalCandidates = 0
    rowHeight       = 0
    renderedStart   = -1
    renderedEnd     = -1
    windowDirty     = false
    topSpacer       = null
    bottomSpacer    = null
    highlightedPlayerId = null
}

/** Clears the candidate table and shows a single centred message row. */
export function showTableMessage(message: string): void {
    table.innerHTML = ''
    resetVirtualState()
    renderCtx = null
    const row = table.createTBody().insertRow()
    const cell = row.insertCell()
    cell.colSpan = 100
    cell.className = 'table-message'
    cell.textContent = message
}

/** Clears the candidate table and rebuilds its width container, column structure,
 *  and header row. Categories, mode, and scoring format are read from the DOM. */
export function buildTableHeader(): void {

    const categories = getSelectedCategories()
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'

    const widthContainer = document.getElementById('panel-content-width-container')!
    // The min-width acts as a desktop floor so the candidate table doesn't get
    // unreadably narrow when the panel shrinks. On mobile the rem-scaled floor
    // (e.g. 38rem in auction = 380px at the 10px root font) exceeds the viewport
    // content width (~371px), and `min-width` beats `width: 100%`, so the
    // container — and the whole chain inside it — gets pushed past the viewport
    // right edge. Clear the floor on mobile and let `width: 100%` constrain.
    widthContainer.style.minWidth = isMobileViewport() ? '' : computeContentMinWidth()
    table.style.minWidth = ''

    table.innerHTML = ''
    // A header rebuild throws the body away, so drop the virtual state too (a re-evaluate repopulates it).
    resetVirtualState()

    const isMobile = isMobileViewport()
    // Auction $ columns on mobile use a two-line header ("Your" then "$") so
    // they can stay narrow — see the .colheader-wrap-dollar rule in styles.css
    // and the className branch in the auction header loop below.
    const dollarColWidth = isMobile ? '1.25rem' : '2.5rem'
    const diffColWidth   = isMobile ? '1.25rem' : '2.5rem'

    const thead = table.createTHead()
    const headerRow = thead.insertRow()

    const playerTh = document.createElement('th')
    playerTh.className = 'tableheader'
    playerTh.textContent = 'Player'
    // Auction needs more horizontal room for 4 score columns + 9 categories,
    // so the player column gets less. Draft has only the H-Score column, so
    // the player column can be wider without squeezing the categories.
    playerTh.style.width = isMobile
        ? (isAuction ? '7rem' : '11rem')
        : '14rem'
    headerRow.append(playerTh)

    if (isAuction) {
        for (const label of ['Diff.', 'Your $', 'Gnrc. $', 'Orig. $']) {
            const th = document.createElement('th')
            // Dollar headers (Your $, Gnrc. $, Orig. $) get the wrap class so
            // mobile breaks the label across two lines ("Your" then "$"),
            // letting the column be narrower than a single-line header would
            // permit. Diff. doesn't have a $ so it stays single-line.
            th.className = label.includes('$') ? 'tableheader colheader-wrap-dollar' : 'tableheader'
            th.textContent = label
            th.style.width = label === 'Diff.' ? diffColWidth : dollarColWidth
            headerRow.append(th)
        }
    } else {
        const th = document.createElement('th')
        th.className = 'tableheader'
        th.textContent = 'H-Score'
        th.style.width = '2.5rem'
        headerRow.append(th)
    }

    // On mobile, swap each category's full name for its short form (e.g. "Points"
    // → "pts") so the headers stay narrow.
    const shortNames = isMobile ? getShortCategoryNames() : {}
    for (const category of categories) {
        const label = shortNames[category] ?? category
        const th = document.createElement('th')
        th.className = label.length >= 10 ? 'tableheader colheader-long' : 'tableheader'
        th.textContent = label
        headerRow.append(th)
    }
}

/** Builds one candidate's display + hidden-expand `<tr>` pair as an HTML string. */
function buildRowPairHtml(player: PlayerResult, ctx: RenderContext): string {
    let html = `<tr>`

    // Player display cell (headshot + name + positions) with expand button
    html += `<th class='playerheader'><div class='playerheaderdiv'>${buildFullPlayerDisplayHtml(player.player_id)}<button class='playerpopup'>▶</button></div></th>`

    // Score column(s)
    if (ctx.isAuction) {
        const av = player.auction_values
        if (av) {
            const diff = av.your_dollar - av.generic_dollar
            html += `<td class='auction-dollar' style='${stat_styler_secondary(diff, 10, 0)}'>${diff.toFixed(ctx.decimals)}</td>`
            for (const val of [av.your_dollar, av.generic_dollar, av.original_dollar]) {
                html += `<td class='auction-dollar celltypeb'>${val.toFixed(ctx.decimals)}</td>`
            }
        } else {
            for (let j = 0; j < 4; j++) html += `<td class='auction-dollar'>—</td>`
        }
    } else {
        // H-Score keeps 1-decimal precision even on mobile — it differentiates
        // closely-ranked players (e.g. 52.5% vs 52.3%) which integer rounding flattens.
        html += `<td class='overallhscore'>${player.h_score.toFixed(1)}</td>`
    }

    // Category win rate cells
    for (const value of player.win_rates) {
        if (ctx.rotoData) {
            const rotoValue = 1 + (value / 100) * (ctx.rotoData.nDrafters - 1)
            html += `<td class='categoricalRotoHscore' style='${stat_styler_primary(rotoValue, H_MULTIPLIER * (ctx.rotoData.nDrafters - 1), ctx.rotoData.rotoMiddle)}'>${rotoValue.toFixed(ctx.decimals)}</td>`
        } else {
            html += `<td class='categoricalhscore' style='${stat_styler_primary(value, H_MULTIPLIER, 50)}'>${value.toFixed(ctx.decimals)}</td>`
        }
    }

    html += `</tr>`
    // Expansion row (content built lazily on expand; only attached to the DOM while expanded + in-window)
    html += `<tr class='expandedview' style='display:none'></tr>`
    return html
}

/** Parses a batch of players into detached CandidateRow pairs with an expand listener bound to each row's
 *  header (closing over the CandidateRow, not an index — indices shift as batches merge in). */
function buildBatchRows(players: PlayerResult[], ctx: RenderContext): CandidateRow[] {
    let html = ''
    for (const player of players) html += buildRowPairHtml(player, ctx)

    // Parse in a scratch table (tables have strict child-parsing rules, so innerHTML
    // needs a real <tbody>). The parsed rows alternate display, expand, display, ...
    const scratch = document.createElement('table')
    scratch.innerHTML = `<tbody>${html}</tbody>`
    const parsed = Array.from(scratch.tBodies[0].rows) as HTMLTableRowElement[]

    const batch: CandidateRow[] = []
    for (let k = 0; k < players.length; k++) {
        const candidate: CandidateRow = {
            player:         players[k],
            displayRow:     parsed[2 * k],
            expandRow:      parsed[2 * k + 1],
            expanded:       false,
            expandedHeight: 0,
        }
        const div = candidate.displayRow.querySelector('.playerheaderdiv') as HTMLElement
        // Larger click target = the whole header div. Closes over the candidate, not an index.
        div.addEventListener('click', () => toggleExpand(candidate, ctx))
        batch.push(candidate)
    }
    return batch
}

/** Opens or closes one candidate's detail panel, then re-syncs the window. On expand we build the panel,
 *  render it (so it is in the DOM), then measure its height for the spacer maths and render once more. */
function toggleExpand(candidate: CandidateRow, ctx: RenderContext): void {
    const button = candidate.displayRow.querySelector('.playerpopup') as HTMLButtonElement
    if (candidate.expanded) {
        candidate.expanded       = false
        candidate.expandedHeight = 0
        candidate.expandRow.innerHTML = ''
        candidate.expandRow.style.display = 'none'
        button.classList.remove('popup-open')
        windowDirty = true
        renderWindow()
    } else {
        candidate.expanded = true
        candidate.expandRow.style.display = 'table-row'
        buildExpandPanel(candidate.expandRow, candidate.player, ctx.categories)
        button.classList.add('popup-open')
        windowDirty = true
        renderWindow()                                          // attach the (now-built) expand row
        candidate.expandedHeight = candidate.expandRow.offsetHeight  // measure once it is laid out
        windowDirty = true
        renderWindow()                                          // correct the spacers with the real height
    }
}

/** Merges an already-descending-sorted batch into `candidateRows` (data only — the DOM is synced by
 *  renderWindow). Fast path: the batch ranks entirely at/below the current bottom → append. */
function mergeBatchData(batch: CandidateRow[]): void {
    if (candidateRows.length === 0) { candidateRows = batch; return }
    const curMin = candidateRows[candidateRows.length - 1].player.h_score
    if (batch[0].player.h_score <= curMin) { for (const r of batch) candidateRows.push(r); return }

    const merged: CandidateRow[] = []
    let i = 0, j = 0
    while (i < candidateRows.length && j < batch.length) {
        if (candidateRows[i].player.h_score >= batch[j].player.h_score) merged.push(candidateRows[i++])
        else                                                            merged.push(batch[j++])
    }
    while (i < candidateRows.length) merged.push(candidateRows[i++])
    while (j < batch.length)         merged.push(batch[j++])
    candidateRows = merged
}

// ── Spacers ─────────────────────────────────────────────────────────────────
function makeSpacerRow(): HTMLTableRowElement {
    const tr = document.createElement('tr')
    tr.className = 'virtual-spacer'
    tr.setAttribute('aria-hidden', 'true')
    const td = document.createElement('td')
    td.colSpan = columnCount
    td.style.padding = '0'
    td.style.border  = '0'
    tr.appendChild(td)
    return tr
}
function setSpacerHeight(spacer: HTMLTableRowElement, px: number): void {
    (spacer.firstElementChild as HTMLElement).style.height = Math.max(0, px) + 'px'
}

// ── Windowed rendering ────────────────────────────────────────────────────────

/** Measures one collapsed row's height by briefly attaching the first display row. */
function ensureRowHeight(): void {
    if (rowHeight > 0 || candidateRows.length === 0 || !tbodyEl) return
    const probe = candidateRows[0].displayRow
    tbodyEl.appendChild(probe)
    rowHeight = probe.offsetHeight || 34   // fallback avoids a blank table if the row can't be measured
    tbodyEl.removeChild(probe)
}

/** Syncs the DOM to the current scroll window: attach only the rows overlapping the viewport (± buffer),
 *  with top/bottom spacer rows standing in for everything off-window (and the not-yet-streamed tail). */
function renderWindow(): void {
    if (!tbodyEl || candidateRows.length === 0) return
    ensureRowHeight()
    if (rowHeight === 0) return

    const n = candidateRows.length

    // Viewport, expressed relative to the top of the candidate rows. Uses only viewport-relative geometry
    // (getBoundingClientRect + innerHeight), so it is correct whichever element actually scrolls — here the
    // document scrolls, not #main-content. Read before mutating the DOM; the tbody's top doesn't move when
    // we swap its children (only its height does).
    const bodyTopVp = tbodyEl.getBoundingClientRect().top
    const viewTop   = -bodyTopVp
    const viewBot   = viewTop + window.innerHeight

    // Cumulative pixel offset of each row's top. Off[i] = height of rows [0, i). Cheap for a few hundred rows.
    const off = new Array<number>(n + 1)
    off[0] = 0
    for (let i = 0; i < n; i++) off[i + 1] = off[i] + rowHeight + candidateRows[i].expandedHeight

    // First row whose bottom is below the viewport top; last whose top is above the viewport bottom.
    let start = n
    for (let i = 0; i < n; i++) { if (off[i + 1] > viewTop) { start = i; break } }
    let end = start
    for (let i = start; i < n; i++) { if (off[i] < viewBot) end = i + 1; else break }
    start = Math.max(0, start - WINDOW_BUFFER)
    end   = Math.min(n, end + WINDOW_BUFFER)

    if (start === renderedStart && end === renderedEnd && !windowDirty) return

    const reserveTail = totalCandidates > n ? (totalCandidates - n) * rowHeight : 0
    if (topSpacer === null)    topSpacer    = makeSpacerRow()
    if (bottomSpacer === null) bottomSpacer = makeSpacerRow()
    setSpacerHeight(topSpacer, off[start])
    setSpacerHeight(bottomSpacer, (off[n] - off[end]) + reserveTail)

    tbodyEl.replaceChildren()
    tbodyEl.appendChild(topSpacer)
    for (let i = start; i < end; i++) {
        const candidate = candidateRows[i]
        const headerCell = candidate.displayRow.firstElementChild as HTMLElement | null
        headerCell?.classList.toggle('waiver-drop-highlight',
                                     highlightedPlayerId !== null && candidate.player.player_id === highlightedPlayerId)
        tbodyEl.appendChild(candidate.displayRow)
        if (candidate.expanded) tbodyEl.appendChild(candidate.expandRow)
    }
    tbodyEl.appendChild(bottomSpacer)

    renderedStart = start
    renderedEnd   = end
    windowDirty   = false
}

/** Scrolls the document so candidate `idx` sits ~30% down the viewport (works with the document scroller). */
function scrollCandidateIntoView(idx: number): void {
    ensureRowHeight()
    if (rowHeight === 0 || !tbodyEl) return
    let off = 0
    for (let i = 0; i < idx; i++) off += rowHeight + candidateRows[i].expandedHeight
    const bodyTopDoc = tbodyEl.getBoundingClientRect().top + window.scrollY
    window.scrollTo({ top: Math.max(0, bodyTopDoc + off - window.innerHeight * 0.3), behavior: 'auto' })
}

/** Marks one candidate (by player id) with the waiver-drop highlight, or clears it when null. The mark
 *  is stored, not written to a fixed DOM row, so it follows the row as the window scrolls. If found, the
 *  candidate is scrolled into view. Safe when the id is absent — nothing is highlighted. */
export function highlightCandidate(playerId: number | null): void {
    highlightedPlayerId = playerId
    if (playerId !== null) {
        const idx = candidateRows.findIndex(candidate => candidate.player.player_id === playerId)
        if (idx >= 0) scrollCandidateIntoView(idx)
    }
    windowDirty = true
    renderWindow()
}

/** Wires the scroll + resize listeners once. Scroll is rAF-throttled so we re-window at most once a frame. */
function ensureScrollWired(): void {
    if (scrollWired) return
    const onScroll = () => {
        if (scrollScheduled) return
        scrollScheduled = true
        requestAnimationFrame(() => { scrollScheduled = false; renderWindow() })
    }
    // Capture phase so we catch scrolls from whichever element scrolls (the document here, or a nested
    // overflow container) — scroll events fire on the scroller and do not bubble.
    document.addEventListener('scroll', onScroll, { passive: true, capture: true })
    window.addEventListener('resize', () => { windowDirty = true; renderWindow() })
    scrollWired = true
}

// ── Streaming / tail reservation (public API kept for callers) ─────────────────

/** Reserves scroll height for `totalCandidates - (already merged)` not-yet-streamed candidates, so the
 *  scrollbar doesn't grow as later batches merge in. Realised as extra height on the bottom spacer. */
export function reserveTailSpace(total: number): void {
    totalCandidates = Math.max(total, candidateRows.length)
    windowDirty = true
    renderWindow()
}

/** Drops the reserved tail once every batch has arrived (or the table is rebuilt). */
export function clearTailSpace(): void {
    totalCandidates = candidateRows.length
    windowDirty = true
    renderWindow()
}

/** Clears the candidate table (header + empty body) and captures the render context,
 *  ready for one or more addBatch() calls. */
export function resetTable(): void {
    buildTableHeader()          // wipes the table and resets virtual state
    tbodyEl = table.createTBody()
    ensureScrollWired()

    const isRoto = getScoringFormat() === 'Rotisserie'
    renderCtx = {
        categories: getSelectedCategories(),
        isAuction:  (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode',
        isRoto,
        // Mobile shows integer-rounded values so columns stay narrow ("47" vs "47.5");
        // desktop keeps 1-decimal precision. H-Score always keeps 1 decimal (see buildRowPairHtml).
        decimals:   isMobileViewport() ? 0 : 1,
        rotoData:   isRoto
            ? (() => { const nDrafters = getLeagueSettings().n_drafters
                       return { nDrafters, rotoMiddle: (nDrafters - 1) / 2 + 1 } })()
            : null,
    }
    columnCount = 1 + (renderCtx.isAuction ? 4 : 1) + renderCtx.categories.length
}

/** Merges one batch of already-H-score-sorted players into the current table.
 *  Call resetTable() first. */
export function addBatch(players: PlayerResult[]): void {
    if (players.length === 0 || renderCtx === null || tbodyEl === null) return
    mergeBatchData(buildBatchRows(players, renderCtx))
    if (totalCandidates < candidateRows.length) totalCandidates = candidateRows.length
    windowDirty = true
    renderWindow()
}

/** Rebuilds the whole candidate table from a single, complete player list.
 *  (Auction and any non-batched caller use this.) */
export function buildTable(players: PlayerResult[]): void {
    resetTable()
    addBatch(players)
}
