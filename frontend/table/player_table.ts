// table/player_table.ts
// Renders the H-score candidate table: headers, player rows, expand buttons.
// Reads current player and category state from app_state.ts.
//
// The table is built incrementally so that draft/waiver evaluations can stream in
// batches (top-ranked players first): resetTable() clears it, then each addBatch()
// call merges a batch of already-H-score-sorted players into place. Because the
// virtual list (candidateRows) and the DOM mirror each other in descending H-score
// order, a batch is merged, not repainted — its tail (which ranks below everything
// already shown) is appended at the bottom, and only the few "crossing" players that
// outrank the current bottom are inserted higher up. Existing rows never move.
// buildTable() is the single-shot entry point (one batch = everything, e.g. auction).

import { stat_styler_primary, stat_styler_secondary } from '../styles/styler_functions.js'
import { toggleExpandView } from './expand_view.js'
import { PlayerResult } from '../types.js'
import { getScoringFormat, getSelectedCategories } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getShortCategoryNames } from '../app_state.js'
import { isMobileViewport } from '../helper_functions.js'

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

// ── Incremental render state ──────────────────────────────────────────────────
// A rendered candidate: its player data plus the two DOM rows (visible + hidden expand).
interface CandidateRow {
    player:     PlayerResult
    displayRow: HTMLTableRowElement
    expandRow:  HTMLTableRowElement
}

// The virtual "dataframe": candidate rows in descending H-score order, mirroring the DOM order.
let candidateRows: CandidateRow[] = []
let tbodyEl: HTMLTableSectionElement | null = null

// Per-render context, captured once in resetTable so every addBatch renders rows consistently.
interface RenderContext {
    categories: string[]
    isAuction:  boolean
    isRoto:     boolean
    rotoData:   { nDrafters: number; rotoMiddle: number } | null
    decimals:   number
}
let renderCtx: RenderContext | null = null

/** Clears the candidate table and shows a single centred message row. */
export function showTableMessage(message: string): void {
    clearTailSpace()
    table.innerHTML = ''
    candidateRows = []
    tbodyEl = null
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

    // Player name cell with expand button
    html += `<th class='playerheader'><div class='playerheaderdiv'><span class='playername'>${player.name}</span><button class='playerpopup'>▶</button></div></th>`

    // Score column(s)
    if (ctx.isAuction) {
        const av = player.auction_values
        if (av) {
            const diff = av.your_dollar - av.gnrc_dollar
            html += `<td class='auction-dollar' style='${stat_styler_secondary(diff, 10, 0)}'>${diff.toFixed(ctx.decimals)}</td>`
            for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
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
            html += `<td class='categoricalRotoHscore' style='${stat_styler_primary(rotoValue, 3 * (ctx.rotoData.nDrafters - 1), ctx.rotoData.rotoMiddle)}'>${rotoValue.toFixed(ctx.decimals)}</td>`
        } else {
            html += `<td class='categoricalhscore' style='${stat_styler_primary(value, 3, 50)}'>${value.toFixed(ctx.decimals)}</td>`
        }
    }

    html += `</tr>`
    // Expansion row (hidden until the header is clicked; content built lazily on expand)
    html += `<tr class='expandedview' style='display:none'></tr>`
    return html
}

/** Parses a batch of players into detached CandidateRow pairs with expand listeners bound
 *  to the rows directly (not by index — indices shift as batches merge in). */
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
        const displayRow = parsed[2 * k]
        const expandRow  = parsed[2 * k + 1]
        const player     = players[k]
        const div = displayRow.querySelector('.playerheaderdiv') as HTMLElement
        const btn = displayRow.querySelector('.playerpopup')     as HTMLButtonElement
        // Larger click target = the whole header div. Closes over the row refs, not an index.
        div.addEventListener('click', () => toggleExpandView(btn, expandRow, player, ctx.categories))
        batch.push({ player, displayRow, expandRow })
    }
    return batch
}

/** Merges an already-descending-sorted batch into candidateRows + the DOM.
 *  Fast path (common): the batch ranks entirely at/below the current bottom → append it.
 *  Otherwise: a two-pointer merge that only ever inserts the batch's rows before an
 *  existing anchor row; existing rows are never moved. */
function mergeBatch(batch: CandidateRow[]): void {
    const tbody = tbodyEl!

    const curMin = candidateRows.length ? candidateRows[candidateRows.length - 1].player.h_score : Infinity
    if (candidateRows.length === 0 || batch[0].player.h_score <= curMin) {
        const frag = document.createDocumentFragment()
        for (const r of batch) {
            frag.appendChild(r.displayRow)
            frag.appendChild(r.expandRow)
            candidateRows.push(r)
        }
        tbody.appendChild(frag)
        return
    }

    // General merge: walk both descending lists; insert batch rows before the first
    // existing row they outrank. The batch's non-crossing tail falls through to append.
    const merged: CandidateRow[] = []
    let i = 0, j = 0
    while (i < candidateRows.length && j < batch.length) {
        if (candidateRows[i].player.h_score >= batch[j].player.h_score) {
            merged.push(candidateRows[i++])
        } else {
            const anchor = candidateRows[i].displayRow
            tbody.insertBefore(batch[j].displayRow, anchor)
            tbody.insertBefore(batch[j].expandRow,  anchor)
            merged.push(batch[j++])
        }
    }
    while (i < candidateRows.length) merged.push(candidateRows[i++])
    if (j < batch.length) {
        const frag = document.createDocumentFragment()
        for (; j < batch.length; j++) {
            frag.appendChild(batch[j].displayRow)
            frag.appendChild(batch[j].expandRow)
            merged.push(batch[j])
        }
        tbody.appendChild(frag)
    }
    candidateRows = merged
}

// ── Tail-space reservation ────────────────────────────────────────────────────
// When batches stream in, the page's scroll height grows as each batch appends rows,
// which shifts the scrollbar under the user. To avoid that, once the first batch is
// painted we reserve whitespace below the table for the candidates still to come, so
// the scroll height reaches its final value immediately; each later batch shrinks the
// reserved space by exactly what it fills, keeping the scrollbar still.
let tailSpacer: HTMLDivElement | null = null
let measuredRowHeight = 0

/** Reserves whitespace below the table for `totalCandidates - (already rendered)` rows, so the
 *  scroll height doesn't grow as later batches merge in. Safe to call after every batch — it
 *  shrinks the reserved height as rows arrive. Row height is measured once from a rendered row. */
export function reserveTailSpace(totalCandidates: number): void {
    if (candidateRows.length === 0) return
    if (measuredRowHeight === 0) measuredRowHeight = candidateRows[0].displayRow.offsetHeight
    const remaining = Math.max(0, totalCandidates - candidateRows.length)
    if (tailSpacer === null) {
        tailSpacer = document.createElement('div')
        tailSpacer.id = 'candidate-tail-spacer'
        table.insertAdjacentElement('afterend', tailSpacer)
    }
    tailSpacer.style.height = (remaining * measuredRowHeight) + 'px'
}

/** Drops the reserved whitespace once every batch has arrived (or the table is rebuilt). */
export function clearTailSpace(): void {
    if (tailSpacer) { tailSpacer.remove(); tailSpacer = null }
    measuredRowHeight = 0
}

/** Clears the candidate table (header + empty body) and captures the render context,
 *  ready for one or more addBatch() calls. */
export function resetTable(): void {
    clearTailSpace()
    buildTableHeader()
    candidateRows = []
    tbodyEl = table.createTBody()

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
}

/** Merges one batch of already-H-score-sorted players into the current table.
 *  Call resetTable() first. */
export function addBatch(players: PlayerResult[]): void {
    if (players.length === 0 || renderCtx === null || tbodyEl === null) return
    mergeBatch(buildBatchRows(players, renderCtx))
}

/** Rebuilds the whole candidate table from a single, complete player list.
 *  (Auction and any non-batched caller use this.) */
export function buildTable(players: PlayerResult[]): void {
    resetTable()
    addBatch(players)
}
