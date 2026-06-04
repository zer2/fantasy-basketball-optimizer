// table/player_table.ts
// Renders the H-score candidate table: headers, player rows, expand buttons.
// Reads current player and category state from app_state.ts.

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

/** Clears the candidate table and shows a single centred message row. */
export function showTableMessage(message: string): void {
    table.innerHTML = ''
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
    // Auction $ columns are quite narrow on mobile (since "Your $"/"Gnrc. $" headers
    // truncate anyway), but on desktop they need room for the full headers.
    const dollarColWidth = isMobile ? '1.5rem'  : '2.5rem'
    const diffColWidth   = isMobile ? '1.25rem' : '2.5rem'

    const thead = table.createTHead()
    const headerRow = thead.insertRow()

    const playerTh = document.createElement('th')
    playerTh.className = 'tableheader'
    playerTh.textContent = 'Player'
    playerTh.style.width = isMobile ? '11rem' : '14rem'
    headerRow.append(playerTh)

    if (isAuction) {
        for (const label of ['Diff.', 'Your $', 'Gnrc. $', 'Orig. $']) {
            const th = document.createElement('th')
            th.className = 'tableheader'
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


/** Rebuilds the H-score candidate table from scratch: clears old rows, creates headers, and populates player rows with styled cells. */
export function buildTable(players: PlayerResult[]): void {

    buildTableHeader()

    const categories = getSelectedCategories()
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'
    const isRoto     = getScoringFormat() === 'Rotisserie'

    // Mobile shows integer-rounded values so columns can stay narrow ("47" vs "47.5").
    // Desktop keeps 1-decimal precision since there's room. H-Score keeps 1-decimal
    // on both viewports (see below) — it differentiates closely-ranked players.
    // Uses the same 768px breakpoint as the smaller root font in styles.css.
    const decimals = isMobileViewport() ? 0 : 1

    // ── Player rows (built as HTML string for performance) ───────────────────
    const rotoData = isRoto
        ? (() => {
              const nDrafters = getLeagueSettings().n_drafters
              return { nDrafters, rotoMiddle: (nDrafters - 1) / 2 + 1 }
          })()
        : null

    let html = ''
    for (const [i, player] of players.entries()) {
        html += `<tr>`

        // Player name cell with expand button
        html += `<th class='playerheader'><div class='playerheaderdiv'><span class='playername'>${player.name}</span><button class='playerpopup'>▶</button></div></th>`

        // Score column(s)
        if (isAuction) {
            const av = player.auction_values
            if (av) {
                const diff = av.your_dollar - av.gnrc_dollar
                html += `<td class='auction-dollar' style='${stat_styler_secondary(diff, 10, 0)}'>${diff.toFixed(decimals)}</td>`
                for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
                    html += `<td class='auction-dollar celltypeb'>${val.toFixed(decimals)}</td>`
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
            if (rotoData) {
                const rotoValue = 1 + (value / 100) * (rotoData.nDrafters - 1)
                html += `<td class='categoricalRotoHscore' style='${stat_styler_primary(rotoValue, 3 * (rotoData.nDrafters - 1), rotoData.rotoMiddle)}'>${rotoValue.toFixed(decimals)}</td>`
            } else {
                html += `<td class='categoricalhscore' style='${stat_styler_primary(value, 3, 50)}'>${value.toFixed(decimals)}</td>`
            }
        }

        html += `</tr>`

        // Expansion row (hidden until button clicked)
        html += `<tr class='expandedview EV${i}' style='display:none'></tr>`
    }

    const tbody = table.createTBody()
    tbody.innerHTML = html

    // Attach expand listeners to the header div (larger click target) — cheap second pass
    const expandButtons  = Array.from(tbody.querySelectorAll<HTMLButtonElement>('.playerpopup'))
    const expandDivs     = Array.from(tbody.querySelectorAll<HTMLDivElement>('.playerheaderdiv'))
    const expandedRows   = Array.from(tbody.querySelectorAll<HTMLTableRowElement>('tr.expandedview'))
    for (const [i, player] of players.entries()) {
        expandDivs[i].addEventListener('click', () => toggleExpandView(expandButtons[i], expandedRows[i], player, categories))
    }
}
