// table/player_table.ts
// Renders the H-score candidate table: headers, player rows, expand buttons.
// Reads current player and category state from app_state.ts.

import { stat_styler_primary, stat_styler_secondary } from '../styles/styler_functions.js'
import { toggleExpandView } from './expand_view.js'
import { Player } from '../types.js'
import { getFormatAndCategories } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'

const table = document.getElementById('hscoretable') as HTMLTableElement

// ── Column widths (shared between buildTable and computeHscoreTableWidth) ────

const PLAYER_COL_W = 224
const SCORE_COL_W  = 72
const CAT_COL_W    = 90   // desired minimum width per category column

/**
 * Computes the H-score table width from current sidebar state.
 * Uses backend-confirmed categories if available, otherwise falls back to the
 * sidebar selection — so the result is correct both before and after the first
 * evaluate response.
 */
export function computeHscoreTableWidth(): string {
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'
    const nScoreCols = isAuction ? 4 : 1
    const categories = getFormatAndCategories().categories
    return (PLAYER_COL_W + nScoreCols * SCORE_COL_W + categories.length * CAT_COL_W) + 'px'
}

/** Rebuilds the H-score candidate table from scratch: clears old rows, creates headers, and populates player rows with styled cells. */
export function buildTable(players: Player[]): void {
    const categories = getFormatAndCategories().categories
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'
    const isRoto     = getFormatAndCategories().scoring_format === 'Rotisserie'

    table.style.width = computeHscoreTableWidth()

    table.innerHTML = ''

    // ── Header ──────────────────────────────────────────────────────────────
    const thead = table.createTHead()
    const headerRow = thead.insertRow()

    const playerTh = document.createElement('th')
    playerTh.className = 'tableheader'
    playerTh.textContent = 'Player'
    playerTh.style.width = '224px'
    headerRow.append(playerTh)

    if (isAuction) {
        for (const label of ['Diff.', 'Your $', 'Gnrc. $', 'Orig. $']) {
            const th = document.createElement('th')
            th.className = 'tableheader'
            th.textContent = label
            th.style.width = '72px'
            headerRow.append(th)
        }
    } else {
        const th = document.createElement('th')
        th.className = 'tableheader'
        th.textContent = 'H-Score'
        th.style.width = '72px'
        headerRow.append(th)
    }

    for (const category of categories) {
        const th = document.createElement('th')
        th.className = 'tableheader'
        th.textContent = category
        headerRow.append(th)
    }

    // ── Player rows (built as HTML string for performance) ───────────────────
    // Roto values hoisted outside the loop to avoid repeated lookups.
    let nDrafters  = 0
    let rotoMiddle = 0
    if (isRoto) {
        nDrafters  = getLeagueSettings().n_drafters
        rotoMiddle = (nDrafters - 1) / 2 + 1
    }

    let html = ''
    for (const [i, player] of players.entries()) {
        html += `<tr>`

        // Player name cell with expand button
        html += `<th class='playerheader'><div class='playerheaderdiv'><div style="width:80%">${player.name}</div><div style="width:20%"><button class='playerpopup'>▶</button></div></div></th>`

        // Score column(s)
        if (isAuction) {
            const av = player.auction_values
            if (av) {
                const diff = av.your_dollar - av.gnrc_dollar
                html += `<td class='auction-dollar' style='${stat_styler_secondary(diff, 10, 0)}'>${diff.toFixed(1)}</td>`
                for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
                    html += `<td class='auction-dollar celltypeb'>${val.toFixed(1)}</td>`
                }
            } else {
                for (let j = 0; j < 4; j++) html += `<td class='auction-dollar'>—</td>`
            }
        } else {
            html += `<td class='overallhscore'>${player.h_score.toFixed(1)}</td>`
        }

        // Category win rate cells
        for (const value of player.win_rates) {
            if (isRoto) {
                const rotoValue = 1 + (value / 100) * (nDrafters - 1)
                html += `<td class='categoricalRotoHscore' style='${stat_styler_primary(rotoValue, 3 * (nDrafters - 1), rotoMiddle)}'>${rotoValue.toFixed(1)}</td>`
            } else {
                html += `<td class='categoricalhscore' style='${stat_styler_primary(value, 3, 50)}'>${value.toFixed(1)}</td>`
            }
        }

        html += `</tr>`

        // Expansion row (hidden until button clicked)
        html += `<tr class='expandedview EV${i}' style='display:none'></tr>`
    }

    const tbody = table.createTBody()
    tbody.innerHTML = html

    // Attach expand button listeners (cheap second pass — no DOM mutations)
    const expandButtons  = Array.from(tbody.querySelectorAll<HTMLButtonElement>('.playerpopup'))
    const expandedRows   = Array.from(tbody.querySelectorAll<HTMLTableRowElement>('tr.expandedview'))
    for (const [i, player] of players.entries()) {
        expandButtons[i].addEventListener('click', () => toggleExpandView(expandButtons[i], expandedRows[i], player, categories))
    }
}
