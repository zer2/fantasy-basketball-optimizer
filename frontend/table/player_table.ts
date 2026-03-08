// table/player_table.ts
// Renders the H-score candidate table: headers, player rows, expand buttons.
// Reads current player and category state from app_state.ts.

import { stat_styler_primary, stat_styler_secondary } from '../styles/styler_functions.js'
import { ExpandView } from './expand_view.js'
import { Player } from '../types.js'
import { getCategories } from '../app_state.js'
import { getFormatAndCategories } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'

const table = document.getElementById('realtable') as HTMLTableElement

/** Rebuilds the H-score candidate table from scratch: clears old rows, creates headers, and populates player rows with styled cells. */
export function buildTable(players: Player[]): void {
    const categories = getCategories()
    const isAuction  = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'
    const isRoto     = getFormatAndCategories().scoring_format === 'Rotisserie'

    // 1 (player) + score cols + N categories
    const PLAYER_COL_W = 224
    const SCORE_COL_W  = 72
    const scoreCols    = isAuction ? 4 : 1  // Diff / Your $ / Gnrc. $ / Orig. $  vs  H-Score

    // Keep category column widths consistent across modes by widening the table
    // to compensate for the extra auction columns (otherwise they'd wrap).
    const CAT_COL_W    = 90   // desired minimum width per category column
    const tableWidth   = PLAYER_COL_W + scoreCols * SCORE_COL_W + categories.length * CAT_COL_W
    table.style.width  = tableWidth + 'px'

    const totalCols = 1 + scoreCols + categories.length

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

    // ── Player rows ─────────────────────────────────────────────────────────
    for (const [i, player] of players.entries()) {
        const row = table.insertRow(-1)

        // Player name cell with expand button
        const nameCell = document.createElement('th')
        nameCell.innerHTML = `
            <div class='playerheaderdiv'>
                <div style="width:80%">${player.name}</div>
                <div style="width:20%">
                    <button class='playerpopup' id='PP${i}'>▶</button>
                </div>
            </div>`
        nameCell.className = 'playerheader'
        row.append(nameCell)

        const button = nameCell.querySelector(`#PP${i}.playerpopup`) as HTMLButtonElement
        button.addEventListener('click', () => ExpandView(i, player, categories, totalCols))

        // Score column(s)
        if (isAuction) {
            const av = player.auction_values
            if (av) {
                const diff = av.your_dollar - av.gnrc_dollar

                const diffCell = row.insertCell(-1)
                diffCell.textContent = diff.toFixed(1)
                diffCell.style.cssText = stat_styler_secondary(diff, 10, 0)
                diffCell.className = 'auction-dollar'

                for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
                    const cell = row.insertCell(-1)
                    cell.textContent = String(val.toFixed(1))
                    cell.className = 'auction-dollar celltypea'
                }
            } else {
                // Auction values not yet available (e.g. stale results from draft mode).
                // Insert empty cells to keep column count correct; runEvaluate will refresh.
                for (let i = 0; i < 4; i++) {
                    const cell = row.insertCell(-1)
                    cell.textContent = '—'
                    cell.className = 'auction-dollar'
                }
            }
        } else {
            const hscoreCell = row.insertCell(-1)
            hscoreCell.className = 'overallhscore'
            hscoreCell.textContent = player.h_score.toFixed(1)
        }

        // Category win rate cells
        for (const value of player.win_rates) {
            const cell = row.insertCell(-1)
            if (isRoto) {
                const n = getLeagueSettings().n_drafters
                const rotoValue = 1 + (value / 100) * (n - 1)
                const rotoMiddle = (n - 1) / 2 + 1
                cell.textContent = rotoValue.toFixed(1)
                cell.style.cssText = stat_styler_primary(rotoValue, 3 * (n - 1), rotoMiddle)
                cell.className = 'categoricalRotoHscore'
            } else {
                cell.textContent = value.toFixed(1)
                cell.style.cssText = stat_styler_primary(value, 3, 50)
                cell.className = 'categoricalhscore'
            }
        }

        // Expansion row (hidden until button clicked)
        const expandedRow = table.insertRow(-1)
        expandedRow.className = `expandedview EV${i}`
        expandedRow.style.display = 'none'
    }
}
