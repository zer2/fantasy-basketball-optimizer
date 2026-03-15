// table/gscore_table.ts
// Renders a G-score breakdown table for a list of players.
// Used by Auction Mode's "My Team" tab and potentially by Season Mode roster inspection.

import { getGScoreByName, getCategories } from '../app_state.js'
import { stat_styler_primary } from '../styles/styler_functions.js'

/**
 * Builds a G-score table for the given players: one row per player plus a
 * team totals row.  Styled to match the season roster inspector tables.
 * Clears and replaces `container` contents on each call.
 */
export function renderTeamGScoreTable(playerNames: string[], container: HTMLElement, width?: string): void {
    container.innerHTML = ''
    const categories = getCategories()
    const gScoreMap  = getGScoreByName()

    // Collect G-scores for each player
    const rows: { name: string; values: number[]; total: number }[] = []
    for (const name of playerNames) {
        const gs = gScoreMap.get(name)
        if (!gs) continue
        rows.push({ name: gs.name, values: gs.values, total: gs.total })
    }

    if (rows.length === 0) return

    // ── Build table ──────────────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table panel-table--rounded'
    tbl.style.tableLayout = 'fixed'
    if (width) tbl.style.width = width

    // Spacer row to lock column widths
    const tHead = tbl.createTHead()
    const spacerRow = tHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh())

    // Header row
    const headerRow = tHead.insertRow(-1)
    headerRow.appendChild(makeSpacerTh())  // invisible label spacer
    const totalTh = document.createElement('th')
    totalTh.className = 'panel-colheader'
    totalTh.textContent = 'Total'
    headerRow.appendChild(totalTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.className = 'panel-colheader'
        th.textContent = cat
        headerRow.appendChild(th)
    }

    // Data rows — one per player
    const tBody = tbl.createTBody()
    const teamTotals = new Array(categories.length).fill(0)
    let teamTotalSum = 0

    for (const row of rows) {
        const tr = tBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.textContent = row.name
        tr.appendChild(labelCell)

        const totalCell = tr.insertCell(-1)
        totalCell.textContent = row.total.toFixed(2)
        totalCell.className = 'panel-datacell celltypea'
        teamTotalSum += row.total

        for (let i = 0; i < row.values.length; i++) {
            const cell = tr.insertCell(-1)
            cell.textContent = row.values[i].toFixed(2)
            cell.style.cssText = stat_styler_primary(row.values[i], 60, 0)
            cell.className = 'panel-datacell'
            teamTotals[i] += row.values[i]
        }
    }

    // Totals row
    const totalsRow = tBody.insertRow(-1)
    const totalsLabel = document.createElement('th')
    totalsLabel.className = 'panel-rowlabel'
    totalsLabel.textContent = 'Team Total'
    totalsRow.appendChild(totalsLabel)

    const totalsCell = totalsRow.insertCell(-1)
    totalsCell.textContent = teamTotalSum.toFixed(2)
    totalsCell.className = 'panel-datacell celltypeb'

    for (const val of teamTotals) {
        const cell = totalsRow.insertCell(-1)
        cell.textContent = val.toFixed(2)
        cell.style.cssText = stat_styler_primary(val, 60, 0)
        cell.className = 'panel-datacell'
    }

    container.appendChild(tbl)
}

/** Creates an invisible `<th>` spacer to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const th = document.createElement('th')
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return th
}
