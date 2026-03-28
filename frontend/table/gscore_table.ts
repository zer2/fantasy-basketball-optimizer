// table/gscore_table.ts
// Renders a G-score breakdown table for a list of players.
// Used by Auction Mode's "My Team" tab and potentially by Season Mode roster inspection.

import { getGScoreByName } from '../app_state.js'
import { getSelectedCategories, getScoringFormat } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { stat_styler_primary } from '../styles/styler_functions.js'

/**
 * Builds a G-score table for the given players: one row per player plus a
 * team totals row.  Styled to match the season roster inspector tables.
 * Clears and replaces `container` contents on each call.
 */
export function renderTeamGScoreTable(
    playerNames: string[]
  , container: HTMLElement
  , fullTeamResult?: { h_score: number; win_rates: number[] } | null
): void {
    container.innerHTML = ''
    const categories = getSelectedCategories()
    const gScoreMap  = getGScoreByName()
    // Collect G-scores for each player
    const rows: { name: string; values: number[]; total: number }[] = []
    if (playerNames.length === 0) return

    for (const name of playerNames) {
        const gs = gScoreMap.get(name)
        if (!gs) throw new Error(`G-score not found for player: ${name}`)
        rows.push({ name: gs.name, values: gs.values, total: gs.total })
    }

    // ── Build table ──────────────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table panel-table--rounded'
    tbl.style.tableLayout = 'fixed'
    tbl.style.width = '100%'

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
        th.className = cat.length >= 10 ? 'panel-colheader colheader-long' : 'panel-colheader'
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

    if (!fullTeamResult) return

    // ── H-score table (separate, columns aligned to the G-score table above) ──

    const isRoto    = getScoringFormat() === 'Rotisserie'
    const nDrafters = isRoto ? getLeagueSettings().n_drafters : 0
    const rotoMiddle = (nDrafters - 1) / 2 + 1

    const hScoreTbl = document.createElement('table')
    hScoreTbl.className = 'panel-table panel-table--rounded panel-table--top-gap'
    hScoreTbl.style.tableLayout = 'fixed'
    hScoreTbl.style.width = '100%'

    // colgroup sets column widths to match the G-score table without a spacer row,
    // so there is no gap between the outer border and the first data row.
    const colgroup = document.createElement('colgroup')
    const nameCol  = document.createElement('col')
    nameCol.style.width = '200px'   // matches .panel-colspacer-name
    const totalCol = document.createElement('col')
    totalCol.style.width = '83px'   // matches .panel-colspacer-total
    colgroup.append(nameCol, totalCol)
    for (let i = 0; i < categories.length; i++) colgroup.appendChild(document.createElement('col'))
    hScoreTbl.appendChild(colgroup)

    const hScoreTBody = hScoreTbl.createTBody()
    const hScoreRow = hScoreTBody.insertRow(-1)

    const hScoreLabel = document.createElement('th')
    hScoreLabel.className = 'panel-rowlabel'
    hScoreLabel.textContent = 'H-Score (est. win rate)'
    hScoreRow.appendChild(hScoreLabel)

    const hScoreTotalCell = hScoreRow.insertCell(-1)
    hScoreTotalCell.textContent = fullTeamResult.h_score.toFixed(1)
    hScoreTotalCell.className = 'overallhscore'

    for (const winRate of fullTeamResult.win_rates) {
        const cell = hScoreRow.insertCell(-1)
        if (isRoto) {
            const rotoValue = 1 + (winRate / 100) * (nDrafters - 1)
            cell.textContent = rotoValue.toFixed(1)
            cell.className = 'categoricalRotoHscore'
            cell.style.cssText = stat_styler_primary(rotoValue, 3 * (nDrafters - 1), rotoMiddle)
        } else {
            cell.textContent = winRate.toFixed(1)
            cell.className = 'categoricalhscore'
            cell.style.cssText = stat_styler_primary(winRate, 3, 50)
        }
    }

    container.appendChild(hScoreTbl)
}

/** Creates an invisible `<th>` spacer to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const th = document.createElement('th')
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return th
}
