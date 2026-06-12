// table/gscore_table.ts
// Renders a G-score breakdown table for a list of players.
// Used by Auction Mode's "My Team" tab and potentially by Season Mode roster inspection.

import { getGScoreByName, getShortCategoryNames } from '../app_state.js'
import { getSelectedCategories, getScoringFormat } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { stat_styler_primary } from '../styles/styler_functions.js'
import { isMobileViewport } from '../helper_functions.js'
import { makeSpacerTh } from './table_helpers.js'

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

    // On mobile, narrow the name/total columns and use short category labels
    // (e.g. "Points" → "Pts") so the table's intrinsic min-content fits inside
    // the candidate panel rather than stretching it. Matches the equivalent
    // mobile treatment in season_rosters.ts's team inspector.
    const isMobile   = isMobileViewport()
    const shortNames = isMobile ? getShortCategoryNames() : {}

    // colgroup locks column widths without introducing a visual spacer row.
    tbl.appendChild(makeNameTotalCategoriesColgroup(categories.length, isMobile))

    // Header row
    const headerRow = tbl.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh())  // invisible label spacer
    const totalTh = document.createElement('th')
    totalTh.className = 'panel-colheader'
    totalTh.textContent = 'Total'
    headerRow.appendChild(totalTh)
    for (const cat of categories) {
        const label = shortNames[cat] ?? cat
        const th = document.createElement('th')
        th.className = label.length >= 10 ? 'panel-colheader colheader-long' : 'panel-colheader'
        th.textContent = label
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

        for (let i = 0; i < categories.length; i++) {
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

    hScoreTbl.appendChild(makeNameTotalCategoriesColgroup(categories.length, isMobile))

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

/** Creates a `<colgroup>` with name and total columns followed by one
 *  auto-sized column per category. Desktop uses 200px/83px (matches
 *  .panel-colspacer-name / .panel-colspacer-total); mobile uses 7rem/3rem so
 *  the table fits inside the panel without forcing it wider. */
function makeNameTotalCategoriesColgroup(
    nCategories: number
  , isMobile: boolean
): HTMLTableColElement {
    const colgroup = document.createElement('colgroup')
    const nameCol  = document.createElement('col')
    nameCol.style.width = isMobile ? '7rem' : '200px'
    const totalCol = document.createElement('col')
    totalCol.style.width = isMobile ? '3rem' : '83px'
    colgroup.append(nameCol, totalCol)
    for (let i = 0; i < nCategories; i++) colgroup.appendChild(document.createElement('col'))
    return colgroup
}
