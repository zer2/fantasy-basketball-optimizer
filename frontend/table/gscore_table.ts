// table/gscore_table.ts
// Renders a G-score breakdown table for a list of players.
// Used by Draft/Auction Mode's team statistics and by Season Mode's roster inspection: the
// exported builders below are the single home for the panel's header, body, totals, and
// H-score row, so the two panels cannot drift apart cell by cell. Each caller keeps its own
// table shell — the shells genuinely differ (rounded corners, and colgroup-locked widths
// here vs. the inspector's CSS-class spacer row).

import { getGScoreById, getShortCategoryNames } from '../app_state.js'
import { makeFullPlayerDisplay } from '../player_display.js'
import { getSelectedCategories, getScoringFormat } from '../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import {
    stat_styler_primary, G_SCORE_MULTIPLIER, H_MULTIPLIER,
    convertWinRateToRotoPoints, computeRotoMiddle,
} from '../styles/styler_functions.js'
import { isMobileViewport } from '../helper_functions.js'
import { makeSpacerTh } from './table_helpers.js'

export interface GScoreRowData { playerId: number; values: number[]; total: number }

/** The stored G-score row for a player id; a missing entry is a programmer error. */
export function getGScoreRowOrThrow(playerId: number): GScoreRowData {
    const gs = getGScoreById().get(playerId)
    if (!gs) throw new Error(`G-score not found for player id ${playerId}`)
    return { playerId, values: gs.values, total: gs.total }
}

/**
 * Builds a G-score table for the given players: one row per player plus a
 * team totals row, and an H-score row when a full-team result is provided.
 * Clears and replaces `container` contents on each call.
 */
export function renderTeamGScoreTable(
    playerIds: number[]
  , container: HTMLElement
  , fullTeamResult?: { h_score: number; win_rates: number[] } | null
): void {
    container.innerHTML = ''
    const categories = getSelectedCategories()
    if (playerIds.length === 0) return
    const rows = playerIds.map(getGScoreRowOrThrow)

    const tbl = document.createElement('table')
    tbl.className = 'panel-table panel-table--rounded'
    tbl.style.tableLayout = 'fixed'
    tbl.style.width = '100%'
    tbl.dataset.testid = 'team-gscore'

    // On mobile, narrow the name/total columns and use short category labels
    // (e.g. "Points" → "Pts") so the table's intrinsic min-content fits inside
    // the candidate panel rather than stretching it.
    const isMobile = isMobileViewport()

    // colgroup locks column widths without introducing a visual spacer row.
    tbl.appendChild(makeNameTotalCategoriesColgroup(categories.length, isMobile))
    appendGScoreHeaderRow(tbl.createTHead(), categories, isMobile)
    appendGScoreBodyAndTotals(tbl, rows, categories)
    container.appendChild(tbl)

    if (!fullTeamResult) return
    container.appendChild(buildAlignedHScoreTable(fullTeamResult, categories.length, isMobile))
}

/** Header row: an invisible label spacer, Total, then one column per category — short
 *  category labels on mobile so the columns stay narrow. */
export function appendGScoreHeaderRow(
    tableHead: HTMLTableSectionElement
  , categories: string[]
  , isMobile: boolean
): void {
    const shortNames = isMobile ? getShortCategoryNames() : {}
    const headerRow = tableHead.insertRow(-1)
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
}

/** Data rows — one per player (name, total, per-category G-scores) — plus the Team Total
 *  row summing them. Creates the table's tbody. */
export function appendGScoreBodyAndTotals(
    table: HTMLTableElement
  , rows: GScoreRowData[]
  , categories: string[]
): void {
    const tBody = table.createTBody()
    const teamTotals = new Array(categories.length).fill(0)
    let teamTotalSum = 0

    for (const row of rows) {
        const tr = tBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.append(makeFullPlayerDisplay(row.playerId))
        tr.appendChild(labelCell)

        const totalCell = tr.insertCell(-1)
        totalCell.textContent = row.total.toFixed(2)
        totalCell.className = 'panel-datacell celltypea'
        teamTotalSum += row.total

        for (let i = 0; i < categories.length; i++) {
            const cell = tr.insertCell(-1)
            cell.textContent = row.values[i].toFixed(2)
            cell.style.cssText = stat_styler_primary(row.values[i], G_SCORE_MULTIPLIER, 0)
            cell.className = 'panel-datacell'
            teamTotals[i] += row.values[i]
        }
    }

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
        cell.style.cssText = stat_styler_primary(val, G_SCORE_MULTIPLIER, 0)
        cell.className = 'panel-datacell'
    }
}

/** The one-row H-score table, column-aligned (via the shared colgroup) with a G-score table
 *  above it. Categorical cells show win rates centred at 50; Rotisserie converts them to
 *  expected standings points on the 1..n_drafters scale, coloured by the same H multiplier
 *  scaled to that range. */
export function buildAlignedHScoreTable(
    fullTeamResult: { h_score: number; win_rates: number[] }
  , nCategories: number
  , isMobile: boolean
  , { testId, fullWidth = true }: { testId?: string; fullWidth?: boolean } = {}
): HTMLTableElement {
    const isRoto     = getScoringFormat() === 'Rotisserie'
    const nDrafters  = isRoto ? getLeagueSettings().n_drafters : 0
    const rotoMiddle = computeRotoMiddle(nDrafters)

    const hScoreTbl = document.createElement('table')
    hScoreTbl.className = 'panel-table panel-table--rounded panel-table--top-gap'
    hScoreTbl.style.tableLayout = 'fixed'
    if (fullWidth) hScoreTbl.style.width = '100%'
    if (testId !== undefined) hScoreTbl.dataset.testid = testId

    hScoreTbl.appendChild(makeNameTotalCategoriesColgroup(nCategories, isMobile))

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
            const rotoValue = convertWinRateToRotoPoints(winRate, nDrafters)
            cell.textContent = rotoValue.toFixed(1)
            cell.className = 'categoricalRotoHscore'
            cell.style.cssText = stat_styler_primary(rotoValue, H_MULTIPLIER * (nDrafters - 1), rotoMiddle)
        } else {
            cell.textContent = winRate.toFixed(1)
            cell.className = 'categoricalhscore'
            cell.style.cssText = stat_styler_primary(winRate, H_MULTIPLIER, 50)
        }
    }

    return hScoreTbl
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
