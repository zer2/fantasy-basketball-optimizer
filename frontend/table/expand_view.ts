// table/expand_view.ts
// Builds the expandable detail panel beneath each player row in the main table.
// Mirrors the detail panels in the original Streamlit app.

import { stat_styler_primary, stat_styler_tertiary } from '../styles/styler_functions.js'
import { PlayerResult, FlexAllocations, Roster } from '../types.js'
import { getPositionNames } from '../app_state.js'
import { makeSpacerTh } from './table_helpers.js'


/** Expands a flex slot row label by expanding the position prefix.
 *  e.g. "G-1" → "Guard-1", "Util-3" → "Utility-3".
 */
function expandFlexLabel(label: string): string {
    const dashIndex = label.indexOf('-')
    const positionNames = getPositionNames()
    const prefix = label.slice(0, dashIndex)
    return (positionNames[prefix] ?? prefix) + label.slice(dashIndex)
}


/**
 * Builds the expandable detail panel (G-score, category weights, flex allocations, roster) into the
 * given expand row's single spanning cell. Pure content builder: the caller (player_table) owns showing,
 * hiding, clearing, and measuring the row — this only fills an *empty* expand row.
 *
 * @param expandedRow - The (empty) detail row to fill
 * @param playerData  - Full player data object
 * @param categories  - Ordered list of category names matching the table columns
 */
export function buildExpandPanel(
    expandedRow: HTMLTableRowElement
    , playerData: PlayerResult
    , categories: string[]
): void {
    const scoreCols = playerData.auction_values ? 4 : 1
    const totalCols = 1 + scoreCols + categories.length

    // Single cell spanning all columns
    const cell = expandedRow.insertCell(-1)
    cell.colSpan = totalCols
    cell.className = 'panel-cell'

    cell.appendChild(makePanelLabel('G-score expectations (difference vs. other teams)', '60px'))
    cell.appendChild(makeGScoreTable(playerData, categories))

    if (playerData.category_weights) {
        cell.appendChild(makePanelLabel('Category strategy', '60px'))
        cell.appendChild(makeWeightsTable(playerData, categories))
    }

    // Position-column count shared between the flex allocations and roster
    // tables so each non-label column has the same width in both. The roster
    // always includes every base position the flex table can allocate to plus
    // extra flex types (G, F, Util), so its position-type count is the larger
    // of the two and the flex table pads with invisible filler columns to match.
    const nRosterPositionTypes = playerData.roster
        ? new Set(playerData.roster.slots.map(slot => slot.replace(/\d+$/, ''))).size
        : 0

    if (playerData.flex_allocations) {
        cell.appendChild(makePanelLabel('Position allocations for future flex spot picks', '60px'))

        if (playerData.auction_values) {
            // Side-by-side with auction values. The flex table keeps its natural
            // per-column width (no roster filler) so the auction block fits beside it.
            const sideRow = document.createElement('div')
            sideRow.className = 'panel-flex-row'
            sideRow.appendChild(makeFlexAllocationsTable(playerData.flex_allocations, false, playerData.flex_allocations.base_positions.length))

            const auctionBlock = document.createElement('div')
            auctionBlock.appendChild(makePanelLabel('All auction values'))
            auctionBlock.appendChild(makeAuctionValuesTable(playerData))
            sideRow.appendChild(auctionBlock)

            cell.appendChild(sideRow)
        } else {
            cell.appendChild(makeFlexAllocationsTable(playerData.flex_allocations, true, nRosterPositionTypes))
        }
    } else if (playerData.auction_values) {
        cell.appendChild(makePanelLabel('All auction values', '60px'))
        cell.appendChild(makeAuctionValuesTable(playerData))
    }

    if (playerData.roster) {
        cell.appendChild(makePanelLabel('Roster assignments', '60px'))
        cell.appendChild(makeRosterGrid(playerData.roster, nRosterPositionTypes))
    }
}


// ─── Panel section label ──────────────────────────────────────────────────────

/** Creates a `.panel-label` div with optional left-padding for indentation. */
function makePanelLabel(text: string, paddingLeft?: string): HTMLDivElement {
    const label = document.createElement('div')
    label.className = 'panel-label'
    label.textContent = text
    if (paddingLeft) label.style.paddingLeft = paddingLeft
    return label
}


// ─── G-score expectations table ───────────────────────────────────────────────
// Rows: current diff, player contribution, future diff, total diff.
// Category cells: stat_styler_primary (middle=0, multiplier=60).
// Total column: celltypea / celltypeb for the summary row.

/**
 * Builds the G-score expectations table for one candidate player.
 * Rows show how the player changes each category's head-to-head win expectation.
 */
function makeGScoreTable(playerData: PlayerResult, categories: string[]): HTMLDivElement {
    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'
    table.dataset.testid = 'gscore-expectations-table'

    // colgroup locks column widths without introducing a visual spacer row.
    const colgroup = document.createElement('colgroup')
    const nameCol  = document.createElement('col')
    nameCol.style.width = '136px'   // matches .panel-colspacer-name-sm
    const totalCol = document.createElement('col')
    totalCol.style.width = '83px'   // matches .panel-colspacer-total
    colgroup.append(nameCol, totalCol)
    for (let index = 0; index < categories.length; index++) colgroup.appendChild(document.createElement('col'))
    table.appendChild(colgroup)

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank'))
    const totalHeader = document.createElement('th')
    totalHeader.className = 'panel-colheader'
    totalHeader.textContent = 'Total'
    headerRow.appendChild(totalHeader)
    for (const category of categories) {
        const categoryHeader = document.createElement('th')
        categoryHeader.className = category.length >= 10 ? 'panel-colheader colheader-long' : 'panel-colheader'
        categoryHeader.textContent = category
        headerRow.appendChild(categoryHeader)
    }

    // Data rows
    const tableBody = table.createTBody()
    for (const rowData of playerData.g_score_rows) {
        const row = tableBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.textContent = rowData.label
        row.appendChild(labelCell)

        const totalCell = row.insertCell(-1)
        totalCell.textContent = rowData.total.toFixed(2)
        totalCell.className = 'panel-datacell' + (rowData.isTotal ? ' celltypeb' : ' celltypea')

        for (const value of rowData.values) {
            const cell = row.insertCell(-1)
            cell.textContent = value.toFixed(2)
            cell.style.cssText = stat_styler_primary(value, 60, 0)
            cell.className = 'panel-datacell'
        }
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'panel-indent'
    wrapper.appendChild(table)
    return wrapper
}


// ─── Category weights table ───────────────────────────────────────────────────
// Single data row: algorithm weight assigned to each category for future picks.
// Values are percentages; '%' suffix is added by CSS .panel-weight::after.

/**
 * Builds the category weights table showing how the algorithm weights each
 * stat category when evaluating future picks after this candidate is drafted.
 */
function makeWeightsTable(playerData: PlayerResult, categories: string[]): HTMLDivElement {
    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'
    table.dataset.testid = 'future-pick-strategy-table'

    // colgroup locks column widths without introducing a visual spacer row.
    const colgroup = document.createElement('colgroup')
    const labelCol = document.createElement('col')
    labelCol.style.width = '219px'   // matches .panel-colspacer-weights
    colgroup.appendChild(labelCol)
    for (let index = 0; index < categories.length; index++) colgroup.appendChild(document.createElement('col'))
    table.appendChild(colgroup)

    const row = table.createTBody().insertRow(-1)
    const labelCell = document.createElement('th')
    labelCell.className = 'panel-rowlabel'
    labelCell.textContent = 'Future pick weight'
    row.appendChild(labelCell)
    for (const value of playerData.category_weights) {
        const cell = row.insertCell(-1)
        cell.textContent = value.toFixed(0)  // '%' appended by CSS panel-weight::after
        // A single blue ramp: white at the low end, deepening with the weight, so the pursued
        // categories are the strong blues and the punts fade toward white. Weights sit ~70–115, so
        // the white anchor is 70 (the bottom of that range) — anchoring at 0 would cram every weight
        // into a narrow, uniform-looking slice of the ramp. Clamp to >=70 so a deep punt stays white
        // rather than being re-coloured by tertiary's |raw|. ~90% lands on light blue; 110% is deep.
        cell.style.cssText = stat_styler_tertiary(Math.max(value, 70), 2.5, 70)
        cell.className = 'panel-datacell panel-weight'
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'panel-indent'
    wrapper.appendChild(table)
    return wrapper
}


// ─── Flex position allocations table ──────────────────────────────────────────
// Rows: flex slot types (G, F, Util, Total); Cols: base positions (PG–C).
// Each cell is the expected number of times that base position fills that flex slot.
// null means that base position is ineligible for that flex slot type.

/**
 * Builds the flex allocations table showing how future picks are expected to
 * fill remaining flex roster slots, given that this candidate is drafted.
 *
 * @param flexData   - The flex allocation data for this player
 * @param useMargin  - Whether to apply the standard left-margin indent (default true)
 */
function makeFlexAllocationsTable(
    flexData: FlexAllocations
    , useMargin: boolean
    , nTotalColumns: number
): HTMLDivElement {
    const positionNames = getPositionNames()

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.dataset.testid = 'flex-allocations-table'
    table.style.tableLayout = 'fixed'
    table.style.width = '100%'

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank panel-colspacer-position-label'))
    for (const basePosition of flexData.base_positions) {
        const positionHeader = document.createElement('th')
        positionHeader.className = 'panel-colheader'
        positionHeader.textContent = positionNames[basePosition] ?? basePosition
        headerRow.appendChild(positionHeader)
    }
    // Invisible filler columns so this table's column count matches the roster's,
    // giving each non-label column an identical width in both tables.
    for (let i = flexData.base_positions.length; i < nTotalColumns; i++) {
        headerRow.appendChild(makeSpacerTh('panel-colheader-blank'))
    }

    const tableBody = table.createTBody()
    for (const rowData of flexData.rows) {
        const row = tableBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = rowData.isTotal ? 'panel-rowlabel panel-rowlabel-total' : 'panel-rowlabel'
        labelCell.textContent = rowData.isTotal ? rowData.label : expandFlexLabel(rowData.label)
        row.appendChild(labelCell)

        for (const value of rowData.values) {
            const cell = row.insertCell(-1)
            if (value === null) {
                cell.className = 'ineligible panel-datacell'
            } else {
                cell.className = 'panel-datacell'
                cell.textContent = value.toFixed(2)
                cell.style.cssText = stat_styler_tertiary(value, 50, 0)
            }
        }
        // Filler cells with border-style:hidden — in border-collapse mode, hidden
        // borders override any neighbor's solid borders at the shared edge, so the
        // row's bottom line does not extend into the filler region.
        for (let i = rowData.values.length; i < nTotalColumns; i++) {
            const filler = row.insertCell(-1)
            filler.style.cssText = 'border-style: hidden; padding: 0;'
        }
    }

    const wrapper = document.createElement('div')
    if (useMargin) wrapper.className = 'panel-indent'
    wrapper.appendChild(table)
    return wrapper
}


// ─── Auction values table ─────────────────────────────────────────────────────
// Two rows (H-score, G-score) × three columns (Your $, Gnrc. $, Orig. $).
// The H-score row mirrors the values shown in the main candidate table.
// The G-score row shows the same dollar columns computed from G-scores instead;
// Your $ has no G-score equivalent and is shown as a dash.

/** Builds the auction values detail table: H-score and G-score rows × Your $ / Gnrc. $ / Orig. $ columns. */
function makeAuctionValuesTable(playerData: PlayerResult): HTMLTableElement {
    const auctionValues = playerData.auction_values!

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'
    table.dataset.testid = 'auction-values-table'

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colspacer-dollar'))
    for (const columnLabel of ['Your $', 'Gnrc. $', 'Orig. $']) {
        const columnHeader = document.createElement('th')
        columnHeader.className = 'panel-colheader'
        columnHeader.style.width = '72px'
        columnHeader.textContent = columnLabel
        headerRow.appendChild(columnHeader)
    }

    const tableBody = table.createTBody()

    const hScoreRow = tableBody.insertRow(-1)
    const hScoreLabel = document.createElement('th')
    hScoreLabel.className = 'panel-rowlabel'
    hScoreLabel.textContent = 'H-score'
    hScoreRow.appendChild(hScoreLabel)
    for (const value of [auctionValues.your_dollar, auctionValues.gnrc_dollar, auctionValues.orig_dollar]) {
        const cell = hScoreRow.insertCell(-1)
        cell.className = 'panel-datacell celltypea'
        cell.textContent = value.toFixed(1)
    }

    // G-score row (Your $ is not applicable for G-scores)
    const gScoreRow = tableBody.insertRow(-1)
    const gScoreLabel = document.createElement('th')
    gScoreLabel.className = 'panel-rowlabel'
    gScoreLabel.textContent = 'G-score'
    gScoreRow.appendChild(gScoreLabel)
    const naCell = gScoreRow.insertCell(-1)
    naCell.className = 'panel-datacell celltypea'
    naCell.textContent = '—'
    for (const value of [auctionValues.gnrc_dollar_g, auctionValues.orig_dollar_g]) {
        const cell = gScoreRow.insertCell(-1)
        cell.className = 'panel-datacell celltypea'
        cell.textContent = value.toFixed(1)
    }

    return table
}


// ─── Roster grid ──────────────────────────────────────────────────────────────
// Rows: depth level (1st slot, 2nd slot, …); Cols: all position types (PG…Util).
// Base-position cols align with the flex allocations table above.

/**
 * Builds the roster grid showing which players are assigned to each slot,
 * and highlights the candidate player being evaluated.
 */
function makeRosterGrid(roster: Roster, nTotalColumns: number): HTMLDivElement {
    const positionNames = getPositionNames()

    // Group slots by position type (e.g. "PG1", "PG2" → type "PG") preserving insertion order.
    const positionTypes: string[] = []
    const slotsByType: Record<string, string[]> = {}
    for (const slot of roster.slots) {
        const positionType = slot.replace(/\d+$/, '')
        if (!slotsByType[positionType]) { slotsByType[positionType] = []; positionTypes.push(positionType) }
        slotsByType[positionType].push(slot)
    }
    const maxDepth = Math.max(...positionTypes.map(positionType => slotsByType[positionType].length))

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.dataset.testid = 'roster-assignments-table'
    table.style.tableLayout = 'fixed'
    table.style.width = '100%'

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank panel-colspacer-position-label'))
    for (const positionType of positionTypes) {
        const positionHeader = document.createElement('th')
        positionHeader.className = 'panel-colheader'
        positionHeader.textContent = positionNames[positionType] ?? positionType
        headerRow.appendChild(positionHeader)
    }
    // Invisible filler columns so this table's column count matches the flex
    // allocations table's, giving each non-label column an identical width.
    for (let i = positionTypes.length; i < nTotalColumns; i++) {
        headerRow.appendChild(makeSpacerTh('panel-colheader-blank'))
    }

    const tableBody = table.createTBody()
    for (let depthIndex = 0; depthIndex < maxDepth; depthIndex++) {
        const row = tableBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.textContent = 'Slot ' + (depthIndex + 1)
        row.appendChild(labelCell)

        for (const positionType of positionTypes) {
            const cell = row.insertCell(-1)
            const slot = slotsByType[positionType][depthIndex]
            if (slot === undefined) {
                cell.className = 'ineligible panel-datacell'
                cell.textContent = '\u00A0'
            } else {
                const assignment = roster.assignments[slot]
                if (!assignment) {
                    cell.className = 'rosterabsent panel-datacell'
                    cell.textContent = '\u00A0'
                } else if (assignment.isCandidate) {
                    cell.className = 'rostercandidate panel-datacell'
                    cell.textContent = assignment.name
                } else {
                    cell.className = 'rosteronteam panel-datacell'
                    cell.textContent = assignment.name
                }
            }
        }
        // See makeFlexAllocationsTable for the filler-cell rationale.
        for (let i = positionTypes.length; i < nTotalColumns; i++) {
            const filler = row.insertCell(-1)
            filler.style.cssText = 'border-style: hidden; padding: 0;'
        }
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'panel-indent'
    wrapper.appendChild(table)
    return wrapper
}
