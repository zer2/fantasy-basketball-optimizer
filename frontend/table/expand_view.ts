// table/expand_view.ts
// Builds the expandable detail panel beneath each player row in the main table.
// Mirrors the detail panels in the original Streamlit app.

import { stat_styler_primary, stat_styler_tertiary } from '../styles/styler_functions.js'
import { Player, FlexAllocations, Roster } from '../types.js'
import { getPositionNames } from '../app_state.js'


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
 * Toggles the expandable detail panel for a player row.
 * On expand: builds G-score, category weights, flex allocations, and roster tables.
 * On collapse: clears the panel and hides the row.
 *
 * @param expandButton - The expand button that was clicked
 * @param expandedRow  - The detail row immediately associated with this player
 * @param playerData   - Full player data object
 * @param categories   - Ordered list of category names matching the table columns
 */
export function toggleExpandView(
    expandButton: HTMLButtonElement
    , expandedRow: HTMLTableRowElement
    , playerData: Player
    , categories: string[]
): void {

    if (expandedRow.style.display === 'table-row') {
        // Collapse: clear content and hide
        expandedRow.style.display = 'none'
        expandedRow.innerHTML = ''
        expandButton.classList.remove('popup-open')
    } else {
        // Expand: build the detail panel as a vertical stack
        expandedRow.style.display = 'table-row'
        expandButton.classList.add('popup-open')

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

        if (playerData.flex_allocations) {
            cell.appendChild(makePanelLabel('Position allocations for future flex spot picks', '60px'))

            // Compute a shared per-position column width so base position columns align
            // between the flex allocations table and the roster grid beneath it.
            // Size columns using the wider table (roster, which has extra flex-slot columns)
            // so both tables fit within the available space.
            const INDENT_W         = 100   // matches .panel-indent margin-left
            const POSITION_LABEL_W = 110   // matches .panel-colspacer-position-label
            const containerWidth   = cell.clientWidth
            const nBasePositions   = playerData.flex_allocations.base_positions.length
            const nRosterPositionTypes = playerData.roster
                ? new Set(playerData.roster.slots.map(slot => slot.replace(/\d+$/, ''))).size
                : nBasePositions
            const nMaxColumns     = Math.max(nBasePositions, nRosterPositionTypes)
            const positionColumnWidth = Math.floor((containerWidth - INDENT_W - POSITION_LABEL_W) / nMaxColumns)

            if (playerData.auction_values) {
                // In auction mode, show flex allocations and auction values side-by-side.
                const sideRow = document.createElement('div')
                sideRow.className = 'panel-flex-row'
                sideRow.appendChild(makeFlexAllocationsTable(playerData.flex_allocations, false, positionColumnWidth))

                const auctionBlock = document.createElement('div')
                auctionBlock.appendChild(makePanelLabel('All auction values'))
                auctionBlock.appendChild(makeAuctionValuesTable(playerData))
                sideRow.appendChild(auctionBlock)

                cell.appendChild(sideRow)
            } else {
                cell.appendChild(makeFlexAllocationsTable(playerData.flex_allocations, true, positionColumnWidth))
            }

            if (playerData.roster) {
                cell.appendChild(makePanelLabel('Roster assignments', '60px'))
                cell.appendChild(makeRosterGrid(playerData.roster, positionColumnWidth))
            }
        } else if (playerData.auction_values) {
            // No position data but in auction mode: still show auction values alone.
            cell.appendChild(makePanelLabel('All auction values', '60px'))
            cell.appendChild(makeAuctionValuesTable(playerData))

            if (playerData.roster) {
                cell.appendChild(makePanelLabel('Roster assignments', '60px'))
                const INDENT_W         = 100
                const POSITION_LABEL_W = 110
                const nRosterPositionTypes = new Set(playerData.roster.slots.map(slot => slot.replace(/\d+$/, ''))).size
                cell.appendChild(makeRosterGrid(playerData.roster, Math.floor((cell.clientWidth - INDENT_W - POSITION_LABEL_W) / nRosterPositionTypes)))
            }
        } else if (playerData.roster) {
            const INDENT_W         = 100
            const POSITION_LABEL_W = 110
            const nRosterPositionTypes = new Set(playerData.roster.slots.map(slot => slot.replace(/\d+$/, ''))).size
            cell.appendChild(makePanelLabel('Roster assignments', '60px'))
            cell.appendChild(makeRosterGrid(playerData.roster, Math.floor((cell.clientWidth - INDENT_W - POSITION_LABEL_W) / nRosterPositionTypes)))
        }
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

/** Creates an invisible `<th>` spacer used to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const spacer = document.createElement('th')
    spacer.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return spacer
}


// ─── G-score expectations table ───────────────────────────────────────────────
// Rows: current diff, player contribution, future diff, total diff.
// Category cells: stat_styler_primary (middle=0, multiplier=60).
// Total column: celltypea / celltypeb for the summary row.

/**
 * Builds the G-score expectations table for one candidate player.
 * Rows show how the player changes each category's head-to-head win expectation.
 */
function makeGScoreTable(playerData: Player, categories: string[]): HTMLDivElement {
    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'

    // Row 1: invisible spacers lock column widths.
    const tableHead = table.createTHead()
    const spacerRow = tableHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name-sm'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let index = 0; index < categories.length; index++) spacerRow.appendChild(makeSpacerTh())

    // Row 2: visible column headers.
    const headerRow = tableHead.insertRow(-1)
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
function makeWeightsTable(playerData: Player, categories: string[]): HTMLDivElement {
    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'

    const spacerRow = table.createTHead().insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-weights'))
    for (let index = 0; index < categories.length; index++) spacerRow.appendChild(makeSpacerTh())

    const row = table.createTBody().insertRow(-1)
    const labelCell = document.createElement('th')
    labelCell.className = 'panel-rowlabel'
    labelCell.textContent = 'Future pick weight'
    row.appendChild(labelCell)
    for (const value of playerData.category_weights) {
        const cell = row.insertCell(-1)
        cell.textContent = value.toFixed(0)  // '%' appended by CSS panel-weight::after
        cell.style.cssText = stat_styler_tertiary(value, 5, 90)
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
    , useMargin = true
    , positionColumnWidth: number
): HTMLDivElement {
    const positionNames = getPositionNames()

    const POSITION_LABEL_W = 110  // matches .panel-colspacer-position-label

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'
    table.style.width = (POSITION_LABEL_W + flexData.base_positions.length * positionColumnWidth) + 'px'

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank panel-colspacer-position-label'))
    for (const basePosition of flexData.base_positions) {
        const positionHeader = document.createElement('th')
        positionHeader.className = 'panel-colheader'
        positionHeader.style.width = positionColumnWidth + 'px'
        positionHeader.textContent = positionNames[basePosition] ?? basePosition
        headerRow.appendChild(positionHeader)
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
function makeAuctionValuesTable(playerData: Player): HTMLTableElement {
    const auctionValues = playerData.auction_values!

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'

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
function makeRosterGrid(roster: Roster, positionColumnWidth: number): HTMLDivElement {
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

    const POSITION_LABEL_W = 110  // matches .panel-colspacer-position-label

    const table = document.createElement('table')
    table.className = 'panel-table'
    table.style.tableLayout = 'fixed'
    table.style.width = (POSITION_LABEL_W + positionTypes.length * positionColumnWidth) + 'px'

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank panel-colspacer-position-label'))
    for (const positionType of positionTypes) {
        const positionHeader = document.createElement('th')
        positionHeader.className = 'panel-colheader'
        positionHeader.style.width = positionColumnWidth + 'px'
        positionHeader.textContent = positionNames[positionType] ?? positionType
        headerRow.appendChild(positionHeader)
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
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'panel-indent'
    wrapper.appendChild(table)
    return wrapper
}
