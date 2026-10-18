// table/expand_view.ts
// Builds the expandable detail panel beneath each player row in the main table.
// Mirrors the detail panels in the original Streamlit app.

import {
    stat_styler_primary, stat_styler_tertiary, G_SCORE_MULTIPLIER, G_SCORE_AGGREGATE_MULTIPLIER,
} from '../styles/styler_functions.js'
import { PlayerResult, FlexAllocations, Roster, HypotheticalTeam, BoardRequest } from '../types.js'
import { getPositionNames, getPositionStructure, getLatestScoredBoard } from '../app_state.js'
import { makeSpacerTh } from './table_helpers.js'
import { buildPlayerGScoreCard } from './gscore_table.js'
import { makeRosterPlayerDisplay, makeHypotheticalPlayerDisplay } from '../player_display.js'
import { fetchHypotheticalTeam, HTTPError } from '../api/client.js'
import { getRegistryEntry } from '../player_registry.js'
import { isExampleTeamShown } from '../preferences.js'
import { getMode } from '../setting_collection/league_settings.js'


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

    if (playerData.flex_allocations) {
        cell.appendChild(makePanelLabel('Position allocations for future flex spot picks', '60px'))

        // The flex table keeps its natural width; in an auction the auction values sit beside it.
        const sideRow = document.createElement('div')
        sideRow.className = 'panel-flex-row'
        sideRow.appendChild(makeFlexAllocationsTable(playerData.flex_allocations))
        if (playerData.auction_values) {
            const auctionBlock = document.createElement('div')
            auctionBlock.appendChild(makePanelLabel('All auction values'))
            auctionBlock.appendChild(makeAuctionValuesTable(playerData))
            sideRow.appendChild(auctionBlock)
        }
        cell.appendChild(sideRow)
    } else if (playerData.auction_values) {
        cell.appendChild(makePanelLabel('All auction values', '60px'))
        cell.appendChild(makeAuctionValuesTable(playerData))
    }

    if (playerData.roster) {
        cell.appendChild(makePanelLabel('Roster assignments', '60px'))
        const rosterHolder = document.createElement('div')
        rosterHolder.dataset.testid = 'roster-assignments-panel'   // the grid plus, with an example team, its note
        cell.appendChild(rosterHolder)
        // Draft and auction only, with the Display section's "Show example team" on: the hypothetical team reads the
        // strategy the last draft/auction evaluate cached. A candidate with no category weights (the last open slot)
        // has no future picks to fill.
        const mode = getMode()
        const canShowExampleTeam = (mode === 'Draft Mode' || mode === 'Auction Mode') && Boolean(playerData.category_weights)
        if (canShowExampleTeam && isExampleTeamShown()) {
            fillRosterWithHypotheticalTeam(playerData, rosterHolder)
        } else {
            rosterHolder.appendChild(makeRosterGrid(playerData.roster, null, false))
            // The note's space is kept with the example team off too, so flipping the toggle never changes the
            // panel's height.
            if (canShowExampleTeam) rosterHolder.appendChild(makeHypotheticalTeamNote(false))
        }
    }
}


// ─── Hypothetical team ────────────────────────────────────────────────────────
// Opening a draft/auction candidate's panel fills the roster grid's empty slots with one team the candidate's
// H-scoring plan could become (backend math/exact_team_solves): spinners in the empty slots while it is found, then
// greyed hypothetical picks. Display only: it never changes the ranking.

// Teams (found or being found) for the board the latest evaluate scored, by candidate; reset whenever that board
// changes, so a pick never shows a team built for an earlier board, and reopening a panel never asks twice.
let hypotheticalTeamsBoard: BoardRequest | null = null
const hypotheticalTeamsByCandidate = new Map<number, Promise<HypotheticalTeam>>()
const foundHypotheticalTeamsByCandidate = new Map<number, HypotheticalTeam>()   // the settled ones, shown at once

/** The candidate's hypothetical team on the given scored board: requested once per board and candidate. */
function requestHypotheticalTeam(
    board: { sessionId: string; request: BoardRequest }
    , candidatePlayerId: number
): Promise<HypotheticalTeam> {
    syncHypotheticalTeamsToBoard(board.request)
    let team = hypotheticalTeamsByCandidate.get(candidatePlayerId)
    if (team === undefined) {
        team = fetchHypotheticalTeam(board.sessionId, { ...board.request, candidate_player_id: candidatePlayerId })
        team.then(
            found => { if (hypotheticalTeamsBoard === board.request) foundHypotheticalTeamsByCandidate.set(candidatePlayerId, found) },
            // A failed request is not kept, so reopening the panel tries again.
            () => hypotheticalTeamsByCandidate.delete(candidatePlayerId),
        )
        hypotheticalTeamsByCandidate.set(candidatePlayerId, team)
    }
    return team
}

/** Forgets every team found for an earlier board. */
function syncHypotheticalTeamsToBoard(request: BoardRequest): void {
    if (request === hypotheticalTeamsBoard) return
    hypotheticalTeamsByCandidate.clear()
    foundHypotheticalTeamsByCandidate.clear()
    hypotheticalTeamsBoard = request
}

/** Each hypothetical pick's grid label: its planned round (draft) or predicted price (auction). */
function makeHypotheticalPickLabels(team: HypotheticalTeam): Map<number, string> {
    return new Map(team.hypothetical_picks.map(pick => [
        pick.player_id,
        pick.predicted_price !== null ? `$${pick.predicted_price}` : `Rd ${pick.pick_round}`,
    ]))
}

/**
 * Shows the roster grid with a spinner in every empty slot, then swaps in the hypothetical team (and its note) when
 * it arrives. On failure the plain grid returns, with the reason under it.
 */
function fillRosterWithHypotheticalTeam(
    playerData: PlayerResult
    , rosterHolder: HTMLDivElement
): void {
    const board = getLatestScoredBoard()
    if (board === null) throw new Error('A candidate panel is open with no scored board on record')
    syncHypotheticalTeamsToBoard(board.request)
    const foundTeam = foundHypotheticalTeamsByCandidate.get(playerData.player_id)
    if (foundTeam !== undefined) {
        rosterHolder.append(
            makeRosterGrid(foundTeam.roster, makeHypotheticalPickLabels(foundTeam), false),
            makeHypotheticalTeamNote(true),
        )
        return
    }
    // The spinner grid and the (hidden) note take the filled panel's full height from the start
    // (every roster row is sized for a hypothetical card), so the panel does not grow when the team arrives.
    rosterHolder.append(makeRosterGrid(playerData.roster!, null, true),
                        makeHypotheticalTeamNote(false))

    requestHypotheticalTeam(board, playerData.player_id)
        .then(team => {
            // A newer board rebuilds the open panels and asks for its own team.
            if (getLatestScoredBoard()?.request !== board.request) return
            rosterHolder.replaceChildren(
                makeRosterGrid(team.roster, makeHypotheticalPickLabels(team), false),
                makeHypotheticalTeamNote(true),
            )
            document.dispatchEvent(new Event('expand-panel-resized'))
        })
        .catch(error => {
            if (getLatestScoredBoard()?.request !== board.request) return
            const message = document.createElement('div')
            message.className = 'hypothetical-team-error'
            message.textContent = error instanceof HTTPError ? readErrorDetail(error) : String(error)
            rosterHolder.replaceChildren(
                makeRosterGrid(playerData.roster!, null, false), message)
            document.dispatchEvent(new Event('expand-panel-resized'))
        })
}

/** The server's error message from an HTTPError body ({"detail": ...}), or the raw body when it is not JSON. */
function readErrorDetail(error: HTTPError): string {
    try {
        return JSON.parse(error.body).detail ?? error.body
    } catch {
        return error.body
    }
}

// ─── G-score hover card ───────────────────────────────────────────────────────
// Hovering a filled roster cell (a tap on a touch screen) shows that player's G-scores -- the Total and each category
// -- in a card floating beside the cell. One card at a time; it floats above the page, so the panel never moves.

let gScoreCard: { element: HTMLDivElement; anchor: HTMLElement } | null = null

/** Shows the player's G-score card on hover, or on a tap for touch input (a second tap closes it). */
function attachGScoreCard(cell: HTMLElement, playerId: number): void {
    cell.addEventListener('mouseenter', () => showGScoreCard(cell, playerId))
    cell.addEventListener('mouseleave', () => hideGScoreCard())
    cell.addEventListener('click', event => {
        if ((event as PointerEvent).pointerType === 'mouse') return   // the mouse already shows it on hover
        if (gScoreCard?.anchor === cell) hideGScoreCard()
        else showGScoreCard(cell, playerId)
        event.stopPropagation()
    })
}

/** Places the card centred under the cell (over it when there is no room below in the window), kept inside the window,
 *  in page coordinates so it stays beside the cell when the page scrolls. */
function showGScoreCard(cell: HTMLElement, playerId: number): void {
    hideGScoreCard()
    const element = document.createElement('div')
    element.className = 'gscore-hover-card'
    element.appendChild(buildPlayerGScoreCard(playerId))
    document.body.appendChild(element)
    const cellBox = cell.getBoundingClientRect()
    const cardBox = element.getBoundingClientRect()
    const margin = 6
    const below = cellBox.bottom + margin
    const top = below + cardBox.height <= window.innerHeight ? below : cellBox.top - margin - cardBox.height
    const centred = cellBox.left + cellBox.width / 2 - cardBox.width / 2
    const left = Math.min(Math.max(margin, centred), window.innerWidth - cardBox.width - margin)
    element.style.top = `${Math.max(margin, top) + window.scrollY}px`
    element.style.left = `${Math.max(margin, left) + window.scrollX}px`
    gScoreCard = { element, anchor: cell }
}

function hideGScoreCard(): void {
    gScoreCard?.element.remove()
    gScoreCard = null
}

// A resize can reflow the cell away from the card, so it closes the card; so does a tap elsewhere.
window.addEventListener('resize', hideGScoreCard)
document.addEventListener('click', hideGScoreCard)


/** A roster cell's corner label listing the positions the player is eligible for, one per line (a narrow column). */
function makePositionEligibilityLabel(playerId: number): HTMLSpanElement {
    const label = document.createElement('span')
    label.className = 'roster-position-eligibility'
    for (const position of getRegistryEntry(playerId).positions) {
        const line = document.createElement('span')
        line.textContent = position
        label.appendChild(line)
    }
    return label
}

/** The one-line note under a grid holding a hypothetical team. While the team is still being found the note is not
 *  shown, but keeps its line, so the panel does not grow when the team arrives. */
function makeHypotheticalTeamNote(isTeamShown: boolean): HTMLDivElement {
    const note = document.createElement('div')
    note.className = 'hypothetical-team-note'
    note.style.visibility = isTeamShown ? 'visible' : 'hidden'
    note.textContent = 'Potential future picks are hypothetical, and for illustrative purposes only. '
        + 'They are found based on the results of H-scoring but not taken into account for scoring'
    return note
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
// Category cells: stat_styler_primary (middle=0), G_SCORE_MULTIPLIER on the candidate's row and
// G_SCORE_AGGREGATE_MULTIPLIER on the team-level rows, which run several times larger.
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

    // Data rows: the candidate's own row first, then the team-level rows. The two blocks colour on different multipliers,
    // so the first team-level row opens with a darker line (.gscore-aggregate-start) marking where the scale changes.
    const tableBody = table.createTBody()
    const orderedRows = [
        ...playerData.g_score_rows.filter(rowData => rowData.isCandidate),
        ...playerData.g_score_rows.filter(rowData => !rowData.isCandidate),
    ]
    for (const [rowIndex, rowData] of orderedRows.entries()) {
        const row = tableBody.insertRow(-1)
        if (!rowData.isCandidate && orderedRows[rowIndex - 1]?.isCandidate) row.className = 'gscore-aggregate-start'

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.textContent = rowData.label
        row.appendChild(labelCell)

        const totalCell = row.insertCell(-1)
        totalCell.textContent = rowData.total.toFixed(2)
        totalCell.className = 'panel-datacell' + (rowData.isTotal ? ' celltypeb' : ' celltypea')

        const multiplier = rowData.isCandidate ? G_SCORE_MULTIPLIER : G_SCORE_AGGREGATE_MULTIPLIER
        for (const value of rowData.values) {
            const cell = row.insertCell(-1)
            cell.textContent = value.toFixed(2)
            cell.style.cssText = stat_styler_primary(value, multiplier, 0)
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
 * @param flexData - The flex allocation data for this player
 */
function makeFlexAllocationsTable(flexData: FlexAllocations): HTMLDivElement {
    const positionNames = getPositionNames()

    const table = document.createElement('table')
    table.className = 'panel-table flex-allocations-table'
    table.dataset.testid = 'flex-allocations-table'
    table.style.setProperty('--flex-position-count', String(flexData.base_positions.length))

    const headerRow = table.createTHead().insertRow(-1)
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank panel-colspacer-position-label'))
    for (const basePosition of flexData.base_positions) {
        const positionHeader = document.createElement('th')
        positionHeader.className = 'panel-colheader'
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
    for (const value of [auctionValues.your_dollar, auctionValues.generic_dollar, auctionValues.original_dollar]) {
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
    for (const value of [auctionValues.generic_dollar_g_score, auctionValues.original_dollar_g_score]) {
        const cell = gScoreRow.insertCell(-1)
        cell.className = 'panel-datacell celltypea'
        cell.textContent = value.toFixed(1)
    }

    return table
}


/** One row of the roster grid: the base positions, or the flex positions. */
function makeRosterBlock(): HTMLDivElement {
    const block = document.createElement('div')
    block.className = 'roster-block'
    return block
}


// ─── Roster grid ──────────────────────────────────────────────────────────────
// One group per position type (PG, SG, ..., Util), its slots side by side under the type's label, in two rows: the
// base positions over the flex positions. Every cell has the same width, so a roster of any shape reads evenly with no
// filler cells; a row too wide for the panel (a phone) wraps within itself.

/**
 * Builds the roster grid showing which players are assigned to each slot,
 * and highlights the candidate player being evaluated. hypotheticalPickLabels labels each hypothetical
 * pick's cell (planned round or predicted price); null when the roster holds none. showPendingSpinners puts a
 * spinner in every empty slot while a hypothetical team is being found.
 */
function makeRosterGrid(
    roster: Roster
    , hypotheticalPickLabels: Map<number, string> | null
    , showPendingSpinners: boolean
): HTMLDivElement {
    const positionNames = getPositionNames()

    // Group slots by position type (e.g. "PG1", "PG2" → type "PG") preserving insertion order.
    const positionTypes: string[] = []
    const slotsByType: Record<string, string[]> = {}
    for (const slot of roster.slots) {
        const positionType = slot.replace(/\d+$/, '')
        if (!slotsByType[positionType]) { slotsByType[positionType] = []; positionTypes.push(positionType) }
        slotsByType[positionType].push(slot)
    }

    hideGScoreCard()
    const groups = document.createElement('div')
    groups.className = 'roster-groups'
    groups.dataset.testid = 'roster-assignments-table'
    const flexPositions = new Set(getPositionStructure().flex_list)
    const baseTypes = positionTypes.filter(positionType => !flexPositions.has(positionType))
    const flexTypes = positionTypes.filter(positionType => flexPositions.has(positionType))
    const baseBlock = makeRosterBlock()
    const flexBlock = makeRosterBlock()
    if (baseTypes.length > 0) groups.appendChild(baseBlock)
    if (flexTypes.length > 0) groups.appendChild(flexBlock)

    for (const positionType of positionTypes) {
        const slots = slotsByType[positionType]
        const group = document.createElement('div')
        group.className = 'roster-group'
        group.style.setProperty('--roster-group-slots', String(slots.length))
        const table = document.createElement('table')
        table.className = 'panel-table'
        group.appendChild(table)

        const header = document.createElement('th')
        header.className = 'panel-colheader'
        header.colSpan = slots.length
        header.textContent = positionNames[positionType] ?? positionType   // full names, as in the position allocation table
        table.createTHead().insertRow(-1).appendChild(header)

        const row = table.createTBody().insertRow(-1)
        for (const slot of slots) {
            const cell = row.insertCell(-1)
            const assignment = roster.assignments[slot]
            if (!assignment && showPendingSpinners) {
                cell.className = 'rosterabsent rosterpending panel-datacell'
                const spinner = document.createElement('span')
                spinner.className = 'roster-slot-spinner'
                cell.appendChild(spinner)
            } else if (!assignment) {
                cell.className = 'rosterabsent panel-datacell'
                cell.textContent = '\u00A0'
            } else if (assignment.kind === 'hypothetical') {
                const label = hypotheticalPickLabels?.get(assignment.player_id)
                if (label === undefined) throw new Error(`No label for hypothetical pick ${assignment.player_id}`)
                cell.className = 'rosterhypothetical panel-datacell roster-player-cell'
                cell.appendChild(makeHypotheticalPlayerDisplay(assignment.player_id, label))
                cell.appendChild(makePositionEligibilityLabel(assignment.player_id))
                attachGScoreCard(cell, assignment.player_id)
            } else {
                cell.className = (assignment.kind === 'candidate' ? 'rostercandidate' : 'rosteronteam')
                    + ' panel-datacell roster-player-cell'
                cell.appendChild(makeRosterPlayerDisplay(assignment.player_id))
                cell.appendChild(makePositionEligibilityLabel(assignment.player_id))
                attachGScoreCard(cell, assignment.player_id)
            }
        }
        ;(flexPositions.has(positionType) ? flexBlock : baseBlock).appendChild(group)
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'panel-indent'
    wrapper.appendChild(groups)
    return wrapper
}
