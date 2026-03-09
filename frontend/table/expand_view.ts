// table/expand_view.ts
// Builds the expandable detail panel beneath each player row in the main table.
// Mirrors the detail panels in the original Streamlit app.

import { stat_styler_primary, stat_styler_tertiary} from '../styles/styler_functions.js'
import { Player, FlexAllocations, Roster } from '../types.js'

const POSITION_NAMES: Record<string, string> = {
    PG:   'Point Guard',
    SG:   'Shooting Guard',
    SF:   'Small Forward',
    PF:   'Power Forward',
    C:    'Center',
    G:    'Guard',
    F:    'Forward',
    Util: 'Utility',
}

/** Expands a position abbreviation to its full name, e.g. "PG" → "Point Guard". */
function expandPosition(abbr: string): string {
    return POSITION_NAMES[abbr] ?? abbr
}

/**
 * Expands flex slot row labels by expanding the position prefix.
 * e.g. "G-1" → "Guard-1", "Util-3" → "Utility-3", "Total" → "Total".
 */
function expandFlexLabel(label: string): string {
    const dashIdx = label.indexOf('-')
    if (dashIdx === -1) return label
    return expandPosition(label.slice(0, dashIdx)) + label.slice(dashIdx)
}

/**
 * Toggles the expandable detail panel for a player row.
 * On expand: builds G-score, category weights, flex allocations, and roster tables.
 * On collapse: clears the panel and hides the row.
 *
 * @param playerIndex - Index of the player row in the table (used for DOM ID lookup)
 * @param playerData  - Full player data object
 * @param categories  - Ordered list of category names matching the table columns
 */
export function ExpandView(playerIndex: number, playerData: Player, categories: string[], totalCols: number): void {

    let evpopup = document.querySelector(`#PP${playerIndex}.playerpopup`) as HTMLButtonElement;
    let expandedRow = document.querySelector(`.EV${playerIndex}.expandedview`) as HTMLTableRowElement;

    if (expandedRow.style.display === 'table-row') {
        // Collapse: clear content and hide
        expandedRow.style.display = 'none';
        expandedRow.innerHTML = '';
        evpopup.classList.remove('popup-open');
    } else {
        // Expand: build the detail panel as a vertical stack
        expandedRow.style.display = 'table-row';
        evpopup.classList.add('popup-open');

        // Single cell spanning all columns
        let cell = expandedRow.insertCell(-1);
        cell.colSpan = totalCols;
        cell.className = 'panel-cell';

        cell.appendChild(makePanelLabel('G-score expectations (difference vs. other teams)', '60px'));
        cell.appendChild(makeGScoreTable(playerData, categories));


        if (playerData.category_weights) {
            cell.appendChild(makePanelLabel('Category strategy', '60px'));
            cell.appendChild(makeWeightsTable(playerData, categories));
        }

        if (playerData.flex_allocations) {
            cell.appendChild(makePanelLabel('Position allocations for future flex spot picks', '60px'));

            if (playerData.auction_values) {
                // In auction mode, show the flex allocations and the auction values table side-by-side.
                const sideRow = document.createElement('div');
                sideRow.style.cssText = 'display:flex; gap:32px; align-items:flex-start; margin-left:100px;';
                sideRow.appendChild(makeFlexAllocationsTable(playerData.flex_allocations, true));

                const auctionBlock = document.createElement('div');
                auctionBlock.appendChild(makePanelLabel('All auction values'));
                auctionBlock.appendChild(makeAuctionValuesTable(playerData));
                sideRow.appendChild(auctionBlock);

                cell.appendChild(sideRow);
            } else {
                cell.appendChild(makeFlexAllocationsTable(playerData.flex_allocations));
            }
        } else if (playerData.auction_values) {
            // No position data but in auction mode: still show auction values alone.
            cell.appendChild(makePanelLabel('All auction values', '60px'));
            cell.appendChild(makeAuctionValuesTable(playerData));
        }

        if (playerData.roster) {
            cell.appendChild(makePanelLabel('Roster assignments', '60px'));
            cell.appendChild(makeRosterGrid(playerData.roster));
        }
    }
}

// ─── Panel section label ──────────────────────────────────────────────────────

/** Creates a `.panel-label` div with optional left-padding for indentation. */
function makePanelLabel(text: string, paddingLeft?: string): HTMLDivElement {
    let label = document.createElement('div');
    label.className = 'panel-label';
    label.textContent = text;
    if (paddingLeft) label.style.paddingLeft = paddingLeft;
    return label;
}

/** Creates an invisible `<th>` spacer used to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    let th = document.createElement('th');
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer';
    return th;
}

// ─── G-score expectations table ───────────────────────────────────────────────
// Rows: current diff, player contribution, future diff, total diff.
// Category cells: stat_styler_primary (middle=0, multiplier=60).
// Total column: styler_a / styler_b for the summary row.

/**
 * Builds the G-score expectations table for one candidate player.
 * Rows show how the player changes each category's head-to-head win expectation.
 */
function makeGScoreTable(playerData: Player, categories: string[]): HTMLDivElement {
    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';

    // Row 1: invisible spacers lock column widths.
    let thead = table.createTHead();
    let spacerRow = thead.insertRow(-1);
    spacerRow.style.border = 'none';
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name-sm'));
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'));
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh());

    // Row 2: visible column headers.
    let headerRow = thead.insertRow(-1);
    headerRow.appendChild(makeSpacerTh('panel-colheader-blank')); // hidden label cell
    let totalTh = document.createElement('th');
    totalTh.className = 'panel-colheader';
    totalTh.textContent = 'Total';
    headerRow.appendChild(totalTh);
    for (let cat of categories) {
        let th = document.createElement('th');
        th.className = 'panel-colheader';
        th.textContent = cat;
        headerRow.appendChild(th);
    }

    // Data rows
    let tbody = table.createTBody();
    for (let rowData of playerData.g_score_rows) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = 'panel-rowlabel';
        labelCell.textContent = rowData.label;
        row.appendChild(labelCell);

        // Total column: flat color, brighter for the summary row
        let totalCell = row.insertCell(-1);
        totalCell.textContent = rowData.total.toFixed(2);
        totalCell.className = 'panel-datacell';
        totalCell.className += rowData.isTotal ? ' celltypeb' : ' celltypea';

        for (let value of rowData.values) {
            let cell = row.insertCell(-1);
            cell.textContent = value.toFixed(2);
            cell.style.cssText = stat_styler_primary(value, 60, 0);
            cell.className = 'panel-datacell';
        }
    }

    let wrapper = document.createElement('div');
    wrapper.style.marginLeft = '100px';
    wrapper.appendChild(table);
    return wrapper;
}

// ─── Category weights table ───────────────────────────────────────────────────
// Single data row: algorithm weight assigned to each category for future picks.
// Values are percentages; '%' suffix is added by CSS .panel-weight::after.

/**
 * Builds the category weights table showing how the algorithm weights each
 * stat category when evaluating future picks after this candidate is drafted.
 */
function makeWeightsTable(playerData: Player, categories: string[]): HTMLDivElement {
    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    let headerRow = table.createTHead().insertRow(-1);
    headerRow.style.border = 'none';
    let emptyTh = document.createElement('th');
    emptyTh.className = 'panel-colspacer';
    emptyTh.style.width = '219px';
    headerRow.appendChild(emptyTh);
    for (let i = 0; i < categories.length; i++) headerRow.appendChild(makeSpacerTh());

    // Data row: row label + N category weight cells
    let row = table.createTBody().insertRow(-1);
    let labelCell = document.createElement('th');
    labelCell.className = 'panel-rowlabel';
    labelCell.textContent = 'Future pick weight';
    row.appendChild(labelCell);
    for (let value of playerData.category_weights) {
        let cell = row.insertCell(-1);
        cell.textContent = value.toFixed(0); // '%' appended by CSS panel-weight::after
        cell.style.cssText = stat_styler_tertiary(value, 5, 90);
        cell.className = 'panel-datacell panel-weight';
    }

    let wrapper = document.createElement('div');
    wrapper.style.marginLeft = '100px';
    wrapper.appendChild(table);
    return wrapper;
}

// ─── Flex position allocations table ──────────────────────────────────────────
// Rows: flex slot types (G, F, Util, Total); Cols: base positions (PG–C).
// Each cell is the expected number of times that base position type fills that flex slot.
// -999 means that base position is ineligible for that flex slot type.

/**
 * Builds the flex allocations table showing how future picks are expected to
 * fill remaining flex roster slots, given that this candidate is drafted.
 */
function makeFlexAllocationsTable(flexData: FlexAllocations, noMargin = false): HTMLDivElement {

    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    table.style.width = (110 + flexData.base_positions.length * 90) + 'px';

    let headerRow = table.createTHead().insertRow(-1);
    let emptyTh = document.createElement('th');
    emptyTh.className = 'panel-colspacer panel-colheader-blank';
    emptyTh.style.width = '110px';
    headerRow.appendChild(emptyTh);
    for (let pos of flexData.base_positions) {
        let th = document.createElement('th');
        th.className = 'panel-colheader';
        th.style.width = '90px';
        th.textContent = expandPosition(pos);
        headerRow.appendChild(th);
    }

    let tbody = table.createTBody();
    for (let rowData of flexData.rows) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = rowData.isTotal ? 'panel-rowlabel panel-rowlabel-total' : 'panel-rowlabel';
        labelCell.textContent = expandFlexLabel(rowData.label);
        row.appendChild(labelCell);

        for (let value of rowData.values) {
            let cell = row.insertCell(-1);
            if (value === null) {
                cell.className = 'ineligible panel-datacell';
            } else {
                cell.className = 'panel-datacell';
                cell.textContent = value.toFixed(2);
                cell.style.cssText = stat_styler_tertiary(value, 50, 0);
            }
        }
    }

    let wrapper = document.createElement('div');
    if (!noMargin) wrapper.style.marginLeft = '100px';
    wrapper.appendChild(table);
    return wrapper;
}

// ─── Auction values table ─────────────────────────────────────────────────────
// Two rows (H-score, G-score) × three columns (Your $, Gnrc. $, Orig. $).
// The H-score row mirrors the values shown in the main candidate table.
// The G-score row shows the same dollar columns computed from G-scores instead;
// Your $ has no G-score equivalent and is shown as a dash.

/** Builds the auction values detail table: H-score and G-score rows × Your $ / Gnrc. $ / Orig. $ columns. */
function makeAuctionValuesTable(playerData: Player): HTMLTableElement {
    const av = playerData.auction_values!;

    const table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';

    // Header
    const headerRow = table.createTHead().insertRow(-1);
    const labelTh = document.createElement('th');
    labelTh.className = 'panel-colspacer';
    labelTh.style.width = '72px';
    headerRow.appendChild(labelTh);
    for (const col of ['Your $', 'Gnrc. $', 'Orig. $']) {
        const th = document.createElement('th');
        th.className = 'panel-colheader';
        th.style.width = '72px';
        th.textContent = col;
        headerRow.appendChild(th);
    }

    const tbody = table.createTBody();

    // H-score row
    const hRow = tbody.insertRow(-1);
    const hLabel = document.createElement('th');
    hLabel.className = 'panel-rowlabel';
    hLabel.textContent = 'H-score';
    hRow.appendChild(hLabel);
    for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
        const cell = hRow.insertCell(-1);
        cell.className = 'panel-datacell celltypea';
        cell.textContent = val.toFixed(1);
    }

    // G-score row (Your $ is not applicable for G-scores)
    const gRow = tbody.insertRow(-1);
    const gLabel = document.createElement('th');
    gLabel.className = 'panel-rowlabel';
    gLabel.textContent = 'G-score';
    gRow.appendChild(gLabel);
    const naCell = gRow.insertCell(-1);
    naCell.className = 'panel-datacell celltypea';
    naCell.textContent = '—';
    for (const val of [av.gnrc_dollar_g, av.orig_dollar_g]) {
        const cell = gRow.insertCell(-1);
        cell.className = 'panel-datacell celltypea';
        cell.textContent = val.toFixed(1);
    }

    return table;
}

// ─── Roster grid ─────────────────────────────────────────────────────────────
// Rows: depth level (1st slot, 2nd slot, …); Cols: all position types (PG…Util).
// Base-position cols align with the flex allocations table above.

/**
 * Builds the roster grid showing which players are assigned to each slot,
 * and highlights the candidate player being evaluated.
 */
function makeRosterGrid(roster: Roster): HTMLDivElement {

    console.log(roster)

    // Derive ordered position types by stripping trailing digits (e.g. "PG1" → "PG")
    let posTypes: string[] = [];
    let groups: Record<string, string[]> = {};
    for (let slot of roster.slots) {
        let type = slot.replace(/\d+$/, '');
        if (!groups[type]) { groups[type] = []; posTypes.push(type); }
        groups[type].push(slot);
    }
    let maxDepth = Math.max(...posTypes.map(t => groups[t].length));

    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    table.style.width = (90 + posTypes.length * 90) + 'px';

    let headerRow = table.createTHead().insertRow(-1);
    let emptyTh = document.createElement('th');
    emptyTh.className = 'panel-colspacer panel-colheader-blank';
    emptyTh.style.width = '90px';
    headerRow.appendChild(emptyTh);
    for (let type of posTypes) {
        let th = document.createElement('th');
        th.className = 'panel-colheader';
        th.style.width = '90px';
        th.textContent = expandPosition(type);
        headerRow.appendChild(th);
    }

    let tbody = table.createTBody();
    for (let d = 0; d < maxDepth; d++) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = 'panel-rowlabel';
        labelCell.textContent = 'Slot ' + (d + 1);
        row.appendChild(labelCell);

        for (let type of posTypes) {
            let cell = row.insertCell(-1);
            let slot = groups[type][d];
            if (slot === undefined) {
                cell.className = 'ineligible panel-datacell';
                cell.textContent = '\u00A0';
            } else {
                let assignment = roster.assignments[slot];
                if (!assignment) {
                    cell.className = 'rosterabsent panel-datacell';
                    cell.textContent = '\u00A0';
                } else if (assignment.isCandidate) {
                    cell.className = 'rostercandidate panel-datacell';
                    cell.textContent = assignment.name;
                } else {
                    cell.className = 'rosteronteam panel-datacell';
                    cell.textContent = assignment.name;
                }
            }
        }
    }

    let wrapper = document.createElement('div');
    wrapper.style.marginLeft = '100px';
    wrapper.appendChild(table);
    return wrapper;
}
