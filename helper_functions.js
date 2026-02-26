import { stat_styler_primary, stat_styler_tertiary, styler_a, styler_b,
         styler_ineligible, styler_absent, styler_candidate, styler_rostered } from './styler_functions.js'

/**
 * Toggle the expanded detail panel for a player row.
 * Shows G-score expectations, category weights, position allocations, and roster assignments.
 *
 * @param {number}   playerIndex - Index of the player row in the table
 * @param {object}   playerData  - Full player data object (see script.js)
 * @param {string[]} categories  - Ordered list of category names
 */
export function ExpandView(playerIndex, playerData, categories) {
    const totalCols = categories.length + 2; // player th + H-score + N category cols

    let evpopup = document.querySelector(`#PP${playerIndex}.playerpopup`);
    let expandedRow = document.querySelector(`.EV${playerIndex}.expandedview`);

    if (expandedRow.style.display === 'table-row') {
        // Collapse: clear content and hide
        expandedRow.style.display = 'none';
        expandedRow.innerHTML = '';
        evpopup.textContent = '▼';
    } else {
        // Expand: build the detail panel as a vertical stack
        expandedRow.style.display = 'table-row';
        evpopup.textContent = '▲';

        // Single cell spanning all columns
        let cell = expandedRow.insertCell(-1);
        cell.colSpan = totalCols;
        cell.className = 'panel-cell';

        cell.appendChild(makeLabel('G-score expectations (difference vs. other teams)', '60px'));
        cell.appendChild(makeGScoreTable(playerData, categories));

        cell.appendChild(makeLabel('Category strategy', '60px'));
        cell.appendChild(makeWeightsTable(playerData, categories));

        cell.appendChild(makeLabel('Flex position allocations for future flex spot picks', '60px'));
        cell.appendChild(makeFlexAllocationsTable(playerData));

        cell.appendChild(makeLabel('Roster assignments', '60px'));
        cell.appendChild(makeRosterGrid(playerData));
    }
}

// ─── Helper: section label ───────────────────────────────────────────────────

/**
 * @param {string}  text          - Label text
 * @param {string}  [paddingLeft] - Optional left padding override (e.g. '60px')
 * @returns {HTMLDivElement}
 */
function makeLabel(text, paddingLeft) {
    let label = document.createElement('div');
    label.className = 'panel-label';
    label.textContent = text;
    if (paddingLeft) label.style.paddingLeft = paddingLeft;
    return label;
}

// ─── Helper: invisible spacer <th> to lock column width ──────────────────────

/**
 * @param {string} [width] - Optional explicit width (e.g. '136px')
 * @returns {HTMLTableCellElement}
 */
function makeSpacerTh(width) {
    let th = document.createElement('th');
    th.className = 'panel-colspacer';
    if (width) th.style.width = width;
    return th;
}


// ─── G-score expectations table ───────────────────────────────────────────────
// Rows: current diff, player, future diff, total diff
// Category cells: stat_styler_primary (middle=0, multiplier=60)
// Total cell: styler_a (flat dark), styler_b for the isTotal row

/**
 * @param {object}   playerData - Player data with g_score_rows array
 * @param {string[]} categories - Category names
 * @returns {HTMLDivElement}
 */
function makeGScoreTable(playerData, categories) {
    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';

    // Row 1: invisible spacers lock column widths (no border/padding so content-box = border-box).
    // 100px wrapper + 136px + 83px = 319px ≈ main table category start (player 224+10+2=236,
    // H-score 72+10+2=84, total=320px). Spacer th widths are content-box=border-box since
    // they have no padding/border, unlike the main table th which adds padding to its width.
    let thead = table.createTHead();
    let spacerRow = thead.insertRow(-1);
    spacerRow.style.border = 'none';
    spacerRow.appendChild(makeSpacerTh('136px'));
    spacerRow.appendChild(makeSpacerTh('83px'));
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh());

    // Row 2: visible column headers — widths inherited from spacer row above.
    let headerRow = thead.insertRow(-1);
    headerRow.appendChild(makeSpacerTh()); // invisible label cell
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

        // Total column (second): flat color, brighter for the summary row
        let totalCell = row.insertCell(-1);
        totalCell.textContent = rowData.total.toFixed(2);
        totalCell.style.cssText = rowData.isTotal ? styler_b() : styler_a();
        totalCell.className = 'panel-datacell';

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
// Single row: weight per category (blue shades, middle=90 since values are ×100)
// The '%' suffix is added via the panel-weight CSS class (::after rule).

/**
 * @param {object}   playerData - Player data with category_weights array
 * @param {string[]} categories - Category names
 * @returns {HTMLDivElement}
 */
function makeWeightsTable(playerData, categories) {
    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    // No explicit table width — inherits 100% from .panel-table so category columns
    // auto-distribute to match the G-score table and main table widths.
    let headerRow = table.createTHead().insertRow(-1);
    headerRow.style.border = 'none'; // suppress global tr border on this invisible spacer row
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
// Rows: slot types (G-1, F-2, Util-3, Total); Cols: base positions (PG–C)
// Explicit table width = 90 + N×72 px so column widths are exact (not scaled by width:100%)

/**
 * @param {object} playerData - Player data with flex_allocations object
 * @returns {HTMLDivElement}
 */
function makeFlexAllocationsTable(playerData) {
    let flexData = playerData.flex_allocations;

    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    table.style.width = (90 + flexData.base_positions.length * 72) + 'px';

    let headerRow = table.createTHead().insertRow(-1);
    let emptyTh = document.createElement('th');
    emptyTh.className = 'panel-colspacer';
    emptyTh.style.width = '90px';
    headerRow.appendChild(emptyTh);
    for (let pos of flexData.base_positions) {
        let th = document.createElement('th');
        th.className = 'panel-colheader';
        th.style.width = '72px';
        th.textContent = pos;
        headerRow.appendChild(th);
    }

    let tbody = table.createTBody();
    for (let rowData of flexData.rows) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = rowData.isTotal ? 'panel-rowlabel panel-rowlabel-total' : 'panel-rowlabel';
        labelCell.textContent = rowData.label;
        row.appendChild(labelCell);

        for (let value of rowData.values) {
            let cell = row.insertCell(-1);
            cell.className = 'panel-datacell';
            if (value === -999) {
                cell.style.cssText = styler_ineligible();
            } else {
                cell.textContent = value.toFixed(2);
                cell.style.cssText = stat_styler_tertiary(value, 150, 0);
            }
        }
    }

    let wrapper = document.createElement('div');
    wrapper.style.marginLeft = '100px';
    wrapper.appendChild(table);
    return wrapper;
}

// ─── Roster grid ─────────────────────────────────────────────────────────────
// Rows = depth level (1st slot, 2nd slot, ...); Cols = all position types (PG…Util)
// Explicit table width = 90 + N×72 px. Base-position cols (PG–C) align with flex table above.

/**
 * @param {object} playerData - Player data with roster object
 * @returns {HTMLDivElement}
 */
function makeRosterGrid(playerData) {
    let roster = playerData.roster;

    // Derive ordered position types by stripping trailing digits (e.g. "PG1" → "PG")
    let posTypes = [];
    let groups = {};
    for (let slot of roster.slots) {
        let type = slot.replace(/\d+$/, '');
        if (!groups[type]) { groups[type] = []; posTypes.push(type); }
        groups[type].push(slot);
    }
    let maxDepth = Math.max(...posTypes.map(t => groups[t].length));

    let table = document.createElement('table');
    table.className = 'panel-table';
    table.style.tableLayout = 'fixed';
    table.style.width = (90 + posTypes.length * 72) + 'px';

    let headerRow = table.createTHead().insertRow(-1);
    let emptyTh = document.createElement('th');
    emptyTh.className = 'panel-colspacer';
    emptyTh.style.width = '90px';
    headerRow.appendChild(emptyTh);
    for (let type of posTypes) {
        let th = document.createElement('th');
        th.className = 'panel-colheader';
        th.style.width = '72px';
        th.textContent = type;
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
            cell.className = 'panel-datacell';
            let slot = groups[type][d];
            if (slot === undefined) {
                cell.style.cssText = styler_ineligible();
                cell.textContent = '\u00A0';
            } else {
                let assignment = roster.assignments[slot];
                if (!assignment) {
                    cell.style.cssText = styler_absent();
                    cell.textContent = '\u00A0';
                } else if (assignment.isCandidate) {
                    cell.style.cssText = styler_candidate();
                    cell.textContent = assignment.name;
                } else {
                    cell.style.cssText = styler_rostered();
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
