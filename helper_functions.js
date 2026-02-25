import { stat_styler_primary, stat_styler_tertiary, styler_a, styler_b } from './styler_functions.js'

/**
 * Toggle the expanded detail view for a player row.
 * Shows G-score expectations, category weights, position allocations, and rank info.
 *
 * @param {number} playerIndex - Index of the player row in the table
 * @param {object} playerData  - Full player data object (see script.js)
 * @param {string[]} categories - Ordered list of category names
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
        cell.className = 'expansion-cell';

        // --- G-SCORE EXPECTATIONS ---
        cell.appendChild(makeLabel('G-score expectations (difference vs. other teams)'));
        cell.appendChild(makeGScoreTable(playerData, categories));

        // --- CATEGORY WEIGHTS ---
        cell.appendChild(makeLabel('Category weights for future picks'));
        cell.appendChild(makeWeightsTable(playerData, categories));

        // --- FLEX POSITION ALLOCATIONS ---
        cell.appendChild(makeLabel('Flex position allocations for future flex spot picks'));
        cell.appendChild(makeFlexAllocationsTable(playerData));

        // --- ROSTER ASSIGNMENTS ---
        cell.appendChild(makeLabel('Roster assignments for chosen players'));
        cell.appendChild(makeRosterTable(playerData));

    }
}

// ─── Helper: section label ───────────────────────────────────────────────────

function makeLabel(text) {
    let label = document.createElement('div');
    label.className = 'expansion-label';
    label.textContent = text;
    return label;
}

// ─── Helper: append a <th> to a row ──────────────────────────────────────────

function appendTh(row, text, className) {
    let th = document.createElement('th');
    th.className = className;
    th.textContent = text;
    row.appendChild(th);
}

// ─── G-score expectations table ───────────────────────────────────────────────
// Rows: current diff, player, future diff, total diff
// Category cells: stat_styler_primary (middle=0, multiplier=60)
// Total cell: styler_a (flat dark), styler_b for the isTotal row

function makeGScoreTable(playerData, categories) {
    let table = document.createElement('table');
    table.className = 'expansion-table';

    // Header
    let headerRow = table.createTHead().insertRow(-1);
    appendTh(headerRow, '', 'exp-rowlabel');
    for (let cat of categories) appendTh(headerRow, cat, 'exp-colheader');
    appendTh(headerRow, 'Total', 'exp-colheader');

    // Data rows
    let tbody = table.createTBody();
    for (let rowData of playerData.g_score_rows) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = 'exp-rowlabel';
        labelCell.textContent = rowData.label;
        row.appendChild(labelCell);

        for (let value of rowData.values) {
            let cell = row.insertCell(-1);
            cell.textContent = value.toFixed(2);
            cell.style.cssText = stat_styler_primary(value, 60, 0);
            cell.className = 'exp-datacell';
        }

        // Total column: flat color, brighter for the summary row
        let totalCell = row.insertCell(-1);
        totalCell.textContent = rowData.total.toFixed(2);
        totalCell.style.cssText = rowData.isTotal ? styler_b() : styler_a();
        totalCell.className = 'exp-datacell';
    }

    return table;
}

// ─── Category weights table ───────────────────────────────────────────────────
// Single row: weight per category (blue shades, middle=90 since values are ×100)

function makeWeightsTable(playerData, categories) {
    let table = document.createElement('table');
    table.className = 'expansion-table';

    // Header
    let headerRow = table.createTHead().insertRow(-1);
    for (let cat of categories) appendTh(headerRow, cat, 'exp-colheader');

    // Data row
    let row = table.createTBody().insertRow(-1);

    for (let value of playerData.category_weights) {
        let cell = row.insertCell(-1);
        cell.textContent = value.toFixed(0) + '%';
        cell.style.cssText = stat_styler_tertiary(value, 5, 90);
        cell.className = 'exp-datacell';
    }

    return table;
}

// ─── Flex position allocations table ──────────────────────────────────────────
// Rows: flex slot types (G-1, F-2, Util-3, ...) + Total
// Columns: base positions (PG, SG, SF, PF, C)
// Values: expected number of that slot type used by this player
// -999 encodes "not eligible" → shown as hidden grey cell

function makeFlexAllocationsTable(playerData) {
    let table = document.createElement('table');
    table.className = 'expansion-table';

    let flexData = playerData.flex_allocations;

    // Header
    let headerRow = table.createTHead().insertRow(-1);
    appendTh(headerRow, '', 'exp-rowlabel');
    for (let pos of flexData.base_positions) appendTh(headerRow, pos, 'exp-colheader');

    // Data rows
    let tbody = table.createTBody();
    for (let rowData of flexData.rows) {
        let row = tbody.insertRow(-1);

        let labelCell = document.createElement('th');
        labelCell.className = rowData.isTotal ? 'exp-rowlabel exp-rowlabel-total' : 'exp-rowlabel';
        labelCell.textContent = rowData.label;
        row.appendChild(labelCell);

        for (let value of rowData.values) {
            let cell = row.insertCell(-1);
            cell.className = 'exp-datacell';
            if (value === -999) {
                cell.style.cssText = 'background-color:#8D8D9E;color:#8D8D9E;';
            } else {
                cell.textContent = value.toFixed(2);
                cell.style.cssText = stat_styler_tertiary(value, 50, 0);
            }
        }
    }

    return table;
}

// ─── Roster assignments table ─────────────────────────────────────────────────
// Single transposed row: columns = position slots, values = player last names
// Existing players: dark blue. Candidate player: bright blue. Empty: hidden grey.

function makeRosterTable(playerData) {
    let table = document.createElement('table');
    table.className = 'expansion-table';

    let roster = playerData.roster;

    // Header row: slot names
    let headerRow = table.createTHead().insertRow(-1);
    appendTh(headerRow, '', 'exp-rowlabel');
    for (let slot of roster.slots) appendTh(headerRow, slot, 'exp-colheader');

    // Data row: player names
    let row = table.createTBody().insertRow(-1);

    let labelCell = document.createElement('th');
    labelCell.className = 'exp-rowlabel';
    labelCell.textContent = 'Players';
    row.appendChild(labelCell);

    for (let slot of roster.slots) {
        let cell = row.insertCell(-1);
        cell.className = 'exp-datacell';
        let assignment = roster.assignments[slot];

        if (!assignment) {
            // Empty slot — hide with grey
            cell.style.cssText = 'background-color:#555566;color:#555566;';
        } else if (assignment.isCandidate) {
            // Candidate player — bright blue
            cell.style.cssText = 'background-color:rgb(90,90,240);color:white;';
            cell.textContent = assignment.name;
        } else {
            // Existing player — dark blue
            cell.style.cssText = 'background-color:rgb(70,70,150);color:white;';
            cell.textContent = assignment.name;
        }
    }

    return table;
}
