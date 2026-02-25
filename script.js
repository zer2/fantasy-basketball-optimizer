import { stat_styler_primary } from './styler_functions.js'
import { ExpandView } from './helper_functions.js'

const categories = ["FG%", "FT%", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]

// Player data. Each player has:
//   h_score            – overall H-score win rate (0–100 scale)
//   win_rates          – per-category win rates (0–100 scale, 50 = average)
//   category_weights   – algorithm's relative weighting for future picks (100 = baseline)
//   g_score_rows       – rows for the G-score expectations table
//   flex_allocations   – expected usage of remaining flex slots
//   roster             – position slot assignments for existing team + candidate
//   h_rank / g_rank    – rank among available players

const players = [
    {
        name: "Nikola Jokic (C)",
        h_score: 53.7,
        h_rank: 1,
        g_rank: 1,
        win_rates: [66.2, 14.2, 33.9, 66.3, 73.4, 72.3, 59.7, 67.7, 29.7],
        category_weights: [95, 83, 98, 114, 95, 102, 103, 111, 100],
        g_score_rows: [
            { label: 'Current diff', values: [ 0.42, -0.18, -0.31,  0.28,  0.61,  0.54,  0.12,  0.19, -0.38], total:  1.29, isTotal: false },
            { label: 'Jokic',        values: [ 1.80, -2.10, -0.90,  1.70,  2.40,  2.20,  0.50,  1.60, -1.30], total:  5.90, isTotal: false },
            { label: 'Future diff',  values: [ 0.31, -0.09, -0.22,  0.19,  0.47,  0.38,  0.09,  0.14, -0.27], total:  1.00, isTotal: false },
            { label: 'Total diff',   values: [ 2.53, -2.37, -1.43,  2.17,  3.48,  3.12,  0.71,  1.93, -1.95], total:  8.19, isTotal: true  },
        ],
        flex_allocations: {
            base_positions: ["PG", "SG", "SF", "PF", "C"],
            rows: [
                { label: "G-1",    values: [0.65,  0.35,  -999,  -999,  -999], isTotal: false },
                { label: "F-2",    values: [-999,  -999,  1.10,  0.90,  -999], isTotal: false },
                { label: "Util-3", values: [0.50,  0.40,  0.70,  0.60,  0.80], isTotal: false },
                { label: "Total",  values: [1.15,  0.75,  1.80,  1.50,  0.80], isTotal: true  },
            ]
        },
        roster: {
            slots: ["PG1", "SG1", "SF1", "PF1", "C1", "C2", "G1", "G2", "F1", "F2", "Util1", "Util2", "Util3"],
            assignments: {
                "PG1":   { name: "Curry",  isCandidate: false },
                "SG1":   null,
                "SF1":   { name: "Durant", isCandidate: false },
                "PF1":   null,
                "C1":    { name: "Jokic",  isCandidate: true  },
                "C2":    null,
                "G1":    { name: "Paul",   isCandidate: false },
                "G2":    null,
                "F1":    null,
                "F2":    null,
                "Util1": null,
                "Util2": null,
                "Util3": null,
            }
        }
    },
    {
        name: "Shai Gilgeous-Alexander (PG)",
        h_score: 53.0,
        h_rank: 2,
        g_rank: 2,
        win_rates: [40.8, 71.9, 65.4, 58.4, 10.8, 55.2, 59.1, 35.2, 58.2],
        category_weights: [88, 108, 112, 96, 72, 95, 118, 96, 98],
        g_score_rows: [
            { label: 'Current diff', values: [ 0.42, -0.18, -0.31,  0.28,  0.61,  0.54,  0.12,  0.19, -0.38], total:  1.29, isTotal: false },
            { label: 'SGA',          values: [-0.60,  1.80,  1.40,  0.80, -2.40,  0.50,  0.60,  0.50,  0.60], total:  3.20, isTotal: false },
            { label: 'Future diff',  values: [ 0.31, -0.09, -0.22,  0.19,  0.47,  0.38,  0.09,  0.14, -0.27], total:  1.00, isTotal: false },
            { label: 'Total diff',   values: [ 0.13,  1.53,  0.87,  1.27, -1.32,  1.42,  0.81,  0.83, -0.05], total:  5.49, isTotal: true  },
        ],
        flex_allocations: {
            base_positions: ["PG", "SG", "SF", "PF", "C"],
            rows: [
                { label: "G-1",    values: [0.40,  0.60,  -999,  -999,  -999], isTotal: false },
                { label: "F-2",    values: [-999,  -999,  1.20,  0.80,  -999], isTotal: false },
                { label: "Util-3", values: [0.60,  0.30,  0.70,  0.50,  0.90], isTotal: false },
                { label: "Total",  values: [1.00,  0.90,  1.90,  1.30,  0.90], isTotal: true  },
            ]
        },
        roster: {
            slots: ["PG1", "SG1", "SF1", "PF1", "C1", "C2", "G1", "G2", "F1", "F2", "Util1", "Util2", "Util3"],
            assignments: {
                "PG1":   { name: "Curry",  isCandidate: false },
                "SG1":   { name: "SGA",    isCandidate: true  },
                "SF1":   { name: "Durant", isCandidate: false },
                "PF1":   null,
                "C1":    null,
                "C2":    null,
                "G1":    { name: "Paul",   isCandidate: false },
                "G2":    null,
                "F1":    null,
                "F2":    null,
                "Util1": null,
                "Util2": null,
                "Util3": null,
            }
        }
    },
    {
        name: "Victor Wembanyama (C)",
        h_score: 52.1,
        h_rank: 3,
        g_rank: 3,
        win_rates: [51.3, 54.2, 66.2, 41.7, 57.4,  9.6, 39.2, 73.2, 76.2],
        category_weights: [102, 97, 114, 88, 108, 72, 85, 138, 112],
        g_score_rows: [
            { label: 'Current diff', values: [ 0.42, -0.18, -0.31,  0.28,  0.61,  0.54,  0.12,  0.19, -0.38], total:  1.29, isTotal: false },
            { label: 'Wembanyama',   values: [ 0.10,  0.20,  1.50, -0.60,  0.40, -2.60, -0.40,  2.00,  1.80], total:  2.40, isTotal: false },
            { label: 'Future diff',  values: [ 0.31, -0.09, -0.22,  0.19,  0.47,  0.38,  0.09,  0.14, -0.27], total:  1.00, isTotal: false },
            { label: 'Total diff',   values: [ 0.83, -0.07,  0.97, -0.13,  1.48, -1.68, -0.19,  2.33,  1.15], total:  4.69, isTotal: true  },
        ],
        flex_allocations: {
            base_positions: ["PG", "SG", "SF", "PF", "C"],
            rows: [
                { label: "G-1",    values: [0.65,  0.35,  -999,  -999,  -999], isTotal: false },
                { label: "F-2",    values: [-999,  -999,  1.10,  0.90,  -999], isTotal: false },
                { label: "Util-3", values: [0.40,  0.30,  0.60,  0.50,  1.20], isTotal: false },
                { label: "Total",  values: [1.05,  0.65,  1.70,  1.40,  1.20], isTotal: true  },
            ]
        },
        roster: {
            slots: ["PG1", "SG1", "SF1", "PF1", "C1", "C2", "G1", "G2", "F1", "F2", "Util1", "Util2", "Util3"],
            assignments: {
                "PG1":   { name: "Curry",      isCandidate: false },
                "SG1":   null,
                "SF1":   { name: "Durant",     isCandidate: false },
                "PF1":   null,
                "C1":    { name: "Wembanyama", isCandidate: true  },
                "C2":    null,
                "G1":    { name: "Paul",       isCandidate: false },
                "G2":    null,
                "F1":    null,
                "F2":    null,
                "Util1": null,
                "Util2": null,
                "Util3": null,
            }
        }
    }
]

// ─── Build the table ──────────────────────────────────────────────────────────

let table = document.getElementById('realtable')

// Header row
let header = table.createTHead()

// Player column — explicit width so table-layout:fixed locks it in
let playerHeaderCell = document.createElement('th')
playerHeaderCell.className = 'tableheader'
playerHeaderCell.textContent = 'Player'
playerHeaderCell.style.width = '220px'
header.append(playerHeaderCell)

// H-score column
let hscoreHeaderCell = document.createElement('th')
hscoreHeaderCell.className = 'tableheader'
hscoreHeaderCell.textContent = 'H-Score'
hscoreHeaderCell.style.width = '68px'
header.append(hscoreHeaderCell)

// Category columns (remaining width split equally)
for (let category of categories) {
    let catHeaderCell = document.createElement('th')
    catHeaderCell.className = 'tableheader'
    catHeaderCell.textContent = category
    header.append(catHeaderCell)
}

// Player rows
for (const [i, player] of players.entries()) {

    let row = table.insertRow(-1)

    // Player name cell with expand button
    let nameCell = document.createElement('th')
    nameCell.innerHTML = `
        <div class='playerheaderdiv'>
            <div style="width:80%">${player.name}</div>
            <div style="width:20%">
                <button class='playerpopup' id='PP${i}'>▼</button>
            </div>
        </div>`
    nameCell.className = 'playerheader'
    row.append(nameCell)

    let button = nameCell.querySelector(`#PP${i}.playerpopup`)
    button.addEventListener('click', () => ExpandView(i, player, categories))

    // H-score cell
    let hscoreCell = row.insertCell(-1)
    hscoreCell.className = 'overallhscore'
    hscoreCell.textContent = player.h_score.toFixed(1)

    // Category win rate cells
    for (let value of player.win_rates) {
        let cell = row.insertCell(-1)
        cell.textContent = value.toFixed(1)
        cell.style.cssText += stat_styler_primary(value, 5, 50)
        cell.className = 'categoricalhscore'
    }

    // Expansion row (hidden until button clicked)
    let expandedRow = table.insertRow(-1)
    expandedRow.className = `expandedview EV${i}`
    expandedRow.style.display = 'none'
}
