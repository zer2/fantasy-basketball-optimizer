import { stat_styler_primary, stat_styler_secondary, styler_a } from './styler_functions.js'
import { ExpandView } from './expand_view.js'
import { Player, SessionRequest } from './types.js'
import { renderLeagueSettings, getLeagueSettings } from './parameter_collection/league_settings.js'
import { renderFormatAndCategories, getFormatAndCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts } from './parameter_collection/slot_counts.js'
import { renderTradeParameters } from './parameter_collection/trade_parameters.js'

// ─── Sidebar ──────────────────────────────────────────────────────────────────

/**
 * Creates a collapsible `<details>` sidebar section and returns its content div.
 * The returned element is the container that `render*` functions should populate.
 */
function createSection(parent: HTMLElement, title: string): HTMLElement {
    const details = document.createElement('details')
    details.className = 'sidebar-section'
    const summary = document.createElement('summary')
    summary.textContent = title
    details.append(summary)
    const content = document.createElement('div')
    content.className = 'sidebar-section-content'
    details.append(content)
    parent.append(details)
    return content
}

const sidebar = document.getElementById('sidebar') as HTMLElement
const sidebarSections = document.getElementById('sidebar-sections')!
renderLeagueSettings(createSection(sidebarSections, 'League Settings'))
renderPlayerStats(createSection(sidebarSections, 'Player Stats'))
renderFormatAndCategories(createSection(sidebarSections, 'Format & Categories'))
renderModelParameters(createSection(sidebarSections, 'Model Parameters'))
renderSlotCounts(createSection(sidebarSections, 'Position Parameters'))
renderTradeParameters(createSection(sidebarSections, 'Trade Parameters'))
// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

/**
 * Collects all sidebar parameter values and assembles a `SessionRequest` object
 * ready to POST to `/sessions`.
 */
export function buildSessionRequest(): SessionRequest {
    const { sport, platform, mode, n_drafters, n_picks, cash_per_team, my_team_id } = getLeagueSettings()
    const { scoring_format, categories } = getFormatAndCategories()
    const { data_source, injured_players } = getPlayerStatsParams()
    const league: SessionRequest['league'] = { sport, n_drafters, n_picks, scoring_format, categories }
    if (mode === 'Auction Mode') league.cash_per_team = cash_per_team
    return {
        league,
        platform,
        slot_counts: getSlotCounts(),
        parameters: getModelParameters(),
        data_source,
        injured_players,
        my_team_id,
    }
}

// ─── Player table ─────────────────────────────────────────────────────────────

let categories: string[] = ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]

let players: Player[] = [
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
        },
        auction_values: { your_dollar: 52, gnrc_dollar: 48, orig_dollar: 50 },
    },
    {
        name: "Shai Gilgeous-Alexander (PG)",
        h_score: 53.0,
        h_rank: 2,
        g_rank: 2,
        win_rates: [40.8, 71.9, 65.4, 58.4, 10.8, 55.2, 59.1, 35.2, 58.2],
        category_weights: [103, 95, 103, 103, 79, 105, 101, 107, 103],
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
        },
        auction_values: { your_dollar: 45, gnrc_dollar: 43, orig_dollar: 44 },
    },
    {
        name: "Victor Wembanyama (C)",
        h_score: 52.1,
        h_rank: 3,
        g_rank: 3,
        win_rates: [51.3, 54.2, 66.2, 41.7, 57.4,  9.6, 39.2, 73.2, 76.2],
        category_weights: [111, 105, 107, 102, 124, 67, 85, 101, 98],
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
        },
        auction_values: { your_dollar: 38, gnrc_dollar: 41, orig_dollar: 39 },
    }
]

// ─── Build the table ──────────────────────────────────────────────────────────

const table = document.getElementById('realtable') as HTMLTableElement

function buildTable(): void {
    const isAuction = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode'

    // 1 (player) + score cols + N categories
    const PLAYER_COL_W = 224
    const SCORE_COL_W  = 72
    const scoreCols    = isAuction ? 4 : 1  // Diff / Your $ / Gnrc. $ / Orig. $  vs  H-Score

    // Keep category column widths consistent across modes by widening the table
    // to compensate for the extra auction columns (otherwise they'd wrap).
    const CAT_COL_W    = 90   // desired minimum width per category column
    const tableWidth   = PLAYER_COL_W + scoreCols * SCORE_COL_W + categories.length * CAT_COL_W
    table.style.width  = tableWidth + 'px'

    const totalCols = 1 + scoreCols + categories.length

    table.innerHTML = ''

    // ── Header ──────────────────────────────────────────────────────────────
    const thead = table.createTHead()
    const headerRow = thead.insertRow()

    const playerTh = document.createElement('th')
    playerTh.className = 'tableheader'
    playerTh.textContent = 'Player'
    playerTh.style.width = '224px'
    headerRow.append(playerTh)

    if (isAuction) {
        for (const label of ['Diff.', 'Your $', 'Gnrc. $', 'Orig. $']) {
            const th = document.createElement('th')
            th.className = 'tableheader'
            th.textContent = label
            th.style.width = '72px'
            headerRow.append(th)
        }
    } else {
        const th = document.createElement('th')
        th.className = 'tableheader'
        th.textContent = 'H-Score'
        th.style.width = '72px'
        headerRow.append(th)
    }

    for (const category of categories) {
        const th = document.createElement('th')
        th.className = 'tableheader'
        th.textContent = category
        headerRow.append(th)
    }

    // ── Player rows ─────────────────────────────────────────────────────────
    for (const [i, player] of players.entries()) {
        const row = table.insertRow(-1)

        // Player name cell with expand button
        const nameCell = document.createElement('th')
        nameCell.innerHTML = `
            <div class='playerheaderdiv'>
                <div style="width:80%">${player.name}</div>
                <div style="width:20%">
                    <button class='playerpopup' id='PP${i}'>▶</button>
                </div>
            </div>`
        nameCell.className = 'playerheader'
        row.append(nameCell)

        const button = nameCell.querySelector(`#PP${i}.playerpopup`) as HTMLButtonElement
        button.addEventListener('click', () => ExpandView(i, player, categories, totalCols))

        // Score column(s)
        if (isAuction) {
            const av = player.auction_values!
            const diff = av.your_dollar - av.gnrc_dollar

            const diffCell = row.insertCell(-1)
            diffCell.textContent = diff.toFixed(1)
            diffCell.style.cssText = stat_styler_secondary(diff, 6, 0)
            diffCell.className = 'auction-dollar'

            for (const val of [av.your_dollar, av.gnrc_dollar, av.orig_dollar]) {
                const cell = row.insertCell(-1)
                cell.textContent = String(val.toFixed(1))
                cell.style.cssText = styler_a()
                cell.className = 'auction-dollar'
            }
        } else {
            const hscoreCell = row.insertCell(-1)
            hscoreCell.className = 'overallhscore'
            hscoreCell.textContent = player.h_score.toFixed(1)
        }

        // Category win rate cells
        for (const value of player.win_rates) {
            const cell = row.insertCell(-1)
            cell.textContent = value.toFixed(1)
            cell.style.cssText = stat_styler_primary(value, 5, 50)
            cell.className = 'categoricalhscore'
        }

        // Expansion row (hidden until button clicked)
        const expandedRow = table.insertRow(-1)
        expandedRow.className = `expandedview EV${i}`
        expandedRow.style.display = 'none'
    }
}

// Initial build
buildTable()

// ── DEV ONLY: mock backend response on mode change ────────────────────────────
// Remove once the real backend call is wired up to the mode change event.
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    updateTable(players, categories)
})

/**
 * Updates the player data and rebuilds the table.
 * Call this whenever the backend returns a new set of results.
 * Reads the current mode from the DOM, so the layout stays in sync automatically.
 *
 * @param newPlayers    - Full player list from the backend response
 * @param newCategories - Category list; omit if unchanged (e.g. same session)
 */
export function updateTable(newPlayers: Player[], newCategories?: string[]): void {
    players = newPlayers
    if (newCategories) categories = newCategories
    buildTable()
}
