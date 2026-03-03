import { stat_styler_primary, stat_styler_secondary, styler_a } from './styler_functions.js'
import { ExpandView } from './expand_view.js'
import { Player, SessionRequest } from './types.js'
import { renderLeagueSettings, getLeagueSettings } from './parameter_collection/league_settings.js'
import { renderFormatAndCategories, getFormatAndCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts } from './parameter_collection/slot_counts.js'
import { renderTradeParameters } from './parameter_collection/trade_parameters.js'
import { initLayout, reapplyLayout, getCurrentSeat } from './layout.js'
import { getDraftState } from './data_entry/draft_board.js'
import { getAuctionState } from './data_entry/auction_entry.js'
import * as api from './api.js'

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

/**
 * Appends a small "Apply" button to a sidebar section content div.
 * The callback may be sync or async; errors are caught and logged.
 */
function addApplyBtn(container: HTMLElement, onClick: () => void | Promise<void>): void {
    const btn = document.createElement('button')
    btn.className   = 'section-apply-btn'
    btn.textContent = 'Apply'
    btn.addEventListener('click', () => {
        const result = onClick()
        if (result instanceof Promise) {
            btn.disabled = true
            result.finally(() => { btn.disabled = false })
                  .catch(err => console.error('Apply failed:', err))
        }
    })
    container.append(btn)
}

renderLeagueSettings(createSection(sidebarSections, 'League Settings'))
// League Settings auto-updates via listeners added after initLayout() below.

const playerStatsSection = createSection(sidebarSections, 'Player Stats')
renderPlayerStats(playerStatsSection)
addApplyBtn(playerStatsSection, async () => {
    const { data_source, injured_players } = getPlayerStatsParams()
    await createOrPatchSession(1, { data_source, injured_players })
    await runEvaluate()
})

const formatSection = createSection(sidebarSections, 'Format & Categories')
renderFormatAndCategories(formatSection)
addApplyBtn(formatSection, async () => {
    const { scoring_format, categories: cats } = getFormatAndCategories()
    await createOrPatchSession(4, { league: { scoring_format, categories: cats } })
    await runEvaluate()
})

const modelSection = createSection(sidebarSections, 'Model Parameters')
renderModelParameters(modelSection)
addApplyBtn(modelSection, async () => {
    const parameters = getModelParameters()
    await createOrPatchSession(3, { parameters })
    await runEvaluate()
})

const slotSection = createSection(sidebarSections, 'Position Parameters')
renderSlotCounts(slotSection)
addApplyBtn(slotSection, async () => {
    const slot_counts = getSlotCounts()
    await createOrPatchSession(4, { slot_counts })
    await runEvaluate()
})

const tradeSection = createSection(sidebarSections, 'Trade Parameters')
renderTradeParameters(tradeSection)
// Trade parameters target a future endpoint; no backend call yet.
addApplyBtn(tradeSection, () => {})

// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

// ─── Session management ───────────────────────────────────────────────────────

let sessionId: string | null = null

/**
 * Collects all sidebar parameter values and assembles a `SessionRequest` object
 * ready to POST to `/sessions`.
 */
export function buildSessionRequest(): SessionRequest {
    const { sport, platform, mode, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
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
        // my_team_id comes from the seat selector in the main content area
    }
}

/**
 * Creates a new session if none exists, or patches the existing one starting from
 * `fromStep` with the given partial parameter body.
 */
async function createOrPatchSession(
    fromStep: number,
    patchBody: Record<string, unknown> = {},
): Promise<void> {
    if (!sessionId) {
        const req = buildSessionRequest()
        const resp = await api.createSession(req)
        sessionId = resp.session_id
        categories = resp.categories
    } else {
        await api.patchSession(sessionId, { from_step: fromStep, ...patchBody })
    }
}

/**
 * Ensures a session exists (creating one from current sidebar state if not).
 * Called before evaluate when the session may not have been explicitly created yet.
 */
async function ensureSession(): Promise<void> {
    if (sessionId) return
    const req = buildSessionRequest()
    const resp = await api.createSession(req)
    sessionId = resp.session_id
    categories = resp.categories
}

/**
 * Runs the evaluate endpoint with the current draft / auction state and
 * updates the candidate table with the response.
 */
export async function runEvaluate(): Promise<void> {
    await ensureSession()

    // Default to the first team name if no seat has been chosen yet
    const seat = getCurrentSeat() || getLeagueSettings().team_names[0] || 'Drafter 1'
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value

    let evalReq: Parameters<typeof api.evaluate>[1]
    if (mode === 'Auction Mode') {
        const { player_assignments, remaining_cash } = getAuctionState()
        evalReq = { player_assignments, my_team_id: seat, remaining_cash }
    } else {
        const { player_assignments } = getDraftState()
        evalReq = { player_assignments, my_team_id: seat }
    }

    const resp = await api.evaluate(sessionId!, evalReq)
    updateTable(api.candidatesToPlayers(resp.candidates))
    reapplyLayout()
}

// ─── Player table ─────────────────────────────────────────────────────────────

let categories: string[] = ["Field Goal %", "Free Throw %", "Threes", "Points", "Rebounds", "Assists", "Steals", "Blocks", "Turnovers"]

let players: Player[] = []

export function getPlayers():    Player[] { return players }
export function getCategories(): string[] { return categories }

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
            diffCell.style.cssText = stat_styler_secondary(diff, 10, 0)
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
            cell.style.cssText = stat_styler_primary(value, 3, 50)
            cell.className = 'categoricalhscore'
        }

        // Expansion row (hidden until button clicked)
        const expandedRow = table.insertRow(-1)
        expandedRow.className = `expandedview EV${i}`
        expandedRow.style.display = 'none'
    }
}

// Initial build (empty; will be populated once the backend responds on load)
buildTable()

// ── Mode change: rebuild table ────────────────────────────────────────────────
// Registered before initLayout so updateTable fires before applyLayout on mode
// change, ensuring realtable.style.width is correct when applyLayout reads it.
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    updateTable(players, categories)
})

initLayout({ onEvaluate: runEvaluate })

// Run all backend steps immediately on page load
runEvaluate().catch(err => console.error('Initial load failed:', err))

// ── League settings auto-update ───────────────────────────────────────────────
// Number inputs fire on 'change' (when focus leaves); updateTable runs first to
// set realtable.style.width before reapplyLayout reads it.
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        updateTable(players, categories)
        const { n_drafters, n_picks, cash_per_team } = getLeagueSettings()
        createOrPatchSession(4, { league: { n_drafters, n_picks, cash_per_team } })
            .then(() => runEvaluate())
            .then(() => reapplyLayout())
            .catch(err => console.error('League settings update failed:', err))
    })
}

// Team names debounce: wait 600 ms after last keystroke to avoid flicker while typing.
let teamNamesTimer: ReturnType<typeof setTimeout> | null = null
document.getElementById('ls-team-names')!.addEventListener('input', () => {
    if (teamNamesTimer) clearTimeout(teamNamesTimer)
    teamNamesTimer = setTimeout(() => {
        updateTable(players, categories)
        reapplyLayout()
    }, 600)
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
