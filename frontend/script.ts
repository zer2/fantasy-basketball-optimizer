// script.ts
// Application entry point. Sets up the sidebar, registers event listeners,
// initializes the layout, and triggers the first backend evaluation.
// All heavy logic lives in api/session.ts, table/player_table.ts, and layout.ts.

import { createSection, addApplyBtn } from './helper_functions.js'
import { getCandidatePlayers } from './app_state.js'
import { createOrPatchSession, runEvaluate } from './api/session.js'
import { buildTable } from './table/player_table.js'
import { initLayout, reapplyLayout } from './layout.js'
import { resetDraftBoard } from './data_entry/draft_board.js'
import { resetAuctionEntry } from './data_entry/auction_entry.js'

import { renderLeagueSettings, getLeagueSettings } from './parameter_collection/league_settings.js'
import { renderFormatAndCategories, getFormatAndCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts } from './parameter_collection/slot_counts.js'
import { renderTradeParameters } from './parameter_collection/trade_parameters.js'

// ─── Sidebar ──────────────────────────────────────────────────────────────────

const sidebar = document.getElementById('sidebar') as HTMLElement
const sidebarSections = document.getElementById('sidebar-sections')!

renderLeagueSettings(createSection(sidebarSections, 'League Settings'))
// League Settings auto-updates via listeners added after initLayout() below.

const playerStatsSection = createSection(sidebarSections, 'Player Stats')
renderPlayerStats(playerStatsSection)
addApplyBtn(playerStatsSection, async () => {
    const { data_source, injured_players } = getPlayerStatsParams()
    resetDraftBoard()
    resetAuctionEntry()
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

// ─── Mode change: rebuild table and sync session ───────────────────────────
// Registered before initLayout so buildTable fires before applyLayout on mode
// change, ensuring realtable.style.width is correct when applyLayout reads it.
//
// When switching to Auction Mode the existing session must be patched with
// cash_per_team; without it the backend cannot compute auction dollar values.
//
// Uses an AbortController so rapid mode switches cancel stale backend calls
// instead of letting them pile up.
let modeChangeController: AbortController | null = null
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    if (modeChangeController) modeChangeController.abort()
    modeChangeController = new AbortController()
    const { signal } = modeChangeController

    const { mode, cash_per_team } = getLeagueSettings()

    if (mode === 'Season Mode') return   // table is hidden in season mode; skip rebuild and backend call

    buildTable(getCandidatePlayers())

    const patch = mode === 'Auction Mode' ? { league: { cash_per_team } } : {}
    createOrPatchSession(4, patch, signal)
        .then(() => { if (!signal.aborted) return runEvaluate() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Mode change failed:', err)
        })
})

initLayout({ onEvaluate: runEvaluate })

// Initial build (empty; will be populated once the backend responds on load)
buildTable(getCandidatePlayers())

// Run all backend steps immediately on page load
runEvaluate().catch(err => console.error('Initial load failed:', err))

// ── League settings auto-update ───────────────────────────────────────────
// Number inputs fire on 'change' (when focus leaves); buildTable runs first to
// set realtable.style.width before reapplyLayout reads it.
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        buildTable(getCandidatePlayers())
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
        buildTable(getCandidatePlayers())
        reapplyLayout()
    }, 600)
})
