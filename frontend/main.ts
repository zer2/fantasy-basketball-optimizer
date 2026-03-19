// main.ts
// Application entry point. Sets up the sidebar, registers event listeners,
// initializes the layout, and triggers the first backend evaluation.
// All heavy logic lives in api/session.ts, table/player_table.ts, and layout.ts.

import { createSection, addApplyBtn, makeSidebarToggle } from './helper_functions.js'
import { getCandidatePlayers, setSportConfig, getCurrentSeat, setCurrentSeat } from './app_state.js'
import { createOrPatchSession, runEvaluate, runSeasonInit } from './api/session.js'
import { fetchConfig } from './api/client.js'
import { buildTable } from './table/player_table.js'
import { applyLayout, getCurrentAuctionTab, getCurrentDraftTab, refreshAuctionGScore, refreshDraftGScore } from './layout.js'
import { resetDraftBoard } from './data_entry/draft_board.js'
import { resetAuctionEntry } from './data_entry/auction_entry.js'
import { makeCustomSelect } from './custom_select.js'
import { pref, savePref } from './preferences.js'
import { setTheme } from './styles/styler_functions.js'

import { renderLeagueSettings, getLeagueSettings } from './parameter_collection/league_settings.js'
import { renderFormatAndCategories, getFormatAndCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts } from './parameter_collection/slot_counts.js'
import { renderTradeParameters } from './parameter_collection/trade_parameters.js'

// Dispatches to runSeasonInit (Season Mode) or runEvaluate (Draft / Auction Mode)
// depending on the current mode selector value.
function runModeEval(): Promise<void> {
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
    return mode === 'Season Mode' ? runSeasonInit() : runEvaluate()
}

// ─── Async init: fetch config, then build sidebar ────────────────────────────

;(async () => {

// Fetch sport config before rendering sidebar so defaults come from parameters.yaml
try {
    const config = await fetchConfig('NBA')
    setSportConfig(config)
} catch (err) {
    console.error('Failed to fetch config; using hard-coded fallbacks:', err)
}

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
    await runModeEval()
    applyLayout()
})

const formatSection = createSection(sidebarSections, 'Format & Categories')
renderFormatAndCategories(formatSection)
addApplyBtn(formatSection, async () => {
    const { scoring_format, categories } = getFormatAndCategories()
    await createOrPatchSession(4, { league: { scoring_format, categories } })
    await runModeEval()
    applyLayout()
})

const modelSection = createSection(sidebarSections, 'Model Parameters')
renderModelParameters(modelSection)
addApplyBtn(modelSection, async () => {
    const parameters = getModelParameters()
    await createOrPatchSession(3, { parameters })
    await runModeEval()
    applyLayout()
})

const slotSection = createSection(sidebarSections, 'Position Parameters')
renderSlotCounts(slotSection)
addApplyBtn(slotSection, async () => {
    const slot_counts = getSlotCounts()
    await createOrPatchSession(4, { slot_counts })
    await runModeEval()
    applyLayout()
})

const tradeSection = createSection(sidebarSections, 'Trade Parameters')
renderTradeParameters(tradeSection)
// Trade parameters target a future endpoint; no backend call yet.
addApplyBtn(tradeSection, () => {})

// ── Display section (theme toggle) ───────────────────────────────────────────
const displaySection = createSection(sidebarSections, 'Display')
const themeToggle = makeSidebarToggle('theme-toggle', 'Light mode', 'Dark mode')
displaySection.append(themeToggle)
const themeInput = document.getElementById('theme-toggle') as HTMLInputElement
const savedLight = pref('light_mode', false)
themeInput.checked = savedLight
setTheme(savedLight ? 'light' : 'dark')
themeInput.addEventListener('change', () => {
    const isLight = themeInput.checked
    savePref('light_mode', isLight)
    setTheme(isLight ? 'light' : 'dark')
})

// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

// ─── Persistent seat selector ─────────────────────────────────────────────────
// Rendered once into the fixed DOM element; layout.ts shows/hides the container.

function readTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

const seatSelectorContainer = document.getElementById('seat-selector-container') as HTMLElement
const initialTeamNames = readTeamNames()
const seatSelect = makeCustomSelect(
    'seat-select',
    initialTeamNames.map(name => ({ value: name, label: name })),
)
seatSelect.element.style.flex = '1'

const seatLabel = document.createElement('div')
seatLabel.className   = 'pick-control-label'
seatLabel.textContent = 'Select team'

const seatSelectorRow = document.createElement('div')
seatSelectorRow.className = 'seat-selector-row'
seatSelectorRow.append(seatLabel, seatSelect.element)

const seatSelectorWrap = document.createElement('div')
seatSelectorWrap.className = 'seat-selector-wrap'
seatSelectorWrap.append(seatSelectorRow)
seatSelectorContainer.append(seatSelectorWrap)

if (initialTeamNames.length > 0) {
    setCurrentSeat(initialTeamNames[0])
    seatSelect.setValue(initialTeamNames[0])
}

seatSelect.element.addEventListener('change', () => {
    setCurrentSeat(seatSelect.getValue() ?? null)
    if (getCurrentAuctionTab() === 'my-team') refreshAuctionGScore()
    if (getCurrentDraftTab()   === 'my-team') refreshDraftGScore()
    runModeEval()
        .then(() => applyLayout())
        .catch(err => console.error('Seat change evaluate failed:', err))
})

// ─── Mode change: rebuild table and sync session ───────────────────────────
// Registered before applyLayout so buildTable fires first, ensuring
// hscoretable.style.width is correct when applyLayout reads it.
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
        .then(() => { if (!signal.aborted) return runModeEval() })
        .then(() => { if (!signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Mode change failed:', err)
        })
})

applyLayout()
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', applyLayout)
document.getElementById('ls-platform')!.parentElement!.addEventListener('change', applyLayout)

// Initial build (empty; will be populated once the backend responds on load)
buildTable(getCandidatePlayers())

// Run all backend steps immediately on page load
runModeEval()
    .then(() => applyLayout())
    .catch(err => console.error('Initial load failed:', err))

// ── League settings auto-update ───────────────────────────────────────────
// Number inputs fire on 'change' (when focus leaves); buildTable runs first to
// set hscoretable.style.width before reapplyLayout reads it.
// AbortController cancels stale calls if the user changes multiple fields quickly.
let leagueSettingsController: AbortController | null = null
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        if (leagueSettingsController) leagueSettingsController.abort()
        leagueSettingsController = new AbortController()
        const { signal } = leagueSettingsController

        buildTable(getCandidatePlayers())
        const { n_drafters, n_picks, cash_per_team } = getLeagueSettings()
        createOrPatchSession(4, { league: { n_drafters, n_picks, cash_per_team } }, signal)
            .then(() => { if (!signal.aborted) return runModeEval() })
            .then(() => { if (!signal.aborted) applyLayout() })
            .catch(err => {
                if (err.name === 'AbortError') return
                console.error('League settings update failed:', err)
            })
    })
}

// Team names debounce: wait 600 ms after last keystroke to avoid flicker while typing.
let teamNamesTimer: ReturnType<typeof setTimeout> | null = null
document.getElementById('ls-team-names')!.addEventListener('input', () => {
    if (teamNamesTimer) clearTimeout(teamNamesTimer)
    teamNamesTimer = setTimeout(() => {
        const updatedTeamNames = readTeamNames()
        seatSelect.setOptions(updatedTeamNames.map(name => ({ value: name, label: name })))
        if (getCurrentSeat() === null && updatedTeamNames.length > 0) {
            setCurrentSeat(updatedTeamNames[0])
            seatSelect.setValue(updatedTeamNames[0])
        }
        buildTable(getCandidatePlayers())
        applyLayout()
    }, 600)
})

})()
