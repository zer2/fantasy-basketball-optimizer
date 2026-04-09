// main.ts
// Application entry point. Sets up the sidebar, registers event listeners,
// initializes the layout, and triggers the first backend evaluation.
// All heavy logic lives in api/session.ts, table/player_table.ts, and layout.ts.

import { createSection, addApplyBtn, makeSidebarToggle, makeDebouncer } from './helper_functions.js'
import { getCandidatePlayerResults, setSportConfig, getCurrentSeat, setCurrentSeat } from './app_state.js'
import { createOrPatchSession, runEvaluate, runSeasonInit, clearFullTeamResult } from './api/session.js'
import { fetchConfig } from './api/client.js'
import { buildTable } from './table/player_table.js'
import { applyLayout } from './layout.js'
import { resetDraftBoard } from './data_entry/draft_board.js'
import { resetAuctionEntry } from './data_entry/auction_entry.js'
import { makeCustomSelect } from './custom_select.js'
import { pref, savePref } from './preferences.js'
import { setTheme } from './styles/styler_functions.js'

import { renderLeagueSettings, getLeagueSettings, getTeamNames } from './parameter_collection/league_settings.js'
import { renderFormatAndCategories, getScoringFormat, getSelectedCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts, isSlotCountsValid, revalidateSlotCounts } from './parameter_collection/slot_counts.js'

// Dispatches to runSeasonInit (Season Mode) or runEvaluate (Draft / Auction Mode)
// depending on the current mode selector value.
function runModeEval(): Promise<void> {
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
    return mode === 'Season Mode' ? runSeasonInit() : runEvaluate()
}

// ─── Async init: fetch config, then build sidebar ────────────────────────────

;(async () => {

// Fetch sport config before rendering sidebar so defaults come from parameters.yaml
const config = await fetchConfig('NBA')
setSportConfig(config)

const sidebar = document.getElementById('sidebar') as HTMLElement
const sidebarSections = document.getElementById('sidebar-sections')!

// ─── 1. League Settings ───────────────────────────────────────────────────────

renderLeagueSettings(createSection(sidebarSections, 'League Settings'))

// Seat selector: rendered once into the fixed DOM element; layout.ts shows/hides the container.
const seatSelectorContainer = document.getElementById('seat-selector-container') as HTMLElement
const initialTeamNames = getTeamNames()
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
    clearFullTeamResult()
    runModeEval()
        .then(() => applyLayout())
        .catch(err => console.error('Seat change evaluate failed:', err))
})

// Mode change: rebuild table and sync session.
// Registered before the applyLayout listener so buildTable fires first, ensuring
// hscoretable.style.width is correct when applyLayout reads it.
// When switching to Auction Mode, patch cash_per_team so the backend can compute dollar values.
// AbortController cancels stale backend calls on rapid mode switches.
let modeChangeController: AbortController | null = null
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    if (modeChangeController) modeChangeController.abort()
    modeChangeController = new AbortController()
    const { signal } = modeChangeController

    const { mode, cash_per_team } = getLeagueSettings()

    if (mode === 'Season Mode') return   // table is hidden in season mode; skip rebuild and backend call

    buildTable(getCandidatePlayerResults())

    const patch = mode === 'Auction Mode' ? { league: { cash_per_team } } : {}
    createOrPatchSession(4, patch, signal)
        .then(() => { if (!signal.aborted) return runModeEval() })
        .then(() => { if (!signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Mode change failed:', err)
        })
})
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', applyLayout)
document.getElementById('ls-platform')!.parentElement!.addEventListener('change', applyLayout)

// Numeric league settings: n_drafters, n_picks, cash_per_team fire on 'change' (focus leaves).
// AbortController cancels stale calls if the user changes multiple fields quickly.
let leagueSettingsController: AbortController | null = null
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        if (leagueSettingsController) leagueSettingsController.abort()
        leagueSettingsController = new AbortController()
        const { signal } = leagueSettingsController

        buildTable(getCandidatePlayerResults())
        applyLayout()  // re-renders draft board with new n_drafters/n_picks so getDraftState() is current before evaluate
        revalidateSlotCounts()
        if (!isSlotCountsValid()) return
        const { n_drafters, n_picks, cash_per_team } = getLeagueSettings()
        createOrPatchSession(4, { league: { n_drafters, n_picks, cash_per_team }, slot_counts: getSlotCounts() }, signal)
            .then(() => { if (!signal.aborted) return runModeEval() })
            .then(() => { if (!signal.aborted) applyLayout() })
            .catch(err => {
                if (err.name === 'AbortError') return
                console.error('League settings update failed:', err)
            })
    })
}

// : debounce 600 ms after last keystroke to avoid flicker while typing.
let teamNamesTimer: ReturnType<typeof setTimeout> | null = null
document.getElementById('ls-team-names')!.addEventListener('input', () => {
    if (teamNamesTimer) clearTimeout(teamNamesTimer)
    teamNamesTimer = setTimeout(() => {
        const updatedTeamNames = getTeamNames()
        seatSelect.setOptions(updatedTeamNames.map(name => ({ value: name, label: name })))
        if (getCurrentSeat() === null && updatedTeamNames.length > 0) {
            setCurrentSeat(updatedTeamNames[0])
            seatSelect.setValue(updatedTeamNames[0])
        }
        buildTable(getCandidatePlayerResults())
        applyLayout()
    }, 600)
})

// ─── 2. Player Stats ──────────────────────────────────────────────────────────

const playerStatsSection = createSection(sidebarSections, 'Player Stats')
renderPlayerStats(playerStatsSection)

const applyPlayerStats = (signal?: AbortSignal) => {
    const { data_source, injured_players } = getPlayerStatsParams()
    resetDraftBoard()
    resetAuctionEntry()
    createOrPatchSession(1, { data_source, injured_players }, signal)
        .then(() => { if (!signal || !signal.aborted) return runModeEval() })
        .then(() => { if (!signal || !signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Player stats apply failed:', err)
        })
}

const playerStatsDebouncer = makeDebouncer(() => applyPlayerStats(), 800)
let playerStatsController: AbortController | null = null

playerStatsSection.addEventListener('input', () => playerStatsDebouncer.fire())
playerStatsSection.addEventListener('change', () => {
    playerStatsDebouncer.cancel()
    if (playerStatsController) playerStatsController.abort()
    playerStatsController = new AbortController()
    applyPlayerStats(playerStatsController.signal)
})

// ─── 3. Format & Categories ───────────────────────────────────────────────────

const formatSection = createSection(sidebarSections, 'Format & Categories')
renderFormatAndCategories(formatSection)
let formatController: AbortController | null = null
formatSection.addEventListener('change', () => {
    if (formatController) formatController.abort()
    formatController = new AbortController()
    const { signal } = formatController
    createOrPatchSession(4, { league: { scoring_format: getScoringFormat(), categories: getSelectedCategories() } }, signal)
        .then(() => { if (!signal.aborted) return runModeEval() })
        .then(() => { if (!signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Format & categories apply failed:', err)
        })
})

// ─── 4. Model Parameters ──────────────────────────────────────────────────────

const modelSection = createSection(sidebarSections, 'Model Parameters')
renderModelParameters(modelSection)
let modelController: AbortController | null = null
modelSection.addEventListener('change', () => {
    if (modelController) modelController.abort()
    modelController = new AbortController()
    const { signal } = modelController
    createOrPatchSession(3, { parameters: getModelParameters() }, signal)
        .then(() => { if (!signal.aborted) return runModeEval() })
        .then(() => { if (!signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Model parameters apply failed:', err)
        })
})

// ─── 5. Position Parameters ───────────────────────────────────────────────────

const slotSection = createSection(sidebarSections, 'Position Parameters')
renderSlotCounts(slotSection)
addApplyBtn(slotSection, async () => {
    if (!isSlotCountsValid()) return
    const slot_counts = getSlotCounts()
    await createOrPatchSession(4, { slot_counts })
    await runModeEval()
    applyLayout()
})

// ─── 6. Display ───────────────────────────────────────────────────────────────

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

// ─── Bootstrap ────────────────────────────────────────────────────────────────

// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

applyLayout()
buildTable(null)
runModeEval()
    .then(() => applyLayout())
    .catch(err => console.error('Initial load failed:', err))

})()
