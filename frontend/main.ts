// main.ts
// Application entry point: wiring, in order. Builds the sidebar sections, registers every
// event listener, and triggers the first backend evaluation. The widgets it places live in
// their own modules (seat_selector.ts, account_row.ts, parameter_collection/*); the heavy
// logic lives in api/session.ts, table/player_table.ts, and layout.ts.

import { createSection, addApplyBtn, makeSidebarToggle } from './helper_functions.js'
import { makeDebouncer } from './api/session.js'
import { setSportConfig } from './app_state.js'
import { createOrPatchSession, runEvaluate, clearFullTeamResult, showDefaultRankings } from './api/draft_and_auction_session.js'
import { runSeasonInit, refreshSeasonRostersFromPlatform, clearLivePlatformRosters } from './api/season_session.js'
import { fetchConfig } from './api/client.js'
import { fetchCurrentUser, setSignedInUser } from './api/auth.js'
import { buildTableHeader } from './table/player_table.js'
import { applyLayout } from './layout.js'
import { resetDraftBoard } from './data_entry/draft_board.js'
import { resetAuctionEntry } from './data_entry/auction_entry.js'
import { pref, savePref } from './preferences.js'
import { setTheme } from './styles/styler_functions.js'
import { renderSeatSelector, refreshSeatOptions, clearSeatOptions } from './seat_selector.js'
import { renderAccountRow } from './account_row.js'

import {
    renderLeagueSettings, getLeagueSettings, isPlatformConnected,
    getModeSelectElement, getPlatformSelectElement,
} from './parameter_collection/league_settings.js'
import { TEAM_LABELS_CHANGED } from './data_entry/team_labels.js'
import {
    renderFormatAndCategories, getScoringFormat, getMostCategoriesWeight, getTiebreakerCategory,
    getSelectedCategories,
} from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams, waitForSeasons, markUploadedSourcesExpired } from './parameter_collection/player_stats.js'
import { renderModelParameters, refreshOpponentConfidenceControl, getModelParameters } from './parameter_collection/model_parameters.js'
import { renderSlotCounts, getSlotCounts, isSlotCountsValid, revalidateSlotCounts } from './parameter_collection/slot_counts.js'

// Dispatches to runSeasonInit (Season Mode) or runEvaluate (Draft / Auction Mode)
// depending on the current mode selector value.
async function runModeEval(): Promise<void> {
    const { platform, mode } = getLeagueSettings()
    if (mode === 'Season Mode') return runSeasonInit()
    // A live platform that isn't connected yet has no draft state to evaluate, and a normal
    // evaluate would throw (then its finally would flip the indicator to 'Updated'). Show the
    // base ("default") rankings instead, which leaves the indicator on 'Unconnected'.
    if (platform !== 'Enter your own data' && !isPlatformConnected()) return showDefaultRankings()
    await runEvaluate()   // discard the autopilot-only top-pick return; runModeEval yields void
}

/** One sidebar section's apply pipeline: abort any in-flight run for the section, rebuild
 *  the table header, patch the session from `fromStep`, then re-evaluate and re-layout —
 *  each post-patch step skipped once a newer run has superseded this one. Every section
 *  owns one chain, so rapid edits cancel their own stale backend calls without cancelling
 *  another section's. Player Stats does not use this: its expired-upload retry and
 *  pool-reset logic are real differences, not boilerplate (see applyPlayerStats). */
function makeApplyChain(label: string) {
    let controller: AbortController | null = null
    return function runApplyChain(
        fromStep: number
      , patch: Record<string, unknown>
      , { rebuildTableHeader = true, evaluate = true } = {}
    ): Promise<void> {
        controller?.abort()
        controller = new AbortController()
        const { signal } = controller
        if (rebuildTableHeader) buildTableHeader()
        return createOrPatchSession(fromStep, patch, signal)
            .then(() => { if (!signal.aborted && evaluate) return runModeEval() })
            .then(() => { if (!signal.aborted) applyLayout() })
            .catch(err => {
                if (err.name === 'AbortError') return
                console.error(`${label} failed:`, err)
            })
    }
}

// ─── Async init: fetch config, then build sidebar ────────────────────────────

;(async () => {

// Signing in is optional: the app runs anonymously, and the per-client rate limits are what
// keep the expensive endpoints from being hammered. What an account buys is the live-platform
// connection (credentials are stored per account server-side) and a rate-limit budget of one's
// own. The session is a same-origin cookie, so once signed in every fetch carries it.
const currentUser = await fetchCurrentUser()
setSignedInUser(currentUser)

// Fetch sport config before rendering sidebar so defaults come from parameters.yaml
const config = await fetchConfig('NBA')
setSportConfig(config)

const sidebar = document.getElementById('sidebar') as HTMLElement
const sidebarSections = document.getElementById('sidebar-sections')!

// ─── 1. League Settings ───────────────────────────────────────────────────────

renderLeagueSettings(createSection(sidebarSections, 'League Settings'))

// Seat selector: rendered once into the fixed DOM element; layout.ts shows/hides the
// container. The widget keeps the app-state seat in sync itself; what a seat change CAUSES
// is decided here — and the same flow runs when a null seat adopts a team, since adoption
// deliberately dispatches no event (see refreshSeatOptions).
function handleSeatChanged(): void {
    clearFullTeamResult()
    buildTableHeader()
    runModeEval()
        .then(() => applyLayout())
        .catch(err => console.error('Seat change evaluate failed:', err))
}
renderSeatSelector().addEventListener('change', handleSeatChanged)

// Mode change: rebuild table and sync session.
// Registered before the applyLayout listener so buildTableHeader fires first, ensuring
// hscoretable.style.width is correct when applyLayout reads it.
// is_auction is a session parameter (its league type): the backend requires remaining_cash
// on every evaluate exactly when it is set. Entering Auction Mode also patches cash_per_team
// so the backend can compute dollar values.
const applyModeChange = makeApplyChain('Mode change')
getModeSelectElement().addEventListener('change', () => {
    const { mode, cash_per_team } = getLeagueSettings()
    const patch = mode === 'Auction Mode'
        ? { is_auction: true, league: { cash_per_team } }
        : { is_auction: false }
    // Season Mode: the table is hidden, so there is no rebuild or evaluate — but the session's
    // league type must be patched BEFORE the season layout renders, because rendering fires
    // season evaluates (waiver, roster inspection) that omit remaining_cash. The standalone
    // applyLayout listener below skips Season Mode for the same reason: layout comes after
    // the patch, not concurrently with it.
    const entersSeasonMode = mode === 'Season Mode'
    applyModeChange(4, patch, { rebuildTableHeader: !entersSeasonMode, evaluate: !entersSeasonMode })
})
// Instant layout switch for draft/auction (their evaluates are sequenced after the patch by
// the handler above). Season Mode's layout is deferred until its session patch lands — see above.
getModeSelectElement().addEventListener('change', () => {
    if (getLeagueSettings().mode !== 'Season Mode') applyLayout()
})
getPlatformSelectElement().addEventListener('change', applyLayout)

// Season Mode + live platform: pull the platform's rosters into the grid when the user
// switches into that state (via either the mode or the platform dropdown). The poll is
// async, so we re-applyLayout once it lands; renderSeasonRosters reads the cache. Leaving
// that state clears the cache so the grid reverts to defaults.
function refreshSeasonRostersIfLive(): void {
    const { platform, mode } = getLeagueSettings()
    if (mode === 'Season Mode' && platform !== 'Enter your own data') {
        refreshSeasonRostersFromPlatform()
            .then(() => applyLayout())
            .catch(err => console.error('Season roster refresh failed:', err))
    } else {
        clearLivePlatformRosters()
    }
}
getModeSelectElement().addEventListener('change', refreshSeasonRostersIfLive)
getPlatformSelectElement().addEventListener('change', refreshSeasonRostersIfLive)

// On a platform switch (Draft/Auction), set up the seat selector and run the right evaluation
// for the new data source. Mode switches don't change the data source, so they're handled by
// the mode-change listener above (via runModeEval) and intentionally don't trigger this.
//   - Unconnected live   -> empty the seat, show base rankings, leave the indicator 'Unconnected'.
//   - Own data / connected -> seat from team names, run the normal evaluate (which renders
//     results and moves the indicator off 'Unconnected').
function syncForPlatformSwitch(): void {
    const { platform, mode } = getLeagueSettings()
    if (mode === 'Season Mode') return   // season handled by refreshSeasonRostersIfLive
    if (platform !== 'Enter your own data' && !isPlatformConnected()) {
        clearSeatOptions()
        showDefaultRankings().catch(err => console.error('Default rankings failed:', err))
    } else {
        refreshSeatOptions()
        buildTableHeader()
        runModeEval()
            .then(() => applyLayout())
            .catch(err => console.error('Platform-switch evaluate failed:', err))
    }
}
getPlatformSelectElement().addEventListener('change', syncForPlatformSwitch)

// Numeric league settings: n_drafters, n_picks, cash_per_team fire on 'change' (focus leaves).
const applyLeagueSettings = makeApplyChain('League settings update')
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        buildTableHeader()
        applyLayout()  // re-renders draft board with new n_drafters/n_picks so getDraftState() is current before evaluate
        revalidateSlotCounts()
        if (!isSlotCountsValid()) return
        const { n_drafters, n_picks, cash_per_team } = getLeagueSettings()
        applyLeagueSettings(
            4,
            { league: { n_drafters, n_picks, cash_per_team }, slot_counts: getSlotCounts() },
            { rebuildTableHeader: false },   // built above, before the layout pre-render
        )
    })
}

// Team identities (#ls-team-names) only change when the drafter count changes (own data) or a
// live platform connects — both already drive a layout re-render elsewhere — so this handler
// keeps the seat-selector options in sync with the identity set. When a null seat adopts the
// first new team, that adoption is a real seat change and runs the same flow a manual
// selection would.
document.getElementById('ls-team-names')!.addEventListener('input', () => {
    if (refreshSeatOptions() !== null) handleSeatChanged()
})

// A team's display label changed (header input). Identity/value is unchanged, so only relabel
// the seat selector's options — preserving the current selection by value.
document.addEventListener(TEAM_LABELS_CHANGED, () => { refreshSeatOptions() })

// ─── 2. Player Stats ──────────────────────────────────────────────────────────

const playerStatsSection = createSection(sidebarSections, 'Player Stats')
renderPlayerStats(playerStatsSection)

const applyPlayerStats = async (signal?: AbortSignal, keepsPlayerPool = false) => {
    // Switching the data source to Historical starts an async seasons fetch and fires this
    // change handler in the same breath, so the ps-season dropdown may not exist yet. Wait
    // for it (already resolved in every other case) rather than read a season that is not
    // there — getPlayerStatsParams rightly refuses a historical source without one.
    await waitForSeasons()
    const { data_source, injured_players } = getPlayerStatsParams()
    // The boards reset when the player pool's identity changes (data source, season,
    // uploads, injured list) so stale names are never sent to the backend. A blend-weight
    // change re-weights the same pool, so in-progress draft/auction boards survive it.
    if (!keepsPlayerPool) {
        resetDraftBoard()
        resetAuctionEntry()
    }
    buildTableHeader()
    createOrPatchSession(1, { data_source, injured_players }, signal)
        .then(() => { if (!signal || !signal.aborted) return runModeEval() })
        .then(() => { if (!signal || !signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            // A dead upload id (backend restart, or the upload store's TTL) fails every
            // patch that carries it, regardless of what the user changed. Surface it on
            // the upload's status line, drop the dead ids, and retry without them so the
            // rest of the change still lands.
            if (String(err).includes('data_id') && markUploadedSourcesExpired()) {
                applyPlayerStats(signal, keepsPlayerPool)
                return
            }
            console.error('Player stats apply failed:', err)
        })
}

/** Blend-weight inputs re-weight the existing pool; everything else in the section
 *  (data type, season, uploads, injured list) changes which players exist. */
const changeKeepsPlayerPool = (eventTarget: EventTarget | null): boolean =>
    eventTarget instanceof HTMLElement && eventTarget.id.startsWith('ps-w-')

let playerStatsChangeKeepsPool = false
const playerStatsDebouncer = makeDebouncer(() => applyPlayerStats(undefined, playerStatsChangeKeepsPool), 800)
let playerStatsController: AbortController | null = null

playerStatsSection.addEventListener('input', (event) => {
    playerStatsChangeKeepsPool = changeKeepsPlayerPool(event.target)
    playerStatsDebouncer.fire()
})
playerStatsSection.addEventListener('change', (event) => {
    playerStatsChangeKeepsPool = changeKeepsPlayerPool(event.target)
    playerStatsDebouncer.cancel()
    if (playerStatsController) playerStatsController.abort()
    playerStatsController = new AbortController()
    applyPlayerStats(playerStatsController.signal, playerStatsChangeKeepsPool)
})

// ─── 3. Format & Categories ───────────────────────────────────────────────────

const formatSection = createSection(sidebarSections, 'Format & Categories')
renderFormatAndCategories(formatSection)
const applyFormatChange = makeApplyChain('Format & categories apply')
formatSection.addEventListener('change', () => {
    refreshOpponentConfidenceControl(getScoringFormat())
    applyFormatChange(4, { league: {
        scoring_format:         getScoringFormat(),
        most_categories_weight: getMostCategoriesWeight(),
        tiebreaker_category:    getTiebreakerCategory(),
        categories:             getSelectedCategories(),
    } })
})

// ─── 4. Model Parameters ──────────────────────────────────────────────────────

const modelSection = createSection(sidebarSections, 'Model Parameters')
renderModelParameters(modelSection)
// The format section is built first, so its current value decides the control's initial visibility.
refreshOpponentConfidenceControl(getScoringFormat())
const applyModelParameters = makeApplyChain('Model parameters apply')
modelSection.addEventListener('change', () => {
    applyModelParameters(3, { parameters: getModelParameters() })
})

// ─── 5. Position Parameters ───────────────────────────────────────────────────

const slotSection = createSection(sidebarSections, 'Position Parameters')
renderSlotCounts(slotSection)
const applySlotCounts = makeApplyChain('Position parameters apply')
addApplyBtn(slotSection, async () => {
    if (!isSlotCountsValid()) return
    await applySlotCounts(4, { slot_counts: getSlotCounts() })
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

// ─── Sidebar toggle ───────────────────────────────────────────────────────────

const SIDEBAR_COLLAPSE_BREAKPOINT = 768  // px; viewports narrower than this default to a collapsed sidebar
const sidebarToggle = document.getElementById('sidebar-toggle') as HTMLButtonElement
const appLayout     = document.getElementById('app-layout') as HTMLElement
const defaultCollapsed = window.innerWidth < SIDEBAR_COLLAPSE_BREAKPOINT
if (pref('sidebar_collapsed', defaultCollapsed)) appLayout.classList.add('sidebar-collapsed')
sidebarToggle.addEventListener('click', () => {
    const collapsed = appLayout.classList.toggle('sidebar-collapsed')
    savePref('sidebar_collapsed', collapsed)
})

// ─── Account (signed-in name + sign out) ──────────────────────────────────────

renderAccountRow(sidebar, sidebarSections, currentUser)

// ─── Bootstrap ────────────────────────────────────────────────────────────────

// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

applyLayout()
buildTableHeader()
waitForSeasons()
    .then(() => runModeEval())
    .then(() => applyLayout())
    .catch(err => console.error('Initial load failed:', err))

})()
