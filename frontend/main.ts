// main.ts
// Application entry point. Sets up the sidebar, registers event listeners,
// initializes the layout, and triggers the first backend evaluation.
// All heavy logic lives in api/session.ts, table/player_table.ts, and layout.ts.

import { createSection, addApplyBtn, makeSidebarToggle } from './helper_functions.js'
import { makeDebouncer } from './api/session.js'
import { setSportConfig, getCurrentSeat, setCurrentSeat } from './app_state.js'
import { createOrPatchSession, runEvaluate, clearFullTeamResult, showDefaultRankings } from './api/draft_and_auction_session.js'
import { runSeasonInit, refreshSeasonRostersFromPlatform, clearLivePlatformRosters } from './api/season_session.js'
import { fetchConfig } from './api/client.js'
import { fetchCurrentUser, renderLoginScreen, logout } from './api/auth.js'
import { buildTableHeader } from './table/player_table.js'
import { applyLayout } from './layout.js'
import { resetDraftBoard } from './data_entry/draft_board.js'
import { resetAuctionEntry } from './data_entry/auction_entry.js'
import { makeCustomSelect } from './custom_select.js'
import { pref, savePref } from './preferences.js'
import { setTheme } from './styles/styler_functions.js'

import { renderLeagueSettings, getLeagueSettings, getTeamNames, isPlatformConnected } from './parameter_collection/league_settings.js'
import { getTeamLabel, defaultTeamLabel, TEAM_LABELS_CHANGED } from './data_entry/team_labels.js'
import { renderFormatAndCategories, getScoringFormat, getSelectedCategories } from './parameter_collection/format_and_categories.js'
import { renderPlayerStats, getPlayerStatsParams, waitForInitialSeasons, markUploadedSourcesExpired } from './parameter_collection/player_stats.js'
import { renderModelParameters, getModelParameters } from './parameter_collection/model_parameters.js'
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

// ─── Async init: fetch config, then build sidebar ────────────────────────────

;(async () => {

// Gate the whole app behind Google login. The session is a same-origin cookie, so once
// signed in every subsequent fetch carries it automatically.
const currentUser = await fetchCurrentUser()
if (!currentUser) {
    renderLoginScreen()
    return
}

// Fetch sport config before rendering sidebar so defaults come from parameters.yaml
const config = await fetchConfig('NBA')
setSportConfig(config)

const sidebar = document.getElementById('sidebar') as HTMLElement
const sidebarSections = document.getElementById('sidebar-sections')!

// ─── 1. League Settings ───────────────────────────────────────────────────────

renderLeagueSettings(createSection(sidebarSections, 'League Settings'))

// Seat selector option from a team identity: the value is always the identity ("Team N" for own
// data; the real name for a live platform). For own-data identities we show the editable display
// label; a live-platform name (which differs from the "Team N" default) is shown as-is.
function seatOption(identityName: string, index: number): { value: string; label: string } {
    const label = identityName === defaultTeamLabel(index) ? getTeamLabel(index) : identityName
    return { value: identityName, label }
}
function seatOptions(names: string[]): { value: string; label: string }[] {
    return names.map(seatOption)
}

// Seat selector: rendered once into the fixed DOM element; layout.ts shows/hides the container.
const seatSelectorContainer = document.getElementById('seat-selector-container') as HTMLElement
const initialTeamNames = getTeamNames()
const seatSelect = makeCustomSelect(
    'seat-select',
    seatOptions(initialTeamNames),
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
    buildTableHeader()
    runModeEval()
        .then(() => applyLayout())
        .catch(err => console.error('Seat change evaluate failed:', err))
})

// Mode change: rebuild table and sync session.
// Registered before the applyLayout listener so buildTableHeader fires first, ensuring
// hscoretable.style.width is correct when applyLayout reads it.
// is_auction is a session parameter (its league type): the backend requires remaining_cash
// on every evaluate exactly when it is set. Entering Auction Mode also patches cash_per_team
// so the backend can compute dollar values.
// AbortController cancels stale backend calls on rapid mode switches.
let modeChangeController: AbortController | null = null
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    if (modeChangeController) modeChangeController.abort()
    modeChangeController = new AbortController()
    const { signal } = modeChangeController

    const { mode, cash_per_team } = getLeagueSettings()
    const patch = mode === 'Auction Mode'
        ? { is_auction: true, league: { cash_per_team } }
        : { is_auction: false }

    if (mode === 'Season Mode') {
        // The table is hidden in season mode, so there is no rebuild or evaluate here — but the
        // session's league type must be patched BEFORE the season layout renders, because
        // rendering fires season evaluates (waiver, roster inspection) that omit remaining_cash.
        // The standalone applyLayout listener below skips Season Mode for the same reason:
        // layout comes after the patch, not concurrently with it.
        createOrPatchSession(4, patch, signal)
            .then(() => { if (!signal.aborted) applyLayout() })
            .catch(err => {
                if (err.name === 'AbortError') return
                console.error('Mode change failed:', err)
            })
        return
    }

    buildTableHeader()

    createOrPatchSession(4, patch, signal)
        .then(() => { if (!signal.aborted) return runModeEval() })
        .then(() => { if (!signal.aborted) applyLayout() })
        .catch(err => {
            if (err.name === 'AbortError') return
            console.error('Mode change failed:', err)
        })
})
// Instant layout switch for draft/auction (their evaluates are sequenced after the patch by
// the handler above). Season Mode's layout is deferred until its session patch lands — see above.
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
    if (getLeagueSettings().mode !== 'Season Mode') applyLayout()
})
document.getElementById('ls-platform')!.parentElement!.addEventListener('change', applyLayout)

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
document.getElementById('ls-mode')!.parentElement!.addEventListener('change', refreshSeasonRostersIfLive)
document.getElementById('ls-platform')!.parentElement!.addEventListener('change', refreshSeasonRostersIfLive)

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
        setCurrentSeat(null)
        seatSelect.setOptions([])
        showDefaultRankings().catch(err => console.error('Default rankings failed:', err))
    } else {
        const names = getTeamNames()
        seatSelect.setOptions(seatOptions(names), getCurrentSeat() ?? names[0])
        if (getCurrentSeat() === null && names.length > 0) setCurrentSeat(names[0])
        buildTableHeader()
        runModeEval()
            .then(() => applyLayout())
            .catch(err => console.error('Platform-switch evaluate failed:', err))
    }
}
document.getElementById('ls-platform')!.parentElement!.addEventListener('change', syncForPlatformSwitch)

// Numeric league settings: n_drafters, n_picks, cash_per_team fire on 'change' (focus leaves).
// AbortController cancels stale calls if the user changes multiple fields quickly.
let leagueSettingsController: AbortController | null = null
for (const id of ['ls-n-drafters', 'ls-n-picks', 'ls-cash-per-team']) {
    document.getElementById(id)!.addEventListener('change', () => {
        if (leagueSettingsController) leagueSettingsController.abort()
        leagueSettingsController = new AbortController()
        const { signal } = leagueSettingsController

        buildTableHeader()
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

// Team identities (#ls-team-names) only change when the drafter count changes (own data) or a
// live platform connects — both already drive a layout re-render elsewhere — so this handler just
// keeps the seat-selector options in sync with the identity set.
document.getElementById('ls-team-names')!.addEventListener('input', () => {
    const updatedTeamNames = getTeamNames()
    seatSelect.setOptions(seatOptions(updatedTeamNames), getCurrentSeat() ?? updatedTeamNames[0])
    if (getCurrentSeat() === null && updatedTeamNames.length > 0) {
        setCurrentSeat(updatedTeamNames[0])
        seatSelect.setValue(updatedTeamNames[0])
    }
})

// A team's display label changed (header input). Identity/value is unchanged, so only relabel the
// seat selector's options — preserving the current selection by value.
document.addEventListener(TEAM_LABELS_CHANGED, () => {
    const names = getTeamNames()
    seatSelect.setOptions(seatOptions(names), getCurrentSeat() ?? names[0])
})

// ─── 2. Player Stats ──────────────────────────────────────────────────────────

const playerStatsSection = createSection(sidebarSections, 'Player Stats')
renderPlayerStats(playerStatsSection)

const applyPlayerStats = (signal?: AbortSignal, keepsPlayerPool = false) => {
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
let formatController: AbortController | null = null
formatSection.addEventListener('change', () => {
    if (formatController) formatController.abort()
    formatController = new AbortController()
    const { signal } = formatController
    buildTableHeader()
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
    buildTableHeader()
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
    buildTableHeader()
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

const accountRow = document.createElement('div')
accountRow.className = 'sidebar-account'

const identity = document.createElement('div')
identity.className = 'account-identity'

const avatar = document.createElement('span')
avatar.className = 'account-avatar'

/** Default person icon — shown when no Google picture is available, and swapped in when a
 *  stored picture URL stops resolving (Google's profile-photo URLs are tokenized and expire;
 *  the session stores the URL from login, so a long-lived login eventually holds a dead link). */
function showDefaultAccountIcon(): void {
    avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
        + '<path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.4 0-8 2.2-8 5v1h16v-1c0-2.8-3.6-5-8-5Z"/></svg>'
}

if (currentUser.picture) {
    const avatarImg = document.createElement('img')
    avatarImg.src = currentUser.picture
    avatarImg.alt = ''
    avatarImg.referrerPolicy = 'no-referrer'   // Google pic URLs 403 without this
    avatarImg.addEventListener('error', showDefaultAccountIcon)
    avatar.append(avatarImg)
} else {
    showDefaultAccountIcon()
}

const accountName = document.createElement('span')
accountName.className   = 'account-name'
accountName.textContent = currentUser.name
identity.append(avatar, accountName)

const logoutBtn = document.createElement('button')
logoutBtn.type        = 'button'
logoutBtn.className    = 'account-logout'
logoutBtn.textContent  = 'Sign out'
logoutBtn.addEventListener('click', () => { logout().catch(err => console.error('Logout failed:', err)) })

accountRow.append(identity, logoutBtn)
// Place it below the title's divider line, above the sidebar options, with its own
// divider separating it from the first option.
sidebar.insertBefore(accountRow, sidebarSections)
const accountDivider = document.createElement('hr')
accountDivider.className = 'sidebar-divider'
sidebar.insertBefore(accountDivider, sidebarSections)

// ─── Bootstrap ────────────────────────────────────────────────────────────────

// All sections are fully built; reveal the sidebar in one repaint
sidebar.style.visibility = ''

applyLayout()
buildTableHeader()
waitForInitialSeasons()
    .then(() => runModeEval())
    .then(() => applyLayout())
    .catch(err => console.error('Initial load failed:', err))

})()
