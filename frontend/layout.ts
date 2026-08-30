// layout.ts
// Manages the main-content area layout based on mode (Draft / Auction / Season)
// and platform (own data vs. live).  Call applyLayout() after the sidebar is built.

import { renderDraftBoard } from './data_entry/draft_board.js'
import { renderAuctionEntry }  from './data_entry/auction_entry.js'
import { renderSeasonRosters } from './data_entry/season/season_rosters.js'
import { renderSeasonTrading } from './data_entry/season/season_trading.js'
import { renderWaiverControls } from './data_entry/season/season_waiver.js'
import { getAuctionState }     from './data_entry/auction_state.js'
import { getDraftState }       from './data_entry/draft_state.js'
import { renderTeamGScoreTable } from './table/gscore_table.js'
import { getLeagueSettings, isPlatformConnected } from './parameter_collection/league_settings.js'
import { getCurrentSeat } from './app_state.js'
import { getFullTeamResult, refreshLiveAnalysis } from './api/draft_and_auction_session.js'
import { setIndicatorState, claimDisplay } from './api/session.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let seasonNavBuilt    = false
let currentSeasonTab: string | null = null  // null means no tab activated yet
let currentAuctionTab = 'candidates'
let currentDraftTab   = 'candidates'

// Refresh the active "my team" panel when session.ts signals that the full-team
// H-score result has arrived (dispatched after the evaluate await resolves).
// Guarded by the CURRENT mode: each mode's tab state persists while the other mode is
// active, so an unguarded refresh would repaint the inactive mode's (hidden, stale)
// panel with the wrong board's rosters.
document.addEventListener('full-team-result-updated', () => {
    const { mode } = getLeagueSettings()
    if (mode === 'Draft Mode'   && currentDraftTab   === 'my-team') refreshDraftGScore()
    if (mode === 'Auction Mode' && currentAuctionTab === 'my-team') refreshAuctionGScore()
})

// Refresh the active "my team" panel whenever the seat changes (including during autopilot).
document.addEventListener('seat-changed', () => {
    const { mode } = getLeagueSettings()
    if (mode === 'Draft Mode'   && currentDraftTab   === 'my-team') refreshDraftGScore()
    if (mode === 'Auction Mode' && currentAuctionTab === 'my-team') refreshAuctionGScore()
})

export function getCurrentAuctionTab(): string { return currentAuctionTab }
export function getCurrentDraftTab():   string { return currentDraftTab   }

// ─── Public API ───────────────────────────────────────────────────────────────

/** Re-dispatches the layout with current DOM state. */
export function applyLayout(): void {
    const { mode, platform } = getLeagueSettings()

    if (mode === 'Season Mode') {
        showSeasonLayout()
    } else {
        // Reset season tab tracking when leaving Season Mode so the next entry defaults to Waiver
        currentSeasonTab = null
        if (platform === 'Enter your own data') {
            showOwnDataLayout(mode)
        } else {
            showLiveLayout()
        }
    }
}


// ─── Own-data layout (Draft or Auction) ───────────────────────────────────────

/** Renders the own-data layout: draft/auction board + seat selector above the H-score table. */
function showOwnDataLayout(mode: string): void {
    hide('season-nav')
    hide('live-bar')
    hide('left-panel')
    hide('season-rosters-row')
    hide('season-trading-row')
    removeLiveRefreshButton()   // live-only; gone in own-data mode

    show('content-row')
    show('eval-indicator')

    const rightHeader    = document.getElementById('right-header')!
    const rightSubHeader = document.getElementById('right-sub-header')!
    const rightFooter    = document.getElementById('right-footer')!
    rightHeader.innerHTML    = ''
    rightSubHeader.innerHTML = ''
    rightFooter.innerHTML    = ''

    show('seat-selector-container')

    if (mode === 'Auction Mode') {
        renderAuctionEntry(rightHeader)
        // The other mode's team panel would otherwise stay visible (and stale) under this
        // layout — each branch must hide its counterpart's container.
        hide('draft-gscore')
        buildModeTabBar(rightSubHeader, 'auction-tab-bar', currentAuctionTab, activateAuctionTab)
        activateAuctionTab(currentAuctionTab)

        // Refresh G-score table when the auction board changes (pick/undo/clear)
        rightHeader.addEventListener('auction-board-change', () => {
            if (currentAuctionTab === 'my-team') refreshAuctionGScore()
        })
    } else {
        renderDraftBoard(rightHeader)
        hide('auction-gscore')
        buildModeTabBar(rightSubHeader, 'draft-tab-bar', currentDraftTab, activateDraftTab)
        activateDraftTab(currentDraftTab)
    }

    rightFooter.innerHTML = ''
}

// ─── Live-platform layout (Draft or Auction) ──────────────────────────────────

/** Renders the live-platform layout: seat selector only (no data-entry grid). */
function showLiveLayout(): void {
    hide('season-nav')
    hide('live-bar')
    hide('left-panel')
    hide('season-rosters-row')
    hide('season-trading-row')

    show('content-row')
    show('eval-indicator')

    const rightHeader    = document.getElementById('right-header')!
    const rightSubHeader = document.getElementById('right-sub-header')!
    const rightFooter    = document.getElementById('right-footer')!
    rightHeader.innerHTML    = ''
    rightSubHeader.innerHTML = ''
    rightFooter.innerHTML    = ''

    show('seat-selector-container')

    // Live platforms have no manual data-entry grid; a Refresh Analysis button pulls the
    // current draft/roster state and re-evaluates. It sits in the tab-row beside the seat
    // selector (left of the status indicator) and is disabled until the platform connection
    // is established — clicking before then would spin the indicator forever.
    removeLiveRefreshButton()
    const refreshButton = document.createElement('button')
    refreshButton.type        = 'button'
    refreshButton.id          = 'live-refresh-btn'
    refreshButton.className    = 'section-apply-btn live-refresh-btn'
    refreshButton.textContent  = 'Refresh Analysis'
    refreshButton.disabled     = !isPlatformConnected()
    refreshButton.addEventListener('click', () => {
        if (refreshButton.disabled) return
        refreshLiveAnalysis().catch(err => console.error('Refresh analysis failed:', err))
    })
    const tabRow = document.getElementById('tab-row')!
    tabRow.insertBefore(refreshButton, document.getElementById('eval-indicator'))

    // Until a platform is connected, the status reads "Unconnected" rather than a stale
    // "Updated"; once connected it returns to the neutral idle state (the default rankings
    // are still shown the whole time — see showDefaultRankings). Claimed so an async run
    // still in flight from the previous layout cannot overwrite this once it settles.
    claimDisplay()
    setIndicatorState(isPlatformConnected() ? 'idle' : 'unconnected')

    show('hscoretable')
    hide('draft-gscore')
}

/** Removes the live-platform Refresh Analysis button from the tab-row, if present. */
function removeLiveRefreshButton(): void {
    document.getElementById('live-refresh-btn')?.remove()
}

// ─── Season layout ────────────────────────────────────────────────────────────

/** Renders the season mode layout: tab nav (Waiver / Trading / Rosters) with content panes. */
function showSeasonLayout(): void {
    hide('live-bar')
    hide('left-panel')
    hide('seat-selector-container')
    hide('eval-indicator')
    removeLiveRefreshButton()   // live draft/auction-only; gone in season mode

    // Clear sub-header so the tab bar from draft/auction mode doesn't bleed in
    const rightSubHeader = document.getElementById('right-sub-header')!
    rightSubHeader.innerHTML = ''
    document.getElementById('tab-row')!.style.maxWidth = ''

    show('season-nav')

    if (!seasonNavBuilt) {
        buildSeasonNav()
        seasonNavBuilt = true
    }

    // Pre-populate roster DOM elements so the Waiver tab can read sr-player-* inputs
    // on first visit, before the user has manually opened the Rosters tab.
    // Must run every time Season Mode is entered, not just on first build, because
    // player data may not be loaded yet on the first applyLayout call.
    renderSeasonRosters(
        document.getElementById('rosters-left')!,
        document.getElementById('rosters-right')!,
    )

    // Only default to Waiver on first entry; leave the active tab alone on subsequent calls
    if (currentSeasonTab === null) {
        activateSeasonTab('waiver')
    }
}

/** Creates the season mode tab buttons (Waiver, Trading, Rosters) in the nav bar. */
function buildSeasonNav(): void {
    const nav = document.getElementById('season-nav')!
    nav.innerHTML = ''

    const tabs: { id: string; label: string }[] = [
        { id: 'waiver',  label: 'Waiver'  },
        { id: 'trading', label: 'Trading' },
        { id: 'rosters', label: 'Rosters' },
    ]

    for (const tab of tabs) {
        const btn = document.createElement('button')
        btn.className   = 'season-tab-btn'
        btn.textContent = tab.label
        btn.dataset.tab = tab.id
        btn.addEventListener('click', () => activateSeasonTab(tab.id))
        nav.append(btn)
    }
}

/** Switches the visible season mode content pane and highlights the active tab button. */
function activateSeasonTab(tabId: string): void {
    currentSeasonTab = tabId

    document.querySelectorAll('.season-tab-btn').forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId)
    })

    if (tabId === 'waiver') {
        hide('season-rosters-row')
        hide('season-trading-row')
        show('content-row')

        const rightHeader = document.getElementById('right-header')!
        rightHeader.innerHTML = ''
        document.getElementById('right-footer')!.innerHTML = ''
        renderWaiverControls(rightHeader)

    } else if (tabId === 'trading') {
        hide('content-row')
        hide('season-rosters-row')
        show('season-trading-row')

        const tradingContainer = document.getElementById('season-trading-row')!
        renderSeasonTrading(tradingContainer)

    } else if (tabId === 'rosters') {
        hide('content-row')
        hide('season-trading-row')
        show('season-rosters-row')

        const rostersLeft  = document.getElementById('rosters-left')!
        const rostersRight = document.getElementById('rosters-right')!
        renderSeasonRosters(rostersLeft, rostersRight)
    }
}

// ─── Candidates / Team view buttons (shared) ─────────────────────────────────

/** Creates the Show candidates / Show team stats button group in the given container. */
function buildModeTabBar(
    container: HTMLElement
    , barClass: string
    , currentTab: string
    , onTabClick: (tabId: string) => void
): void {
    const bar = document.createElement('div')
    bar.className = barClass

    for (const tab of [
        { id: 'candidates', label: 'Show candidates' },
        { id: 'my-team',    label: 'Show team statistics' },
    ]) {
        const btn = document.createElement('button')
        btn.className   = 'pick-btn'
        btn.textContent = tab.label
        btn.dataset.tab = tab.id
        if (tab.id === currentTab) btn.classList.add('active')
        btn.addEventListener('click', () => onTabClick(tab.id))
        bar.append(btn)
    }

    container.append(bar)
}

// ─── Draft and Auction tab activation ────────────────────────────────────────

function activateDraftTab(tabId: string): void {
    currentDraftTab = tabId
    document.querySelectorAll('.draft-tab-bar .pick-btn').forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId)
    })
    if (tabId === 'candidates') {
        show('hscoretable')
        hide('draft-gscore')
    } else {
        hide('hscoretable')
        show('draft-gscore')
        refreshDraftGScore()
    }
}

function activateAuctionTab(tabId: string): void {
    currentAuctionTab = tabId
    document.querySelectorAll('.auction-tab-bar .pick-btn').forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId)
    })
    if (tabId === 'candidates') {
        show('hscoretable')
        hide('auction-gscore')
    } else {
        hide('hscoretable')
        show('auction-gscore')
        refreshAuctionGScore()
    }
}

export function refreshDraftGScore(): void {
    const { player_assignments } = getDraftState()
    renderTeamGScoreTable(
        player_assignments[getCurrentSeat() ?? ''] ?? []
        , document.getElementById('draft-gscore')!
        , getFullTeamResult()
    )
}

export function refreshAuctionGScore(): void {
    const { player_assignments } = getAuctionState()
    renderTeamGScoreTable(
        player_assignments[getCurrentSeat() ?? ''] ?? []
        , document.getElementById('auction-gscore')!
        , getFullTeamResult()
    )
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function show(id: string): void {
    document.getElementById(id)!.style.display = ''
}

function hide(id: string): void {
    document.getElementById(id)!.style.display = 'none'
}
