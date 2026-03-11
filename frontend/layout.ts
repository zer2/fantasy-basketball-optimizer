// layout.ts
// Manages the main-content area layout based on mode (Draft / Auction / Season)
// and platform (own data vs. live).  Call initLayout() once after the sidebar is built.

import { renderDraftBoard, getDraftState } from './data_entry/draft_board.js'
import { renderAuctionEntry }  from './data_entry/auction_entry.js'
import { renderSeasonRosters } from './data_entry/season/season_rosters.js'
import { renderSeasonTrading } from './data_entry/season/season_trading.js'
import { renderWaiverControls } from './data_entry/season/season_waiver.js'
import { makeCustomSelect }    from './custom_select.js'
import { getAuctionState }     from './data_entry/auction_entry.js'
import { renderTeamGScoreTable } from './table/gscore_table.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let seasonNavBuilt  = false
let currentSeasonTab = ''   // tracks active season tab; '' means no tab activated yet
let currentAuctionTab = 'candidates'  // 'candidates' or 'my-team'
let currentDraftTab   = 'candidates'  // 'candidates' or 'my-team'
let currentSeat     = ''
let _onEvaluate: (() => void | Promise<void>) | undefined

// ─── Public API ───────────────────────────────────────────────────────────────

/** Initializes the main content area and registers change listeners for mode/platform selectors. */
export function initLayout(opts: { onEvaluate?: () => void | Promise<void> } = {}): void {
    _onEvaluate = opts.onEvaluate
    applyLayout()
    // Re-apply whenever mode or platform changes
    document.getElementById('ls-mode')!.parentElement!
        .addEventListener('change', applyLayout)
    document.getElementById('ls-platform')!.parentElement!
        .addEventListener('change', applyLayout)
}

/** Re-runs the layout dispatcher with current DOM state. Called by the Apply button. */
export function reapplyLayout(): void { applyLayout() }

/** Returns the currently selected seat (team name). Empty string if none selected. */
export function getCurrentSeat(): string { return currentSeat }

// ─── Layout dispatcher ────────────────────────────────────────────────────────

/** Reads current mode/platform from the DOM and dispatches to the appropriate layout builder. */
function applyLayout(): void {
    const mode     = (document.getElementById('ls-mode')     as HTMLInputElement).value
    const platform = (document.getElementById('ls-platform') as HTMLInputElement).value

    const isOwnData = platform === 'Enter your own data'
    const isSeason  = mode === 'Season Mode'

    // Reset season tab tracking when leaving Season Mode so the next entry defaults to Waiver
    if (!isSeason) currentSeasonTab = ''

    if (isSeason) {
        showSeasonLayout()
    } else if (isOwnData) {
        showOwnDataLayout(mode)
    } else {
        showLiveLayout()
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

    show('content-row')

    const rightHeader    = document.getElementById('right-header')!
    const rightSubHeader = document.getElementById('right-sub-header')!
    const rightFooter    = document.getElementById('right-footer')!
    rightHeader.innerHTML    = ''
    rightSubHeader.innerHTML = ''
    rightFooter.innerHTML    = ''

    // Align data-entry section, seat selector, and divider to the H-score table width
    const hscoreW = getHscoreWidth()
    rightHeader.style.maxWidth    = hscoreW
    rightSubHeader.style.maxWidth = hscoreW

    if (mode === 'Auction Mode') {
        renderAuctionEntry(rightHeader, _onEvaluate)
    } else {
        renderDraftBoard(rightHeader, _onEvaluate)
    }

    // Seat selector lives below the divider, directly above the H-score table
    renderSeatSelector(rightSubHeader)

    // Add Candidates / My Team tab bar for both Draft and Auction Mode
    if (mode === 'Auction Mode') {
        buildAuctionTabBar(rightSubHeader)
        activateAuctionTab(currentAuctionTab)

        // Refresh G-score table when the auction board changes (pick/undo/clear)
        rightHeader.addEventListener('auction-board-change', () => {
            if (currentAuctionTab === 'my-team') refreshAuctionGScore()
        })
    } else {
        hide('auction-gscore')
        buildDraftTabBar(rightSubHeader)
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

    const rightHeader    = document.getElementById('right-header')!
    const rightSubHeader = document.getElementById('right-sub-header')!
    const rightFooter    = document.getElementById('right-footer')!
    rightHeader.innerHTML    = ''
    rightSubHeader.innerHTML = ''
    rightFooter.innerHTML    = ''

    const hscoreW = getHscoreWidth()
    rightSubHeader.style.maxWidth = hscoreW

    renderSeatSelector(rightSubHeader)

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightFooter.append(stub)
}

// ─── Season layout ────────────────────────────────────────────────────────────

/** Renders the season mode layout: tab nav (Waiver / Trading / Rosters) with content panes. */
function showSeasonLayout(): void {
    hide('live-bar')
    hide('left-panel')

    // Clear sub-header so the seat selector from draft/auction mode doesn't bleed in
    const rightSubHeader = document.getElementById('right-sub-header')!
    rightSubHeader.innerHTML    = ''
    rightSubHeader.style.maxWidth = ''

    show('season-nav')

    if (!seasonNavBuilt) {
        buildSeasonNav()
        seasonNavBuilt = true
        // Pre-populate roster DOM elements so the Waiver tab can read sr-player-* inputs
        // on first visit, before the user has manually opened the Rosters tab.
        renderSeasonRosters(
            document.getElementById('rosters-left')!,
            document.getElementById('rosters-right')!,
        )
    }

    // Only default to Waiver on first entry; leave the active tab alone on subsequent calls
    if (!currentSeasonTab) {
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

    // Update active button styling
    document.querySelectorAll('.season-tab-btn').forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId)
    })

    if (tabId === 'waiver') {
        hide('season-rosters-row')
        hide('season-trading-row')
        show('content-row')

        const rightHeader = document.getElementById('right-header')!
        rightHeader.innerHTML = ''
        rightHeader.style.maxWidth = getHscoreWidth()
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

// ─── Draft tab bar (Candidates / My Team) ────────────────────────────────

/** Creates the draft tab buttons in the given container. */
function buildDraftTabBar(container: HTMLElement): void {
    const bar = document.createElement('div')
    bar.className = 'draft-tab-bar'

    for (const tab of [
        { id: 'candidates', label: 'Candidates' },
        { id: 'my-team',    label: 'My Team' },
    ]) {
        const btn = document.createElement('button')
        btn.className   = 'season-tab-btn'
        btn.textContent = tab.label
        btn.dataset.tab = tab.id
        if (tab.id === currentDraftTab) btn.classList.add('active')
        btn.addEventListener('click', () => activateDraftTab(tab.id))
        bar.append(btn)
    }

    container.append(bar)
}

/** Switches between the H-score candidate table and the My Team G-score view. */
function activateDraftTab(tabId: string): void {
    currentDraftTab = tabId

    document.querySelectorAll('.draft-tab-bar .season-tab-btn').forEach(btn => {
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

/** Rebuilds the G-score table for the current seat's draft picks. */
function refreshDraftGScore(): void {
    const container = document.getElementById('draft-gscore')
    if (!container) return
    const seat = currentSeat
    if (!seat) { container.innerHTML = ''; return }
    const { player_assignments } = getDraftState()
    const playerNames = player_assignments[seat] ?? []
    const width = getHscoreWidth()
    renderTeamGScoreTable(playerNames, container, width)
}

// ─── Auction tab bar (Candidates / My Team) ──────────────────────────────

/** Creates the auction tab buttons in the given container. */
function buildAuctionTabBar(container: HTMLElement): void {
    const bar = document.createElement('div')
    bar.className = 'auction-tab-bar'

    for (const tab of [
        { id: 'candidates', label: 'Candidates' },
        { id: 'my-team',    label: 'My Team' },
    ]) {
        const btn = document.createElement('button')
        btn.className   = 'season-tab-btn'   // reuse season tab styling
        btn.textContent = tab.label
        btn.dataset.tab = tab.id
        if (tab.id === currentAuctionTab) btn.classList.add('active')
        btn.addEventListener('click', () => activateAuctionTab(tab.id))
        bar.append(btn)
    }

    container.append(bar)
}

/** Switches between the H-score candidate table and the My Team G-score view. */
function activateAuctionTab(tabId: string): void {
    currentAuctionTab = tabId

    // Update active button styling
    document.querySelectorAll('.auction-tab-bar .season-tab-btn').forEach(btn => {
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

/** Rebuilds the G-score table for the current seat's auction picks. */
function refreshAuctionGScore(): void {
    const container = document.getElementById('auction-gscore')
    if (!container) return
    const seat = currentSeat
    if (!seat) { container.innerHTML = ''; return }
    const { player_assignments } = getAuctionState()
    const playerNames = player_assignments[seat] ?? []
    const width = getHscoreWidth()
    renderTeamGScoreTable(playerNames, container, width)
}

// ─── Seat selector ────────────────────────────────────────────────────────────

/** Shows or hides the "Updating..." spinner next to the seat selector. */
export function setEvaluating(active: boolean): void {
    const el = document.getElementById('eval-indicator')
    if (el) (el as HTMLElement).style.visibility = active ? 'visible' : 'hidden'
}

/** Builds the "Which team are you?" selector. Changing the seat triggers an immediate re-evaluate. */
function renderSeatSelector(container: HTMLElement): void {
    const teamNames = readTeamNames()
    const wrap = document.createElement('div')
    wrap.className = 'seat-selector-wrap'

    const sel = makeCustomSelect(
        'seat-select',
        teamNames.map(n => ({ value: n, label: n })),
    )

    const indicator = document.createElement('span')
    indicator.id        = 'eval-indicator'
    indicator.className = 'eval-indicator'
    indicator.textContent = 'Updating...'

    const label = document.createElement('div')
    label.className   = 'pick-control-label'
    label.textContent = 'Which team are you?'

    const selRow = document.createElement('div')
    selRow.style.cssText = 'display:flex;align-items:center;gap:10px'
    selRow.append(label, sel.element, indicator)
    wrap.append(selRow)

    // Initialize currentSeat to the first team if not yet set
    if (!currentSeat && teamNames.length > 0) currentSeat = teamNames[0]
    if (currentSeat) sel.setValue(currentSeat)
    sel.element.addEventListener('change', () => {
        currentSeat = sel.getValue() ?? ''
        // Refresh G-score table if the My Team tab is active
        if (currentAuctionTab === 'my-team') refreshAuctionGScore()
        if (currentDraftTab   === 'my-team') refreshDraftGScore()
        // Re-evaluate immediately when the user switches their seat
        if (_onEvaluate) {
            const result = _onEvaluate()
            if (result instanceof Promise) result.catch(err => console.error('Evaluate failed:', err))
        }
    })

    container.append(wrap)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Returns the H-score table's current width string.
 *  Falls back to getComputedStyle when the inline style isn't set yet (initial load). */
function getHscoreWidth(): string {
    const el = document.getElementById('hscoretable') as HTMLTableElement
    return el.style.width || getComputedStyle(el).width
}

function readTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

function show(id: string): void {
    const el = document.getElementById(id)
    if (el) el.style.display = ''
}

function hide(id: string): void {
    const el = document.getElementById(id)
    if (el) el.style.display = 'none'
}
