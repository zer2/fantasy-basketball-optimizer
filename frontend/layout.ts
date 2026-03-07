// layout.ts
// Manages the main-content area layout based on mode (Draft / Auction / Season)
// and platform (own data vs. live).  Call initLayout() once after the sidebar is built.

import { renderDraftBoard }    from './data_entry/draft_board.js'
import { renderAuctionEntry }  from './data_entry/auction_entry.js'
import { renderSeasonRosters } from './data_entry/season/season_rosters.js'
import { renderSeasonTrading } from './data_entry/season/season_trading.js'
import { renderWaiverControls } from './data_entry/season/season_waiver.js'
import { makeCustomSelect }    from './custom_select.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let seasonNavBuilt  = false
let currentSeasonTab = ''   // tracks active season tab; '' means no tab activated yet
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
    const hscoreW = (document.getElementById('realtable') as HTMLTableElement).style.width
    rightHeader.style.maxWidth    = hscoreW
    rightSubHeader.style.maxWidth = hscoreW

    if (mode === 'Auction Mode') {
        renderAuctionEntry(rightHeader, _onEvaluate)
    } else {
        renderDraftBoard(rightHeader, _onEvaluate)
    }

    // Seat selector lives below the divider, directly above the H-score table
    renderSeatSelector(rightSubHeader)

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightFooter.append(stub)
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

    const hscoreW = (document.getElementById('realtable') as HTMLTableElement).style.width
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

// ─── Seat selector ────────────────────────────────────────────────────────────

/** Builds the "Which team are you?" selector. Changing the seat triggers an immediate re-evaluate. */
function renderSeatSelector(container: HTMLElement): void {
    const teamNames = readTeamNames()
    const wrap = document.createElement('div')
    wrap.className = 'seat-selector-wrap'

    const label = document.createElement('div')
    label.className   = 'pick-control-label'
    label.textContent = 'Which team are you?'
    wrap.append(label)

    const sel = makeCustomSelect(
        'seat-select',
        teamNames.map(n => ({ value: n, label: n })),
    )
    if (currentSeat) sel.setValue(currentSeat)
    sel.element.addEventListener('change', () => {
        currentSeat = sel.getValue() ?? ''
        // Re-evaluate immediately when the user switches their seat
        if (_onEvaluate) {
            const result = _onEvaluate()
            if (result instanceof Promise) result.catch(err => console.error('Evaluate failed:', err))
        }
    })
    wrap.append(sel.element)

    container.append(wrap)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

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
