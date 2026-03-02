// layout.ts
// Manages the main-content area layout based on mode (Draft / Auction / Season)
// and platform (own data vs. live).  Call initLayout() once after the sidebar is built.

import { renderDraftBoard }    from './data_entry/draft_board.js'
import { renderAuctionEntry }  from './data_entry/auction_entry.js'
import { renderSeasonRosters } from './data_entry/season_rosters.js'
import { makeCustomSelect }    from './custom_select.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let seasonNavBuilt = false
let currentSeat    = ''

// ─── Public API ───────────────────────────────────────────────────────────────

export function initLayout(): void {
    applyLayout()
    // Re-apply whenever mode or platform changes
    document.getElementById('ls-mode')!.parentElement!
        .addEventListener('change', applyLayout)
    document.getElementById('ls-platform')!.parentElement!
        .addEventListener('change', applyLayout)
}

// ─── Layout dispatcher ────────────────────────────────────────────────────────

function applyLayout(): void {
    const mode     = (document.getElementById('ls-mode')     as HTMLInputElement).value
    const platform = (document.getElementById('ls-platform') as HTMLInputElement).value

    const isOwnData = platform === 'Enter your own data'
    const isSeason  = mode === 'Season Mode'

    if (isSeason) {
        showSeasonLayout()
    } else if (isOwnData) {
        showOwnDataLayout(mode)
    } else {
        showLiveLayout()
    }
}

// ─── Own-data layout (Draft or Auction) ───────────────────────────────────────

function showOwnDataLayout(mode: string): void {
    hide('season-nav')
    hide('live-bar')
    hide('left-panel')
    hide('season-rosters-row')
    hide('season-trading-row')

    show('content-row')
    show('candidate-panel')

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
        renderAuctionEntry(rightHeader)
    } else {
        renderDraftBoard(rightHeader)
    }

    // Seat selector lives below the divider, directly above the H-score table
    renderSeatSelector(rightSubHeader)

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightFooter.append(stub)
}

// ─── Live-platform layout (Draft or Auction) ──────────────────────────────────

function showLiveLayout(): void {
    hide('season-nav')
    hide('live-bar')
    hide('left-panel')
    hide('season-rosters-row')
    hide('season-trading-row')

    show('content-row')
    show('candidate-panel')

    const rightHeader    = document.getElementById('right-header')!
    const rightSubHeader = document.getElementById('right-sub-header')!
    const rightFooter    = document.getElementById('right-footer')!
    rightHeader.innerHTML    = ''
    rightSubHeader.innerHTML = ''
    rightFooter.innerHTML    = ''

    const hscoreW = (document.getElementById('realtable') as HTMLTableElement).style.width
    rightSubHeader.style.maxWidth = hscoreW

    renderSeatSelector(rightSubHeader, 'Refresh & Run Algorithm')

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightFooter.append(stub)
}

// ─── Season layout ────────────────────────────────────────────────────────────

function showSeasonLayout(): void {
    hide('live-bar')
    hide('left-panel')

    show('season-nav')

    if (!seasonNavBuilt) {
        buildSeasonNav()
        seasonNavBuilt = true
    }

    // Default to Waiver tab on first show
    activateSeasonTab('waiver')
}

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

function activateSeasonTab(tabId: string): void {
    // Update active button styling
    document.querySelectorAll('.season-tab-btn').forEach(btn => {
        btn.classList.toggle('active', (btn as HTMLElement).dataset.tab === tabId)
    })

    if (tabId === 'waiver') {
        hide('season-rosters-row')
        hide('season-trading-row')
        show('content-row')
        show('candidate-panel')

        const rightHeader = document.getElementById('right-header')!
        rightHeader.innerHTML = ''
        document.getElementById('right-footer')!.innerHTML = ''
        renderWaiverControls(rightHeader)

    } else if (tabId === 'trading') {
        hide('content-row')
        hide('season-rosters-row')
        show('season-trading-row')

    } else if (tabId === 'rosters') {
        hide('content-row')
        hide('season-trading-row')
        show('season-rosters-row')

        const rostersLeft  = document.getElementById('rosters-left')!
        const rostersRight = document.getElementById('rosters-right')!
        renderSeasonRosters(rostersLeft, rostersRight)
    }
}

// ─── Waiver controls ──────────────────────────────────────────────────────────

function renderWaiverControls(container: HTMLElement): void {
    const teamNames   = readTeamNames()
    const bar = document.createElement('div')
    bar.className = 'waiver-filter-bar'

    const teamSel = makeCustomSelect(
        'waiver-team-select',
        teamNames.map(n => ({ value: n, label: n })),
    )
    bar.append(teamSel.element)

    container.append(bar)
}

// ─── Seat selector ────────────────────────────────────────────────────────────

function renderSeatSelector(container: HTMLElement, btnLabel = 'Run algorithm'): void {
    const teamNames = readTeamNames()
    const wrap = document.createElement('div')
    wrap.className = 'seat-selector-wrap'

    const label = document.createElement('div')
    label.className   = 'pick-control-label'
    label.textContent = 'Which team are you?'
    wrap.append(label)

    const row = document.createElement('div')
    row.className = 'pick-control-row'

    const sel = makeCustomSelect(
        'seat-select',
        teamNames.map(n => ({ value: n, label: n })),
    )
    if (currentSeat) sel.setValue(currentSeat)
    sel.element.addEventListener('change', () => {
        currentSeat = sel.getValue() ?? ''
    })
    row.append(sel.element)

    const runBtn = document.createElement('button')
    runBtn.className   = 'pick-btn'
    runBtn.textContent = btnLabel
    runBtn.addEventListener('click', () => {
        // TODO: trigger backend call
    })
    row.append(runBtn)

    wrap.append(row)
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
