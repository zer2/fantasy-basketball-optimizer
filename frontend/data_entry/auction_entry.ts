// data_entry/auction_entry.ts
// Renders the auction pick control and board grid for Auction Mode + own data.
// Mirrors the draft board structure: pick control on top, grid below.

import { makeCustomSelect } from '../custom_select.js'
import { getPlayers } from '../script.js'

// ─── Types ────────────────────────────────────────────────────────────────────

interface AuctionPick { player: string; cost: number }

// ─── Module state ─────────────────────────────────────────────────────────────

let picks: (AuctionPick | null)[][] = []   // [row][drafter]
let history: [number, number][]     = []   // undo stack: [row, drafter]
let teamNames:  string[] = []
let nDrafters = 0
let nPicks    = 0
let cashPerTeam = 0
let configKey = ''

const ROUND_W = 46
const TEAM_W  = 85

// ─── Public API ───────────────────────────────────────────────────────────────

/** Returns the current auction state for use in /evaluate requests. */
export function getAuctionState(): {
    player_assignments: Record<string, string[]>
    remaining_cash: Record<string, number>
} {
    const player_assignments: Record<string, string[]> = {}
    const remaining_cash: Record<string, number> = {}
    for (let d = 0; d < nDrafters; d++) {
        const name = teamNames[d] ?? `Drafter ${d + 1}`
        const teamPicks = picks.map(row => row[d]).filter(Boolean) as AuctionPick[]
        player_assignments[name] = teamPicks.map(p => p.player)
        const spent = teamPicks.reduce((sum, p) => sum + p.cost, 0)
        remaining_cash[name] = cashPerTeam - spent
    }
    return { player_assignments, remaining_cash }
}

export function renderAuctionEntry(container: HTMLElement): void {
    const cfg = readConfig()

    if (cfg.key !== configKey) {
        picks       = Array.from({ length: cfg.nPicks }, () => Array(cfg.nDrafters).fill(null))
        history     = []
        teamNames   = cfg.teamNames
        nDrafters   = cfg.nDrafters
        nPicks      = cfg.nPicks
        cashPerTeam = cfg.cashPerTeam
        configKey   = cfg.key
    }

    container.innerHTML = ''
    container.append(buildPickControl(container))
    container.append(buildAuctionBoard())
}

// ─── Pick control ─────────────────────────────────────────────────────────────

function buildPickControl(container: HTMLElement): HTMLElement {
    const wrap = document.createElement('div')

    const row = document.createElement('div')
    row.className = 'pick-control-row'

    // Player dropdown — grows to fill available space
    const available = getAvailablePlayers()
    const playerSel = makeCustomSelect(
        'auction-pick-player',
        [{ value: '', label: '' }, ...available.map(n => ({ value: n, label: n }))],
    )
    playerSel.element.style.width = '100%'
    const playerCol = makePickCol('Player', playerSel.element)
    playerCol.style.flex = '1'
    row.append(playerCol)

    // Cost input — fixed width
    const costInput = document.createElement('input')
    costInput.type        = 'number'
    costInput.min         = '0'
    costInput.placeholder = '$'
    costInput.className   = 'auction-cost-input'
    row.append(makePickCol('Cost', costInput))

    // Team dropdown — grows to fill available space
    const teamSel = makeCustomSelect(
        'auction-pick-team',
        [{ value: '', label: '' }, ...teamNames.map(n => ({ value: n, label: n }))],
    )
    teamSel.element.style.width = '100%'
    const teamCol = makePickCol('Drafter', teamSel.element)
    teamCol.style.flex = '1'
    row.append(teamCol)

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    const lockBtn = document.createElement('button')
    lockBtn.className   = 'pick-btn'
    lockBtn.textContent = 'Lock in selection'
    lockBtn.addEventListener('click', () => {
        const player = playerSel.getValue()
        const team   = teamSel.getValue()
        const cost   = parseFloat(costInput.value)
        if (!player || !team || isNaN(cost) || cost <= 0) return
        const dIdx    = teamNames.indexOf(team)
        const emptyRow = picks.findIndex(r => r[dIdx] === null)
        if (emptyRow === -1) return   // team is full
        picks[emptyRow][dIdx] = { player, cost }
        history.push([emptyRow, dIdx])
        renderAuctionEntry(container)
    })

    const undoBtn = document.createElement('button')
    undoBtn.className   = 'pick-btn'
    undoBtn.textContent = 'Undo previous selection'
    undoBtn.disabled    = history.length === 0
    undoBtn.addEventListener('click', () => {
        const last = history.pop()
        if (!last) return
        picks[last[0]][last[1]] = null
        renderAuctionEntry(container)
    })

    const clearBtn = document.createElement('button')
    clearBtn.className   = 'pick-btn'
    clearBtn.textContent = 'Clear auction board'
    clearBtn.addEventListener('click', () => {
        picks   = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
        history = []
        renderAuctionEntry(container)
    })

    btns.append(lockBtn, undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Auction board table ──────────────────────────────────────────────────────

function buildAuctionBoard(): HTMLElement {
    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'

    const table = document.createElement('table')
    table.className    = 'entry-table'
    table.style.width    = '100%'
    table.style.minWidth = (ROUND_W + nDrafters * TEAM_W) + 'px'

    // Header: Round | Team1 | Team2 | …
    const thead = table.createTHead()
    const hrow  = thead.insertRow()
    const roundTh = document.createElement('th')
    roundTh.textContent = 'Round'
    roundTh.style.width = ROUND_W + 'px'
    hrow.append(roundTh)
    for (const name of teamNames) {
        const th = document.createElement('th')
        th.textContent = name
        hrow.append(th)
    }

    // Body rows
    const tbody = table.createTBody()
    for (let r = 0; r < nPicks; r++) {
        const row = tbody.insertRow()

        const roundCell = row.insertCell()
        roundCell.className   = 'entry-cell-label'
        roundCell.textContent = String(r + 1)

        for (let d = 0; d < nDrafters; d++) {
            const cell = row.insertCell()
            const pick = picks[r][d]
            if (pick) {
                const nameEl = document.createElement('div')
                nameEl.className   = 'auction-cell-name'
                nameEl.textContent = pick.player

                const costEl = document.createElement('div')
                costEl.className   = 'auction-cell-cost'
                costEl.textContent = `$${pick.cost}`

                cell.append(nameEl, costEl)
                cell.classList.add('drafted')
            }
        }
    }

    // Footer: remaining budget per team
    const tfoot = table.createTFoot()
    const frow  = tfoot.insertRow()

    const budgetLabel = document.createElement('td')
    budgetLabel.className   = 'entry-cell-label auction-budget-label'
    budgetLabel.textContent = 'Remaining'
    frow.append(budgetLabel)

    for (let d = 0; d < nDrafters; d++) {
        const spent = picks.reduce((sum, r) => sum + (r[d] ? r[d]!.cost : 0), 0)
        const td = document.createElement('td')
        td.className   = 'auction-budget-cell'
        td.textContent = `$${cashPerTeam - spent}`
        frow.append(td)
    }

    scroll.append(table)
    return scroll
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makePickCol(labelText: string, input: HTMLElement): HTMLElement {
    const col = document.createElement('div')
    col.className = 'pick-col'
    const lbl = document.createElement('div')
    lbl.className   = 'pick-col-label'
    lbl.textContent = labelText
    col.append(lbl, input)
    return col
}

function getAvailablePlayers(): string[] {
    const allPlayers = getPlayers().map(p => p.name)
    const pickedSet  = new Set(
        picks.flat().filter(Boolean).map(p => (p as AuctionPick).player),
    )
    return allPlayers.filter(n => !pickedSet.has(n))
}

function readConfig(): {
    nDrafters: number; nPicks: number; cashPerTeam: number
    teamNames: string[]; key: string
} {
    const nD   = parseInt((document.getElementById('ls-n-drafters')   as HTMLInputElement).value)  || 12
    const nP   = parseInt((document.getElementById('ls-n-picks')       as HTMLInputElement).value)  || 13
    const cash = parseInt((document.getElementById('ls-cash-per-team') as HTMLInputElement).value)  || 200
    const names = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    return { nDrafters: nD, nPicks: nP, cashPerTeam: cash, teamNames: names,
             key: `${nD}:${nP}:${cash}:${names.join(',')}` }
}
