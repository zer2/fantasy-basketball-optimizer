// data_entry/draft_board.ts
// Renders the draft pick control and draft board grid for Draft Mode + own data.
// Mirrors make_drafting_tab_own_data() in src/tabs/drafting.py.

import { makeCustomSelect } from '../custom_select.js'
import { getPlayers } from '../script.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let pickRow     = 0
let pickDrafter = 0
let drafted: (string | null)[][] = []   // [row][drafter], null = not yet picked
let teamNames:  string[] = []
let nDrafters = 0
let nPicks    = 0
let configKey = ''   // detects sidebar changes that require a reset
let _onPick: (() => void | Promise<void>) | undefined

const ROUND_W = 46   // px — Round label column
const TEAM_W  = 85   // px — per-drafter column

// ─── Public API ───────────────────────────────────────────────────────────────

/** Returns the current draft state for use in /evaluate requests. */
export function getDraftState(): { player_assignments: Record<string, string[]> } {
    const player_assignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const name = teamNames[d] ?? `Drafter ${d + 1}`
        player_assignments[name] = drafted.map(row => row[d]).filter(Boolean) as string[]
    }
    return { player_assignments }
}

export function renderDraftBoard(
    container: HTMLElement,
    onPick?: () => void | Promise<void>,
): void {
    if (onPick !== undefined) _onPick = onPick
    const cfg = readConfig()

    // Reset state if league settings changed (different drafter/pick counts or teams)
    if (cfg.key !== configKey) {
        pickRow     = 0
        pickDrafter = 0
        drafted     = Array.from({ length: cfg.nPicks }, () => Array(cfg.nDrafters).fill(null))
        teamNames   = cfg.teamNames
        nDrafters   = cfg.nDrafters
        nPicks      = cfg.nPicks
        configKey   = cfg.key
    }

    container.innerHTML = ''
    container.append(buildPickControl(container))
    container.append(buildDraftBoard())
}

// ─── Pick control ─────────────────────────────────────────────────────────────

function buildPickControl(container: HTMLElement): HTMLElement {
    const wrap = document.createElement('div')

    const isDone = pickRow >= nPicks

    const label = document.createElement('div')
    label.className = 'pick-control-label'
    label.textContent = isDone
        ? 'Draft complete'
        : `Select Pick ${pickRow + 1} for ${teamNames[pickDrafter] ?? `Drafter ${pickDrafter + 1}`}`
    wrap.append(label)

    // Inline row: player select + action buttons on the same line
    const row = document.createElement('div')
    row.className = 'pick-control-row'

    const available = getAvailablePlayers()
    const sel = makeCustomSelect('draft-pick-select', available.map(n => ({ value: n, label: n })))
    row.append(sel.element)

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    const lockBtn = document.createElement('button')
    lockBtn.className = 'pick-btn'
    lockBtn.textContent = 'Lock in selection'
    lockBtn.disabled = isDone || available.length === 0
    lockBtn.addEventListener('click', () => {
        const chosen = sel.getValue()
        if (!chosen || isDone) return
        drafted[pickRow][pickDrafter] = chosen
        advance()
        renderDraftBoard(container)
        if (_onPick) {
            const result = _onPick()
            if (result instanceof Promise) {
                lockBtn.disabled = true
                result.finally(() => { lockBtn.disabled = false })
                       .catch(err => console.error('Evaluate after pick failed:', err))
            }
        }
    })

    const undoBtn = document.createElement('button')
    undoBtn.className = 'pick-btn'
    undoBtn.textContent = 'Undo previous selection'
    undoBtn.disabled = pickRow === 0 && pickDrafter === 0
    undoBtn.addEventListener('click', () => {
        goBack()
        drafted[pickRow][pickDrafter] = null
        renderDraftBoard(container)
        if (_onPick) {
            const result = _onPick()
            if (result instanceof Promise) {
                result.catch(err => console.error('Evaluate after undo failed:', err))
            }
        }
    })

    const clearBtn = document.createElement('button')
    clearBtn.className = 'pick-btn'
    clearBtn.textContent = 'Clear draft board'
    clearBtn.addEventListener('click', () => {
        pickRow = 0
        pickDrafter = 0
        drafted = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
        renderDraftBoard(container)
        if (_onPick) {
            const result = _onPick()
            if (result instanceof Promise) {
                result.catch(err => console.error('Evaluate after clear failed:', err))
            }
        }
    })

    btns.append(lockBtn, undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Draft board table ────────────────────────────────────────────────────────

function buildDraftBoard(): HTMLElement {
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

    // Data rows
    const tbody = table.createTBody()
    for (let r = 0; r < nPicks; r++) {
        const row = tbody.insertRow()

        const roundCell = row.insertCell()
        roundCell.className = 'entry-cell-label'
        roundCell.textContent = String(r + 1)

        for (let d = 0; d < nDrafters; d++) {
            const cell = row.insertCell()
            const player = drafted[r][d]
            if (player) {
                cell.textContent = player
                cell.classList.add('drafted')
            } else if (r === pickRow && d === pickDrafter) {
                cell.classList.add('current-pick')
            }
        }
    }

    scroll.append(table)
    return scroll
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function getAvailablePlayers(): string[] {
    const allPlayers = getPlayers().map(p => p.name)
    const draftedSet = new Set(drafted.flat().filter(Boolean) as string[])
    return allPlayers.filter(n => !draftedSet.has(n))
}

/** Advance pick position in serpentine order. */
function advance(): void {
    const isForward = pickRow % 2 === 0   // even rounds: left→right
    if (isForward) {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    } else {
        if (pickDrafter > 0) { pickDrafter--; return }
    }
    // Move to next round
    if (pickRow < nPicks - 1) {
        pickRow++
        pickDrafter = pickRow % 2 === 0 ? 0 : nDrafters - 1
    } else {
        pickRow = nPicks   // draft complete sentinel
    }
}

/** Move pick position back one step in serpentine order. */
function goBack(): void {
    if (pickRow >= nPicks) { pickRow = nPicks - 1 }
    const isForward = pickRow % 2 === 0
    if (isForward) {
        if (pickDrafter > 0) { pickDrafter--; return }
    } else {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    }
    if (pickRow > 0) {
        pickRow--
        const prevForward = pickRow % 2 === 0
        pickDrafter = prevForward ? nDrafters - 1 : 0
    }
}

function readConfig(): { nDrafters: number; nPicks: number; teamNames: string[]; key: string } {
    const nD = parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value) || 12
    const nP = parseInt((document.getElementById('ls-n-picks')    as HTMLInputElement).value) || 13
    const names = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    return { nDrafters: nD, nPicks: nP, teamNames: names, key: `${nD}:${nP}:${names.join(',')}` }
}

