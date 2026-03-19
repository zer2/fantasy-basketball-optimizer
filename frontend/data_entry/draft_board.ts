// data_entry/draft_board.ts
// Renders the draft pick control and draft board grid for Draft Mode + own data.
// Mirrors make_drafting_tab_own_data() in src/tabs/drafting.py.

import { makeCustomSelect } from '../custom_select.js'
import { getCandidatePlayers } from '../app_state.js'
import { makeDebouncer, Debouncer } from '../helper_functions.js'
import { runEvaluate } from '../api/session.js'
import {
    DraftConfig,
    getPickRow, getPickDrafter, getDrafted, getTeamNames, getNDrafters, getNPicks, getConfigKey,
    resetDraftState, applyDraftConfig,
    recordDraftPick, clearDraftPick, clearAllDraftPicks,
    advanceDraftPick, goBackDraftPick,
} from './draft_state.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let _debouncer: Debouncer | null = null

const ROUND_W = 46   // px — Round label column
const TEAM_W  = 85   // px — per-drafter column

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Clears the draft board state. Call when the player pool changes (e.g. data source switch)
 * so drafted names from the old dataset are not sent to the backend.
 * The next renderDraftBoard call will reinitialise from current sidebar values.
 */
export function resetDraftBoard(): void {
    resetDraftState()
    _debouncer?.cancel()
}

/** Renders the draft board UI into the container. Resets state if sidebar config changed. */
export function renderDraftBoard(container: HTMLElement): void {
    if (!_debouncer) {
        _debouncer = makeDebouncer(() => { runEvaluate().catch(err => console.error('Draft evaluate failed:', err)) })
    }
    const cfg = readDraftConfig()

    // Reset state if league settings changed (different drafter/pick counts or teams)
    if (cfg.key !== getConfigKey()) {
        applyDraftConfig(cfg)
    }

    container.innerHTML = ''
    container.append(buildPickControl(container))
    container.append(buildDraftBoard())
}

// ─── Pick control ─────────────────────────────────────────────────────────────

/** Builds the pick control row: player dropdown, lock-in / undo / clear buttons. */
function buildPickControl(container: HTMLElement): HTMLElement {
    const wrap = document.createElement('div')

    const pickRow     = getPickRow()
    const pickDrafter = getPickDrafter()
    const nPicks      = getNPicks()
    const nDrafters   = getNDrafters()
    const teamNames   = getTeamNames()

    const isDone = pickRow >= nPicks

    // Inline row: label + player select + action buttons on the same line
    const row = document.createElement('div')
    row.className = 'pick-control-row'

    const label = document.createElement('div')
    label.className = 'pick-control-label'
    label.textContent = isDone
        ? 'Draft complete'
        : `Select Pick ${pickRow + 1} for ${teamNames[pickDrafter] ?? `Drafter ${pickDrafter + 1}`}`
    row.append(label)

    const draftedSet = new Set(getDrafted().flat().filter(Boolean) as string[])
    const available  = getCandidatePlayers().map(p => p.name).filter(n => !draftedSet.has(n))
    const sel = makeCustomSelect('draft-pick-select', available.map(n => ({ value: n, label: n })))
    sel.element.style.flex = '1'
    row.append(sel.element)

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    const lockBtn = document.createElement('button')
    lockBtn.className = 'pick-btn'
    lockBtn.textContent = 'Lock in selection'
    lockBtn.disabled = isDone || available.length === 0
    lockBtn.addEventListener('click', () => {
        const chosen = sel.getValue()
        if (!chosen || getPickRow() >= getNPicks()) return
        recordDraftPick(getPickRow(), getPickDrafter(), chosen)
        advanceDraftPick()
        renderDraftBoard(container)
        _debouncer?.fire()
    })

    const undoBtn = document.createElement('button')
    undoBtn.className = 'pick-btn'
    undoBtn.textContent = 'Undo previous selection'
    undoBtn.disabled = pickRow === 0 && pickDrafter === 0
    undoBtn.addEventListener('click', () => {
        goBackDraftPick()
        clearDraftPick(getPickRow(), getPickDrafter())
        renderDraftBoard(container)
        _debouncer?.fire()
    })

    const clearBtn = document.createElement('button')
    clearBtn.className = 'pick-btn'
    clearBtn.textContent = 'Clear draft board'
    clearBtn.addEventListener('click', () => {
        clearAllDraftPicks()
        renderDraftBoard(container)
        runEvaluate().catch(err => console.error('Evaluate after clear failed:', err))
    })

    btns.append(lockBtn, undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Draft board table ────────────────────────────────────────────────────────

/** Builds the draft board grid table: rounds × drafters with serpentine pick highlighting. */
function buildDraftBoard(): HTMLElement {
    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'

    const nPicks    = getNPicks()
    const nDrafters = getNDrafters()
    const teamNames = getTeamNames()
    const drafted   = getDrafted()
    const pickRow   = getPickRow()
    const pickDrafter = getPickDrafter()

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
            const cell   = row.insertCell()
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

/** Reads current sidebar league settings and returns them with a composite key for change detection. */
function readDraftConfig(): DraftConfig {
    const nD  = parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value) || 12
    const nP  = parseInt((document.getElementById('ls-n-picks')    as HTMLInputElement).value) || 13
    const src = (document.getElementById('ps-data-type') as HTMLInputElement).value
    const names = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    return { nDrafters: nD, nPicks: nP, teamNames: names, key: `${nD}:${nP}:${src}:${names.join(',')}` }
}
