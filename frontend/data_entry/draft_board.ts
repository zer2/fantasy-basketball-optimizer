// data_entry/draft_board.ts
// Renders the draft pick control and draft board grid for Draft Mode + own data.
// Mirrors make_drafting_tab_own_data() in src/tabs/drafting.py.

import { makeCustomSelect } from '../custom_select.js'
import { readRequiredIntInput } from '../helper_functions.js'
import { getPlayerResults, getCandidatePlayerResults, getPlayerNamesByGScore, getSessionPhase, getCurrentSeat, setCurrentSeat } from '../app_state.js'
import { makeDebouncer } from '../api/session.js'
import { runEvaluate, clearFullTeamResult } from '../api/draft_and_auction_session.js'
import { setAutopilotOn, setAutopilotOff } from '../api/session.js'
import { getDrafterMethod } from './drafter_methods.js'
import { makeMethodDropdown } from './method_dropdown.js'
import { getTeamLabel, makeTeamLabelInput } from './team_labels.js'
import {
    DraftConfig,
    getPickRow, getPickDrafter, getDrafted, getTeamNames, getNDrafters, getNPicks, getConfigKey,
    resetDraftState, applyDraftConfig,
    recordDraftPick, clearDraftPick, clearAllDraftPicks,
    advanceDraftPick, goBackDraftPick,
} from './draft_state.js'

// ─── Module state ─────────────────────────────────────────────────────────────

const _draftDebouncer = makeDebouncer(() => { runEvaluate().catch(err => console.error('Draft evaluate failed:', err)) })

// The header (team-label inputs + method dropdowns) is built ONCE — on first render, on a
// config change, or if the container was cleared (e.g. applyLayout). Only the pick control and
// the table body rebuild per pick, so editing a label isn't interrupted mid-keystroke.
let boardTable:       HTMLTableElement | null = null
let pickControlSlot:  HTMLElement      | null = null

// Pick-control listeners (the candidate select) are detached every render; header listeners
// (label inputs + method dropdowns) only when the header itself is rebuilt. Without this, each
// rebuild would leave the previous custom select's internal listeners (~9) bound to detached nodes.
let pickListenerController:   AbortController | null = null
let headerListenerController: AbortController | null = null

let _autopilotRunning = false

const ROUND_W = 32   // px — Round label column
const TEAM_W  = 50   // px — per-drafter column (min-width floor only; table fills available width)

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Clears the draft board state. Call when the player pool changes (e.g. data source switch)
 * so drafted names from the old dataset are not sent to the backend.
 * The next renderDraftBoard call will reinitialise from current sidebar values.
 */
export function resetDraftBoard(): void {
    resetDraftState()
    _draftDebouncer?.cancel()
}

/** Renders the draft board UI into the container. Resets state if sidebar config changed. */
export function renderDraftBoard(container: HTMLElement): void {
    const cfg = readDraftConfig()
    const configChanged = cfg.key !== getConfigKey()

    // Reset state if league settings changed (different drafter/pick counts, reversal, data source)
    if (configChanged) applyDraftConfig(cfg)

    // (Re)build the persistent scaffold (pick-control slot + table with its header) on first
    // render, on a config change, or if the table was detached (container cleared elsewhere).
    if (configChanged || !boardTable || !container.contains(boardTable)) {
        buildScaffold(container)
    }

    // Pick control: rebuilt every render — its candidate dropdown changes each pick.
    pickListenerController?.abort()
    pickListenerController = new AbortController()
    pickControlSlot!.innerHTML = ''
    pickControlSlot!.append(buildPickControl(container))

    // Body: rebuilt every render (text + classes only; no components → no header churn).
    rebuildBoardBody(boardTable!)

    // Auto-fire autopilot when the current drafter is an autopilot drafter and the loop is not already running.
    // Guard on G-scores being loaded — on initial page render the session hasn't been created yet,
    // and applyLayout() will re-render once runModeEval() completes, at which point this fires correctly.
    const dataIsReady = getSessionPhase() !== 'uninitialized'
    if (!_autopilotRunning && dataIsReady && getPickRow() < getNPicks() && getDrafterMethod(getPickDrafter()) !== 'Manual input') {
        fireAutopilotPicks(container).catch(err => console.error('Autopilot failed:', err))
    }
}

/** Builds the stable scaffold: a pick-control slot followed by the board table (header built once,
 *  empty body). Detaches stale header listeners; the body is filled by rebuildBoardBody. */
function buildScaffold(container: HTMLElement): void {
    container.innerHTML = ''

    pickControlSlot = document.createElement('div')
    pickControlSlot.className = 'draft-pick-control-slot'
    container.append(pickControlSlot)

    headerListenerController?.abort()
    headerListenerController = new AbortController()

    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'
    boardTable = buildBoardTable(container)
    scroll.append(boardTable)
    container.append(scroll)
}

// ─── Autopilot helpers ────────────────────────────────────────────────────────

/** Returns the undrafted player with the highest G-score total.
 *  Uses the pre-sorted G-score list (computed once at session creation) so this
 *  is O(drafted) rather than O(all players). */
function pickByGScore(draftedSet: Set<string>): string | null {
    for (const name of getPlayerNamesByGScore()) {
        if (!draftedSet.has(name)) return name
    }
    return null
}

/** Returns the top-ranked candidate from the latest evaluate response.
 *  Candidates are always undrafted (the backend only includes available players). */
function pickByHScore(): string | null {
    return getCandidatePlayerResults()?.[0]?.name ?? null
}

/**
 * Runs autopilot picks for consecutive autopilot drafters starting from the current position.
 * Stops when the next drafter is Manual input, the draft is complete, or a pick cannot be found.
 * Fires `runEvaluate` at the end to refresh candidates for the next manual drafter.
 */
async function fireAutopilotPicks(container: HTMLElement): Promise<void> {
    if (_autopilotRunning) return
    _autopilotRunning = true
    setAutopilotOn()
    const userSeat = getCurrentSeat()
    ;(document.getElementById('seat-selector-container') as HTMLElement).style.visibility = 'hidden'
    try {
        while (getPickRow() < getNPicks()) {
            const mode = getDrafterMethod(getPickDrafter())
            if (mode === 'Manual input') break

            const draftedSet = new Set(getDrafted().flat().filter(Boolean) as string[])

            let player: string | null
            if (mode === 'H-scoring') {
                clearFullTeamResult()
                setCurrentSeat(getTeamNames()[getPickDrafter()] ?? `Team ${getPickDrafter() + 1}`)
                await runEvaluate()
                player = pickByHScore()
            } else {
                player = pickByGScore(draftedSet)
            }

            if (!player) break
            recordDraftPick(getPickRow(), getPickDrafter(), player)
            advanceDraftPick()
            renderDraftBoard(container)
        }
    } finally {
        setCurrentSeat(userSeat)
        _autopilotRunning = false
        setAutopilotOff()
        ;(document.getElementById('seat-selector-container') as HTMLElement).style.visibility = ''
    }
    // Re-render now that _autopilotRunning is false so the pick control
    // switches back to the normal Manual input state.
    renderDraftBoard(container)
    _draftDebouncer.fire()
}

// ─── Pick control ─────────────────────────────────────────────────────────────

/** Builds the pick control row: player dropdown, lock-in / undo / clear buttons. */
function buildPickControl(container: HTMLElement): HTMLElement {
    const wrap = document.createElement('div')

    const pickRowVal     = getPickRow()
    const pickDrafterVal = getPickDrafter()
    const nPicks         = getNPicks()

    const isDone      = pickRowVal >= nPicks
    const currentMode = isDone ? ('Manual input' as const) : getDrafterMethod(pickDrafterVal)
    const isAutopilot = currentMode !== 'Manual input'

    // Inline row: label + (player select if manual) + action buttons on the same line
    const row = document.createElement('div')
    row.className = 'pick-control-row'

    const label = document.createElement('div')
    label.className = 'pick-control-label'
    label.textContent = isDone
        ? 'Draft complete'
        : isAutopilot
            ? `${getTeamLabel(pickDrafterVal)} (${currentMode})`
            : `Select Pick ${pickRowVal + 1} for ${getTeamLabel(pickDrafterVal)}`
    row.append(label)

    if (_autopilotRunning) {
        // Keep the label; replace the input area with a spinner
        const indicator = document.createElement('div')
        indicator.className = 'eval-indicator evaluating autopilot-running-indicator'
        indicator.textContent = 'Running autopilot'
        row.append(indicator)
        wrap.append(row)
        return wrap
    }

    const draftedSet = new Set(getDrafted().flat().filter(Boolean) as string[])
    const available  = getPlayerResults()?.map(p => p.name).filter(n => !draftedSet.has(n)) ?? []

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    if (!isDone && !isAutopilot) {
        const sel = makeCustomSelect(
            'draft-pick-select',
            available.map(n => ({ value: n, label: n })),
            undefined,
            undefined,
            pickListenerController?.signal,
        )
        sel.element.style.flex = '1'
        row.append(sel.element)

        const lockBtn = document.createElement('button')
        lockBtn.className = 'pick-btn'
        lockBtn.textContent = 'Lock in selection'
        lockBtn.disabled = available.length === 0
        lockBtn.addEventListener('click', async () => {
            const chosen = sel.getValue()
            if (!chosen || getPickRow() >= getNPicks()) return
            recordDraftPick(getPickRow(), getPickDrafter(), chosen)
            advanceDraftPick()
            renderDraftBoard(container)
            await fireAutopilotPicks(container)
        })
        btns.append(lockBtn)
    }

    const undoBtn = document.createElement('button')
    undoBtn.className = 'pick-btn'
    undoBtn.textContent = 'Undo previous selection'
    undoBtn.disabled = pickRowVal === 0 && pickDrafterVal === 0
    undoBtn.addEventListener('click', () => {
        if (getPickRow() === 0 && getPickDrafter() === 0) return
        // Rewind until we land on a Manual input drafter (or reach the very start)
        do {
            goBackDraftPick()
            clearDraftPick(getPickRow(), getPickDrafter())
        } while (
            getDrafterMethod(getPickDrafter()) !== 'Manual input'
            && !(getPickRow() === 0 && getPickDrafter() === 0)
        )
        renderDraftBoard(container)
        _draftDebouncer.fire()
    })

    const clearBtn = document.createElement('button')
    clearBtn.className = 'pick-btn'
    clearBtn.textContent = 'Clear draft board'
    clearBtn.addEventListener('click', () => {
        clearAllDraftPicks()
        renderDraftBoard(container)
        runEvaluate().catch(err => console.error('Evaluate after clear failed:', err))
    })

    btns.append(undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Draft board table ────────────────────────────────────────────────────────

/** Builds the draft board table with its header (Round | per-drafter [label input][method ▾])
 *  and an empty body. The header is built once per scaffold; rebuildBoardBody fills the rows. */
function buildBoardTable(container: HTMLElement): HTMLTableElement {
    const nDrafters = getNDrafters()

    const table = document.createElement('table')
    table.className      = 'entry-table'
    table.style.width    = '100%'
    table.style.minWidth = (ROUND_W + nDrafters * TEAM_W) + 'px'

    const thead = table.createTHead()
    const hrow  = thead.insertRow()
    const roundTh = document.createElement('th')
    roundTh.textContent = 'Round'
    roundTh.style.width = ROUND_W + 'px'
    hrow.append(roundTh)
    for (let d = 0; d < nDrafters; d++) {
        const th = document.createElement('th')
        th.className = 'team-header-cell'
        th.append(buildTeamHeader(d, container))
        hrow.append(th)
    }

    table.createTBody()   // filled by rebuildBoardBody
    return table
}

/** Builds one column header: an editable display-label input plus the compact method dropdown.
 *  Editing the label only changes presentation (identity stays "Team N") — it persists and
 *  relabels the seat selector via the team-labels-changed event, without rebuilding the header. */
function buildTeamHeader(drafterIndex: number, container: HTMLElement): HTMLElement {
    const cell = document.createElement('div')
    cell.className = 'team-header'

    cell.append(makeTeamLabelInput(drafterIndex, headerListenerController?.signal))

    cell.append(makeMethodDropdown(
        drafterIndex
      , () => renderDraftBoard(container)
      , headerListenerController?.signal
    ))
    return cell
}

/** Replaces the table body with the current picks: rounds × drafters, serpentine highlighting. */
function rebuildBoardBody(table: HTMLTableElement): void {
    const nPicks      = getNPicks()
    const nDrafters   = getNDrafters()
    const drafted     = getDrafted()
    const pickRow     = getPickRow()
    const pickDrafter = getPickDrafter()

    const tbody = document.createElement('tbody')
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

    const oldBody = table.tBodies[0]
    if (oldBody) table.replaceChild(tbody, oldBody)
    else         table.append(tbody)
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Reads current sidebar league settings and returns them with a composite key for change detection. */
function readDraftConfig(): DraftConfig {
    const nDrafters          = readRequiredIntInput('ls-n-drafters')
    const nPicks             = readRequiredIntInput('ls-n-picks')
    const dataSource         = (document.getElementById('ps-data-type') as HTMLInputElement).value
    const thirdRoundReversal = (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked
    const teamNames = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    // teamNames are excluded from the key: identities are constant "Team N" (so they never
    // trigger a reset), and editable display labels are presentation-only — a rename must not
    // reset the draft or rebuild the header. n_drafters covers any change in team count.
    return {
        nDrafters
        , nPicks
        , teamNames
        , thirdRoundReversal
        , key: `${nDrafters}:${nPicks}:${dataSource}:${thirdRoundReversal}`
    }
}
