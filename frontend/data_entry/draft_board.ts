// data_entry/draft_board.ts
// Renders the draft pick control and draft board grid for Draft Mode + own data.
// Mirrors make_drafting_tab_own_data() in src/tabs/drafting.py.

import { makeCustomSelect } from '../custom_select.js'
import { getPlayerResults, getCandidatePlayerResults, getPlayerNamesByGScore, getSessionPhase, getCurrentSeat, setCurrentSeat } from '../app_state.js'
import { makeDebouncer } from '../api/session.js'
import { runEvaluate, clearFullTeamResult } from '../api/draft_and_auction_session.js'
import { setAutopilotOn, setAutopilotOff } from '../api/session.js'
import { getDrafterMethodByIndex } from '../parameter_collection/league_settings.js'
import {
    DraftConfig,
    getPickRow, getPickDrafter, getDrafted, getTeamNames, getNDrafters, getNPicks, getConfigKey,
    resetDraftState, applyDraftConfig,
    recordDraftPick, clearDraftPick, clearAllDraftPicks,
    advanceDraftPick, goBackDraftPick,
} from './draft_state.js'

// ─── Module state ─────────────────────────────────────────────────────────────

const _draftDebouncer = makeDebouncer(() => { runEvaluate().catch(err => console.error('Draft evaluate failed:', err)) })

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

    // Reset state if league settings changed (different drafter/pick counts or teams)
    if (cfg.key !== getConfigKey()) {
        applyDraftConfig(cfg)
    }

    container.innerHTML = ''
    container.append(buildPickControl(container))
    container.append(buildDraftBoard())

    // Auto-fire autopilot when the current drafter is an autopilot drafter and the loop is not already running.
    // Guard on G-scores being loaded — on initial page render the session hasn't been created yet,
    // and applyLayout() will re-render once runModeEval() completes, at which point this fires correctly.
    const dataIsReady = getSessionPhase() !== 'uninitialized'
    if (!_autopilotRunning && dataIsReady && getPickRow() < getNPicks() && getDrafterMethodByIndex(getPickDrafter()) !== 'Manual input') {
        fireAutopilotPicks(container).catch(err => console.error('Autopilot failed:', err))
    }
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
            const mode = getDrafterMethodByIndex(getPickDrafter())
            if (mode === 'Manual input') break

            const draftedSet = new Set(getDrafted().flat().filter(Boolean) as string[])

            let player: string | null
            if (mode === 'H-scoring') {
                clearFullTeamResult()
                setCurrentSeat(getTeamNames()[getPickDrafter()] ?? `Drafter ${getPickDrafter() + 1}`)
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
    const teamNames      = getTeamNames()

    const isDone      = pickRowVal >= nPicks
    const currentMode = isDone ? ('Manual input' as const) : getDrafterMethodByIndex(pickDrafterVal)
    const isAutopilot = currentMode !== 'Manual input'

    // Inline row: label + (player select if manual) + action buttons on the same line
    const row = document.createElement('div')
    row.className = 'pick-control-row'

    const label = document.createElement('div')
    label.className = 'pick-control-label'
    label.textContent = isDone
        ? 'Draft complete'
        : isAutopilot
            ? `${teamNames[pickDrafterVal] ?? `Drafter ${pickDrafterVal + 1}`} (${currentMode})`
            : `Select Pick ${pickRowVal + 1} for ${teamNames[pickDrafterVal] ?? `Drafter ${pickDrafterVal + 1}`}`
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
        const sel = makeCustomSelect('draft-pick-select', available.map(n => ({ value: n, label: n })))
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
            getDrafterMethodByIndex(getPickDrafter()) !== 'Manual input'
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
    const trr = (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked
    const names = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    return { nDrafters: nD
        , nPicks: nP
        , teamNames: names
        , thirdRoundReversal: trr
        , key: `${nD}:${nP}:${src}:${trr}:${names.join(',')}` }
}
