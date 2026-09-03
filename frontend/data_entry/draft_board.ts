// data_entry/draft_board.ts
// Renders the draft pick control and draft board grid for Draft Mode + own data.
// Mirrors make_drafting_tab_own_data() in src/tabs/drafting.py.

import { makeCustomSelect } from '../custom_select.js'
import { isMobileViewport, readRequiredIntInput, makeBoardToggleHeaderCell, buildBoardTableShell } from '../helper_functions.js'
import { getPlayerResults, getSessionPhase, getCurrentSeat, setCurrentSeat } from '../app_state.js'
import { buildPlayerOption, makeMinimalPlayerDisplay } from '../player_display.js'
import { makeDebouncer } from '../api/session.js'
import { runEvaluate, clearFullTeamResult } from '../api/draft_and_auction_session.js'
import { setAutopilotOn, setAutopilotOff } from '../api/session.js'
import { getDrafterMethod } from './drafter_methods.js'
import { makeAutodraftToggle } from './autodraft_toggle.js'
import { getTeamLabel, defaultTeamLabel, makeTeamLabelInput } from './team_labels.js'
import { setSeatSelectorVisible } from '../seat_selector.js'
import { getTeamIdentitiesFromSidebar } from '../setting_collection/league_settings.js'
import {
    DraftConfig,
    getPickRow, getPickDrafter, getDrafted, getTeamIdentitiesFromBoard, getNDrafters, getNPicks, getConfigKey,
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
// Set only around autopilot's own end-of-run re-render, so that render doesn't re-trigger autopilot via
// renderDraftBoard's auto-start. Without it, an autopilot run that stops on an autopilot seat (e.g. an
// evaluate came back empty) would be restarted by its own final render — a tight "autopiloting…" loop.
let _suppressAutostart = false

// Mobile transposes the board (teams as rows, rounds as columns) with a frozen team-name column.
// `transposed` is the orientation the current scaffold was built for; the scaffold rebuilds when
// the viewport crosses the mobile breakpoint and this flips.
let transposed        = false
let lastScrolledRound = -1   // mobile auto-scroll: only nudge when the current round changes

const ROUND_W    = 46   // px — Round label column incl. the collapse arrow (desktop)
const TEAM_W     = 50   // px — per-drafter column, desktop (min-width floor; table fills width)
const TEAMCOL_W  = 90   // px — frozen team-name column (transposed mobile; fits name + method dropdown on one row)
const ROUNDCOL_W = 64   // px — per-round column (transposed mobile)

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Clears the draft board state. Call when the player pool changes (e.g. data source switch)
 * so drafted player ids from the old dataset are not sent to the backend.
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
    // render, on a config change, if the table was detached (container cleared elsewhere), or on
    // an orientation flip (desktop ⇄ mobile transpose).
    if (configChanged || !boardTable || !container.contains(boardTable) || isMobileViewport() !== transposed) {
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
    if (!_autopilotRunning && !_suppressAutostart && dataIsReady && getPickRow() < getNPicks() && getDrafterMethod(getPickDrafter()) !== 'Manual input') {
        fireAutopilotPicks(container).catch(err => console.error('Autopilot failed:', err))
    }
}

/** Builds the stable scaffold: a pick-control slot followed by the board table (header built once,
 *  empty body). Detaches stale header listeners; the body is filled by rebuildBoardBody. */
function buildScaffold(container: HTMLElement): void {
    container.innerHTML = ''
    transposed = isMobileViewport()
    lastScrolledRound = -1

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

/** Re-render the board after a pick-state change, then refresh whatever the *new* current drafter
 *  needs. renderDraftBoard already fires autopilot for an autopilot drafter, so here we only trigger
 *  the available-players evaluate for a manual drafter. Deliberately scoped to real pick mutations
 *  (clear / undo / lock-in / autopilot-end) rather than baked into renderDraftBoard — the latter also
 *  runs on cosmetic and config re-renders, which runModeEval already evaluates, so evaluating there
 *  would double-fire and reset the table needlessly. */
function reflectDraftChange(container: HTMLElement): void {
    renderDraftBoard(container)
    // Fire the evaluate whenever renderDraftBoard did NOT hand off to autopilot — i.e. a manual drafter
    // is on the clock, or the draft is complete. The done case matters: it is the evaluate that computes
    // the full-team result, so the team-statistics tab shows the finished team's H-score. (A guard that
    // required picks remaining skipped that final evaluate, leaving the H-score blank once a team filled.)
    const willAutostart = getPickRow() < getNPicks() && getDrafterMethod(getPickDrafter()) !== 'Manual input'
    if (!_autopilotRunning && getSessionPhase() !== 'uninitialized' && !willAutostart) {
        _draftDebouncer.fire()
    }
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
    // Cancel any pending debounced evaluate: if it fired mid-run it would start a competing evaluate that
    // aborts autopilot's own evaluate, which now returns null and stops the run short (then the end render
    // could restart it — the "autopiloting…" loop). Autopilot refreshes the board itself when it finishes.
    _draftDebouncer.cancel()
    const userSeat = getCurrentSeat()
    setSeatSelectorVisible(false)
    try {
        while (getPickRow() < getNPicks()) {
            if (getDrafterMethod(getPickDrafter()) === 'Manual input') break

            clearFullTeamResult()
            setCurrentSeat(getTeamIdentitiesFromBoard()[getPickDrafter()] ?? defaultTeamLabel(getPickDrafter()))
            // Autopilot only needs the top pick: score just the first batch and render nothing. Use the
            // value this evaluate returns — not a shared global — so a competing evaluate that aborts
            // this one yields null (stop cleanly) instead of a stale pick from a previous drafter.
            const topPlayerId = await runEvaluate({ forAutopilot: true })

            if (topPlayerId === null) break
            recordDraftPick(getPickRow(), getPickDrafter(), topPlayerId)
            advanceDraftPick(readDraftConfig().thirdRoundReversal)
            renderDraftBoard(container)
        }
    } finally {
        setCurrentSeat(userSeat)
        _autopilotRunning = false
        setAutopilotOff()
        setSeatSelectorVisible(true)
    }
    // Re-render now that _autopilotRunning is false so the pick control switches back to the normal
    // Manual input state, and refresh the available players / full-team result. Suppress auto-start for
    // this one render: autopilot has already run to its stopping point (a manual seat, the draft's end,
    // or a seat it couldn't pick for), so re-triggering it here would only spin.
    _suppressAutostart = true
    reflectDraftChange(container)
    _suppressAutostart = false
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

    const draftedSet = new Set(getDrafted().flat().filter(playerId => playerId !== null) as number[])
    const available  = getPlayerResults()?.map(p => p.player_id).filter(playerId => !draftedSet.has(playerId)) ?? []

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    if (!isDone && !isAutopilot) {
        const sel = makeCustomSelect(
            'draft-pick-select',
            available.map(buildPlayerOption),
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
        lockBtn.addEventListener('click', () => {
            const chosen = sel.getValue()
            if (!chosen || getPickRow() >= getNPicks()) return
            const chosenPlayerId = Number(chosen)
            if (Number.isNaN(chosenPlayerId)) throw new Error(`Draft pick select carried a non-numeric value: "${chosen}"`)
            recordDraftPick(getPickRow(), getPickDrafter(), chosenPlayerId)
            // Read the reversal setting fresh from the sidebar: the stepping consumes it at the
            // round-2/3 boundary, so a toggle made any time before then must take effect.
            advanceDraftPick(readDraftConfig().thirdRoundReversal)
            // renderDraftBoard (inside reflectDraftChange) fires autopilot for the next drafter if it is
            // an autodrafter, so no explicit fireAutopilotPicks call is needed here.
            reflectDraftChange(container)
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
        reflectDraftChange(container)
    })

    const clearBtn = document.createElement('button')
    clearBtn.className = 'pick-btn'
    clearBtn.textContent = 'Clear draft board'
    clearBtn.addEventListener('click', () => {
        clearAllDraftPicks()
        // reflectDraftChange re-renders and, if drafter 0 is an autodrafter, lets renderDraftBoard own
        // the autopilot run — so we no longer fire a competing runEvaluate that would abort it and leave
        // the first autopilot pick reading a stale candidate list.
        reflectDraftChange(container)
    })

    btns.append(undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Draft board table ────────────────────────────────────────────────────────

/** Builds the board table for the current orientation. Header widgets (team label + method
 *  dropdown) are built once per scaffold; rebuildBoardBody fills the data cells. */
function buildBoardTable(container: HTMLElement): HTMLTableElement {
    return transposed ? buildTransposedTable(container) : buildStandardTable(container)
}

/** Desktop: rounds as rows, teams as columns. Header row = Round | per-team [label input][method ▾];
 *  empty body filled by rebuildStandardBody. */
function buildStandardTable(container: HTMLElement): HTMLTableElement {
    const nDrafters = getNDrafters()

    const table = buildBoardTableShell(nDrafters, TEAM_W, ROUND_W, 'draft_board_open',
        d => buildTeamHeader(d, container))
    table.createTBody()   // filled by rebuildStandardBody
    return table
}

/** Mobile: teams as rows, rounds as columns, with a frozen team-name first column. The header row is
 *  round numbers; each team row's first cell is the persistent team header (sticky). The trailing
 *  pick cells are filled/rebuilt by rebuildTransposedBody. */
function buildTransposedTable(container: HTMLElement): HTMLTableElement {
    const nDrafters = getNDrafters()
    const nPicks    = getNPicks()

    const table = document.createElement('table')
    table.className      = 'entry-table transposed'
    table.style.width    = '100%'
    table.style.minWidth = (TEAMCOL_W + nPicks * ROUNDCOL_W) + 'px'

    // Header: [corner] | 1 | 2 | … | nPicks
    const thead = table.createTHead()
    const hrow  = thead.insertRow()
    const corner = makeBoardToggleHeaderCell(table, 'draft_board_open', '')
    corner.style.width = TEAMCOL_W + 'px'
    hrow.append(corner)
    for (let r = 0; r < nPicks; r++) {
        const th = document.createElement('th')
        th.textContent = String(r + 1)
        th.style.width = ROUNDCOL_W + 'px'
        hrow.append(th)
    }

    // One body row per team; first cell = persistent team header (kept across picks by
    // rebuildTransposedBody, which only rebuilds the trailing pick cells).
    const tbody = table.createTBody()
    for (let d = 0; d < nDrafters; d++) {
        const row = tbody.insertRow()
        const teamCell = document.createElement('th')
        teamCell.className = 'team-header-cell'
        teamCell.style.width = TEAMCOL_W + 'px'
        teamCell.append(buildTeamHeader(d, container))
        row.append(teamCell)
    }
    return table
}

/** Builds one column header: an editable display-label input plus the compact autodraft toggle.
 *  Editing the label only changes presentation (identity stays "Team N") — it persists and
 *  relabels the seat selector via the team-labels-changed event, without rebuilding the header. */
function buildTeamHeader(drafterIndex: number, container: HTMLElement): HTMLElement {
    const cell = document.createElement('div')
    cell.className = 'team-header'

    cell.append(makeTeamLabelInput(drafterIndex, headerListenerController?.signal))

    cell.append(makeAutodraftToggle(
        drafterIndex
      , () => renderDraftBoard(container)
      , headerListenerController?.signal
    ))
    return cell
}

/** Repaints the picks for the current orientation. */
function rebuildBoardBody(table: HTMLTableElement): void {
    if (transposed) rebuildTransposedBody(table)
    else            rebuildStandardBody(table)
}

/** Desktop: replace the whole body with rounds × drafters (team headers live in <thead>). */
function rebuildStandardBody(table: HTMLTableElement): void {
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
            const cell     = row.insertCell()
            const playerId = drafted[r][d]
            if (playerId !== null) {
                cell.append(makeMinimalPlayerDisplay(playerId))
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

/** Mobile: per team row, keep the sticky team-header cell (cell 0) and rebuild only the trailing
 *  pick cells (one per round). Auto-scrolls the current round into view when the round changes. */
function rebuildTransposedBody(table: HTMLTableElement): void {
    const nPicks      = getNPicks()
    const nDrafters   = getNDrafters()
    const drafted     = getDrafted()
    const pickRow     = getPickRow()
    const pickDrafter = getPickDrafter()
    const tbody       = table.tBodies[0]

    let currentCell: HTMLTableCellElement | null = null
    for (let d = 0; d < nDrafters; d++) {
        const row = tbody.rows[d]
        while (row.cells.length > 1) row.deleteCell(-1)   // keep the sticky team header (cell 0)
        for (let r = 0; r < nPicks; r++) {
            const cell     = row.insertCell()
            const playerId = drafted[r][d]
            if (playerId !== null) {
                cell.append(makeMinimalPlayerDisplay(playerId))
                cell.classList.add('drafted')
            } else if (r === pickRow && d === pickDrafter) {
                cell.classList.add('current-pick')
                currentCell = cell
            }
        }
    }

    // Follow the draft: bring the current round column into view, but only when the round actually
    // changes, so it doesn't fight the user's manual horizontal scrolling.
    if (currentCell && pickRow !== lastScrolledRound) {
        currentCell.scrollIntoView({ inline: 'nearest', block: 'nearest' })
        lastScrolledRound = pickRow
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Reads current sidebar league settings and returns them with a composite key for change detection. */
function readDraftConfig(): DraftConfig {
    const nDrafters          = readRequiredIntInput('ls-n-drafters')
    const nPicks             = readRequiredIntInput('ls-n-picks')
    const dataSource         = (document.getElementById('ps-data-type') as HTMLInputElement).value
    const thirdRoundReversal = (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked
    const teamNames = getTeamIdentitiesFromSidebar()
    // teamNames are excluded from the key: identities are constant "Team N" (so they never
    // trigger a reset), and editable display labels are presentation-only — a rename must not
    // reset the draft or rebuild the header. n_drafters covers any change in team count.
    // thirdRoundReversal is also excluded: toggling it must not reset the board — the setting
    // is passed into advanceDraftPick, which consumes it at the round-2/3 boundary.
    return {
        nDrafters
        , nPicks
        , teamNames
        , thirdRoundReversal
        , key: `${nDrafters}:${nPicks}:${dataSource}`
    }
}
