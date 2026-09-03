// data_entry/auction_entry.ts
// Renders the auction pick control and board grid for Auction Mode + own data.
// Mirrors the draft board structure: pick control on top, grid below.

import { makeCustomSelect } from '../custom_select.js'
import { readRequiredIntInput, buildBoardTableShell } from '../helper_functions.js'
import { getPlayerResults } from '../app_state.js'
import { buildPlayerOption, makeMinimalPlayerDisplay } from '../player_display.js'
import { getTeamIdentitiesFromSidebar } from '../parameter_collection/league_settings.js'
import { makeDebouncer } from '../api/session.js'
import { runEvaluate } from '../api/draft_and_auction_session.js'
import { getTeamLabel, makeTeamLabelInput } from './team_labels.js'
import {
    AuctionConfig,
    getPicks, getTeamIdentitiesFromBoard, getNDrafters, getNPicks, getCashPerTeam, getConfigKey, getHistory,
    resetAuctionState, applyAuctionConfig,
    recordAuctionPick, undoLastAuctionPick, clearAllAuctionPicks,
} from './auction_state.js'

// ─── Module state ─────────────────────────────────────────────────────────────

const _auctionDebouncer = makeDebouncer(() => { runEvaluate().catch(err => console.error('Auction evaluate failed:', err)) })

// Tracks the listeners attached by the most recent renderAuctionEntry call so
// they can be detached before the next one. renderAuctionEntry is called on
// every pick — without this, each rebuild would leave the previous pick-control
// custom selects' internal listeners (~9 each × 2 selects) bound to detached
// nodes. The closures keep the old wrapper DOM alive until the cycle is broken.
let auctionListenerController: AbortController | null = null

/** Total auction dollars a drafter has spent across the board's picks. The budget cap on
 *  the cost input and the lock-in enforcement both read this, so they cannot disagree. */
function sumSpentByDrafter(picks: ReturnType<typeof getPicks>, drafterIndex: number): number {
    return picks.reduce((sum, pickRow) => sum + (pickRow[drafterIndex]?.cost ?? 0), 0)
}

const ROUND_W = 46   // fits the collapse arrow beside 'Round'
const TEAM_W  = 60

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Clears the auction board state. Call when the player pool changes (e.g. data source switch)
 * so picks referencing old player ids are not sent to the backend.
 * The next renderAuctionEntry call will reinitialise from current sidebar values.
 */
export function resetAuctionEntry(): void {
    resetAuctionState()
    _auctionDebouncer?.cancel()
}

/** Renders the auction entry UI into the container. Resets state if sidebar config changed. */
export function renderAuctionEntry(container: HTMLElement): void {
    const cfg = readAuctionConfig()

    if (cfg.key !== getConfigKey()) {
        applyAuctionConfig(cfg)
    }

    // Detach listeners from the previous render's custom selects so their
    // closures can be garbage-collected. See comment on auctionListenerController.
    auctionListenerController?.abort()
    auctionListenerController = new AbortController()

    container.innerHTML = ''
    container.append(buildPickControl(container))
    container.append(buildAuctionBoard())

    // Notify layout that the board changed so the G-score tab can refresh
    container.dispatchEvent(new CustomEvent('auction-board-change', { bubbles: true }))
}

// ─── Pick control ─────────────────────────────────────────────────────────────

/** Builds the auction pick control row: player + drafter + cost inputs, lock-in / undo / clear buttons. */
function buildPickControl(container: HTMLElement): HTMLElement {
    const wrap = document.createElement('div')

    const row = document.createElement('div')
    row.className = 'pick-control-row'

    // Player dropdown — grows to fill available space. Filter the FULL player pool by the
    // board's own picks (like the draft board), not the last evaluate's candidate list: that
    // list already excludes picked players, so after an undo it is stale and would leave the
    // undone player missing from the dropdown until some later re-render.
    const currentPicks = getPicks()
    const pickedSet = new Set(currentPicks.flat().filter(Boolean).map(p => p!.playerId))
    const available = getPlayerResults()?.map(p => p.player_id).filter(playerId => !pickedSet.has(playerId)) ?? []
    const playerSel = makeCustomSelect(
        'auction-pick-player',
        [
            { value: '', label: '' },
            ...available.map(buildPlayerOption),
        ],
        undefined,
        undefined,
        auctionListenerController?.signal,
    )
    playerSel.element.style.width = '100%'
    const playerCol = makePickCol('Player', playerSel.element)
    playerCol.style.flex = '1'
    row.append(playerCol)

    // Drafter dropdown — before cost so the cap makes sense visually; full teams excluded.
    // Option value is the team identity ("Team N", mapped back via indexOf on lock-in); the
    // shown label is the editable display label.
    const availableTeamOptions = getTeamIdentitiesFromBoard()
        .map((name, index) => ({ value: name, label: getTeamLabel(index), index }))
        .filter(({ index }) => currentPicks.some(pickRow => pickRow[index] === null))
        .map(({ value, label }) => ({ value, label }))
    const teamSel = makeCustomSelect(
        'auction-pick-team',
        [{ value: '', label: '' }, ...availableTeamOptions],
        undefined,
        undefined,
        auctionListenerController?.signal,
    )
    teamSel.element.style.width = '100%'
    const teamCol = makePickCol('Drafter', teamSel.element)
    teamCol.style.flex = '1'
    row.append(teamCol)

    // Cost input — capped to selected drafter's remaining cash
    const costInput = document.createElement('input')
    costInput.type        = 'number'
    costInput.min         = '1'
    costInput.placeholder = '$'
    costInput.className   = 'auction-cost-input'

    function updateCostOverBudget(): void {
        const max = parseFloat(costInput.max)
        const cost = parseFloat(costInput.value)
        costInput.classList.toggle('over-budget', !isNaN(max) && !isNaN(cost) && cost > max)
    }

    function updateCostMax(): void {
        const team = teamSel.getValue()
        if (team) {
            const drafterIndex = getTeamIdentitiesFromBoard().indexOf(team)
            const spent = sumSpentByDrafter(getPicks(), drafterIndex)
            costInput.max = String(getCashPerTeam() - spent)
        } else {
            costInput.removeAttribute('max')
        }
        updateCostOverBudget()
    }

    costInput.addEventListener('input', updateCostOverBudget)
    teamSel.element.addEventListener('change', updateCostMax)
    row.append(makePickCol('Cost', costInput))

    const btns = document.createElement('div')
    btns.className = 'pick-control-buttons'

    const lockBtn = document.createElement('button')
    lockBtn.className   = 'pick-btn'
    lockBtn.textContent = 'Lock in selection'
    lockBtn.addEventListener('click', () => {
        const chosen = playerSel.getValue()
        if (!chosen) return
        const chosenPlayerId = Number(chosen)
        if (Number.isNaN(chosenPlayerId)) throw new Error(`Auction pick select carried a non-numeric value: "${chosen}"`)
        const team = teamSel.getValue()
        if (!team) return
        const drafterIndex = getTeamIdentitiesFromBoard().indexOf(team)
        const cost = parseFloat(costInput.value)
        if (isNaN(cost) || cost <= 0) return
        const spent = sumSpentByDrafter(getPicks(), drafterIndex)
        if (cost > getCashPerTeam() - spent) return
        const succeeded    = recordAuctionPick(chosenPlayerId, cost, drafterIndex)
        if (succeeded) {
            renderAuctionEntry(container)
            _auctionDebouncer?.fire()
        }
    })

    const undoBtn = document.createElement('button')
    undoBtn.className   = 'pick-btn'
    undoBtn.textContent = 'Undo previous selection'
    undoBtn.disabled    = getHistory().length === 0
    undoBtn.addEventListener('click', () => {
        const undone = undoLastAuctionPick()
        if (undone) {
            renderAuctionEntry(container)
            _auctionDebouncer?.fire()
        }
    })

    const clearBtn = document.createElement('button')
    clearBtn.className   = 'pick-btn'
    clearBtn.textContent = 'Clear auction board'
    clearBtn.addEventListener('click', () => {
        const cleared = clearAllAuctionPicks()
        if (cleared) {
            renderAuctionEntry(container)
            runEvaluate().catch(err => console.error('Evaluate after clear failed:', err))
        }
    })

    btns.append(lockBtn, undoBtn, clearBtn)
    row.append(btns)
    wrap.append(row)
    return wrap
}

// ─── Auction board table ──────────────────────────────────────────────────────

/** Builds the auction board grid: rounds × drafters with player names, costs, and remaining budget footer. */
function buildAuctionBoard(): HTMLElement {
    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'

    const nPicks      = getNPicks()
    const nDrafters   = getNDrafters()
    const picks       = getPicks()
    const cashPerTeam = getCashPerTeam()

    // Header columns follow nDrafters rather than the name list: board identities are
    // positional, so a half-edited textarea must not change the column count.
    const table = buildBoardTableShell(nDrafters, TEAM_W, ROUND_W, 'auction_board_open', d => {
        const headerWrap = document.createElement('div')
        headerWrap.className = 'team-header'
        headerWrap.append(makeTeamLabelInput(d, auctionListenerController?.signal))
        return headerWrap
    })

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
                const costEl = document.createElement('div')
                costEl.className   = 'auction-cell-cost'
                costEl.textContent = `$${pick.cost}`

                cell.append(makeMinimalPlayerDisplay(pick.playerId), costEl)
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
        const spent = sumSpentByDrafter(picks, d)
        const td = document.createElement('td')
        td.className   = 'auction-budget-cell'
        td.textContent = `$${cashPerTeam - spent}`
        frow.append(td)
    }

    scroll.append(table)
    return scroll
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Creates a labelled column wrapper for an input element in the pick control row. */
function makePickCol(labelText: string, input: HTMLElement): HTMLElement {
    const col = document.createElement('div')
    col.className = 'pick-col'
    const lbl = document.createElement('div')
    lbl.className   = 'pick-col-label'
    lbl.textContent = labelText
    col.append(lbl, input)
    return col
}


/** Reads current sidebar league settings and returns them with a composite key for change detection. */
function readAuctionConfig(): AuctionConfig {
    const nDrafters   = readRequiredIntInput('ls-n-drafters')
    const nPicks      = readRequiredIntInput('ls-n-picks')
    const cashPerTeam = readRequiredIntInput('ls-cash-per-team')
    const dataSource  = (document.getElementById('ps-data-type') as HTMLInputElement).value
    const teamNames   = getTeamIdentitiesFromSidebar()
    return {
        nDrafters
        , nPicks
        , cashPerTeam
        , teamNames
        , key: `${nDrafters}:${nPicks}:${cashPerTeam}:${dataSource}:${teamNames.join(',')}`
    }
}
