// data_entry/auction_entry.ts
// Renders the auction pick control and board grid for Auction Mode + own data.
// Mirrors the draft board structure: pick control on top, grid below.

import { makeCustomSelect } from '../custom_select.js'
import { getCandidatePlayerResults } from '../app_state.js'
import { makeDebouncer } from '../helper_functions.js'
import { runEvaluate } from '../api/session.js'
import {
    AuctionConfig,
    getPicks, getTeamNames, getNDrafters, getNPicks, getCashPerTeam, getConfigKey, getHistory,
    resetAuctionState, applyAuctionConfig,
    recordAuctionPick, undoLastAuctionPick, clearAllAuctionPicks,
} from './auction_state.js'

// ─── Module state ─────────────────────────────────────────────────────────────

const _auctionDebouncer = makeDebouncer(() => { runEvaluate().catch(err => console.error('Auction evaluate failed:', err)) })

const ROUND_W = 46
const TEAM_W  = 85

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Clears the auction board state. Call when the player pool changes (e.g. data source switch)
 * so picks referencing old player names are not sent to the backend.
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

    // Player dropdown — grows to fill available space
    const currentPicks = getPicks()
    const pickedSet = new Set(currentPicks.flat().filter(Boolean).map(p => p!.player))
    const available = getCandidatePlayerResults()?.map(p => p.name).filter(n => !pickedSet.has(n)) ?? []
    const playerSel = makeCustomSelect(
        'auction-pick-player',
        [{ value: '', label: '' }, ...available.map(n => ({ value: n, label: n }))],
    )
    playerSel.element.style.width = '100%'
    const playerCol = makePickCol('Player', playerSel.element)
    playerCol.style.flex = '1'
    row.append(playerCol)

    // Drafter dropdown — before cost so the cap makes sense visually; full teams excluded
    const availableTeams = getTeamNames().filter((_, index) =>
        currentPicks.some(pickRow => pickRow[index] === null)
    )
    const teamSel = makeCustomSelect(
        'auction-pick-team',
        [{ value: '', label: '' }, ...availableTeams.map(n => ({ value: n, label: n }))],
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
            const drafterIndex = getTeamNames().indexOf(team)
            const spent = getPicks().reduce((sum, pickRow) => sum + (pickRow[drafterIndex]?.cost ?? 0), 0)
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
        const player = playerSel.getValue()
        if (!player) return
        const team = teamSel.getValue()
        if (!team) return
        const drafterIndex = getTeamNames().indexOf(team)
        const cost = parseFloat(costInput.value)
        if (isNaN(cost) || cost <= 0) return
        const spent = getPicks().reduce((sum, pickRow) => sum + (pickRow[drafterIndex]?.cost ?? 0), 0)
        if (cost > getCashPerTeam() - spent) return
        const succeeded    = recordAuctionPick(player, cost, drafterIndex)
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
    const teamNames   = getTeamNames()
    const picks       = getPicks()
    const cashPerTeam = getCashPerTeam()

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
    const nD   = parseInt((document.getElementById('ls-n-drafters')   as HTMLInputElement).value) || 12
    const nP   = parseInt((document.getElementById('ls-n-picks')       as HTMLInputElement).value) || 13
    const cash = parseInt((document.getElementById('ls-cash-per-team') as HTMLInputElement).value) || 200
    const src  = (document.getElementById('ps-data-type') as HTMLInputElement).value
    const names = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    return {
        nDrafters: nD, nPicks: nP, cashPerTeam: cash, teamNames: names,
        key: `${nD}:${nP}:${cash}:${src}:${names.join(',')}`,
    }
}
