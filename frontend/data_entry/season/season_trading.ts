// data_entry/season/season_trading.ts
// Renders the Season → Trading tab.
//   1. Your team
//   2. Trade inspection: the team traded with, player multi-selects (send / receive), result sub-tabs (H-score / G-score)
//   3. Suggested trades, searched across every other team in value-gap bands (one request per band, merged in by
//      your score)

import { makeCustomSelect } from '../../custom_select.js'
import {
    makeMultiSelectWidget, MultiSelectWidget, makeNumberInput, makeSidebarToggle, readRequiredIntInput, normalizeSearchText,
} from '../../helper_functions.js'
import { readRosterAssignments, readEmptyRostersMessage, listMovablePlayers } from './season_helpers.js'
import { getTeamIdentitiesFromSidebar } from '../../setting_collection/league_settings.js'
import { getGScoreById } from '../../app_state.js'
import { getRegistryEntry } from '../../player_registry.js'
import { buildPlayerOption, buildMinimalPlayerDisplayHtml } from '../../player_display.js'
import { getSelectedCategories } from '../../setting_collection/format_and_categories.js'
import { stat_styler_primary, G_SCORE_MULTIPLIER, G_SCORE_AGGREGATE_MULTIPLIER } from '../../styles/styler_functions.js'
import {
    DEFAULT_COMBOS, TRADE_SEARCH_ROUNDS, TRADE_THRESHOLD_FLOOR_PERCENT, SUGGESTION_ROWS_PER_PAGE,
} from '../../setting_collection/trade_parameters.js'
import { pref, savePref } from '../../preferences.js'
import { runTradeAnalyze, runTradeSuggest } from '../../api/season_session.js'
import type { TradeAnalyzeResponse, TradeSuggestion } from '../../api/client.js'

// ─── Player id <-> multiselect value helpers ─────────────────────────────────

// Styler multipliers for this pane's fraction-scaled scores (the app-wide G/H multipliers
// live in styler_functions). Matchup H-scores and win rates arrive as 0-1 fractions centred
// at even (0.5); suggestion score deltas are far smaller fractions, hence the larger gain.
const MATCHUP_H_MULTIPLIER        = 200
const SUGGESTION_SCORE_MULTIPLIER = 15000

/** Multiselect options for a roster: player ids as values, registry names as labels. */
function buildPlayerOptions(playerIds: number[]): { value: string; label: string; html: string }[] {
    return playerIds.map(buildPlayerOption)
}

/** Parses multiselect values back to player ids. Throws on a non-numeric value — the
 *  send/receive selectors only ever carry stringified ids. */
function parseSelectedPlayerIds(values: string[]): number[] {
    return values.map(value => {
        const playerId = Number(value)
        if (Number.isNaN(playerId)) throw new Error(`Trade multiselect carried a non-numeric value: "${value}"`)
        return playerId
    })
}

// ─── G-score comparison table ────────────────────────────────────────────────

/** Builds a G-score comparison table for the selected send/receive players. */
function buildGScoreTable(sent: number[], received: number[]): HTMLElement {
    const container = document.createElement('div')
    container.className = 'trade-gscore-section'

    const gScoreMap = getGScoreById()
    const categories = getSelectedCategories()

    type Row = { label: string; total: number; values: number[] }

    function getPlayerGRow(playerId: number): Row | null {
        const gs = gScoreMap.get(playerId)
        if (!gs) return null
        return { label: getRegistryEntry(playerId).name, total: gs.total, values: gs.values }
    }

    const sentRows:     Row[] = []
    const receivedRows: Row[] = []
    for (const playerId of sent) {
        const r = getPlayerGRow(playerId)
        if (r) sentRows.push(r)
    }
    for (const playerId of received) {
        const r = getPlayerGRow(playerId)
        if (r) receivedRows.push(r)
    }

    function sumRows(rows: Row[]): Row {
        const values = categories.map((_, i) => rows.reduce((s, r) => s + r.values[i], 0))
        return { label: '', total: values.reduce((a, b) => a + b, 0), values }
    }

    const sentTotal = sumRows(sentRows)
    const recvTotal = sumRows(receivedRows)
    const diff: Row = {
        label: 'Total Difference',
        total: recvTotal.total - sentTotal.total,
        values: categories.map((_, i) => recvTotal.values[i] - sentTotal.values[i]),
    }

    // Render table
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    hrow.insertCell().textContent = ''
    const totalTh = document.createElement('th')
    totalTh.textContent = 'Total'
    hrow.append(totalTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.textContent = cat
        hrow.append(th)
    }

    const tbody = table.createTBody()

    /** gradientMultiplier colours the cells (a player's row, or the team-level difference); null leaves them plain. */
    function addRow(label: string, row: Row, cssClass: string, gradientMultiplier: number | null): void {
        const tr = tbody.insertRow()
        tr.className = cssClass
        const labelCell = tr.insertCell()
        labelCell.textContent = label
        labelCell.className = 'trade-row-label'
        const totalCell = tr.insertCell()
        totalCell.textContent = row.total.toFixed(2)
        if (gradientMultiplier !== null) {
            totalCell.style.cssText = stat_styler_primary(row.total, gradientMultiplier, 0)
        } else {
            totalCell.className = 'celltypeb'
        }
        for (const v of row.values) {
            const td = tr.insertCell()
            td.textContent = v.toFixed(2)
            if (gradientMultiplier !== null) {
                td.style.cssText = stat_styler_primary(v, gradientMultiplier, 0)
            } else {
                td.className = 'celltypeb'
            }
        }
    }

    for (const row of sentRows) {
        addRow(row.label, row, 'trade-row-sent', G_SCORE_MULTIPLIER)
    }
    addRow('Total Sent', sentTotal, 'trade-row-subtotal', null)

    for (const row of receivedRows) {
        addRow(row.label, row, 'trade-row-received', G_SCORE_MULTIPLIER)
    }
    addRow('Total Received', recvTotal, 'trade-row-subtotal', null)

    addRow('Total Difference', diff, 'trade-row-diff', G_SCORE_AGGREGATE_MULTIPLIER)

    container.append(table)
    return container
}

// ─── H-score trade analysis ─────────────────────────────────────────────────

/** Builds the H-score trade result display with pre/post comparison for both teams. */
function buildHScoreResult(
    pane: HTMLElement
    , assignments: Record<string, number[]>
    , yourTeam: string
    , theirTeam: string
    , sent: number[]
    , received: number[]
): void {
    pane.innerHTML = ''

    const loading = document.createElement('div')
    loading.className = 'trade-hscore-stub'
    loading.textContent = 'Analyzing trade...'
    pane.append(loading)

    const positionCheck = (document.getElementById('ts-check-positions') as HTMLInputElement).checked
    runTradeAnalyze(assignments, yourTeam, theirTeam, sent, received, positionCheck)
        .then(resp => renderHScoreResult(pane, resp))
        .catch(err => {
            pane.innerHTML = ''
            const msg = document.createElement('div')
            msg.className = 'trade-hscore-stub'
            msg.textContent = `Error: ${err.message}`
            pane.append(msg)
        })
}

function renderHScoreResult(pane: HTMLElement, resp: TradeAnalyzeResponse): void {
    pane.innerHTML = ''

    if (resp.error) {
        const msg = document.createElement('div')
        msg.className = 'trade-hscore-stub'
        msg.textContent = resp.error
        pane.append(msg)
        return
    }

    if (!resp.your_team || !resp.their_team) return

    const categories = getSelectedCategories()
    const yourImproved = resp.your_team.post.h_score > resp.your_team.pre.h_score
    const theirImproved = resp.their_team.post.h_score > resp.their_team.pre.h_score

    // Sub-tabs: Your Team / Their Team
    const tabNav = document.createElement('div')
    tabNav.className = 'trade-tab-nav'

    const yourBtn = document.createElement('button')
    yourBtn.type = 'button'
    yourBtn.className = 'trade-tab-btn active'
    yourBtn.textContent = `Your Team ${yourImproved ? '\u{1F44D}' : '\u{1F44E}'}`

    const theirBtn = document.createElement('button')
    theirBtn.type = 'button'
    theirBtn.className = 'trade-tab-btn'
    theirBtn.textContent = `Their Team ${theirImproved ? '\u{1F44D}' : '\u{1F44E}'}`

    tabNav.append(yourBtn, theirBtn)
    pane.append(tabNav)

    const yourPane = document.createElement('div')
    const theirPane = document.createElement('div')
    theirPane.style.display = 'none'
    pane.append(yourPane, theirPane)

    yourBtn.addEventListener('click', () => {
        yourBtn.classList.add('active');  theirBtn.classList.remove('active')
        yourPane.style.display = '';      theirPane.style.display = 'none'
    })
    theirBtn.addEventListener('click', () => {
        theirBtn.classList.add('active'); yourBtn.classList.remove('active')
        theirPane.style.display = '';     yourPane.style.display = 'none'
    })

    yourPane.append(buildHScoreComparisonTable(resp.your_team.pre, resp.your_team.post, categories))
    theirPane.append(buildHScoreComparisonTable(resp.their_team.pre, resp.their_team.post, categories))
}

function buildHScoreComparisonTable(
    pre: { h_score: number; rates: number[] }
    , post: { h_score: number; rates: number[] }
    , categories: string[]
): HTMLTableElement {
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    hrow.insertCell().textContent = ''
    const hTh = document.createElement('th')
    hTh.textContent = 'H-score'
    hrow.append(hTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.textContent = cat
        hrow.append(th)
    }

    const tbody = table.createTBody()

    function addRow(label: string, hScore: number, rates: number[]): void {
        const tr = tbody.insertRow()
        const labelCell = tr.insertCell()
        labelCell.textContent = label
        labelCell.className = 'trade-row-label'

        const hCell = tr.insertCell()
        hCell.textContent = (hScore * 100).toFixed(2) + '%'
        hCell.style.cssText = stat_styler_primary(hScore - 0.5, MATCHUP_H_MULTIPLIER, 0)

        for (const rate of rates) {
            const td = tr.insertCell()
            td.textContent = (rate * 100).toFixed(1) + '%'
            td.style.cssText = stat_styler_primary(rate - 0.5, MATCHUP_H_MULTIPLIER, 0)
        }
    }

    addRow('Pre-trade', pre.h_score, pre.rates)
    addRow('Post-trade', post.h_score, post.rates)

    return table
}

// ─── Suggested trades ───────────────────────────────────────────────────────

/** A suggested trade together with the team it is made with: suggestions are searched across every other team. */
interface TeamTradeSuggestion {
    team:       string
    suggestion: TradeSuggestion
}

/** The trade-size label a suggestion belongs to ("2 for 2"), as the Trade sizes selector names them. */
function readTradeSizeLabel(suggestion: TradeSuggestion): string {
    return `${suggestion.send.length} for ${suggestion.receive.length}`
}

/**
 * Merges a band's batch of suggestions into the suggestions already found, both sorted by your score, highest first:
 * the merge step of mergesort, so each batch slots in as it arrives without re-sorting what is already shown. On a tie
 * the suggestion found earlier stays first.
 */
function mergeSuggestionsByYourScore(
    found: TeamTradeSuggestion[]
    , batch: TeamTradeSuggestion[]
): TeamTradeSuggestion[] {
    const merged: TeamTradeSuggestion[] = []
    let foundIndex = 0
    let batchIndex = 0
    while (foundIndex < found.length && batchIndex < batch.length) {
        if (batch[batchIndex].suggestion.your_score > found[foundIndex].suggestion.your_score) {
            merged.push(batch[batchIndex++])
        } else {
            merged.push(found[foundIndex++])
        }
    }
    return merged.concat(found.slice(foundIndex), batch.slice(batchIndex))
}

/** Whether a suggestion involves a player whose name contains the (normalised) search text; any does when empty. */
function isSuggestionMatchingSearch(
    suggestion: TradeSuggestion
    , normalizedSearchText: string
): boolean {
    if (normalizedSearchText === '') return true
    return [...suggestion.send, ...suggestion.receive].some(playerId =>
        normalizeSearchText(getRegistryEntry(playerId).name).includes(normalizedSearchText))
}

/** Builds the suggestions table. A row click hands its trade to `onSuggestionChosen`. The team traded with has its
 *  own column; narrow screens hide that column and show the team above the received players instead. */
function buildSuggestionTable(
    suggestions: TeamTradeSuggestion[]
    , onSuggestionChosen: (chosen: TeamTradeSuggestion) => void
): HTMLTableElement {
    const table = document.createElement('table')
    table.className = 'trade-result-table trade-suggestion-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    for (const header of ['Team', 'Send', 'Receive', 'Your Score', 'Their Score']) {
        const th = document.createElement('th')
        th.textContent = header
        if (header === 'Team') th.className = 'trade-suggestion-team-cell'
        if (header === 'Your Score' || header === 'Their Score') th.className = 'trade-suggestion-score'
        if (header === 'Team' || header === 'Send' || header === 'Receive') th.style.textAlign = 'left'
        hrow.append(th)
    }

    const tbody = table.createTBody()

    for (const chosen of suggestions) {
        const { team, suggestion } = chosen
        const tr = tbody.insertRow()
        tr.style.cursor = 'pointer'
        tr.dataset.team = team

        const teamCell = tr.insertCell()
        teamCell.className = 'trade-suggestion-team-cell'
        teamCell.textContent = team
        teamCell.style.textAlign = 'left'

        const sendCell = tr.insertCell()
        sendCell.className = 'trade-suggestion-players'
        sendCell.innerHTML = suggestion.send.map(buildMinimalPlayerDisplayHtml).join('')
        sendCell.style.textAlign = 'left'

        const receiveCell = tr.insertCell()
        receiveCell.className = 'trade-suggestion-players'
        const teamLine = document.createElement('div')
        teamLine.className = 'trade-suggestion-team-line'
        teamLine.textContent = `from ${team}`
        receiveCell.append(teamLine)
        receiveCell.insertAdjacentHTML('beforeend', suggestion.receive.map(buildMinimalPlayerDisplayHtml).join(''))
        receiveCell.style.textAlign = 'left'

        const yourCell = tr.insertCell()
        yourCell.className = 'trade-suggestion-score'
        yourCell.textContent = (suggestion.your_score * 100).toFixed(2) + '%'
        yourCell.style.cssText = stat_styler_primary(suggestion.your_score, SUGGESTION_SCORE_MULTIPLIER, 0)

        const theirCell = tr.insertCell()
        theirCell.className = 'trade-suggestion-score'
        theirCell.textContent = (suggestion.their_score * 100).toFixed(2) + '%'
        theirCell.style.cssText = stat_styler_primary(suggestion.their_score, SUGGESTION_SCORE_MULTIPLIER, 0)

        tr.addEventListener('click', () => onSuggestionChosen(chosen))
    }

    return table
}

/** A threshold input's value as the backend takes it: a percentage, as a fraction. Throws on a value that is not a
 *  number, rather than sending NaN (which matches no trade). */
function readThreshold(input: HTMLInputElement): number {
    const percentage = parseFloat(input.value)
    if (Number.isNaN(percentage)) throw new Error(`Trade threshold ${input.id} is not a number: "${input.value}"`)
    return percentage / 100
}

/** The Trade sizes the user chose last time, among those offered; the first size when none of them is remembered. */
function readRememberedTradeSizes(tradeSizeLabels: string[]): string[] {
    const remembered = pref<string[]>('ts-trade-sizes', []).filter(label => tradeSizeLabels.includes(label))
    return remembered.length > 0 ? remembered : [tradeSizeLabels[0]]
}

// Tracks the listeners attached by the most recent renderSeasonTrading call so
// they can be detached before the next one. Without this, switching tabs in and
// out of Trading would leave the previous render's custom selects and
// multiselect widgets' internal listeners bound to detached nodes.
let tradingListenerController: AbortController | null = null

// ─── Main render ─────────────────────────────────────────────────────────────

/** Renders the full Trading tab into the given container element. */
export function renderSeasonTrading(container: HTMLElement): void {
    tradingListenerController?.abort()
    tradingListenerController = new AbortController()

    container.innerHTML = ''

    const teamNames   = getTeamIdentitiesFromSidebar()
    const assignments = readRosterAssignments()

    const nPicks = readRequiredIntInput('ls-n-picks')
    const fullTeams = teamNames.filter(t => (assignments[t]?.length ?? 0) >= nPicks)

    if (fullTeams.length < 2) {
        // Nothing while a connected league's rosters are pending: the indicator beside the tabs says so.
        const message = readEmptyRostersMessage('two full teams', 'Trading')
        if (message !== null) {
            const msg = document.createElement('p')
            msg.className   = 'coming-soon'
            msg.textContent = message
            container.append(msg)
        }
        return
    }

    // ── Top line: your team ──────────────────────────────────────────────────

    const selectorRow = document.createElement('div')
    selectorRow.className = 'trade-selector-row'

    const yourWrap = document.createElement('div')
    yourWrap.className = 'trade-selector-col'
    const yourLabel = document.createElement('div')
    yourLabel.className = 'pick-control-label'
    yourLabel.textContent = 'Which team do you want to trade from?'
    yourWrap.append(yourLabel)
    // Both teams are remembered across visits and reloads -- your team is shared with the Waiver tab -- and a
    // remembered team this league does not have (another league, or a roster no longer full) falls back to the first.
    const savedYourTeam  = pref<string | null>('season-your-team', null)
    const initialYourTeam = savedYourTeam !== null && fullTeams.includes(savedYourTeam) ? savedYourTeam : fullTeams[0]
    const yourTeamSel = makeCustomSelect(
        'trade-your-team',
        fullTeams.map(n => ({ value: n, label: n })),
        initialYourTeam,
        undefined,
        tradingListenerController.signal,
    )
    yourWrap.append(yourTeamSel.element)
    selectorRow.append(yourWrap)
    container.append(selectorRow)

    const divider = document.createElement('hr')
    divider.className = 'trade-divider'
    container.append(divider)

    // ── Trade inspection: the team traded with, the players, the results ─────
    // The counterparty belongs here rather than on the top line: suggestions are searched across every other team, so
    // it only says which trade is being inspected.

    const counterpartyRow = document.createElement('div')
    counterpartyRow.className = 'trade-selector-row'
    const theirWrap = document.createElement('div')
    theirWrap.className = 'trade-selector-col'
    const theirLabel = document.createElement('div')
    theirLabel.className = 'pick-control-label'
    theirLabel.textContent = 'Which team do you want to trade with?'
    theirWrap.append(theirLabel)
    const counterpartyOptions = fullTeams.filter(n => n !== initialYourTeam)
    const savedTheirTeam = pref<string | null>('ts-their-team', null)
    const initialTheirTeam = savedTheirTeam !== null && counterpartyOptions.includes(savedTheirTeam)
        ? savedTheirTeam : counterpartyOptions[0]
    const theirTeamSel = makeCustomSelect(
        'trade-their-team',
        counterpartyOptions.map(n => ({ value: n, label: n })),
        initialTheirTeam,
        undefined,
        tradingListenerController.signal,
    )
    theirWrap.append(theirTeamSel.element)
    counterpartyRow.append(theirWrap)
    container.append(counterpartyRow)

    const inspectionArea = document.createElement('div')
    container.append(inspectionArea)

    // ── Suggested trades ─────────────────────────────────────────────────────

    const divider2 = document.createElement('hr')
    divider2.className = 'trade-divider'
    container.append(divider2)

    // Combo filter row — "Suggested trades" label on left, controls on right
    const comboRow = document.createElement('div')
    comboRow.className = 'trade-combo-row'

    const suggestHeader = document.createElement('span')
    suggestHeader.className = 'trade-suggest-header'
    suggestHeader.textContent = 'Suggestions'

    comboRow.append(suggestHeader)

    // Find player / Find team: narrow the suggestions to trades involving a player (as the draft and auction candidate
    // search does) and to trades with a team. Their placeholders are their labels, to save room on the row.
    const searchGroup = document.createElement('div')
    searchGroup.className = 'trade-threshold-group trade-search-group'
    function makeSuggestionSearchInput(
        id: string
        , placeholder: string
    ): HTMLInputElement {
        const input = document.createElement('input')
        input.id = id
        input.className = 'player-search-input trade-search-input'
        input.type = 'search'
        input.placeholder = placeholder
        input.setAttribute('aria-label', placeholder)
        input.autocomplete = 'off'
        return input
    }
    const searchInput     = makeSuggestionSearchInput('trade-suggestion-search', 'Find player')
    const teamSearchInput = makeSuggestionSearchInput('trade-team-search', 'Find team')
    searchGroup.append(searchInput, teamSearchInput)
    comboRow.append(searchGroup)

    // Combo filter group — right-justified, with "Trade sizes" caption
    const comboGroup = document.createElement('div')
    comboGroup.className = 'trade-threshold-group'
    comboGroup.style.marginLeft = 'auto'

    const comboSizeLabel = document.createElement('span')
    comboSizeLabel.className = 'trade-threshold-label'
    comboSizeLabel.textContent = 'Trade sizes'

    const comboOptions = DEFAULT_COMBOS.map(cp => `${cp.n_traded} for ${cp.n_received}`)
    const comboSel = makeMultiSelectWidget('', comboOptions.map(combo => ({ value: combo, label: combo })))
    comboSel.element.classList.add('trade-combo-select')
    comboSel.setSelectedSilently(readRememberedTradeSizes(comboOptions))

    comboGroup.append(comboSizeLabel, comboSel.element)
    comboRow.append(comboGroup)

    // Your threshold control group
    const yourThreshGroup = document.createElement('div')
    yourThreshGroup.className = 'trade-threshold-group'
    const yourThreshLabel = document.createElement('label')
    yourThreshLabel.htmlFor = 'ts-your-threshold'
    yourThreshLabel.textContent = 'Your threshold'
    const yourThreshInfo = document.createElement('button')
    yourThreshInfo.type = 'button'
    yourThreshInfo.className = 'info-btn'
    yourThreshInfo.textContent = 'ⓘ'
    yourThreshInfo.dataset.tooltip = 'Minimum H-score improvement required for a trade to be suggested for your team, as a percentage, down to -2. Default: 0.'
    const yourThreshInput = makeNumberInput('ts-your-threshold', Math.max(pref('ts-your-threshold', 0), TRADE_THRESHOLD_FLOOR_PERCENT),
        TRADE_THRESHOLD_FLOOR_PERCENT)
    yourThreshInput.className = ''
    yourThreshInput.step = '0.1'
    const yourThreshWrap = document.createElement('div')
    yourThreshWrap.className = 'trade-threshold-input-wrap'
    yourThreshWrap.append(yourThreshInput)
    yourThreshGroup.append(yourThreshLabel, yourThreshInfo, yourThreshWrap)
    comboRow.append(yourThreshGroup)

    // Their threshold control group
    const theirThreshGroup = document.createElement('div')
    theirThreshGroup.className = 'trade-threshold-group'
    const theirThreshLabel = document.createElement('label')
    theirThreshLabel.htmlFor = 'ts-their-threshold'
    theirThreshLabel.textContent = 'Their threshold'
    const theirThreshInfo = document.createElement('button')
    theirThreshInfo.type = 'button'
    theirThreshInfo.className = 'info-btn'
    theirThreshInfo.textContent = 'ⓘ'
    theirThreshInfo.dataset.tooltip = 'Minimum H-score improvement for the counterparty team, as a percentage. Negative values allow trades that hurt them slightly, down to -2. Default: 0.'
    const theirThreshInput = makeNumberInput('ts-their-threshold', Math.max(pref('ts-their-threshold', 0), TRADE_THRESHOLD_FLOOR_PERCENT),
        TRADE_THRESHOLD_FLOOR_PERCENT)
    theirThreshInput.className = ''
    theirThreshInput.step = '0.1'
    const theirThreshWrap = document.createElement('div')
    theirThreshWrap.className = 'trade-threshold-input-wrap'
    theirThreshWrap.append(theirThreshInput)
    theirThreshGroup.append(theirThreshLabel, theirThreshInfo, theirThreshWrap)
    comboRow.append(theirThreshGroup)

    // Positively framed (CLAUDE.md): the toggle says what it does when on, and it is on by
    // default. The old 'ts-ignore-position' preference is deliberately orphaned — its polarity
    // is inverted, so carrying it over would flip every stored choice.
    // Its label on the left, like every other control in this row.
    const checkPositionsToggle = makeSidebarToggle('ts-check-positions', '', 'Check positions')
    comboRow.append(checkPositionsToggle)

    container.append(comboRow)

    const checkPositionsInput = document.getElementById('ts-check-positions') as HTMLInputElement
    checkPositionsInput.checked = pref('ts-check-positions', true)

    const suggestResults = document.createElement('div')
    suggestResults.dataset.testid = 'trade-suggestions'
    container.append(suggestResults)

    // ── Suggestion state ─────────────────────────────────────────────────────
    // Every suggestion found so far, across teams and trade sizes, sorted by your score; the display filters it by the
    // chosen trade sizes and the search. Each team's batch merges in as it arrives. A change that invalidates what
    // was found (your team, a threshold, the position check) starts a new search generation: a response from an
    // older one is dropped rather than mixed in.
    let rankedSuggestions: TeamTradeSuggestion[] = []
    const fetchedTradeSizesByRound = new Map<number, Set<string>>()   // round -> trade sizes searched, every team
    const pendingRoundTradeSizes   = new Set<string>()                // `${round}|${trade size}`
    let suggestionGeneration = 0
    // How many of the matching suggestions are drawn; "Show more" raises it. Back to one page whenever what matches
    // changes (a new search, a threshold, a search box, the trade sizes).
    let suggestionRowsShown = SUGGESTION_ROWS_PER_PAGE
    // Searches of the current generation still running. "No promising trades found" waits for all of them: between two
    // teams' batches nothing is pending, and the message showed after the first team that had none.
    let runningSearchCount = 0

    let currentSendSel:       MultiSelectWidget | null = null
    let currentReceiveSel:    MultiSelectWidget | null = null
    let currentUpdateResults: (() => void) | null      = null

    function readYourTeam(): string {
        return yourTeamSel.getValue()
    }

    function readCounterparties(): string[] {
        return fullTeams.filter(team => team !== readYourTeam())
    }

    function refreshSuggestionDisplay(): void {
        suggestResults.innerHTML = ''
        const selected = comboSel.getSelected()
        if (selected.length === 0) {
            const msg = document.createElement('div')
            msg.className = 'trade-suggestion-stub'
            msg.textContent = 'Select at least one trade combination.'
            suggestResults.append(msg)
            return
        }
        const normalizedSearchText = normalizeSearchText(searchInput.value)
        const normalizedTeamSearchText = normalizeSearchText(teamSearchInput.value)
        // Strictly above, as the backend filters.
        const yourThreshold  = readThreshold(yourThreshInput)
        const theirThreshold = readThreshold(theirThreshInput)
        const shown = rankedSuggestions.filter(found =>
            found.suggestion.your_score > yourThreshold
            && found.suggestion.their_score > theirThreshold
            && selected.includes(readTradeSizeLabel(found.suggestion))
            && isSuggestionMatchingSearch(found.suggestion, normalizedSearchText)
            && normalizeSearchText(found.team).includes(normalizedTeamSearchText))
        if (shown.length > 0) {
            suggestResults.append(buildSuggestionTable(shown.slice(0, suggestionRowsShown), showSuggestedTrade))
            if (shown.length > suggestionRowsShown) {
                const moreButton = document.createElement('button')
                moreButton.type = 'button'
                moreButton.className = 'section-apply-btn trade-show-more-btn'
                moreButton.textContent = `Show more (${suggestionRowsShown} of ${shown.length} shown)`
                moreButton.addEventListener('click', () => {
                    suggestionRowsShown += SUGGESTION_ROWS_PER_PAGE
                    refreshSuggestionDisplay()
                })
                suggestResults.append(moreButton)
            }
        } else if (runningSearchCount === 0) {
            const msg = document.createElement('div')
            msg.className = 'trade-suggestion-stub'
            msg.textContent = normalizedSearchText === '' && normalizedTeamSearchText === ''
                ? 'No promising trades found.'
                : 'No suggested trade matches the search.'
            suggestResults.append(msg)
        }
    }

    /** Searches every other team for the chosen trade sizes not yet searched (searchEveryTeam), counting the search
     *  as running until it ends. */
    async function fetchMissingSuggestions(): Promise<void> {
        const generation = suggestionGeneration
        runningSearchCount += 1
        try {
            await searchEveryTeam(generation)
        } finally {
            if (generation === suggestionGeneration) {
                runningSearchCount -= 1
                refreshSuggestionDisplay()
            }
        }
    }

    /** Searches in value-gap bands (TRADE_SEARCH_ROUNDS), the closest trades first. One request (one batch) per band,
     *  covering every team, in turn: the backend runs one request per session at a time anyway, and in turn a newer
     *  generation stops this one between bands. */
    async function searchEveryTeam(generation: number): Promise<void> {
        const yourTeam = readYourTeam()
        // Always at the floor: the thresholds filter what this finds (refreshSuggestionDisplay).
        const your_differential_threshold  = TRADE_THRESHOLD_FLOOR_PERCENT / 100
        const their_differential_threshold = TRADE_THRESHOLD_FLOOR_PERCENT / 100
        const positionCheck = checkPositionsInput.checked
        const counterparties = readCounterparties()
        refreshSuggestionDisplay()
        for (const [roundIndex, round] of TRADE_SEARCH_ROUNDS.entries()) {
            if (generation !== suggestionGeneration) return
            const fetched = fetchedTradeSizesByRound.get(roundIndex) ?? new Set<string>()
            const missing = comboSel.getSelected().filter(label =>
                !fetched.has(label) && !pendingRoundTradeSizes.has(`${roundIndex}|${label}`))
            if (missing.length === 0) continue
            const comboParams = missing.map(label => {
                const combo = DEFAULT_COMBOS.find(cp => `${cp.n_traded} for ${cp.n_received}` === label)
                if (!combo) throw new Error(`Unknown trade size "${label}"`)
                return { ...combo, threshold: round.threshold, searched_threshold: round.searchedThreshold }
            })
            missing.forEach(label => pendingRoundTradeSizes.add(`${roundIndex}|${label}`))
            let suggestions: TradeSuggestion[]
            try {
                const response = await runTradeSuggest(assignments, yourTeam, counterparties, comboParams,
                    your_differential_threshold, their_differential_threshold, positionCheck)
                suggestions = response.suggestions
            } catch (err) {
                if (generation !== suggestionGeneration) return
                missing.forEach(label => pendingRoundTradeSizes.delete(`${roundIndex}|${label}`))
                console.error('Trade suggestions failed:', err)
                continue
            }
            if (generation !== suggestionGeneration) return
            missing.forEach(label => pendingRoundTradeSizes.delete(`${roundIndex}|${label}`))
            fetchedTradeSizesByRound.set(roundIndex, new Set([...fetched, ...missing]))
            // The backend ranks a band across its teams already; sorted again here, the merge does not depend on it.
            const batch = suggestions
                .map(suggestion => ({ team: suggestion.their_team, suggestion }))
                .sort((a, b) => b.suggestion.your_score - a.suggestion.your_score)
            rankedSuggestions = mergeSuggestionsByYourScore(rankedSuggestions, batch)
            refreshSuggestionDisplay()
        }
    }

    function restartSuggestionSearch(): void {
        if (!document.contains(suggestResults)) return
        suggestionGeneration += 1
        suggestionRowsShown = SUGGESTION_ROWS_PER_PAGE
        runningSearchCount = 0
        rankedSuggestions = []
        fetchedTradeSizesByRound.clear()
        pendingRoundTradeSizes.clear()
        fetchMissingSuggestions().catch(err => console.error('Trade suggestions failed:', err))
    }

    // ── Inspection body: rebuilt when either team changes ────────────────────

    function rebuildInspection(): void {
        inspectionArea.innerHTML = ''

        const yourTeam  = readYourTeam()
        const theirTeam = theirTeamSel.getValue()

        const yourPlayers  = listMovablePlayers(assignments[yourTeam])
        const theirPlayers = listMovablePlayers(assignments[theirTeam])

        const bodyRow = document.createElement('div')
        bodyRow.className = 'trade-body-row'

        // ── Left: multi-selects ──────────────────────────────────────────────

        const leftCol = document.createElement('div')
        leftCol.className = 'trade-left-col'

        const sendSel    = makeMultiSelectWidget('Which players are you trading?',   buildPlayerOptions(yourPlayers))
        const receiveSel = makeMultiSelectWidget('Which players are you receiving?', buildPlayerOptions(theirPlayers))
        currentSendSel = sendSel
        currentReceiveSel = receiveSel

        leftCol.append(sendSel.element)
        leftCol.append(receiveSel.element)
        bodyRow.append(leftCol)

        // ── Right: result tabs ───────────────────────────────────────────────

        const rightCol = document.createElement('div')
        rightCol.className = 'trade-right-col'

        // Tab buttons
        const tabNav = document.createElement('div')
        tabNav.className = 'trade-tab-nav'
        const hBtn = document.createElement('button')
        hBtn.type = 'button'
        hBtn.className = 'trade-tab-btn active'
        hBtn.textContent = 'H-score'
        const gBtn = document.createElement('button')
        gBtn.type = 'button'
        gBtn.className = 'trade-tab-btn'
        gBtn.textContent = 'G-score'
        tabNav.append(hBtn, gBtn)
        rightCol.append(tabNav)

        // Tab panes
        const hPane = document.createElement('div')
        hPane.className = 'trade-tab-pane'
        hPane.dataset.testid = 'trade-hscore-pane'
        const gPane = document.createElement('div')
        gPane.className = 'trade-tab-pane'
        gPane.dataset.testid = 'trade-gscore-pane'
        gPane.style.display = 'none'
        rightCol.append(hPane, gPane)

        hBtn.addEventListener('click', () => {
            hBtn.classList.add('active');  gBtn.classList.remove('active')
            hPane.style.display = '';      gPane.style.display = 'none'
        })
        gBtn.addEventListener('click', () => {
            gBtn.classList.add('active');  hBtn.classList.remove('active')
            gPane.style.display = '';      hPane.style.display = 'none'
        })

        function updateResults(): void {
            hPane.innerHTML = ''
            gPane.innerHTML = ''

            const sent     = parseSelectedPlayerIds(sendSel.getSelected())
            const received = parseSelectedPlayerIds(receiveSel.getSelected())

            if (sent.length === 0 || received.length === 0) {
                const msg = document.createElement('p')
                msg.className = 'coming-soon'
                msg.textContent = 'A trade must include at least one player from each team.'
                hPane.append(msg)
                gPane.append(msg.cloneNode(true))
                return
            }

            if (sent.length !== received.length) {
                const msg = document.createElement('p')
                msg.className = 'coming-soon'
                msg.textContent = 'Trades must include an equal number of players from both teams.'
                hPane.append(msg)
                gPane.append(msg.cloneNode(true))
                return
            }

            buildHScoreResult(hPane, assignments, yourTeam, theirTeam, sent, received)
            gPane.append(buildGScoreTable(sent, received))
        }

        currentUpdateResults = updateResults
        updateResults()
        sendSel.onChange(updateResults)
        receiveSel.onChange(updateResults)

        bodyRow.append(rightCol)
        inspectionArea.append(bodyRow)
    }

    /** Inspects a suggested trade: switches the inspection to the team it is made with, then fills in the players. */
    function showSuggestedTrade(chosen: TeamTradeSuggestion): void {
        if (theirTeamSel.getValue() !== chosen.team) {
            theirTeamSel.setValue(chosen.team)   // its change listener rebuilds the inspection for that team
        }
        if (!currentSendSel || !currentReceiveSel || !currentUpdateResults) {
            throw new Error('The trade inspection was not built before a suggestion was chosen')
        }
        // Both set silently, then one update: two onChange calls would send a stale analysis on the first.
        currentSendSel.setSelectedSilently(chosen.suggestion.send.map(String))
        currentReceiveSel.setSelectedSilently(chosen.suggestion.receive.map(String))
        currentUpdateResults()
    }

    rebuildInspection()
    restartSuggestionSearch()

    /** A threshold only filters the trades found (every search runs at the floor). One below the floor is raised to
     *  it: nothing below was searched. */
    function applyThresholdChange(input: HTMLInputElement, prefKey: string): void {
        if (parseFloat(input.value) < TRADE_THRESHOLD_FLOOR_PERCENT) input.value = String(TRADE_THRESHOLD_FLOOR_PERCENT)
        savePref(prefKey, parseFloat(input.value))
        suggestionRowsShown = SUGGESTION_ROWS_PER_PAGE
        refreshSuggestionDisplay()
    }
    yourThreshInput.addEventListener('change', () => applyThresholdChange(yourThreshInput, 'ts-your-threshold'))
    theirThreshInput.addEventListener('change', () => applyThresholdChange(theirThreshInput, 'ts-their-threshold'))
    checkPositionsInput.addEventListener('change', () => {
        savePref('ts-check-positions', checkPositionsInput.checked)
        restartSuggestionSearch()
    })
    comboSel.onChange(() => {
        savePref('ts-trade-sizes', comboSel.getSelected())
        suggestionRowsShown = SUGGESTION_ROWS_PER_PAGE
        fetchMissingSuggestions().catch(err => console.error('Trade suggestions failed:', err))
    })
    function applySearchChange(): void {
        suggestionRowsShown = SUGGESTION_ROWS_PER_PAGE
        refreshSuggestionDisplay()
    }
    searchInput.addEventListener('input', applySearchChange, { signal: tradingListenerController.signal })
    teamSearchInput.addEventListener('input', applySearchChange, { signal: tradingListenerController.signal })
    yourTeamSel.element.addEventListener('change', () => {
        const yourTeam = readYourTeam()
        savePref('season-your-team', yourTeam)
        // Keep the counterparty when it is still a valid one (anyone but your new team).
        const options = fullTeams.filter(n => n !== yourTeam)
        const theirTeam = theirTeamSel.getValue()
        theirTeamSel.setOptions(options.map(n => ({ value: n, label: n })),
                                options.includes(theirTeam) ? theirTeam : options[0])
        savePref('ts-their-team', theirTeamSel.getValue())
        rebuildInspection()
        restartSuggestionSearch()
    })
    // The team traded with only changes which trade is inspected: the suggestions already cover every team.
    theirTeamSel.element.addEventListener('change', () => {
        savePref('ts-their-team', theirTeamSel.getValue())
        rebuildInspection()
    })
}
