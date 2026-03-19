// data_entry/season/season_trading.ts
// Renders the Season → Trading tab.
// Replicates the structure of src/tabs/trading.py:
//   1. Team selectors (your team / counterparty)
//   2. Player multi-selects (send / receive) + result sub-tabs (H-score / G-score)
//   3. Suggested trades table

import { makeCustomSelect } from '../../custom_select.js'
import { makeMultiSelectWidget, MultiSelectWidget } from '../../helper_functions.js'
import { getGScoreByName } from '../../app_state.js'
import { getFormatAndCategories } from '../../parameter_collection/format_and_categories.js'
import { stat_styler_primary } from '../../styles/styler_functions.js'
import { getTradeParameters } from '../../parameter_collection/trade_parameters.js'
import { runTradeAnalyze, runTradeSuggest } from '../../api/session.js'
import type { TradeAnalyzeResponse, TradeSuggestion } from '../../api/client.js'

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Reads team names from the sidebar textarea. */
function readTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

/** Reads roster assignments from the Rosters tab grid (sr-player-{row}-{col}). */
function readRosterAssignments(): Record<string, string[]> {
    const teamNames = readTeamNames()
    const nDrafters = parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value) || 12
    const nPicks    = parseInt((document.getElementById('ls-n-picks')    as HTMLInputElement).value) || 13

    const assignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d] ?? `Team ${d + 1}`
        const players: string[] = []
        for (let r = 0; r < nPicks; r++) {
            const input = document.getElementById(`sr-player-${r}-${d}`) as HTMLInputElement | null
            const val = input?.value ?? ''
            if (val) players.push(val)
        }
        assignments[team] = players
    }
    return assignments
}


// ─── G-score comparison table ────────────────────────────────────────────────

/** Builds a G-score comparison table for the selected send/receive players. */
function buildGScoreTable(sent: string[], received: string[]): HTMLElement {
    const container = document.createElement('div')
    container.className = 'trade-gscore-section'

    const gScoreMap = getGScoreByName()
    const categories = getFormatAndCategories().categories

    type Row = { label: string; total: number; values: number[] }

    function getPlayerGRow(name: string): Row | null {
        const gs = gScoreMap.get(name)
        if (!gs) return null
        return { label: gs.name, total: gs.total, values: gs.values }
    }

    const sentRows:     Row[] = []
    const receivedRows: Row[] = []
    for (const name of sent) {
        const r = getPlayerGRow(name)
        if (r) sentRows.push(r)
    }
    for (const name of received) {
        const r = getPlayerGRow(name)
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

    function addRow(label: string, row: Row, cssClass: string): void {
        const tr = tbody.insertRow()
        tr.className = cssClass
        const labelCell = tr.insertCell()
        labelCell.textContent = label
        labelCell.className = 'trade-row-label'
        const totalCell = tr.insertCell()
        totalCell.textContent = row.total.toFixed(2)
        totalCell.style.cssText = stat_styler_primary(row.total, 60, 0)
        for (const v of row.values) {
            const td = tr.insertCell()
            td.textContent = v.toFixed(2)
            td.style.cssText = stat_styler_primary(v, 60, 0)
        }
    }

    for (let i = 0; i < sent.length; i++) {
        if (sentRows[i]) addRow(sent[i], sentRows[i], 'trade-row-sent')
    }
    addRow('Total Sent', sentTotal, 'trade-row-subtotal')

    for (let i = 0; i < received.length; i++) {
        if (receivedRows[i]) addRow(received[i], receivedRows[i], 'trade-row-received')
    }
    addRow('Total Received', recvTotal, 'trade-row-subtotal')

    addRow('Total Difference', diff, 'trade-row-diff')

    container.append(table)
    return container
}

// ─── H-score trade analysis ─────────────────────────────────────────────────

/** Builds the H-score trade result display with pre/post comparison for both teams. */
function buildHScoreResult(
    pane: HTMLElement,
    assignments: Record<string, string[]>,
    yourTeam: string,
    theirTeam: string,
    sent: string[],
    received: string[],
): void {
    pane.innerHTML = ''

    const loading = document.createElement('div')
    loading.className = 'trade-hscore-stub'
    loading.textContent = 'Analyzing trade...'
    pane.append(loading)

    const { ignore_position_check } = getTradeParameters()
    runTradeAnalyze(assignments, yourTeam, theirTeam, sent, received, ignore_position_check)
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

    const categories = getFormatAndCategories().categories
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
    pre: { h_score: number; rates: number[] },
    post: { h_score: number; rates: number[] },
    categories: string[],
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
        hCell.style.cssText = stat_styler_primary(hScore - 0.5, 200, 0)

        for (const rate of rates) {
            const td = tr.insertCell()
            td.textContent = (rate * 100).toFixed(1) + '%'
            td.style.cssText = stat_styler_primary(rate - 0.5, 200, 0)
        }
    }

    addRow('Pre-trade', pre.h_score, pre.rates)
    addRow('Post-trade', post.h_score, post.rates)

    return table
}

// ─── Suggested trades ───────────────────────────────────────────────────────

function runSuggestionSearch(
    resultsArea: HTMLElement,
    comboSel: MultiSelectWidget,
    assignments: Record<string, string[]>,
    yourTeam: string,
    theirTeam: string,
    sendSel: MultiSelectWidget,
    receiveSel: MultiSelectWidget,
): void {
    resultsArea.innerHTML = ''
    const { combo_params, your_differential_threshold, their_differential_threshold, ignore_position_check } = getTradeParameters()

    const selected = comboSel.getSelected()
    if (selected.length === 0) {
        const msg = document.createElement('div')
        msg.className = 'trade-suggestion-stub'
        msg.textContent = 'Select at least one trade combination.'
        resultsArea.append(msg)
        return
    }

    // Filter combo_params to only the selected ones
    const filteredCombos = combo_params.filter(cp =>
        selected.includes(`${cp.n_traded} for ${cp.n_received}`)
    )

    const loading = document.createElement('div')
    loading.className = 'trade-suggestion-stub'
    loading.textContent = 'Finding suggested trades...'
    resultsArea.append(loading)

    runTradeSuggest(
        assignments, yourTeam, theirTeam,
        filteredCombos, your_differential_threshold, their_differential_threshold,
        ignore_position_check,
    )
        .then(resp => {
            loading.remove()
            if (resp.suggestions.length === 0) {
                const msg = document.createElement('div')
                msg.className = 'trade-suggestion-stub'
                msg.textContent = 'No promising trades found.'
                resultsArea.append(msg)
                return
            }
            resultsArea.append(buildSuggestionTable(resp.suggestions, sendSel, receiveSel))
        })
        .catch(err => {
            loading.remove()
            const msg = document.createElement('div')
            msg.className = 'trade-suggestion-stub'
            msg.textContent = `Error: ${err.message}`
            resultsArea.append(msg)
        })
}

function buildSuggestionTable(
    suggestions: TradeSuggestion[],
    sendSel: MultiSelectWidget,
    receiveSel: MultiSelectWidget,
): HTMLTableElement {
    const table = document.createElement('table')
    table.className = 'trade-result-table'

    const thead = table.createTHead()
    const hrow = thead.insertRow()
    for (const header of ['Send', 'Receive', 'Your Score', 'Their Score']) {
        const th = document.createElement('th')
        th.textContent = header
        if (header === 'Send' || header === 'Receive') th.style.textAlign = 'left'
        hrow.append(th)
    }

    const tbody = table.createTBody()

    for (const sug of suggestions) {
        const tr = tbody.insertRow()
        tr.style.cursor = 'pointer'

        const sendCell = tr.insertCell()
        sendCell.textContent = sug.send.join(', ')
        sendCell.style.textAlign = 'left'

        const recvCell = tr.insertCell()
        recvCell.textContent = sug.receive.join(', ')
        recvCell.style.textAlign = 'left'

        const yourCell = tr.insertCell()
        yourCell.textContent = (sug.your_score * 100).toFixed(2) + '%'
        yourCell.style.cssText = stat_styler_primary(sug.your_score, 15000, 0)

        const theirCell = tr.insertCell()
        theirCell.textContent = (sug.their_score * 100).toFixed(2) + '%'
        theirCell.style.cssText = stat_styler_primary(sug.their_score, 15000, 0)

        // Clicking a suggestion populates the send/receive selectors
        tr.addEventListener('click', () => {
            sendSel.setSelected(sug.send)
            receiveSel.setSelected(sug.receive)
        })
    }

    return table
}

// ─── Main render ─────────────────────────────────────────────────────────────

/** Renders the full Trading tab into the given container element. */
export function renderSeasonTrading(container: HTMLElement): void {
    container.innerHTML = ''

    const teamNames   = readTeamNames()
    const assignments = readRosterAssignments()

    const nPicks = parseInt((document.getElementById('ls-n-picks') as HTMLInputElement).value) || 13
    const fullTeams = teamNames.filter(t => (assignments[t]?.length ?? 0) >= nPicks)

    if (fullTeams.length < 2) {
        const msg = document.createElement('p')
        msg.className = 'coming-soon'
        msg.textContent = 'Fill out at least two full teams on the Rosters tab to use Trading.'
        container.append(msg)
        return
    }

    // ── Row 1: team selectors ────────────────────────────────────────────────

    const selectorRow = document.createElement('div')
    selectorRow.className = 'trade-selector-row'

    // Your team
    const yourWrap = document.createElement('div')
    yourWrap.className = 'trade-selector-col'
    const yourLabel = document.createElement('div')
    yourLabel.className = 'pick-control-label'
    yourLabel.textContent = 'Which team do you want to trade from?'
    yourWrap.append(yourLabel)
    const yourTeamSel = makeCustomSelect(
        'trade-your-team',
        fullTeams.map(n => ({ value: n, label: n })),
    )
    yourWrap.append(yourTeamSel.element)
    selectorRow.append(yourWrap)

    // Counterparty
    const theirWrap = document.createElement('div')
    theirWrap.className = 'trade-selector-col'
    const theirLabel = document.createElement('div')
    theirLabel.className = 'pick-control-label'
    theirLabel.textContent = 'Which team do you want to trade with?'
    theirWrap.append(theirLabel)
    const counterpartyOptions = fullTeams.filter(n => n !== fullTeams[0])
    const theirTeamSel = makeCustomSelect(
        'trade-their-team',
        counterpartyOptions.map(n => ({ value: n, label: n })),
    )
    theirWrap.append(theirTeamSel.element)
    selectorRow.append(theirWrap)

    container.append(selectorRow)

    // ── Divider ──────────────────────────────────────────────────────────────

    const divider = document.createElement('hr')
    divider.className = 'trade-divider'
    container.append(divider)

    // ── Row 2: player selects + results (re-rendered on team change) ─────────

    const bodyArea = document.createElement('div')
    container.append(bodyArea)

    // ── Suggested trades (persists across team changes) ──────────────────

    const divider2 = document.createElement('hr')
    divider2.className = 'trade-divider'
    container.append(divider2)

    const suggestHeader = document.createElement('div')
    suggestHeader.className = 'trade-suggest-header'
    suggestHeader.textContent = 'Suggested trades'
    container.append(suggestHeader)

    // Combo filter multiselect — created once, persists across team changes
    const { combo_params } = getTradeParameters()
    const comboOptions = combo_params.map(cp => `${cp.n_traded} for ${cp.n_received}`)
    const comboSel = makeMultiSelectWidget('Trade combinations to search', comboOptions)
    comboSel.setSelected([comboOptions[0]])
    container.append(comboSel.element)

    const suggestResults = document.createElement('div')
    container.append(suggestResults)

    // Track current send/receive selectors so combo change can re-trigger search
    let currentSendSel: MultiSelectWidget | null = null
    let currentReceiveSel: MultiSelectWidget | null = null

    function rebuildBody(): void {
        bodyArea.innerHTML = ''

        const yourTeam  = yourTeamSel.getValue() || fullTeams[0]
        const theirTeam = theirTeamSel.getValue() || fullTeams[1]

        if (!yourTeam || !theirTeam) {
            bodyArea.innerHTML = '<p class="coming-soon">Select two different teams.</p>'
            return
        }

        const yourPlayers  = assignments[yourTeam]  ?? []
        const theirPlayers = assignments[theirTeam] ?? []

        if (yourPlayers.length < nPicks || theirPlayers.length < nPicks) {
            bodyArea.innerHTML = '<p class="coming-soon">Both teams must have full rosters.</p>'
            return
        }

        const bodyRow = document.createElement('div')
        bodyRow.className = 'trade-body-row'

        // ── Left: multi-selects ──────────────────────────────────────────────

        const leftCol = document.createElement('div')
        leftCol.className = 'trade-left-col'

        const sendSel    = makeMultiSelectWidget('Which players are you trading?',   yourPlayers)
        const receiveSel = makeMultiSelectWidget('Which players are you receiving?', theirPlayers)
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
        const gPane = document.createElement('div')
        gPane.className = 'trade-tab-pane'
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

            const sent     = sendSel.getSelected()
            const received = receiveSel.getSelected()

            if (sent.length === 0 || received.length === 0) {
                const msg = document.createElement('p')
                msg.className = 'coming-soon'
                msg.textContent = 'A trade must include at least one player from each team.'
                hPane.append(msg)
                gPane.append(msg.cloneNode(true))
                return
            }

            buildHScoreResult(hPane, assignments, yourTeam, theirTeam, sent, received)
            gPane.append(buildGScoreTable(sent, received))
        }

        updateResults()
        sendSel.onChange(updateResults)
        receiveSel.onChange(updateResults)

        bodyRow.append(rightCol)
        bodyArea.append(bodyRow)

        // Trigger suggestion search for current teams
        runSuggestionSearch(suggestResults, comboSel, assignments, yourTeam, theirTeam, sendSel, receiveSel)
    }

    rebuildBody()

    comboSel.onChange(() => {
        if (!currentSendSel || !currentReceiveSel) return
        const yourTeam  = yourTeamSel.getValue() || fullTeams[0]
        const theirTeam = theirTeamSel.getValue() || fullTeams[1]
        runSuggestionSearch(suggestResults, comboSel, assignments, yourTeam, theirTeam, currentSendSel, currentReceiveSel)
    })
    yourTeamSel.element.addEventListener('change', () => {
        const yourTeam = yourTeamSel.getValue()
        theirTeamSel.setOptions(
            fullTeams.filter(n => n !== yourTeam).map(n => ({ value: n, label: n })),
        )
        rebuildBody()
    })
    theirTeamSel.element.addEventListener('change', rebuildBody)
}
