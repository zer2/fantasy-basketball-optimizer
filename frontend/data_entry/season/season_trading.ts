// data_entry/season/season_trading.ts
// Renders the Season → Trading tab.
// Replicates the structure of src/tabs/trading.py:
//   1. Team selectors (your team / counterparty)
//   2. Player multi-selects (send / receive) + result sub-tabs (H-score / G-score)
//   3. Suggested trades table
//
// Backend integration is stubbed — the UI structure is complete.

import { makeCustomSelect } from '../../custom_select.js'
import { makeMultiSelectWidget } from '../../helper_functions.js'
import { getPlayerByName, getCategories } from '../../app_state.js'

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

/** Builds a G-score comparison table for the selected send/receive players.
 *  Uses actual G-score data from getPlayers() when available. */
function buildGScoreTable(sent: string[], received: string[]): HTMLElement {
    const container = document.createElement('div')
    container.className = 'trade-gscore-section'

    const playerByName = getPlayerByName()
    const categories = getCategories()

    type Row = { label: string; total: number; values: number[] }

    function getPlayerGRow(name: string): Row | null {
        const p = playerByName.get(name)
        if (!p) return null
        return p.g_score_rows.find((r: Row) => r.label === name) ?? null
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
        for (const v of row.values) {
            const td = tr.insertCell()
            td.textContent = v.toFixed(2)
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

// ─── H-score stub ────────────────────────────────────────────────────────────

function buildHScoreStub(): HTMLElement {
    const el = document.createElement('div')
    el.className = 'trade-hscore-stub'
    el.textContent = 'H-score trade analysis will appear here once the backend is connected.'
    return el
}

// ─── Suggestion table stub ───────────────────────────────────────────────────

function buildSuggestionStub(): HTMLElement {
    const el = document.createElement('div')
    el.className = 'trade-suggestion-stub'
    el.textContent = 'Suggested trades will appear here once the backend is connected.'
    return el
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

            hPane.append(buildHScoreStub())
            gPane.append(buildGScoreTable(sent, received))
        }

        updateResults()
        sendSel.onChange(updateResults)
        receiveSel.onChange(updateResults)

        bodyRow.append(rightCol)
        bodyArea.append(bodyRow)

        // ── Divider + Suggested trades ───────────────────────────────────────

        const divider2 = document.createElement('hr')
        divider2.className = 'trade-divider'
        bodyArea.append(divider2)

        const suggestHeader = document.createElement('div')
        suggestHeader.className = 'trade-suggest-header'
        suggestHeader.textContent = 'Suggested trades'
        bodyArea.append(suggestHeader)

        bodyArea.append(buildSuggestionStub())
    }

    rebuildBody()
    yourTeamSel.element.addEventListener('change', () => {
        const yourTeam = yourTeamSel.getValue()
        theirTeamSel.setOptions(
            fullTeams.filter(n => n !== yourTeam).map(n => ({ value: n, label: n })),
        )
        rebuildBody()
    })
    theirTeamSel.element.addEventListener('change', rebuildBody)
}
