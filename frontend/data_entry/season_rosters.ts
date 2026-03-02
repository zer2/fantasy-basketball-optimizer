// data_entry/season_rosters.ts
// Renders the season roster entry table (left) and team selector / stub (right).
// Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../custom_select.js'
import { getPlayers } from '../script.js'

export function renderSeasonRosters(leftEl: HTMLElement, rightEl: HTMLElement): void {
    const nDrafters = parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value) || 12
    const nPicks    = parseInt((document.getElementById('ls-n-picks')    as HTMLInputElement).value) || 13
    const teamNames = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    const playerNames = getPlayers().map(p => p.name)

    leftEl.innerHTML  = ''
    rightEl.innerHTML = ''

    // ── Left: roster entry table ────────────────────────────────────────────

    const scroll = document.createElement('div')
    scroll.className = 'entry-table-scroll'

    const table = document.createElement('table')
    table.className = 'entry-table'

    // Header: Pick | Team1 | Team2 | …
    const thead = table.createTHead()
    const hrow  = thead.insertRow()
    const pickTh = document.createElement('th')
    pickTh.textContent = 'Pick'
    pickTh.style.width = '48px'
    hrow.append(pickTh)
    for (const name of teamNames) {
        const th = document.createElement('th')
        th.textContent = name
        hrow.append(th)
    }

    // Data rows — one row per pick, one column per team
    const selects: CustomSelect[][] = []   // [row][col]
    const blankOption = [{ value: '', label: '' }]
    const tbody = table.createTBody()

    for (let r = 0; r < nPicks; r++) {
        const row  = tbody.insertRow()
        const rowSelects: CustomSelect[] = []

        const pickCell = row.insertCell()
        pickCell.className   = 'entry-cell-label'
        pickCell.textContent = String(r + 1)

        for (let d = 0; d < nDrafters; d++) {
            const cell = row.insertCell()
            const sel  = makeCustomSelect(
                `sr-player-${r}-${d}`,
                [...blankOption, ...playerNames.map(n => ({ value: n, label: n }))],
            )
            sel.element.style.fontSize = '0.75rem'
            cell.append(sel.element)
            rowSelects.push(sel)
        }
        selects.push(rowSelects)
    }

    scroll.append(table)
    leftEl.append(scroll)

    // Lock-in button
    const lockBtn = document.createElement('button')
    lockBtn.className   = 'lock-in-btn'
    lockBtn.textContent = 'Lock in'
    leftEl.append(lockBtn)

    // ── Right: team selector + stub ─────────────────────────────────────────

    const teamSel = makeCustomSelect(
        'sr-team-select',
        teamNames.map(n => ({ value: n, label: n })),
    )
    teamSel.element.style.marginBottom = '10px'
    rightEl.append(teamSel.element)

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightEl.append(stub)
}
