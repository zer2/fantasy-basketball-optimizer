// data_entry/season/season_rosters.ts
// Renders the season roster entry table (left) and team selector / stub (right).
// Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../../custom_select.js'
import { getPlayers } from '../../app_state.js'

/** Renders the season roster entry grid (left) and team inspector selector with stub (right). */
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

    // Sort players by G-score rank and snake-draft to pre-fill the table
    const sorted = [...getPlayers()].sort((a, b) => a.g_rank - b.g_rank)
    const totalSlots = nDrafters * nPicks
    const snakeDraft: string[][] = Array.from({ length: nDrafters }, () => [])
    for (let i = 0; i < Math.min(sorted.length, totalSlots); i++) {
        const round = Math.floor(i / nDrafters)
        const pos   = i % nDrafters
        const team  = round % 2 === 0 ? pos : nDrafters - 1 - pos
        snakeDraft[team].push(sorted[i].name)
    }

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
            // Pre-fill from snake draft if a player is available for this slot
            const prefill = snakeDraft[d]?.[r]
            if (prefill) sel.setValue(prefill)
            cell.append(sel.element)
            rowSelects.push(sel)
        }
        selects.push(rowSelects)
    }

    scroll.append(table)
    leftEl.append(scroll)

    // ── Right: team selector + stub ─────────────────────────────────────────

    const wrap = document.createElement('div')
    wrap.className = 'seat-selector-wrap'

    const label = document.createElement('div')
    label.className   = 'pick-control-label'
    label.textContent = 'Which team do you want to inspect?'
    wrap.append(label)

    const teamSel = makeCustomSelect(
        'sr-team-select',
        teamNames.map(n => ({ value: n, label: n })),
    )
    wrap.append(teamSel.element)
    rightEl.append(wrap)

    const stub = document.createElement('div')
    stub.className   = 'team-display-stub'
    stub.textContent = 'Team statistics will appear here once the backend is connected.'
    rightEl.append(stub)
}
