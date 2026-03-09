// data_entry/season/season_rosters.ts
// Renders the season roster entry table (left) and team selector + G-score
// inspector table (right).  Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../../custom_select.js'
import { getPlayers, getGScoreByName, getCategories } from '../../app_state.js'
import { stat_styler_primary } from '../../styles/styler_functions.js'

/** Renders the season roster entry grid (left) and team inspector with G-score table (right). */
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

    // ── Copy support: copy the full grid as tab/newline-separated text ───
    table.addEventListener('copy', (e: ClipboardEvent) => {
        // Only intercept when focus is inside the table body
        const active = document.activeElement
        if (!active || !table.contains(active)) return

        const lines: string[] = []
        for (let r = 0; r < nPicks; r++) {
            const cols: string[] = []
            for (let d = 0; d < nDrafters; d++) {
                cols.push(selects[r]?.[d]?.getValue() ?? '')
            }
            lines.push(cols.join('\t'))
        }

        e.preventDefault()
        e.clipboardData?.setData('text/plain', lines.join('\n'))
    })

    // ── Paste support: paste tab/newline-separated data into the grid ────
    table.addEventListener('paste', (e: ClipboardEvent) => {
        const text = e.clipboardData?.getData('text/plain')
        if (!text) return

        // Find which cell is focused
        const active = document.activeElement
        if (!active) return
        const cell = active.closest('td')
        if (!cell) return
        const row = cell.parentElement as HTMLTableRowElement
        if (!row) return

        const startRow = row.rowIndex - 1  // subtract 1 for thead row
        const startCol = cell.cellIndex - 1 // subtract 1 for pick label column
        if (startRow < 0 || startCol < 0) return

        const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trimEnd().split('\n')

        let changed = false
        for (let dr = 0; dr < lines.length; dr++) {
            const r = startRow + dr
            if (r >= nPicks) break
            const values = lines[dr].split('\t')
            for (let dc = 0; dc < values.length; dc++) {
                const d = startCol + dc
                if (d >= nDrafters) break
                const val = values[dc].trim()
                if (val && selects[r]?.[d]) {
                    selects[r][d].setValue(val)
                    changed = true
                }
            }
        }

        if (changed) {
            e.preventDefault()
            rebuildInspector()
        }
    })

    scroll.append(table)
    leftEl.append(scroll)

    // ── Right: team selector + G-score table ─────────────────────────────────

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

    // Container for the G-score table — rebuilt when team or roster changes
    const tableContainer = document.createElement('div')
    rightEl.append(tableContainer)

    function rebuildInspector(): void {
        const selectedTeam = teamSel.getValue()
        const teamIdx = teamNames.indexOf(selectedTeam)
        if (teamIdx < 0) { tableContainer.innerHTML = ''; return }
        buildTeamGScoreTable(teamIdx, selects, nPicks, tableContainer)
    }

    // Rebuild when the team selector changes
    teamSel.element.addEventListener('change', rebuildInspector)

    // Rebuild when any roster select changes
    for (const rowSelects of selects) {
        for (const sel of rowSelects) {
            sel.element.addEventListener('change', rebuildInspector)
        }
    }

    // Initial render
    rebuildInspector()
}

// ─── G-score team inspector table ────────────────────────────────────────────

/**
 * Builds a G-score table for the selected team: one row per rostered player
 * plus a totals row.  Styled to match the expanded-view G-score tables.
 */
function buildTeamGScoreTable(
    teamIdx: number,
    selects: CustomSelect[][],
    nPicks: number,
    container: HTMLElement,
): void {
    container.innerHTML = ''
    const categories = getCategories()
    const gScoreMap  = getGScoreByName()

    // Collect G-scores for players on this team
    const rows: { name: string; values: number[]; total: number }[] = []
    for (let r = 0; r < nPicks; r++) {
        const name = selects[r][teamIdx].getValue()
        if (!name) continue
        const gs = gScoreMap.get(name)
        if (!gs) continue
        rows.push({ name: gs.name, values: gs.values, total: gs.total })
    }

    if (rows.length === 0) return

    // ── Build table ──────────────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table'
    tbl.style.tableLayout = 'fixed'

    // Spacer row to lock column widths
    const tHead = tbl.createTHead()
    const spacerRow = tHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh())

    // Header row
    const headerRow = tHead.insertRow(-1)
    headerRow.appendChild(makeSpacerTh())  // invisible label spacer
    const totalTh = document.createElement('th')
    totalTh.className = 'panel-colheader'
    totalTh.textContent = 'Total'
    headerRow.appendChild(totalTh)
    for (const cat of categories) {
        const th = document.createElement('th')
        th.className = 'panel-colheader'
        th.textContent = cat
        headerRow.appendChild(th)
    }

    // Data rows — one per player
    const tBody = tbl.createTBody()
    const teamTotals = new Array(categories.length).fill(0)
    let teamTotalSum = 0

    for (const row of rows) {
        const tr = tBody.insertRow(-1)

        const labelCell = document.createElement('th')
        labelCell.className = 'panel-rowlabel'
        labelCell.textContent = row.name
        tr.appendChild(labelCell)

        const totalCell = tr.insertCell(-1)
        totalCell.textContent = row.total.toFixed(2)
        totalCell.className = 'panel-datacell celltypea'
        teamTotalSum += row.total

        for (let i = 0; i < row.values.length; i++) {
            const cell = tr.insertCell(-1)
            cell.textContent = row.values[i].toFixed(2)
            cell.style.cssText = stat_styler_primary(row.values[i], 60, 0)
            cell.className = 'panel-datacell'
            teamTotals[i] += row.values[i]
        }
    }

    // Totals row
    const totalsRow = tBody.insertRow(-1)
    const totalsLabel = document.createElement('th')
    totalsLabel.className = 'panel-rowlabel'
    totalsLabel.textContent = 'Team Total'
    totalsRow.appendChild(totalsLabel)

    const totalsCell = totalsRow.insertCell(-1)
    totalsCell.textContent = teamTotalSum.toFixed(2)
    totalsCell.className = 'panel-datacell celltypeb'

    for (const val of teamTotals) {
        const cell = totalsRow.insertCell(-1)
        cell.textContent = val.toFixed(2)
        cell.style.cssText = stat_styler_primary(val, 60, 0)
        cell.className = 'panel-datacell'
    }

    container.appendChild(tbl)
}

/** Creates an invisible `<th>` spacer to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const th = document.createElement('th')
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return th
}
