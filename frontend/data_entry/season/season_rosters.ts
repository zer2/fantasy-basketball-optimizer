// data_entry/season/season_rosters.ts
// Renders the season roster entry table (left) and team selector + G-score
// inspector table (right).  Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../../custom_select.js'
import { getPlayerResults, getGScoreByName, getShortCategoryNames } from '../../app_state.js'
import { isMobileViewport, readRequiredIntInput } from '../../helper_functions.js'
import { DEFAULT_SEASON_ROSTERS } from './default_season_rosters.js'
import { getTeamLabel, makeTeamLabelInput } from '../team_labels.js'
import { getSelectedCategories, getScoringFormat } from '../../parameter_collection/format_and_categories.js'
import { getLeagueSettings } from '../../parameter_collection/league_settings.js'
import { stat_styler_primary } from '../../styles/styler_functions.js'
import { evaluateTeamHScore, getLivePlatformRosters } from '../../api/season_session.js'

// Tracks the change-event listeners attached by the most recent render so they
// can be removed before the next one. Without this, calling renderSeasonRosters
// a second time (e.g. revisiting Season Mode) leaves listener closures bound to
// the now-detached previous selects. Each closure captures `selects` (156 custom
// selects × ~300 options) and `rebuildInspector`, so the entire previous render
// tree stays alive in memory. Aborting this controller before the new render
// detaches all listeners at once — no manual iteration, no missed cleanup.
let rosterListenerController: AbortController | null = null

/** Renders the season roster entry grid (left) and team inspector with G-score table (right). */
export function renderSeasonRosters(leftEl: HTMLElement, rightEl: HTMLElement): void {

    // Remove change listeners from any previously-rendered selects so their
    // closures can be garbage-collected. See comment on rosterListenerController.
    rosterListenerController?.abort()
    rosterListenerController = new AbortController()
    const listenerOpts = { signal: rosterListenerController.signal }
    const playerResults = getPlayerResults()
    if (playerResults === null) return

    const nDrafters = readRequiredIntInput('ls-n-drafters')
    const nPicks    = readRequiredIntInput('ls-n-picks')
    const teamNames = (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
    const playerNames = playerResults.map(p => p.name)

    // Live platform → prefill the grid from the platform's rosters (blank until the
    // async poll populates the cache); own data → the hardcoded defaults.
    const isLivePlatform = getLeagueSettings().platform !== 'Enter your own data'
    const platformRosters = isLivePlatform ? getLivePlatformRosters() : null

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
    teamNames.forEach((name, d) => {
        const th = document.createElement('th')
        // Own data → editable display label ("Team N" by default); live platform → the real
        // team name (identity), shown read-only since it comes from the platform.
        if (isLivePlatform) {
            th.textContent = name
        } else {
            th.className = 'team-header-cell'
            const headerWrap = document.createElement('div')
            headerWrap.className = 'team-header'
            headerWrap.append(makeTeamLabelInput(d, rosterListenerController!.signal))
            th.append(headerWrap)
        }
        hrow.append(th)
    })

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
                `sr-player-${r}-${d}`
              , [...blankOption, ...playerNames.map(n => ({ value: n, label: n }))]
              , undefined
              , true
              , rosterListenerController.signal
            )
            const prefill = isLivePlatform
                ? platformRosters?.[teamNames[d]]?.[r]
                : DEFAULT_SEASON_ROSTERS[teamNames[d]]?.[r]
            if (prefill) sel.setValue(prefill)
            cell.append(sel.element)
            rowSelects.push(sel)
        }
        selects.push(rowSelects)
    }

    // ── Excel-style cell selection + Copy/Paste ────────────────────────────────

    // Selection range (grid coordinates, not DOM indices)
    let anchorRow = 0, anchorCol = 0
    let focusRow  = 0, focusCol  = 0

    function selectionBounds() {
        return {
            r1: Math.min(anchorRow, focusRow)
          , r2: Math.max(anchorRow, focusRow)
          , c1: Math.min(anchorCol, focusCol)
          , c2: Math.max(anchorCol, focusCol)
        }
    }

    function clearSelectionHighlight(): void {
        table.querySelectorAll('.entry-cell-selected').forEach(el =>
            el.classList.remove('entry-cell-selected')
        )
    }

    function applySelectionHighlight(): void {
        clearSelectionHighlight()
        const { r1, r2, c1, c2 } = selectionBounds()
        for (let r = r1; r <= r2; r++) {
            for (let c = c1; c <= c2; c++) {
                // tbody row r, cell c+1 (cell 0 is the pick label)
                const cell = tbody.rows[r]?.cells[c + 1]
                if (cell) cell.classList.add('entry-cell-selected')
            }
        }
    }

    /** Resolves a click target to grid coordinates, or null if outside data cells. */
    function cellCoordsFromEvent(e: MouseEvent): { row: number; col: number } | null {
        const td = (e.target as HTMLElement).closest('td') as HTMLTableCellElement | null
        if (!td || !tbody.contains(td)) return null
        const tr = td.parentElement as HTMLTableRowElement
        const col = td.cellIndex - 1   // subtract pick label column
        const row = tr.rowIndex - 1     // subtract thead row
        if (row < 0 || col < 0 || row >= nPicks || col >= nDrafters) return null
        return { row, col }
    }

    let dragging = false

    table.addEventListener('mousedown', (e: MouseEvent) => {
        const coords = cellCoordsFromEvent(e)
        if (!coords) return
        if (e.shiftKey) {
            e.preventDefault()
            focusRow = coords.row
            focusCol = coords.col
        } else {
            anchorRow = coords.row
            anchorCol = coords.col
            focusRow  = coords.row
            focusCol  = coords.col
            dragging  = true
        }
        applySelectionHighlight()
    }, listenerOpts)

    table.addEventListener('mousemove', (e: MouseEvent) => {
        if (!dragging) return
        const coords = cellCoordsFromEvent(e)
        if (!coords) return
        if (coords.row !== focusRow || coords.col !== focusCol) {
            focusRow = coords.row
            focusCol = coords.col
            applySelectionHighlight()
        }
    }, listenerOpts)

    document.addEventListener('mouseup', () => { dragging = false }, listenerOpts)

    table.addEventListener('keydown', (e: KeyboardEvent) => {
        const ctrl = e.ctrlKey || e.metaKey
        if (!ctrl) return

        if (e.key === 'a') {
            // Select all data cells
            e.preventDefault()
            anchorRow = 0; anchorCol = 0
            focusRow = nPicks - 1; focusCol = nDrafters - 1
            applySelectionHighlight()
            return
        }

        if (e.key === 'c') {
            e.preventDefault()
            const { r1, r2, c1, c2 } = selectionBounds()
            const lines: string[] = []
            for (let r = r1; r <= r2; r++) {
                const cols: string[] = []
                for (let c = c1; c <= c2; c++) {
                    cols.push(selects[r]?.[c]?.getValue() ?? '')
                }
                lines.push(cols.join('\t'))
            }
            navigator.clipboard.writeText(lines.join('\n'))

        } else if (e.key === 'v') {
            e.preventDefault()
            const { r1, c1 } = selectionBounds()

            navigator.clipboard.readText().then(text => {
                if (!text) return
                const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').trimEnd().split('\n')

                let changed = false
                for (let dr = 0; dr < lines.length; dr++) {
                    const r = r1 + dr
                    if (r >= nPicks) break
                    const values = lines[dr].split('\t')
                    for (let dc = 0; dc < values.length; dc++) {
                        const d = c1 + dc
                        if (d >= nDrafters) break
                        const val = values[dc].trim()
                        if (selects[r]?.[d]) {
                            selects[r][d].setValue(val)
                            changed = true
                        }
                    }
                }

                if (changed) rebuildInspector()
            })
        }
    }, listenerOpts)

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
        teamNames.map((name, index) => ({ value: name, label: isLivePlatform ? name : getTeamLabel(index) })),
        undefined,
        undefined,
        rosterListenerController.signal,
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
        buildTeamGScoreTable(teamIdx, selects, nPicks, teamNames, tableContainer)
    }

    // Rebuild when the team selector changes
    teamSel.element.addEventListener('change', rebuildInspector, listenerOpts)

    // Rebuild when any roster select changes
    for (const rowSelects of selects) {
        for (const sel of rowSelects) {
            sel.element.addEventListener('change', rebuildInspector, listenerOpts)
        }
    }

    // Initial render
    rebuildInspector()
}

// ─── G-score team inspector table ────────────────────────────────────────────

/**
 * Builds a G-score table for the selected team: one row per rostered player
 * plus a totals row, followed by an H-score row fetched from the backend.
 * Styled to match the expanded-view G-score tables.
 */
async function buildTeamGScoreTable(
    teamIdx: number
    , selects: CustomSelect[][]
    , nPicks: number
    , teamNames: string[]
    , container: HTMLElement
): Promise<void> {
    container.innerHTML = ''
    const categories = getSelectedCategories()
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

    // ── Build G-score table ──────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table'
    tbl.style.tableLayout = 'fixed'
    tbl.dataset.testid = 'roster-inspection-gscore'

    // Spacer row to lock column widths
    const tHead = tbl.createTHead()
    const spacerRow = tHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh())

    // Header row — on mobile, swap full category names for their short form
    // (e.g. "Points" → "Pts") to keep columns narrow.
    const isMobile   = isMobileViewport()
    const shortNames = isMobile ? getShortCategoryNames() : {}
    const headerRow = tHead.insertRow(-1)
    headerRow.appendChild(makeSpacerTh())  // invisible label spacer
    const totalTh = document.createElement('th')
    totalTh.className = 'panel-colheader'
    totalTh.textContent = 'Total'
    headerRow.appendChild(totalTh)
    for (const cat of categories) {
        const label = shortNames[cat] ?? cat
        const th = document.createElement('th')
        th.className = label.length >= 10 ? 'panel-colheader colheader-long' : 'panel-colheader'
        th.textContent = label
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

    // ── H-score row (fetched from backend, only for full rosters) ────────────

    if (rows.length < nPicks) return

    const nDrafters = teamNames.length
    const playerAssignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d]
        const players: string[] = []
        for (let r = 0; r < nPicks; r++) {
            const name = selects[r][d].getValue()
            if (name) players.push(name)
        }
        playerAssignments[team] = players
    }

    const teamName = teamNames[teamIdx]
    const result = await evaluateTeamHScore(playerAssignments, teamName)
    if (!result) return

    const isRoto     = getScoringFormat() === 'Rotisserie'
    const rotoNDrafters = isRoto ? getLeagueSettings().n_drafters : 0
    const rotoMiddle = (rotoNDrafters - 1) / 2 + 1

    const hScoreTbl = document.createElement('table')
    hScoreTbl.className = 'panel-table panel-table--rounded panel-table--top-gap'
    hScoreTbl.style.tableLayout = 'fixed'
    hScoreTbl.dataset.testid = 'roster-inspection-hscore'

    // Match the team-inspector's column widths above so the two tables line up.
    // Mobile values mirror the #rosters-right .panel-colspacer-* overrides.
    const colgroup = document.createElement('colgroup')
    const nameCol  = document.createElement('col')
    nameCol.style.width = isMobile ? '7rem' : '200px'
    const totalCol = document.createElement('col')
    totalCol.style.width = isMobile ? '3rem' : '83px'
    colgroup.append(nameCol, totalCol)
    for (let i = 0; i < categories.length; i++) colgroup.appendChild(document.createElement('col'))
    hScoreTbl.appendChild(colgroup)

    const hScoreTBody = hScoreTbl.createTBody()
    const hScoreRow = hScoreTBody.insertRow(-1)

    const hScoreLabel = document.createElement('th')
    hScoreLabel.className = 'panel-rowlabel'
    hScoreLabel.textContent = 'H-Score (est. win rate)'
    hScoreRow.appendChild(hScoreLabel)

    const hScoreTotalCell = hScoreRow.insertCell(-1)
    hScoreTotalCell.textContent = result.h_score.toFixed(1)
    hScoreTotalCell.className = 'overallhscore'

    for (const winRate of result.win_rates) {
        const cell = hScoreRow.insertCell(-1)
        if (isRoto) {
            const rotoValue = 1 + (winRate / 100) * (rotoNDrafters - 1)
            cell.textContent = rotoValue.toFixed(1)
            cell.className = 'categoricalRotoHscore'
            cell.style.cssText = stat_styler_primary(rotoValue, 3 * (rotoNDrafters - 1), rotoMiddle)
        } else {
            cell.textContent = winRate.toFixed(1)
            cell.className = 'categoricalhscore'
            cell.style.cssText = stat_styler_primary(winRate, 3, 50)
        }
    }

    container.appendChild(hScoreTbl)
}

/** Creates an invisible `<th>` spacer to lock column widths in panel tables. */
function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const th = document.createElement('th')
    th.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return th
}
