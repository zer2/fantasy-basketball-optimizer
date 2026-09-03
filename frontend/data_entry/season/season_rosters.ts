// data_entry/season/season_rosters.ts
// Renders the season roster entry table (left) and team selector + G-score
// inspector table (right).  Used by layout.ts for Season → Rosters tab.

import { makeCustomSelect, CustomSelect } from '../../custom_select.js'
import { getPlayerResults } from '../../app_state.js'
import { getRegistryEntry, findPlayerIdByName } from '../../player_registry.js'
import { buildFullPlayerDisplayHtml, buildPlayerOptionLabel } from '../../player_display.js'
import { isMobileViewport, readRequiredIntInput } from '../../helper_functions.js'
import { DEFAULT_SEASON_ROSTERS } from './default_season_rosters.js'
import { getTeamLabel, makeTeamLabelInput } from '../team_labels.js'
import { getSelectedCategories } from '../../setting_collection/format_and_categories.js'
import { getLeagueSettings, getTeamIdentitiesFromSidebar } from '../../setting_collection/league_settings.js'
import { evaluateTeamHScore, getLivePlatformRosters } from '../../api/season_session.js'
import { makeSpacerTh } from '../../table/table_helpers.js'
import {
    GScoreRowData, getGScoreRowOrThrow, appendGScoreHeaderRow,
    appendGScoreBodyAndTotals, buildAlignedHScoreTable,
} from '../../table/gscore_table.js'

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
    const teamNames = getTeamIdentitiesFromSidebar()
    // Option order follows playerResults (G-rank descending), so the dropdowns list the
    // best players first; labels come from the registry.
    const playerOptions = playerResults.map(p => ({
        value: String(p.player_id)
      , label: buildPlayerOptionLabel(p.player_id)
      , html:  buildFullPlayerDisplayHtml(p.player_id)
    }))

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
              , [...blankOption, ...playerOptions]
              , undefined
              , true
              , rosterListenerController.signal
            )
            // Live platform rosters arrive as player ids; DEFAULT_SEASON_ROSTERS stays a
            // name-keyed file and is resolved through the registry at render time. A default
            // name missing from the current pool simply leaves the cell blank.
            if (isLivePlatform) {
                const prefillPlayerId = platformRosters?.[teamNames[d]]?.[r]
                if (prefillPlayerId !== undefined) sel.setValue(String(prefillPlayerId))
            } else {
                const defaultName = DEFAULT_SEASON_ROSTERS[teamNames[d]]?.[r]
                if (defaultName) {
                    const prefillPlayerId = resolveRosterEntryName(defaultName)
                    if (prefillPlayerId !== null) sel.setValue(String(prefillPlayerId))
                }
            }
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
            // Copy writes registry names (tab-separated), so the clipboard stays human-readable
            // and round-trips through the name-resolving paste below.
            e.preventDefault()
            const { r1, r2, c1, c2 } = selectionBounds()
            const lines: string[] = []
            for (let r = r1; r <= r2; r++) {
                const cols: string[] = []
                for (let c = c1; c <= c2; c++) {
                    const value = selects[r]?.[c]?.getValue() ?? ''
                    cols.push(value ? getRegistryEntry(Number(value)).name : '')
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
                const unmatchedNames: string[] = []
                for (let dr = 0; dr < lines.length; dr++) {
                    const r = r1 + dr
                    if (r >= nPicks) break
                    const values = lines[dr].split('\t')
                    for (let dc = 0; dc < values.length; dc++) {
                        const d = c1 + dc
                        if (d >= nDrafters) break
                        if (!selects[r]?.[d]) continue
                        const pastedName = values[dc].trim()
                        const pastedPlayerId = pastedName ? resolveRosterEntryName(pastedName) : null
                        // A name the registry doesn't know leaves the cell blank and is
                        // reported below — never silently kept or guessed at.
                        if (pastedName && pastedPlayerId === null) unmatchedNames.push(pastedName)
                        selects[r][d].setValue(pastedPlayerId === null ? '' : String(pastedPlayerId))
                        changed = true
                    }
                }

                pasteWarning.hidden = unmatchedNames.length === 0
                pasteWarning.textContent = unmatchedNames.length > 0
                    ? `Not recognized (left blank): ${unmatchedNames.join(', ')}`
                    : ''
                if (changed) rebuildInspector()
            })
        }
    }, listenerOpts)

    scroll.append(table)
    leftEl.append(scroll)

    // Paste feedback: names the registry doesn't recognize are listed here so a partial
    // paste can't silently masquerade as a complete one.
    const pasteWarning = document.createElement('div')
    pasteWarning.className = 'paste-warning'
    pasteWarning.hidden = true
    leftEl.append(pasteWarning)

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

    // Collect G-scores for players on this team
    const rows: GScoreRowData[] = []
    for (let r = 0; r < nPicks; r++) {
        const playerId = readSelectedPlayerId(selects[r][teamIdx])
        if (playerId === null) continue
        rows.push(getGScoreRowOrThrow(playerId))
    }

    if (rows.length === 0) return

    // ── Build G-score table ──────────────────────────────────────────────────

    const tbl = document.createElement('table')
    tbl.className = 'panel-table'
    tbl.style.tableLayout = 'fixed'
    tbl.dataset.testid = 'roster-inspection-gscore'

    // Spacer row to lock column widths: this panel's widths come from the
    // .panel-colspacer-* classes (and their #rosters-right mobile overrides) rather than
    // the colgroup the team-statistics panel uses — the one deliberate shell difference.
    const tHead = tbl.createTHead()
    const spacerRow = tHead.insertRow(-1)
    spacerRow.style.border = 'none'
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-name'))
    spacerRow.appendChild(makeSpacerTh('panel-colspacer-total'))
    for (let i = 0; i < categories.length; i++) spacerRow.appendChild(makeSpacerTh())

    const isMobile = isMobileViewport()
    appendGScoreHeaderRow(tHead, categories, isMobile)
    appendGScoreBodyAndTotals(tbl, rows, categories)

    container.appendChild(tbl)

    // ── H-score row (fetched from backend, only for full rosters) ────────────

    if (rows.length < nPicks) return

    const nDrafters = teamNames.length
    const playerAssignments: Record<string, number[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d]
        const players: number[] = []
        for (let r = 0; r < nPicks; r++) {
            const playerId = readSelectedPlayerId(selects[r][d])
            if (playerId !== null) players.push(playerId)
        }
        playerAssignments[team] = players
    }

    const teamName = teamNames[teamIdx]
    const result = await evaluateTeamHScore(playerAssignments, teamName)
    if (!result) return

    // This panel's H-table keeps the colgroup-derived widths but not the 100% stretch —
    // it aligns to the spacer-row-sized table above it.
    container.appendChild(buildAlignedHScoreTable(result, categories.length, isMobile,
        { testId: 'roster-inspection-hscore', fullWidth: false }))
}

/** Resolves a human-entered roster name to a player id, or null when the registry doesn't
 *  know it. Accepts both bare names and the pre-refactor "Name (POS)" form, which
 *  DEFAULT_SEASON_ROSTERS (and old clipboard copies) still carry. */
function resolveRosterEntryName(entryName: string): number | null {
    const bareName = entryName.replace(/\s*\([A-Z,\-]+\)\s*$/, '')
    return findPlayerIdByName(bareName)
}

/** The player id a roster select holds, or null when the cell is blank. Throws on a
 *  non-numeric value — the selects only ever carry stringified ids. */
function readSelectedPlayerId(select: CustomSelect): number | null {
    const value = select.getValue()
    if (!value) return null
    const playerId = Number(value)
    if (Number.isNaN(playerId)) throw new Error(`Roster select carried a non-numeric value: "${value}"`)
    return playerId
}
