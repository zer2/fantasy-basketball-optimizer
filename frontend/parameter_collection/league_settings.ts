// Collects: sport, platform, mode, n_drafters, n_picks,
//           cash_per_team (Auction Mode only), third_round_reversal, team_names, my_team_id
// Mirrors league_settings_popover() in src/parameter_collection/league_settings.py

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, makeNumberInput, makeSidebarToggle, readRequiredIntInput } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'
import { fetchDivisions, connectPlatform } from '../api/client.js'

const DRAFT_MODE_OPTIONS = ['Draft Mode', 'Auction Mode', 'Season Mode'] as const
export type DraftMode = typeof DRAFT_MODE_OPTIONS[number]

const PLATFORM_OPTIONS = [
    'Enter your own data',
    'Retrieve from Yahoo',
    'Retrieve from Fantrax',
    'Retrieve from ESPN',
] as const
export type Platform = typeof PLATFORM_OPTIONS[number]

const DRAFTER_METHOD_OPTIONS = ['Manual input', 'H-scoring', 'G-scoring'] as const
export type DrafterMethod = typeof DRAFTER_METHOD_OPTIONS[number]

/**
 * Renders the League Settings section into `container`.
 * Layout: 2-column grid for compact items, followed by full-width items.
 */
export function renderLeagueSettings(container: HTMLElement): void {
    const config = getSportConfig()
    if (!config) throw new Error('renderLeagueSettings called before sport config loaded')
    const nDraftersDefault = config.options.n_drafters.default
    const nPicksDefault    = config.options.n_picks.default

    const grid = document.createElement('div')
    grid.className = 'ls-grid'
    container.append(grid)

    // ── Sport (left col) ──────────────────────────────────────────────────
    // Only NBA is supported now; structured as a selector for future expansion.
    const sportCell = makeCell()
    sportCell.append(makeLabel('ls-sport', 'Sport'))
    const sportSelect = makeCustomSelect(
        'ls-sport',
        [{ value: 'NBA', label: 'NBA' }],
        pref('sport', 'NBA'),
    )
    sportSelect.element.addEventListener('change', () => savePref('sport', sportSelect.getValue()))
    sportCell.append(sportSelect.element)
    grid.append(sportCell)

    // ── Mode (right col) ──────────────────────────────────────────────────
    const modeCell = makeCell()
    modeCell.append(makeLabel('ls-mode', 'Mode'))
    const modeSelect = makeCustomSelect(
        'ls-mode',
        DRAFT_MODE_OPTIONS.map(m => ({ value: m, label: m })),
        pref('mode', 'Draft Mode'),
    )
    modeSelect.element.addEventListener('change', () => savePref('mode', modeSelect.getValue()))
    modeCell.append(modeSelect.element)
    grid.append(modeCell)

    // ── Platform (full-width) ─────────────────────────────────────────────
    // Controls whether league data is entered manually or pulled from a platform.
    const platformCell = makeCell('ls-cell-full')
    platformCell.append(makeLabel('ls-platform', 'Fantasy Platform'))
    const platformSelect = makeCustomSelect(
        'ls-platform',
        PLATFORM_OPTIONS.map(p => ({ value: p, label: p })),
        pref('platform', 'Enter your own data'),
    )
    platformSelect.element.addEventListener('change', () => savePref('platform', platformSelect.getValue()))
    platformCell.append(platformSelect.element)
    grid.append(platformCell)

    // ── Number of drafters (left col) ─────────────────────────────────────
    const draftersCell = makeCell()
    draftersCell.append(makeLabel('ls-n-drafters', 'Drafters'))
    const nDraftersInput = makeNumberInput('ls-n-drafters', pref('n_drafters', nDraftersDefault), 2)
    nDraftersInput.addEventListener('change', () => savePref('n_drafters', parseInt(nDraftersInput.value)))
    draftersCell.append(nDraftersInput)
    grid.append(draftersCell)

    // ── Picks per drafter (right col) ─────────────────────────────────────
    const picksCell = makeCell()
    picksCell.append(makeLabel('ls-n-picks', 'Picks / drafter'))
    const nPicksInput = makeNumberInput('ls-n-picks', pref('n_picks', nPicksDefault), 1)
    nPicksInput.addEventListener('change', () => savePref('n_picks', parseInt(nPicksInput.value)))
    picksCell.append(nPicksInput)
    grid.append(picksCell)

    // ── Budget per team (left col, Auction Mode only) ─────────────────────
    const cashCell = makeCell()
    cashCell.style.display = 'none'
    cashCell.append(makeLabel('ls-cash-per-team', 'Budget / team ($)'))
    const cashInput = makeNumberInput('ls-cash-per-team', pref('cash_per_team', 200), 1)
    cashInput.addEventListener('change', () => savePref('cash_per_team', parseInt(cashInput.value)))
    cashCell.append(cashInput)
    grid.append(cashCell)

    // ── Third round reversal toggle (full-width, Draft Mode only) ─────────
    const trrToggle = makeSidebarToggle('ls-third-round-reversal', 'Third round reversal')
    trrToggle.id = 'ls-trr-row'

    const trrCheckbox = trrToggle.querySelector('input') as HTMLInputElement
    trrCheckbox.checked = pref('third_round_reversal', false)
    trrCheckbox.addEventListener('change', () => savePref('third_round_reversal', trrCheckbox.checked))

    // ── Team names list + per-drafter mode dropdowns (own-data only) ──────
    // A hidden textarea keeps #ls-team-names in the DOM so all existing
    // readers (draft_board, season modules, main.ts) continue to work.
    // The visible UI is a scrollable list of rows: [editable name] [mode ▼]
    const teamNamesWrap = document.createElement('div')
    teamNamesWrap.append(makeLabel('ls-team-names', 'Teams'))

    const hiddenNamesTextarea = document.createElement('textarea')
    hiddenNamesTextarea.id    = 'ls-team-names'
    hiddenNamesTextarea.style.display = 'none'
    hiddenNamesTextarea.value = pref('team_names', defaultTeamNames(pref('n_drafters', nDraftersDefault)))
    teamNamesWrap.append(hiddenNamesTextarea)

    const teamNamesList = document.createElement('div')
    teamNamesList.className = 'team-names-list'
    teamNamesWrap.append(teamNamesList)
    container.append(teamNamesWrap)
    container.append(trrToggle)

    /** Sync visible inputs → hidden textarea, persist, and notify main.ts. */
    function syncTeamNames(): void {
        const nDrafters = parseInt(nDraftersInput.value)
        const names = Array.from({ length: nDrafters }, (_, i) => {
            const el = document.getElementById(`ls-team-name-${i}`) as HTMLInputElement | null
            if (!el) throw new Error(`Team name input ls-team-name-${i} not found`)
            return el.value
        })
        hiddenNamesTextarea.value = names.join('\n')
        savePref('team_names', hiddenNamesTextarea.value)
        hiddenNamesTextarea.dispatchEvent(new Event('input', { bubbles: true }))
    }

    /** Rebuild the visible rows. Pass resetToDefaults=true when n_drafters changes. */
    function rebuildTeamNameRows(resetToDefaults: boolean): void {
        const nDrafters = parseInt(nDraftersInput.value)

        const names: string[] = []
        if (resetToDefaults) {
            for (let i = 0; i < nDrafters; i++) names.push(`Drafter ${i + 1}`)
        } else {
            const saved = hiddenNamesTextarea.value.split('\n').map(s => s.trim())
            for (let i = 0; i < nDrafters; i++) {
                const existing = document.getElementById(`ls-team-name-${i}`) as HTMLInputElement | null
                names.push(existing?.value.trim() || saved[i] || `Drafter ${i + 1}`)
            }
        }

        teamNamesList.innerHTML = ''

        for (let i = 0; i < nDrafters; i++) {
            const row = document.createElement('div')
            row.className = 'team-name-row'

            const nameInput = document.createElement('input')
            nameInput.type      = 'text'
            nameInput.id        = `ls-team-name-${i}`
            nameInput.className = 'team-name-input'
            nameInput.value     = names[i]
            nameInput.addEventListener('input', syncTeamNames)
            row.append(nameInput)

            const initialMode = resetToDefaults ? 'Manual input' : pref(`drafter_mode_${i}`, 'Manual input') as DrafterMethod
            const drafterModeSelect = makeCustomSelect(
                `ls-drafter-mode-${i}`
              , DRAFTER_METHOD_OPTIONS.map(m => ({ value: m, label: m }))
              , initialMode
            )
            drafterModeSelect.element.classList.add('drafter-mode-cell')
            drafterModeSelect.element.addEventListener('change', () => savePref(`drafter_mode_${i}`, drafterModeSelect.getValue()))
            row.append(drafterModeSelect.element)

            teamNamesList.append(row)
        }

        hiddenNamesTextarea.value = names.join('\n')
        savePref('team_names', hiddenNamesTextarea.value)
    }

    rebuildTeamNameRows(false)

    nDraftersInput.addEventListener('change', () => {
        const nDrafters = parseInt(nDraftersInput.value)
        if (!isNaN(nDrafters) && nDrafters > 0) rebuildTeamNameRows(true)
    })

    // ── Live-platform connect (shown only when a live platform is selected) ───
    const connectCell = makeCell('ls-cell-full')
    connectCell.id = 'ls-connect-cell'

    connectCell.append(makeLabel('ls-league-id', 'League ID'))
    const leagueIdInput = document.createElement('input')
    leagueIdInput.type      = 'text'
    leagueIdInput.id        = 'ls-league-id'
    leagueIdInput.className = 'team-name-input'
    leagueIdInput.value     = pref('platform_league_id', '')
    connectCell.append(leagueIdInput)

    const divisionWrap = document.createElement('div')
    divisionWrap.id = 'ls-division-wrap'
    divisionWrap.style.display = 'none'
    divisionWrap.append(makeLabel('ls-division', 'Division'))
    const divisionSelect = makeCustomSelect('ls-division', [{ value: '', label: '(none)' }])
    divisionWrap.append(divisionSelect.element)
    connectCell.append(divisionWrap)

    const connectButton = document.createElement('button')
    connectButton.type        = 'button'
    connectButton.id          = 'ls-connect-btn'
    connectButton.className   = 'section-apply-btn'
    connectButton.textContent = 'Connect'
    connectCell.append(connectButton)

    const connectStatus = document.createElement('div')
    connectStatus.id        = 'ls-connect-status'
    connectStatus.className  = 'pick-control-label'
    connectCell.append(connectStatus)

    container.append(connectCell)

    /** Loads the league's divisions into the division select (hidden when none). */
    async function loadDivisions(): Promise<void> {
        const platform = platformSelect.getValue()
        const leagueId = leagueIdInput.value.trim()
        if (platform === 'Enter your own data' || !leagueId) return
        const divisions = await fetchDivisions(platform, leagueId)
        if (divisions.length === 0) {
            divisionWrap.style.display = 'none'
            divisionSelect.setOptions([{ value: '', label: '(none)' }])
        } else {
            divisionWrap.style.display = ''
            divisionSelect.setOptions(divisions.map(division => ({ value: division.id, label: division.name })))
        }
    }

    leagueIdInput.addEventListener('change', () => {
        savePref('platform_league_id', leagueIdInput.value)
        loadDivisions().catch(err => { connectStatus.textContent = `Could not load divisions: ${err.message}` })
    })

    connectButton.addEventListener('click', () => {
        const platform = platformSelect.getValue()
        const leagueId = leagueIdInput.value.trim()
        if (platform === 'Enter your own data' || !leagueId) {
            connectStatus.textContent = 'Enter a league ID first.'
            return
        }
        const divisionId = divisionSelect.getValue() || null
        connectStatus.textContent = 'Connecting...'
        connectPlatform(platform, leagueId, divisionId)
            .then(resp => {
                // Restrict the mode selector to what this platform supports.
                modeSelect.setOptions(resp.available_modes.map(m => ({ value: m, label: m })))
                if (!resp.available_modes.includes(modeSelect.getValue())) {
                    modeSelect.setValue(resp.available_modes[0])
                }
                // Set n_drafters / n_picks directly — NOT via a change event, which
                // would reset the team-name rows — and drive the seat selector via
                // the hidden textarea's input event (main.ts repopulates it).
                nDraftersInput.value = String(resp.n_drafters)
                nPicksInput.value    = String(resp.n_picks)
                hiddenNamesTextarea.value = resp.team_names.join('\n')
                hiddenNamesTextarea.dispatchEvent(new Event('input', { bubbles: true }))
                connectStatus.textContent =
                    `Connected — ${resp.team_names.length} teams, ${resp.n_picks} picks. Click Refresh Analysis.`
            })
            .catch(err => { connectStatus.textContent = `Connect failed: ${err.message}` })
    })

    // ── Own-data-dependent and mode-dependent visibility ──────────────────
    function updateVisibility(): void {
        const isOwnData = platformSelect.getValue() === 'Enter your own data'
        const mode      = modeSelect.getValue()
        cashCell.style.display       = mode === 'Auction Mode' ? '' : 'none'
        trrToggle.style.display      = isOwnData && mode === 'Draft Mode' ? '' : 'none'
        teamNamesWrap.style.display  = isOwnData ? '' : 'none'
        connectCell.style.display    = isOwnData ? 'none' : ''
        if (!isOwnData || mode !== 'Draft Mode') trrCheckbox.checked = false

        // Show drafter mode dropdowns only in Draft Mode + own data
        const showModes = isOwnData && mode === 'Draft Mode'
        teamNamesList.querySelectorAll<HTMLElement>('.drafter-mode-cell').forEach(el => {
            el.style.display = showModes ? '' : 'none'
        })
    }

    updateVisibility()
    modeSelect.element.addEventListener('change', updateVisibility)
    platformSelect.element.addEventListener('change', updateVisibility)
}

export function getTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

export function getDrafterMethodByIndex(index: number): DrafterMethod {
    const el = document.getElementById(`ls-drafter-mode-${index}`) as HTMLInputElement | null
    if (!el) throw new Error(`Drafter method select ls-drafter-mode-${index} not found`)
    return el.value as DrafterMethod
}

export function getDrafterMethods(): DrafterMethod[] {
    const modes: DrafterMethod[] = []
    let i = 0
    while (document.getElementById(`ls-drafter-mode-${i}`)) {
        modes.push(getDrafterMethodByIndex(i))
        i++
    }
    return modes
}

/**
 * Reads all League Settings values from the DOM and returns them as a plain object.
 */
export function getLeagueSettings(): {
    sport: string
    platform: Platform
    mode: DraftMode
    n_drafters: number
    n_picks: number
    cash_per_team: number
    third_round_reversal: boolean
    team_names: string[]
} {
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value as DraftMode
    return {
        sport:                (document.getElementById('ls-sport') as HTMLInputElement).value,
        platform:             (document.getElementById('ls-platform') as HTMLInputElement).value as Platform,
        mode,
        n_drafters:           readRequiredIntInput('ls-n-drafters'),
        n_picks:              readRequiredIntInput('ls-n-picks'),
        cash_per_team:        readRequiredIntInput('ls-cash-per-team'),
        third_round_reversal: (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked,
        team_names:           (document.getElementById('ls-team-names') as HTMLTextAreaElement)
                                  .value.split('\n').map(s => s.trim()).filter(s => s.length > 0),
    }
}

/**
 * Returns the live-platform connection for the session request, or null for
 * 'Enter your own data' (or a live platform with no league ID entered yet).
 */
export function getPlatformConfig(): { league_id: string; division_id?: string | null } | null {
    const platform = (document.getElementById('ls-platform') as HTMLInputElement).value
    if (platform === 'Enter your own data') return null
    const leagueIdEl = document.getElementById('ls-league-id') as HTMLInputElement | null
    const leagueId   = leagueIdEl?.value.trim() ?? ''
    if (!leagueId) return null
    const divisionEl = document.getElementById('ls-division') as HTMLInputElement | null
    return { league_id: leagueId, division_id: (divisionEl?.value || null) }
}

/** Creates a grid cell `<div>` that stacks its label and input vertically. */
function makeCell(extraClass?: string): HTMLDivElement {
    const cell = document.createElement('div')
    cell.className = extraClass ? `ls-cell ${extraClass}` : 'ls-cell'
    return cell
}

/** Generates default team names ("Drafter 1", "Drafter 2", …) for `n` drafters. */
function defaultTeamNames(n: number): string {
    return Array.from({ length: n }, (_, i) => `Drafter ${i + 1}`).join('\n')
}
