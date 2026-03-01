// Collects: sport, platform, mode, n_drafters, n_picks,
//           cash_per_team (Auction Mode only), third_round_reversal, team_names, my_team_id
// Mirrors league_settings_popover() in src/parameter_collection/league_settings.py

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, makeNumberInput, makeSidebarToggle } from '../helper_functions.js'

export type DraftMode = 'Draft Mode' | 'Auction Mode' | 'Season Mode'
export type Platform = 'Enter your own data' | 'Retrieve from Yahoo' | 'Retrieve from Fantrax' | 'Retrieve from ESPN'

const PLATFORM_OPTIONS: Platform[] = [
    'Enter your own data',
    'Retrieve from Yahoo',
    'Retrieve from Fantrax',
    'Retrieve from ESPN',
]

/**
 * Renders the League Settings section into `container`.
 * Layout: 2-column grid for compact items, followed by full-width items.
 */
export function renderLeagueSettings(container: HTMLElement): void {

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
        'NBA',
    )
    sportCell.append(sportSelect.element)
    grid.append(sportCell)

    // ── Mode (right col) ──────────────────────────────────────────────────
    const modeCell = makeCell()
    modeCell.append(makeLabel('ls-mode', 'Mode'))
    const modeSelect = makeCustomSelect(
        'ls-mode',
        ['Draft Mode', 'Auction Mode', 'Season Mode'].map(m => ({ value: m, label: m })),
        'Draft Mode',
    )
    modeCell.append(modeSelect.element)
    grid.append(modeCell)

    // ── Platform (full-width) ─────────────────────────────────────────────
    // Controls whether league data is entered manually or pulled from a platform.
    const platformCell = makeCell('ls-cell-full')
    platformCell.append(makeLabel('ls-platform', 'Fantasy Platform'))
    const platformSelect = makeCustomSelect(
        'ls-platform',
        PLATFORM_OPTIONS.map(p => ({ value: p, label: p })),
        'Enter your own data',
    )
    platformCell.append(platformSelect.element)
    grid.append(platformCell)

    // ── Number of drafters (left col) ─────────────────────────────────────
    const draftersCell = makeCell()
    draftersCell.append(makeLabel('ls-n-drafters', 'Drafters'))
    const nDraftersInput = makeNumberInput('ls-n-drafters', 12, 2)
    draftersCell.append(nDraftersInput)
    grid.append(draftersCell)

    // ── Picks per drafter (right col) ─────────────────────────────────────
    const picksCell = makeCell()
    picksCell.append(makeLabel('ls-n-picks', 'Picks / drafter'))
    picksCell.append(makeNumberInput('ls-n-picks', 13, 1))
    grid.append(picksCell)

    // ── Budget per team (left col, Auction Mode only) ─────────────────────
    const cashCell = makeCell()
    cashCell.style.display = 'none'
    cashCell.append(makeLabel('ls-cash-per-team', 'Budget / team ($)'))
    cashCell.append(makeNumberInput('ls-cash-per-team', 200, 1))
    grid.append(cashCell)

    // ── Third round reversal toggle (full-width, Draft Mode only) ─────────
    const trrToggle = makeSidebarToggle('ls-third-round-reversal', 'Third round reversal')
    trrToggle.id = 'ls-trr-row'
    container.append(trrToggle)

    const trrCheckbox = trrToggle.querySelector('input') as HTMLInputElement

    modeSelect.element.addEventListener('change', () => {
        const mode = modeSelect.getValue()
        cashCell.style.display = mode === 'Auction Mode' ? '' : 'none'
        trrToggle.style.display = mode === 'Draft Mode' ? '' : 'none'
        if (mode !== 'Draft Mode') trrCheckbox.checked = false
    })

    // ── Team names textarea (full-width) ───────────────────────────────────
    container.append(makeLabel('ls-team-names', 'Team names (one per line)'))
    const teamNamesInput = document.createElement('textarea')
    teamNamesInput.id = 'ls-team-names'
    teamNamesInput.className = 'sidebar-input'
    teamNamesInput.rows = 4
    teamNamesInput.value = defaultTeamNames(12)
    container.append(teamNamesInput)

    // ── Your team selector (full-width) ───────────────────────────────────
    const myTeamSelect = makeCustomSelect('ls-my-team-id', [])

    function syncMyTeamSelect(): void {
        const names = teamNamesInput.value.split('\n').map(s => s.trim()).filter(s => s.length > 0)
        myTeamSelect.setOptions(names.map(n => ({ value: n, label: n })))
    }

    syncMyTeamSelect()

    container.append(makeLabel('ls-my-team-id', 'Your team'))
    container.append(myTeamSelect.element)

    // Re-sync team list when drafter count changes
    nDraftersInput.addEventListener('change', () => {
        const n = parseInt(nDraftersInput.value)
        if (!isNaN(n) && n > 0) {
            teamNamesInput.value = defaultTeamNames(n)
            syncMyTeamSelect()
        }
    })

    teamNamesInput.addEventListener('input', syncMyTeamSelect)
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
    my_team_id: string
} {
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value as DraftMode
    return {
        sport:                (document.getElementById('ls-sport') as HTMLInputElement).value,
        platform:             (document.getElementById('ls-platform') as HTMLInputElement).value as Platform,
        mode,
        n_drafters:           parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value),
        n_picks:              parseInt((document.getElementById('ls-n-picks') as HTMLInputElement).value),
        cash_per_team:        parseInt((document.getElementById('ls-cash-per-team') as HTMLInputElement).value),
        third_round_reversal: (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked,
        team_names:           (document.getElementById('ls-team-names') as HTMLTextAreaElement)
                                  .value.split('\n').map(s => s.trim()).filter(s => s.length > 0),
        my_team_id:           (document.getElementById('ls-my-team-id') as HTMLInputElement).value,
    }
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
