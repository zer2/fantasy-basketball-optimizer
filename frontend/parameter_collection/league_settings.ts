// Collects: sport (NBA only), mode, n_drafters, n_picks, third_round_reversal, team_names, my_team_id
// Mirrors league_settings_popover() in src/parameter_collection/league_settings.py

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, makeNumberInput, makeSidebarToggle } from '../helper_functions.js'

export type DraftMode = 'Draft Mode' | 'Auction Mode' | 'Season Mode'

/**
 * Renders the League Settings section into `container`.
 * Includes sport (static), mode, third-round-reversal toggle (Draft Mode only),
 * drafter/pick counts, team names textarea, and "Your team" selector.
 */
export function renderLeagueSettings(container: HTMLElement): void {

    // Sport — static for now
    const sportLabel = document.createElement('div')
    sportLabel.className = 'sidebar-label'
    sportLabel.textContent = 'Sport'
    container.append(sportLabel)

    const sportValue = document.createElement('div')
    sportValue.className = 'sidebar-static'
    sportValue.textContent = 'NBA'
    container.append(sportValue)

    // Mode
    container.append(makeLabel('ls-mode', 'Mode'))
    const modeSelect = makeCustomSelect(
        'ls-mode',
        ['Draft Mode', 'Auction Mode', 'Season Mode'].map(m => ({ value: m, label: m })),
        'Draft Mode',
    )
    container.append(modeSelect.element)

    // Third round reversal toggle (Draft Mode only)
    const trrToggle = makeSidebarToggle('ls-third-round-reversal', 'Third round reversal')
    trrToggle.id = 'ls-trr-row'
    container.append(trrToggle)

    // The checkbox inside the toggle is what the getter reads; get a reference for resetting it.
    const trrCheckbox = trrToggle.querySelector('input') as HTMLInputElement

    modeSelect.element.addEventListener('change', () => {
        const mode = modeSelect.getValue()
        trrToggle.style.display = mode === 'Draft Mode' ? '' : 'none'
        if (mode !== 'Draft Mode') trrCheckbox.checked = false
    })

    // Number of drafters
    container.append(makeLabel('ls-n-drafters', 'Number of drafters'))
    const nDraftersInput = makeNumberInput('ls-n-drafters', 12, 2)
    container.append(nDraftersInput)

    // Picks per drafter
    container.append(makeLabel('ls-n-picks', 'Picks per drafter'))
    container.append(makeNumberInput('ls-n-picks', 13, 1))

    // Team names
    container.append(makeLabel('ls-team-names', 'Team names (one per line)'))
    const teamNamesInput = document.createElement('textarea')
    teamNamesInput.id = 'ls-team-names'
    teamNamesInput.className = 'sidebar-input'
    teamNamesInput.rows = 4
    teamNamesInput.value = defaultTeamNames(12)
    container.append(teamNamesInput)

    // My team — a select whose options are kept in sync with the team names textarea,
    // so the user can only choose a name that is actually in the league.
    const myTeamSelect = makeCustomSelect('ls-my-team-id', [])

    function syncMyTeamSelect(): void {
        const names = teamNamesInput.value.split('\n').map(s => s.trim()).filter(s => s.length > 0)
        myTeamSelect.setOptions(names.map(n => ({ value: n, label: n })))
    }

    // Populate options before appending to DOM so the element is never seen in an empty state.
    syncMyTeamSelect()

    container.append(makeLabel('ls-my-team-id', 'Your team'))
    container.append(myTeamSelect.element)

    // Re-sync the team names textarea and "Your team" dropdown whenever n_drafters changes,
    // so the textarea always has exactly n_drafters entries and the dropdown matches.
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
 * The `my_team_id` value is one of the strings from `team_names`.
 */
export function getLeagueSettings(): {
    sport: string
    mode: DraftMode
    n_drafters: number
    n_picks: number
    third_round_reversal: boolean
    team_names: string[]
    my_team_id: string
} {
    return {
        sport:                'NBA',
        mode:                 (document.getElementById('ls-mode') as HTMLInputElement).value as DraftMode,
        n_drafters:           parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value),
        n_picks:              parseInt((document.getElementById('ls-n-picks') as HTMLInputElement).value),
        third_round_reversal: (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked,
        team_names:           (document.getElementById('ls-team-names') as HTMLTextAreaElement)
                                  .value.split('\n').map(s => s.trim()).filter(s => s.length > 0),
        my_team_id:           (document.getElementById('ls-my-team-id') as HTMLInputElement).value,
    }
}

/** Generates default team names ("Drafter 1", "Drafter 2", …) for `n` drafters. */
function defaultTeamNames(n: number): string {
    return Array.from({ length: n }, (_, i) => `Drafter ${i + 1}`).join('\n')
}
