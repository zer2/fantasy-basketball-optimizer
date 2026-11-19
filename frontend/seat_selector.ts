// seat_selector.ts
// The "Select team" control above the H-score table: which seat the app evaluates for.
// This module owns the widget — how it is built, how team ids map onto display labels, and
// how its option set stays consistent with the league's teams (team_registry). What the
// app DOES about a seat change (evaluate, re-layout) is orchestration and lives in main.ts,
// which attaches its own listener to the element this module renders.

import { makeCustomSelect } from './custom_select.js'
import { getCurrentSeat, setCurrentSeat } from './app_state.js'
import { getTeamLabel, defaultTeamLabel } from './data_entry/team_labels.js'
import { getAllTeamIdentities, type TeamRegistryEntry } from './team_registry.js'

let seatSelect: ReturnType<typeof makeCustomSelect> | null = null

// Seat selector option from a team: the value is always the team id. An own-data team (named "Team N") shows its
// editable display label; a live platform's team shows its own name as-is.
function buildSeatOption(team: TeamRegistryEntry, index: number): { value: string; label: string } {
    const label = team.name === defaultTeamLabel(index) ? getTeamLabel(index) : team.name
    return { value: team.team_id, label }
}

function requireSeatSelect(): ReturnType<typeof makeCustomSelect> {
    if (seatSelect === null) throw new Error('Seat selector used before renderSeatSelector')
    return seatSelect
}

/** Builds the seat selector once into #seat-selector-container (layout.ts shows/hides the
 *  container) and adopts the first team as the initial seat. Returns the widget's root
 *  element so the caller can attach its seat-change listener. */
export function renderSeatSelector(): HTMLElement {
    const seatSelectorContainer = document.getElementById('seat-selector-container') as HTMLElement
    const initialTeams = getAllTeamIdentities()
    seatSelect = makeCustomSelect(
        'seat-select',
        initialTeams.map(buildSeatOption),
    )
    seatSelect.element.style.flex = '1'

    const seatLabel = document.createElement('div')
    seatLabel.className   = 'pick-control-label'
    seatLabel.textContent = 'Select team'

    const seatSelectorRow = document.createElement('div')
    seatSelectorRow.className = 'seat-selector-row'
    seatSelectorRow.append(seatLabel, seatSelect.element)

    const seatSelectorWrap = document.createElement('div')
    seatSelectorWrap.className = 'seat-selector-wrap'
    seatSelectorWrap.append(seatSelectorRow)
    seatSelectorContainer.append(seatSelectorWrap)

    // The widget keeps the app-state seat in lockstep with the user's selection. Reactions
    // (evaluate, re-layout) are the caller's, attached to the returned element — and since
    // this listener registers first, the seat is already consistent when any reaction reads it.
    seatSelect.element.addEventListener('change', () => setCurrentSeat(requireSeatSelect().getValue() ?? null))

    if (initialTeams.length > 0) {
        setCurrentSeat(initialTeams[0].team_id)
        seatSelect.setValue(initialTeams[0].team_id)
    }
    return seatSelect.element
}

/** Rebuilds the option list from the league's teams and labels, preserving the selected
 *  seat by value. A null seat adopts the first team — in app state and the control together.
 *  Never dispatches a change event: each caller knows whether the moment warrants an
 *  evaluate, so the reaction is theirs. Returns the adopted team id, or null when the
 *  existing selection survived. */
export function refreshSeatOptions(): string | null {
    const teams = getAllTeamIdentities()
    const teamIds = teams.map(team => team.team_id)
    // A seat is kept across a refresh only while it is still one of the teams. Connecting a
    // live platform replaces the own-data seats with the league's teams, and a seat left
    // pointing at 'seat-1' is not merely stale: it is sent as my_team_id on every evaluate,
    // against a board keyed by the platform's team ids, where no such team exists. Ids are
    // compared, not labels -- a relabel changes what a team is called, never which team it is.
    const seat = getCurrentSeat()
    const keptSeat = seat !== null && teamIds.includes(seat) ? seat : null
    requireSeatSelect().setOptions(teams.map(buildSeatOption), keptSeat ?? teamIds[0])
    if (keptSeat === null && teamIds.length > 0) {
        setCurrentSeat(teamIds[0])
        return teamIds[0]
    }
    return null
}

/** Shows or hides the whole selector container. Autopilot hides it while it drives the
 *  seat itself; layout.ts separately toggles display per layout, on a different property,
 *  so the two writers cannot fight. */
export function setSeatSelectorVisible(visible: boolean): void {
    ;(document.getElementById('seat-selector-container') as HTMLElement).style.visibility =
        visible ? '' : 'hidden'
}

/** Empties the selector and the seat — the unconnected-live-platform state, where there is
 *  no draft board and nothing to evaluate for. */
export function clearSeatOptions(): void {
    setCurrentSeat(null)
    requireSeatSelect().setOptions([])
}
