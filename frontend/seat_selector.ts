// seat_selector.ts
// The "Select team" control above the H-score table: which seat the app evaluates for.
// This module owns the widget — how it is built, how team identities map onto display
// labels, and how its option set stays consistent with the current identities. What the
// app DOES about a seat change (evaluate, re-layout) is orchestration and lives in main.ts,
// which attaches its own listener to the element this module renders.

import { makeCustomSelect } from './custom_select.js'
import { getCurrentSeat, setCurrentSeat } from './app_state.js'
import { getTeamLabel, defaultTeamLabel } from './data_entry/team_labels.js'
import { getTeamIdentitiesFromSidebar } from './parameter_collection/league_settings.js'

let seatSelect: ReturnType<typeof makeCustomSelect> | null = null

// Seat selector option from a team identity: the value is always the identity ("Team N" for own
// data; the real name for a live platform). For own-data identities we show the editable display
// label; a live-platform name (which differs from the "Team N" default) is shown as-is.
function buildSeatOption(identityName: string, index: number): { value: string; label: string } {
    const label = identityName === defaultTeamLabel(index) ? getTeamLabel(index) : identityName
    return { value: identityName, label }
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
    const initialTeamNames = getTeamIdentitiesFromSidebar()
    seatSelect = makeCustomSelect(
        'seat-select',
        initialTeamNames.map(buildSeatOption),
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

    if (initialTeamNames.length > 0) {
        setCurrentSeat(initialTeamNames[0])
        seatSelect.setValue(initialTeamNames[0])
    }
    return seatSelect.element
}

/** Rebuilds the option list from the current identities and labels, preserving the selected
 *  seat by value. A null seat adopts the first team — in app state and the control together.
 *  Never dispatches a change event: each caller knows whether the moment warrants an
 *  evaluate, so the reaction is theirs. Returns the adopted identity, or null when the
 *  existing selection survived. */
export function refreshSeatOptions(): string | null {
    const names = getTeamIdentitiesFromSidebar()
    requireSeatSelect().setOptions(names.map(buildSeatOption), getCurrentSeat() ?? names[0])
    if (getCurrentSeat() === null && names.length > 0) {
        setCurrentSeat(names[0])
        return names[0]
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
