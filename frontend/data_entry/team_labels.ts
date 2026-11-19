// data_entry/team_labels.ts
// Per-drafter DISPLAY labels — presentation only, keyed by seat index (the board's columns). A team's identity is its
// team id (team_registry); a label is purely what's shown in the UI. Editing a label changes nothing in logic
// (my_team_id, draft/auction state, the backend all use the team id), so it can never reset the draft or affect an
// evaluate.

import { pref, savePref } from '../preferences.js'
import { getAllTeamIdentities } from '../team_registry.js'

/** Fired (on document) whenever a label changes, so the seat selector can relabel. */
export const TEAM_LABELS_CHANGED = 'team-labels-changed'

/** The default (identity) label for a drafter index: "Team 1", "Team 2", … */
export function defaultTeamLabel(index: number): string {
    return `Team ${index + 1}`
}

/** The raw saved custom label (may be empty). Use for the header input's value. */
function getRawTeamLabel(index: number): string {
    return pref(`team_label_${index}`, '') as string
}

/** The team's own name: "Team N" for own data, the league's real team name when connected to a live platform, which
 *  is what a header should say. A seat index past the league's teams is a bug, and throws. */
function identityLabel(index: number): string {
    const teams = getAllTeamIdentities()
    if (index < 0 || index >= teams.length) {
        throw new Error(`Seat ${index + 1} is not one of the league's ${teams.length} teams`)
    }
    return teams[index].name
}

/** The resolved display label: the saved custom label, else the team's own name. */
export function getTeamLabel(index: number): string {
    return getRawTeamLabel(index).trim() || identityLabel(index)
}

/** Persists a custom label (empty/whitespace clears back to the default) and notifies listeners. */
function setTeamLabel(index: number, text: string): void {
    savePref(`team_label_${index}`, text.trim())
    document.dispatchEvent(new Event(TEAM_LABELS_CHANGED))
}

/** Builds the editable display-label input for a column header (draft + auction). Typing persists
 *  the label (firing TEAM_LABELS_CHANGED) without touching identity, so it never resets the board. */
export function makeTeamLabelInput(index: number, signal?: AbortSignal): HTMLInputElement {
    const input = document.createElement('input')
    input.type        = 'text'
    input.className    = 'team-label-input'
    input.value       = getRawTeamLabel(index)
    input.placeholder = identityLabel(index)
    input.addEventListener('input', () => setTeamLabel(index, input.value), { signal })
    return input
}
