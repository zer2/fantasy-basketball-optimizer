// data_entry/team_labels.ts
// Per-drafter DISPLAY labels — presentation only. Team *identity* stays "Team N" (see
// league_settings #ls-team-names / getTeamNames); a label is purely what's shown in the UI.
// Editing a label changes nothing in logic (my_team_id, draft/auction state, the backend all
// use the identity), so it can never reset the draft or affect an evaluate.

import { pref, savePref } from '../preferences.js'

/** Fired (on document) whenever a label changes, so the seat selector can relabel. */
export const TEAM_LABELS_CHANGED = 'team-labels-changed'

/** The default (identity) label for a drafter index: "Team 1", "Team 2", … */
export function defaultTeamLabel(index: number): string {
    return `Team ${index + 1}`
}

/** The raw saved custom label (may be empty). Use for the header input's value. */
export function getRawTeamLabel(index: number): string {
    return pref(`team_label_${index}`, '') as string
}

/** The resolved display label: the saved custom label, or the default "Team N". */
export function getTeamLabel(index: number): string {
    return getRawTeamLabel(index).trim() || defaultTeamLabel(index)
}

/** Persists a custom label (empty/whitespace clears back to the default) and notifies listeners. */
export function setTeamLabel(index: number, text: string): void {
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
    input.placeholder = defaultTeamLabel(index)
    input.addEventListener('input', () => setTeamLabel(index, input.value), { signal })
    return input
}
