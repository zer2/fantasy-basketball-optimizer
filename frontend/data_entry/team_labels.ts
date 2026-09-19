// data_entry/team_labels.ts
// Per-drafter DISPLAY labels — presentation only. Team *identity* is what the sidebar holds (see
// league_settings #ls-team-names / getTeamIdentitiesFromSidebar); a label is purely what's shown in the UI.
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
function getRawTeamLabel(index: number): string {
    return pref(`team_label_${index}`, '') as string
}

/** The team's own name: the identity the sidebar holds for this seat, or "Team N" if it has none.
 *
 *  Read from the DOM rather than imported, because league_settings imports this module and the
 *  hidden #ls-team-names textarea exists precisely so that readers can get identities without
 *  one. With own data the identities ARE "Team N", so this changes nothing there; connected to a
 *  live platform they are the league's real team names, which is what a header should say.
 */
function identityLabel(index: number): string {
    const identities = document.getElementById('ls-team-names') as HTMLTextAreaElement | null
    return identities?.value.split('\n')[index]?.trim() || defaultTeamLabel(index)
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
