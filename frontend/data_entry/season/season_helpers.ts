// data_entry/season/season_helpers.ts
// Shared readers for the Season tabs (Trading, Waiver Wire), which both need to
// pull the league's team names and the roster grid out of the DOM.

import { readRequiredIntInput } from '../../helper_functions.js'

/** Reads team names from the sidebar textarea. */
export function readTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

/** Reads roster assignments from the Rosters tab grid (sr-player-{row}-{col}). */
export function readRosterAssignments(): Record<string, string[]> {
    const teamNames = readTeamNames()
    const nDrafters = readRequiredIntInput('ls-n-drafters')
    const nPicks    = readRequiredIntInput('ls-n-picks')

    const assignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d] ?? `Team ${d + 1}`   // d can exceed teamNames.length
        const players: string[] = []
        for (let r = 0; r < nPicks; r++) {
            const input = document.getElementById(`sr-player-${r}-${d}`) as HTMLInputElement | null
            const val = input?.value ?? ''              // getElementById can return null
            if (val) players.push(val)
        }
        assignments[team] = players
    }
    return assignments
}
