// data_entry/season/season_helpers.ts
// Shared readers for the Season tabs (Trading, Waiver Wire), which both need to
// pull the league's team names and the roster grid out of the DOM.

import { readRequiredIntInput } from '../../helper_functions.js'
import { defaultTeamLabel } from '../team_labels.js'
import { getTeamIdentitiesFromSidebar } from '../../parameter_collection/league_settings.js'


/** Reads roster assignments from the Rosters tab grid (sr-player-{row}-{col}). The single
 *  DOM-scrape choke point: the grid's hidden inputs carry stringified player ids, so a
 *  non-numeric non-empty value is a programming error and throws rather than being sent on. */
export function readRosterAssignments(): Record<string, number[]> {
    const teamNames = getTeamIdentitiesFromSidebar()
    const nDrafters = readRequiredIntInput('ls-n-drafters')
    const nPicks    = readRequiredIntInput('ls-n-picks')

    const assignments: Record<string, number[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d] ?? defaultTeamLabel(d)   // d can exceed teamNames.length
        const players: number[] = []
        for (let r = 0; r < nPicks; r++) {
            const input = document.getElementById(`sr-player-${r}-${d}`) as HTMLInputElement | null
            const value = input?.value ?? ''            // getElementById can return null
            if (!value) continue
            const playerId = Number(value)
            if (Number.isNaN(playerId)) {
                throw new Error(`Roster cell sr-player-${r}-${d} carried a non-numeric value: "${value}"`)
            }
            players.push(playerId)
        }
        assignments[team] = players
    }
    return assignments
}
