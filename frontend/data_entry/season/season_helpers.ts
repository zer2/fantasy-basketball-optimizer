// data_entry/season/season_helpers.ts
// Shared readers for the Season tabs (Trading, Waiver Wire), which both need to
// pull the league's team names and the roster grid out of the DOM.

import { readRequiredIntInput } from '../../helper_functions.js'
import { defaultTeamLabel } from '../team_labels.js'
import { getTeamIdentitiesFromSidebar, getLeagueSettings } from '../../setting_collection/league_settings.js'
import { REPLACEMENT_PLAYER_ID } from '../../player_registry.js'
import { getLivePlatformRosterError, getLivePlatformRosters } from '../../api/season_session.js'


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

/** The players of a roster that can be dropped or traded: all but the unmatched ones, which the roster carries as the
 *  replacement player (a stand-in for whoever the platform rostered, not a player to move). */
export function listMovablePlayers(roster: number[]): number[] {
    return roster.filter(playerId => playerId !== REPLACEMENT_PLAYER_ID)
}

/** What a Season Mode tab says before it has the full teams it needs, or null while a connected league's rosters are
 *  still on their way: the tab is drawn again when they arrive, and meanwhile the indicator beside the tabs says
 *  "Starting..." (a "Loading your league's rosters…" line of its own sat in a different place on each tab and fell out
 *  of step with the indicator, 2026-10-28). Asking the user to fill in a connected league's rosters read as broken
 *  (2026-10-13). */
export function readEmptyRostersMessage(
    teamsNeeded: string
    , toolName: string
): string | null {
    if (getLeagueSettings().platform === 'Enter your own data') {
        return `Fill out at least ${teamsNeeded} on the Rosters tab to use ${toolName}.`
    }
    const rosterError = getLivePlatformRosterError()
    if (rosterError !== null) return `Could not load your league's rosters: ${rosterError}`
    // The league answered and its rosters are empty: a season that has not drafted yet (an ESPN league keeps its id
    // across seasons, and the newest season is chosen by default). Saying "loading" there never ended (2026-10-21).
    const platformRosters = getLivePlatformRosters()
    if (platformRosters !== null && Object.values(platformRosters).every(roster => roster.length === 0)) {
        return "This league's rosters are empty for the season it is connected to (has it drafted yet?)."
            + (getLeagueSettings().platform === 'Retrieve from ESPN'
                ? ' Choose another season under League Settings to see that one.' : '')
    }
    if (platformRosters === null) return null
    // The rosters arrived but too few teams are full (a league still filling, or one smaller than the tool needs).
    return `Your league doesn't have ${teamsNeeded} yet, which ${toolName} needs.`
}
