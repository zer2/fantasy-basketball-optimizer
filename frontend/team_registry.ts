// team_registry.ts
// The league's teams, in seat order: {team_id -> name}, mirroring player_registry.ts. The interior is keyed by team
// id everywhere (player_assignments, my_team_id, remaining_cash, the seat, the boards); a team's name is read here
// only to display it. The ids (backend/team_identity.py): a connected league's own platform team ids, "empty-N" for
// a Yahoo seat nobody has taken, and "seat-1" ... "seat-N" for the user's own data.
//
// Keyed by name, two ESPN teams whose names ended in a space showed every roster slot as undefined, and were scored
// as empty, without an error (2026-11-18): the sidebar trimmed the names and the rosters were keyed by the raw ones.

export interface TeamRegistryEntry {
    team_id: string
    name: string
}

/** Fired (on document) whenever the registry is replaced, so the seat selector can rebuild its options. */
export const TEAM_REGISTRY_CHANGED = 'team-registry-changed'

let teamsInSeatOrder: TeamRegistryEntry[] | null = null
let entriesById: Map<string, TeamRegistryEntry> | null = null

/** Installs the league's teams: the user's own seats (makeOwnDataTeams) or a connected league's (ConnectResponse). */
export function setTeamRegistry(teams: TeamRegistryEntry[]): void {
    entriesById = new Map(teams.map(team => [team.team_id, team]))
    if (entriesById.size !== teams.length) {
        throw new Error(`The league lists a team id more than once: ${teams.map(team => team.team_id).join(', ')}`)
    }
    teamsInSeatOrder = [...teams]
    document.dispatchEvent(new Event(TEAM_REGISTRY_CHANGED))
}

/** The registry entry for a team id. Throws on an unknown id: a payload or a seat naming a team the league lacks is
 *  a contract violation, not a renderable state. */
export function getTeamEntry(teamId: string): TeamRegistryEntry {
    const entry = entriesById?.get(teamId)
    if (!entry) throw new Error(`Team id ${teamId} is not in the team registry`)
    return entry
}

/** Every team, in seat order. */
export function getAllTeamIdentities(): TeamRegistryEntry[] {
    if (teamsInSeatOrder === null) throw new Error('getAllTeamIdentities called before the team registry was installed')
    return [...teamsInSeatOrder]
}

/** Every team id, in seat order. */
export function listTeamIds(): string[] {
    return getAllTeamIdentities().map(team => team.team_id)
}

/** The user's own league: seats "seat-1" ... "seat-N", named "Team 1" ... "Team N" (backend
 *  make_own_data_team_identities). */
export function makeOwnDataTeams(nDrafters: number): TeamRegistryEntry[] {
    return Array.from({ length: nDrafters }, (_, index) => ({ team_id: `seat-${index + 1}`, name: `Team ${index + 1}` }))
}
