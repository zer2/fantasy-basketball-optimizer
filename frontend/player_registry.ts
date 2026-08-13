// player_registry.ts
// The session's player identities: {player_id -> name, last name, positions, headshot
// availability}, delivered inside SessionResponse/GScoresResponse whenever the pool
// changes. The single frontend source of player names.
//
// Stage B of the identity refactor: only the api/ translation layer reads this, keeping
// the rest of the frontend name-keyed until Stage C makes it id-native. The legacy-label
// helpers below reconstruct the string-identity era's 'Name (POS)' keys for that interior
// and are deleted with it.

export interface PlayerRegistryEntry {
    player_id: number
    name: string
    last_name: string
    positions: string[]
    has_headshot: boolean
}

export const RP_PLAYER_ID = -1

let entriesById: Map<number, PlayerRegistryEntry> | null = null
let playerIdByLabel: Map<string, number> | null = null

/** Installs a session's registry (called by the api layer whenever a payload carries one). */
export function setPlayerRegistry(entries: PlayerRegistryEntry[]): void {
    entriesById = new Map(entries.map(entry => [entry.player_id, entry]))
    playerIdByLabel = new Map()
    for (const entry of entries) {
        playerIdByLabel.set(buildLegacyLabelFromEntry(entry), entry.player_id)
        playerIdByLabel.set(entry.name, entry.player_id)
    }
}

/** The registry entry for a player id. Throws on an unknown id — a payload referencing a
 *  player the registry lacks is a contract violation, not a renderable state. */
export function getRegistryEntry(playerId: number): PlayerRegistryEntry {
    const entry = entriesById?.get(playerId)
    if (!entry) throw new Error(`Player id ${playerId} is not in the session registry`)
    return entry
}

function buildLegacyLabelFromEntry(entry: PlayerRegistryEntry): string {
    if (entry.player_id === RP_PLAYER_ID) return 'RP'
    const positionDisplay = entry.positions.length > 0 ? entry.positions.join(',') : 'NP'
    return `${entry.name} (${positionDisplay})`
}

/** The string-identity era's 'Name (POS)' key for a player id (Stage-B interior only). */
export function buildLegacyLabel(playerId: number): string {
    return buildLegacyLabelFromEntry(getRegistryEntry(playerId))
}

/** Resolves a legacy label (or bare name) back to a player id. Throws on an unknown
 *  label — silently dropping a player from a payload would corrupt the board. */
export function findPlayerIdByLegacyLabel(label: string): number {
    const playerId = playerIdByLabel?.get(label)
    if (playerId === undefined) throw new Error(`Player ${label!} is not in the session registry`)
    return playerId
}
