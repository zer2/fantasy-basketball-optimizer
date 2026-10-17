// player_registry.ts
// The session's player identities: {player_id -> name, last name, positions, headshot
// availability}, delivered inside SessionResponse/GScoresResponse whenever the pool
// changes. The single frontend source of player names: the interior is keyed by
// player id everywhere, and names are resolved here only at display time (or, for
// human-text inputs like clipboard pastes, resolved back to ids by name).

export interface PlayerRegistryEntry {
    player_id: number
    name: string
    last_name: string
    positions: string[]
    has_headshot: boolean
}

/** The replacement-player sentinel's id (backend RP_PLAYER_ID): a stand-in the pipeline injects, not a player. */
export const REPLACEMENT_PLAYER_ID = -1

let entriesById: Map<number, PlayerRegistryEntry> | null = null
let playerIdByName: Map<string, number> | null = null

// Bumped by every install, so a response can tell whether the registry it would be read against is still the one in
// place when it was requested (getRegistryGeneration).
let registryGeneration = 0

/** Which registry install is current: compare one taken before a request with one taken after it to see whether the
 *  registry was replaced while the request was out. */
export function getRegistryGeneration(): number {
    return registryGeneration
}

/** Installs a session's registry (called by the api layer whenever a payload carries one).
 *  Announces the install as a document event rather than calling listeners directly, so
 *  reactions that need the api layer (the headshot preloader in player_display.ts) don't
 *  create an import cycle through this module. */
export function setPlayerRegistry(entries: PlayerRegistryEntry[]): void {
    registryGeneration += 1
    entriesById = new Map(entries.map(entry => [entry.player_id, entry]))
    playerIdByName = new Map(entries.map(entry => [entry.name, entry.player_id]))
    document.dispatchEvent(new Event('player-registry-updated'))
}

/** The registry entry for a player id. Throws on an unknown id — a payload referencing a
 *  player the registry lacks is a contract violation, not a renderable state. */
export function getRegistryEntry(playerId: number): PlayerRegistryEntry {
    const entry = entriesById?.get(playerId)
    if (!entry) throw new Error(`Player id ${playerId} is not in the session registry`)
    return entry
}

/** Resolves a bare display name to a player id (case-sensitive exact match), or null when
 *  the name is not in the session registry. For human-text inputs — clipboard pastes —
 *  where an unknown name is a legitimate
 *  outcome the caller must handle visibly, not a contract violation. */
export function findPlayerIdByName(name: string): number | null {
    return playerIdByName?.get(name) ?? null
}

/** Every identity in the session registry (registry order, RP included). */
export function getAllPlayerIdentities(): PlayerRegistryEntry[] {
    if (entriesById === null) throw new Error('getAllPlayerIdentities called before the registry was installed')
    return [...entriesById.values()]
}
