// app_state.ts
// Shared application state: the current player list and category names.
// Kept in a standalone module with no app-internal imports (only types.ts)
// so that any module can read the state without creating circular dependencies.

import { Player } from './types.js'

let allPlayers:  Player[] = []   // full dataset — only grows, never shrinks
let candidates:  Player[] = []   // current evaluate output (may be a subset, e.g. waiver free agents)
let playerByName: Map<string, Player> = new Map()
let categories: string[] = [
    'Field Goal %', 'Free Throw %', 'Threes', 'Points',
    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
]

/** Returns every player in the loaded dataset, regardless of draft/waiver state. */
export function getPlayers(): Player[] { return allPlayers }

/** Returns the current list of ranked candidate players displayed in the table. */
export function getCandidatePlayers(): Player[] { return candidates }

/** Returns a name → Player map, rebuilt whenever the player list is updated. */
export function getPlayerByName(): Map<string, Player> { return playerByName }

/** Returns the current list of scoring category names (e.g. "Points", "Rebounds"). */
export function getCategories(): string[] { return categories }

/** Replaces the candidate list. Called by session.ts after an evaluate response.
 *  Also expands allPlayers when the new list is larger (i.e. a full-pool evaluate). */
export function setPlayers(p: Player[]): void {
    candidates = p
    playerByName = new Map(p.map(pl => [pl.name, pl]))
    if (p.length >= allPlayers.length) allPlayers = p
}

/** Replaces the category list. Called by session.ts when categories change. */
export function setCategories(c: string[]): void { categories = c }
