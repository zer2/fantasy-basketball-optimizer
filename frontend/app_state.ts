// app_state.ts
// Shared application state: the current player list and category names.
// Kept in a standalone module with no app-internal imports (only types.ts)
// so that any module can read the state without creating circular dependencies.

import { Player } from './types.js'

let players: Player[] = []
let playerByName: Map<string, Player> = new Map()
let categories: string[] = [
    'Field Goal %', 'Free Throw %', 'Threes', 'Points',
    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
]

/** Returns the current list of ranked candidate players displayed in the table. */
export function getPlayers(): Player[] { return players }

/** Returns a name → Player map, rebuilt whenever the player list is updated. */
export function getPlayerByName(): Map<string, Player> { return playerByName }

/** Returns the current list of scoring category names (e.g. "Points", "Rebounds"). */
export function getCategories(): string[] { return categories }

/** Replaces the player list. Called by session.ts after an evaluate response. */
export function setPlayers(p: Player[]): void {
    players = p
    playerByName = new Map(p.map(pl => [pl.name, pl]))
}

/** Replaces the category list. Called by session.ts when categories change. */
export function setCategories(c: string[]): void { categories = c }
