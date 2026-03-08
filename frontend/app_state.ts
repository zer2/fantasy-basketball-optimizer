// app_state.ts
// Shared application state: the current player list and category names.
// Kept in a standalone module with no app-internal imports (only types.ts)
// so that any module can read the state without creating circular dependencies.

import { Player, PlayerGScore, SportConfig } from './types.js'

let allPlayers:      Player[] = []   // full dataset — only grows, never shrinks
let candidates:      Player[] = []   // current evaluate output (may be a subset, e.g. waiver free agents)
let allPlayerByName: Map<string, Player> = new Map()
let gScoreByName:    Map<string, PlayerGScore> = new Map()
let categories: string[] = [
    'Field Goal %', 'Free Throw %', 'Threes', 'Points',
    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
]

/** Returns every player in the loaded dataset, regardless of draft/waiver state. */
export function getPlayers(): Player[] { return allPlayers }

/** Returns the current list of ranked candidate players displayed in the table. */
export function getCandidatePlayers(): Player[] { return candidates }

/** Returns a name → Player map for every player in the full dataset. */
export function getPlayerByName(): Map<string, Player> { return allPlayerByName }

/** Returns the current list of scoring category names (e.g. "Points", "Rebounds"). */
export function getCategories(): string[] { return categories }

/** Replaces the full player dataset. Call after a full evaluate (draft/auction/season).
 *  Also updates candidates, since a full evaluate's candidate list = the full pool. */
export function setAllPlayers(p: Player[]): void {
    allPlayers      = p
    allPlayerByName = new Map(p.map(pl => [pl.name, pl]))
    candidates      = p
}

/** Replaces only the candidate list. Call after a partial evaluate (e.g. waiver wire)
 *  where allPlayers should not change. */
export function setCandidates(p: Player[]): void {
    candidates = p
}

/** Returns the name → PlayerGScore map (raw G-scores from the pipeline). */
export function getGScoreByName(): Map<string, PlayerGScore> { return gScoreByName }

/** Replaces the G-score map. Called by session.ts after session creation. */
export function setGScores(scores: PlayerGScore[]): void {
    gScoreByName = new Map(scores.map(s => [s.name, s]))
}

/** Replaces the category list. Called by session.ts when categories change. */
export function setCategories(c: string[]): void { categories = c }

// ── Sport config (from GET /config/{sport}) ──────────────────────────────────

let sportConfig: SportConfig | null = null

/** Returns the current sport config, or null if not yet loaded. */
export function getSportConfig(): SportConfig | null { return sportConfig }

/** Stores the sport config fetched from the backend. */
export function setSportConfig(c: SportConfig): void { sportConfig = c }
