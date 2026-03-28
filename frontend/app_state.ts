// app_state.ts
// Shared application state: the current player list and category names.
// Kept in a standalone module with no app-internal imports (only types.ts)
// so that any module can read the state without creating circular dependencies.

import { Player, PlayerGScore, SportConfig } from './types.js'

let allPlayers:      Player[] = []   // full dataset — only grows, never shrinks
let candidates:      Player[] = []   // current evaluate output (may be a subset, e.g. waiver free agents)
let allPlayerByName: Map<string, Player> = new Map()
let gScoreByName:    Map<string, PlayerGScore> = new Map()
let playerNamesByGScore: string[] = []   // player names sorted descending by G-score total
/** Returns every player in the loaded dataset, regardless of draft/waiver state. */
export function getPlayers(): Player[] { return allPlayers }

/** Returns the current list of ranked candidate players displayed in the table. */
export function getCandidatePlayers(): Player[] { return candidates }

/** Returns a name → Player map for every player in the full dataset. */
export function getPlayerByName(): Map<string, Player> { return allPlayerByName }

/** Replaces the full player dataset. Call after a full evaluate (draft/auction/season).
 *  Also updates candidates, since a full evaluate's candidate list = the full pool. */
export function setAllPlayers(p: Player[]): void {
    allPlayers      = p
    allPlayerByName = new Map(p.map(pl => [pl.name, pl]))
    candidates      = p
}

/** Replaces only the base player list (allPlayers + name map) without touching candidates.
 *  Call with the empty-board evaluate result so the draft dropdown always shows base H-score ordering. */
export function setBasePlayers(p: Player[]): void {
    allPlayers      = p
    allPlayerByName = new Map(p.map(pl => [pl.name, pl]))
}

/** Replaces only the candidate list. Call after a partial evaluate (e.g. waiver wire)
 *  where allPlayers should not change. */
export function setCandidates(p: Player[]): void {
    candidates = p
}

/** Returns the name → PlayerGScore map (raw G-scores from the pipeline). */
export function getGScoreByName(): Map<string, PlayerGScore> { return gScoreByName }

/** Returns player names sorted descending by G-score total (pre-computed at session creation). */
export function getPlayerNamesByGScore(): string[] { return playerNamesByGScore }

/** Replaces the G-score map. Called by session.ts after session creation.
 *  `scores` arrives from the backend already sorted by total descending. */
export function setGScores(scores: PlayerGScore[]): void {
    gScoreByName        = new Map(scores.map(s => [s.name, s]))
    playerNamesByGScore = scores.map(s => s.name)
}

/**
 * Builds minimal Player objects from the G-score map so that Season Mode
 * has player names and G-ranks available for roster dropdowns, without
 * needing a full evaluate call.
 */
export function setPlayersFromGScores(): void {
    const scores = [...gScoreByName.values()].sort((a, b) => b.total - a.total)
    allPlayers = scores.map((gs, i) => ({
        name: gs.name,
        h_score: 0,
        h_rank: 0,
        g_rank: i + 1,
        win_rates: [],
        category_weights: [],
        g_score_rows: [],
    }))
    allPlayerByName = new Map(allPlayers.map(p => [p.name, p]))
    candidates = allPlayers
}

// ── Seat selector state ───────────────────────────────────────────────────────

let currentSeat: string | null = null

/** Returns the currently selected seat (team name), or null if none selected. */
export function getCurrentSeat(): string | null { return currentSeat }

/** Sets the currently selected seat (team name). Dispatches 'seat-changed' so layout can refresh team stats. */
export function setCurrentSeat(seat: string | null): void {
    currentSeat = seat
    document.dispatchEvent(new Event('seat-changed'))
}

// ── Sport config (from GET /config/{sport}) ──────────────────────────────────

let sportConfig: SportConfig | null = null

/** Returns the current sport config, or null if not yet loaded. */
export function getSportConfig(): SportConfig | null { return sportConfig }

/** Stores the sport config fetched from the backend. */
export function setSportConfig(c: SportConfig): void { sportConfig = c }

/** Returns the position abbreviation → full name map (e.g. "PG" → "Point Guard"). */
export function getPositionNames(): Record<string, string> { return sportConfig?.position_names ?? {} }
