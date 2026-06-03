// app_state.ts
// Shared application state: the current player list and category names.
// Kept in a standalone module with no app-internal imports (only types.ts)
// so that any module can read the state without creating circular dependencies.

import { PlayerResult, PlayerGScore, SportConfig } from './types.js'

// ── Session phase ─────────────────────────────────────────────────────────────

export type SessionPhase = 'uninitialized' | 'session-created' | 'evaluated'

let sessionPhase: SessionPhase = 'uninitialized'

/** Returns how far the backend initialization has progressed. */
export function getSessionPhase(): SessionPhase { return sessionPhase }

// ── Player data ───────────────────────────────────────────────────────────────

let allPlayerResults:        PlayerResult[] | null = null   // full dataset — only grows, never shrinks
let candidates:              PlayerResult[] | null = null   // current evaluate output (may be a subset, e.g. waiver free agents)
let playerResultsByName:     Map<string, PlayerResult> | null = null
let gScoreByName:        Map<string, PlayerGScore> | null = null
let playerNamesByGScore: string[] | null = null   // player names sorted descending by G-score total

/** Returns every player in the loaded dataset, or null if no evaluate has run yet. */
export function getPlayerResults(): PlayerResult[] | null { return allPlayerResults }

/** Returns the current candidate player list, or null if no evaluate has run yet. */
export function getCandidatePlayerResults(): PlayerResult[] | null { return candidates }

/** Returns a name → Player map for every player in the full dataset. */
export function getPlayerResultsByName(): Map<string, PlayerResult> {
    if (playerResultsByName === null) throw new Error('getPlayerResultsByName called before player data was loaded')
    else return playerResultsByName
}

/** Replaces the full player dataset. Call after a full evaluate (draft/auction/season).
 *  Also updates candidates, since a full evaluate's candidate list = the full pool. */
export function setAllPlayerResults(p: PlayerResult[]): void {
    setBasePlayerResults(p)
    setCandidatePlayerResults(p)
}

/** Replaces only the base player list (allPlayerResults + name map) without touching candidates.
 *  Call with the empty-board evaluate result so the draft dropdown always shows base H-score ordering. */
export function setBasePlayerResults(p: PlayerResult[]): void {
    allPlayerResults    = p
    playerResultsByName = new Map(p.map(pl => [pl.name, pl]))
}

/** Replaces only the candidate list. Call after a partial evaluate (e.g. waiver wire)
 *  where allPlayerResults should not change. */
export function setCandidatePlayerResults(p: PlayerResult[]): void {
    candidates   = p
    sessionPhase = 'evaluated'
}

/** Returns the name → PlayerGScore map (raw G-scores from the pipeline). */
export function getGScoreByName(): Map<string, PlayerGScore> {
    if (gScoreByName === null) throw new Error('getGScoreByName called before G-scores were loaded')
    else return gScoreByName
}

/** Returns player names sorted descending by G-score total (pre-computed at session creation). */
export function getPlayerNamesByGScore(): string[] {
    if (playerNamesByGScore === null) throw new Error('getPlayerNamesByGScore called before G-scores were loaded')
    else return playerNamesByGScore
}

/** Replaces the G-score map. Called by session.ts after session creation.
 *  `scores` arrives from the backend already sorted by total descending. */
export function setGScores(scores: PlayerGScore[]): void {
    gScoreByName        = new Map(scores.map(s => [s.name, s]))
    playerNamesByGScore = scores.map(s => s.name)
    if (sessionPhase === 'uninitialized') sessionPhase = 'session-created'
}

/**
 * Populates the player list from G-scores alone, for use in Season Mode before
 * any evaluate has run. The resulting PlayerResult objects contain only name and
 * g_rank — all other fields are zeroed/empty. H-score fields are meaningless in
 * this context; Season Mode roster dropdowns only use name and g_rank.
 */
export function setPlayerResultsFromGScores(): void {
    const scores  = [...getGScoreByName().values()].sort((a, b) => b.total - a.total)
    const players = scores.map((gs, i) => ({
        name: gs.name,
        h_score: 0,
        h_rank: 0,
        g_rank: i + 1,
        win_rates: [],
        category_weights: [],
        g_score_rows: [],
    }))
    setAllPlayerResults(players)
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

/** Returns the full category name → short label map (e.g. "Points" → "pts").
 *  Used by the candidate table on mobile to keep column headers compact. */
export function getShortCategoryNames(): Record<string, string> { return sportConfig?.short_category_names ?? {} }
