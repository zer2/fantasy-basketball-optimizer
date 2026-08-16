// api/season_session.ts
// Season mode API: trade analysis, trade suggestions, waiver wire evaluate, and season init.
// Owns #ts-suggest-indicator for trade suggestions.
// Directly manages #eval-indicator for waiver wire (no autopilot protection needed in season mode).

import { setCandidatePlayerResults, setPlayerResultsFromGScores } from '../app_state.js'
import { buildTable } from '../table/player_table.js'
import { ensureSession, getSessionId, setIndicatorState, applyIndicatorState, withSessionRetry } from './session.js'
import { evaluate, candidatesToPlayerResults, analyzeTrade, suggestTrades, fetchDraftState } from './client.js'
import type { TradeAnalyzeResponse, TradeSuggestResponse } from './client.js'

// ─── Live-platform season rosters ──────────────────────────────────────────────
// When a live platform is connected in Season Mode, the roster grid is prefilled
// from the platform instead of DEFAULT_SEASON_ROSTERS. The rosters are fetched async
// (draft-state needs a session), cached here, and read by renderSeasonRosters as its
// prefill source; the caller re-renders (applyLayout) once they arrive. Living here
// (not in season_rosters) keeps the import one-way: season_rosters → season_session.

let livePlatformRosters: Record<string, number[]> | null = null

/** Platform-provided season rosters ({team: [player ids]}), or null when not loaded
 *  (own data, or before the first platform refresh). */
export function getLivePlatformRosters(): Record<string, number[]> | null {
    return livePlatformRosters
}

export function clearLivePlatformRosters(): void {
    livePlatformRosters = null
}

/** Pull the connected platform's current rosters for Season Mode and cache them.
 *  The caller re-renders the season layout so renderSeasonRosters picks them up. */
export async function refreshSeasonRostersFromPlatform(): Promise<void> {
    await withSessionRetry(async () => {
        const state = await fetchDraftState(getSessionId()!, 'Season Mode')
        livePlatformRosters = state.player_assignments
    })
}

// ─── Team H-score ────────────────────────────────────────────────────────────

/**
 * Evaluates the full-team H-score for a specific team given roster assignments.
 * Returns h_score and per-category win rates, or null if the backend returns no candidates.
 */
export async function evaluateTeamHScore(
    playerAssignments: Record<string, number[]>
  , teamId: string
): Promise<{ h_score: number; win_rates: number[] } | null> {
    await ensureSession()
    const resp = await evaluate(getSessionId()!, { player_assignments: playerAssignments, my_team_id: teamId })
    if (resp.candidates.length === 0) return null
    return {
        h_score:   resp.candidates[0].h_score,
        win_rates: resp.candidates[0].win_rates,
    }
}

// ─── Season init ─────────────────────────────────────────────────────────────

/**
 * Initialises Season Mode: ensures a session exists so G-scores are available,
 * then builds minimal Player objects for the roster dropdowns.
 */
export async function runSeasonInit(): Promise<void> {
    await ensureSession()
    setIndicatorState('idle')
    setPlayerResultsFromGScores()
}

// ─── Trade analysis ──────────────────────────────────────────────────────────

/**
 * Analyzes a proposed trade, returning pre/post H-scores for both teams.
 * Ensures a session exists before calling the backend.
 */
export async function runTradeAnalyze(
    playerAssignments: Record<string, number[]>
  , myTeam: string
  , theirTeam: string
  , myTrade: number[]
  , theirTrade: number[]
  , ignorePositionCheck?: boolean
): Promise<TradeAnalyzeResponse> {
    return await withSessionRetry(() => analyzeTrade(getSessionId()!, {
        player_assignments:    playerAssignments,
        my_team:               myTeam,
        their_team:            theirTeam,
        my_trade:              myTrade,
        their_trade:           theirTrade,
        ignore_position_check: ignorePositionCheck,
    }))
}

// ─── Trade suggestions ───────────────────────────────────────────────────────

/**
 * Generates trade suggestions for two teams.
 * Updates #ts-suggest-indicator: "Starting..." while waiting for a session,
 * "Updating..." once the session is ready and computation begins.
 * Hiding the indicator when all fetches complete is the caller's responsibility.
 */
export async function runTradeSuggest(
    playerAssignments: Record<string, number[]>
  , myTeam: string
  , theirTeam: string
  , comboParams: { n_traded: number; n_received: number; threshold: number }[]
  , yourThreshold: number
  , theirThreshold: number
  , ignorePositionCheck?: boolean
): Promise<TradeSuggestResponse> {
    return await withSessionRetry(
        async () => {
            applyIndicatorState('suggest-indicator', 'evaluating')
            return await suggestTrades(getSessionId()!, {
                player_assignments:           playerAssignments,
                my_team:                      myTeam,
                their_team:                   theirTeam,
                combo_params:                 comboParams,
                your_differential_threshold:  yourThreshold,
                their_differential_threshold: theirThreshold,
                ignore_position_check:        ignorePositionCheck,
            })
        }
        , () => applyIndicatorState('suggest-indicator', 'fetching')
    )
}

// ─── Waiver wire ─────────────────────────────────────────────────────────────

/**
 * Runs evaluate for waiver wire analysis with a caller-supplied roster state.
 * Unlike runEvaluate, the caller is responsible for passing modified player_assignments
 * (with the dropped player removed from their team) so they appear as a free-agent candidate.
 */
export async function runWaiverEvaluate(
    playerAssignments: Record<string, number[]>
  , myTeamId: string
): Promise<void> {
    setIndicatorState('evaluating')
    try {
        await withSessionRetry(async () => {
            const resp = await evaluate(getSessionId()!, { player_assignments: playerAssignments, my_team_id: myTeamId })
            const players = candidatesToPlayerResults(resp.candidates)
            setCandidatePlayerResults(players)
            buildTable(players)
        })
    } finally {
        setIndicatorState('idle')
    }
}
