// api/season_session.ts
// Season mode API: trade analysis, trade suggestions, waiver wire evaluate, and season init.
// Trade suggestions show no indicator of their own: the one beside the Season Mode tabs covers every request.
// Directly manages #eval-indicator for waiver wire (no autopilot protection needed in season mode).

import { setCandidatePlayerResults, setPlayerResultsFromGScores } from '../app_state.js'
import { buildTable } from '../table/player_table.js'
import {
    ensureSession, getSessionId,
    withDisplayOwnership, withSessionRetry,
} from './session.js'
import { evaluate, candidatesToPlayerResults, analyzeTrade, suggestTrades, fetchDraftState } from './client.js'
import type { TradeAnalyzeResponse, TradeSuggestResponse, TradeComboParam } from './client.js'

// ─── Live-platform season rosters ──────────────────────────────────────────────
// When a live platform is connected in Season Mode, the roster grid is prefilled
// from the platform (without one it starts empty). The rosters are fetched async
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
    if (resp.candidates.length === 0) return null //ZR: Is there any reasonable situation where this gets triggered? 
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
    // The build can take seconds; if the user has since left Season Mode, a newer actor
    // owns the display and the 'idle' is skipped rather than stamped over its state. busy is
    // omitted — the mode-change flow has already set its own spinner state — and onFailure is
    // omitted to preserve the pre-wrapper behaviour of writing nothing on a failed init.
    await withDisplayOwnership({ onSuccess: 'idle' }, () => ensureSession())
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
  , checkPositions?: boolean
): Promise<TradeAnalyzeResponse> {
    return await withSessionRetry(() => analyzeTrade(getSessionId()!, {
        player_assignments:    playerAssignments,
        my_team:               myTeam,
        their_team:            theirTeam,
        my_trade:              myTrade,
        their_trade:           theirTrade,
        position_check: checkPositions,
    }))
}

// ─── Trade suggestions ───────────────────────────────────────────────────────

/**
 * Generates trade suggestions between your team and each of `theirTeams`, ranked together by your score.
 */
export async function runTradeSuggest(
    playerAssignments: Record<string, number[]>
  , myTeam: string
  , theirTeams: string[]
  , comboParams: TradeComboParam[]
  , yourThreshold: number
  , theirThreshold: number
  , checkPositions?: boolean
): Promise<TradeSuggestResponse> {
    return await withSessionRetry(() => suggestTrades(getSessionId()!, {
        player_assignments:           playerAssignments,
        my_team:                      myTeam,
        their_teams:                  theirTeams,
        combo_params:                 comboParams,
        your_differential_threshold:  yourThreshold,
        their_differential_threshold: theirThreshold,
        position_check:               checkPositions,
    }))
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
    await withDisplayOwnership(
        { busy: 'evaluating', onSuccess: 'idle', onFailure: 'idle' }
        , stillOwner => withSessionRetry(async () => {
            const resp = await evaluate(getSessionId()!, { player_assignments: playerAssignments, my_team_id: myTeamId })
            // No abort signal cancels this evaluate; by the time it resolves, a newer actor
            // — another waiver check, or a mode switch into a draft evaluate — may own the
            // display, and season candidates must not repaint that actor's board.
            if (stillOwner()) {
                const players = candidatesToPlayerResults(resp.candidates)
                setCandidatePlayerResults(players)
                buildTable(players)
            }
        })
    )
}
