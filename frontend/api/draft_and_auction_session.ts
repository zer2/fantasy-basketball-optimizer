// api/draft_and_auction_session.ts
// Draft and auction mode: session management, evaluate orchestration, and indicator state.
// Owns the #eval-indicator element and all draft/auction-specific session state.

import { PlayerResult } from '../types.js'
import { setBasePlayerResults, setCandidatePlayerResults, getCandidatePlayerResults, setGScores, getCurrentSeat } from '../app_state.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getDraftState } from '../data_entry/draft_state.js'
import { getAuctionState } from '../data_entry/auction_state.js'
import { buildTable, showTableMessage } from '../table/player_table.js'
import { startFreshSession, ensureSession, getSessionId, resetSession, setIndicatorState } from './session.js'
import { patchSession, fetchGScores, evaluate, candidatesToPlayerResults } from './client.js'

// ─── Draft/auction state ─────────────────────────────────────────────────────

const basePlayersBySession: Map<string, PlayerResult[]> = new Map()
let evaluateController: AbortController | null = null
let evaluateGeneration = 0
let latestFullTeamResult: { h_score: number; win_rates: number[] } | null = null

export function getFullTeamResult(): { h_score: number; win_rates: number[] } | null {
    return latestFullTeamResult
}

export function clearFullTeamResult(): void {
    latestFullTeamResult = null
}

// ─── Session management ──────────────────────────────────────────────────────

/**
 * Creates a new session if none exists, or patches the existing one starting from
 * `fromStep` with the given partial parameter body.
 * If the session has expired (404), creates a fresh one from current sidebar state.
 */
export async function createOrPatchSession(
    fromStep: number
  , patchBody: Record<string, unknown> = {}
  , signal?: AbortSignal
): Promise<void> {
    setIndicatorState('fetching')
    if (!getSessionId()) {
        await startFreshSession(signal)
        return
    }
    try {
        const patchResp = await patchSession(getSessionId()!, { from_step: fromStep, ...patchBody }, signal)
        basePlayersBySession.delete(getSessionId()!)
        latestFullTeamResult = null
        if (patchResp.steps_rerun.includes(4)) {
            const freshGScores = await fetchGScores(getSessionId()!)
            setGScores(freshGScores)
        }
    } catch (err: any) {
        if (!err.message?.includes('(404)')) throw err
        resetSession()
        await startFreshSession(signal)
    }
}

// ─── Evaluate ────────────────────────────────────────────────────────────────

/**
 * Runs the evaluate endpoint for the given seat, updates candidates,
 * base players cache, and full-team result.
 * Retries once if the session has expired (404).
 */
async function evaluateSeat(seat: string): Promise<void> {
    if (evaluateController) evaluateController.abort()
    evaluateController = new AbortController()
    const { signal } = evaluateController
    const generation = ++evaluateGeneration
    setIndicatorState('evaluating')

    try {
        for (let attempt = 0; attempt < 2; attempt++) {
            try {
                await ensureSession()
                setIndicatorState('evaluating')

                const mode = (document.getElementById('ls-mode') as HTMLInputElement).value

                let evalReq: Parameters<typeof evaluate>[1]
                if (mode === 'Auction Mode') {
                    const { player_assignments, remaining_cash } = getAuctionState()
                    evalReq = { player_assignments, my_team_id: seat, remaining_cash }
                } else {
                    const { player_assignments } = getDraftState()
                    evalReq = { player_assignments, my_team_id: seat }
                }

                const boardIsEmpty = Object.values(evalReq.player_assignments).flat().length === 0
                if (!basePlayersBySession.has(getSessionId()!) && !boardIsEmpty) {
                    const { n_drafters } = getLeagueSettings()
                    const genericTeams   = Array.from({ length: n_drafters }, (_, i) => `Team ${i + 1}`)
                    const emptyAssignments: Record<string, string[]> = Object.fromEntries(
                        genericTeams.map(name => [name, []])
                    )
                    const baseResp = await evaluate(
                        getSessionId()!
                    ,   { player_assignments: emptyAssignments, my_team_id: genericTeams[0] }
                    ,   signal
                    )
                    basePlayersBySession.set(getSessionId()!, candidatesToPlayerResults(baseResp.candidates))
                }

                const myTeamSize = (evalReq.player_assignments[seat] ?? []).length
                if (myTeamSize >= getLeagueSettings().n_picks) {
                    latestFullTeamResult = null
                    const fullTeamResp = await evaluate(getSessionId()!, evalReq, signal)
                    if (fullTeamResp.candidates.length > 0) {
                        latestFullTeamResult = {
                            h_score:   fullTeamResp.candidates[0].h_score,
                            win_rates: fullTeamResp.candidates[0].win_rates,
                        }
                        document.dispatchEvent(new Event('full-team-result-updated'))
                    }
                    return
                }
                latestFullTeamResult = null

                const resp = await evaluate(getSessionId()!, evalReq, signal)
                const players = candidatesToPlayerResults(resp.candidates)

                if (!basePlayersBySession.has(getSessionId()!)) {
                    basePlayersBySession.set(getSessionId()!, players)
                }

                setBasePlayerResults(basePlayersBySession.get(getSessionId()!)!)
                setCandidatePlayerResults(players)
                return
            } catch (err: any) {
                if (err.name === 'AbortError') return
                if (attempt === 0 && err.message?.includes('(404)')) {
                    resetSession()
                    continue
                }
                throw err
            }
        }
    } finally {
        if (evaluateGeneration === generation) setIndicatorState('idle')
    }
}

/** Evaluates the current draft/auction state for the current seat and rebuilds the candidate table. */
export async function runEvaluate(): Promise<void> {
    await evaluateSeat(getCurrentSeat() ?? getLeagueSettings().team_names[0] ?? 'Drafter 1')
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
    if (mode !== 'Season Mode') {
        if (getFullTeamResult()) {
            showTableMessage('Your team is full.')
        } else {
            buildTable(getCandidatePlayerResults())
        }
    }
}
