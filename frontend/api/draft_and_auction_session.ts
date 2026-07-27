// api/draft_and_auction_session.ts
// Draft and auction mode: session management, evaluate orchestration, and indicator state.
// Owns the #eval-indicator element and all draft/auction-specific session state.

import { PlayerResult } from '../types.js'
import { setBasePlayerResults, setCandidatePlayerResults, getCandidatePlayerResults, setGScores, getCurrentSeat } from '../app_state.js'
import { getLeagueSettings, getPlatformConfig } from '../parameter_collection/league_settings.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { getDraftState } from '../data_entry/draft_state.js'
import { getAuctionState } from '../data_entry/auction_state.js'
import { buildTable, resetTable, addBatch, showTableMessage, reserveTailSpace, clearTailSpace } from '../table/player_table.js'

// Draft/waiver candidate batch size: score + paint the top players first, then fill in the
// bench in follow-up requests. Auction is never batched (its $ values need the whole pool).
const CANDIDATE_BATCH_SIZE = 100
import { startFreshSession, getSessionId, resetSession, setIndicatorState, withSessionRetry } from './session.js'
import { patchSession, fetchGScores, evaluate, fetchDraftState, candidatesToPlayerResults, HTTPError } from './client.js'

// ─── Draft/auction state ─────────────────────────────────────────────────────

const basePlayersBySession: Map<string, PlayerResult[]> = new Map()
let evaluateController: AbortController | null = null
let evaluateGeneration = 0
let latestFullTeamResult: { h_score: number; win_rates: number[] } | null = null

// Player assignments pulled from a live platform (Refresh Analysis). When set,
// evaluateSeat uses these instead of reading the manual draft/auction board,
// which does not exist in the live-platform layout.
let livePlayerAssignments: Record<string, string[]> | null = null
let liveRemainingCash: Record<string, number> | null = null

export function setLivePlayerAssignments(
    assignments: Record<string, string[]>
    , remainingCash?: Record<string, number>
): void {
    livePlayerAssignments = assignments
    liveRemainingCash = remainingCash ?? null
}

export function clearLivePlayerAssignments(): void {
    livePlayerAssignments = null
    liveRemainingCash = null
}

export function getFullTeamResult(): { h_score: number; win_rates: number[] } | null {
    return latestFullTeamResult
}

/** Per-team full budgets — the remaining cash of an auction board with no picks. Auction sessions
 *  require remaining_cash on every evaluate, including the empty-board base evaluation. */
function buildFullBudgets(teamNames: string[]): Record<string, number> {
    const { cash_per_team } = getLeagueSettings()
    return Object.fromEntries(teamNames.map(name => [name, cash_per_team]))
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
    } catch (err) {
        if (!(err instanceof HTTPError) || err.status !== 404) throw err
        resetSession()
        await startFreshSession(signal)
    }
}

// When a live platform connects, patch the session so it carries the platform's config
// (which drives the draft-state poll + name lookup) and the platform's drafter/pick counts.
// No recreate needed — platform_config feeds nothing the pipeline computes. Driven by an
// event so league_settings doesn't import this module (it imports league_settings — a cycle).
document.addEventListener('platform-connected', () => {
    const { platform, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
    createOrPatchSession(4, {
        league: { n_drafters, n_picks, cash_per_team },
        slot_counts: getSlotCounts(),
        platform,
        platform_config: getPlatformConfig(),
    }).catch(err => console.error('Platform connect patch failed:', err))
})

// ─── Evaluate ────────────────────────────────────────────────────────────────

/**
 * Runs the evaluate endpoint for the given seat, updates candidates,
 * base players cache, and full-team result.
 * Retries once if the session has expired (404).
 */
async function evaluateSeat(seat: string, forAutopilot = false): Promise<string | null> {
    if (evaluateController) evaluateController.abort()
    evaluateController = new AbortController()
    const { signal } = evaluateController
    const generation = ++evaluateGeneration
    setIndicatorState('evaluating')

    try {
        return await withSessionRetry(async () => {
            const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
            const isLivePlatform = getLeagueSettings().platform !== 'Enter your own data'

            let evalReq: Parameters<typeof evaluate>[1]
            if (isLivePlatform) {
                // Live platforms supply assignments (and, for auctions, remaining cash)
                // via the Refresh Analysis poll instead of a manual board.
                if (livePlayerAssignments === null) {
                    throw new Error('No live draft state loaded; click Refresh Analysis first')
                }
                evalReq = (mode === 'Auction Mode')
                    ? { player_assignments: livePlayerAssignments, my_team_id: seat, remaining_cash: liveRemainingCash ?? undefined }
                    : { player_assignments: livePlayerAssignments, my_team_id: seat }
            } else if (mode === 'Auction Mode') {
                const { player_assignments, remaining_cash } = getAuctionState()
                evalReq = { player_assignments, my_team_id: seat, remaining_cash }
            } else {
                const { player_assignments } = getDraftState()
                evalReq = { player_assignments, my_team_id: seat }
            }

            // Autopilot never renders the board, so it needs neither the base-player comparison nor
            // a full base evaluation to establish it — skip that work entirely.
            const boardIsEmpty = Object.values(evalReq.player_assignments).flat().length === 0
            if (!forAutopilot && !basePlayersBySession.has(getSessionId()!) && !boardIsEmpty) {
                const { n_drafters } = getLeagueSettings()
                const genericTeams   = Array.from({ length: n_drafters }, (_, i) => `Team ${i + 1}`)
                const emptyAssignments: Record<string, string[]> = Object.fromEntries(
                    genericTeams.map(name => [name, []])
                )
                // An auction session requires remaining_cash on every evaluate; the empty board
                // means every team still holds its full budget.
                const baseEvalRequest: Parameters<typeof evaluate>[1] =
                    { player_assignments: emptyAssignments, my_team_id: genericTeams[0] }
                if (mode === 'Auction Mode') baseEvalRequest.remaining_cash = buildFullBudgets(genericTeams)
                const baseResp = await evaluate(getSessionId()!, baseEvalRequest, signal)
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
                return null
            }
            latestFullTeamResult = null

            let players: PlayerResult[]
            if (mode === 'Auction Mode') {
                // Auction scores the whole pool in one call (dollar values anchor on the full
                // distribution). Rendered by runEvaluate, as before.
                const resp = await evaluate(getSessionId()!, evalReq, signal)
                players = candidatesToPlayerResults(resp.candidates)
            } else {
                // Draft: score + paint in batches (top-ranked first) so the top of the board appears
                // before the deep bench is scored. Each batch is merged into the table incrementally.
                // Autopilot only needs the single top pick and never shows the board, so it scores just
                // the first batch and skips all rendering.
                players = []
                for (let offset = 0, first = true; ; offset += CANDIDATE_BATCH_SIZE, first = false) {
                    const resp = await evaluate(
                        getSessionId()!,
                        { ...evalReq, candidate_offset: offset, candidate_limit: CANDIDATE_BATCH_SIZE },
                        signal,
                    )
                    if (signal.aborted) return null
                    const batch = candidatesToPlayerResults(resp.candidates)
                    players.push(...batch)
                    if (forAutopilot) break   // the top pick is in the first batch; the rest is wasted work
                    if (first) resetTable()
                    addBatch(batch)
                    // The top of the board is what users look at; once the first batch is painted, drop
                    // the spinner even though the deep bench is still scoring. The remaining batches merge
                    // in silently — by the time anyone scrolls past the first ~100, they've arrived.
                    if (first) setIndicatorState('idle')
                    // Reserve whitespace for the not-yet-scored candidates so the scrollbar stays put as
                    // later batches fill in; drop it once the last batch has arrived.
                    if (resp.has_more) reserveTailSpace(resp.total_candidates ?? 0)
                    else clearTailSpace()
                    if (!resp.has_more) break
                }
            }

            if (forAutopilot) {
                // Return the top candidate for the pick decision. The caller uses this return value
                // rather than a shared global, so a later evaluate that aborts this one can't leave a
                // stale pick behind. Don't cache this partial first-batch-only list as the base players.
                setCandidatePlayerResults(players)
                return players[0]?.name ?? null
            }

            if (!basePlayersBySession.has(getSessionId()!)) {
                basePlayersBySession.set(getSessionId()!, players)
            }

            setBasePlayerResults(basePlayersBySession.get(getSessionId()!)!)
            setCandidatePlayerResults(players)
            return null
        })
    } catch (err: any) {
        if (err.name === 'AbortError') return null
        throw err
    } finally {
        if (evaluateGeneration === generation) setIndicatorState('idle')
    }
}

/** Evaluates the current draft/auction state for the current seat and rebuilds the candidate table.
 *  With `forAutopilot`, it scores only the first batch and renders nothing — the caller only needs the
 *  top candidate for an autopilot pick. */
export async function runEvaluate(options: { forAutopilot?: boolean } = {}): Promise<string | null> {
    const forAutopilot = options.forAutopilot ?? false
    // No explicit seat selected falls back to the first team; an empty league is a bug, not a default.
    const seat = getCurrentSeat() ?? getLeagueSettings().team_names[0]
    if (seat === undefined) throw new Error('runEvaluate: no seat selected and league has no team names')
    const topPick = await evaluateSeat(seat, forAutopilot)
    if (forAutopilot) return topPick   // autopilot needs only the top candidate; nothing is shown
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
    if (mode !== 'Season Mode') {
        if (getFullTeamResult()) {
            showTableMessage('Your team is full.')
        } else if (mode === 'Auction Mode') {
            // Draft renders incrementally inside evaluateSeat; only auction renders in one shot here.
            buildTable(getCandidatePlayerResults()!)
        }
    }
    return null
}

/**
 * Live-platform refresh: polls the connected platform for the current draft /
 * roster state, stores it as the live player assignments, then re-evaluates.
 * Backs the "Refresh Analysis" button in the live-platform layout.
 */
/**
 * Before a live platform is connected, show the base player rankings (everyone vs.
 * empty teams) so the user still sees default rankings pre-auth. Uses generic teams
 * for the evaluation, so it doesn't depend on a selected seat.
 */
export async function showDefaultRankings(): Promise<void> {
    // Cancel any in-flight per-seat evaluate and invalidate its generation, so its finally
    // block can't flip the indicator back to 'idle' after we settle on 'unconnected'.
    if (evaluateController) evaluateController.abort()
    evaluateGeneration++
    setIndicatorState('fetching')
    try {
        await withSessionRetry(async () => {
            const { n_drafters, mode } = getLeagueSettings()
            const genericTeams = Array.from({ length: n_drafters }, (_, i) => `Team ${i + 1}`)
            const emptyAssignments: Record<string, string[]> = Object.fromEntries(
                genericTeams.map(name => [name, []])
            )
            // An auction session requires remaining_cash on every evaluate; with no picks made,
            // every team still holds its full budget.
            const evalRequest: Parameters<typeof evaluate>[1] =
                { player_assignments: emptyAssignments, my_team_id: genericTeams[0] }
            if (mode === 'Auction Mode') evalRequest.remaining_cash = buildFullBudgets(genericTeams)
            const resp = await evaluate(getSessionId()!, evalRequest)
            const players = candidatesToPlayerResults(resp.candidates)
            setBasePlayerResults(players)
            setCandidatePlayerResults(players)
        })
        buildTable(getCandidatePlayerResults()!)
    } finally {
        setIndicatorState('unconnected')
    }
}

export async function refreshLiveAnalysis(): Promise<void> {
    setIndicatorState('fetching')
    try {
        await withSessionRetry(async () => {
            const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
            const state = await fetchDraftState(getSessionId()!, mode)
            setLivePlayerAssignments(state.player_assignments, state.remaining_cash)
        })
        await runEvaluate()
    } catch (err) {
        setIndicatorState('idle')   // never leave the spinner stuck on a failed poll
        throw err
    }
}
