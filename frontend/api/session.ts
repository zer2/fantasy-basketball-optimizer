// api/session.ts
// Session lifecycle management and evaluate orchestration.
// Owns the session ID, abort controller, and the updateTable bridge between
// backend responses and the player table.

import { Player, SessionRequest } from '../types.js'
import { setBasePlayers, setCandidates, getCandidatePlayers, setGScores, setPlayersFromGScores, getCurrentSeat } from '../app_state.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getScoringFormat, getSelectedCategories } from '../parameter_collection/format_and_categories.js'
import { getPlayerStatsParams } from '../parameter_collection/player_stats.js'
import { getModelParameters } from '../parameter_collection/model_parameters.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { getDraftState } from '../data_entry/draft_state.js'
import { getAuctionState } from '../data_entry/auction_state.js'
import { buildTable, showTableMessage } from '../table/player_table.js'
import * as client from './client.js'

// ─── Local UI helpers ─────────────────────────────────────────────────────────
// Direct DOM access for the eval indicator, avoiding a circular dependency with
// layout.ts (which imports runEvaluate from this module).

type IndicatorState = 'idle' | 'fetching' | 'evaluating' | 'autopiloting'

let currentIndicatorState: IndicatorState = 'idle'

function applyIndicatorDisplay(state: IndicatorState): void {
    const el = document.getElementById('eval-indicator') as HTMLElement
    el.classList.toggle('evaluating', state === 'evaluating')
    el.classList.toggle('fetching', state === 'fetching')
    el.textContent = state === 'fetching'     ? 'Fetching...'
                   : state === 'evaluating'   ? 'Updating...'
                   : state === 'autopiloting' ? 'Autopiloting...'
                   :                           'Updated'
}

/** Sets the indicator state. Suppressed while autopiloting — the autopilot indicator owns the display. */
function setIndicatorState(state: 'idle' | 'fetching' | 'evaluating'): void {
    if (currentIndicatorState === 'autopiloting') return
    currentIndicatorState = state
    applyIndicatorDisplay(state)
}

/** Enters or exits autopilot indicator mode. Bypasses normal suppression. */
export function setAutopilotIndicator(active: boolean): void {
    currentIndicatorState = active ? 'autopiloting' : 'idle'
    applyIndicatorDisplay(currentIndicatorState)
}

// ─── Module state ────────────────────────────────────────────────────────────

let sessionId: string | null = null
// Tracks the in-flight evaluate request so it can be aborted when a newer one starts.
let evaluateController: AbortController | null = null
// Incremented on each new evaluate call; lets the finally block avoid hiding the
// indicator when a newer call has already shown it.
let evaluateGeneration = 0
// Caches the empty-board evaluate result per session ID so the draft dropdown always
// shows base H-score ordering regardless of how many picks have been made.
// Keyed by session ID; invalidated whenever the session is patched (parameters changed).
const basePlayersBySession: Map<string, Player[]> = new Map()
// Stores the H-score result from the most recent full-team evaluate call.
// Set when my team is complete; null at all other times.
let latestFullTeamResult: { h_score: number; win_rates: number[] } | null = null

export function getFullTeamResult(): { h_score: number; win_rates: number[] } | null {
    return latestFullTeamResult
}

export function clearFullTeamResult(): void {
    latestFullTeamResult = null
}

// ─── Public API ──────────────────────────────────────────────────────────────

// Reads all sidebar parameters, POSTs to /sessions, and stores the returned session ID and G-scores.
// Every code path that needs a new session calls this.
async function startFreshSession(signal?: AbortSignal): Promise<void> {
    const { sport, platform, mode, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
    const scoring_format = getScoringFormat()
    const categories     = getSelectedCategories()
    const { data_source, injured_players } = getPlayerStatsParams()
    const league: SessionRequest['league'] = { sport, n_drafters, n_picks, scoring_format, categories }
    if (mode === 'Auction Mode') league.cash_per_team = cash_per_team
    const req: SessionRequest = {
        league,
        platform,
        slot_counts: getSlotCounts(),
        parameters: getModelParameters(),
        data_source,
        injured_players,
        // my_team_id comes from the seat selector in the main content area
    }
    setIndicatorState('fetching')
    const resp = await client.createSession(req, signal)
    sessionId = resp.session_id
    setGScores(resp.g_scores)
}

/**
 * Creates a new session if none exists, or patches the existing one starting from
 * `fromStep` with the given partial parameter body.
 * If the session has expired (404), creates a fresh one from current sidebar state.
 */
export async function createOrPatchSession(
    fromStep: number,
    patchBody: Record<string, unknown> = {},
    signal?: AbortSignal,
): Promise<void> {
    if (!sessionId) {
        await startFreshSession(signal)
        return
    }
    try {
        await client.patchSession(sessionId, { from_step: fromStep, ...patchBody }, signal)
        // Parameters changed — cached base result is now stale.
        basePlayersBySession.delete(sessionId)
        latestFullTeamResult = null
    } catch (err: any) {
        if (!err.message?.includes('(404)')) throw err
        // Session expired; rebuild from current sidebar state.
        sessionId = null
        await startFreshSession(signal)
    }
}

/**
 * Ensures a session exists (creating one from current sidebar state if not).
 * Called before evaluate when the session may not have been explicitly created yet.
 */
async function ensureSession(): Promise<void> {
    if (sessionId) return
    await startFreshSession()
}

/**
 * Initialises Season Mode: ensures a session exists so G-scores are available,
 * then builds minimal Player objects for the roster dropdowns.
 * Separate from runEvaluate because Season Mode does not run the H-score algorithm.
 */
export async function runSeasonInit(): Promise<void> {
    await ensureSession()
    setIndicatorState('idle')
    setPlayersFromGScores()
}

/**
 * Runs the evaluate endpoint for the given seat and applies the result to app state:
 * updates candidates, base players cache, and full-team result.
 * Does NOT rebuild the candidate table — call runEvaluate() for that.
 * Retries once if the session has expired (404), creating a fresh session first.
 */
async function evaluateSeat(seat: string): Promise<void> {
    // Abort any in-flight evaluate request so the backend is only computing
    // the most recent draft state, not every intermediate pick.
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

                let evalReq: Parameters<typeof client.evaluate>[1]
                if (mode === 'Auction Mode') {
                    const { player_assignments, remaining_cash } = getAuctionState()
                    evalReq = { player_assignments, my_team_id: seat, remaining_cash }
                } else {
                    const { player_assignments } = getDraftState()
                    evalReq = { player_assignments, my_team_id: seat }
                }

                // If the base result is not yet cached for this session, fetch it now.
                // When the board is non-empty this requires a separate empty-board evaluate;
                // when the board is already empty the result below will serve as the base.
                const boardIsEmpty = Object.values(evalReq.player_assignments).flat().length === 0
                if (!basePlayersBySession.has(sessionId!) && !boardIsEmpty) {
                    // Use generic team names so the base evaluate is independent of configured team names.
                    const { n_drafters } = getLeagueSettings()
                    const genericTeams   = Array.from({ length: n_drafters }, (_, i) => `Team ${i + 1}`)
                    const emptyAssignments: Record<string, string[]> = Object.fromEntries(
                        genericTeams.map(name => [name, []])
                    )
                    const baseResp = await client.evaluate(
                        sessionId!
                    ,   { player_assignments: emptyAssignments, my_team_id: genericTeams[0] }
                    ,   signal
                    )
                    basePlayersBySession.set(sessionId!, client.candidatesToPlayers(baseResp.candidates))
                }

                const myTeamSize = (evalReq.player_assignments[seat] ?? []).length
                if (myTeamSize >= getLeagueSettings().n_picks) {
                    // Clear immediately so no stale result is shown while the request is in-flight.
                    latestFullTeamResult = null
                    // Evaluate with the full team; the backend returns one candidate
                    // representing the team's overall H-score.
                    const fullTeamResp = await client.evaluate(sessionId!, evalReq, signal)
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

                const resp = await client.evaluate(sessionId!, evalReq, signal)
                const players = client.candidatesToPlayers(resp.candidates)

                // Cache the base result if the board was empty (this result IS the base).
                if (!basePlayersBySession.has(sessionId!)) {
                    basePlayersBySession.set(sessionId!, players)
                }

                // Base ordering drives the draft dropdown; current candidates drive the table.
                setBasePlayers(basePlayersBySession.get(sessionId!)!)
                setCandidates(players)
                return
            } catch (err: any) {
                if (err.name === 'AbortError') return  // superseded by a newer call
                if (attempt === 0 && err.message?.includes('(404)')) {
                    // Session expired; reset so ensureSession() creates a fresh one.
                    sessionId = null
                    continue
                }
                throw err
            }
        }
    } finally {
        if (evaluateGeneration === generation) setIndicatorState('idle')
    }
}

/**
 * Evaluates the current draft/auction state for the current seat and rebuilds the candidate table.
 */
export async function runEvaluate(): Promise<void> {
    await evaluateSeat(getCurrentSeat() ?? getLeagueSettings().team_names[0] ?? 'Drafter 1')
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value
    if (mode !== 'Season Mode') {
        if (getFullTeamResult()) {
            showTableMessage('Your team is full.')
        } else {
            buildTable(getCandidatePlayers())
        }
    }
}

/**
 * Analyzes a proposed trade, returning pre/post H-scores for both teams.
 * Ensures a session exists before calling the backend.
 */
export async function runTradeAnalyze(
    playerAssignments: Record<string, string[]>,
    myTeam: string,
    theirTeam: string,
    myTrade: string[],
    theirTrade: string[],
    ignorePositionCheck?: boolean,
): Promise<client.TradeAnalyzeResponse> {
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            await ensureSession()
            return await client.analyzeTrade(sessionId!, {
                player_assignments: playerAssignments,
                my_team: myTeam,
                their_team: theirTeam,
                my_trade: myTrade,
                their_trade: theirTrade,
                ignore_position_check: ignorePositionCheck,
            })
        } catch (err: any) {
            if (attempt === 0 && err.message?.includes('(404)')) {
                sessionId = null
                continue
            }
            throw err
        }
    }
    throw new Error('Trade analyze failed after retry')
}

/**
 * Generates trade suggestions for two teams.
 * Ensures a session exists before calling the backend.
 */
export async function runTradeSuggest(
    playerAssignments: Record<string, string[]>,
    myTeam: string,
    theirTeam: string,
    comboParams: { n_traded: number; n_received: number; threshold: number }[],
    yourThreshold: number,
    theirThreshold: number,
    ignorePositionCheck?: boolean,
): Promise<client.TradeSuggestResponse> {
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            await ensureSession()
            return await client.suggestTrades(sessionId!, {
                player_assignments: playerAssignments,
                my_team: myTeam,
                their_team: theirTeam,
                combo_params: comboParams,
                your_differential_threshold: yourThreshold,
                their_differential_threshold: theirThreshold,
                ignore_position_check: ignorePositionCheck,
            })
        } catch (err: any) {
            if (attempt === 0 && err.message?.includes('(404)')) {
                sessionId = null
                continue
            }
            throw err
        }
    }
    throw new Error('Trade suggest failed after retry')
}

/**
 * Runs evaluate for waiver wire analysis with a caller-supplied roster state.
 * Unlike runEvaluate, the caller is responsible for passing modified player_assignments
 * (with the dropped player removed from their team) so they appear as a free-agent candidate.
 * Does not reapply layout — the waiver tab layout is already correct.
 */
export async function runWaiverEvaluate(
    playerAssignments: Record<string, string[]>,
    myTeamId: string,
): Promise<void> {
    setIndicatorState('evaluating')
    try {
        for (let attempt = 0; attempt < 2; attempt++) {
            try {
                await ensureSession()
                const resp = await client.evaluate(sessionId!, { player_assignments: playerAssignments, my_team_id: myTeamId })
                const players = client.candidatesToPlayers(resp.candidates)

                setCandidates(players)
                buildTable(players)
                return
            } catch (err: any) {
                if (attempt === 0 && err.message?.includes('(404)')) {
                    sessionId = null
                    continue
                }
                throw err
            }
        }
    } finally {
        setIndicatorState('idle')
    }
}

/**
 * Evaluates the full-team H-score for a specific team given roster assignments.
 * Returns h_score and per-category win rates, or null if the backend returns no candidates.
 */
export async function evaluateTeamHScore(
    playerAssignments: Record<string, string[]>
  , teamId: string
): Promise<{ h_score: number; win_rates: number[] } | null> {
    await ensureSession()
    const resp = await client.evaluate(sessionId!, { player_assignments: playerAssignments, my_team_id: teamId })
    if (resp.candidates.length === 0) return null
    return {
        h_score:   resp.candidates[0].h_score,
        win_rates: resp.candidates[0].win_rates,
    }
}

