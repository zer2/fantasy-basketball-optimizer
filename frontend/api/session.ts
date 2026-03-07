// api/session.ts
// Session lifecycle management and evaluate orchestration.
// Owns the session ID, abort controller, and the updateTable bridge between
// backend responses and the player table.

import { SessionRequest } from '../types.js'
import { setPlayers, setCategories } from '../app_state.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getFormatAndCategories } from '../parameter_collection/format_and_categories.js'
import { getPlayerStatsParams } from '../parameter_collection/player_stats.js'
import { getModelParameters } from '../parameter_collection/model_parameters.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { reapplyLayout, getCurrentSeat } from '../layout.js'
import { getDraftState } from '../data_entry/draft_board.js'
import { getAuctionState } from '../data_entry/auction_entry.js'
import { buildTable } from '../table/player_table.js'
import * as api from './client.js'
import { Player } from '../types.js'

// ─── Module state ────────────────────────────────────────────────────────────

let sessionId: string | null = null
// Tracks the in-flight evaluate request so it can be aborted when a newer one starts.
let evaluateController: AbortController | null = null

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Collects all sidebar parameter values and assembles a `SessionRequest` object
 * ready to POST to `/sessions`.
 */
export function buildSessionRequest(): SessionRequest {
    const { sport, platform, mode, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
    const { scoring_format, categories } = getFormatAndCategories()
    const { data_source, injured_players } = getPlayerStatsParams()
    const league: SessionRequest['league'] = { sport, n_drafters, n_picks, scoring_format, categories }
    if (mode === 'Auction Mode') league.cash_per_team = cash_per_team
    return {
        league,
        platform,
        slot_counts: getSlotCounts(),
        parameters: getModelParameters(),
        data_source,
        injured_players,
        // my_team_id comes from the seat selector in the main content area
    }
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
        const req = buildSessionRequest()
        const resp = await api.createSession(req, signal)
        sessionId = resp.session_id
        setCategories(resp.categories)
        return
    }
    try {
        await api.patchSession(sessionId, { from_step: fromStep, ...patchBody }, signal)
    } catch (err: any) {
        if (!err.message?.includes('(404)')) throw err
        // Session expired; rebuild from current sidebar state.
        sessionId = null
        const req = buildSessionRequest()
        const resp = await api.createSession(req, signal)
        sessionId = resp.session_id
        setCategories(resp.categories)
    }
}

/**
 * Ensures a session exists (creating one from current sidebar state if not).
 * Called before evaluate when the session may not have been explicitly created yet.
 */
async function ensureSession(): Promise<void> {
    if (sessionId) return
    const req = buildSessionRequest()
    const resp = await api.createSession(req)
    sessionId = resp.session_id
    setCategories(resp.categories)
}

/**
 * Runs the evaluate endpoint with the current draft / auction state and
 * updates the candidate table with the response.
 * Retries once if the session has expired (404), creating a fresh session first.
 */
export async function runEvaluate(): Promise<void> {
    // Abort any in-flight evaluate request so the backend is only computing
    // the most recent draft state, not every intermediate pick.
    if (evaluateController) evaluateController.abort()
    evaluateController = new AbortController()
    const { signal } = evaluateController

    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            await ensureSession()

            // Default to the first team name if no seat has been chosen yet
            const seat = getCurrentSeat() || getLeagueSettings().team_names[0] || 'Drafter 1'
            const mode = (document.getElementById('ls-mode') as HTMLInputElement).value

            let evalReq: Parameters<typeof api.evaluate>[1]
            if (mode === 'Auction Mode') {
                const { player_assignments, remaining_cash } = getAuctionState()
                evalReq = { player_assignments, my_team_id: seat, remaining_cash }
            } else {
                const { player_assignments } = getDraftState()
                evalReq = { player_assignments, my_team_id: seat }
            }

            const resp = await api.evaluate(sessionId!, evalReq, signal)
            updateTable(api.candidatesToPlayers(resp.candidates))
            reapplyLayout()
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
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            await ensureSession()
            const resp = await api.evaluate(sessionId!, { player_assignments: playerAssignments, my_team_id: myTeamId })
            updateTable(api.candidatesToPlayers(resp.candidates))
            return
        } catch (err: any) {
            if (attempt === 0 && err.message?.includes('(404)')) {
                sessionId = null
                continue
            }
            throw err
        }
    }
}

/**
 * Updates the player data and rebuilds the table.
 * Call this whenever the backend returns a new set of results.
 * Reads the current mode from the DOM, so the layout stays in sync automatically.
 *
 * @param newPlayers    - Full player list from the backend response
 * @param newCategories - Category list; omit if unchanged (e.g. same session)
 */
export function updateTable(newPlayers: Player[], newCategories?: string[]): void {
    setPlayers(newPlayers)
    if (newCategories) setCategories(newCategories)
    buildTable()
}
