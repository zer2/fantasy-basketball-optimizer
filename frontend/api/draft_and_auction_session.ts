// api/draft_and_auction_session.ts
// Draft and auction mode: session management, evaluate orchestration, and indicator state.
// Owns the #eval-indicator element and all draft/auction-specific session state.

import { PlayerResult } from '../types.js'
import { setBasePlayerResults, setCandidatePlayerResults, getCandidatePlayerResults, setGScores, getCurrentSeat } from '../app_state.js'
import { getLeagueSettings, getPlatformConfig, getMode } from '../parameter_collection/league_settings.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { getDraftState } from '../data_entry/draft_state.js'
import { getAuctionState } from '../data_entry/auction_state.js'
import { defaultTeamLabel } from '../data_entry/team_labels.js'
import { buildTable, resetTable, addBatch, showTableMessage, reserveTailSpace, clearTailSpace } from '../table/player_table.js'

import {
    startFreshSession, getSessionId, resetSession, setIndicatorState,
    withDisplayOwnership, withSessionRetry,
} from './session.js'
import { prefetchHeadshotsForDataSource } from '../player_display.js'
import { patchSession, fetchGScores, evaluate, fetchDraftState, candidatesToPlayerResults, HTTPError } from './client.js'

// Draft/waiver candidate batch size: score + paint the top players first, then fill in the
// bench in follow-up requests. Auction is never batched (its $ values need the whole pool).
// A local constant rather than a parameters.yaml entry: batch size is rendering cadence, not
// model configuration — it changes no score, varies by neither sport nor user, and reading it
// from config would turn a compile-time constant into an async dependency. The value is still
// deliberate: 100 covers everyone plausibly picked next, so the first paint is the whole
// decision-relevant board — and it is the autodraft consideration window, since autopilot
// scores only the first batch (drafts.md documents autodrafters as "top 100" for this reason).
const CANDIDATE_BATCH_SIZE = 100

// ─── Draft/auction state ─────────────────────────────────────────────────────

const basePlayersBySession: Map<string, PlayerResult[]> = new Map()
let evaluateController: AbortController | null = null
let latestFullTeamResult: { h_score: number; win_rates: number[] } | null = null

// Display ownership (withDisplayOwnership, claimDisplay) lives in session.js, beside the
// indicator it protects; every async writer in this module runs through the wrapper.

// Player assignments pulled from a live platform (Refresh Analysis). When set,
// evaluateSeat uses these instead of reading the manual draft/auction board,
// which does not exist in the live-platform layout.
let livePlayerAssignments: Record<string, number[]> | null = null
let liveRemainingCash: Record<string, number> | null = null

function setLivePlayerAssignments(
    assignments: Record<string, number[]>
    , remainingCash?: Record<string, number>
): void {
    livePlayerAssignments = assignments
    liveRemainingCash = remainingCash ?? null
}

function clearLivePlayerAssignments(): void {
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

/** The evaluate request for an empty board: generic team names, no picks, evaluated from the
 *  first seat. An auction session requires remaining_cash on every evaluate — with no picks
 *  made, every team still holds its full budget. */
function buildEmptyBoardEvaluateRequest(mode: string): Parameters<typeof evaluate>[1] {
    const { n_drafters } = getLeagueSettings()
    const genericTeams = Array.from({ length: n_drafters }, (_, index) => defaultTeamLabel(index))
    const request: Parameters<typeof evaluate>[1] = {
        player_assignments: Object.fromEntries(genericTeams.map(name => [name, []])),
        my_team_id: genericTeams[0],
    }
    if (mode === 'Auction Mode') request.remaining_cash = buildFullBudgets(genericTeams)
    return request
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
    // A data-source change means a new player pool: start warming its headshots now, in
    // parallel with the rebuild (startFreshSession does the same for brand-new sessions).
    const patchDataSource = patchBody.data_source as { type: string; season?: string | null } | undefined
    if (patchDataSource) prefetchHeadshotsForDataSource(patchDataSource.type, patchDataSource.season)
    // On failure the indicator must not stay stuck on "Starting..." — no evaluate follows a
    // failed create/patch to reset it. onSuccess is omitted because the indicator deliberately
    // stays 'fetching': the caller chains an evaluate, which claims the display itself.
    await withDisplayOwnership({ busy: 'fetching', onFailure: 'idle' }, async () => {
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
    })
}


// When a live platform connects, patch the session so it carries the platform's config
// (which drives the draft-state poll + name lookup) and the platform's drafter/pick counts.
// A patch suffices: the loaded player data is untouched, so the session does not need to be
// torn down and rebuilt from step 1 the way a data-source change would require. The counts
// rerun the later pipeline steps (4-5); platform_config itself is merely stored on the
// session — no pipeline step reads it. Driven by an event so league_settings doesn't import
// this module (it imports league_settings — a cycle).
document.addEventListener('platform-connected', () => {
    // A fresh connection invalidates any polled board from the previous league. This is not
    // automatic: reconnecting to a different league on the SAME platform never touches the
    // platform-selection reset, so without this clear the old league's assignments would be
    // evaluated against the new league's session by any evaluate that runs before the first
    // Refresh Analysis.
    clearLivePlayerAssignments()
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
async function evaluateSeat(seat: string, forAutopilot = false): Promise<number | null> {
    if (evaluateController) evaluateController.abort()
    evaluateController = new AbortController()
    const { signal } = evaluateController
    try {
        const scoreSeat = (stillOwner: () => boolean) => withSessionRetry(async () => {
            const mode = getMode()
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
                const baseResp = await evaluate(getSessionId()!, buildEmptyBoardEvaluateRequest(mode), signal)
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
                    // Ownership-checked for non-aborting claimants (the debounce spinner, a patch):
                    // their state must not be knocked back to 'idle' by this still-running evaluate.
                    if (first && stillOwner()) setIndicatorState('idle')
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
                return players[0]?.player_id ?? null
            }

            if (!basePlayersBySession.has(getSessionId()!)) {
                basePlayersBySession.set(getSessionId()!, players)
            }

            setBasePlayerResults(basePlayersBySession.get(getSessionId()!)!)
            setCandidatePlayerResults(players)
            return null
        })
        return await withDisplayOwnership(
            { busy: 'evaluating', onSuccess: 'idle', onFailure: 'idle' }, scoreSeat)
    } catch (err: any) {
        if (err.name === 'AbortError') return null
        throw err
    }
}

/** Evaluates the current draft/auction state for the current seat and rebuilds the candidate table.
 *  With `forAutopilot`, it scores only the first batch and renders nothing — the caller only needs the
 *  top candidate for an autopilot pick. */
export async function runEvaluate(options: { forAutopilot?: boolean } = {}): Promise<number | null> {
    const forAutopilot = options.forAutopilot ?? false
    // No explicit seat selected falls back to the first team; an empty league is a bug, not a default.
    const seat = getCurrentSeat() ?? getLeagueSettings().team_names[0]
    if (seat === undefined) throw new Error('runEvaluate: no seat selected and league has no team names')
    const topPick = await evaluateSeat(seat, forAutopilot)
    if (forAutopilot) return topPick   // autopilot needs only the top candidate; nothing is shown
    const mode = getMode()
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
 * Before a live platform is connected, show the base player rankings (everyone vs.
 * empty teams) so the user still sees default rankings pre-auth. Uses generic teams
 * for the evaluation, so it doesn't depend on a selected seat.
 */
export async function showDefaultRankings(): Promise<void> {
    // A per-seat evaluate can still be in flight here — e.g. the user switches from manual
    // entry to a live platform while the board is mid-score. Abort it and claim the
    // display: the aborted run still executes its finally block, and only the ownership
    // check there keeps it from stamping 'idle' over the states set here.
    if (evaluateController) evaluateController.abort()
    await withDisplayOwnership(
        { busy: 'fetching', onSuccess: 'unconnected', onFailure: 'unconnected' }
        , stillOwner => withSessionRetry(async () => {
            const { mode } = getLeagueSettings()
            const resp = await evaluate(getSessionId()!, buildEmptyBoardEvaluateRequest(mode))
            // Nothing cancels this request — it carries no abort signal — so by the time it
            // resolves, a newer run may own the display. Its board must not be repainted
            // with empty-board rankings, so the writes are ownership-gated like the
            // indicator resets.

            if (stillOwner()) {
                const players = candidatesToPlayerResults(resp.candidates)
                setBasePlayerResults(players)
                setCandidatePlayerResults(players)
                buildTable(players)
            }
        })
    )
}

/**
 * Live-platform refresh: polls the connected platform for the current draft /
 * roster state, stores it as the live player assignments, then re-evaluates.
 * Backs the "Refresh Analysis" button in the live-platform layout.
 */
export async function refreshLiveAnalysis(): Promise<void> {
    // On a failed poll the spinner must not stay stuck. On success there is no terminal
    // write to make: runEvaluate has claimed the display itself and owns the ending state,
    // including for its own failures.
    await withDisplayOwnership({ busy: 'fetching', onFailure: 'idle' }, async () => {
        await withSessionRetry(async () => {
            const mode = getMode()
            const state = await fetchDraftState(getSessionId()!, mode)
            setLivePlayerAssignments(state.player_assignments, state.remaining_cash)
        })
        await runEvaluate()
    })
}
