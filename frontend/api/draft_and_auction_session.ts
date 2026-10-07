// api/draft_and_auction_session.ts
// Draft and auction mode: session management, evaluate orchestration, and indicator state.
// Owns the #eval-indicator element and all draft/auction-specific session state.

import { PlayerResult } from '../types.js'
import { setBasePlayerResults, setCandidatePlayerResults, getCandidatePlayerResults, setGScores, getCurrentSeat } from '../app_state.js'
import {
    getLeagueSettings, getPlatformConfig, getMode, DraftMode, isPlatformConnected, PLATFORM_SELECTION_CHANGED,
} from '../setting_collection/league_settings.js'
import { getSlotCounts } from '../setting_collection/slot_counts.js'
import { getDraftState } from '../data_entry/draft_state.js'
import { getAuctionState } from '../data_entry/auction_state.js'
import { defaultTeamLabel } from '../data_entry/team_labels.js'
import { buildTable, resetTable, addBatch, showTableMessage, reserveTailSpace, clearTailSpace } from '../table/player_table.js'
import { showFailureInTable } from '../table/failure_message.js'

import {
    startFreshSession, getSessionId, resetSession, setIndicatorState,
    withDisplayOwnership, withSessionRetry,
} from './session.js'
import { prefetchHeadshotsForDataSource } from '../player_display.js'
import { patchSession, fetchGScores, evaluate, fetchDraftState, candidatesToPlayerResults, HTTPError } from './client.js'
import { getForcedCategoryWeights } from '../table/force_weights.js'

// Draft/waiver candidate batch size: score + paint the top players first, then fill in the
// bench in follow-up requests. Auction is never batched (its $ values need the whole pool).
// A local constant rather than a parameters.yaml entry: batch size is rendering cadence, not
// model configuration — it changes no score, varies by neither sport nor user, and reading it
// from config would turn a compile-time constant into an async dependency. The value is still
// deliberate: 100 covers everyone plausibly picked next, so the first paint is the whole
// decision-relevant board — and it is the autodraft consideration window, since autopilot
// scores only the first batch (drafts.md documents autodrafters as "top 100" for this reason).
const CANDIDATE_BATCH_SIZE = 100

/** Dispatched on document after a platform poll stores a new board. */
export const LIVE_BOARD_UPDATED = 'live-board-updated'

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

/** The board most recently polled from the connected platform, or null before the first poll.
 *  Read by the layout's team panel, which has no data-entry grid to read from when live. */
export function getLivePlayerAssignments(): Record<string, number[]> | null {
    return livePlayerAssignments
}

function clearLivePlayerAssignments(): void {
    livePlayerAssignments = null
    liveRemainingCash = null
}

/**
 * Polls the connected platform for the current board and stores it as the live state.
 *
 * Shared by the Refresh Analysis button and by the first evaluate after a connection, which
 * has no board yet: connecting clears the previous league's assignments, so every evaluate
 * between the connection and the first manual refresh -- a sidebar apply, a seat change, the
 * bootstrap -- would otherwise fail on a state the user had no way to have loaded.
 * Must run inside withSessionRetry: it reads the session id.
 */
async function pollLiveDraftState(mode: DraftMode): Promise<Record<string, number[]>> {
    const state = await fetchDraftState(getSessionId()!, mode)
    if (isRoomClosedAfterPicks(state.player_assignments)) return livePlayerAssignments!
    setLivePlayerAssignments(state.player_assignments, state.remaining_cash)
    // The board has moved on, which is the whole reason for polling: a live seat's team panel
    // reads these assignments, and nothing else would tell it they changed. The full-team event
    // only fires once a roster is COMPLETE, and a seat change only when the seat changes — so
    // without this the team statistics sat on the roster from whenever the tab was opened.
    document.dispatchEvent(new Event(LIVE_BOARD_UPDATED))
    return state.player_assignments
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


/** Fired (on document) once a Season Mode connection has reached the session, so main.ts can load the
 *  league's rosters into the grid. */
export const SEASON_PLATFORM_CONNECTED = 'season-platform-connected'

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
    // ...and the poll of the previous league with it; it restarts below, once this league's board is loaded.
    stopLivePolling()
    const { platform, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
    createOrPatchSession(4, {
        league: { n_drafters, n_picks, cash_per_team },
        slot_counts: getSlotCounts(),
        platform,
        platform_config: getPlatformConfig(),
    })
        .then(() => {
            // Connecting is the moment the league's board becomes readable, so evaluate it here
            // rather than leaving the table empty until something else happens to trigger a run.
            // Nothing else reliably does: the seat selector re-evaluates only when the seat
            // actually CHANGES, and a league whose teams are named "Team 1".."Team 4" — Yahoo's
            // own naming for unclaimed mock seats — leaves the seat exactly where it was.
            // Evaluating after the patch, not beside it, because the patch is what puts the
            // platform config and the league's counts on the session this reads.
            // Season Mode has no evaluate to run: it fills its roster grid instead, which main.ts owns. It is
            // told here, after the patch, for the same reason the evaluate waits for it -- the roster poll reads
            // the platform config this patch puts on the session.
            // Polling starts once that first evaluate has loaded the board, so its first poll compares
            // against the board on screen instead of racing the evaluate to load it.
            if (getMode() !== 'Season Mode') return runEvaluate().then(() => startLivePolling())
            document.dispatchEvent(new Event(SEASON_PLATFORM_CONNECTED))
        })
        .catch(err => {
            console.error('Platform connect patch failed:', err)
            showFailureInTable(err)
        })
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
            // The session is in hand from here on: what follows is scoring, so the indicator
            // reads "Updating..." — the session build above (the one-time self-play populate)
            // showed "Starting..." (see the busy state below and issue #420).
            if (stillOwner()) setIndicatorState('evaluating')
            const mode = getMode()
            const isLivePlatform = getLeagueSettings().platform !== 'Enter your own data'

            let evalReq: Parameters<typeof evaluate>[1]
            if (isLivePlatform) {
                // Live platforms supply assignments (and, for auctions, remaining cash) from the
                // platform poll instead of a manual board. Polling here when there is nothing
                // stored covers the first evaluate after a connection; Refresh Analysis is then
                // the way to pick up picks made SINCE, not a precondition for evaluating at all.
                const assignments = livePlayerAssignments ?? await pollLiveDraftState(mode)
                evalReq = (mode === 'Auction Mode')
                    ? { player_assignments: assignments, my_team_id: seat, remaining_cash: liveRemainingCash ?? undefined }
                    : { player_assignments: assignments, my_team_id: seat }
            } else if (mode === 'Auction Mode') {
                const { player_assignments, remaining_cash } = getAuctionState()
                evalReq = { player_assignments, my_team_id: seat, remaining_cash }
            } else {
                const { player_assignments } = getDraftState()
                evalReq = { player_assignments, my_team_id: seat }
            }

            // Force-weighting: the user's pinned category weights, when any are set. Attached to the
            // personalised request only — the base ("generic") evaluate below is deliberately left
            // unpinned, because it is the neutral reference the board compares against. Batched
            // follow-up requests spread this same object, so every batch is solved under the pins.
            const forcedCategoryWeights = getForcedCategoryWeights()
            if (forcedCategoryWeights) {
                evalReq = { ...evalReq, forced_category_weights: forcedCategoryWeights }
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
        }, () => {
            // Before each attempt's ensureSession: a missing session means the expensive
            // one-time build (self-play populate) is about to run inside this evaluate —
            // that phase is "Starting...", not "Updating..." (issue #420). Covers both the
            // first load and the recreate after a 404-expired session.
            if (!getSessionId() && stillOwner()) setIndicatorState('fetching')
        })
        return await withDisplayOwnership(
            // First load: no session yet, so the claim itself starts at "Starting..." too —
            // the flip to 'evaluating' happens inside scoreSeat once the session exists.
            { busy: getSessionId() ? 'evaluating' : 'fetching', onSuccess: 'idle', onFailure: 'idle' },
            scoreSeat)
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

// ─── Live draft polling ──────────────────────────────────────────────────────
// While a league is connected in Draft or Auction Mode, the platform's board is polled every
// LIVE_POLL_INTERVAL_MS while the page is in view and every LIVE_POLL_HIDDEN_INTERVAL_MS while it is
// not, and the analysis re-runs when -- and only when -- the board changed. No platform pushes draft
// events, so polling is the only way to follow a draft without a click per pick. One poll is one
// platform request (Yahoo: one draft-results call).
//
// Polls never overlap: the next is scheduled only once the last has answered. A changed board
// starts a new evaluate, which aborts one still running, so a burst of picks costs one finished
// evaluate for the newest board rather than one per pick. An unchanged board does nothing at all --
// not even the indicator moves -- except on coming back to the tab (see pollLiveBoardForChanges).
// Season Mode is not polled: rosters move by the day, not the second.

const LIVE_POLL_INTERVAL_MS = 1000
// Hidden tabs keep following the draft, more gently: the board is current when the user comes back.
// (Browsers throttle a long-hidden tab's timers further on their own; the poll on return covers that.)
const LIVE_POLL_HIDDEN_INTERVAL_MS = 5000
const LIVE_POLL_MAX_BACKOFF_MS = 30000
// The least time "Updating..." shows when a poll on coming back finds nothing new: long enough to be
// seen, so the user knows the board on screen was just checked.
const UNCHANGED_BOARD_FLASH_MS = 700

let livePollTimer: ReturnType<typeof setTimeout> | null = null
// Bumped by every start and stop, so a poll still in flight from an earlier run cannot act on or
// reschedule a run that has since been stopped or replaced.
let livePollGeneration = 0
let livePollFailuresInARow = 0
// Whether the last evaluate a poll started has finished painting its board. A tab coming back with
// one still running (or one that failed) re-evaluates instead of only flashing.
let livePollEvaluateSettled = true

/** Starts polling the connected league's board, unless it is already running or there is nothing to poll. */
export function startLivePolling(): void {
    if (livePollTimer !== null || !shouldPollLiveBoard()) return
    livePollGeneration += 1
    livePollFailuresInARow = 0
    scheduleLivePoll(livePollGeneration, livePollInterval(), false)
}

/** Polls right away, as a catch-up, replacing the slower background wait: coming back to the tab shows "Updating..."
 *  at once, so the user sees the board brought up to date. */
function restartLivePollingImmediately(): void {
    stopLivePolling()
    if (!shouldPollLiveBoard()) return
    livePollGeneration += 1
    livePollFailuresInARow = 0
    scheduleLivePoll(livePollGeneration, 0, true)
}

function livePollInterval(): number {
    return document.visibilityState === 'visible' ? LIVE_POLL_INTERVAL_MS : LIVE_POLL_HIDDEN_INTERVAL_MS
}

export function stopLivePolling(): void {
    if (livePollTimer !== null) clearTimeout(livePollTimer)
    livePollTimer = null
    livePollGeneration += 1
}

/** A live league is connected in a mode that drafts, and there is a session. Whether the tab is in view decides only
 *  how often (livePollInterval): a hidden tab keeps following the draft, so the board is current when the user comes
 *  back -- and a user away for their last pick still sees their final team, though platforms stop answering once a
 *  draft ends (Yahoo's draft results return errors from then on). */
function shouldPollLiveBoard(): boolean {
    return getLeagueSettings().platform !== 'Enter your own data'
        && getMode() !== 'Season Mode'
        && isPlatformConnected()
        && getSessionId() !== null
}

function scheduleLivePoll(
    generation: number
  , delayMs: number
  , catchingUp: boolean
): void {
    livePollTimer = setTimeout(() => { pollLiveBoardForChanges(generation, catchingUp) }, delayMs)
}

/** One poll of the live board; a changed board is stored and re-evaluated. A catching-up poll (the first after the tab
 *  comes back into view) shows "Updating..." from the moment it starts, before the platform has answered, so the user
 *  always sees the board on screen being checked:
 *    - a changed board is evaluated, as any poll's is;
 *    - an unchanged board whose last poll-started evaluate is still running or never finished is evaluated afresh,
 *      repainting whatever was left unfinished while the tab was away;
 *    - an unchanged board already current -- background polling kept it so -- is not re-evaluated: "Updating..." is
 *      held for UNCHANGED_BOARD_FLASH_MS and settles to "Updated", the same assurance without the cost. */
async function pollLiveBoardForChanges(
    generation: number
  , catchingUp: boolean
): Promise<void> {
    if (!shouldPollLiveBoard()) { stopLivePolling(); return }
    const mode = getMode()
    const startedAt = Date.now()
    try {
        const fetchLiveBoard = () => withSessionRetry(() => fetchDraftState(getSessionId()!, mode))
        // No onSuccess: a catch-up hands the display to the evaluate or the flash below.
        let catchUpOwnsDisplay = () => false
        const state = catchingUp
            ? await withDisplayOwnership({ busy: 'evaluating', onFailure: 'idle' }, stillOwner => {
                catchUpOwnsDisplay = stillOwner
                return fetchLiveBoard()
            })
            : await fetchLiveBoard()
        if (generation !== livePollGeneration) return
        livePollFailuresInARow = 0
        // The draft is over and the room gone: keep the final board's results, and stop asking.
        const roomClosed = isRoomClosedAfterPicks(state.player_assignments)
        const boardChanged = !roomClosed
            && describeLiveBoard(state.player_assignments, state.remaining_cash)
               !== describeLiveBoard(livePlayerAssignments, liveRemainingCash)
        if (boardChanged) {
            setLivePlayerAssignments(state.player_assignments, state.remaining_cash)
            document.dispatchEvent(new Event(LIVE_BOARD_UPDATED))
        }
        if (boardChanged || (catchingUp && !livePollEvaluateSettled)) {
            runLivePollEvaluate()
        } else if (catchingUp) {
            flashUnchangedBoard(startedAt, catchUpOwnsDisplay)
        }
        if (roomClosed || isLiveBoardFull(state.player_assignments)) { stopLivePolling(); return }
        scheduleLivePoll(generation, livePollInterval(), false)
    } catch (err) {
        if (generation !== livePollGeneration) return
        // An expired or revoked authorization is not going to fix itself: say so and stop, and
        // reconnecting (which restarts the poll) is the way back.
        if (err instanceof HTTPError && err.status === 401) {
            stopLivePolling()
            showFailureInTable(err)
            return
        }
        // Anything else -- the platform briefly unreachable, a rate limit -- is waited out, backing
        // off so a struggling platform is not hit every second.
        livePollFailuresInARow += 1
        const backoffMs = Math.min(LIVE_POLL_INTERVAL_MS * 2 ** livePollFailuresInARow, LIVE_POLL_MAX_BACKOFF_MS)
        console.warn(`Live draft poll failed (${livePollFailuresInARow} in a row); retrying in ${backoffMs} ms:`, err)
        scheduleLivePoll(generation, backoffMs, false)
    }
}

// Counts the evaluates polls start, so only the newest one marks the board settled: an older one finishing late
// (or being aborted by its successor) says nothing about the board now on screen.
let livePollEvaluateCount = 0

/** Re-evaluates for the live board, recording whether that evaluate finished (livePollEvaluateSettled). */
function runLivePollEvaluate(): void {
    const evaluateNumber = ++livePollEvaluateCount
    livePollEvaluateSettled = false
    runEvaluate().then(() => {
        if (evaluateNumber === livePollEvaluateCount) livePollEvaluateSettled = true
    }).catch(err => {
        if (err.name === 'AbortError') return   // a newer board's evaluate replaced it
        console.error('Evaluate after a live board poll failed:', err)
        showFailureInTable(err)
    })
}

/** Holds "Updating..." until UNCHANGED_BOARD_FLASH_MS after the poll began, then settles to "Updated" -- unless
 *  something newer has taken over the display meanwhile, which then owns what it shows. */
function flashUnchangedBoard(
    startedAt: number
  , stillOwnsDisplay: () => boolean
): void {
    const remainingMs = Math.max(0, UNCHANGED_BOARD_FLASH_MS - (Date.now() - startedAt))
    setTimeout(() => { if (stillOwnsDisplay()) setIndicatorState('idle') }, remainingMs)
}

/** The board as a canonical string (teams sorted), so a poll that returns the same board in another
 *  key order does not count as a change. */
function describeLiveBoard(
    assignments: Record<string, number[]> | null
  , remainingCash: Record<string, number> | null | undefined
): string {
    if (assignments === null) return 'no board'
    const sortedEntries = (record: Record<string, unknown>) =>
        Object.keys(record).sort().map(key => [key, record[key]])
    return JSON.stringify([sortedEntries(assignments), remainingCash ? sortedEntries(remainingCash) : null])
}

/** A board with no picks, arriving after one that had them. Picks are not un-made during a draft, so this is not
 *  the draft's state: it is the platform no longer reporting the draft -- a Yahoo mock room, for one, stops
 *  returning its results once the draft ends, which the integration (rightly, before a draft) reads as "not
 *  started". Taking it at its word replaced the final results with an empty board's base rankings. A commissioner
 *  undoing a pick still comes through: that board is smaller, not empty. */
function isRoomClosedAfterPicks(polledAssignments: Record<string, number[]>): boolean {
    const countPicks = (assignments: Record<string, number[]>) =>
        Object.values(assignments).reduce((total, roster) => total + roster.length, 0)
    return countPicks(polledAssignments) === 0
        && livePlayerAssignments !== null
        && countPicks(livePlayerAssignments) > 0
}

/** Every roster spot taken: the draft is over, and nothing more will change. */
function isLiveBoardFull(assignments: Record<string, number[]>): boolean {
    const { n_drafters, n_picks } = getLeagueSettings()
    const picksMade = Object.values(assignments).reduce((total, roster) => total + roster.length, 0)
    return picksMade >= n_drafters * n_picks
}

// Coming back polls at once rather than waiting out the slower background interval; going away changes nothing but
// the next wait (livePollInterval reads the visibility when it schedules).
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') restartLivePollingImmediately()
})

// Naming another league ends the connection as far as polling is concerned: polling on would keep
// refreshing the OLD league's board under the new one's name.
document.addEventListener(PLATFORM_SELECTION_CHANGED, () => {
    if (!isPlatformConnected()) stopLivePolling()
})
