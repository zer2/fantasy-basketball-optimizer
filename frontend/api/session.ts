// api/session.ts
// Core session lifecycle: ID management, session creation, indicator state, and shared utilities.
// Indicator state is centralised here so all modes share a single source of truth.

import { SessionRequest } from '../types.js'
import { setGScores } from '../app_state.js'
import { getLeagueSettings, getPlatformConfig } from '../parameter_collection/league_settings.js'
import {
    getScoringFormat, getMostCategoriesWeight, getTiebreakerCategory, getSelectedCategories,
    syncCategoriesFromBackend,
} from '../parameter_collection/format_and_categories.js'
import { getPlayerStatsParams } from '../parameter_collection/player_stats.js'
import { getModelParameters } from '../parameter_collection/model_parameters.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { createSession, HTTPError } from './client.js'
import { prefetchHeadshotsForDataSource } from '../player_display.js'

// ─── Module state ─────────────────────────────────────────────────────────────

let sessionId: string | null = null

// Indicator state: single source of truth for the loading indicator (#eval-indicator)
// across all modes. The current state is stored as a data-state attribute so styles.css
// can apply different colours and animations per state (e.g. spinner while fetching/evaluating).
const INDICATOR_LABELS: Record<string, string> = {
    idle:         'Updated',
    fetching:     'Starting...',
    evaluating:   'Updating...',
    autopiloting: 'Autopiloting...',
    unconnected:  'Unconnected',
}
type IndicatorState = keyof typeof INDICATOR_LABELS

// Default to 'fetching' (not 'idle') so the indicator says "Starting..." from page
// load until the first session-creating action runs setIndicatorState. The HTML
// default in app.html mirrors this so the very first paint matches.
let currentIndicatorState: IndicatorState = 'fetching'

// ─── Session ID ───────────────────────────────────────────────────────────────

export function getSessionId(): string | null { return sessionId }
export function resetSession(): void          { sessionId = null }

/** Applies a state to any indicator element, looked up by ID. */
export function applyIndicatorState(
    elementId: string
    , state: IndicatorState
): void {
    const element = document.getElementById(elementId)
    if (!element) return
    element.dataset.state = state
    element.textContent = INDICATOR_LABELS[state]
}

/** Sets the primary #eval-indicator state.  Suppressed while autopilot is active. */
export function setIndicatorState(state: 'idle' | 'fetching' | 'evaluating' | 'unconnected'): void {
    if (currentIndicatorState === 'autopiloting') return
    currentIndicatorState = state
    applyIndicatorState('eval-indicator', state)
}

/** Enters autopilot indicator mode. While active, setIndicatorState calls are suppressed. */
export function setAutopilotOn(): void {
    currentIndicatorState = 'autopiloting'
    applyIndicatorState('eval-indicator', 'autopiloting')
}

/** Exits autopilot indicator mode and resets to idle. */
export function setAutopilotOff(): void {
    currentIndicatorState = 'idle'
    applyIndicatorState('eval-indicator', 'idle')
}

// ─── Session creation ────────────────────────────────────────────────────────

/**
 * Reads all sidebar parameters, POSTs to /sessions, and stores the returned
 * session ID and G-scores.  Every code path that needs a new session calls this.
 */
export async function startFreshSession(signal?: AbortSignal): Promise<void> {
    const { sport, platform, mode, n_drafters, n_picks, cash_per_team } = getLeagueSettings()
    const scoring_format = getScoringFormat()
    const most_categories_weight = getMostCategoriesWeight()
    const categories     = getSelectedCategories()
    const { data_source, injured_players } = getPlayerStatsParams()
    const league: SessionRequest['league'] = {
        sport, n_drafters, n_picks, scoring_format, most_categories_weight,
        tiebreaker_category: getTiebreakerCategory(), categories,
    }
    if (mode === 'Auction Mode') league.cash_per_team = cash_per_team
    const req: SessionRequest = {
        league,
        is_auction: mode === 'Auction Mode',
        platform,
        slot_counts: getSlotCounts(),
        parameters: getModelParameters(),
        data_source,
        injured_players,
    }
    const platformConfig = getPlatformConfig()
    if (platformConfig) req.platform_config = platformConfig
    // Warm the pool's headshots in parallel with the build: H-score setup is CPU-bound
    // server-side while image serving is pure I/O, so the build window is free time.
    prefetchHeadshotsForDataSource(data_source.type, data_source.season)
    const resp = await createSession(req, signal)
    sessionId = resp.session_id
    syncCategoriesFromBackend(resp.categories)
    setGScores(resp.g_scores)
}

/**
 * Ensures a session exists (creating one from current sidebar state if not).
 * Called before any backend operation that requires a session.
 */
export async function ensureSession(): Promise<void> {
    if (sessionId) return
    await startFreshSession()
}

// ─── Indicator-aware debouncer ────────────────────────────────────────────────
// Used by draft_board, auction_entry, and the player-stats sidebar so rapid
// edits only trigger one backend call after `delayMs` of inactivity. Each fire()
// also flips #eval-indicator to "evaluating" immediately, so the spinner appears
// as soon as the user acts rather than after the debounce window expires.

export interface Debouncer {
    /** Schedule the callback; resets the timer on each call. */
    fire(): void
    /** Cancel any pending invocation (e.g. on board reset). */
    cancel(): void
}

/** Creates a debouncer that calls `fn` only after `delayMs` ms of inactivity.
 *  Each `fire()` also sets the eval indicator to "evaluating" so the spinner is
 *  visible during the debounce window; the eventual `fn()` is expected to clear
 *  it (e.g. via runEvaluate's finally). */
export function makeDebouncer(fn: () => void, delayMs = 300): Debouncer {
    let timer: ReturnType<typeof setTimeout> | null = null
    return {
        fire() {
            if (timer) clearTimeout(timer)
            setIndicatorState('evaluating')
            timer = setTimeout(() => { timer = null; fn() }, delayMs)
        },
        cancel() {
            if (timer) { clearTimeout(timer); timer = null }
        },
    }
}

// ─── Session retry ────────────────────────────────────────────────────────────

/**
 * Ensures a session exists, runs `fn`, and retries once with a fresh session if
 * the backend returns 404 (session expired).  Other errors propagate.
 *
 * `onBeforeAttempt` runs before each attempt's ensureSession call — useful for
 * resetting an indicator back to a "fetching" state for the retry.
 */
export async function withSessionRetry<T>(
    fn: () => Promise<T>
    , onBeforeAttempt?: () => void
): Promise<T> {
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            onBeforeAttempt?.()
            await ensureSession()
            return await fn()
        } catch (err) {
            if (attempt === 0 && err instanceof HTTPError && err.status === 404) {
                resetSession()
                continue
            }
            throw err
        }
    }
    throw new Error('withSessionRetry: loop exited without returning')
}

