// api/session.ts
// Core session lifecycle: ID management, session creation, indicator state, and shared utilities.
// Indicator state is centralised here so all modes share a single source of truth.

import { SessionRequest } from '../types.js'
import { setGScores } from '../app_state.js'
import { getLeagueSettings } from '../parameter_collection/league_settings.js'
import { getScoringFormat, getSelectedCategories, syncCategoriesFromBackend } from '../parameter_collection/format_and_categories.js'
import { getPlayerStatsParams } from '../parameter_collection/player_stats.js'
import { getModelParameters } from '../parameter_collection/model_parameters.js'
import { getSlotCounts } from '../parameter_collection/slot_counts.js'
import { createSession } from './client.js'

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
}
type IndicatorState = keyof typeof INDICATOR_LABELS

let currentIndicatorState: IndicatorState = 'idle'

// ─── Session ID ───────────────────────────────────────────────────────────────

export function getSessionId(): string | null { return sessionId }
export function resetSession(): void          { sessionId = null }

/** Applies a state to any indicator element, looked up by ID. */
export function applyIndicatorState(
    elementId: string
    , state: string
): void {
    const element = document.getElementById(elementId)
    if (!element) return
    element.dataset.state = state
    element.textContent = INDICATOR_LABELS[state] ?? ''
}

/** Sets the primary #eval-indicator to fetching, evaluating, or idle.  Suppressed while autopilot is active. */
export function setIndicatorState(state: 'idle' | 'fetching' | 'evaluating'): void {
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
    }
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

