// api/client.ts
// HTTP client for the Fantasy Basketball Optimizer backend.
// All fetch calls go through this module; callers receive typed results.

import { PlayerResult, PlayerGScore, SessionRequest, SportConfig } from '../types.js'


// Empty string = same-origin deployment (frontend and backend served from the same host).
// Change to an absolute URL (e.g. 'https://api.example.com') if the frontend
// and backend are ever deployed on different origins.
export const BASE_URL = ''


// ── HTTP helpers ──────────────────────────────────────────────────────────────

/** Thrown by jsonRequest when the backend returns a non-2xx status. Callers that
 *  want to react to specific statuses (e.g. retry on session expiry) should catch
 *  this and check `.status` rather than parsing the message string. */
export class HTTPError extends Error {
    constructor(
        public readonly status: number
        , public readonly body: string
        , label: string
    ) {
        super(`${label} failed (${status}): ${body}`)
        this.name = 'HTTPError'
    }
}

/** Issues `fetch(url, init)` and parses the JSON response, throwing HTTPError on
 *  non-2xx. The `label` prefixes error messages so callers don't each repeat the
 *  "X failed (status): body" boilerplate. */
async function jsonRequest<T>(
    url: string
    , label: string
    , init?: RequestInit
): Promise<T> {
    const res = await fetch(url, init)
    if (!res.ok) {
        const body = await res.text()
        throw new HTTPError(res.status, body, label)
    }
    if (res.status === 204) return undefined as T   // no content (e.g. token exchange)
    return res.json() as Promise<T>
}


// ── Map backend Candidate → frontend PlayerResult ─────────────────────────────


// Adapter between backend snake_case Candidate objects and frontend PlayerResult objects.
// The backend follows Python naming conventions; this is the single place where that translation happens.
/** Converts raw backend Candidate objects to frontend PlayerResult objects, remapping snake_case keys to camelCase. */
export function candidatesToPlayerResults(candidates: any[]): PlayerResult[] {
    return candidates.filter(c => c.h_score != null).map((c, i) => ({
        name:             c.name,
        h_score:          c.h_score,
        h_rank:           c.h_rank,
        g_rank:           i + 1,
        win_rates:        c.win_rates,
        category_weights: c.category_weights,
        g_score_rows: c.g_score_rows.map((r: any) => ({
            label:   r.label,
            values:  r.values,
            total:   r.total,
            isTotal: r.is_total,
        })),
        flex_allocations: c.flex_allocations ? {
            base_positions: c.flex_allocations.base_positions,
            rows: c.flex_allocations.rows.map((r: any) => ({
                label:   r.label,
                values:  r.values,
                isTotal: r.is_total,
            })),
        } : undefined,
        roster: c.roster ? {
            slots: c.roster.slots,
            assignments: Object.fromEntries(
                Object.entries(c.roster.assignments).map(([slot, a]: [string, any]) => [
                    slot,
                    a ? { name: a.name, full_name: a.full_name, isCandidate: a.is_candidate } : null,
                ]),
            ),
        } : undefined,
        auction_values: c.auction_values ? {
            your_dollar:   c.auction_values.your_dollar,
            gnrc_dollar:   c.auction_values.gnrc_dollar,
            orig_dollar:   c.auction_values.orig_dollar,
            gnrc_dollar_g: c.auction_values.gnrc_dollar_g,
            orig_dollar_g: c.auction_values.orig_dollar_g,
        } : undefined,
    }))
}

// ── GET /config/{sport} ──────────────────────────────────────────────────────

/** Fetches sport-specific configuration (defaults, categories, positions) from parameters.yaml. */
export async function fetchConfig(sport: string): Promise<SportConfig> {
    return jsonRequest(`${BASE_URL}/config/${encodeURIComponent(sport)}`, 'Config fetch')
}

// ── POST /data/upload ─────────────────────────────────────────────────────────

/** Uploads a projection CSV (export format auto-detected server-side) and returns its data_id. */
export async function uploadCsv(
    file: File
): Promise<{ data_id: string; detected_format: string; n_players: number; expires_at: string }> {
    const form = new FormData()
    form.append('file', file)
    return jsonRequest(`${BASE_URL}/data/upload`, 'Upload', { method: 'POST', body: form })
}

// ── GET /seasons ───────────────────────────────────────────────────────────────

/** Fetches the list of available historical seasons from the backend. */
export async function getSeasons(): Promise<string[]> {
    const data = await jsonRequest<{ seasons: string[] }>(`${BASE_URL}/seasons`, 'Get seasons')
    return data.seasons
}

// ── POST /sessions ─────────────────────────────────────────────────────────────

/** Creates a new backend session, running the full 5-step pipeline. Returns session_id and resolved categories. */
export async function createSession(
    req: SessionRequest
    , signal?: AbortSignal
): Promise<{ session_id: string; categories: string[]; g_scores: PlayerGScore[]; n_players_loaded: number; expires_at: string }> {
    return jsonRequest(`${BASE_URL}/sessions`, 'Create session', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
}

// ── PATCH /sessions/{id} ──────────────────────────────────────────────────────

/** Patches an existing session. Only call via `createOrPatchSession`, which handles 404 (expired session) recovery. */
export async function patchSession(
    sessionId: string
    , req: Record<string, unknown>
    , signal?: AbortSignal
): Promise<{ ok: boolean; steps_rerun: number[] }> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}`, 'Patch session', {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
}

// ── GET /sessions/{id}/g-scores ───────────────────────────────────────────────

/** Fetches the current G-scores for a session directly from session state. */
export async function fetchGScores(sessionId: string): Promise<PlayerGScore[]> {
    const data = await jsonRequest<{ g_scores: PlayerGScore[] }>(`${BASE_URL}/sessions/${sessionId}/g-scores`, 'Fetch G-scores')
    return data.g_scores
}

// ── POST /sessions/{id}/evaluate ──────────────────────────────────────────────

/** Runs the H-score algorithm for the given draft/auction state and returns ranked candidates. Supports AbortSignal for cancellation. */
export async function evaluate(
    sessionId: string
    , req: {
        player_assignments: Record<string, string[]>
        my_team_id: string
        exclusion_list?: string[]
        remaining_cash?: Record<string, number>
        candidate_offset?: number   // draft/waiver batching: slice start
        candidate_limit?: number    // draft/waiver batching: slice size (omit = whole pool)
    }
    , signal?: AbortSignal
): Promise<{ iteration: number; candidates: any[]; has_more?: boolean; total_candidates?: number }> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/evaluate`, 'Evaluate', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
}

// ── POST /sessions/{id}/trade/analyze ────────────────────────────────────────

export interface TeamHScore {
    h_score: number
    rates: number[]
}

export interface TeamTradeResult {
    pre: TeamHScore
    post: TeamHScore
}

export interface TradeAnalyzeResponse {
    your_team: TeamTradeResult | null
    their_team: TeamTradeResult | null
    error: string | null
}

/** Analyzes a trade and returns pre/post H-scores for both teams. */
export async function analyzeTrade(
    sessionId: string
    , req: {
        player_assignments: Record<string, string[]>
        my_team: string
        their_team: string
        my_trade: string[]
        their_trade: string[]
        ignore_position_check?: boolean
    }
): Promise<TradeAnalyzeResponse> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/trade/analyze`, 'Trade analyze', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
}

// ── POST /sessions/{id}/trade/suggest ────────────────────────────────────────

export interface TradeSuggestion {
    send: string[]
    receive: string[]
    your_score: number
    their_score: number
}

export interface TradeSuggestResponse {
    suggestions: TradeSuggestion[]
}

/** Generates trade suggestions for two teams. May take 10-30s. */
export async function suggestTrades(
    sessionId: string
    , req: {
        player_assignments: Record<string, string[]>
        my_team: string
        their_team: string
        combo_params: { n_traded: number; n_received: number; threshold: number }[]
        your_differential_threshold: number
        their_differential_threshold: number
        ignore_position_check?: boolean
    }
): Promise<TradeSuggestResponse> {
    return jsonRequest(`${BASE_URL}/sessions/${sessionId}/trade/suggest`, 'Trade suggest', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
}

// ── Platform integration (live: ESPN / Yahoo / Fantrax) ───────────────────────
// `platform` is the exact label from PLATFORM_OPTIONS (e.g. 'Retrieve from Fantrax').

export interface PlatformDivision {
    name: string
    id: string
}

export interface PlatformConnectResponse {
    team_names: string[]
    n_drafters: number
    n_picks: number
    available_modes: string[]
}

export interface DraftStateResponse {
    player_assignments: Record<string, string[]>
    injured_players: string[]
    status: string
    remaining_cash?: Record<string, number>   // Auction Mode only
}

export interface PlatformLeague {
    id: string
    name: string
    season?: number
}

/** Lists a platform league's divisions (empty when the league has none). */
export async function fetchDivisions(
    platform: string
    , leagueId: string
): Promise<PlatformDivision[]> {
    const data = await jsonRequest<{ divisions: PlatformDivision[] }>(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/divisions?league_id=${encodeURIComponent(leagueId)}`
        , 'Platform divisions'
    )
    return data.divisions
}

/** Lists the user's leagues for an auth-based platform (empty for manual-id platforms).
 *  Credentials are resolved server-side from the signed-in user's session cookie. */
export async function fetchLeagues(platform: string): Promise<PlatformLeague[]> {
    const data = await jsonRequest<{ leagues: PlatformLeague[] }>(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/leagues`
        , 'Platform leagues'
    )
    return data.leagues
}

/** Returns the Yahoo OAuth authorization URL for the user to visit. */
export async function fetchYahooAuthUrl(): Promise<string> {
    const data = await jsonRequest<{ auth_url: string }>(`${BASE_URL}/platforms/yahoo/auth-url`, 'Yahoo auth URL')
    return data.auth_url
}

/** Exchanges a pasted Yahoo authorization code for tokens, persisted for the signed-in user. */
export async function submitYahooToken(authCode: string): Promise<void> {
    await jsonRequest<void>(`${BASE_URL}/platforms/yahoo/token`, 'Yahoo token', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ auth_code: authCode }),
    })
}

/** Persists the signed-in user's ESPN s2 + SWID cookies. */
export async function submitEspnCredentials(s2: string, swid: string): Promise<void> {
    await jsonRequest<void>(`${BASE_URL}/platforms/espn/credentials`, 'ESPN credentials', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ s2, swid }),
    })
}

/** Connects to a platform league and returns its team/shape metadata. */
export async function connectPlatform(
    platform: string
    , leagueId: string
    , divisionId: string | null
): Promise<PlatformConnectResponse> {
    return jsonRequest(
        `${BASE_URL}/platforms/${encodeURIComponent(platform)}/connect`
        , 'Platform connect'
        , {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ league_id: leagueId, division_id: divisionId }),
        }
    )
}

/** Polls the current draft/roster state from the connected platform. */
export async function fetchDraftState(
    sessionId: string
    , mode: string
): Promise<DraftStateResponse> {
    return jsonRequest(
        `${BASE_URL}/sessions/${sessionId}/draft-state?mode=${encodeURIComponent(mode)}`
        , 'Draft state'
    )
}
