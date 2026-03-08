// api/client.ts
// HTTP client for the Fantasy Basketball Optimizer backend.
// All fetch calls go through this module; callers receive typed results.

import { Player, PlayerGScore, SessionRequest } from '../types.js'

export const BASE_URL = 'http://127.0.0.1:8000'

// ── Map backend Candidate → frontend Player ────────────────────────────────────

/** Converts raw backend Candidate objects to frontend Player objects, remapping snake_case keys to camelCase. */
export function candidatesToPlayers(candidates: any[]): Player[] {
    return candidates.map((c, i) => ({
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
                    a ? { name: a.name, isCandidate: a.is_candidate } : null,
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

// ── POST /data/upload ─────────────────────────────────────────────────────────

/** Uploads a projection CSV (HTB or BBM format) and returns a data_id for later reference. */
export async function uploadCsv(
    file: File,
    fileType: 'HTB' | 'BBM',
): Promise<{ data_id: string; file_type: string; n_players: number; expires_at: string }> {
    const form = new FormData()
    form.append('file', file)
    form.append('file_type', fileType)
    const res = await fetch(`${BASE_URL}/data/upload`, { method: 'POST', body: form })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Upload failed (${res.status}): ${detail}`)
    }
    return res.json()
}

// ── GET /seasons ───────────────────────────────────────────────────────────────

/** Fetches the list of available historical seasons from the backend. */
export async function getSeasons(): Promise<string[]> {
    const res = await fetch(`${BASE_URL}/seasons`)
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Get seasons failed (${res.status}): ${detail}`)
    }
    const data = await res.json()
    return data.seasons as string[]
}

// ── POST /sessions ─────────────────────────────────────────────────────────────

/** Creates a new backend session, running the full 5-step pipeline. Returns session_id and resolved categories. */
export async function createSession(
    req: SessionRequest,
    signal?: AbortSignal,
): Promise<{ session_id: string; categories: string[]; g_scores: PlayerGScore[]; n_players_loaded: number; expires_at: string }> {
    const res = await fetch(`${BASE_URL}/sessions`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Create session failed (${res.status}): ${detail}`)
    }
    return res.json()
}

// ── PATCH /sessions/{id} ──────────────────────────────────────────────────────

/* This function might fail if it is called by itself, outside of CreateorPatchSession.
That function has logic to handle the potential for this function to fail due to 
an expired session
*/
export async function patchSession(
    sessionId: string,
    req: Record<string, unknown>,
    signal?: AbortSignal,
): Promise<{ ok: boolean; steps_rerun: number[] }> {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
        signal,
    })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Patch session failed (${res.status}): ${detail}`)
    }
    return res.json()
}

// ── POST /sessions/{id}/evaluate ──────────────────────────────────────────────

/** Runs the H-score algorithm for the given draft/auction state and returns ranked candidates. Supports AbortSignal for cancellation. */
export async function evaluate(
    sessionId: string,
    req: {
        player_assignments: Record<string, string[]>
        my_team_id: string
        exclusion_list?: string[]
        remaining_cash?: Record<string, number>
    },
    signal?: AbortSignal,
): Promise<{ iteration: number; candidates: any[] }> {
    const body = { exclusion_list: [], ...req }
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}/evaluate`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
        signal,
    })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Evaluate failed (${res.status}): ${detail}`)
    }
    return res.json()
}
