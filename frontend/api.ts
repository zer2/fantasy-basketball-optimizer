// api.ts
// HTTP client for the Fantasy Basketball Optimizer backend.
// All fetch calls go through this module; callers receive typed results.

import { Player, SessionRequest } from './types.js'

export const BASE_URL = 'http://127.0.0.1:8000'

// ── Map backend Candidate → frontend Player ────────────────────────────────────

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
        flex_allocations: {
            base_positions: c.flex_allocations.base_positions,
            rows: c.flex_allocations.rows.map((r: any) => ({
                label:   r.label,
                values:  r.values,
                isTotal: r.is_total,
            })),
        },
        roster: {
            slots: c.roster.slots,
            assignments: Object.fromEntries(
                Object.entries(c.roster.assignments).map(([slot, a]: [string, any]) => [
                    slot,
                    a ? { name: a.name, isCandidate: a.is_candidate } : null,
                ]),
            ),
        },
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

// ── POST /sessions ─────────────────────────────────────────────────────────────

export async function createSession(
    req: SessionRequest,
): Promise<{ session_id: string; categories: string[]; n_players_loaded: number; expires_at: string }> {
    const res = await fetch(`${BASE_URL}/sessions`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Create session failed (${res.status}): ${detail}`)
    }
    return res.json()
}

// ── PATCH /sessions/{id} ──────────────────────────────────────────────────────

export async function patchSession(
    sessionId: string,
    req: Record<string, unknown>,
): Promise<{ ok: boolean; steps_rerun: number[] }> {
    const res = await fetch(`${BASE_URL}/sessions/${sessionId}`, {
        method:  'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(req),
    })
    if (!res.ok) {
        const detail = await res.text()
        throw new Error(`Patch session failed (${res.status}): ${detail}`)
    }
    return res.json()
}

// ── POST /sessions/{id}/evaluate ──────────────────────────────────────────────

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
