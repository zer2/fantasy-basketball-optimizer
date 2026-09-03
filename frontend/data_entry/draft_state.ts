// data_entry/draft_state.ts
// Pure state for the draft board. No DOM access; no imports from session or layout.
// Imported by draft_board.ts (UI rendering) and session.ts (evaluate requests).

// ─── Types ────────────────────────────────────────────────────────────────────

export interface DraftConfig {
    nDrafters:           number
    nPicks:              number
    teamNames:           string[]
    thirdRoundReversal:  boolean
    key:                 string
}

// ─── Module state ─────────────────────────────────────────────────────────────

let pickRow     = 0
let pickDrafter = 0
let drafted: (number | null)[][] = []   // [row][drafter] player id, null = not yet picked
let teamNames:          string[]
let nDrafters:          number
let nPicks:             number
let thirdRoundReversal: boolean
let configKey = ''   // detects sidebar changes that require a reset

// ─── Getters ──────────────────────────────────────────────────────────────────

export function getPickRow():     number              { return pickRow     }
export function getPickDrafter(): number              { return pickDrafter }
export function getDrafted():     (number | null)[][] { return drafted     }
export function getTeamIdentitiesFromBoard():   string[]            { return teamNames   }
export function getNDrafters(): number { return nDrafters }
export function getNPicks():    number { return nPicks    }
export function getConfigKey():   string              { return configKey   }

// ─── Config ───────────────────────────────────────────────────────────────────

/** Resets pick position and all picks; clears configKey so the next render re-applies config. */
export function resetDraftState(): void {
    pickRow     = 0
    pickDrafter = 0
    drafted     = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
    configKey   = ''
}

/** Applies a new league config, resetting all pick data. */
export function applyDraftConfig(cfg: DraftConfig): void {
    pickRow     = 0
    pickDrafter = 0
    drafted     = Array.from({ length: cfg.nPicks }, () => Array(cfg.nDrafters).fill(null))
    teamNames          = cfg.teamNames
    nDrafters          = cfg.nDrafters
    nPicks             = cfg.nPicks
    thirdRoundReversal = cfg.thirdRoundReversal
    configKey          = cfg.key
}

// ─── Pick mutations ───────────────────────────────────────────────────────────

export function recordDraftPick(row: number, drafter: number, playerId: number): void {
    drafted[row][drafter] = playerId
}

export function clearDraftPick(row: number, drafter: number): void {
    drafted[row][drafter] = null
}

export function clearAllDraftPicks(): void {
    pickRow     = 0
    pickDrafter = 0
    drafted     = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
}

// ─── Pick position navigation ─────────────────────────────────────────────────

/** Returns whether the given row goes left→right (forward) under the current reversal setting. */
function isForwardRow(row: number): boolean {
    if (thirdRoundReversal) {
        return row < 2 ? row % 2 === 0 : row % 2 === 1
    }
    return row % 2 === 0
}

/** Advance pick position one step in serpentine order.
 *
 *  The third-round-reversal decision is consumed exactly when the draft steps past the second
 *  round, so the in-effect value refreshes from the sidebar setting at that moment — toggling
 *  the sidebar before the boundary is crossed always applies. Past the boundary the stored
 *  value keeps governing navigation (an undo must retrace the path the draft actually took),
 *  until an undo or clear brings the position back and a later crossing refreshes it again. */
export function advanceDraftPick(thirdRoundReversalSetting: boolean): void {
    const atSecondRoundBoundary = pickRow === 1 && pickDrafter === 0
    if (atSecondRoundBoundary) thirdRoundReversal = thirdRoundReversalSetting

    // Third-round reversal: end of row 1 jumps directly to row 2 at the far end
    if (thirdRoundReversal && atSecondRoundBoundary) {
        pickRow     = 2
        pickDrafter = nDrafters - 1
        return
    }

    const isForward = isForwardRow(pickRow)
    if (isForward) {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    } else {
        if (pickDrafter > 0) { pickDrafter--; return }
    }
    // Move to next round
    if (pickRow < nPicks - 1) {
        pickRow++
        pickDrafter = isForwardRow(pickRow) ? 0 : nDrafters - 1
    } else {
        pickRow = nPicks   // draft complete sentinel
    }
}

/** Move pick position back one step in serpentine order. */
export function goBackDraftPick(): void {
    if (pickRow >= nPicks) { pickRow = nPicks - 1 }

    // Third-round reversal: start of row 2 jumps back to end of row 1
    if (thirdRoundReversal && pickRow === 2 && pickDrafter === nDrafters - 1) {
        pickRow     = 1
        pickDrafter = 0
        return
    }

    const isForward = isForwardRow(pickRow)
    if (isForward) {
        if (pickDrafter > 0) { pickDrafter--; return }
    } else {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    }
    if (pickRow > 0) {
        pickRow--
        pickDrafter = isForwardRow(pickRow) ? nDrafters - 1 : 0
    }
}

// ─── Derived state ────────────────────────────────────────────────────────────

/** Returns the current draft state shaped for /evaluate requests. */
export function getDraftState(): { player_assignments: Record<string, number[]> } {
    const player_assignments: Record<string, number[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const name = teamNames[d] ?? `Team ${d + 1}`
        player_assignments[name] = drafted
            .map(row => row[d])
            .filter(playerId => playerId !== null) as number[]
    }
    return { player_assignments }
}
