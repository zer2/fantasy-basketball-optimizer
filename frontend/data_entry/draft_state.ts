// data_entry/draft_state.ts
// Pure state for the draft board. No DOM access; no imports from session or layout.
// Imported by draft_board.ts (UI rendering) and session.ts (evaluate requests).

// ─── Types ────────────────────────────────────────────────────────────────────

export interface DraftConfig {
    nDrafters: number
    nPicks:    number
    teamNames: string[]
    key:       string
}

// ─── Module state ─────────────────────────────────────────────────────────────

let pickRow     = 0
let pickDrafter = 0
let drafted: (string | null)[][] = []   // [row][drafter], null = not yet picked
let teamNames:  string[] = []
let nDrafters = 0
let nPicks    = 0
let configKey = ''   // detects sidebar changes that require a reset

// ─── Getters ──────────────────────────────────────────────────────────────────

export function getPickRow():     number              { return pickRow     }
export function getPickDrafter(): number              { return pickDrafter }
export function getDrafted():     (string | null)[][] { return drafted     }
export function getTeamNames():   string[]            { return teamNames   }
export function getNDrafters():   number              { return nDrafters   }
export function getNPicks():      number              { return nPicks      }
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
    teamNames   = cfg.teamNames
    nDrafters   = cfg.nDrafters
    nPicks      = cfg.nPicks
    configKey   = cfg.key
}

// ─── Pick mutations ───────────────────────────────────────────────────────────

export function recordDraftPick(row: number, drafter: number, player: string): void {
    drafted[row][drafter] = player
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

/** Advance pick position one step in serpentine order. */
export function advanceDraftPick(): void {
    const isForward = pickRow % 2 === 0   // even rounds: left→right
    if (isForward) {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    } else {
        if (pickDrafter > 0) { pickDrafter--; return }
    }
    // Move to next round
    if (pickRow < nPicks - 1) {
        pickRow++
        pickDrafter = pickRow % 2 === 0 ? 0 : nDrafters - 1
    } else {
        pickRow = nPicks   // draft complete sentinel
    }
}

/** Move pick position back one step in serpentine order. */
export function goBackDraftPick(): void {
    if (pickRow >= nPicks) { pickRow = nPicks - 1 }
    const isForward = pickRow % 2 === 0
    if (isForward) {
        if (pickDrafter > 0) { pickDrafter--; return }
    } else {
        if (pickDrafter < nDrafters - 1) { pickDrafter++; return }
    }
    if (pickRow > 0) {
        pickRow--
        const prevForward = pickRow % 2 === 0
        pickDrafter = prevForward ? nDrafters - 1 : 0
    }
}

// ─── Derived state ────────────────────────────────────────────────────────────

/** Returns the current draft state shaped for /evaluate requests. */
export function getDraftState(): { player_assignments: Record<string, string[]> } {
    const player_assignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const name = teamNames[d] ?? `Drafter ${d + 1}`
        player_assignments[name] = drafted.map(row => row[d]).filter(Boolean) as string[]
    }
    return { player_assignments }
}
