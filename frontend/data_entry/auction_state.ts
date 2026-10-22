// data_entry/auction_state.ts
// Pure state for the auction board. No DOM access; no imports from session or layout.
// Imported by auction_entry.ts (UI rendering) and session.ts (evaluate requests).

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AuctionPick { playerId: number; cost: number }

export interface AuctionConfig {
    nDrafters:   number
    nPicks:      number
    cashPerTeam: number
    teamNames:   string[]
    key:         string
}

// ─── Module state ─────────────────────────────────────────────────────────────

let picks:     (AuctionPick | null)[][] = []   // [row][drafter]
let history:   [number, number, number][] = [] // undo stack: [row, drafter, nominator]
let teamNames: string[]
let nDrafters:   number
let nPicks:      number
let cashPerTeam: number
let configKey   = ''   // detects sidebar changes that require a reset
let nominatorIndex = 0

// ─── Getters ──────────────────────────────────────────────────────────────────

export function getPicks():       (AuctionPick | null)[][] { return picks       }
export function getHistory():     [number, number, number][] { return history     }
export function getTeamIdentitiesFromBoard():   string[]                 { return teamNames   }
export function getNDrafters():   number                   { return nDrafters   }
export function getNPicks():      number                   { return nPicks      }
export function getCashPerTeam(): number                   { return cashPerTeam }
export function getConfigKey():   string                   { return configKey   }
export function getNominatorIndex(): number               { return nominatorIndex }
export function setNominatorIndex(index: number): void    { nominatorIndex = index }

export function isAuctionComplete(): boolean {
    if (nDrafters <= 0 || nPicks <= 0) return false
    for (let d = 0; d < nDrafters; d++) {
        if (picks.some(r => r[d] === null)) return false
    }
    return true
}

export function advanceNominatorIndex(): void {
    if (nDrafters <= 0) return
    for (let step = 1; step <= nDrafters; step++) {
        const next = (nominatorIndex + step) % nDrafters
        const isFull = picks.every(r => r[next] !== null)
        if (!isFull) {
            nominatorIndex = next
            return
        }
    }
}

// ─── Config ───────────────────────────────────────────────────────────────────

/** Resets all picks and history; clears configKey so the next render re-applies config. */
export function resetAuctionState(): void {
    picks          = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
    history        = []
    nominatorIndex = 0
    configKey      = ''
}

/** Applies a new league config, resetting all pick data. */
export function applyAuctionConfig(cfg: AuctionConfig): void {
    picks          = Array.from({ length: cfg.nPicks }, () => Array(cfg.nDrafters).fill(null))
    history        = []
    nominatorIndex = 0
    teamNames      = cfg.teamNames
    nDrafters      = cfg.nDrafters
    nPicks         = cfg.nPicks
    cashPerTeam    = cfg.cashPerTeam
    configKey      = cfg.key
}

// ─── Pick mutations ───────────────────────────────────────────────────────────

/**
 * Records a pick for the given drafter in the first empty row.
 * Returns true if successful, false if the team is already full.
 */
export function recordAuctionPick(playerId: number, cost: number, drafterIndex: number): boolean {
    const emptyRow = picks.findIndex(r => r[drafterIndex] === null)
    if (emptyRow === -1) return false
    picks[emptyRow][drafterIndex] = { playerId, cost }
    history.push([emptyRow, drafterIndex, nominatorIndex])
    advanceNominatorIndex()
    return true
}

/**
 * Removes the last pick from the board.
 * Returns true if a pick was undone, false if history is empty.
 */
export function undoLastAuctionPick(): boolean {
    const last = history.pop()
    if (!last) return false
    picks[last[0]][last[1]] = null
    nominatorIndex = last[2]
    return true
}

export function clearAllAuctionPicks(): boolean {
    const hadPicks = history.length > 0
    picks          = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
    history        = []
    nominatorIndex = 0
    return hadPicks
}

// ─── Derived state ────────────────────────────────────────────────────────────

/** Returns the current auction state shaped for /evaluate requests. */
export function getAuctionState(): {
    player_assignments: Record<string, number[]>
    remaining_cash:     Record<string, number>
} {
    const player_assignments: Record<string, number[]> = {}
    const remaining_cash:     Record<string, number>   = {}
    for (let d = 0; d < nDrafters; d++) {
        const name      = teamNames[d] ?? `Team ${d + 1}`
        const teamPicks = picks.map(row => row[d]).filter(Boolean) as AuctionPick[]
        player_assignments[name] = teamPicks.map(p => p.playerId)
        const spent              = teamPicks.reduce((sum, p) => sum + p.cost, 0)
        remaining_cash[name]     = cashPerTeam - spent
    }
    return { player_assignments, remaining_cash }
}
