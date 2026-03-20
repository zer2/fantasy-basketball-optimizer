// data_entry/auction_state.ts
// Pure state for the auction board. No DOM access; no imports from session or layout.
// Imported by auction_entry.ts (UI rendering) and session.ts (evaluate requests).

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AuctionPick { player: string; cost: number }

export interface AuctionConfig {
    nDrafters:   number
    nPicks:      number
    cashPerTeam: number
    teamNames:   string[]
    key:         string
}

// ─── Module state ─────────────────────────────────────────────────────────────

let picks:     (AuctionPick | null)[][] = []   // [row][drafter]
let history:   [number, number][]       = []   // undo stack: [row, drafter]
let teamNames: string[]
let nDrafters:   number
let nPicks:      number
let cashPerTeam: number
let configKey   = ''   // detects sidebar changes that require a reset

// ─── Getters ──────────────────────────────────────────────────────────────────

export function getPicks():       (AuctionPick | null)[][] { return picks       }
export function getHistory():     [number, number][]       { return history     }
export function getTeamNames():   string[]                 { return teamNames   }
export function getNDrafters():   number                   { return nDrafters   }
export function getNPicks():      number                   { return nPicks      }
export function getCashPerTeam(): number                   { return cashPerTeam }
export function getConfigKey():   string                   { return configKey   }

// ─── Config ───────────────────────────────────────────────────────────────────

/** Resets all picks and history; clears configKey so the next render re-applies config. */
export function resetAuctionState(): void {
    picks     = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
    history   = []
    configKey = ''
}

/** Applies a new league config, resetting all pick data. */
export function applyAuctionConfig(cfg: AuctionConfig): void {
    picks       = Array.from({ length: cfg.nPicks }, () => Array(cfg.nDrafters).fill(null))
    history     = []
    teamNames   = cfg.teamNames
    nDrafters   = cfg.nDrafters
    nPicks      = cfg.nPicks
    cashPerTeam = cfg.cashPerTeam
    configKey   = cfg.key
}

// ─── Pick mutations ───────────────────────────────────────────────────────────

/**
 * Records a pick for the given drafter in the first empty row.
 * Returns true if successful, false if the team is already full.
 */
export function recordAuctionPick(player: string, cost: number, drafterIndex: number): boolean {
    const emptyRow = picks.findIndex(r => r[drafterIndex] === null)
    if (emptyRow === -1) return false
    picks[emptyRow][drafterIndex] = { player, cost }
    history.push([emptyRow, drafterIndex])
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
    return true
}

export function clearAllAuctionPicks(): void {
    picks   = Array.from({ length: nPicks }, () => Array(nDrafters).fill(null))
    history = []
}

// ─── Derived state ────────────────────────────────────────────────────────────

/** Returns the current auction state shaped for /evaluate requests. */
export function getAuctionState(): {
    player_assignments: Record<string, string[]>
    remaining_cash:     Record<string, number>
} {
    const player_assignments: Record<string, string[]> = {}
    const remaining_cash:     Record<string, number>   = {}
    for (let d = 0; d < nDrafters; d++) {
        const name      = teamNames[d] ?? `Drafter ${d + 1}`
        const teamPicks = picks.map(row => row[d]).filter(Boolean) as AuctionPick[]
        player_assignments[name] = teamPicks.map(p => p.player)
        const spent              = teamPicks.reduce((sum, p) => sum + p.cost, 0)
        remaining_cash[name]     = cashPerTeam - spent
    }
    return { player_assignments, remaining_cash }
}
