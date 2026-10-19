// Collects: trading evaluation parameters
// The threshold inputs and position toggle are rendered inline by season_trading.ts.
// Combo sizes are fixed (DEFAULT_COMBOS) and not user-editable.

interface TradeComboRow {
    n_traded:   number
    n_received: number
}

export const DEFAULT_COMBOS: TradeComboRow[] = [
    { n_traded: 1, n_received: 1 },
    { n_traded: 2, n_received: 2 },
    { n_traded: 3, n_received: 3 },
]

export const POINTS_COMBOS: TradeComboRow[] = [
    { n_traded: 1, n_received: 1 },
    { n_traded: 2, n_received: 1 },
    { n_traded: 1, n_received: 2 },
    { n_traded: 2, n_received: 2 },
]

/** One round of the suggestion search: the value window it covers, in percentage points of summed general value (base
 *  H-score) between the two sides. `threshold` is the widest gap searched (null: every gap); gaps within
 *  `searchedThreshold` were searched by the round before (null: none was). */
export interface TradeSearchRound {
    threshold:         number | null
    searchedThreshold: number | null
}

/** The suggestion search runs in bands of value gap, each across every team: within 1 point first, then 1 to 2, and so
 *  on to 10, then everything beyond 10. The closest trades -- most of the good ones -- show first. No good trade in
 *  the league measured (2026-10-13: 960 across 1-for-1 to 3-for-3) had a gap past 0.084, so the last band is there
 *  only so nothing is left out by assumption. */
export const TRADE_SEARCH_ROUNDS: TradeSearchRound[] = [
    ...Array.from({ length: 10 }, (_, band) => ({ threshold: band + 1, searchedThreshold: band === 0 ? null : band })),
    { threshold: null, searchedThreshold: 10 },
]

/** The lowest threshold (a percentage of H-score) either side may be filtered at. Every search runs at this floor and
 *  the thresholds filter what it found, so changing one never searches again; the backend refuses anything lower
 *  (TRADE_THRESHOLD_FLOOR there). At the floor a 16-team league's 3-for-3 search found 101,480 trades (2026-10-13). */
export const TRADE_THRESHOLD_FLOOR_PERCENT = -2

/** Suggestions drawn at a time; "Show more" draws the next batch. */
export const SUGGESTION_ROWS_PER_PAGE = 100
