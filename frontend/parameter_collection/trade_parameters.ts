// Collects: trading evaluation parameters
// The threshold inputs and position toggle are rendered inline by season_trading.ts.
// Combo sizes are fixed (DEFAULT_COMBOS) and not user-editable.

export interface TradeComboRow {
    n_traded:   number
    n_received: number
    threshold:  number
}

export const DEFAULT_COMBOS: TradeComboRow[] = [
    { n_traded: 1, n_received: 1, threshold: 2 },
    { n_traded: 2, n_received: 2, threshold: 2 },
    { n_traded: 3, n_received: 3, threshold: 2 },
]
