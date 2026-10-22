// table/table_helpers.ts
// Shared helpers for the panel tables (expand view, G-score table).

/** Creates an invisible `<th>` spacer used to lock column widths in panel tables. */
export function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const spacer = document.createElement('th')
    spacer.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return spacer
}

// The colour scale on "Your $" by the team's cash-to-cost ratio r (backend/services/auction_pricing.py): no colour
// below CASH_PRESSURE_START, then CASH_PRESSURE_STEPS steps from amber to red, the last reached at CASH_PRESSURE_FULL
// and held above it. The smooth rule raises values from a little below r = 1, so the colour starts there too; red
// from r = 1, where the team holds more cash than its best options cost.
const CASH_PRESSURE_START = 0.7
const CASH_PRESSURE_FULL = 1.0
const CASH_PRESSURE_STEPS = 5

/** The styles.css class for a "Your $" at this cash-to-cost ratio: '' below the start of the scale, else
 *  'cash-pressure cash-pressure-N' with N from 1 (amber) deepening across the scale, and CASH_PRESSURE_STEPS (red)
 *  from CASH_PRESSURE_FULL up. */
export function classifyCashPressure(ratio: number | null): string {
    if (ratio === null || ratio < CASH_PRESSURE_START) return ''
    const position = (ratio - CASH_PRESSURE_START) / (CASH_PRESSURE_FULL - CASH_PRESSURE_START)
    if (position >= 1) return `cash-pressure cash-pressure-${CASH_PRESSURE_STEPS}`
    const step = 1 + Math.floor(position * (CASH_PRESSURE_STEPS - 1))
    return `cash-pressure cash-pressure-${step}`
}

/** The hover text on a coloured "Your $": how the team's cash compares with what its best options cost, and how
 *  much of this value comes from that cash. */
export function describeCashPressure(ratio: number, share: number): string {
    return `Values increasing to match your remaining money supply: your cash is ${Math.round(100 * ratio)}% of `
        + `what your best options cost, adding $${share.toFixed(1)} to this player.`
}
