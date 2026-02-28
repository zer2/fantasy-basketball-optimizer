// Collects: slot_counts, bench_slots
// Mirrors position_requirement_popover() in src/parameter_collection/position_requirement.py
//
// Renamed from position_requirement.py to slot_counts.ts to reflect that
// the user specifies counts (e.g. C: 2) rather than an explicit slot list.
// Valid position types are sport-level config; hardcoded here for NBA.

import { makeLabel, makeNumberInput } from '../helper_functions.js'

const BASE_POSITIONS: string[] = ['PG', 'SG', 'SF', 'PF', 'C']
const FLEX_POSITIONS: string[] = ['G', 'F', 'Util']

const DEFAULTS: Record<string, number> = {
    PG: 1, SG: 1, SF: 1, PF: 1, C: 2,
    G: 2, F: 2, Util: 3,
}

/**
 * Renders the Position Parameters section: slot counts for each position type,
 * a bench slots field, and a live validation warning.
 */
export function renderSlotCounts(container: HTMLElement): void {

    const warning = document.createElement('div')
    warning.className = 'sidebar-caption'
    warning.textContent = 'Slot counts must sum to the number of starter slots (picks minus bench).'
    container.append(warning)

    const grid = document.createElement('div')
    grid.className = 'slot-grid'
    container.append(grid)

    const leftCol = document.createElement('div')
    const leftHeader = document.createElement('div')
    leftHeader.className = 'sidebar-label'
    leftHeader.textContent = 'Base positions'
    leftCol.append(leftHeader)
    grid.append(leftCol)

    const rightCol = document.createElement('div')
    const rightHeader = document.createElement('div')
    rightHeader.className = 'sidebar-label'
    rightHeader.textContent = 'Flex positions'
    rightCol.append(rightHeader)
    grid.append(rightCol)

    for (const pos of BASE_POSITIONS) {
        leftCol.append(makeSlotRow(pos))
    }

    for (const pos of FLEX_POSITIONS) {
        rightCol.append(makeSlotRow(pos))
    }

    // Bench slots
    container.append(makeLabel('sc-bench-slots', 'Bench slots'))

    const benchCaption = document.createElement('div')
    benchCaption.className = 'sidebar-caption'
    benchCaption.textContent =
        'Roster spots to treat as bench (e.g. if your platform shows 16 total slots but ' +
        'only 13 are start/sit relevant, set this to 3). These are excluded from the optimizer.'
    container.append(benchCaption)

    container.append(makeNumberInput('sc-bench-slots', 0, 0))

    // Validation message
    const validationMsg = document.createElement('div')
    validationMsg.id = 'sc-validation'
    validationMsg.className = 'sidebar-error'
    container.append(validationMsg)

    // Update validation on any input change
    grid.addEventListener('input', () => validateSlotCounts(validationMsg))
    validateSlotCounts(validationMsg)
}

/** Creates a labelled number input row for a single position slot count. */
function makeSlotRow(pos: string): HTMLElement {
    const row = document.createElement('div')
    row.className = 'slot-row'

    const label = document.createElement('label')
    label.htmlFor = `sc-${pos.toLowerCase()}`
    label.textContent = pos
    row.append(label)

    const input = document.createElement('input')
    input.type = 'number'
    input.id = `sc-${pos.toLowerCase()}`
    input.className = 'sidebar-input slot-input'
    input.min = '0'
    input.value = String(DEFAULTS[pos])
    row.append(input)

    return row
}

/**
 * Validates that the sum of all slot counts does not exceed picks-per-drafter.
 * Writes an error message to `msgEl` if over budget, or clears it if valid.
 */
function validateSlotCounts(msgEl: HTMLElement): void {
    const counts = getSlotCounts()
    const total = Object.values(counts).reduce((a, b) => a + b, 0)
    const nPicksEl = document.getElementById('ls-n-picks') as HTMLInputElement | null
    if (!nPicksEl) return
    const nPicks = parseInt(nPicksEl.value)
    if (isNaN(total) || isNaN(nPicks)) return
    msgEl.textContent = total > nPicks
        ? `Slot total (${total}) exceeds picks per drafter (${nPicks}).`
        : ''
}

/**
 * Returns the current slot count for each position type.
 * Keys are position names (e.g. `"PG"`, `"G"`, `"Util"`); values are integer counts.
 */
export function getSlotCounts(): Record<string, number> {
    const result: Record<string, number> = {}
    for (const pos of [...BASE_POSITIONS, ...FLEX_POSITIONS]) {
        const input = document.getElementById(`sc-${pos.toLowerCase()}`) as HTMLInputElement
        result[pos] = parseInt(input.value) || 0
    }
    return result
}

/**
 * Returns the number of bench slots to exclude from optimization.
 * These are roster spots that the platform counts but the optimizer should ignore.
 */
export function getBenchSlots(): number {
    return parseInt((document.getElementById('sc-bench-slots') as HTMLInputElement).value) || 0
}
