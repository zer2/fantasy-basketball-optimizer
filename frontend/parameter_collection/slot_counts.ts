// Collects: slot_counts, bench_slots
// Mirrors position_requirement_popover() in src/parameter_collection/position_requirement.py
//
// Position types and default slot counts are loaded from the backend config
// (parameters.yaml) via getSportConfig().

import { makeLabel, makeNumberInput } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

/** Resolves base positions, flex positions, and default slot counts from config. */
function resolvePositionConfig(): {
    basePositions: string[]
    flexPositions: string[]
    defaults: Record<string, number>
} {
    const config = getSportConfig()
    if (!config) throw new Error('resolvePositionConfig called before sport config loaded')

    const basePositions = config.position_structure.base_list
    const flexPositions = config.position_structure.flex_list

    const nPicksDefault = config.options.n_picks.default
    const posEntry = config.positions[String(nPicksDefault)]
    if (!posEntry) throw new Error(`No position entry found for n_picks default ${nPicksDefault}`)

    const defaults = { ...posEntry.base, ...posEntry.flex }

    return { basePositions, flexPositions, defaults }
}

/**
 * Renders the Position Parameters section: slot counts for each position type,
 * a bench slots field, and a live validation warning.
 */
export function renderSlotCounts(container: HTMLElement): void {
    const { basePositions, flexPositions, defaults } = resolvePositionConfig()

    const warning = document.createElement('div')
    warning.className = 'sidebar-caption'
    warning.textContent = `The H-scoring algorithm will choose players assuming that
                            its team ultimately needs to fit this structure. This
                            includes players who will start on the bench, but will
                            be rotated to real slots when they have games.`
    container.append(warning)

    const grid = document.createElement('div')
    grid.className = 'slot-grid'
    grid.dataset.testid = 'position-structure'
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

    for (const pos of basePositions) {
        leftCol.append(makeSlotRow(pos, pref(`slot_${pos}`, defaults[pos])))
    }

    for (const pos of flexPositions) {
        rightCol.append(makeSlotRow(pos, pref(`slot_${pos}`, defaults[pos])))
    }

    // Bench slots
    container.append(makeLabel('sc-bench-slots', 'Perma-bench slots'))

    const benchCaption = document.createElement('div')
    benchCaption.className = 'sidebar-caption'
    benchCaption.textContent =
        'Roster spots to treat as bench (e.g. if your platform shows 16 total slots but ' +
        'only 13 are start/sit relevant, set this to 3). These are excluded from the optimizer.'
    container.append(benchCaption)

    const benchInput = makeNumberInput('sc-bench-slots', pref('bench_slots', 0), 0)
    benchInput.addEventListener('change', () => savePref('bench_slots', parseInt(benchInput.value)))
    container.append(benchInput)

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
function makeSlotRow(pos: string, defaultValue: number): HTMLElement {
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
    input.value = String(defaultValue)
    input.addEventListener('change', () => savePref(`slot_${pos}`, parseInt(input.value)))
    row.append(input)

    return row
}

/** The slot total and the picks-per-drafter it must equal. A missing sport config or a
 *  missing input element is a programmer error and throws (fail noisily); NaN values from
 *  in-progress typing are a user state and flow through for the callers to judge. */
function readSlotTotalAndPicks(): { total: number; nPicks: number } {
    const config = getSportConfig()
    if (!config) throw new Error('Sport config not loaded')
    const counts = getSlotCountsFromPositions(
        config.position_structure.base_list, config.position_structure.flex_list)
    const total = Object.values(counts).reduce((a, b) => a + b, 0)
    const nPicksEl = document.getElementById('ls-n-picks') as HTMLInputElement | null
    if (!nPicksEl) throw new Error('ls-n-picks element not found')
    return { total, nPicks: parseInt(nPicksEl.value) }
}

/**
 * Validates that the sum of all slot counts equals picks-per-drafter.
 * Writes an error message to `msgEl` if mismatched, or clears it if valid.
 */
function validateSlotCounts(msgEl: HTMLElement): void {
    const { total, nPicks } = readSlotTotalAndPicks()
    if (isNaN(total) || isNaN(nPicks)) return
    if (total > nPicks) {
        msgEl.textContent = `Slot total (${total}) exceeds picks per drafter (${nPicks}).`
    } else if (total < nPicks) {
        msgEl.textContent = `Slot total (${total}) is less than picks per drafter (${nPicks}). Increase slot counts or reduce picks.`
    } else {
        msgEl.textContent = ''
    }
}

/**
 * Returns true if the current slot count total equals the current picks-per-drafter value.
 * Call this before applying slot count or n_picks changes to the backend. Throws (rather
 * than quietly answering false) when the config or the inputs are missing — those are
 * programmer errors, not invalid user input.
 */
export function isSlotCountsValid(): boolean {
    const { total, nPicks } = readSlotTotalAndPicks()
    return total === nPicks
}

/**
 * Re-runs the slot count validation against the current n_picks value.
 * Call this when n_picks changes in League Settings so the error stays in sync.
 */
export function revalidateSlotCounts(): void {
    const msgEl = document.getElementById('sc-validation')
    if (!msgEl) throw new Error('sc-validation element not found')
    validateSlotCounts(msgEl)
}

/** Reads slot counts for a given set of positions from the DOM. */
function getSlotCountsFromPositions(basePositions: string[], flexPositions: string[]): Record<string, number> {
    const result: Record<string, number> = {}
    for (const pos of [...basePositions, ...flexPositions]) {
        const input = document.getElementById(`sc-${pos.toLowerCase()}`) as HTMLInputElement | null
        if (!input) throw new Error(`Slot count input sc-${pos.toLowerCase()} not found`)
        result[pos] = parseInt(input.value)
    }
    return result
}

/**
 * Returns the current slot count for each position type.
 * Keys are position names (e.g. `"PG"`, `"G"`, `"Util"`); values are integer counts.
 */
export function getSlotCounts(): Record<string, number> {
    const config = getSportConfig()
    if (!config) throw new Error('getSlotCounts called before sport config loaded')
    const basePositions = config.position_structure.base_list
    const flexPositions = config.position_structure.flex_list
    return getSlotCountsFromPositions(basePositions, flexPositions)
}

