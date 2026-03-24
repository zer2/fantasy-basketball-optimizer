// Collects: slot_counts, bench_slots
// Mirrors position_requirement_popover() in src/parameter_collection/position_requirement.py
//
// Position types and default slot counts are loaded from the backend config
// (parameters.yaml) via getSportConfig(). Hard-coded fallbacks are used
// only if the config fetch failed.

import { makeLabel, makeNumberInput } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

const FALLBACK_BASE: string[] = ['PG', 'SG', 'SF', 'PF', 'C']
const FALLBACK_FLEX: string[] = ['G', 'F', 'Util']
const FALLBACK_DEFAULTS: Record<string, number> = {
    PG: 1, SG: 1, SF: 1, PF: 1, C: 2,
    G: 2, F: 2, Util: 3,
}

/** Resolves base positions, flex positions, and default slot counts from config. */
function resolvePositionConfig(): {
    basePositions: string[]
    flexPositions: string[]
    defaults: Record<string, number>
} {
    const config = getSportConfig()
    if (!config) {
        return { basePositions: FALLBACK_BASE, flexPositions: FALLBACK_FLEX, defaults: FALLBACK_DEFAULTS }
    }

    const basePositions = config.position_structure.base_list
    const flexPositions = config.position_structure.flex_list

    // Find the default slot counts from positions[n_picks]
    const nPicksDefault = config.options?.n_picks?.default
    const posEntry = nPicksDefault != null ? config.positions[String(nPicksDefault)] : null

    const defaults: Record<string, number> = {}
    if (posEntry) {
        for (const [pos, count] of Object.entries(posEntry.base)) defaults[pos] = count
        for (const [pos, count] of Object.entries(posEntry.flex)) defaults[pos] = count
    } else {
        // Fallback: all zeros
        for (const pos of [...basePositions, ...flexPositions]) defaults[pos] = 0
    }

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
        leftCol.append(makeSlotRow(pos, pref(`slot_${pos}`, defaults[pos] ?? 0)))
    }

    for (const pos of flexPositions) {
        rightCol.append(makeSlotRow(pos, pref(`slot_${pos}`, defaults[pos] ?? 0)))
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
    benchInput.addEventListener('change', () => savePref('bench_slots', parseInt(benchInput.value) || 0))
    container.append(benchInput)

    // Validation message
    const validationMsg = document.createElement('div')
    validationMsg.id = 'sc-validation'
    validationMsg.className = 'sidebar-error'
    container.append(validationMsg)

    // Update validation on any input change
    grid.addEventListener('input', () => validateSlotCounts(validationMsg, basePositions, flexPositions))
    validateSlotCounts(validationMsg, basePositions, flexPositions)
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
    input.addEventListener('change', () => savePref(`slot_${pos}`, parseInt(input.value) || 0))
    row.append(input)

    return row
}

/**
 * Validates that the sum of all slot counts does not exceed picks-per-drafter.
 * Writes an error message to `msgEl` if over budget, or clears it if valid.
 */
function validateSlotCounts(msgEl: HTMLElement, basePositions: string[], flexPositions: string[]): void {
    const counts = getSlotCountsFromPositions(basePositions, flexPositions)
    const total = Object.values(counts).reduce((a, b) => a + b, 0)
    const nPicksEl = document.getElementById('ls-n-picks') as HTMLInputElement | null
    if (!nPicksEl) return
    const nPicks = parseInt(nPicksEl.value)
    if (isNaN(total) || isNaN(nPicks)) return
    msgEl.textContent = total > nPicks
        ? `Slot total (${total}) exceeds picks per drafter (${nPicks}).`
        : ''
}

/** Reads slot counts for a given set of positions from the DOM. */
function getSlotCountsFromPositions(basePositions: string[], flexPositions: string[]): Record<string, number> {
    const result: Record<string, number> = {}
    for (const pos of [...basePositions, ...flexPositions]) {
        const input = document.getElementById(`sc-${pos.toLowerCase()}`) as HTMLInputElement | null
        result[pos] = input ? (parseInt(input.value) || 0) : 0
    }
    return result
}

/**
 * Returns the current slot count for each position type.
 * Keys are position names (e.g. `"PG"`, `"G"`, `"Util"`); values are integer counts.
 */
export function getSlotCounts(): Record<string, number> {
    const config = getSportConfig()
    const basePositions = config?.position_structure.base_list ?? FALLBACK_BASE
    const flexPositions = config?.position_structure.flex_list ?? FALLBACK_FLEX
    return getSlotCountsFromPositions(basePositions, flexPositions)
}

/**
 * Returns the number of bench slots to exclude from optimization.
 * These are roster spots that the platform counts but the optimizer should ignore.
 */
export function getBenchSlots(): number {
    return parseInt((document.getElementById('sc-bench-slots') as HTMLInputElement).value) || 0
}
