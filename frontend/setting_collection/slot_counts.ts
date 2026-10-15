// Collects: slot_counts, bench_slots
// Mirrors position_requirement_popover() in src/setting_collection/position_requirement.py
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
    benchInput.addEventListener('change', savePositionStructure)
    container.append(benchInput)

    // Validation message
    const validationMsg = document.createElement('div')
    validationMsg.id = 'sc-validation'
    validationMsg.className = 'sidebar-error'
    container.append(validationMsg)

    // Update validation on any input change, the perma-bench count included
    grid.addEventListener('input', () => validateSlotCounts(validationMsg))
    benchInput.addEventListener('input', () => validateSlotCounts(validationMsg))
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
    input.addEventListener('change', savePositionStructure)
    row.append(input)

    return row
}

/** The slot total, the perma-bench count, and the picks-per-drafter the two must add up to. A
 *  missing sport config or a missing input element is a programmer error and throws (fail
 *  noisily); NaN values from in-progress typing are a user state and flow through for the
 *  callers to judge. */
function readSlotTotalAndPicks(): { total: number; bench: number; nPicks: number } {
    const config = getSportConfig()
    if (!config) throw new Error('Sport config not loaded')
    const counts = getSlotCountsFromPositions(
        config.position_structure.base_list, config.position_structure.flex_list)
    const total = Object.values(counts).reduce((a, b) => a + b, 0)
    const nPicksEl = document.getElementById('ls-n-picks') as HTMLInputElement | null
    if (!nPicksEl) throw new Error('ls-n-picks element not found')
    const benchEl = document.getElementById('sc-bench-slots') as HTMLInputElement | null
    if (!benchEl) throw new Error('sc-bench-slots element not found')
    return { total, bench: parseInt(benchEl.value), nPicks: parseInt(nPicksEl.value) }
}

/**
 * Validates that the slot counts plus the perma-bench slots equal picks-per-drafter: every
 * pick fills either a position slot or a bench spot (the Streamlit app's rule). Before
 * 2026-10-08 the bench was left out of the sum, so a 13-slot, 3-bench, 16-pick league was
 * told its slots fell short.
 * Writes an error message to `msgEl` if mismatched, or clears it if valid.
 */
function validateSlotCounts(msgEl: HTMLElement): void {
    const { total, bench, nPicks } = readSlotTotalAndPicks()
    if (isNaN(total) || isNaN(bench) || isNaN(nPicks)) return
    const spots = total + bench
    const breakdown = bench > 0 ? `Slot total (${total}) plus perma-bench (${bench})` : `Slot total (${total})`
    if (spots > nPicks) {
        msgEl.textContent = `${breakdown} exceeds picks per drafter (${nPicks}).`
    } else if (spots < nPicks) {
        msgEl.textContent = `${breakdown} is less than picks per drafter (${nPicks}). Increase slot counts, add bench slots or reduce picks.`
    } else {
        msgEl.textContent = ''
    }
}

/**
 * Returns true if the current slot count total plus the perma-bench slots equals the current
 * picks-per-drafter value.
 * Call this before applying slot count or n_picks changes to the backend. Throws (rather
 * than quietly answering false) when the config or the inputs are missing — those are
 * programmer errors, not invalid user input.
 */
export function isSlotCountsValid(): boolean {
    const { total, bench, nPicks } = readSlotTotalAndPicks()
    return total + bench === nPicks
}

/** How many of a team's picks are active -- the slot total; picks past it are perma-bench. The
 *  backend scores each team on its first this-many picks (services/ranking.split_off_bench). */
export function getActiveSlotCount(): number {
    return Object.values(getSlotCounts()).reduce((a, b) => a + b, 0)
}

// ─── Where the position structure is saved ────────────────────────────────────
// Manual entry and each connected league keep their own: manual entry in the slot_<pos> and bench_slots preferences, a
// league in league_shape:<platform>:<league id>. Connecting a league never touches manual entry's, so switching back
// to "Enter your own data" restores exactly what was there; a hand edit while connected is saved to that league.
// (They used to share one set, which a connection overwrote with the league's slots while its pick count went
// unsaved: manual entry came back with its own picks and the league's slots, which did not fit.)

/** A connected league's saved structure: the slots its platform last reported (null for a platform that reports
 *  none), and the structure in use for it -- that report, or the user's edit of it. */
interface LeaguePositionStructure {
    reported: Record<string, number> | null
    slots: Record<string, number>
    bench: number
}

let connectedLeagueKey: string | null = null   // whose structure is on screen; null = manual entry

/** Saves the structure on screen to its owner: manual entry's preferences, or the connected league's entry. */
function savePositionStructure(): void {
    const slots = getSlotCounts()
    const bench = parseInt((document.getElementById('sc-bench-slots') as HTMLInputElement).value)
    if (connectedLeagueKey === null) {
        for (const [pos, count] of Object.entries(slots)) savePref(`slot_${pos}`, count)
        savePref('bench_slots', bench)
        return
    }
    const saved = pref<LeaguePositionStructure | null>(connectedLeagueKey, null)
    if (saved === null) throw new Error(`No saved structure for the connected league ${connectedLeagueKey}`)
    savePref(connectedLeagueKey, { ...saved, slots, bench })
}

/** Puts a structure in the Position Parameters inputs (a slot it does not name is zero) and revalidates. */
function showPositionStructure(slots: Record<string, number>, bench: number): void {
    const config = getSportConfig()
    if (!config) throw new Error('Sport config not loaded')
    const positions = [...config.position_structure.base_list, ...config.position_structure.flex_list]
    const unknown = Object.keys(slots).filter(slot => !positions.includes(slot))
    if (unknown.length > 0) throw new Error(`A position structure names slots this sport does not have: ${unknown.join(', ')}`)
    for (const pos of positions) {
        const input = document.getElementById(`sc-${pos.toLowerCase()}`) as HTMLInputElement | null
        if (!input) throw new Error(`Slot count input sc-${pos.toLowerCase()} not found`)
        input.value = String(slots[pos] ?? 0)
    }
    const benchInput = document.getElementById('sc-bench-slots') as HTMLInputElement | null
    if (!benchInput) throw new Error('Perma-bench input sc-bench-slots not found')
    benchInput.value = String(bench)
    revalidateSlotCounts()
}

/** Manual entry's own structure, from its preferences (the config's default where never set), back on screen and in
 *  charge of hand edits again. */
export function showManualPositionStructure(): void {
    const { basePositions, flexPositions, defaults } = resolvePositionConfig()
    connectedLeagueKey = null
    showPositionStructure(
        Object.fromEntries([...basePositions, ...flexPositions].map(pos => [pos, pref(`slot_${pos}`, defaults[pos])])),
        pref('bench_slots', 0),
    )
}

/**
 * Shows a connected league's structure and puts it in charge of hand edits. The league's platform report is the
 * default, not an override: a structure saved for this league stays -- the user's edits included -- as long as the
 * league still reports the slots it reported when that was saved and the structure fits its picks. Otherwise the
 * report is applied fresh with no perma-bench (the platform's slots already account for every pick; Yahoo's bench
 * arrives as Util slots), or for a platform that reports no slots, the default structure parameters.yaml defines for
 * the league's pick count (left as it is, with the mismatch stated, when it defines none).
 */
export function showLeaguePositionStructure(
    platform: string
    , leagueId: string
    , reportedSlots: Record<string, number> | null
    , nPicks: number
): void {
    connectedLeagueKey = `league_shape:${platform}:${leagueId}`
    const saved = pref<LeaguePositionStructure | null>(connectedLeagueKey, null)
    if (saved !== null && haveSameReport(saved.reported, reportedSlots)) {
        showPositionStructure(saved.slots, saved.bench)
        if (isSlotCountsValid()) return
    }
    if (reportedSlots !== null) {
        showPositionStructure(reportedSlots, 0)
    } else {
        const defaults = getDefaultSlotCountsForPicks(nPicks)
        if (defaults !== null) showPositionStructure(defaults, 0)
        else revalidateSlotCounts()
    }
    savePref(connectedLeagueKey, {
        reported: reportedSlots,
        slots: getSlotCounts(),
        bench: parseInt((document.getElementById('sc-bench-slots') as HTMLInputElement).value),
    })
}

/** The default structure parameters.yaml defines for a pick count (13, 14, 15, 16 for the NBA), or null if none. */
function getDefaultSlotCountsForPicks(nPicks: number): Record<string, number> | null {
    const config = getSportConfig()
    if (!config) throw new Error('Sport config not loaded')
    const posEntry = config.positions[String(nPicks)]
    return posEntry ? { ...posEntry.base, ...posEntry.flex } : null
}

/** Whether two platform reports are the same (both none, or the same count for every slot type). */
function haveSameReport(
    first: Record<string, number> | null
    , second: Record<string, number> | null
): boolean {
    if (first === null || second === null) return first === second
    const slotTypes = new Set([...Object.keys(first), ...Object.keys(second)])
    return [...slotTypes].every(slotType => (first[slotType] ?? 0) === (second[slotType] ?? 0))
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

