// Force-weighting: the per-category pin boxes that sit in the candidate table's header.
//
// A pinned category is held at the weight the user typed and excluded from the optimiser's descent,
// constraining it into a chosen build. Blank means the algorithm sets that weight as usual.
//
// The numbers are on the same "100 = neutral" scale the expand view displays, so a pin round-trips
// exactly: type 40 and the displayed weight reads 40.0. See HAgent.set_forced_category_weights.
//
// The pins live in this module and in saved preferences, NOT in the inputs themselves, because
// buildTableHeader() throws the whole header away and rebuilds it on every settings change — values
// left only in the DOM would silently vanish the first time an unrelated sidebar control moved.

import { pref, savePref } from '../preferences.js'

const PREFERENCE_KEY = 'forced_category_weights'

/** category -> the typed weight, for categories the user has pinned. Blank boxes are absent. */
let pinnedWeights: Record<string, number> = readSavedWeights()

function readSavedWeights(): Record<string, number> {
    const saved = pref<Record<string, unknown>>(PREFERENCE_KEY, {})
    if (saved === null || typeof saved !== 'object') return {}
    const cleaned: Record<string, number> = {}
    for (const [category, value] of Object.entries(saved)) {
        const asNumber = typeof value === 'number' ? value : parseFloat(String(value))
        if (Number.isFinite(asNumber)) cleaned[category] = asNumber
    }
    return cleaned
}

/** The pins as the evaluate request carries them, or undefined when nothing is pinned — so an
 *  unpinned session sends exactly the request it sends today. */
export function getForcedCategoryWeights(): Record<string, number> | undefined {
    return Object.keys(pinnedWeights).length > 0 ? { ...pinnedWeights } : undefined
}

/** True when force-weighting is switched on in Model Parameters. Read from saved preferences
 *  rather than the DOM so the table header can be built before the sidebar exists. */
export function isForceWeightingAllowed(): boolean {
    return pref<boolean>('allow_force_weighting', false)
}

/** Builds the header row of pin boxes, one cell per column, or returns null when force-weighting is
 *  off — in which case the table looks exactly as it always has.
 *  `leadingColumnCount` is the number of non-category columns to leave blank (Player plus either
 *  H-Score or the four auction dollar columns). */
export function buildForcedWeightsRow(
    categories: string[]
  , leadingColumnCount: number
): HTMLTableRowElement | null {
    if (!isForceWeightingAllowed()) return null

    const row = document.createElement('tr')
    row.className = 'forced-weights-row'

    for (let column = 0; column < leadingColumnCount; column++) {
        const blank = document.createElement('th')
        blank.className = 'tableheader forced-weights-label'
        // The label goes in the last leading cell, closest to the boxes it explains.
        if (column === leadingColumnCount - 1) blank.textContent = 'Pin'
        row.append(blank)
    }

    for (const category of categories) {
        const cell = document.createElement('th')
        cell.className = 'tableheader'

        const input = document.createElement('input')
        input.type = 'number'
        input.className = 'forced-weight-input'
        input.min = '0'
        input.step = '5'
        input.placeholder = 'auto'
        input.title = `Pin ${category}'s weight (100 = neutral emphasis). Blank lets the algorithm `
                    + `choose it.`
        input.value = category in pinnedWeights ? String(pinnedWeights[category]) : ''
        input.addEventListener('change', () => {
            const typed = input.value.trim()
            const parsed = parseFloat(typed)
            if (typed === '' || !Number.isFinite(parsed) || parsed < 0) {
                // A blank, or anything that is not a usable number, releases the category back to
                // the algorithm. Rewriting the box makes that visible rather than leaving junk text
                // sitting in a field the backend is ignoring.
                delete pinnedWeights[category]
                input.value = ''
            } else {
                pinnedWeights[category] = parsed
            }
            savePref(PREFERENCE_KEY, pinnedWeights)
            // An event rather than a direct call: the re-evaluate lives in main.ts, and importing it
            // here would close a cycle (main -> player_table -> force_weights -> main). Same pattern
            // the platform-connected hand-off uses.
            document.dispatchEvent(new Event('forced-weights-changed'))
        })

        cell.append(input)
        row.append(cell)
    }
    return row
}
