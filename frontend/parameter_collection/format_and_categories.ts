// Collects: scoring_format, categories
// Mirrors format_popover() in src/parameter_collection/format.py
//
// Available categories and defaults are loaded from the backend config
// (parameters.yaml) via getSportConfig(). Throws if the config is not loaded.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, renderMultiselect } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

// Live reference to the multiselect's selected-items array.
// renderMultiselect returns the array it mutates, so this reference always
// reflects the current state without re-querying the DOM.
let _selectedCategories: string[]

// Set to true while syncCategoriesFromBackend is mutating the DOM so the
// MutationObserver skips saving prefs / dispatching a change event.
let _suppressCategoryEvents = false

/**
 * Renders the Format & Categories section: scoring format selector and
 * a chip-style multiselect for stat categories.
 */
export function renderFormatAndCategories(container: HTMLElement): void {
    const config = getSportConfig()
    if (!config) throw new Error('Sport config not loaded')
    const allCategories     = config.all_categories
    const defaultCategories = config.default_categories

    const savedCategories = pref<string[] | null>('categories', null)
    const initialCategories = savedCategories ?? defaultCategories

    _selectedCategories = [...initialCategories]

    // Scoring format
    container.append(makeLabel('fc-scoring-format', 'Scoring format'))
    const fmtSelect = makeCustomSelect(
        'fc-scoring-format',
        ['Head to Head: Each Category', 'Head to Head: Most Categories', 'Rotisserie']
            .map(format => ({ value: format, label: format })),
        pref('scoring_format', 'Head to Head: Each Category'),
    )
    fmtSelect.element.addEventListener('change', () => savePref('scoring_format', fmtSelect.getValue()))
    container.append(fmtSelect.element)

    // Categories multiselect
    const catLabel = document.createElement('div')
    catLabel.className = 'sidebar-label'
    catLabel.textContent = 'Categories'
    container.append(catLabel)

    _selectedCategories = renderMultiselect(
        container,
        allCategories,
        initialCategories,
    )

    // Save categories on chip add/remove (observe DOM mutations on the chip area)
    const inputArea = container.querySelector('.ms-input-area')
    if (inputArea) {
        new MutationObserver(() => {
            if (_suppressCategoryEvents) return
            savePref('categories', [..._selectedCategories])
            container.dispatchEvent(new Event('change', { bubbles: true }))
        }).observe(inputArea, { childList: true })
    }
}

/**
 * Removes any categories not in `backendCategories` from both `_selectedCategories`
 * and the chip DOM, without triggering a session patch or preference save.
 * Call this after session creation when the backend has filtered unavailable categories.
 */
export function syncCategoriesFromBackend(backendCategories: string[]): void {
    const backendSet = new Set(backendCategories)
    const invalidIndices: number[] = []
    for (let i = _selectedCategories.length - 1; i >= 0; i--) {
        if (!backendSet.has(_selectedCategories[i])) invalidIndices.push(i)
    }
    if (invalidIndices.length === 0) return

    _suppressCategoryEvents = true
    try {
        // Remove stale chips from DOM first (chip order mirrors _selectedCategories)
        const inputArea = document.querySelector('.ms-input-area')
        if (inputArea) {
            const chips = Array.from(inputArea.querySelectorAll<HTMLElement>('.ms-chip'))
            for (const idx of invalidIndices) {
                chips[idx]?.remove()
            }
        }
        // Mutate the live array in-place to match
        for (const idx of invalidIndices) {
            _selectedCategories.splice(idx, 1)
        }
        savePref('categories', [..._selectedCategories])
    } finally {
        _suppressCategoryEvents = false
    }
}

export function getScoringFormat(): string {
    return (document.getElementById('fc-scoring-format') as HTMLInputElement).value
}

/** Returns a snapshot of the currently selected stat categories, ordered canonically — the order
 *  they are listed in the parameters (percentage/ratio stats then counting stats, via the config's
 *  all_categories) — rather than the chip/insertion order of the multiselect. This keeps category
 *  order independent of how the user happened to arrange them in the picker. */
export function getSelectedCategories(): string[] {
    const canonicalOrder = getSportConfig()?.all_categories ?? []
    const orderIndex = new Map(canonicalOrder.map((category, index) => [category, index]))
    return [..._selectedCategories].sort(
        (a, b) => (orderIndex.get(a) ?? Infinity) - (orderIndex.get(b) ?? Infinity),
    )
}
