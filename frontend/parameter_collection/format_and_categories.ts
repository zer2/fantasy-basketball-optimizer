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
            savePref('categories', [..._selectedCategories])
            container.dispatchEvent(new Event('change', { bubbles: true }))
        }).observe(inputArea, { childList: true })
    }
}

export function getScoringFormat(): string {
    return (document.getElementById('fc-scoring-format') as HTMLInputElement).value
}

/** Returns a snapshot of the currently selected stat categories. */
export function getSelectedCategories(): string[] {
    return [..._selectedCategories]
}
