// Collects: scoring_format, categories
// Mirrors format_popover() in src/parameter_collection/format.py
//
// Available categories and defaults are loaded from the backend config
// (parameters.yaml) via getSportConfig(). Hard-coded fallbacks are used
// only if the config fetch failed.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, renderMultiselect } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

const SCORING_FORMAT_OPTIONS: { label: string; value: string }[] = [
    { label: 'Head to Head: Each Category',   value: 'Head to Head: Each Category'   },
    { label: 'Head to Head: Most Categories', value: 'Head to Head: Most Categories' },
    { label: 'Rotisserie',                    value: 'Rotisserie'                    },
]

const FALLBACK_ALL_CATEGORIES: string[] = [
    'Field Goal %', 'Free Throw %', 'Three %',
    'Threes', 'Points', 'Rebounds', 'Off Rebounds', 'Def Rebounds',
    'Assists', 'Steals', 'Blocks', 'Turnovers',
    'Double Doubles', 'Field Goals Made', 'Free Throws Made',
]

const FALLBACK_DEFAULT_CATEGORIES: string[] = [
    'Field Goal %', 'Free Throw %', 'Threes', 'Points',
    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
]

// Live reference to the multiselect's selected-items array.
// renderMultiselect returns the array it mutates, so this reference always
// reflects the current state without re-querying the DOM.
let _selectedCategories: string[] = []

/**
 * Renders the Format & Categories section: scoring format selector and
 * a chip-style multiselect for stat categories.
 */
export function renderFormatAndCategories(container: HTMLElement): void {
    const config = getSportConfig()
    const allCategories     = config?.all_categories     ?? FALLBACK_ALL_CATEGORIES
    const defaultCategories = config?.default_categories  ?? FALLBACK_DEFAULT_CATEGORIES

    const savedCategories = pref<string[] | null>('categories', null)
    const initialCategories = savedCategories ?? defaultCategories

    _selectedCategories = [...initialCategories]

    // Scoring format
    container.append(makeLabel('fc-scoring-format', 'Scoring format'))
    const fmtSelect = makeCustomSelect(
        'fc-scoring-format',
        SCORING_FORMAT_OPTIONS,
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
        new MutationObserver(() => savePref('categories', [..._selectedCategories]))
            .observe(inputArea, { childList: true })
    }
}

/**
 * Returns the selected scoring format and the current list of active stat categories.
 * `categories` is a live snapshot of the chip selections at the time of calling.
 */
export function getFormatAndCategories(): { scoring_format: string; categories: string[] } {
    const scoring_format = (document.getElementById('fc-scoring-format') as HTMLInputElement).value
    return { scoring_format, categories: [..._selectedCategories] }
}
