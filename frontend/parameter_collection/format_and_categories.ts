// Collects: scoring_format, categories
// Mirrors format_popover() in src/parameter_collection/format.py
//
// Available categories are sport-level server-side config; hardcoded here for NBA.
// The categories widget mimics st.multiselect: chips for selected items,
// text-filter dropdown for adding more.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, renderMultiselect } from '../helper_functions.js'

const SCORING_FORMAT_OPTIONS: { label: string; value: string }[] = [
    { label: 'Head to Head: Each Category',   value: 'Head to Head: Each Category'   },
    { label: 'Head to Head: Most Categories', value: 'Head to Head: Most Categories' },
    { label: 'Rotisserie',                    value: 'Rotisserie'                    },
]

const ALL_CATEGORIES: string[] = [
    // Ratio
    'Field Goal %', 'Free Throw %', 'Three %',
    // Counting
    'Threes', 'Points', 'Rebounds', 'Off Rebounds', 'Def Rebounds',
    'Assists', 'Steals', 'Blocks', 'Turnovers',
    'Double Doubles', 'Field Goals Made', 'Free Throws Made',
]

const DEFAULT_CATEGORIES: string[] = [
    'Field Goal %', 'Free Throw %', 'Threes', 'Points',
    'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers',
]

// Live reference to the multiselect's selected-items array.
// renderMultiselect returns the array it mutates, so this reference always
// reflects the current state without re-querying the DOM.
let _selectedCategories: string[] = [...DEFAULT_CATEGORIES]

/**
 * Renders the Format & Categories section: scoring format selector and
 * a chip-style multiselect for stat categories.
 */
export function renderFormatAndCategories(container: HTMLElement): void {

    // Scoring format
    container.append(makeLabel('fc-scoring-format', 'Scoring format'))
    const fmtSelect = makeCustomSelect(
        'fc-scoring-format',
        SCORING_FORMAT_OPTIONS,
        'Head to Head: Each Category',
    )
    container.append(fmtSelect.element)

    // Categories multiselect
    const catLabel = document.createElement('div')
    catLabel.className = 'sidebar-label'
    catLabel.textContent = 'Categories'
    container.append(catLabel)

    _selectedCategories = renderMultiselect(
        container,
        ALL_CATEGORIES,
        DEFAULT_CATEGORIES,
    )
}

/**
 * Returns the selected scoring format and the current list of active stat categories.
 * `categories` is a live snapshot of the chip selections at the time of calling.
 */
export function getFormatAndCategories(): { scoring_format: string; categories: string[] } {
    const scoring_format = (document.getElementById('fc-scoring-format') as HTMLInputElement).value
    return { scoring_format, categories: [..._selectedCategories] }
}
