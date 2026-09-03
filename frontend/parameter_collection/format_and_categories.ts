// Collects: scoring_format, categories
// Mirrors format_popover() in src/parameter_collection/format.py
//
// Available categories and defaults are loaded from the backend config
// (parameters.yaml) via getSportConfig(). Throws if the config is not loaded.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, makeWeightSlider, renderMultiselect } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

// Live reference to the multiselect's selected-items array.
// renderMultiselect returns the array it mutates, so this reference always
// reflects the current state without re-querying the DOM.
let _selectedCategories: string[]

// Head to Head used to be two formats. It is now one format plus a dial: how much of the
// objective is winning the majority of categories (Most Categories) rather than each category on
// its own (Each Category). The two old names are the dial's endpoints, so a stored preference
// from before the change maps onto the new pair exactly.
const HEAD_TO_HEAD = 'Head to Head'
const ROTISSERIE   = 'Rotisserie'

const LEGACY_FORMAT_ENDPOINTS: Record<string, { format: string; weight: number }> = {
    'Head to Head: Each Category':   { format: HEAD_TO_HEAD, weight: 0 },
    'Head to Head: Most Categories': { format: HEAD_TO_HEAD, weight: 1 },
}

/** The stored format and dial, translating a preference written before the two Head-to-Head
 *  formats became one. Rotisserie has no dial. */
function readStoredFormatAndWeight(): { format: string; weight: number } {
    const storedFormat = pref('scoring_format', HEAD_TO_HEAD)
    const legacy = LEGACY_FORMAT_ENDPOINTS[storedFormat]
    if (legacy) {
        // Write the translation back, so this runs once rather than on every load. Left in place,
        // the old string would keep mapping to its endpoint and overwrite the dial each time the
        // page opened — the slider could be set repeatedly and would always come back at the
        // endpoint, which is not a caching artefact but this function never finishing its job.
        savePref('scoring_format', legacy.format)
        savePref('most_categories_weight', legacy.weight)
        return legacy
    }
    const storedWeight = pref('most_categories_weight', 0)
    return { format: storedFormat, weight: storedWeight }
}

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

    // Scoring format, and — for Head to Head — how its two objectives are mixed.
    const storedFormat = readStoredFormatAndWeight()

    // The ⓘ sits on the section's own label: it explains the format and its objective together,
    // which is one idea, not two.
    const formatLabelRow = document.createElement('div')
    formatLabelRow.className = 'param-label-row'
    formatLabelRow.append(makeLabel('fc-scoring-format', 'Scoring format'), makeObjectiveInfoButton())
    container.append(formatLabelRow)

    const fmtSelect = makeCustomSelect(
        'fc-scoring-format',
        [HEAD_TO_HEAD, ROTISSERIE].map(format => ({ value: format, label: format })),
        storedFormat.format,
    )
    container.append(fmtSelect.element)

    const weightRow = makeMostCategoriesWeightRow(storedFormat.weight)
    container.append(weightRow)

    const showWeightForFormat = (): void => {
        weightRow.style.display = fmtSelect.getValue() === ROTISSERIE ? 'none' : ''
    }
    showWeightForFormat()
    fmtSelect.element.addEventListener('change', () => {
        savePref('scoring_format', fmtSelect.getValue())
        showWeightForFormat()
    })

    // Categories multiselect
    const catLabel = document.createElement('div')
    catLabel.className = 'sidebar-label'
    catLabel.textContent = 'Categories'
    container.append(catLabel)

    _selectedCategories = renderMultiselect(
        container,
        allCategories.map(category => ({ value: category, label: category })),
        initialCategories,
    )

    // Tiebreaker: only reachable when a matchup can actually tie — an even number of categories,
    // scored by majority. Its options are the categories currently in play, so it is built after
    // the multiselect above.
    const tiebreakerRow = makeTiebreakerRow()
    container.append(tiebreakerRow)
    refreshTiebreakerControl()
    fmtSelect.element.addEventListener('change', refreshTiebreakerControl)
    weightRow.addEventListener('input', refreshTiebreakerControl)

    // Save categories on chip add/remove (observe DOM mutations on the chip area)
    const inputArea = container.querySelector('.ms-input-area')
    if (inputArea) {
        new MutationObserver(() => {
            if (_suppressCategoryEvents) return
            savePref('categories', [..._selectedCategories])
            refreshTiebreakerControl()
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

// The tiebreaker select, and the row that hides it when a tie cannot arise. Module-level so the
// category multiselect's change handler can refresh the options without threading references.
const NO_TIEBREAKER = ''
let tiebreakerSelect: ReturnType<typeof makeCustomSelect> | null = null

/** The tiebreaker control: a category that counts twice, settling matchups that would otherwise
 *  be tied. Sits under the categories it chooses from. */
function makeTiebreakerRow(): HTMLElement {
    const row = document.createElement('div')
    row.id = 'fc-tiebreaker-row'

    const labelRow = document.createElement('div')
    labelRow.className = 'param-label-row'
    labelRow.append(makeLabel('fc-tiebreaker', 'Tiebreaker'))

    const infoButton = document.createElement('button')
    infoButton.type = 'button'
    infoButton.className = 'info-btn'
    infoButton.textContent = 'ⓘ'
    infoButton.dataset.tooltip =
        'With an even number of categories a matchup can end level. Naming a tiebreaker settles '
        + 'it: that category counts twice, so whoever wins it takes an otherwise tied matchup. '
        + 'Matchups that are not level are unaffected. Leave it empty to treat a level matchup as '
        + 'half a win.'
    labelRow.append(infoButton)
    row.append(labelRow)

    tiebreakerSelect = makeCustomSelect('fc-tiebreaker', [{ value: NO_TIEBREAKER, label: 'None' }])
    tiebreakerSelect.element.addEventListener('change', () => {
        // Remembered even while the control is hidden, so that returning to an even number of
        // categories brings the choice back rather than making the user find it again.
        savePref('tiebreaker_category', tiebreakerSelect!.getValue())
    })
    row.append(tiebreakerSelect.element)
    return row
}

/** Whether a tiebreaker can currently apply: majority scoring with an even category count.
 *  The row's visibility AND the value sent to the backend both derive from this one
 *  predicate, so the request payload can never depend on how the row happens to be hidden. */
function tiebreakerCanApply(): boolean {
    return getScoringFormat() !== ROTISSERIE
        && getSelectedCategories().length % 2 === 0
        && (getMostCategoriesWeight() ?? 0) > 0
}

/** Shows the tiebreaker only where it can bite — majority scoring, even category count — and
 *  keeps its options in step with the categories in play. */
function refreshTiebreakerControl(): void {
    const row = document.getElementById('fc-tiebreaker-row')
    if (row === null || tiebreakerSelect === null) return

    const categories = getSelectedCategories()
    const applies = tiebreakerCanApply()
    row.style.display = applies ? '' : 'none'
    if (!applies) return

    const remembered = pref('tiebreaker_category', NO_TIEBREAKER)
    tiebreakerSelect.setOptions(
        [{ value: NO_TIEBREAKER, label: 'None' },
         ...categories.map(category => ({ value: category, label: category }))],
        categories.includes(remembered) ? remembered : NO_TIEBREAKER,
    )
}

/** The category that counts twice, or null when no tie can arise (Rotisserie, an odd number of
 *  categories, or a purely per-category objective) — the backend rejects a value it would ignore. */
export function getTiebreakerCategory(): string | null {
    if (tiebreakerSelect === null || !tiebreakerCanApply()) return null
    const selected = tiebreakerSelect.getValue()
    return selected === NO_TIEBREAKER ? null : selected
}

/** Explains the format selector and the objective dial beneath it — one control in two parts. */
function makeObjectiveInfoButton(): HTMLButtonElement {
    const infoButton = document.createElement('button')
    infoButton.type = 'button'
    infoButton.className = 'info-btn'
    infoButton.textContent = 'ⓘ'
    infoButton.dataset.tooltip =
        'Head to Head scoring supports both the total number of categories won (Each Category) and the  '
        + 'probability of winning a majority of categories (Most Categories). The slider controls '
        + 'how much each kind of scoring is weighed. Setting it in the middle is sensible for leagues '
        + 'which determine regular season standings by total category wins, and playoff matchups '
        + 'based on who wins a majority of categories. Rotisserie scoring is standalone since it '
        + 'has a unique mechanism for determining the winner.'
    return infoButton
}

/** The Head-to-Head objective dial: a slider from scoring every category (0) to scoring only the
 *  majority (1), with both ends named above the track by the format names people already know. */
function makeMostCategoriesWeightRow(initialWeight: number): HTMLElement {
    const row = document.createElement('div')
    row.id = 'fc-objective-row'

    const endpointRow = document.createElement('div')
    endpointRow.className = 'objective-endpoint-row'

    const eachCategoryLabel = document.createElement('label')
    eachCategoryLabel.className   = 'objective-endpoint-label'
    eachCategoryLabel.htmlFor     = 'fc-most-categories-weight'
    eachCategoryLabel.textContent = 'Each Category'

    const mostCategoriesLabel = document.createElement('span')
    mostCategoriesLabel.className   = 'objective-endpoint-label'
    mostCategoriesLabel.textContent = 'Most Categories'

    endpointRow.append(eachCategoryLabel, mostCategoriesLabel)

    const sliderRow = document.createElement('div')
    sliderRow.className = 'sidebar-slider-row'

    const { slider, valueDisplay } = makeWeightSlider('fc-most-categories-weight', initialWeight)
    slider.addEventListener('change', () => savePref('most_categories_weight', parseFloat(slider.value)))

    sliderRow.append(slider, valueDisplay)
    row.append(endpointRow, sliderRow)
    return row
}

export function getScoringFormat(): string {
    return (document.getElementById('fc-scoring-format') as HTMLInputElement).value
}

/** How much of the Head-to-Head objective is winning the majority of categories. Null under
 *  Rotisserie, which scores neither way — the backend rejects a number there rather than
 *  ignoring it. */
export function getMostCategoriesWeight(): number | null {
    if (getScoringFormat() === ROTISSERIE) return null
    return parseFloat((document.getElementById('fc-most-categories-weight') as HTMLInputElement).value)
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
