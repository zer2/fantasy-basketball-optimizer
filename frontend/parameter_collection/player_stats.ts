// Collects: data_source (type, blend_weights, custom_data_ids), injured_players
// Mirrors player_stats_popover() in src/parameter_collection/player_stats.py

import { makeCustomSelect } from '../custom_select.js'
import { uploadCsv, getSeasons } from '../api/client.js'
import { DataSource } from '../types.js'
import { pref, savePref } from '../preferences.js'

// ─── Module state ──────────────────────────────────────────────────────────────

// One row per custom projection slot. The upload's data_id is its identity everywhere
// (blend-weight key, session requests); the title is presentation-only, like team labels
// on the draft board. Deliberately not persisted: uploads cannot survive a reload (they
// live in the backend's memory), so the whole structure resets with the page.
interface CustomUploadRow {
    dataId: string | null
    titleInput: HTMLInputElement
    slider: HTMLInputElement
    valueDisplay: HTMLSpanElement
    statusSpan: HTMLSpanElement
    uploadInput: HTMLInputElement
}

const MAX_CUSTOM_UPLOADS = 5
let customUploadRows: CustomUploadRow[] = []

/**
 * Marks every uploaded source as expired: clears its data_id, locks its weight back to
 * zero, and says so in its status line. Called when a session patch fails because a
 * data_id no longer exists server-side (backend restart, or the upload store's TTL).
 * Returns whether anything was cleared, so the caller can retry without the dead uploads.
 */
export function markUploadedSourcesExpired(): boolean {
    let clearedAny = false
    for (const row of customUploadRows) {
        if (row.dataId === null) continue
        row.dataId = null
        clearedAny = true
        row.slider.disabled = true
        row.slider.value = '0'
        row.valueDisplay.textContent = '0.00'
        row.statusSpan.textContent = 'Upload expired — please re-upload the file.'
        // Clear the input so re-selecting the same file fires a fresh change event.
        row.uploadInput.value = ''
    }
    return clearedAny
}

// Resolves when the in-flight historical-seasons fetch has finished populating the
// ps-season dropdown; already resolved when none is running. Covers BOTH the fetch
// kicked off at render (when the restored data source is 'historical') and the one
// kicked off by switching the data source to Historical later.
let _seasonsPromise: Promise<void> = Promise.resolve()

/** Returns a promise that resolves once the seasons dropdown is ready (immediately when
 *  no fetch is needed or one has already completed). Anything that reads the data source
 *  must await this first: until the fetch lands there is no `ps-season` element, and
 *  getPlayerStatsParams() refuses to report a historical source without a season. */
export function waitForSeasons(): Promise<void> {
    return _seasonsPromise
}

// ─── Render ───────────────────────────────────────────────────────────────────

/**
 * Renders the Player Stats section: data source selector, projection blend weight
 * sliders, optional CSV uploads, and injured/excluded player list.
 */
export function renderPlayerStats(container: HTMLElement): void {

    // Data source type
    const typeLabel = document.createElement('label')
    typeLabel.className = 'sidebar-label'
    typeLabel.htmlFor = 'ps-data-type'
    typeLabel.textContent = 'Data source'
    container.append(typeLabel)

    const typeSelect = makeCustomSelect(
        'ps-data-type',
        [
            { value: 'projections',   label: 'Projections'  },
            { value: 'historical', label: 'Historical'  },
        ],
        pref('data_source_type', 'historical'),
    )
    typeSelect.element.addEventListener('change', () => savePref('data_source_type', typeSelect.getValue()))
    container.append(typeSelect.element)

    // Projection blend weights section (only relevant for 'projections' type)
    const projSection = document.createElement('div')
    projSection.id = 'ps-proj-section'
    projSection.style.display = typeSelect.getValue() === 'projections' ? '' : 'none'
    container.append(projSection)

    renderBlendWeights(projSection)

    // Historical season selector (only relevant for 'historical' type)
    const histSection = document.createElement('div')
    histSection.id = 'ps-hist-section'
    histSection.style.display = typeSelect.getValue() === 'historical' ? '' : 'none'
    container.append(histSection)

    let seasonsLoaded = false
    /** Fetches available seasons from the backend and renders the season dropdown (once). */
    async function loadSeasons(): Promise<void> {
        if (seasonsLoaded) return
        const loadingEl = document.createElement('div')
        loadingEl.className = 'sidebar-caption'
        loadingEl.textContent = 'Loading seasons…'
        histSection.append(loadingEl)
        try {
            const seasons = await getSeasons()
            if (seasons.length === 0) throw new Error('Backend returned an empty season list')
            loadingEl.remove()
            const label = document.createElement('label')
            label.className = 'sidebar-label'
            label.htmlFor = 'ps-season'
            label.textContent = 'Season'
            histSection.append(label)
            const seasonSelect = makeCustomSelect(
                'ps-season',
                seasons.map(s => ({ value: s, label: s })),
                seasons[0],
            )
            histSection.append(seasonSelect.element)
            seasonsLoaded = true
        } catch (err) {
            loadingEl.textContent = `Failed to load seasons: ${err}`
            console.error('Failed to load seasons:', err)
        }
    }

    typeSelect.element.addEventListener('change', () => {
        const type = typeSelect.getValue()
        projSection.style.display = type === 'projections'    ? '' : 'none'
        histSection.style.display = type === 'historical' ? '' : 'none'
        // Published so the change handlers that react to this same event can await the
        // fetch; without it they read ps-season before the dropdown exists. Cheap to
        // re-assign — loadSeasons returns immediately once the seasons are in.
        if (type === 'historical') _seasonsPromise = loadSeasons()
    })

    // Load seasons immediately if restored type is 'historical'
    if (typeSelect.getValue() === 'historical') {
        _seasonsPromise = loadSeasons()
    }

    // Injured players
    const injuredLabel = document.createElement('label')
    injuredLabel.className = 'sidebar-label'
    injuredLabel.htmlFor = 'ps-injured'
    injuredLabel.textContent = 'Injured / excluded players'
    container.append(injuredLabel)

    const injuredInput = document.createElement('textarea')
    injuredInput.id = 'ps-injured'
    injuredInput.className = 'sidebar-input'
    injuredInput.placeholder = 'One player name per line'
    injuredInput.rows = 3
    container.append(injuredInput)
    
}

/** Renders the projection source weights: ESPN and DARKO sliders, then the custom
 *  projections section — up to five uploadable sources, each with an editable title,
 *  a weight slider locked until its upload succeeds, and a file chooser. A fresh empty
 *  row appears after each successful upload. */
function renderBlendWeights(container: HTMLElement): void {

    const weightLabel = document.createElement('div')
    weightLabel.className = 'sidebar-label'
    weightLabel.textContent = 'Projection blend weights'
    container.append(weightLabel)

    const snowflakeSources: { id: string; label: string; prefKey: string; defaultValue: number }[] = [
        { id: 'ps-w-espn',  label: 'ESPN',  prefKey: 'blend_w_espn',  defaultValue: 0.5 },
        { id: 'ps-w-darko', label: 'DARKO', prefKey: 'blend_w_darko', defaultValue: 0.5 },
    ]

    for (const source of snowflakeSources) {
        const row = document.createElement('div')
        row.className = 'sidebar-slider-row'

        const label = document.createElement('label')
        label.htmlFor = source.id
        label.textContent = source.label
        row.append(label)

        const savedWeight = pref(source.prefKey, source.defaultValue)
        const { slider, valueDisplay } = makeWeightSlider(source.id, savedWeight)
        slider.addEventListener('input', () => savePref(source.prefKey, parseFloat(slider.value)))
        row.append(slider, valueDisplay)
        container.append(row)
    }

    const customLabel = document.createElement('div')
    customLabel.className = 'sidebar-label'
    customLabel.textContent = 'Custom projections'
    container.append(customLabel)

    const customCaption = document.createElement('div')
    customCaption.className = 'sidebar-caption'
    customCaption.textContent =
        'Upload projection CSVs (Hashtag Basketball, Basketball Monster, or any file with ' +
        'standard stat columns — the format is detected automatically). Uploads last for ' +
        'this page visit only.'
    container.append(customCaption)

    const customRowsContainer = document.createElement('div')
    customRowsContainer.id = 'ps-custom-uploads'
    container.append(customRowsContainer)

    customUploadRows = []
    appendCustomUploadRow(customRowsContainer)
}

/** Builds a weight slider + its value display (shared by Snowflake and custom sources). */
function makeWeightSlider(
    id: string
    , initialWeight: number
): { slider: HTMLInputElement; valueDisplay: HTMLSpanElement } {
    const slider = document.createElement('input')
    slider.type = 'range'
    slider.id = id
    slider.min = '0'
    slider.max = '1'
    slider.step = '0.05'
    slider.value = String(initialWeight)

    const valueDisplay = document.createElement('span')
    valueDisplay.className = 'slider-value'
    valueDisplay.textContent = initialWeight.toFixed(2)

    slider.addEventListener('input', () => {
        valueDisplay.textContent = parseFloat(slider.value).toFixed(2)
    })
    return { slider, valueDisplay }
}

/** Appends one custom-projection row: [editable title | weight slider] + [file chooser | status].
 *  The slider stays locked at zero until this row's upload succeeds. */
function appendCustomUploadRow(customRowsContainer: HTMLElement): void {
    const rowNumber = customUploadRows.length + 1

    const sliderRow = document.createElement('div')
    sliderRow.className = 'sidebar-slider-row'

    // Editable display title, like team labels on the draft board: presentation only —
    // the upload's identity is its data_id. Title events stay out of the section's
    // change/input listeners so typing a name never triggers a session patch.
    const titleInput = document.createElement('input')
    titleInput.type = 'text'
    titleInput.id = `ps-custom-title-${rowNumber}`
    titleInput.className = 'sidebar-input custom-projection-title'
    titleInput.value = `Data ${rowNumber}`
    titleInput.addEventListener('input', event => event.stopPropagation())
    titleInput.addEventListener('change', event => event.stopPropagation())
    sliderRow.append(titleInput)

    const { slider, valueDisplay } = makeWeightSlider(`ps-w-custom-${rowNumber}`, 0)
    // A weight for a source with no file behind it is meaningless — locked until upload succeeds.
    slider.disabled = true
    sliderRow.append(slider, valueDisplay)
    customRowsContainer.append(sliderRow)

    const uploadRow = document.createElement('div')
    uploadRow.className = 'sidebar-upload-row'

    const uploadInput = document.createElement('input')
    uploadInput.type = 'file'
    uploadInput.id = `ps-upload-custom-${rowNumber}`
    uploadInput.accept = '.csv'
    uploadInput.className = 'sidebar-file-input'
    uploadRow.append(uploadInput)

    const statusSpan = document.createElement('span')
    statusSpan.className = 'sidebar-caption'
    uploadRow.append(statusSpan)
    customRowsContainer.append(uploadRow)

    const row: CustomUploadRow = { dataId: null, titleInput, slider, valueDisplay, statusSpan, uploadInput }
    customUploadRows.push(row)

    uploadInput.addEventListener('change', async () => {
        const file = uploadInput.files?.[0]
        if (!file) return
        statusSpan.textContent = 'Uploading…'
        const hadUploadAlready = row.dataId !== null
        try {
            const resp = await uploadCsv(file)
            row.dataId = resp.data_id
            // A source that pairs projections with a league only carries that league's
            // categories — say which standard stats this file lacks so a lighter file
            // reads as deliberate rather than as a parsing failure.
            const missingNote = resp.missing_stats.length > 0
                ? ` (no ${resp.missing_stats.join('/')})` : ''
            statusSpan.textContent = `✓ ${resp.n_players} players loaded${missingNote}`
            slider.disabled = false
            // Open the next slot once this one is filled (first upload into this row only).
            if (!hadUploadAlready && customUploadRows.length < MAX_CUSTOM_UPLOADS) {
                appendCustomUploadRow(customRowsContainer)
            }
        } catch (err) {
            row.dataId = null
            statusSpan.textContent = `Upload failed: ${err}`
            console.error('Custom projection upload failed:', err)
            // No file behind the source any more — lock its weight back to zero.
            slider.disabled = true
            slider.value = '0'
            valueDisplay.textContent = '0.00'
            // Browsers don't fire change when the same file is re-selected while the
            // input still holds it — and retrying the same file is the normal recovery
            // from a failure. Clear the input so the retry fires. On success the input
            // keeps the filename (clearing it would display "No file chosen" beside a
            // live upload); the expiry path clears it too, in markUploadedSourcesExpired.
            uploadInput.value = ''
        }
    })
}

// ─── Getter ───────────────────────────────────────────────────────────────────

/**
 * Reads data source type, blend weights, and excluded player list from the DOM.
 * custom_data_ids are populated by the CSV upload handlers above.
 */
export function getPlayerStatsParams(): { data_source: DataSource; injured_players: string[] } {
    const type = (document.getElementById('ps-data-type') as HTMLInputElement).value as DataSource['type']

    // Snowflake sources plus one entry per live upload, keyed by its data_id.
    const blend_weights: Record<string, number> = {
        ESPN:  parseFloat((document.getElementById('ps-w-espn')  as HTMLInputElement).value),
        DARKO: parseFloat((document.getElementById('ps-w-darko') as HTMLInputElement).value),
    }
    const custom_data_ids: string[] = []
    for (const row of customUploadRows) {
        if (row.dataId === null) continue
        custom_data_ids.push(row.dataId)
        blend_weights[row.dataId] = parseFloat(row.slider.value)
    }

    let season: string | null = null
    if (type === 'historical') {
        const seasonEl = document.getElementById('ps-season') as HTMLInputElement | null
        if (!seasonEl || !seasonEl.value) {
            throw new Error('Historical data source selected but #ps-season is missing or empty')
        }
        season = seasonEl.value
    }
    const data_source: DataSource = {
        type,
        blend_weights,
        custom_data_ids,
        season,
    }

    const injuredRaw = (document.getElementById('ps-injured') as HTMLTextAreaElement).value
    const injured_players = injuredRaw
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0)

    return { data_source, injured_players }
}
