// Collects: data_source (type, blend_weights, custom_data_ids), injured_players
// Mirrors player_stats_popover() in src/parameter_collection/player_stats.py

import { makeCustomSelect } from '../custom_select.js'
import { uploadProjectionFile, getSeasons } from '../api/client.js'
import { DataSource } from '../types.js'
import { pref, savePref } from '../preferences.js'

// ─── Module state ──────────────────────────────────────────────────────────────

// One row per custom projection slot. The upload's data_id is its identity everywhere
// (blend-weight key, session requests); the file's own name is what identifies the slot
// to the reader.
interface CustomUploadRow {
    dataId: string | null
    fileName: string | null
    slider: HTMLInputElement
    valueDisplay: HTMLSpanElement
    statusSpan: HTMLSpanElement
    uploadInput: HTMLInputElement
    fileNameLabel: HTMLSpanElement
}

/** Shows a chosen file's name, or the browser's own wording when a slot is empty. The full
 *  name rides in the tooltip, since a long one is ellipsized in the narrow sidebar. */
function setFileNameLabel(label: HTMLSpanElement, fileName: string | null): void {
    label.textContent = fileName ?? 'No file chosen'
    label.title = fileName ?? ''
    label.classList.toggle('sidebar-file-name-empty', fileName === null)
}

// What is remembered across reloads for a filled slot. Only the id is load-bearing — the
// file itself lives server-side, kept for a day on a clock that resets whenever a session
// uses it, so a remembered id normally still resolves. When it does not (a longer gap, a
// wiped cache), the patch that carries it fails and markUploadedSourcesExpired clears the
// slot with a visible message, which is the same recovery a mid-session expiry gets.
interface StoredCustomUpload {
    dataId: string
    weight: number
    statusText: string
    fileName: string | null
}

const CUSTOM_UPLOADS_PREF = 'custom_uploads'
const MAX_CUSTOM_UPLOADS = 5
let customUploadRows: CustomUploadRow[] = []

/** Persists every filled slot (id, filename, weight, status line) so uploads survive a reload. */
function saveCustomUploads(): void {
    savePref(CUSTOM_UPLOADS_PREF, customUploadRows
        .filter(row => row.dataId !== null)
        .map(row => ({
            dataId:     row.dataId as string,
            weight:     parseFloat(row.slider.value),
            statusText: row.statusSpan.textContent ?? '',
            fileName:   row.fileName,
        })))
}

/** The remembered slots, ignoring anything malformed (hand-edited or older storage). */
function readStoredCustomUploads(): StoredCustomUpload[] {
    const stored = pref<unknown>(CUSTOM_UPLOADS_PREF, [])
    if (!Array.isArray(stored)) return []
    return stored.filter((entry): entry is StoredCustomUpload =>
        entry !== null && typeof entry === 'object'
        && typeof (entry as StoredCustomUpload).dataId === 'string'
        && typeof (entry as StoredCustomUpload).weight === 'number')
}

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
        // The slot holds nothing now; showing the old filename would suggest otherwise.
        row.fileName = null
        setFileNameLabel(row.fileNameLabel, null)
        // Clear the input so re-selecting the same file fires a fresh change event.
        row.uploadInput.value = ''
    }
    // Forget the dead ids too, so a reload does not restore them and fail all over again.
    if (clearedAny) saveCustomUploads()
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
    // Two rows: the box scrolls, and most leagues exclude nobody or a name or two, so the taller
    // default was mostly empty space in a sidebar that has none to spare.
    injuredInput.rows = 2
    container.append(injuredInput)
    
}

/** Renders the projection source weights: ESPN and DARKO sliders, then the custom
 *  projections section — up to five uploadable sources, each a file chooser plus a weight
 *  slider locked until its upload succeeds. A fresh empty row appears after each successful
 *  upload. */
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

    // No heading or explanation above the upload slots: a file chooser sitting under the source
    // weights, with a weight slider of its own, already says what it is.
    const customRowsContainer = document.createElement('div')
    customRowsContainer.id = 'ps-custom-uploads'
    container.append(customRowsContainer)

    // Restore the slots filled before the last reload, then leave one empty slot open.
    customUploadRows = []
    for (const storedUpload of readStoredCustomUploads()) {
        if (customUploadRows.length >= MAX_CUSTOM_UPLOADS) break
        appendCustomUploadRow(customRowsContainer, storedUpload)
    }
    if (customUploadRows.length < MAX_CUSTOM_UPLOADS) appendCustomUploadRow(customRowsContainer)
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

/** Appends one custom-projection slot: [file chooser | filename] over [weight slider], with
 *  the upload's status line beneath. The file's own name identifies the source, so the slider
 *  needs no label of its own. The slider stays locked at zero until this slot's upload
 *  succeeds. `storedUpload` restores a slot remembered from a previous visit, already filled
 *  and unlocked. */
function appendCustomUploadRow(
    customRowsContainer: HTMLElement
    , storedUpload?: StoredCustomUpload
): void {
    const rowNumber = customUploadRows.length + 1

    const uploadRow = document.createElement('div')
    uploadRow.className = 'sidebar-upload-row'

    const uploadInput = document.createElement('input')
    uploadInput.type = 'file'
    uploadInput.id = `ps-upload-custom-${rowNumber}`
    // Spreadsheets are as common as CSVs here: people copy a projection table into Excel and
    // upload what they saved. The backend decides format from the file's own signature, so
    // this only widens the picker.
    uploadInput.accept = '.csv,.xlsx'
    uploadInput.className = 'sidebar-file-input'

    // The file input's own text ("No file chosen", or the filename) is drawn by the browser
    // and cannot be set from script — a page that could would be able to fake a chosen file.
    // So a restored slot could never show the file behind it, however much else we remembered.
    // The input is visually hidden (still focusable, still the thing the label opens) and the
    // filename is rendered here instead, which makes it ours to persist like the rest.
    const fileButton = document.createElement('label')
    fileButton.className = 'sidebar-file-button'
    fileButton.htmlFor = uploadInput.id
    fileButton.textContent = 'Choose file'

    const fileNameLabel = document.createElement('span')
    fileNameLabel.className = 'sidebar-file-name'
    setFileNameLabel(fileNameLabel, storedUpload?.fileName ?? null)

    uploadRow.append(fileButton, uploadInput, fileNameLabel)

    const statusSpan = document.createElement('span')
    statusSpan.className = 'sidebar-caption'
    statusSpan.textContent = storedUpload?.statusText ?? ''
    uploadRow.append(statusSpan)
    customRowsContainer.append(uploadRow)

    const sliderRow = document.createElement('div')
    sliderRow.className = 'sidebar-slider-row custom-upload-weight-row'

    const { slider, valueDisplay } = makeWeightSlider(`ps-w-custom-${rowNumber}`, storedUpload?.weight ?? 0)
    // A weight for a source with no file behind it is meaningless — locked until upload succeeds.
    slider.disabled = storedUpload === undefined
    slider.addEventListener('input', saveCustomUploads)
    sliderRow.append(slider, valueDisplay)
    customRowsContainer.append(sliderRow)

    const row: CustomUploadRow = {
        dataId: storedUpload?.dataId ?? null,
        fileName: storedUpload?.fileName ?? null,
        slider, valueDisplay, statusSpan, uploadInput, fileNameLabel,
    }
    customUploadRows.push(row)

    uploadInput.addEventListener('change', async () => {
        const file = uploadInput.files?.[0]
        if (!file) return
        statusSpan.textContent = 'Uploading…'
        const hadUploadAlready = row.dataId !== null
        try {
            const resp = await uploadProjectionFile(file)
            row.dataId = resp.data_id
            // A source that pairs projections with a league only carries that league's
            // categories — say which standard stats this file lacks so a lighter file
            // reads as deliberate rather than as a parsing failure.
            const missingNote = resp.missing_stats.length > 0
                ? ` (no ${resp.missing_stats.join('/')})` : ''
            statusSpan.textContent = `✓ ${resp.n_players} players loaded${missingNote}`
            slider.disabled = false
            row.fileName = file.name
            setFileNameLabel(fileNameLabel, file.name)
            saveCustomUploads()
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
            saveCustomUploads()
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
