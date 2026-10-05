// Collects: data_source (type, blend_weights, custom_data_ids), injured_players
// Mirrors player_stats_popover() in src/setting_collection/player_stats.py

import { makeCustomSelect } from '../custom_select.js'
import { makeWeightSlider, makeMultiSelectWidget, MultiSelectWidget } from '../helper_functions.js'
import { getAllPlayerIdentities, REPLACEMENT_PLAYER_ID } from '../player_registry.js'
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

// The injured / excluded players: chosen from the current pool, whose registry each session build delivers.
let injuredPlayersWidget: MultiSelectWidget | null = null

/** Every player in the pool, as injured-list options. The registry includes players already marked injured -- they
 *  leave the pool a step after it is built -- so a pick can always be seen and undone. A player the new pool lacks is
 *  unpicked silently: nothing excludes him, and the backend skips an id it does not have anyway. */
document.addEventListener('player-registry-updated', () => {
    if (injuredPlayersWidget === null) return
    const options = getAllPlayerIdentities()
        .filter(identity => identity.player_id !== REPLACEMENT_PLAYER_ID)
        .map(identity => ({
            value: String(identity.player_id),
            label: identity.positions.length > 0 ? `${identity.name} (${identity.positions.join(',')})` : identity.name,
        }))
        .sort((left, right) => left.label.localeCompare(right.label))
    injuredPlayersWidget.setOptionsSilently(options)
})

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

// The data-source select and the sections it shows, held for limitDataSourcesToPlatform.
let dataTypeSelect: ReturnType<typeof makeCustomSelect> | null = null
let showSectionsForDataType: ((type: string) => void) | null = null

const PROJECTIONS_OPTION = { value: 'projections', label: 'Projections' }
const HISTORICAL_OPTION  = { value: 'historical',  label: 'Historical'  }

/**
 * Offers Historical only with your own data, as the Streamlit app did: a live platform means a draft
 * or season being played now, which only projections describe -- a past season's stats would rank
 * players for a year that is over. Switching to a live platform therefore moves a Historical source
 * to Projections; switching back to your own data offers Historical again (without choosing it).
 *
 * `announceChange` sends the switch through the select's change event, which saves it and rebuilds
 * the session like a choice made by hand. At start-up it is left quiet instead: no session exists
 * yet, and the first one is built from the corrected source.
 */
export function limitDataSourcesToPlatform(
    platform: string
  , { announceChange }: { announceChange: boolean }
): void {
    if (dataTypeSelect === null || showSectionsForDataType === null) {
        throw new Error('limitDataSourcesToPlatform called before renderPlayerStats')
    }
    const isOwnData = platform === 'Enter your own data'
    const wasHistorical = dataTypeSelect.getValue() === 'historical'
    dataTypeSelect.setOptions(isOwnData ? [PROJECTIONS_OPTION, HISTORICAL_OPTION] : [PROJECTIONS_OPTION])
    if (isOwnData || !wasHistorical) return
    if (announceChange) {
        dataTypeSelect.setValue(PROJECTIONS_OPTION.value)
    } else {
        showSectionsForDataType(PROJECTIONS_OPTION.value)
    }
}

/** Returns a promise that resolves once the seasons dropdown is ready (immediately when
 *  no fetch is needed or one has already completed). Anything that reads the data source
 *  must await this first: until the fetch lands there is no `ps-season` element, and
 *  getPlayerStatsSettings() refuses to report a historical source without a season. */
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
        [PROJECTIONS_OPTION, HISTORICAL_OPTION],
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

    dataTypeSelect = typeSelect
    showSectionsForDataType = (type: string) => {
        projSection.style.display = type === 'projections' ? '' : 'none'
        histSection.style.display = type === 'historical'  ? '' : 'none'
    }
    typeSelect.element.addEventListener('change', () => {
        const type = typeSelect.getValue()
        showSectionsForDataType!(type)
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

    // Picked from the players in the pool rather than typed: a typed name that matched nobody (a missing accent, a
    // missing position) excluded no one and said nothing. Empty until the first session delivers its registry.
    injuredPlayersWidget = makeMultiSelectWidget('', [])
    injuredPlayersWidget.element.id = 'ps-injured'
    // The section rebuilds the session on any input inside it, and searching this list is not a change of anything.
    injuredPlayersWidget.element.addEventListener('input', event => event.stopPropagation())
    // A pick or a removal is: announced as a change, from the widget, like any other control in the section.
    const widgetElement = injuredPlayersWidget.element
    injuredPlayersWidget.onChange(() => widgetElement.dispatchEvent(new Event('change', { bubbles: true })))
    container.append(widgetElement)

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
        // DARKO starts at zero: its app has been down and its projections are stale (see docs/projections.md).
        { id: 'ps-w-espn',  label: 'ESPN',  prefKey: 'blend_w_espn',  defaultValue: 1.0 },
        { id: 'ps-w-darko', label: 'DARKO', prefKey: 'blend_w_darko', defaultValue: 0.0 },
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

    uploadInput.addEventListener('change', async (event) => {
        // The Player Stats section rebuilds the session on any change inside it, reading each slot's
        // dataId. Let this event through and that rebuild runs NOW, while the upload is still in
        // flight, with the slot's PREVIOUS data — and nothing rebuilds again when the new id lands.
        // So this event stops here, and the handler announces the change itself once the slot holds
        // the id the rebuild should use (see announceUploadChanged).
        event.stopPropagation()
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
        }
        // Browsers don't fire change when the same file is chosen again while the input still
        // holds it — and choosing it again is exactly what re-uploading an edited file, or retrying
        // a failed one, looks like. So the input is cleared after EVERY attempt. The input itself is
        // visually hidden and the filename is drawn by fileNameLabel, so clearing it shows nothing.
        uploadInput.value = ''
        announceUploadChanged(uploadRow)
    })
}

/** Tells the Player Stats section that a slot's upload changed, once the slot holds the data id
 *  a rebuild should use. Dispatched from the slot's row rather than the file input (whose own
 *  change event is stopped) and never from an id starting "ps-w-", so the section treats it as
 *  a change of which players exist, not a re-weighting of the same pool. */
function announceUploadChanged(uploadRow: HTMLElement): void {
    uploadRow.dispatchEvent(new Event('change', { bubbles: true }))
}

// ─── Getter ───────────────────────────────────────────────────────────────────

/**
 * Reads data source type, blend weights, and excluded player list from the DOM.
 * custom_data_ids are populated by the CSV upload handlers above.
 */
export function getPlayerStatsSettings(): { data_source: DataSource; injured_players: number[] } {
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

    if (injuredPlayersWidget === null) throw new Error('getPlayerStatsSettings called before renderPlayerStats')
    const injured_players = injuredPlayersWidget.getSelected().map(Number)

    return { data_source, injured_players }
}
