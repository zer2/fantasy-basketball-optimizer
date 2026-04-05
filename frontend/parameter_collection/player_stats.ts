// Collects: data_source (type, blend_weights, custom_data_ids), injured_players
// Mirrors player_stats_popover() in src/parameter_collection/player_stats.py

import { makeCustomSelect } from '../custom_select.js'
import { uploadCsv, getSeasons } from '../api/client.js'
import { DataSource } from '../types.js'
import { pref, savePref } from '../preferences.js'

// ─── Module state ──────────────────────────────────────────────────────────────

// Stored data_ids from POST /data/upload; updated immediately on file selection.
let customDataIds: { HTB: string | null; BBM: string | null } = { HTB: null, BBM: null }

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
            loadingEl.remove()
            const label = document.createElement('label')
            label.className = 'sidebar-label'
            label.htmlFor = 'ps-season'
            label.textContent = 'Season'
            histSection.append(label)
            const seasonSelect = makeCustomSelect(
                'ps-season',
                seasons.map(s => ({ value: s, label: s })),
                seasons[0] ?? '',
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
        if (type === 'historical') loadSeasons()
    })

    // Load seasons immediately if restored type is 'historical'
    if (typeSelect.getValue() === 'historical') loadSeasons()

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

/** Renders blend weight sliders and CSV upload inputs for HTB and BBM sources. */
function renderBlendWeights(container: HTMLElement): void {

    const weightLabel = document.createElement('div')
    weightLabel.className = 'sidebar-label'
    weightLabel.textContent = 'Projection blend weights'
    container.append(weightLabel)

    const sources: { id: string; label: string; prefKey: string; defaultValue: number }[] = [
        { id: 'ps-w-espn',  label: 'ESPN',  prefKey: 'blend_w_espn',  defaultValue: 0.5 },
        { id: 'ps-w-darko', label: 'DARKO', prefKey: 'blend_w_darko', defaultValue: 0.5 },
        { id: 'ps-w-htb',   label: 'HTB',   prefKey: 'blend_w_htb',   defaultValue: 0.0 },
        { id: 'ps-w-bbm',   label: 'BBM',   prefKey: 'blend_w_bbm',   defaultValue: 0.0 },
    ]

    for (const source of sources) {
        const row = document.createElement('div')
        row.className = 'sidebar-slider-row'

        const label = document.createElement('label')
        label.htmlFor = source.id
        label.textContent = source.label
        row.append(label)

        const savedWeight = pref(source.prefKey, source.defaultValue)

        const slider = document.createElement('input')
        slider.type = 'range'
        slider.id = source.id
        slider.min = '0'
        slider.max = '1'
        slider.step = '0.05'
        slider.value = String(savedWeight)
        row.append(slider)

        const valueDisplay = document.createElement('span')
        valueDisplay.className = 'slider-value'
        valueDisplay.textContent = savedWeight.toFixed(2)
        row.append(valueDisplay)

        slider.addEventListener('input', () => {
            valueDisplay.textContent = parseFloat(slider.value).toFixed(2)
            savePref(source.prefKey, parseFloat(slider.value))
        })

        container.append(row)

        // File upload for HTB and BBM (user-supplied CSV projections)
        if (source.id === 'ps-w-htb' || source.id === 'ps-w-bbm') {
            const fileType = source.id === 'ps-w-htb' ? 'HTB' : 'BBM'
            const uploadId = `ps-upload-${fileType.toLowerCase()}`

            const uploadRow = document.createElement('div')
            uploadRow.className = 'sidebar-upload-row'

            const uploadInput = document.createElement('input')
            uploadInput.type = 'file'
            uploadInput.id = uploadId
            uploadInput.accept = '.csv'
            uploadInput.className = 'sidebar-file-input'
            uploadRow.append(uploadInput)

            const statusSpan = document.createElement('span')
            statusSpan.className = 'sidebar-caption'
            uploadRow.append(statusSpan)

            uploadInput.addEventListener('change', async () => {
                const file = uploadInput.files?.[0]
                if (!file) return
                statusSpan.textContent = 'Uploading…'
                try {
                    const resp = await uploadCsv(file, fileType)
                    customDataIds[fileType] = resp.data_id
                    statusSpan.textContent = `✓ ${resp.n_players} players loaded`
                } catch (err) {
                    customDataIds[fileType] = null
                    statusSpan.textContent = `Upload failed: ${err}`
                    console.error(`${fileType} upload failed:`, err)
                }
            })

            container.append(uploadRow)
        }
    }
}

// ─── Getter ───────────────────────────────────────────────────────────────────

/**
 * Reads data source type, blend weights, and excluded player list from the DOM.
 * custom_data_ids are populated by the CSV upload handlers above.
 */
export function getPlayerStatsParams(): { data_source: DataSource; injured_players: string[] } {
    const type = (document.getElementById('ps-data-type') as HTMLInputElement).value as DataSource['type']

    const blend_weights = {
        ESPN:  parseFloat((document.getElementById('ps-w-espn')  as HTMLInputElement).value),
        DARKO: parseFloat((document.getElementById('ps-w-darko') as HTMLInputElement).value),
        HTB:   parseFloat((document.getElementById('ps-w-htb')   as HTMLInputElement).value),
        BBM:   parseFloat((document.getElementById('ps-w-bbm')   as HTMLInputElement).value),
    }

    const seasonEl = document.getElementById('ps-season') as HTMLInputElement | null
    const data_source: DataSource = {
        type,
        blend_weights,
        custom_data_ids: { HTB: customDataIds.HTB, BBM: customDataIds.BBM },
        season: type === 'historical' ? (seasonEl?.value ?? null) : null,
    }

    const injuredRaw = (document.getElementById('ps-injured') as HTMLTextAreaElement).value
    const injured_players = injuredRaw
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0)

    return { data_source, injured_players }
}
