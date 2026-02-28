// Collects: data_source (type, blend_weights, custom_data_ids), injured_players
// Mirrors player_stats_popover() in src/parameter_collection/player_stats.py
//
// CSV upload handling is a skeleton — actual upload logic requires POST /data/upload.

import { makeCustomSelect } from '../custom_select.js'
import { DataSource } from '../types.js'

/**
 * Renders the Player Stats section: data source selector, projection blend weight
 * sliders, optional CSV uploads (skeleton), and injured/excluded player list.
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
        [{ value: 'projections', label: 'Projections' }, { value: 'historical', label: 'Historical' }],
        'projections',
    )
    container.append(typeSelect.element)

    // Projection blend weights section (hidden when historical is selected)
    const projSection = document.createElement('div')
    projSection.id = 'ps-proj-section'
    container.append(projSection)

    renderBlendWeights(projSection)

    typeSelect.element.addEventListener('change', () => {
        projSection.style.display = typeSelect.getValue() === 'projections' ? '' : 'none'
    })

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

    const injuredCaption = document.createElement('div')
    injuredCaption.className = 'sidebar-caption'
    injuredCaption.textContent = 'These players are dropped before coefficient calculation.'
    container.append(injuredCaption)
}

/** Renders blend weight sliders and optional CSV upload inputs for each data source. */
function renderBlendWeights(container: HTMLElement): void {

    const weightLabel = document.createElement('div')
    weightLabel.className = 'sidebar-label'
    weightLabel.textContent = 'Projection blend weights'
    container.append(weightLabel)

    const sources: { id: string; label: string; defaultValue: number }[] = [
        { id: 'ps-w-espn', label: 'ESPN',  defaultValue: 0.5 },
        { id: 'ps-w-darko', label: 'DARKO', defaultValue: 0.5 },
        { id: 'ps-w-htb',  label: 'HTB',   defaultValue: 0.0 },
        { id: 'ps-w-bbm',  label: 'BBM',   defaultValue: 0.0 },
    ]

    for (const source of sources) {
        const row = document.createElement('div')
        row.className = 'sidebar-slider-row'

        const label = document.createElement('label')
        label.htmlFor = source.id
        label.textContent = source.label
        row.append(label)

        const slider = document.createElement('input')
        slider.type = 'range'
        slider.id = source.id
        slider.min = '0'
        slider.max = '1'
        slider.step = '0.05'
        slider.value = String(source.defaultValue)
        row.append(slider)

        const valueDisplay = document.createElement('span')
        valueDisplay.className = 'slider-value'
        valueDisplay.textContent = source.defaultValue.toFixed(2)
        row.append(valueDisplay)

        slider.addEventListener('input', () => {
            valueDisplay.textContent = parseFloat(slider.value).toFixed(2)
        })

        container.append(row)

        // File upload for sources that accept custom CSVs
        if (source.id === 'ps-w-htb' || source.id === 'ps-w-bbm') {
            const uploadId = source.id === 'ps-w-htb' ? 'ps-upload-htb' : 'ps-upload-bbm'
            const uploadInput = document.createElement('input')
            uploadInput.type = 'file'
            uploadInput.id = uploadId
            uploadInput.accept = '.csv'
            uploadInput.className = 'sidebar-file-input'
            // TODO: on change, POST file to /data/upload and store returned data_id
            container.append(uploadInput)
        }
    }
}

/**
 * Reads data source type, blend weights, and excluded player list from the DOM.
 * `custom_data_ids` are always `null` until the CSV upload flow is implemented.
 */
export function getPlayerStatsParams(): { data_source: DataSource; injured_players: string[] } {
    const type = (document.getElementById('ps-data-type') as HTMLInputElement).value as 'projections' | 'historical'

    const blend_weights = {
        espn:  parseFloat((document.getElementById('ps-w-espn')  as HTMLInputElement).value),
        darko: parseFloat((document.getElementById('ps-w-darko') as HTMLInputElement).value),
        htb:   parseFloat((document.getElementById('ps-w-htb')   as HTMLInputElement).value),
        bbm:   parseFloat((document.getElementById('ps-w-bbm')   as HTMLInputElement).value),
    }

    // TODO: replace nulls with data_ids returned by POST /data/upload once upload is wired up
    const custom_data_ids = {
        HTB: null as string | null,
        BBM: null as string | null,
    }

    const data_source: DataSource = { type, blend_weights, custom_data_ids }

    const injuredRaw = (document.getElementById('ps-injured') as HTMLTextAreaElement).value
    const injured_players = injuredRaw
        .split('\n')
        .map(s => s.trim())
        .filter(s => s.length > 0)

    return { data_source, injured_players }
}
