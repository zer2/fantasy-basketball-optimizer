// Collects: trading evaluation parameters
// Mirrors trade_param_popover() in src/parameter_collection/parameters.py
// Note: these parameters are not part of POST /sessions; they will be used
// by a future trade evaluation endpoint.

import { makeLabel, makeNumberInput } from '../helper_functions.js'

export interface TradeComboRow {
    n_traded:   number
    n_received: number
    threshold:  number
}

export interface TradeParameters {
    your_differential_threshold:  number
    their_differential_threshold: number
    combo_params:                 TradeComboRow[]
}

const DEFAULT_COMBOS: TradeComboRow[] = [
    { n_traded: 1, n_received: 1, threshold: 3 },
    { n_traded: 2, n_received: 2, threshold: 3 },
    { n_traded: 3, n_received: 3, threshold: 3 },
]

/**
 * Renders the Trade Parameters section: improvement thresholds for your team and
 * the counterparty, plus an editable table of trade combination sizes to evaluate.
 * These parameters are not part of POST /sessions; they target a future trade endpoint.
 */
export function renderTradeParameters(container: HTMLElement): void {

    // Your threshold
    container.append(makeLabel('tp-your-threshold', 'Your improvement threshold (%)'))
    container.append(makeNumberInput('tp-your-threshold', 0))

    const yourCaption = document.createElement('div')
    yourCaption.className = 'sidebar-caption'
    yourCaption.textContent = 'Only trades that improve your H-score by at least this percent will be shown.'
    container.append(yourCaption)

    // Their threshold
    container.append(makeLabel('tp-their-threshold', 'Counterparty threshold (%)'))
    container.append(makeNumberInput('tp-their-threshold', -0.2))

    const theirCaption = document.createElement('div')
    theirCaption.className = 'sidebar-caption'
    theirCaption.textContent = 'Trades that hurt their H-score by more than this percent will not be shown.'
    container.append(theirCaption)

    // Combo params table
    const tableLabel = document.createElement('div')
    tableLabel.className = 'sidebar-label'
    tableLabel.style.marginTop = '10px'
    tableLabel.textContent = 'Trade combinations to evaluate'
    container.append(tableLabel)

    const table = document.createElement('table')
    table.id = 'tp-combo-table'
    table.className = 'trade-combo-table'

    const thead = document.createElement('thead')
    const headerRow = document.createElement('tr')
    for (const header of ['Traded', 'Received', 'Threshold', '']) {
        const th = document.createElement('th')
        th.textContent = header
        headerRow.append(th)
    }
    thead.append(headerRow)
    table.append(thead)

    const tbody = document.createElement('tbody')
    tbody.id = 'tp-combo-tbody'
    table.append(tbody)
    container.append(table)

    for (const row of DEFAULT_COMBOS) {
        addComboRow(tbody, row)
    }

    const addBtn = document.createElement('button')
    addBtn.type = 'button'
    addBtn.className = 'trade-add-btn'
    addBtn.textContent = '+ Add row'
    addBtn.addEventListener('click', () => addComboRow(tbody, { n_traded: 1, n_received: 1, threshold: 3 }))
    container.append(addBtn)
}

/** Appends one editable (traded, received, threshold) row to the combo table body. */
function addComboRow(tbody: HTMLElement, defaults: TradeComboRow): void {
    const row = document.createElement('tr')

    for (const key of ['n_traded', 'n_received', 'threshold'] as const) {
        const td = document.createElement('td')
        const input = document.createElement('input')
        input.type = 'number'
        input.className = 'sidebar-input trade-combo-input'
        input.value = String(defaults[key])
        input.min = '0'
        td.append(input)
        row.append(td)
    }

    const removeTd = document.createElement('td')
    const removeBtn = document.createElement('button')
    removeBtn.type = 'button'
    removeBtn.className = 'info-btn'
    removeBtn.textContent = '✕'
    removeBtn.addEventListener('click', () => row.remove())
    removeTd.append(removeBtn)
    row.append(removeTd)

    tbody.append(row)
}

/**
 * Reads all trade evaluation parameters from the DOM.
 * Threshold percentages are converted from display units (%) to decimal fractions.
 */
export function getTradeParameters(): TradeParameters {
    const your_differential_threshold =
        parseFloat((document.getElementById('tp-your-threshold') as HTMLInputElement).value) / 100
    const their_differential_threshold =
        parseFloat((document.getElementById('tp-their-threshold') as HTMLInputElement).value) / 100

    const rows = Array.from(document.querySelectorAll<HTMLTableRowElement>('#tp-combo-tbody tr'))
    const combo_params: TradeComboRow[] = rows.map(row => {
        const inputs = row.querySelectorAll<HTMLInputElement>('input')
        return {
            n_traded:   parseInt(inputs[0].value),
            n_received: parseInt(inputs[1].value),
            threshold:  parseFloat(inputs[2].value),
        }
    })

    return { your_differential_threshold, their_differential_threshold, combo_params }
}
