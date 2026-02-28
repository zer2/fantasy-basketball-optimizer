// Collects: upsilon, psi, chi, aleph, beth
// These parameters affect earlier steps of the initialization chain (steps 2–5),
// so they are kept in a separate sidebar section from the H-score parameters.
// Mirrors player_stat_param_popover() in src/parameter_collection/parameters.py

interface ParamSpec {
    id:      string
    label:   string
    caption: string
    default: number
    min:     number
    max:     number | null
    step:    number
}

const SPECS: ParamSpec[] = [
    {
        id: 'psp-upsilon', label: 'υ (upsilon)',
        default: 1.0, min: 0.0, max: 1.0, step: 0.05,
        caption: 'Scales injury rates down. At 1.0 full projected injury rates apply; at 0.0 all players are treated as healthy.',
    },
    {
        id: 'psp-psi', label: 'ψ (psi)',
        default: 0.8, min: 0.0, max: 1.0, step: 0.05,
        caption: 'Fraction of missed games assumed to be replaced by a replacement-level player.',
    },
    {
        id: 'psp-chi', label: 'χ (chi)',
        default: 0.6, min: 0.0, max: 1.0, step: 0.05,
        caption: 'Estimated season-long variance relative to empirical week-to-week variance (for Rotisserie).',
    },
    {
        id: 'psp-aleph', label: 'ℵ (aleph)',
        default: 0.2, min: 0.0, max: 1.0, step: 0.05,
        caption: 'Extra correlation added between volume-based categories (for Rotisserie).',
    },
    {
        id: 'psp-beth', label: 'ב (beth)',
        default: 3, min: 0.0, max: null, step: 0.5,
        caption: "Bayesian shrinkage applied to your team's projected stats. Higher values pull projections closer to the average.",
    },
]

/**
 * Renders the Player Stat Parameters section: compact 2-column grid of υ, ψ, χ, ℵ, ב
 * inputs with collapsible ⓘ captions. These parameters affect data processing
 * steps 2–5 (earlier than the H-score parameters in algo_parameters.ts).
 */
export function renderPlayerStatParameters(container: HTMLElement): void {
    const grid = document.createElement('div')
    grid.className = 'param-grid'
    container.append(grid)

    for (const spec of SPECS) {
        grid.append(makeParamItem(spec))
    }
}

/** Reads all player stat parameter values from the DOM. */
export function getPlayerStatParameters(): { upsilon: number; psi: number; chi: number; aleph: number; beth: number } {
    return {
        upsilon: readNumberInput('psp-upsilon'),
        psi:     readNumberInput('psp-psi'),
        chi:     readNumberInput('psp-chi'),
        aleph:   readNumberInput('psp-aleph'),
        beth:    readNumberInput('psp-beth'),
    }
}

/** Builds one parameter row: label + ⓘ button, number input, collapsible caption. */
function makeParamItem(spec: ParamSpec): HTMLElement {
    const item = document.createElement('div')
    item.className = 'param-item'

    const labelRow = document.createElement('div')
    labelRow.className = 'param-label-row'

    const label = document.createElement('label')
    label.htmlFor = spec.id
    label.textContent = spec.label
    labelRow.append(label)

    const infoBtn = document.createElement('button')
    infoBtn.type = 'button'
    infoBtn.className = 'info-btn'
    infoBtn.textContent = 'ⓘ'
    labelRow.append(infoBtn)

    const caption = document.createElement('div')
    caption.className = 'param-caption'
    caption.textContent = spec.caption
    caption.hidden = true

    infoBtn.addEventListener('click', () => { caption.hidden = !caption.hidden })

    const input = document.createElement('input')
    input.type = 'number'
    input.id = spec.id
    input.className = 'sidebar-input'
    input.min = String(spec.min)
    if (spec.max !== null) input.max = String(spec.max)
    input.step = String(spec.step)
    input.value = String(spec.default)

    item.append(labelRow, input, caption)
    return item
}

function readNumberInput(id: string): number {
    return parseFloat((document.getElementById(id) as HTMLInputElement).value)
}

