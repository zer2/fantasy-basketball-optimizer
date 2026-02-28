// Collects: omega, gamma, n_iterations
// These only affect step 5 (build_h_agent), so they are kept in a separate section
// from the player stat parameters which affect earlier steps.
// Mirrors algorithm_param_popover() in src/parameter_collection/parameters.py

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
        id: 'ap-omega', label: 'ω (omega)',
        default: 0.7, min: 0.0, max: 2.0, step: 0.05,
        caption: 'Controls punting aggressiveness. Higher values cause the algorithm to punt more aggressively.',
    },
    {
        id: 'ap-gamma', label: 'γ (gamma)',
        default: 0.25, min: 0.0, max: 1.0, step: 0.05,
        caption: 'Higher values require more general value to be sacrificed in order to pursue a punting strategy.',
    },
    {
        id: 'ap-n-iterations', label: 'Iterations',
        default: 30, min: 0, max: 10000, step: 1,
        caption: 'Number of gradient descent iterations. More iterations improve convergence but increase compute time.',
    },
]

/**
 * Renders the H-score Parameters section: compact 2-column grid of ω, γ, and
 * iterations inputs with collapsible ⓘ captions. These parameters only affect
 * step 5 (build_h_agent) and are separate from the player stat parameters.
 */
export function renderAlgoParameters(container: HTMLElement): void {
    const grid = document.createElement('div')
    grid.className = 'param-grid'
    container.append(grid)

    for (const spec of SPECS) {
        grid.append(makeParamItem(spec))
    }
}

/** Reads all H-score algorithm parameter values from the DOM. */
export function getAlgoParameters(): { omega: number; gamma: number; n_iterations: number } {
    return {
        omega:        readNumberInput('ap-omega'),
        gamma:        readNumberInput('ap-gamma'),
        n_iterations: readNumberInput('ap-n-iterations'),
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

