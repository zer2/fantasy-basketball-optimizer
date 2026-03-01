// Collects: all model parameters (upsilon, psi, chi, aleph, omega, gamma, beth, n_iterations)
// Mirrors player_stat_param_popover() + algorithm_param_popover()
//   in src/parameter_collection/parameters.py
//
// Renamed from parameters.py to model_parameters.ts to better reflect
// that this covers all mathematical model parameters, not just player stats.

import { ModelParameters } from '../types.js'

interface ParamSpec {
    id:      string
    label:   string
    caption: string
    default: number
    min:     number
    max:     number | null
    step:    number
}

// S_σ is shown only in Auction Mode; kept separate so it can be toggled independently.
const S_SPEC: ParamSpec = {
    id: 'mp-s', label: 'S\u03C3 (SAVOR)', step: 1,
    default: 10, min: 0, max: 200,
    caption: 'SAVOR noise parameter. Roughly represents the standard deviation of dollar values expected for players during the season. Higher values down-weight low-dollar players more aggressively.',
}

const PARAM_SPECS: ParamSpec[] = [
    {
        id: 'mp-upsilon', label: 'υ (upsilon)', step: 0.05,
        default: 1.0, min: 0.0, max: 1.0,
        caption: 'Scales injury rates down. At 1.0 full projected injury rates apply; at 0.0 all players are treated as healthy.',
    },
    {
        id: 'mp-psi', label: 'ψ (psi)', step: 0.05,
        default: 0.8, min: 0.0, max: 1.0,
        caption: 'Fraction of missed games assumed to be replaced by a replacement-level player.',
    },
    {
        id: 'mp-chi', label: 'χ (chi)', step: 0.05,
        default: 0.6, min: 0.0, max: 1.0,
        caption: 'Estimated season-long variance relative to empirical week-to-week variance (for Rotisserie).',
    },
    {
        id: 'mp-aleph', label: 'ℵ (aleph)', step: 0.05,
        default: 0.2, min: 0.0, max: 1.0,
        caption: 'Extra correlation added between volume-based categories (for Rotisserie).',
    },
    {
        id: 'mp-omega', label: 'ω (omega)', step: 0.05,
        default: 1.2, min: 0.0, max: 2.0,
        caption: 'Controls punting aggressiveness. Higher values cause the algorithm to punt more aggressively.',
    },
    {
        id: 'mp-gamma', label: 'γ (gamma)', step: 0.05,
        default: 0.1, min: 0.0, max: 1.0,
        caption: 'Complements omega. Higher values require more general value to be sacrificed to pursue a punting strategy.',
    },
    {
        id: 'mp-beth', label: 'ב (beth)', step: 0.5,
        default: 3, min: 0.0, max: null,
        caption: "Bayesian shrinkage applied to your team's projected stats. Higher values pull projections closer to the average.",
    },
    {
        id: 'mp-n-iterations', label: 'Iterations', step: 1,
        default: 30, min: 0, max: 10000,
        caption: 'Number of gradient descent iterations. More iterations improve convergence but increase compute time.',
    },
]

/**
 * Renders all model parameters as a compact 2-column grid of inputs with
 * collapsible ⓘ captions. Covers all parameters from both player_stat_param_popover()
 * and algorithm_param_popover() in parameters.py.
 */
export function renderModelParameters(container: HTMLElement): void {
    const grid = document.createElement('div')
    grid.className = 'param-grid'
    container.append(grid)

    for (const spec of PARAM_SPECS) {
        grid.append(makeParamItem(spec))
    }

    // S_σ is only relevant in Auction Mode; hidden otherwise.
    const sItem = makeParamItem(S_SPEC)
    sItem.style.display = 'none'
    grid.append(sItem)

    document.getElementById('ls-mode')!.parentElement!.addEventListener('change', () => {
        sItem.style.display = (document.getElementById('ls-mode') as HTMLInputElement).value === 'Auction Mode' ? '' : 'none'
    })
}

/** Builds one parameter item: label row with ⓘ info button, number input, collapsible caption. */
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
    infoBtn.dataset.tooltip = spec.caption
    labelRow.append(infoBtn)

    const input = document.createElement('input')
    input.type = 'number'
    input.id = spec.id
    input.className = 'sidebar-input'
    input.min = String(spec.min)
    if (spec.max !== null) input.max = String(spec.max)
    input.step = String(spec.step)
    input.value = String(spec.default)

    item.append(labelRow, input)
    return item
}

export function getModelParameters(): ModelParameters {
    return {
        upsilon:         readNumberInput('mp-upsilon'),
        psi:             readNumberInput('mp-psi'),
        chi:             readNumberInput('mp-chi'),
        aleph:           readNumberInput('mp-aleph'),
        omega:           readNumberInput('mp-omega'),
        gamma:           readNumberInput('mp-gamma'),
        beth:            readNumberInput('mp-beth'),
        n_iterations:    readNumberInput('mp-n-iterations'),
        streaming_noise: readNumberInput('mp-s'),
    }
}

function readNumberInput(id: string): number {
    return parseFloat((document.getElementById(id) as HTMLInputElement).value)
}
