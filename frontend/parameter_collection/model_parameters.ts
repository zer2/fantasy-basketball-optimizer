// Collects: all model parameters (upsilon, psi, chi, aleph, omega, gamma, beth, n_iterations)
// Mirrors player_stat_param_popover() + algorithm_param_popover()
//   in src/parameter_collection/parameters.py
//
// Defaults are loaded from the backend config (parameters.yaml) via
// getSportConfig(). The config already applies the active punting preset
// (e.g. "Moderate punting") so omega/gamma/n_iterations reflect the
// preset defaults rather than the raw option defaults.

import { ModelParameters } from '../types.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

interface ParamSpec {
    id:      string
    key:     string        // key in config.options (e.g. 'omega', 'S')
    label:   string
    caption: string
    fallback: number       // used if config is unavailable
    step:    number
}

// S_σ is shown only in Auction Mode; kept separate so it can be toggled independently.
const S_SPEC: ParamSpec = {
    id: 'mp-s', key: 'S', label: 'S\u03C3 (SAVOR)', step: 1,
    fallback: 10,
    caption: 'SAVOR noise parameter. Roughly represents the standard deviation of dollar values expected for players during the season. Higher values down-weight low-dollar players more aggressively.',
}

const PARAM_SPECS: ParamSpec[] = [
    {
        id: 'mp-upsilon', key: 'upsilon', label: 'υ (upsilon)', step: 0.05,
        fallback: 1.0,
        caption: 'Scales injury rates down. At 1.0 full projected injury rates apply; at 0.0 all players are treated as healthy.',
    },
    {
        id: 'mp-psi', key: 'psi', label: 'ψ (psi)', step: 0.05,
        fallback: 0.8,
        caption: 'Fraction of missed games assumed to be replaced by a replacement-level player.',
    },
    {
        id: 'mp-chi', key: 'chi', label: 'χ (chi)', step: 0.05,
        fallback: 0.6,
        caption: 'Estimated season-long variance relative to empirical week-to-week variance (for Rotisserie).',
    },
    {
        id: 'mp-aleph', key: 'aleph', label: 'ℵ (aleph)', step: 0.05,
        fallback: 0.2,
        caption: 'Extra correlation added between volume-based categories (for Rotisserie).',
    },
    {
        id: 'mp-omega', key: 'omega', label: 'ω (omega)', step: 0.05,
        fallback: 0.7,
        caption: 'Controls punting aggressiveness. Higher values cause the algorithm to punt more aggressively.',
    },
    {
        id: 'mp-gamma', key: 'gamma', label: 'γ (gamma)', step: 0.05,
        fallback: 0.25,
        caption: 'Complements omega. Higher values require more general value to be sacrificed to pursue a punting strategy.',
    },
    {
        id: 'mp-beth', key: 'beth', label: 'ב (beth)', step: 0.5,
        fallback: 3,
        caption: "Bayesian shrinkage applied to your team's projected stats. Higher values pull projections closer to the average.",
    },
    {
        id: 'mp-n-iterations', key: 'n_iterations', label: 'Iterations', step: 1,
        fallback: 30,
        caption: 'Number of gradient descent iterations. More iterations improve convergence but increase compute time.',
    },
]

/** Resolves the effective default, min, and max for a parameter from the config. */
function resolveSpec(spec: ParamSpec): { default: number; min: number; max: number | null } {
    const config = getSportConfig()
    const opt = config?.options?.[spec.key]
    return {
        default: opt?.default ?? spec.fallback,
        min:     opt?.min     ?? 0,
        max:     opt?.max     ?? null,
    }
}

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
    const resolved = resolveSpec(spec)

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
    input.min = String(resolved.min)
    if (resolved.max !== null) input.max = String(resolved.max)
    input.step = String(spec.step)
    input.value = String(pref(spec.key, resolved.default))
    input.addEventListener('change', () => savePref(spec.key, parseFloat(input.value)))

    item.append(labelRow, input)
    return item
}

/** Reads all model parameter values from the DOM and returns them as a typed object. */
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

/** Reads a numeric input element's value by DOM id. */
function readNumberInput(id: string): number {
    return parseFloat((document.getElementById(id) as HTMLInputElement).value)
}
