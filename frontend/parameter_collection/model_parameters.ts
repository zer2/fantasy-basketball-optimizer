// Collects: all model parameters (upsilon, psi, chi, aleph, omega, gamma, beth, n_iterations)
// Mirrors player_stat_param_popover() + algorithm_param_popover()
//   in src/parameter_collection/parameters.py
//
// Defaults are loaded from the backend config (parameters.yaml) via
// getSportConfig().

import { ModelParameters } from '../types.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

interface ParamSpec {
    id:      string
    key:     string        // key in config.options (e.g. 'omega', 'S')
    label:   string
    caption: string
    step:    number
}

const PARAM_SPECS: ParamSpec[] = [
    {
        id: 'mp-s', key: 'S', label: 'S<sub>\u03C3</sub> (SAVOR)', step: 1,
        caption: 'SAVOR noise parameter. Only used in Auction Mode. Roughly represents the standard deviation of dollar values expected for players during the season. Higher values down-weight low-dollar players more aggressively.',
    },
    {
        id: 'mp-upsilon', key: 'upsilon', label: 'υ (upsilon)', step: 0.05,
        caption: 'Scales injury rates down. At 1.0 full projected injury rates apply; at 0.0 all players are treated as healthy.',
    },
    {
        id: 'mp-psi', key: 'psi', label: 'ψ (psi)', step: 0.05,
        caption: 'Fraction of missed games assumed to be replaced by a replacement-level player.',
    },
    {
        id: 'mp-chi', key: 'chi', label: 'χ (chi)', step: 0.05,
        caption: 'Estimated season-long variance relative to empirical week-to-week variance (for Rotisserie).',
    },
    {
        id: 'mp-aleph', key: 'aleph', label: 'ℵ (aleph)', step: 0.05,
        caption: 'Extra correlation added between volume-based categories (for Rotisserie).',
    },
    {
        id: 'mp-kappa', key: 'kappa', label: 'κ (kappa)', step: 0.1,
        caption: 'Anti-crowded-punt strength. Early picks are gently steered away from punts the field is crowding into. 0 disables it.',
    },
    {
        id: 'mp-opponent-confidence', key: 'opponent_model_confidence', label: 'C (confidence)', step: 0.1,
        caption: 'How confident the algorithm is that other drafters are pursuing the punt strategies '
               + 'it predicts for them. 0 treats them as neutral pickers with no strategy at all; 1 '
               + 'takes the prediction at face value. Rotisserie always uses 1.',
    },
    {
        id: 'mp-reg-lambda', key: 'reg_lambda', label: 'λ (lambda)', step: 0.01,
        caption: 'How strongly early-round category weights are pulled back toward balanced. Higher '
               + 'values punt less committally, keeping more room to pivot later; 0 turns the pull '
               + 'off entirely.',
    },
    {
        id: 'mp-omega', key: 'omega', label: 'ω (omega)', step: 0.05,
        caption: 'Controls punting aggressiveness. Higher values cause the algorithm to punt more aggressively.',
    },
    {
        id: 'mp-gamma', key: 'gamma', label: 'γ (gamma)', step: 0.05,
        caption: 'Complements omega. Higher values require more general value to be sacrificed to pursue a punting strategy.',
    },
    {
        id: 'mp-beth', key: 'beth', label: 'ב (beth)', step: 0.5,
        caption: "Bayesian shrinkage applied to your team's projected stats. Higher values pull projections closer to the average.",
    },
    {
        id: 'mp-n-iterations', key: 'n_iterations', label: 'Iterations', step: 1,
        caption: 'Number of gradient descent iterations. More iterations improve convergence but increase compute time.',
    },
]

/** Resolves the effective default, min, and max for a parameter from the config. */
function resolveSpec(spec: ParamSpec): { default: number; min: number; max: number | null } {
    const config = getSportConfig()
    if (!config) throw new Error('resolveSpec called before sport config loaded')
    const opt = config.options[spec.key]
    if (!opt) throw new Error(`No config option found for parameter key '${spec.key}'`)
    return {
        default: opt.default,
        min:     opt.min,
        max:     opt.max,
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
    // The grid is built before the caller can call refreshOpponentConfidenceControl, so the item is
    // in the DOM by the time the format section asks for its visibility to be reconsidered.

    for (const spec of PARAM_SPECS) {
        grid.append(makeParamItem(spec))
    }
}

/** Shows the opponent-confidence control only where it can apply. Rotisserie pins it to full
 *  confidence — there is no equivalent uncertainty about punting in a format that barely punts — so
 *  offering the number there would imply a choice the algorithm ignores. */
export function refreshOpponentConfidenceControl(scoringFormat: string): void {
    const item = document.getElementById('mp-opponent-confidence')?.closest('.param-item')
    if (item instanceof HTMLElement) {
        item.style.display = scoringFormat === 'Rotisserie' ? 'none' : ''
    }
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
    label.innerHTML = spec.label
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
        kappa:           readNumberInput('mp-kappa'),
        reg_lambda:      readNumberInput('mp-reg-lambda'),
        opponent_model_confidence: readNumberInput('mp-opponent-confidence'),
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
