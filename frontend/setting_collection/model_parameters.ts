// Collects: all model parameters (upsilon, psi, chi, aleph, pick_pool_size, beth, n_iterations)
// Mirrors player_stat_param_popover() + algorithm_param_popover()
//   in src/setting_collection/parameters.py
//
// Defaults are loaded from the backend config (parameters.yaml) via
// getSportConfig().

import { ModelSettings } from '../types.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'

interface ParamSpec {
    id:      string
    key:     string        // key in config.options (e.g. 'pick_pool_size', 'S')
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
        id: 'mp-opponent-confidence', key: 'opponent_model_confidence', label: 'C (confidence)', step: 0.1,
        caption: 'How confident the algorithm is that other drafters are pursuing the punt strategies '
               + 'it predicts for them. 0 treats them as neutral pickers with no strategy at all; 1 '
               + 'takes the prediction at face value. Rotisserie always uses 1.',
    },
    {
        id: 'mp-lambda-c', key: 'lambda_c', label: 'λ<sub>c</sub> (category reg)', step: 0.01,
        caption: 'How strongly early-round category weights are pulled back toward balanced. Higher '
               + 'values punt less committally, keeping more room to pivot later; 0 turns the pull '
               + 'off entirely.',
    },
    {
        id: 'mp-lambda-p', key: 'lambda_p', label: 'λ<sub>p</sub> (position reg)', step: 0.5,
        caption: 'How strongly the flex-position strategy for future picks is pulled back toward a '
               + 'balanced positional mix. Higher values hedge across positions; 0 lets the strategy '
               + 'commit entirely to the positions that fit the build.',
    },
    {
        id: 'mp-pick-pool-size', key: 'pick_pool_size', label: 'M (pool size)', step: 1,
        caption: 'How many available players each future pick effectively chooses among. Larger '
               + 'windows give future picks more freedom to pursue the build, making punting more '
               + 'aggressive; smaller windows keep builds balanced.',
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
export function renderModelSettings(container: HTMLElement): void {
    const grid = document.createElement('div')
    grid.className = 'param-grid'
    container.append(grid)
    // The grid is built before the caller can call refreshOpponentConfidenceControl, so the item is
    // in the DOM by the time the format section asks for its visibility to be reconsidered.

    for (const spec of PARAM_SPECS) {
        grid.append(makeParamItem(spec))
    }
}

/** Shows each format-dependent parameter only where it can apply — the sidebar's standard is to
 *  hide parameters the current format ignores.
 *  - Opponent confidence: hidden under Rotisserie, which pins it to full confidence.
 *  - chi: week-to-season variance scaling, consumed only by Rotisserie's objective.
 *  - aleph: adjusts the category-correlation matrix, which is built under Rotisserie and
 *    whenever the Most-Categories weight is above zero. */
export function refreshFormatParameterControls(
    scoringFormat: string
    , mostCategoriesWeight: number | null
): void {
    const isRotisserie = scoringFormat === 'Rotisserie'
    const setVisible = (id: string, visible: boolean) => {
        const item = document.getElementById(id)?.closest('.param-item')
        if (item instanceof HTMLElement) item.style.display = visible ? '' : 'none'
    }
    setVisible('mp-opponent-confidence', !isRotisserie)
    setVisible('mp-chi', isRotisserie)
    setVisible('mp-aleph', isRotisserie || (mostCategoriesWeight ?? 0) > 0)
}

/** Shows the SAVOR streaming-noise control only in Auction Mode — the parameter is inert in
 *  Draft and Season Modes, and the sidebar's standard is to hide parameters that cannot apply. */
export function refreshStreamingNoiseControl(mode: string): void {
    const item = document.getElementById('mp-s')?.closest('.param-item')
    if (item instanceof HTMLElement) {
        item.style.display = mode === 'Auction Mode' ? '' : 'none'
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
export function getModelSettings(): ModelSettings {
    return {
        upsilon:         readNumberInput('mp-upsilon'),
        psi:             readNumberInput('mp-psi'),
        chi:             readNumberInput('mp-chi'),
        aleph:           readNumberInput('mp-aleph'),
        lambda_c:        readNumberInput('mp-lambda-c'),
        lambda_p:        readNumberInput('mp-lambda-p'),
        opponent_model_confidence: readNumberInput('mp-opponent-confidence'),
        pick_pool_size:  readNumberInput('mp-pick-pool-size'),
        beth:            readNumberInput('mp-beth'),
        n_iterations:    readNumberInput('mp-n-iterations'),
        streaming_noise: readNumberInput('mp-s'),
    }
}

/** Reads a numeric input element's value by DOM id. */
function readNumberInput(id: string): number {
    return parseFloat((document.getElementById(id) as HTMLInputElement).value)
}
