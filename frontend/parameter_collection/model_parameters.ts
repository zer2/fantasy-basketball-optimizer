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

const PARAM_SPECS: ParamSpec[] = [
    {
        id: 'mp-upsilon', label: 'υ (upsilon)', step: 0.05,
        default: 1.0, min: 0.0, max: 1.0,
        caption: 'Scales injury rates down. At 1.0 full projected injury rates apply; at 0.0 all players are treated as healthy.',
    },
    {
        id: 'mp-psi', label: 'ψ (psi)', step: 0.05,
        default: 0.5, min: 0.0, max: 1.0,
        caption: 'Fraction of missed games assumed to be replaced by a replacement-level player.',
    },
    {
        id: 'mp-chi', label: 'χ (chi)', step: 0.05,
        default: 0.6, min: 0.0, max: 1.0,
        caption: 'Estimated season-long variance relative to empirical week-to-week variance (for Rotisserie).',
    },
    {
        id: 'mp-aleph', label: 'ℵ (aleph)', step: 0.05,
        default: 0.0, min: 0.0, max: 1.0,
        caption: 'Extra correlation added between volume-based categories (for Rotisserie).',
    },
    {
        id: 'mp-omega', label: 'ω (omega)', step: 0.05,
        default: 0.85, min: 0.0, max: 2.0,
        caption: 'Controls punting aggressiveness. Higher values cause the algorithm to punt more aggressively.',
    },
    {
        id: 'mp-gamma', label: 'γ (gamma)', step: 0.05,
        default: 1.0, min: 0.0, max: 5.0,
        caption: 'Complements omega. Higher values require more general value to be sacrificed to pursue a punting strategy.',
    },
    {
        id: 'mp-beth', label: 'ב (beth)', step: 0.1,
        default: 0.0, min: 0.0, max: null,
        caption: "Bayesian shrinkage applied to your team's projected stats. Higher values pull projections closer to the average.",
    },
    {
        id: 'mp-n-iterations', label: 'Iterations', step: 1,
        default: 15, min: 1, max: 50,
        caption: 'Number of gradient descent iterations. More iterations improve convergence but increase compute time.',
    },
]

export function renderModelParameters(container: HTMLElement): void {
    for (const spec of PARAM_SPECS) {
        const label = document.createElement('label')
        label.className = 'sidebar-label'
        label.htmlFor = spec.id
        label.textContent = spec.label
        container.append(label)

        const input = document.createElement('input')
        input.type = 'number'
        input.id = spec.id
        input.className = 'sidebar-input'
        input.min = String(spec.min)
        if (spec.max !== null) input.max = String(spec.max)
        input.step = String(spec.step)
        input.value = String(spec.default)
        container.append(input)

        const caption = document.createElement('div')
        caption.className = 'sidebar-caption'
        caption.textContent = spec.caption
        container.append(caption)
    }
}

export function getModelParameters(): ModelParameters {
    function val(id: string): number {
        return parseFloat((document.getElementById(id) as HTMLInputElement).value)
    }
    return {
        upsilon:      val('mp-upsilon'),
        psi:          val('mp-psi'),
        chi:          val('mp-chi'),
        aleph:        val('mp-aleph'),
        omega:        val('mp-omega'),
        gamma:        val('mp-gamma'),
        beth:         val('mp-beth'),
        n_iterations: val('mp-n-iterations'),
    }
}
