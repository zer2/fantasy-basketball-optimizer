// data_entry/drafter_methods.ts
// Per-drafter draft METHOD (Manual input / H-scoring / G-scoring), used by Draft-mode
// autopilot. Held here (pref-backed) rather than in the DOM, so the reader (draft_board)
// and the writer (the header method dropdown) don't depend on element existence/ordering.

import { pref, savePref } from '../preferences.js'

export const DRAFTER_METHOD_OPTIONS = ['Manual input', 'H-scoring', 'G-scoring'] as const
export type DrafterMethod = typeof DRAFTER_METHOD_OPTIONS[number]

/** The method for a drafter index; defaults to 'Manual input'. */
export function getDrafterMethod(index: number): DrafterMethod {
    return pref(`drafter_mode_${index}`, 'Manual input') as DrafterMethod
}

/** Persists a drafter's method. */
export function setDrafterMethod(index: number, method: DrafterMethod): void {
    savePref(`drafter_mode_${index}`, method)
}
