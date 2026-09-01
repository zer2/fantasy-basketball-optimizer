// data_entry/drafter_methods.ts
// Per-drafter draft METHOD, used by Draft-mode autopilot: 'Manual input' (the default) or
// 'H-scoring' (the drafter is an H-scoring autodrafter). Held here (pref-backed) rather than in
// the DOM, so the reader (draft_board) and the writer (the header autodraft toggle) don't depend
// on element existence/ordering.

import { pref, savePref } from '../preferences.js'

const DRAFTER_METHOD_OPTIONS = ['Manual input', 'H-scoring'] as const
export type DrafterMethod = typeof DRAFTER_METHOD_OPTIONS[number]

/** The method for a drafter index; defaults to 'Manual input'. Any stored value that is not
 *  'Manual input' (including the removed legacy 'G-scoring') is treated as 'H-scoring'. */
export function getDrafterMethod(index: number): DrafterMethod {
    return pref(`drafter_mode_${index}`, 'Manual input') === 'Manual input' ? 'Manual input' : 'H-scoring'
}

/** Persists a drafter's method. */
export function setDrafterMethod(index: number, method: DrafterMethod): void {
    savePref(`drafter_mode_${index}`, method)
}
