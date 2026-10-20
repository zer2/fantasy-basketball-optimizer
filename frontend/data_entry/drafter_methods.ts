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

// Stored under a key of its own, not the first build's `auction_drafter_mode_N`: that build defaulted every seat
// but the user's to autodraft, so choices saved under it would switch random seats back on.
const AUCTION_DRAFTER_METHOD_KEY = 'auction_drafter_method_'

/** The auction method for a drafter index; defaults to 'Manual input', as in draft mode, so the board stays a
 *  plain record of an auction until the user turns autodrafters on. */
export function getAuctionDrafterMethod(index: number): DrafterMethod {
    return pref(`${AUCTION_DRAFTER_METHOD_KEY}${index}`, 'Manual input') === 'Manual input' ? 'Manual input' : 'H-scoring'
}

/** Persists an auction drafter's method. */
export function setAuctionDrafterMethod(index: number, method: DrafterMethod): void {
    savePref(`${AUCTION_DRAFTER_METHOD_KEY}${index}`, method)
}

