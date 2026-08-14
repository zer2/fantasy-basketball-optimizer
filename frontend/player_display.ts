// player_display.ts
// The two standard player displays, keyed by player id and reading the session registry:
//   full    = headshot + full name + positions (candidate rows, G-score row labels)
//   minimal = headshot + last name             (roster grids, draft/auction board cells)
// Each comes in a string-built variant (for contexts that assemble row HTML as one string)
// and a DOM variant (for contexts that append elements directly).
//
// Headshots: the NBA CDN serves a transparent-background image per NBA player id; the
// backend proxies and caches those at /players/headshots/{player_id}. The registry's
// has_headshot flag marks the ids with no CDN image (RP, synthetic ids), and players
// whose image still 404s hide their <img> via the error handler, so a missing headshot
// costs nothing visually.

import { BASE_URL } from './api/client.js'
import { getRegistryEntry } from './player_registry.js'

const HEADSHOT_BASE_URL = `${BASE_URL}/players/headshots/`

/** The proxied headshot URL for a player id, or null when the player has no headshot. */
export function getHeadshotUrl(playerId: number): string | null {
    if (!getRegistryEntry(playerId).has_headshot) return null
    return `${HEADSHOT_BASE_URL}${playerId}.png`
}

/** Headshot <img> HTML for string-built rows. Empty string when the player has no
 *  headshot. The inline error handler hides the element when the CDN has no image
 *  for the id.
 *
 *  loading='lazy' is LOAD-BEARING here: the virtualized table pre-builds EVERY candidate
 *  row as detached DOM and attaches only the visible window. A non-lazy <img> starts
 *  fetching the moment it is created (the new Image() preload behavior), so without lazy
 *  every evaluate fires one request per candidate (~570) instead of per visible row (~25)
 *  — which starved autodraft evaluates behind thousands of image fetches. */
function buildHeadshotImageHtml(playerId: number, className: string): string {
    const url = getHeadshotUrl(playerId)
    if (url === null) return ''
    return `<img class='${className}' src='${url}' loading='lazy' alt='' onerror="this.style.display='none'">`
}

/** Headshot <img> element for DOM-built displays. These attach to the live DOM
 *  immediately, so they stay eager (no lazy). Null when the player has no headshot. */
function makeHeadshotImage(playerId: number, className: string): HTMLImageElement | null {
    const url = getHeadshotUrl(playerId)
    if (url === null) return null
    const image = document.createElement('img')
    image.className = className
    image.src = url
    image.alt = ''
    image.addEventListener('error', () => { image.style.display = 'none' })
    return image
}

/** Positions suffix HTML for the full display; empty for a position-less player (e.g. RP). */
function buildPositionsHtml(positions: string[]): string {
    if (positions.length === 0) return ''
    return ` <span class='player-positions'>${positions.join(',')}</span>`
}

/** Full display (headshot + full name + positions) as an HTML string, for string-built
 *  contexts: virtualized candidate rows, trade-suggestion cells. */
export function buildFullPlayerDisplayHtml(playerId: number): string {
    const entry = getRegistryEntry(playerId)
    return buildHeadshotImageHtml(playerId, 'player-avatar')
        + `<span class='playername'>${entry.name}${buildPositionsHtml(entry.positions)}</span>`
}

/** Minimal display (headshot + last name) as an HTML string, for string-built contexts. */
export function buildMinimalPlayerDisplayHtml(playerId: number): string {
    const entry = getRegistryEntry(playerId)
    return `<span class='player-minimal'>`
        + buildHeadshotImageHtml(playerId, 'player-avatar')
        + `<span>${entry.last_name}</span></span>`
}

/** Full display (headshot + full name + positions) as a DOM element, for G-score table
 *  row labels and team views. */
export function makeFullPlayerDisplay(playerId: number): HTMLElement {
    const entry = getRegistryEntry(playerId)
    const display = document.createElement('div')
    display.className = 'player-full-display'
    const headshot = makeHeadshotImage(playerId, 'player-avatar')
    if (headshot) display.appendChild(headshot)
    const nameLabel = document.createElement('span')
    nameLabel.className = 'playername'
    nameLabel.textContent = entry.name
    if (entry.positions.length > 0) {
        const positionsLabel = document.createElement('span')
        positionsLabel.className = 'player-positions'
        positionsLabel.textContent = entry.positions.join(',')
        nameLabel.append(' ', positionsLabel)
    }
    display.appendChild(nameLabel)
    return display
}

/** Minimal display (headshot stacked above last name) as a DOM element, for the
 *  expanded-view roster grid and draft/auction board cells. */
export function makeMinimalPlayerDisplay(playerId: number): HTMLElement {
    const entry = getRegistryEntry(playerId)
    const stack = document.createElement('div')
    stack.className = 'roster-player-stack'
    const headshot = makeHeadshotImage(playerId, 'roster-headshot')
    if (headshot) stack.appendChild(headshot)
    const nameLabel = document.createElement('span')
    nameLabel.textContent = entry.last_name
    stack.appendChild(nameLabel)
    return stack
}
