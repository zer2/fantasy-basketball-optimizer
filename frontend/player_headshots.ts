// player_headshots.ts
// NBA player headshots for the displays.
//
// The NBA CDN serves a transparent-background headshot per NBA player id; the backend
// proxies and caches those at /players/headshots/{id} (the CDN's bot filtering rejects
// some clients when hit directly, and the proxy fetches each image from the NBA at most
// once). This module holds the piece of state that makes those URLs usable: a cache of
// player name (any source's spelling) -> NBA player id, fetched once at startup. The
// image bytes are cached by the backend and the browser's HTTP cache. Players whose
// image 404s (some historical players) hide their <img> via the error handler, so a
// missing headshot costs nothing visually.

import { BASE_URL } from './api/client.js'

const HEADSHOT_BASE_URL = `${BASE_URL}/players/headshots/`

// null until loadNbaPlayerIds resolves; displays rendered before then simply skip the
// headshot (the id map loads at startup, well before the first evaluate returns).
let nbaPlayerIdsByName: Record<string, number> | null = null

/** Fetches the canonical-name -> NBA-player-id map once. Call at startup (fire and forget);
 *  a failure leaves the map empty and every display degrades to no headshot. */
export async function loadNbaPlayerIds(): Promise<void> {
    if (nbaPlayerIdsByName !== null) return
    try {
        const response = await fetch(`${BASE_URL}/players/nba-ids`)
        if (!response.ok) return
        const payload = await response.json() as { nba_player_ids: Record<string, number> }
        nbaPlayerIdsByName = payload.nba_player_ids
    } catch {
        // Headshots are decoration: never let their absence break a display.
    }
}

/** Display names carry a position suffix ("Nikola Jokic (C)"); the id map is keyed by
 *  the bare canonical name. */
function stripPositionSuffix(displayName: string): string {
    return displayName.replace(/\s*\([A-Z,\-]+\)\s*$/, '')
}

/** The CDN headshot URL for a player display name, or null when the id is unknown
 *  (map not loaded yet, or the player has no NBA id). */
export function getHeadshotUrl(displayName: string): string | null {
    if (nbaPlayerIdsByName === null) return null
    const nbaPlayerId = nbaPlayerIdsByName[stripPositionSuffix(displayName)]
    return nbaPlayerId === undefined ? null : `${HEADSHOT_BASE_URL}${nbaPlayerId}.png`
}

/** Headshot <img> HTML for string-built rows (the candidate table builds row HTML as one
 *  string for speed). Empty string when the player has no known headshot. The inline
 *  error handler hides the element when the CDN has no image for the id. */
export function buildHeadshotImageHtml(displayName: string, className: string): string {
    const url = getHeadshotUrl(displayName)
    if (url === null) return ''
    return `<img class='${className}' src='${url}' loading='lazy' alt='' onerror="this.style.display='none'">`
}

/** Headshot <img> element for DOM-built displays (roster grid, team views).
 *  Null when the player has no known headshot. */
export function makeHeadshotImage(displayName: string, className: string): HTMLImageElement | null {
    const url = getHeadshotUrl(displayName)
    if (url === null) return null
    const image = document.createElement('img')
    image.className = className
    image.src = url
    image.loading = 'lazy'
    image.alt = ''
    image.addEventListener('error', () => { image.style.display = 'none' })
    return image
}

