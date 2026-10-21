// platforms/espn_league_id.ts
// The league id out of whatever the user pasted into the ESPN league-ID box.

/** A league id as typed or read from a URL, split into the league and its season (null when it names none). */
export function splitEspnLeagueId(leagueId: string): { league: string, season: number | null } {
    const [league, season] = leagueId.split('::')
    return { league, season: season !== undefined && /^\d{4}$/.test(season) ? Number(season) : null }
}

/** ESPN numbers a season by the year it ends: 2027 is 2026-27. */
export function formatEspnSeason(season: number): string {
    return `${season - 1}-${String(season).slice(2)}`
}

/** A bare id is returned trimmed (the connector pairs it with the season chosen for it). An ESPN URL is read for its
 *  `leagueId` query parameter, and for `seasonId` when present, as in
 *      https://fantasy.espn.com/basketball/league/settings?leagueId=1801165221&seasonId=2026
 *  which gives '1801165221::2026' (the backend's composite '<id>::<season>'; the connector chooses that season). A
 *  women's basketball URL (/womens-basketball/) prefixes the id 'wfba:'. Anything else is returned trimmed and
 *  unchanged, so a wrong value fails at ESPN with a message about that value rather than being silently reinterpreted
 *  here. */
export function extractEspnLeagueId(raw: string): string {
    const trimmed = raw.trim()
    let url: URL
    try {
        // A link copied from the address bar can come without its scheme (fantasy.espn.com/basketball/league?...).
        url = new URL(/^https?:\/\//i.test(trimmed) ? trimmed : `https://${trimmed}`)
    } catch {
        return trimmed
    }
    const leagueId = url.searchParams.get('leagueId')
    if (leagueId === null || !/^\d+$/.test(leagueId)) return trimmed
    const prefix = url.pathname.includes('/womens-basketball/') ? 'wfba:' : ''
    const seasonId = url.searchParams.get('seasonId')
    const season = seasonId !== null && /^\d{4}$/.test(seasonId) ? `::${seasonId}` : ''
    return `${prefix}${leagueId}${season}`
}
