// platforms/yahoo_league_id.ts
// The league id out of whatever the user pasted into the Yahoo league-ID box.

/** A bare id is returned trimmed. A Yahoo URL is read for its league id, which Yahoo writes in
 *  two places: the path segment after the game code, as in
 *      https://basketball.fantasysports.yahoo.com/nba/12345
 *      https://basketball.fantasysports.yahoo.com/draftclient/nba/2678336/5?auth=
 *  or the `mlid` query parameter of a mock-draft lobby page, as in
 *      https://basketball.fantasysports.yahoo.com/nba/mock_waiting?mlid=2678371&lobby=standard
 *  The LAST numeric path segment is the wrong one: in the draft-client URL it is the seat (5), and
 *  on a team page it is the team number. Anything that is not a URL is returned trimmed and
 *  unchanged, so a wrong value fails at Yahoo with a message about that value rather than being
 *  silently reinterpreted here. */
export function extractYahooLeagueId(raw: string): string {
    const trimmed = raw.trim()
    let url: URL
    try {
        url = new URL(trimmed)
    } catch {
        return trimmed
    }
    const mockLeagueId = url.searchParams.get('mlid')
    if (mockLeagueId !== null && /^\d+$/.test(mockLeagueId)) return mockLeagueId
    const segments = url.pathname.split('/').filter(segment => segment !== '')
    const gameCodeIndex = segments.indexOf('nba')
    const candidate = gameCodeIndex === -1 ? undefined : segments[gameCodeIndex + 1]
    if (candidate !== undefined && /^\d+$/.test(candidate)) return candidate
    return trimmed
}
