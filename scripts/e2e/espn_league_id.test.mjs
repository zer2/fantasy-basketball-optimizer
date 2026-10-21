// scripts/e2e/espn_league_id.test.mjs
// The ESPN league-ID box (public leagues, no cookies) accepts a bare id or a pasted league URL. The season in a URL
// matters: ESPN keeps a league's id across seasons, and the current season of a league that has not drafted yet has
// empty rosters, so a pasted 2026 URL must not be read as the current season.
// No browser: the parser is pure, so this loads the compiled module directly (see yahoo_league_id.test.mjs for why it
// is loaded from its source text).

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const compiledSource = await readFile(new URL('../../frontend/dist/platforms/espn_league_id.js', import.meta.url), 'utf8')
const { extractEspnLeagueId, splitEspnLeagueId, formatEspnSeason } = await import(`data:text/javascript;base64,${Buffer.from(compiledSource).toString('base64')}`)

test('a bare id is returned trimmed, for the backend to put in the current season', () => {
    assert.equal(extractEspnLeagueId('  1801165221 '), '1801165221')
})

test('a league URL gives its id and season', () => {
    assert.equal(extractEspnLeagueId('https://fantasy.espn.com/basketball/league/settings?leagueId=1801165221&seasonId=2026'),
                 '1801165221::2026')
})

test('a league URL copied without its scheme is still read', () => {
    assert.equal(extractEspnLeagueId('fantasy.espn.com/basketball/league?leagueId=1801165221&seasonId=2026'),
                 '1801165221::2026')
})

test('a league URL without a season gives the bare id', () => {
    assert.equal(extractEspnLeagueId('https://fantasy.espn.com/basketball/league?leagueId=1801165221'), '1801165221')
})

test("a women's basketball URL is marked as one", () => {
    assert.equal(extractEspnLeagueId('https://fantasy.espn.com/womens-basketball/league?leagueId=98765&seasonId=2026'),
                 'wfba:98765::2026')
})

test('anything else is passed through unchanged, to fail at ESPN with a message about it', () => {
    assert.equal(extractEspnLeagueId('https://fantasy.espn.com/basketball/league'),
                 'https://fantasy.espn.com/basketball/league')
    assert.equal(extractEspnLeagueId('not a league'), 'not a league')
})

test('an id splits into its league and season', () => {
    assert.deepEqual(splitEspnLeagueId('1801165221::2026'), { league: '1801165221', season: 2026 })
    assert.deepEqual(splitEspnLeagueId('1801165221'), { league: '1801165221', season: null })
    assert.deepEqual(splitEspnLeagueId('wfba:98765::2026'), { league: 'wfba:98765', season: 2026 })
})

test('a season is shown as the years it spans', () => {
    assert.equal(formatEspnSeason(2027), '2026-27')
})
