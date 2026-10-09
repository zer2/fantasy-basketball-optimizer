// scripts/e2e/yahoo_league_id.test.mjs
// The Yahoo league-ID box accepts a pasted URL. Before 2026-10-09 the parser took the LAST numeric
// path segment, which is the seat in a draft-client URL, and could not read a mock lobby's ?mlid=
// at all -- so a pasted mock URL was sent to Yahoo whole ("Invalid league key 478.l.https:").
// No browser: the parser is pure, so this loads the compiled module directly. Loaded from its source text rather than
// by path because package.json says "type": "commonjs", so Node would read the ES module in dist as CommonJS; the
// module imports nothing, so a data URL carries it whole.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'

const compiledSource = await readFile(new URL('../../frontend/dist/platforms/yahoo_league_id.js', import.meta.url), 'utf8')
const { extractYahooLeagueId } = await import(`data:text/javascript;base64,${Buffer.from(compiledSource).toString('base64')}`)

test('a bare id is returned trimmed', () => {
    assert.equal(extractYahooLeagueId('  12345 '), '12345')
})

test('a league page URL gives the segment after the game code', () => {
    assert.equal(extractYahooLeagueId('https://basketball.fantasysports.yahoo.com/nba/12345'), '12345')
    assert.equal(extractYahooLeagueId('https://basketball.fantasysports.yahoo.com/nba/12345/1'), '12345',
                 'the team number after the league is not the league')
})

test('a draft-client URL gives the league, not the seat', () => {
    assert.equal(extractYahooLeagueId('https://basketball.fantasysports.yahoo.com/draftclient/nba/2678336/5?auth='),
                 '2678336')
})

test('a mock lobby URL gives its mlid', () => {
    assert.equal(extractYahooLeagueId('https://basketball.fantasysports.yahoo.com/nba/mock_waiting?mlid=2678371&lobby=standard'),
                 '2678371')
})

test('anything else is passed through unchanged, to fail at Yahoo with a message about it', () => {
    assert.equal(extractYahooLeagueId('https://basketball.fantasysports.yahoo.com/nba/'),
                 'https://basketball.fantasysports.yahoo.com/nba/')
    assert.equal(extractYahooLeagueId('not a league'), 'not a league')
})
