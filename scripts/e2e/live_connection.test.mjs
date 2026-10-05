// scripts/e2e/live_connection.test.mjs
// Controls that must follow a live connection after they are drawn. Every bug here was the same mistake: a control's
// state computed when it was rendered, from state that a later step changed without re-rendering it -- Refresh
// Analysis disabled forever after connecting, a Season Mode connect that never loaded rosters, a mode list left
// narrowed after leaving the platform that narrowed it.
//
// The platform is stubbed, the backend is real. The stub reproduces the one server rule these flows hinge on: the
// draft state is refused until a session update carrying the league's platform_config has COMPLETED (as
// get_draft_state_route does). The platform fields are stripped before the real backend sees them, since it would
// otherwise try to reach the platform; the frontend still believes it is connected, which is all these tests read.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, setSelect, waitAppSettled, drainSessionFailures, expectCleanSession,
    readDropdownOptionLabels,
} from './helpers.mjs'

const LEAGUE_TEAMS = ['Ann', 'Bob', 'Cat', 'Dan']
const ALL_MODES = ['Draft Mode', 'Auction Mode', 'Season Mode']
const JOKIC_ID = 203999

/** Stubs the live platform (see the header). `availableModes` is what the platform's connect reports. Returns the
 *  platform's board, which a test can change to stand in for picks being made on the platform. */
async function stubLivePlatform(page, { availableModes }) {
    const platformBoard = { assignments: { Ann: [JOKIC_ID], Bob: [], Cat: [], Dan: [] } }
    const connectedSessions = new Set()
    await page.route(url => url.pathname.endsWith('/connect'), route => route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ team_names: LEAGUE_TEAMS, n_drafters: LEAGUE_TEAMS.length, n_picks: 13,
                               available_modes: availableModes, is_auction_draft: false }),
    }))
    await page.route(url => /^\/sessions(\/[^/]+)?$/.test(url.pathname), async route => {
        const request = route.request()
        if (!['PATCH', 'POST'].includes(request.method())) return route.continue()
        const body = request.postDataJSON()
        const carriesConfig = body.platform_config != null
        delete body.platform
        delete body.platform_config
        const response = await route.fetch({ postData: JSON.stringify(body) })
        const json = await response.json()
        if (carriesConfig && response.ok()) {
            connectedSessions.add(request.method() === 'POST' ? json.session_id : new URL(request.url()).pathname.split('/')[2])
        }
        await route.fulfill({ response, json })
    })
    await page.route(url => url.pathname.endsWith('/draft-state'), route => {
        const sessionId = new URL(route.request().url()).pathname.split('/')[2]
        if (!connectedSessions.has(sessionId)) {
            return route.fulfill({ status: 400, contentType: 'application/json',
                                   body: JSON.stringify({ detail: 'Session is not connected to a live platform.' }) })
        }
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({
            player_assignments: platformBoard.assignments,
            injured_players: [], status: 'Success', remaining_cash: null,
        }) })
    })
    return platformBoard
}

async function connectYahooLeague(app, leagueId) {
    const { page } = app
    await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
    await waitAppSettled(app, { timeout: 120000 })
    await typeYahooLeagueId(page, leagueId)
    await page.locator('#ls-connect-btn').click()
    await page.locator('#ls-n-drafters').evaluate(
        (input, count) => new Promise(resolve => {
            const check = () => input.value === String(count) ? resolve() : setTimeout(check, 50)
            check()
        }), LEAGUE_TEAMS.length)
    await waitAppSettled(app, { timeout: 120000 })
}

async function typeYahooLeagueId(page, leagueId) {
    const input = page.locator('#ls-yahoo-league-id')
    await input.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
    await input.fill(leagueId)
}

const indicatorText = page => page.locator('#eval-indicator').innerText()

test('the status follows the connection', async t => {
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })

        await t.test('before connecting it says so, and connecting clears it', async () => {
            await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
            await waitAppSettled(app, { timeout: 120000 })
            assert.equal(await indicatorText(page), 'Unconnected')
            await connectYahooLeague(app, '12345')
            assert.notEqual(await indicatorText(page), 'Unconnected')
            expectCleanSession(app, 'connected')
        })

        await t.test('naming another league says unconnected, and naming the connected one again clears it', async () => {
            // Connecting and editing the selection draw no new layout, so nothing else would update the status.
            await typeYahooLeagueId(page, '67890')
            assert.equal(await indicatorText(page), 'Unconnected')
            await typeYahooLeagueId(page, '12345')
            assert.notEqual(await indicatorText(page), 'Unconnected')
            expectCleanSession(app, 'league id retyped')
        })
    } finally {
        await app.close()
    }
})

test("a connected league's size is locked until the connection ends", async () => {
    // Drafters and picks are the platform's, read at connect; editing them rebuilt the session for a board shape the
    // polled draft does not have.
    const app = await launchAppPage()
    const { page } = app
    const isLocked = async () => [
        await page.locator('#ls-n-drafters').isDisabled(),
        await page.locator('#ls-n-picks').isDisabled(),
    ]
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await isLocked(), [false, false], 'unlocked before connecting')

        await connectYahooLeague(app, '12345')
        assert.deepEqual(await isLocked(), [true, true], 'locked while connected')

        await typeYahooLeagueId(page, '67890')
        assert.deepEqual(await isLocked(), [false, false], 'unlocked once another league is named')
        await typeYahooLeagueId(page, '12345')
        assert.deepEqual(await isLocked(), [true, true], 'locked again when the connected league is named again')
        expectCleanSession(app, 'size lock')
    } finally {
        await app.close()
    }
})

test('coming back to the tab shows the board being brought up to date', async () => {
    // Polling stops while the tab is hidden. Coming back must show "Updating..." at once and settle on "Updated" -- even
    // when no pick was made meanwhile, since that settling is how the user knows the board on screen is current.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')
        const indicatorStates = await page.evaluate(() => new Promise(resolve => {
            const indicator = document.getElementById('eval-indicator')
            const seen = []
            new MutationObserver(() => seen.push(indicator.textContent)).observe(indicator, { childList: true, characterData: true, subtree: true })
            const setVisibility = state => {
                Object.defineProperty(document, 'visibilityState', { value: state, configurable: true })
                document.dispatchEvent(new Event('visibilitychange'))
            }
            setVisibility('hidden')
            setVisibility('visible')
            const waitForSettled = () => seen.at(-1) === 'Updated' ? resolve(seen) : setTimeout(waitForSettled, 50)
            setTimeout(waitForSettled, 50)
        }))
        assert.equal(indicatorStates[0], 'Updating...', `the spinner must start first, saw ${JSON.stringify(indicatorStates)}`)
        assert.equal(indicatorStates.at(-1), 'Updated')
        expectCleanSession(app, 'returned to the tab')
    } finally {
        await app.close()
    }
})

test("while the user's last pick is pending, the board is followed in the background", async () => {
    // Platforms stop answering once a draft ends, so a user who leaves for the platform's tab on their last pick and
    // comes back after the draft could never see their final team. Polling carries on in the background just for
    // that pick, then stops.
    const app = await launchAppPage()
    const { page } = app
    const requestsSince = (start, fragment) =>
        app.sessionRequestLog.slice(start).filter(entry => entry.includes(fragment)).length
    const setVisibility = state => page.evaluate(visibility => {
        Object.defineProperty(document, 'visibilityState', { value: visibility, configurable: true })
        document.dispatchEvent(new Event('visibilitychange'))
    }, state)
    // Real players, twelve a team: every seat (whichever is the user's) is one short of a 13-player roster.
    const players = [1626157, 201142, 1628374, 1627783, 1629636, 1630202, 1628969, 1641717, 1641764, 1642349, 1630591,
                     1642263, 1628983, 201935, 202695, 1630217, 1627826, 1628384, 1626181, 1643407, 1628392, 1630180,
                     202710, 1642856, 1628415, 1630162, 1630567, 1631096, 1628991, 1630700, 1642851, 1641709, 1629651,
                     1630245, 1629674, 1631099, 1631221, 1627832, 203999, 1626164, 1628389, 1627759, 1629638, 1642276,
                     203497, 1630166, 1628404, 1629614]
    const lastPicks = [1642273, 1631217, 1630174, 1629029]
    const boardWith = picksMade => Object.fromEntries(LEAGUE_TEAMS.map((team, index) => [
        team, [...players.slice(index * 12, index * 12 + 12), ...(index < picksMade ? [lastPicks[index]] : [])]]))
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')
        platformBoard.assignments = boardWith(0)
        await page.waitForFunction(() => document.getElementById('eval-indicator').textContent === 'Updated',
                                   null, { timeout: 120000 })
        await page.waitForTimeout(3000)   // the twelve-player board has been polled and evaluated

        await setVisibility('hidden')
        const start = app.sessionRequestLog.length
        await page.waitForTimeout(3000)
        assert.ok(requestsSince(start, '/draft-state') >= 1, 'polling must carry on while the last pick is pending')

        platformBoard.assignments = boardWith(LEAGUE_TEAMS.length)   // every last pick made, the user's included
        await page.waitForTimeout(4000)
        assert.ok(requestsSince(start, '/evaluate') >= 1, 'the finished team must be evaluated in the background')
        const settled = app.sessionRequestLog.length
        await page.waitForTimeout(4000)
        assert.equal(requestsSince(settled, '/draft-state'), 0, 'once the last pick is in, a hidden tab stops polling')
        await setVisibility('visible')
        await waitAppSettled(app, { timeout: 120000 })
        expectCleanSession(app, 'last pick followed in the background')
    } finally {
        await app.close()
    }
})

test('a live platform uses projections: Historical is offered only with your own data', async () => {
    // As in the Streamlit app. A live platform is a draft or season being played now, which a past season's stats
    // would rank for a year that is over (a user drafting from Historical data saw a board that made no sense).
    const app = await launchAppPage()
    const { page } = app
    const dataSource = () => page.locator('[data-testid="ps-data-type-wrapper"] .cs-search-input').inputValue()
    // The Player Stats section can be collapsed (a reload collapses it), and a closed section's options cannot be read.
    const readDataSourceOptions = async () => {
        await page.locator('[data-testid="ps-data-type-wrapper"]').evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        return readDropdownOptionLabels(page, 'ps-data-type-wrapper')
    }
    try {
        await loadApp(app)
        await setSelect(page, 'ps-data-type', 'Historical')
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await dataSource(), 'Historical')

        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await dataSource(), 'Projections', 'a live platform must switch the data source to projections')
        assert.deepEqual(await readDataSourceOptions(), ['Projections'])

        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await dataSource(), 'Projections', 'a remembered live platform must not reload into Historical')

        await setSelect(page, 'ls-platform', 'Enter your own data')
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await readDataSourceOptions(), ['Projections', 'Historical'])
        expectCleanSession(app, 'data source follows the platform')
    } finally {
        await app.close()
    }
})

test('a typed league id shows in the league dropdown, and clearing it restores the dropdown', async () => {
    // The typed id wins over the dropdown; a dropdown still naming a league beside one read as connecting to it.
    const app = await launchAppPage()
    const { page } = app
    const shownLeague = () => page.locator('[data-testid="ls-yahoo-league-wrapper"] .cs-search-input').inputValue()
    try {
        await loadApp(app)
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await shownLeague(), '(authenticate first)')
        await typeYahooLeagueId(page, '2606456')
        assert.equal(await shownLeague(), '(using the league ID below)')
        await typeYahooLeagueId(page, '')
        assert.equal(await shownLeague(), '(authenticate first)')
        expectCleanSession(app, 'league id typed and cleared')
    } finally {
        await app.close()
    }
})

test('a connected draft is followed without clicking Refresh', async t => {
    // The platform is polled every second; a changed board re-runs the analysis, an unchanged one does nothing.
    const app = await launchAppPage()
    const { page } = app
    const requestsSince = (start, fragment) =>
        app.sessionRequestLog.slice(start).filter(entry => entry.includes(fragment)).length
    const candidateRowsWith = name => page.locator('#hscoretable .playerheaderdiv', { hasText: name }).count()
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')

        await t.test('a pick made on the platform reaches the analysis, leaving open detail panels open', async () => {
            assert.ok(await candidateRowsWith('Bam Adebayo') > 0, 'Adebayo starts available')
            // Open the top candidate's details: the re-ranking a pick causes must not close them.
            const topCandidate = page.locator('#hscoretable .playerheaderdiv').first()
            const openedName = (await topCandidate.locator('.playername').first().innerText()).split('\n')[0].trim()
            await topCandidate.click()
            await page.locator('#hscoretable .playerpopup.popup-open').first().waitFor({ timeout: 10000 })

            platformBoard.assignments = { Ann: [JOKIC_ID], Bob: [1628389], Cat: [], Dan: [] }   // Bob takes Adebayo
            await page.waitForFunction(
                () => ![...document.querySelectorAll('#hscoretable .playerheaderdiv')]
                    .some(row => row.textContent.includes('Bam Adebayo')),
                { timeout: 15000 })
            await waitAppSettled(app, { timeout: 120000 })
            const openRow = page.locator('#hscoretable .playerheaderdiv', { hasText: openedName })
                .filter({ has: page.locator('.playerpopup.popup-open') })
            assert.equal(await openRow.count(), 1, `${openedName}'s details must still be open after the update`)
            expectCleanSession(app, 'pick picked up by polling')
        })

        await t.test('an unchanged board is polled quietly', async () => {
            const start = app.sessionRequestLog.length
            await page.waitForTimeout(5000)
            assert.ok(requestsSince(start, '/draft-state') >= 2, 'the board must keep being polled')
            assert.equal(requestsSince(start, '/evaluate'), 0, 'an unchanged board must not re-run the analysis')
            assert.notEqual(await indicatorText(page), 'Unconnected')
            expectCleanSession(app, 'quiet polling')
        })

        await t.test('naming another league stops the polling', async () => {
            await typeYahooLeagueId(page, '67890')
            await page.waitForTimeout(500)    // a poll already in flight may still land
            const start = app.sessionRequestLog.length
            await page.waitForTimeout(5000)
            assert.equal(requestsSince(start, '/draft-state'), 0, 'the old league must not keep being polled')
            expectCleanSession(app, 'polling stopped')
        })
    } finally {
        await app.close()
    }
})

test('when the platform stops reporting a finished draft, its results stay up', async () => {
    // A Yahoo mock room stops returning its results once the draft ends, and the integration reads "no results" as
    // "not started": an empty board. Polled as-is, it replaced the final analysis with base rankings.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')
        const resultsBefore = await page.locator('#hscoretable .overallhscore').allInnerTexts()

        platformBoard.assignments = { Ann: [], Bob: [], Cat: [], Dan: [] }
        await page.waitForTimeout(5000)
        assert.deepEqual(await page.locator('#hscoretable .overallhscore').allInnerTexts(), resultsBefore,
                         'the final results must stay up')
        const pollsBefore = app.sessionRequestLog.filter(entry => entry.includes('/draft-state')).length
        await page.waitForTimeout(5000)
        assert.equal(app.sessionRequestLog.filter(entry => entry.includes('/draft-state')).length, pollsBefore,
                     'a room that has stopped reporting is no longer polled')
        expectCleanSession(app, 'room closed')
    } finally {
        await app.close()
    }
})

for (const startMode of ['Season Mode', 'Draft Mode']) {
    test(`connecting in Season Mode loads the league's rosters (starting from ${startMode})`, async () => {
        // Rosters were fetched only on a mode or platform change, never on connecting, so the grid stayed blank under
        // the default team names. From Draft Mode the connect switches to Season Mode itself (the platform supports
        // nothing else, as with ESPN), which fired that change before the connection existed.
        const app = await launchAppPage()
        const { page } = app
        try {
            await loadApp(app)
            await setSelect(page, 'ls-mode', startMode)
            await waitAppSettled(app, { timeout: 120000 })
            await stubLivePlatform(page, { availableModes: ['Season Mode'] })
            await connectYahooLeague(app, '12345')

            const headers = await page.locator('.entry-table thead th').allInnerTexts()
            assert.deepEqual(headers.slice(1), LEAGUE_TEAMS, 'the grid must show the league\'s own teams')
            assert.match(await page.locator('.entry-table').first().innerText(), /Joki/,
                         'the grid must be filled from the platform\'s rosters')
            expectCleanSession(app, 'Season Mode connect')
        } finally {
            await app.close()
        }
    })
}

test('leaving a platform restores the modes its connection narrowed', async () => {
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await setSelect(page, 'ls-mode', 'Season Mode')
        await waitAppSettled(app, { timeout: 120000 })
        await stubLivePlatform(page, { availableModes: ['Season Mode'] })
        await connectYahooLeague(app, '12345')
        assert.deepEqual(await readDropdownOptionLabels(page, 'ls-mode-wrapper'), ['Season Mode'])

        await setSelect(page, 'ls-platform', 'Enter your own data')
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await readDropdownOptionLabels(page, 'ls-mode-wrapper'), ALL_MODES,
                         'own data supports every mode, whatever the last connection allowed')
        drainSessionFailures(app)
    } finally {
        await app.close()
    }
})

test('renaming a team relabels the pick controls', async t => {
    const app = await launchAppPage()
    const { page } = app
    async function renameFirstTeam(name) {
        await page.locator('.team-label-input').first().fill(name)
    }
    try {
        await loadApp(app)

        await t.test('draft: the pick line', async () => {
            await renameFirstTeam('Sharks')
            assert.equal(await page.locator('.pick-control-label').first().innerText(), 'Select Pick 1 for Sharks')
            await renameFirstTeam('')
        })

        await t.test('auction: the drafter list', async () => {
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app, { timeout: 120000 })
            // A name of its own: the draft subtest's must not be what this one reads, should that one fail before
            // clearing it (the row would then be DRAWN with it, and pass without relabelling anything).
            await renameFirstTeam('Jets')
            const drafters = await readDropdownOptionLabels(page, 'auction-pick-team-wrapper')
            assert.ok(drafters.includes('Jets'), `the renamed team must be offered under its new name — saw ${drafters}`)
            await renameFirstTeam('')
            await setSelect(page, 'ls-mode', 'Draft Mode')
            await waitAppSettled(app, { timeout: 120000 })
        })
        expectCleanSession(app, 'renames')
    } finally {
        await app.close()
    }
})

test('a category the backend drops does not send a session update', async () => {
    // syncCategoriesFromBackend removes the chip of a category the backend no longer scores. That edit is the
    // backend's own, so it must not fire the change a user's edit does -- which patched the session for nothing.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await page.route(url => url.pathname === '/sessions', async route => {
            if (route.request().method() !== 'POST') return route.continue()
            const response = await route.fetch()
            const json = await response.json()
            json.categories = json.categories.filter(category => category !== 'Turnovers')
            await route.fulfill({ response, json })
        })
        // Recorded from before the reload: the patch it would send is debounced, and waiting for the app to settle
        // waits for that patch too, so a log cleared after settling has already lost it.
        app.sessionRequestLog.length = 0
        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })
        await page.waitForTimeout(2000)
        await waitAppSettled(app, { timeout: 120000 })

        assert.equal(await page.locator('.ms-chip', { hasText: 'Turnovers' }).count(), 0, 'the dropped category\'s chip must go')
        const patches = app.sessionRequestLog.filter(entry => entry.startsWith('PATCH'))
        assert.deepEqual(patches, [], 'dropping a category the backend does not score is not a settings change')
        expectCleanSession(app, 'backend category drop')
    } finally {
        await app.close()
    }
})
