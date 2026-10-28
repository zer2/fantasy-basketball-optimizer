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
    readDropdownOptionLabels, APP,
} from './helpers.mjs'

const LEAGUE_TEAMS = ['Ann', 'Bob', 'Cat', 'Dan']
const ALL_MODES = ['Draft Mode', 'Auction Mode', 'Season Mode']
const JOKIC_ID = 203999

/** Stubs the live platform (see the header). `availableModes`, `picksPerTeam` and `slotCounts` are what the platform's
 *  connect reports (`slotCounts` null: a platform that does not report its positions). Returns the platform's board,
 *  which a test can change to stand in for picks being made on the platform; `configuredSlotCounts` collects the
 *  slot_counts of every session update that carried the connection. */
async function stubLivePlatform(page, { availableModes, picksPerTeam = 13, slotCounts = null }) {
    const platformBoard = { assignments: { Ann: [JOKIC_ID], Bob: [], Cat: [], Dan: [] }, configuredSlotCounts: [] }
    const connectedSessions = new Set()
    await page.route(url => url.pathname.endsWith('/connect'), route => route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ team_names: LEAGUE_TEAMS, n_drafters: LEAGUE_TEAMS.length, n_picks: picksPerTeam,
                               available_modes: availableModes, is_auction_draft: false, slot_counts: slotCounts }),
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
        if (carriesConfig) platformBoard.configuredSlotCounts.push(body.slot_counts)
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
            injured_players: [], unmatched_player_names: {}, empty_slots: {}, status: 'Success',
            remaining_cash: null,
        }) })
    })
    return platformBoard
}

async function connectYahooLeague(app, leagueId) {
    const { page } = app
    await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
    await waitAppSettled(app, { timeout: 120000 })
    await typeYahooLeagueId(page, leagueId)
    await page.locator('#ls-yahoo-league-id').press('Enter')   // naming the league connects
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
    // A hidden tab keeps polling, more slowly. Coming back must show "Updating..." at once and settle on "Updated" --
    // even when no pick was made meanwhile, since that settling is how the user knows the board on screen is current.
    // A board that background polling already kept current is only flashed, not re-evaluated.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')
        await page.waitForTimeout(3000)   // the first polls have answered: the board on screen is current
        const evaluatesBefore = app.sessionRequestLog.filter(entry => entry.includes('/evaluate')).length
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
        const evaluatesAfter = app.sessionRequestLog.filter(entry => entry.includes('/evaluate')).length
        assert.equal(evaluatesAfter, evaluatesBefore, 'an unchanged, current board is flashed, not re-evaluated')
        expectCleanSession(app, 'returned to the tab')
    } finally {
        await app.close()
    }
})

test('a hidden tab keeps following the draft, through the last pick', async () => {
    // A hidden tab polls every few seconds, so the board is current on coming back. That matters most for the last
    // pick: platforms stop answering once a draft ends, so a user who leaves for the platform's tab on their last pick
    // and comes back after the draft could otherwise never see their final team. Once the board is full, it stops.
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
        // The stub's rosters are current players (2025 rookies among them), who are in the projections pool only.
        await setSelect(page, 'ps-data-type', 'Projections')
        await waitAppSettled(app, { timeout: 120000 })
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        await connectYahooLeague(app, '12345')
        platformBoard.assignments = boardWith(0)
        await page.waitForFunction(() => document.getElementById('eval-indicator').textContent === 'Updated',
                                   null, { timeout: 120000 })
        await page.waitForTimeout(3000)   // the twelve-player board has been polled and evaluated

        await setVisibility('hidden')
        const start = app.sessionRequestLog.length
        await page.waitForTimeout(7000)   // longer than the hidden-tab interval (5 s)
        assert.ok(requestsSince(start, '/draft-state') >= 1, 'polling must carry on while the tab is hidden')

        platformBoard.assignments = boardWith(LEAGUE_TEAMS.length)   // every last pick made, the user's included
        await page.waitForTimeout(8000)
        assert.ok(requestsSince(start, '/evaluate') >= 1, 'the finished team must be evaluated in the background')
        const settled = app.sessionRequestLog.length
        await page.waitForTimeout(7000)
        assert.equal(requestsSince(settled, '/draft-state'), 0, 'once the board is full, a hidden tab stops polling')
        await setVisibility('visible')
        await waitAppSettled(app, { timeout: 120000 })
        expectCleanSession(app, 'last pick followed in the background')
    } finally {
        await app.close()
    }
})

test('Historical data stays available, and chosen, with a live platform', async () => {
    // A league can be connected in a past season (ESPN keeps a league's id across seasons), and that season's own stats
    // describe it. Historical used to be own-data only, switching the source to Projections on connecting (until
    // 2026-10-21).
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
        assert.equal(await dataSource(), 'Historical', 'a live platform keeps the historical source')
        assert.deepEqual(await readDataSourceOptions(), ['Projections', 'Historical'])

        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await dataSource(), 'Historical', 'a remembered live platform reloads with the remembered source')
        expectCleanSession(app, 'historical data with a live platform')
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

            const pickEvaluated = page.waitForRequest(request => request.url().endsWith('/evaluate')
                && request.postDataJSON().player_assignments.Bob?.includes(1628389), { timeout: 15000 })
            platformBoard.assignments = { Ann: [JOKIC_ID], Bob: [1628389], Cat: [], Dan: [] }   // Bob takes Adebayo
            // Waits for the evaluate that carries the pick: the table empties for a moment while an earlier evaluate
            // redraws it, which satisfied "Adebayo is gone" before the pick was even polled, and the pick's evaluate
            // then landed in the next step's quiet window.
            await pickEvaluated
            await page.waitForFunction(() => {
                const rows = [...document.querySelectorAll('#hscoretable .playerheaderdiv')]
                return rows.length > 0 && !rows.some(row => row.textContent.includes('Bam Adebayo'))
            }, null, { timeout: 15000 })
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

test('choosing a seat while the connection is still reaching the session waits for it', async () => {
    // Connect answers at once, but the session update it sends can queue behind a rebuild for many seconds, and the
    // board cannot be polled until that update lands. Choosing a seat meanwhile evaluated at once, polled, and painted
    // "Session is not connected to a live platform" over the table (2026-10-09, in a real mock draft).
    const app = await launchAppPage()
    const { page } = app
    const PATCH_DELAY_MS = 4000
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await waitAppSettled(app, { timeout: 120000 })
        // Holds up only the update that carries the connection, the way a queued rebuild would. Registered after the
        // stub's own session route, so it runs first (Playwright matches the newest route first) and the stub still
        // strips the platform fields and records the connection when the delayed request goes through.
        await page.route(url => /^\/sessions\/[^/]+$/.test(url.pathname), async route => {
            const request = route.request()
            if (request.method() === 'PATCH' && request.postDataJSON().platform_config != null) {
                await new Promise(resolve => setTimeout(resolve, PATCH_DELAY_MS))
            }
            await route.fallback()
        })
        await typeYahooLeagueId(page, '12345')
        await page.locator('#ls-yahoo-league-id').press('Enter')   // naming the league connects
        await page.locator('#ls-n-drafters').evaluate(
            (input, count) => new Promise(resolve => {
                const check = () => input.value === String(count) ? resolve() : setTimeout(check, 50)
                check()
            }), LEAGUE_TEAMS.length)
        await setSelect(page, 'seat-select', 'Bob')   // while the delayed update is still in flight
        await page.waitForTimeout(PATCH_DELAY_MS + 1000)
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await page.locator('#hscoretable .table-message-error').count(), 0,
                     'the seat change must not paint the not-connected refusal')
        assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0, 'the board must be up')
        assert.notEqual(await indicatorText(page), 'Unconnected')
        expectCleanSession(app, 'seat chosen during the connect update')
    } finally {
        await app.close()
    }
})

test('before the first pick the board is polled slowly, and at full speed once a pick is made', async () => {
    // A room can wait a long time for its draft, and a pick arriving a few seconds late before anyone has picked costs
    // nothing. Polling an empty room once a second -- in two forgotten tabs, all day -- drew Yahoo's rate limit.
    const app = await launchAppPage()
    const { page } = app
    const pollsDuring = async milliseconds => {
        const start = app.sessionRequestLog.length
        await page.waitForTimeout(milliseconds)
        return app.sessionRequestLog.slice(start).filter(entry => entry.includes('/draft-state')).length
    }
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        platformBoard.assignments = { Ann: [], Bob: [], Cat: [], Dan: [] }
        await connectYahooLeague(app, '12345')
        assert.ok(await pollsDuring(12000) <= 2, 'an empty board is polled about every 10 seconds')

        platformBoard.assignments = { Ann: [JOKIC_ID], Bob: [], Cat: [], Dan: [] }
        await page.waitForTimeout(11000)   // the slow poll picks up the first pick
        await waitAppSettled(app, { timeout: 120000 })
        assert.ok(await pollsDuring(5000) >= 3, 'after the first pick the board is polled every second')
        expectCleanSession(app, 'slow then fast polling')
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

test("a Waiver tab drawn before the league's rosters arrive is redrawn when they do", async () => {
    // Season Mode opens on the Waiver tab, which drew while the rosters were still empty ("Fill out at least one full
    // team") and was never drawn again: the rosters' arrival redrew only the hidden Rosters grid (2026-10-13).
    const app = await launchAppPage()
    const { page } = app
    const players = [1626157, 201142, 1628374, 1627783, 1629636, 1630202, 1628969, 1641717, 1641764, 1642349, 1630591,
                     1642263, 1628983, 201935, 202695, 1630217, 1627826, 1628384, 1626181, 1643407, 1628392, 1630180,
                     202710, 1642856, 1628415, 1630162, 1630567, 1631096, 1628991, 1630700, 1642851, 1641709, 1629651,
                     1630245, 1629674, 1631099, 1631221, 1627832, 203999, 1626164, 1628389, 1627759, 1629638, 1642276,
                     203497, 1630166, 1628404, 1629614, 1642273, 1631217, 1630174, 1629029]
    try {
        await loadApp(app)
        // The stub's rosters are current players (2025 rookies among them), who are in the projections pool only.
        await setSelect(page, 'ps-data-type', 'Projections')
        await waitAppSettled(app, { timeout: 120000 })
        await setSelect(page, 'ls-mode', 'Season Mode')
        await waitAppSettled(app, { timeout: 120000 })
        const platformBoard = await stubLivePlatform(page, { availableModes: ['Season Mode'] })
        platformBoard.assignments = Object.fromEntries(LEAGUE_TEAMS.map((team, index) =>
            [team, players.slice(index * 13, index * 13 + 13)]))
        // Hold the league's rosters: while they load, the indicator beside the tabs says "Starting...", the tab says
        // nothing (no request to fill rosters in, 2026-10-13, and no loading line of its own, 2026-10-28), and the empty
        // table stays hidden.
        let releaseRosters
        const rostersHeld = new Promise(resolve => { releaseRosters = resolve })
        await page.route(url => url.pathname.endsWith('/draft-state'), async route => {
            await rostersHeld
            await route.fallback()
        })
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await typeYahooLeagueId(page, '12345')
        await page.locator('#ls-yahoo-league-id').press('Enter')
        await page.waitForFunction(() => document.querySelector('#season-indicator')?.dataset.state === 'fetching',
                                   null, { timeout: 30000 })
        assert.match(await page.locator('#season-indicator').innerText(), /Starting/)
        const tabText = await page.locator('#right-header').innerText()
        assert.doesNotMatch(tabText, /Fill out|Loading/, 'the tab says nothing while the rosters load')
        assert.equal(await page.locator('#hscoretable').isVisible(), false, 'no empty table while the rosters load')
        releaseRosters()
        await waitAppSettled(app, { timeout: 120000 })
        await page.waitForFunction(() => document.querySelector('[data-testid="waiver-drop-select-wrapper"]') !== null,
                                   null, { timeout: 30000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await page.locator('.season-tab-btn.active').getAttribute('data-tab'), 'waiver')
        assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                  'the waiver candidates must render without clicking the tab again')
        expectCleanSession(app, 'waiver redrawn on roster arrival')
    } finally {
        await app.close()
    }
})

/** The Position Parameters inputs, by slot, as the page shows them. */
const readSlotInputs = page => page.evaluate(() => Object.fromEntries(
    [...document.querySelectorAll('input[id^="sc-"]')].filter(input => input.id !== 'sc-bench-slots')
        .map(input => [input.id.slice(3), Number(input.value)])))

test("connecting takes the league's own position structure", async () => {
    // An 11-pick league kept the 13-slot default and was scored as a roster shape it does not have (2026-10-12).
    const app = await launchAppPage()
    const { page } = app
    const leagueSlots = { PG: 1, SG: 1, G: 1, SF: 1, PF: 1, F: 1, C: 1, Util: 4 }
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES, picksPerTeam: 11,
                                                             slotCounts: leagueSlots })
        await connectYahooLeague(app, '12345')
        const shown = await readSlotInputs(page)
        assert.deepEqual(Object.fromEntries(Object.entries(shown).filter(([, count]) => count > 0)),
                         Object.fromEntries(Object.entries(leagueSlots).map(([slot, count]) => [slot.toLowerCase(), count])))
        assert.equal(await page.locator('#sc-bench-slots').inputValue(), '0')
        assert.deepEqual(platformBoard.configuredSlotCounts.at(-1), leagueSlots, 'the session gets the league structure')
        expectCleanSession(app, 'league position structure')
    } finally {
        await app.close()
    }
})

test("the league's position structure is a default: edits survive a reconnect until the league's own changes", async () => {
    // Every page load and mode change reconnects, and each reconnect used to put the platform's slots back, so moving
    // Utils to perma-bench lasted only until the next one (2026-10-14).
    const app = await launchAppPage()
    const { page } = app
    const leagueSlots = { PG: 1, SG: 1, G: 1, SF: 1, PF: 1, F: 1, C: 1, Util: 4 }
    const readBench = () => page.locator('#sc-bench-slots').inputValue()
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES, picksPerTeam: 11,
                                                             slotCounts: leagueSlots })
        // The test account has no Yahoo token; an empty league list stands in for an authorized one, so a reload
        // reconnects to the league named last.
        await page.route(url => url.pathname.endsWith('/leagues'), route => route.fulfill({
            status: 200, contentType: 'application/json', body: JSON.stringify({ leagues: [] }) }))
        await connectYahooLeague(app, '12345')

        // Two Utils to perma-bench: the slot box first (the check fails), then the bench box (it passes and applies).
        await page.evaluate(() => {
            for (const [id, value] of [['sc-util', '2'], ['sc-bench-slots', '2']]) {
                const input = document.getElementById(id)
                input.value = value
                input.dispatchEvent(new Event('input', { bubbles: true }))
                input.dispatchEvent(new Event('change', { bubbles: true }))
            }
        })
        await waitAppSettled(app, { timeout: 120000 })

        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.waitForFunction(() => document.querySelector('#ls-n-drafters')?.value === '4', null, { timeout: 60000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal((await readSlotInputs(page)).util, 2, 'the reconnect keeps the edited Util count')
        assert.equal(await readBench(), '2', '...and the perma-bench')
        assert.equal(platformBoard.configuredSlotCounts.at(-1).Util, 2, 'the session gets the edited structure')

        // The league's own roster settings change: the platform's new structure replaces the edit.
        const changedSlots = { ...leagueSlots, C: 2, Util: 3 }
        await page.route(url => url.pathname.endsWith('/connect'), route => route.fulfill({
            status: 200, contentType: 'application/json',
            body: JSON.stringify({ team_names: LEAGUE_TEAMS, n_drafters: LEAGUE_TEAMS.length, n_picks: 11,
                                   available_modes: ALL_MODES, is_auction_draft: false, slot_counts: changedSlots }),
        }))
        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.waitForFunction(() => document.querySelector('#ls-n-drafters')?.value === '4', null, { timeout: 60000 })
        await waitAppSettled(app, { timeout: 120000 })
        const shown = await readSlotInputs(page)
        assert.equal(shown.c, 2)
        assert.equal(shown.util, 3)
        assert.equal(await readBench(), '0')
        expectCleanSession(app, 'platform structure as a default')
    } finally {
        await app.close()
    }
})

test("manual entry keeps its own league shape: connecting a league never overwrites it", async () => {
    // A connection used to save the league's slots over manual entry's while leaving its pick count unsaved, so after
    // a reload manual entry showed its own picks beside the league's slots (2026-10-15).
    const app = await launchAppPage()
    const { page } = app
    const leagueSlots = { PG: 1, SG: 1, G: 1, SF: 1, PF: 1, F: 1, C: 1, Util: 4 }
    const readShape = () => page.evaluate(() => ({
        drafters: document.getElementById('ls-n-drafters').value,
        picks: document.getElementById('ls-n-picks').value,
        util: document.getElementById('sc-util').value,
        bench: document.getElementById('sc-bench-slots').value,
    }))
    try {
        await loadApp(app)
        const manual = await readShape()
        await stubLivePlatform(page, { availableModes: ALL_MODES, picksPerTeam: 11, slotCounts: leagueSlots })
        await page.route(url => url.pathname.endsWith('/leagues'), route => route.fulfill({
            status: 200, contentType: 'application/json', body: JSON.stringify({ leagues: [] }) }))
        await connectYahooLeague(app, '12345')
        assert.deepEqual(await readShape(), { drafters: String(LEAGUE_TEAMS.length), picks: '11', util: '4', bench: '0' })

        const sessionUpdates = []
        page.on('request', request => {
            if (request.method() === 'PATCH' && /\/sessions\/[^/]+$/.test(new URL(request.url()).pathname)) {
                sessionUpdates.push(request.postDataJSON())
            }
        })
        await setSelect(page, 'ls-platform', 'Enter your own data')
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await readShape(), manual, "own data shows manual entry's shape again")
        const update = sessionUpdates.at(-1)
        assert.equal(update.league.n_picks, Number(manual.picks), "...and the session gets it")
        assert.equal(update.league.n_drafters, Number(manual.drafters))
        assert.equal(update.slot_counts.Util, Number(manual.util))

        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.waitForFunction(() => document.getElementById('ls-n-picks') !== null, null, { timeout: 60000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await readShape(), manual, 'a reload keeps it')

        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await page.waitForFunction(() => document.getElementById('ls-n-picks').value === '11', null, { timeout: 60000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.deepEqual(await readShape(), { drafters: String(LEAGUE_TEAMS.length), picks: '11', util: '4', bench: '0' },
                         "reconnecting brings the league's shape back")
        expectCleanSession(app, 'manual and league shapes kept apart')
    } finally {
        await app.close()
    }
})

test('a connection whose picks no structure fits waits for the Position Parameters, then loads', async () => {
    // A platform that does not report positions, at a pick count with no default structure: the 13-slot default was
    // sent for 11 picks. Now nothing reaches the session until the slots fit, and applying them sends the connection.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES, picksPerTeam: 11 })
        await connectYahooLeague(app, '12345')
        assert.match(await page.locator('.table-message-error').innerText(), /11 roster spots/)
        assert.deepEqual(platformBoard.configuredSlotCounts, [], 'a structure that does not fit is never sent')

        await page.evaluate(() => {
            for (const input of document.querySelectorAll('input[id^="sc-"]')) {
                input.value = input.id === 'sc-util' ? '11' : '0'
                input.dispatchEvent(new Event('input', { bubbles: true }))
            }
            let section = document.getElementById('sc-util').parentElement
            while (!section.querySelector('.section-apply-btn')) section = section.parentElement
            section.querySelector('.section-apply-btn').click()
        })
        await page.waitForFunction(() => document.querySelector('.table-message-error') === null, null, { timeout: 30000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(platformBoard.configuredSlotCounts.length, 1, 'applying a fitting structure sends the connection once')
        const sentSlots = platformBoard.configuredSlotCounts[0]
        assert.equal(sentSlots.Util, 11)
        assert.equal(Object.values(sentSlots).reduce((total, count) => total + count, 0), 11, '...with that structure')
        expectCleanSession(app, 'connection held for slots')
    } finally {
        await app.close()
    }
})

test('connecting while the first session is still building updates it instead of building a second', async () => {
    // The connect found no session id while the page-load create ran, and created another beside it: two builds
    // slowed each other, and whichever finished last became the session -- the page-load one carried the old
    // position structure and no league (2026-10-13).
    const app = await launchAppPage()
    const { page } = app
    try {
        const platformBoard = await stubLivePlatform(page, { availableModes: ALL_MODES })
        let sessionCreates = 0
        let releaseFirstCreate
        const firstCreateHeld = new Promise(resolve => { releaseFirstCreate = resolve })
        await page.route(url => url.pathname === '/sessions', async route => {
            if (route.request().method() === 'POST') {
                sessionCreates += 1
                if (sessionCreates === 1) await firstCreateHeld
            }
            await route.fallback()
        })
        await page.goto(APP, { waitUntil: 'domcontentloaded' })
        await page.waitForFunction(() => document.querySelector('[data-testid="ls-platform-wrapper"]') !== null)
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await typeYahooLeagueId(page, '12345')
        await page.locator('#ls-yahoo-league-id').press('Enter')   // naming the league connects
        await page.waitForTimeout(1500)   // the connect lands while the first create is still held
        releaseFirstCreate()
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(sessionCreates, 1, 'one session, updated by the connection')
        assert.equal(platformBoard.configuredSlotCounts.length, 1, 'the connection reached that session')
        expectCleanSession(app, 'connect during the first create')
    } finally {
        await app.close()
    }
})

test('the app connects whenever it can: on naming a league, after a failure on Try again, and after a reload', async () => {
    // There is no Connect button. Naming a league connects; a failed connect offers Try again; the league named last
    // reconnects once the page is loaded again.
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })
        // The test account has no Yahoo token; an empty league list stands in for an authorized one.
        await page.route(url => url.pathname.endsWith('/leagues'), route => route.fulfill({
            status: 200, contentType: 'application/json', body: JSON.stringify({ leagues: [] }) }))
        let connectRequests = 0
        let failNextConnect = false
        page.on('request', request => { if (new URL(request.url()).pathname.endsWith('/connect')) connectRequests += 1 })
        await page.route(url => url.pathname.endsWith('/connect'), async route => {
            if (!failNextConnect) return route.fallback()
            failNextConnect = false
            await route.fulfill({ status: 429, contentType: 'application/json',
                                  body: JSON.stringify({ detail: 'Yahoo is rate limiting; try again shortly.' }) })
        })
        await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
        await waitAppSettled(app, { timeout: 120000 })

        await typeYahooLeagueId(page, '12345')
        await page.locator('#ls-yahoo-league-id').press('Enter')
        await page.waitForFunction(() => document.querySelector('#ls-n-drafters').value === '4', null, { timeout: 30000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(connectRequests, 1, 'Enter in the league-id box connects')
        expectCleanSession(app, 'connect on naming a league')

        failNextConnect = true
        await typeYahooLeagueId(page, '67890')
        await page.locator('#ls-yahoo-league-id').blur()   // leaving the box names the league too
        await page.locator('#ls-connect-retry').waitFor({ timeout: 30000 })
        assert.match(await page.locator('#ls-connect-status').innerText(), /Connect failed/)
        assert.equal(connectRequests, 2)
        await page.locator('#ls-connect-retry').click()
        await page.waitForFunction(() => document.querySelector('#ls-connect-status').textContent === '', null, { timeout: 30000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(connectRequests, 3, 'Try again connects once more')
        drainSessionFailures(app)   // the failed connect was the point

        await page.reload({ waitUntil: 'domcontentloaded' })
        await page.waitForFunction(() => document.querySelector('#ls-n-drafters')?.value === '4', null, { timeout: 60000 })
        await waitAppSettled(app, { timeout: 120000 })
        assert.equal(await page.locator('#ls-yahoo-league-id').inputValue(), '67890', 'the league named last is named again')
        assert.equal(connectRequests, 4, 'and connects without being asked')
        expectCleanSession(app, 'reconnect after a reload')
    } finally {
        await app.close()
    }
})

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
