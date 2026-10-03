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

/** Stubs the live platform (see the header). `availableModes` is what the platform's connect reports. */
async function stubLivePlatform(page, { availableModes }) {
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
            player_assignments: { Ann: [JOKIC_ID], Bob: [], Cat: [], Dan: [] },
            injured_players: [], status: 'Success', remaining_cash: null,
        }) })
    })
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

test('Refresh Analysis follows the connection', async t => {
    const app = await launchAppPage()
    const { page } = app
    const refreshButton = page.locator('#live-refresh-btn')
    try {
        await loadApp(app)
        await stubLivePlatform(page, { availableModes: ALL_MODES })

        await t.test('connecting enables it', async () => {
            // The live layout is drawn when the platform is picked, before connecting, and connecting draws no new
            // layout -- the button kept the disabled state it was drawn with.
            await connectYahooLeague(app, '12345')
            assert.equal(await refreshButton.isDisabled(), false, 'a connected league must be refreshable')
            await refreshButton.click()
            await waitAppSettled(app, { timeout: 120000 })
            expectCleanSession(app, 'refresh after connecting')
        })

        await t.test('naming another league disables it, and naming the connected one again restores it', async () => {
            await typeYahooLeagueId(page, '67890')
            assert.equal(await refreshButton.isDisabled(), true,
                         'the button would poll the connected league while the sidebar names another')
            assert.equal(await indicatorText(page), 'Unconnected')
            await typeYahooLeagueId(page, '12345')
            assert.equal(await refreshButton.isDisabled(), false)
            assert.notEqual(await indicatorText(page), 'Unconnected')
            expectCleanSession(app, 'league id retyped')
        })
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
