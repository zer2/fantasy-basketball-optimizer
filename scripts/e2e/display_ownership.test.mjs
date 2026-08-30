// scripts/e2e/display_ownership.test.mjs
// Deterministic race test for the eval-display ownership protocol (withDisplayOwnership in
// api/session.ts): a response that arrives after a newer actor has taken over the display
// must be discarded — no board repaint, no indicator overwrite.
//
// The race is real but timing-dependent in normal use, so it is made deterministic here by
// holding one /evaluate request at the network layer while the user moves on, then releasing
// it. The held request is showDefaultRankings' empty-board evaluation: it carries no abort
// signal, so nothing cancels it — ownership is the only thing standing between its late
// response and the board. Before the ownership protocol, this test's release step repainted
// the draft board with empty-board rankings and stamped the indicator 'unconnected'.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInDraftPick, setSelect,
} from './helpers.mjs'

const readIndicatorState = page => page.evaluate(() =>
    document.querySelector('#eval-indicator')?.dataset.state)

// Bare name of the top candidate row, extracted the same way readDropdownPlayerNames does:
// the .playername span's first text node, without the trailing positions span.
const readTopCandidateName = page => page.evaluate(() => {
    const nameSpan = document.querySelector('#hscoretable .playername')
    return nameSpan ? nameSpan.childNodes[0].textContent.trim() : null
})

test('a superseded evaluate cannot repaint the board or the indicator', async t => {
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2024-25')
        expectCleanSession(app, 'initial load')
        const baseTopName = await readTopCandidateName(page)
        assert.ok(baseTopName, 'the base board should have a top candidate')

        // Hold the NEXT /evaluate at the network layer. Switching to an unconnected live
        // platform fires showDefaultRankings, whose evaluate is the first to arrive here;
        // everything after it passes through untouched.
        let heldRoute = null
        await page.route('**/evaluate', route => {
            if (heldRoute === null) { heldRoute = route; return }
            return route.continue()
        })
        await setSelect(page, 'ls-platform', 'Retrieve from Fantrax')
        const holdDeadline = Date.now() + 15000
        while (heldRoute === null && Date.now() < holdDeadline) await page.waitForTimeout(50)
        assert.ok(heldRoute, 'switching to an unconnected platform should fire an evaluate')
        // The held request never finishes while held; waitAppSettled must not wait on it.
        app.pendingSessionRequests.delete(heldRoute.request())

        await t.test('the user moves on while the response is pending', async () => {
            await setSelect(page, 'ls-platform', 'Enter your own data')
            await waitAppSettled(app)
            await lockInDraftPick(page, baseTopName)
            await waitAppSettled(app)
            assert.notEqual(await readTopCandidateName(page), baseTopName,
                            'drafting the top player must change the top candidate')
            expectCleanSession(app, 'draft while held')
        })

        await t.test('the stale response is discarded on release', async () => {
            const pickTwoTopName = await readTopCandidateName(page)
            await heldRoute.continue()
            // Give the released response ample time to land and (incorrectly) render.
            await page.waitForTimeout(1500)
            assert.equal(await readTopCandidateName(page), pickTwoTopName,
                         'a stale empty-board response must not repaint the board')
            assert.equal(await readIndicatorState(page), 'idle',
                         "a stale run must not overwrite the indicator (e.g. with 'unconnected')")
            expectCleanSession(app, 'release')
        })
    } finally {
        await app.close()
    }
})
