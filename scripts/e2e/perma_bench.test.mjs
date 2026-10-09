// scripts/e2e/perma_bench.test.mjs
// Perma-bench slots are picks: a league with 13 position slots and 3 bench slots drafts 16 rounds, the slot check
// counts the bench, the team is full once its 13 active slots are filled, and a bench pick is drafted without error
// (the backend scores each team on its active picks alone -- services/ranking.split_off_bench). Before 2026-10-08 the
// bench count was saved and never read: the check called 13 slots short of 16 picks.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, setLeagueDrafterCount, lockInTopDraftPick, countBoardPlayers, waitAppSettled,
    expectCleanSession,
} from './helpers.mjs'

const ACTIVE_SLOTS = 13
const BENCH_SLOTS = 3
const PICKS = ACTIVE_SLOTS + BENCH_SLOTS

async function setNumber(app, id, value) {
    const input = app.page.locator(`#${id}`)
    await input.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
    await input.fill(String(value))
    await input.evaluate(el => el.blur())
    await waitAppSettled(app)
}

test('perma-bench slots count as picks and are not scored', async () => {
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await setLeagueDrafterCount(app, 2)

        // the picks first (held back: 13 slots fall short of 16), then the bench that completes them -- which must
        // send the held-back picks change, or the session keeps 13 picks
        const leaguePatches = []
        page.on('request', request => {
            if (request.method() === 'PATCH' && /\/sessions\/[^/]+$/.test(new URL(request.url()).pathname)) {
                const league = request.postDataJSON()?.league
                if (league) leaguePatches.push(league.n_picks)
            }
        })
        await setNumber(app, 'ls-n-picks', PICKS)
        assert.match(await page.locator('#sc-validation').textContent(), /Slot total \(13\) is less than picks per drafter \(16\)/)
        assert.deepEqual(leaguePatches, [], 'an invalid slot structure must not be sent')
        await setNumber(app, 'sc-bench-slots', BENCH_SLOTS)
        assert.equal(await page.locator('#sc-validation').textContent(), '',
                     '13 slots plus 3 bench slots must satisfy 16 picks per drafter')
        assert.deepEqual(leaguePatches, [PICKS], 'the bench that completes the structure sends the 16 picks')
        expectCleanSession(app, 'picks and bench slots set')

        // two drafters, snake order: our team (the first) has its 13th pick at overall pick 26
        for (let pick = 1; pick <= 2 * ACTIVE_SLOTS; pick++) {
            await lockInTopDraftPick(app)
        }
        assert.equal(await countBoardPlayers(page), 2 * ACTIVE_SLOTS)
        const message = page.locator('#hscoretable .table-message')
        assert.equal(await message.textContent(), 'Your team is full.',
                     'with the 13 active slots filled, the team is full though 3 bench picks remain')

        // a bench pick for each team: drafted, nothing fails, the team still reads full
        await lockInTopDraftPick(app)
        await lockInTopDraftPick(app)
        assert.equal(await countBoardPlayers(page), 2 * ACTIVE_SLOTS + 2)
        assert.equal(await message.textContent(), 'Your team is full.')
        assert.equal(await page.locator('#hscoretable .table-message-error').count(), 0)
        expectCleanSession(app, 'bench picks drafted')
    } finally {
        await app.close()
    }
})
