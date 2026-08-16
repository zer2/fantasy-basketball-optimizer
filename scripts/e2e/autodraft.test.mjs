// scripts/e2e/autodraft.test.mjs
// Autodraft: toggling a seat's "A" makes autopilot draft for that seat when its turn
// comes, control returns to the next manual seat, and undo rewinds through autodrafted
// picks back to the human drafter. One autodraft seat keeps the test fast (a single
// autopilot evaluate). Uses the 2024-25 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInDraftPick, readDropdownPlayerNames, readBoardPlayerNames, pickControlButton,
} from './helpers.mjs'

test('autodraft seats', async t => {
    const app = await launchAppPage()
    const { page } = app
    const pickLabel = () => page.locator('.pick-control-row .pick-control-label').first().textContent()
    const seatToggle = (index) => page.locator('.entry-table thead .method-dd-trigger').nth(index)

    let topCandidates = []
    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2024-25')
        expectCleanSession(app, 'initial load')

        await t.test('toggling a seat marks it as an autodrafter without starting a draft', async () => {
            // The board's base ordering — used later to judge the autopick's sanity.
            topCandidates = (await readDropdownPlayerNames(page, 'draft-pick-select-wrapper')).slice(0, 15)
            assert.ok(topCandidates.length === 15, 'pick dropdown should list the player pool')

            await seatToggle(1).click()
            assert.equal(await seatToggle(1).getAttribute('aria-pressed'), 'true',
                         "Team 2's autodraft toggle should read as pressed")
            assert.match(await pickLabel(), /Select Pick 1 for Team 1/,
                         'toggling a later seat must not move the draft off the manual Team 1')
            expectCleanSession(app, 'toggle autodraft seat')
        })

        await t.test('autopilot drafts a sensible player for the toggled seat, then yields to the next manual seat', async () => {
            await lockInDraftPick(page, 'Nikola Jokic')
            await waitAppSettled(app)

            assert.match(await pickLabel(), /Select Pick 1 for Team 3/,
                         'after the autopilot pick, the next manual seat should be on the clock')

            const drafted = await readBoardPlayerNames(page)
            assert.equal(drafted.length, 2, 'the board should hold the manual pick plus the autopilot pick')
            const autoPick = drafted.find(name => name !== 'Nikola Jokic')
            assert.ok(autoPick, 'the autopilot pick should not duplicate the manual pick')
            assert.ok(topCandidates.some(candidate => candidate === autoPick),
                      `autopilot's round-1 pick should come from the top of the pool — got: ${autoPick}`)

            const seatSelectorVisible = await page.evaluate(() =>
                getComputedStyle(document.getElementById('seat-selector-container')).visibility !== 'hidden')
            assert.ok(seatSelectorVisible, 'the seat selector should be restored once autopilot finishes')
            expectCleanSession(app, 'autopilot pick')
        })

        await t.test('undo rewinds through the autodrafted pick back to the manual drafter', async () => {
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)

            assert.equal((await readBoardPlayerNames(page)).length, 0,
                         'undo should clear the autodrafted pick and the manual pick behind it')
            assert.match(await pickLabel(), /Select Pick 1 for Team 1/,
                         'the draft should rewind to the manual drafter, not stop on the autodrafter')
            expectCleanSession(app, 'undo through autodraft')
        })
    } finally {
        await app.close()
    }
})
