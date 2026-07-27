// scripts/e2e/draft_board.test.mjs
// Draft-board entry controls: lock in / undo / clear, pick-order label, and
// dropdown membership. Uses the 2024-25 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInDraftPick, readDropdownOptionLabels, pickControlButton,
} from './helpers.mjs'

test('draft board entry controls', async t => {
    const app = await launchAppPage()
    const { page } = app
    // Scoped to the pick-control row: the sidebar's connect-status element shares the
    // pick-control-label class and would otherwise match first.
    const pickLabel = () => page.locator('.pick-control-row .pick-control-label').first().textContent()
    const boardCellsWith = (text) => page.locator('.entry-table td', { hasText: text }).count()
    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2024-25')
        expectCleanSession(app, 'initial load')

        await t.test('board starts empty with Team 1 on the clock', async () => {
            assert.match(await pickLabel(), /Select Pick 1 for Team 1/)
            assert.equal(await boardCellsWith('('), 0, 'no board cell should contain a player yet')
        })

        await t.test('lock in fills the cell, advances the pick, and removes the player from the dropdown', async () => {
            await lockInDraftPick(page, 'Nikola Jokic')
            await waitAppSettled(app)
            assert.equal(await boardCellsWith('Nikola Jokic'), 1, 'locked player should appear on the board')
            assert.match(await pickLabel(), /Select Pick 1 for Team 2/, 'the next drafter should be on the clock')

            const optionLabels = await readDropdownOptionLabels(page, 'draft-pick-select-wrapper')
            assert.ok(!optionLabels.some(label => label.includes('Nikola Jokic')),
                      'locked player should leave the pick dropdown')
            expectCleanSession(app, 'lock in')
        })

        await t.test('undo clears the cell, rewinds the pick, and restores the dropdown', async () => {
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)

            assert.equal(await boardCellsWith('Nikola Jokic'), 0, 'undone player should leave the board')
            assert.match(await pickLabel(), /Select Pick 1 for Team 1/, 'the pick should rewind to Team 1')

            const optionLabels = await readDropdownOptionLabels(page, 'draft-pick-select-wrapper')
            assert.ok(optionLabels.some(label => label.includes('Nikola Jokic')),
                      'undone player should reappear in the pick dropdown')
            expectCleanSession(app, 'undo')
        })

        await t.test('clear resets a board with multiple picks', async () => {
            await lockInDraftPick(page, 'Nikola Jokic')
            await waitAppSettled(app)
            await lockInDraftPick(page, 'Shai Gilgeous-Alexander')
            await waitAppSettled(app)
            assert.match(await pickLabel(), /Select Pick 1 for Team 3/)

            await pickControlButton(page, 'Clear draft board').click()
            await waitAppSettled(app)

            assert.equal(await boardCellsWith('('), 0, 'clear should empty every board cell')
            assert.match(await pickLabel(), /Select Pick 1 for Team 1/, 'clear should rewind to the first pick')
            expectCleanSession(app, 'clear board')
        })
    } finally {
        await app.close()
    }
})
