// scripts/e2e/auction_board.test.mjs
// Auction-board entry controls: lock in / undo / clear, remaining-budget updates,
// and the over-budget guard. Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect, readDropdownOptionLabels, pickControlButton, lockInAuctionPick,
} from './helpers.mjs'

test('auction board entry controls', async t => {
    const app = await launchAppPage()
    const { page } = app
    const remainingRow = () => page.locator('.entry-table tr', { hasText: 'Remaining' }).first()
    const boardCellsWith = (text) => page.locator('.entry-table td', { hasText: text }).count()
    try {
        await loadApp(app)
        await setSelect(page, 'ls-mode', 'Auction Mode')
        await waitAppSettled(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load into auction mode')

        await t.test('lock in fills the board and reduces the remaining budget', async () => {
            await lockInAuctionPick(app, 'Nikola Jokic', 'Team 1', 70)

            assert.equal(await boardCellsWith('Nikola Jokic'), 1, 'locked player should appear on the board')
            const remainingText = await remainingRow().textContent()
            assert.ok(remainingText.includes('$130'), `Team 1's remaining budget should drop to $130 — got: ${remainingText}`)

            const optionLabels = await readDropdownOptionLabels(page, 'auction-pick-player-wrapper')
            assert.ok(!optionLabels.some(label => label.includes('Nikola Jokic')),
                      'locked player should leave the Player dropdown')
            expectCleanSession(app, 'lock in')
        })

        await t.test('a cost above the remaining budget is flagged and refused', async () => {
            await setSelect(page, 'auction-pick-player', 'Shai Gilgeous-Alexander')
            await setSelect(page, 'auction-pick-team', 'Team 1')
            await page.fill('.auction-cost-input', '200')   // Team 1 has $130 left

            const costInputClass = await page.locator('.auction-cost-input').getAttribute('class')
            assert.ok(costInputClass.includes('over-budget'), 'cost input should be flagged over-budget')

            await pickControlButton(page, 'Lock in selection').click()
            await waitAppSettled(app)
            assert.equal(await boardCellsWith('Shai Gilgeous-Alexander'), 0,
                         'an over-budget pick must not land on the board')
            expectCleanSession(app, 'over-budget lock attempt')
        })

        await t.test('undo restores the board, the budget, and the dropdown', async () => {
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)

            assert.equal(await boardCellsWith('Nikola Jokic'), 0, 'undone player should leave the board')
            const remainingText = await remainingRow().textContent()
            assert.ok(!remainingText.includes('$130'), 'the $130 budget entry should be gone after undo')

            const optionLabels = await readDropdownOptionLabels(page, 'auction-pick-player-wrapper')
            assert.ok(optionLabels.some(label => label.includes('Nikola Jokic')),
                      'undone player should reappear in the Player dropdown')
            expectCleanSession(app, 'undo')
        })

        await t.test('clear resets the board and every budget', async () => {
            await lockInAuctionPick(app, 'Nikola Jokic', 'Team 1', 70)
            await pickControlButton(page, 'Clear auction board').click()
            await waitAppSettled(app)

            assert.equal(await boardCellsWith('('), 0, 'clear should empty every board cell')
            const remainingText = await remainingRow().textContent()
            assert.ok(!remainingText.includes('$130'), 'every budget should reset after clear')
            expectCleanSession(app, 'clear board')
        })
    } finally {
        await app.close()
    }
})
