// scripts/e2e/draft_board.test.mjs
// Draft-board entry controls: lock in / undo / clear, pick-order label, and
// dropdown membership. Uses the 2024-25 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInDraftPick, lockInTopDraftPick, readDropdownOptionLabels, pickControlButton,
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

        // ── Pick order across the round-2/3 boundary, with and without third round reversal ──
        // A 2-drafter league reaches round 3 in four locks: rounds go [T1,T2], [T2,T1], then
        // round 3 is [T1,T2] under normal snaking but repeats [T2,T1] under reversal.

        async function setDrafterCount(count) {
            const draftersInput = page.locator('#ls-n-drafters')
            await draftersInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            await draftersInput.fill(String(count))
            await draftersInput.dispatchEvent('change')
            await waitAppSettled(app)
        }

        async function setThirdRoundReversal(enabled) {
            const checkbox = page.locator('#ls-third-round-reversal')
            await checkbox.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            const currentlyEnabled = await checkbox.evaluate(el => el.checked)
            if (currentlyEnabled !== enabled) {
                await checkbox.evaluate(el => el.click())   // styled toggle; the input itself is not clickable
                // The board applies a config change on its next render — force one now so the
                // following locks run against the new pick order from a clean board.
                await pickControlButton(page, 'Clear draft board').click()
                await waitAppSettled(app)
            }
        }

        async function walkToRoundThree() {
            for (let lockCount = 0; lockCount < 4; lockCount++) await lockInTopDraftPick(app)
        }

        await t.test('third round reversal repeats the round-two order', async () => {
            await setDrafterCount(2)
            await setThirdRoundReversal(true)
            await walkToRoundThree()
            assert.match(await pickLabel(), /Select Pick 3 for Team 2/,
                         'with reversal, round 3 should open with the same drafter that opened round 2')
            expectCleanSession(app, 'third round reversal on')
        })

        await t.test('without reversal, round three snakes back to the first drafter', async () => {
            await setThirdRoundReversal(false)
            await walkToRoundThree()
            assert.match(await pickLabel(), /Select Pick 3 for Team 1/,
                         'without reversal, round 3 should open with the first drafter again')
            expectCleanSession(app, 'third round reversal off')
        })
    } finally {
        await app.close()
    }
})
