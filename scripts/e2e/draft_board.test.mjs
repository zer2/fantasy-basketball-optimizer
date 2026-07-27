// scripts/e2e/draft_board.test.mjs
// Draft-board entry controls: lock in / undo / clear, pick-order label, and
// dropdown membership. Uses the 2024-25 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInDraftPick, lockInTopDraftPick, readDropdownOptionLabels, pickControlButton,
    setLeagueDrafterCount,
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
        // round 3 is [T1,T2] under normal snaking but repeats [T2,T1] under reversal. The
        // stepping consumes the sidebar setting exactly at the round-2/3 boundary: a toggle
        // before the crossing applies live (and never resets the board); a toggle after it
        // stays pending until an undo brings the draft back before the boundary.

        async function setThirdRoundReversal(enabled) {
            const checkbox = page.locator('#ls-third-round-reversal')
            await checkbox.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            const currentlyEnabled = await checkbox.evaluate(el => el.checked)
            if (currentlyEnabled !== enabled) {
                await checkbox.evaluate(el => el.click())   // styled toggle; the input itself is not clickable
            }
        }

        await t.test('reversal toggled before round 3 applies at the crossing without a reset', async () => {
            await setLeagueDrafterCount(app, 2)
            await setThirdRoundReversal(true)
            for (let lockCount = 0; lockCount < 4; lockCount++) await lockInTopDraftPick(app)
            assert.equal((await page.locator('.entry-table td.drafted').count()), 4,
                         'toggling reversal before the draft must not have reset the board')
            assert.match(await pickLabel(), /Select Pick 3 for Team 2/,
                         'with reversal, round 3 should open with the same drafter that opened round 2')
            expectCleanSession(app, 'reversal applied at crossing')
        })

        await t.test('reversal toggled past the boundary stays pending until undone back before it', async () => {
            // The draft sits at round 3 (laid out under reversal). Toggling off now must NOT
            // re-interpret the current position: the next pick still follows the [T2, T1]
            // reversal order rather than jumping to round 4.
            await setThirdRoundReversal(false)
            await lockInTopDraftPick(app)
            assert.match(await pickLabel(), /Select Pick 3 for Team 1/,
                         "a toggle past the boundary must not change round 3's in-progress order")

            // Undo back across the boundary (two steps: T2's round-3 pick, then T1's round-2
            // pick). The retrace must follow the reversal path the draft actually took.
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)
            assert.match(await pickLabel(), /Select Pick 2 for Team 1/,
                         'undoing across the boundary should land on the last round-2 pick')

            // Re-crossing the boundary consumes the pending OFF: round 3 now snakes normally,
            // opening with Team 1 (it would open with Team 2 if reversal still applied).
            await lockInTopDraftPick(app)
            assert.match(await pickLabel(), /Select Pick 3 for Team 1/,
                         're-crossing the boundary should apply the pending reversal-off setting')
            expectCleanSession(app, 'pending reversal consumed on re-crossing')
        })
    } finally {
        await app.close()
    }
})
