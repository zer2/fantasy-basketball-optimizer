// scripts/e2e/regressions.test.mjs
// Regression tests for the auction-league session bugs: every evaluate on an auction
// session must carry remaining_cash, and a session must stop being an auction league
// when the user leaves Auction Mode. Each subtest drains the failure monitors, so a
// failing backend call is attributed to the step that caused it.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect, readDropdownOptionLabels, pickControlButton, lockInAuctionPick,
    countBoardCellsWithPlayer,
} from './helpers.mjs'

test('auction-league session regressions', async t => {
    const app = await launchAppPage()
    const { page } = app
    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('auction: parameter change with picks on the board re-evaluates cleanly', async () => {
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app)
            await lockInAuctionPick(app, 'Nikola Jokic', 'Team 1', 70)
            expectCleanSession(app, 'enter auction + lock pick')

            // Changing a model parameter clears the cached base players, forcing the
            // empty-board base evaluation to re-run — the original bug's trigger.
            const upsilonInput = page.locator('#mp-upsilon')
            await upsilonInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            const currentValue = parseFloat(await upsilonInput.inputValue())
            await upsilonInput.fill(String(currentValue + 0.05))
            await upsilonInput.dispatchEvent('change')
            await waitAppSettled(app)

            const candidateRows = await page.locator('#hscoretable .playerheaderdiv').count()
            assert.ok(candidateRows > 0, 'candidate table should re-render after the parameter change')
            expectCleanSession(app, 'parameter change in auction mode')
        })

        await t.test('auction: undo restores the player dropdown and the board', async () => {
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)

            const optionLabels = await readDropdownOptionLabels(page, 'auction-pick-player-wrapper')
            assert.ok(optionLabels.some(label => label.includes('Nikola Jokic')),
                      'undone player should reappear in the Player dropdown')
            assert.equal(await countBoardCellsWithPlayer(page, 'Nikola Jokic'), 0,
                         'undone player should be removed from the board')
            expectCleanSession(app, 'undo in auction mode')
        })

        await t.test('auction -> draft: draft evaluates keep working after leaving auction mode', async () => {
            // Leaving Auction Mode must patch is_auction off (the session's league type) — a
            // draft evaluate never sends remaining_cash, so an auction-league session 400s.
            await setSelect(page, 'ls-mode', 'Draft Mode')
            await waitAppSettled(app)

            const candidateRows = await page.locator('#hscoretable .playerheaderdiv').count()
            assert.ok(candidateRows > 0, 'candidate table should render after switching back to Draft Mode')
            expectCleanSession(app, 'switch auction -> draft')
        })

        await t.test('draft: changing the drafter count keeps evaluates working', async () => {
            // League-settings patches carry cash_per_team, which must be inert outside
            // Auction Mode — the session's mode, not cash presence, is its league type.
            const draftersInput = page.locator('#ls-n-drafters')
            await draftersInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            await draftersInput.fill('8')
            await draftersInput.dispatchEvent('change')
            await waitAppSettled(app)

            const teamHeaderCount = await page.locator('.entry-table thead .team-header-cell').count()
            assert.equal(teamHeaderCount, 8, 'draft board should resize to the new drafter count')
            expectCleanSession(app, 'drafter count change in draft mode')

            await draftersInput.fill('12')
            await draftersInput.dispatchEvent('change')
            await waitAppSettled(app)
            expectCleanSession(app, 'drafter count restored')
        })

        await t.test('auction -> season: roster inspection H-score renders', async () => {
            // Season evaluates (waiver, roster inspection) also omit remaining_cash, so
            // they break the same way when the session is still an auction league.
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app)
            await setSelect(page, 'ls-mode', 'Season Mode')
            await waitAppSettled(app)
            await page.locator('.season-tab-btn[data-tab="rosters"]').click()

            // Toggle the inspected team so the selection fires a fresh change-driven eval.
            await setSelect(page, 'sr-team-select', 'Team 2')
            await setSelect(page, 'sr-team-select', 'Team 1')

            await page.locator('[data-testid="roster-inspection-hscore"]').waitFor({ timeout: 30000 })
            expectCleanSession(app, 'auction -> season roster inspection')
        })
    } finally {
        await app.close()
    }
})
