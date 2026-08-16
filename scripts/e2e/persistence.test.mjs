// scripts/e2e/persistence.test.mjs
// Preference persistence: mode, drafter count, and the category selection must survive
// a page reload (they are stored as localStorage preferences and restored on load).

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect, setLeagueDrafterCount,
} from './helpers.mjs'

test('preferences survive a reload', async t => {
    const app = await launchAppPage()
    const { page } = app

    const categoryPicker = () =>
        page.locator('details.sidebar-section:has(#fc-scoring-format) .ms-container').first()

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')

        // Change a spread of preferences: league size, a category removal, then the mode.
        await setLeagueDrafterCount(app, 8)
        await categoryPicker().evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        await categoryPicker().locator('.ms-chip', { hasText: 'Turnovers' }).first()
            .locator('.ms-chip-remove').click()
        await waitAppSettled(app)
        await setSelect(page, 'ls-mode', 'Auction Mode')
        await waitAppSettled(app)
        expectCleanSession(app, 'preferences changed')

        await t.test('mode, drafter count, and categories are restored after reload', async () => {
            await loadApp(app)   // full page reload; a fresh backend session is created from prefs

            assert.equal(await page.locator('#ls-mode').inputValue(), 'Auction Mode',
                         'the mode should be restored')
            assert.equal(await page.locator('#ls-n-drafters').inputValue(), '8',
                         'the drafter count should be restored')
            assert.equal(await categoryPicker().locator('.ms-chip', { hasText: 'Turnovers' }).count(), 0,
                         'the removed category should stay removed')
            assert.equal(await categoryPicker().locator('.ms-chip').count(), 8,
                         'the remaining categories should all be restored')

            // The restored state must actually drive the page: an auction board with one
            // budget cell per restored drafter.
            const remainingRow = page.locator('.entry-table tr', { hasText: 'Remaining' }).first()
            await remainingRow.waitFor({ timeout: 30000 })
            assert.equal(await remainingRow.locator('td', { hasText: '$' }).count(), 8,
                         'the auction board should be built from the restored preferences')
            expectCleanSession(app, 'reload')
        })
    } finally {
        await app.close()
    }
})
