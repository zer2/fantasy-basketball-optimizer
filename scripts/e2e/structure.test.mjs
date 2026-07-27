// scripts/e2e/structure.test.mjs
// Table structure: entry tables must track the league dimensions (drafters / picks)
// across draft, auction, and season views. Assertions are structural (counts), never
// numeric, so they stay stable as data updates.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect,
} from './helpers.mjs'

test('tables track league dimensions', async t => {
    const app = await launchAppPage()
    const { page } = app

    async function setDrafterCount(count) {
        const draftersInput = page.locator('#ls-n-drafters')
        await draftersInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        await draftersInput.fill(String(count))
        await draftersInput.dispatchEvent('change')
        await waitAppSettled(app)
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('draft board matches the default league dimensions', async () => {
            assert.equal(await page.locator('.entry-table thead .team-header-cell').count(), 12)
            assert.equal(await page.locator('.entry-table tbody tr').count(), 13)
        })

        await t.test('changing the drafter count resizes the draft board', async () => {
            await setDrafterCount(8)
            assert.equal(await page.locator('.entry-table thead .team-header-cell').count(), 8)
            assert.equal(await page.locator('.entry-table tbody tr').count(), 13,
                         'row count should be unchanged by a drafter-count change')
            expectCleanSession(app, 'drafter count 12 -> 8')
        })

        await t.test('auction board tracks the drafter count, including the budget row', async () => {
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app)

            const remainingRow = page.locator('.entry-table tr', { hasText: 'Remaining' }).first()
            const budgetCellCount = await remainingRow.locator('td', { hasText: '$' }).count()
            assert.equal(budgetCellCount, 8, 'one Remaining-budget cell per drafter')
            expectCleanSession(app, 'auction board at 8 drafters')
        })

        await t.test('season rosters grid tracks the league dimensions', async () => {
            await setSelect(page, 'ls-mode', 'Season Mode')
            await waitAppSettled(app)
            await page.locator('.season-tab-btn[data-tab="rosters"]').click()
            const grid = page.locator('#rosters-left .entry-table').first()
            await grid.waitFor()

            assert.equal(await grid.locator('thead th').count(), 8 + 1, 'one header per team plus the pick column')
            assert.equal(await grid.locator('tbody tr').count(), 13, 'one row per pick')
            expectCleanSession(app, 'season rosters grid at 8 drafters')
        })
    } finally {
        await app.close()
    }
})
