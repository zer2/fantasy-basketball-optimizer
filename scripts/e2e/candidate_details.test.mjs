// scripts/e2e/candidate_details.test.mjs
// Candidate table content: rows must be ordered by descending H-score, and expanding a
// candidate must render its detail tables — visibly, with the stat-styler backgrounds
// applied (the "visual display works" check: styled cells, not just DOM presence).
// Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect,
} from './helpers.mjs'

test('candidate table content and detail views', async t => {
    const app = await launchAppPage()
    const { page } = app

    /** Asserts a detail table is visible with rows and real layout width; tables whose cells
     *  carry stat-styler colouring must also have styled cells (the visual-display check —
     *  roster assignments and auction values are plain-value tables, so they skip that part). */
    async function expectStyledTable(testId, { expectStyledCells = true } = {}) {
        const table = page.locator(`[data-testid="${testId}"]`)
        await table.waitFor({ state: 'visible', timeout: 15000 })
        const shape = await table.evaluate(el => ({
            rows: el.querySelectorAll('tbody tr').length,
            styledCells: el.querySelectorAll('td[style*="background"]').length,
            width: el.getBoundingClientRect().width,
        }))
        assert.ok(shape.rows > 0, `${testId} should have rows`)
        if (expectStyledCells) assert.ok(shape.styledCells > 0, `${testId} should have stat-styled cells`)
        assert.ok(shape.width > 0, `${testId} should be laid out with real width`)
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('candidates are ordered by descending H-score', async () => {
            const hScores = await page.evaluate(() =>
                [...document.querySelectorAll('#hscoretable td.overallhscore')]
                    .slice(0, 25).map(cell => parseFloat(cell.textContent)))
            assert.ok(hScores.length >= 10, 'the table should render enough rows to check ordering')
            for (let index = 1; index < hScores.length; index++) {
                assert.ok(hScores[index] <= hScores[index - 1],
                          `H-scores must be non-increasing (row ${index}: ${hScores[index]} after ${hScores[index - 1]})`)
            }
        })

        await t.test('expanding a draft candidate renders the four detail tables, styled', async () => {
            await page.locator('#hscoretable .playerheaderdiv').first().click()
            await expectStyledTable('gscore-expectations-table')
            await expectStyledTable('future-pick-strategy-table')
            await expectStyledTable('flex-allocations-table')
            await expectStyledTable('roster-assignments-table', { expectStyledCells: false })
            expectCleanSession(app, 'draft candidate expanded')

            await page.locator('#hscoretable .playerheaderdiv').first().click()   // collapse again
            await page.locator('[data-testid="gscore-expectations-table"]')
                .waitFor({ state: 'hidden', timeout: 5000 })
        })

        await t.test('expanding an auction candidate adds the auction values table', async () => {
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app)
            await page.locator('#hscoretable .playerheaderdiv').first().click()
            await expectStyledTable('auction-values-table', { expectStyledCells: false })
            await expectStyledTable('gscore-expectations-table')
            expectCleanSession(app, 'auction candidate expanded')
        })
    } finally {
        await app.close()
    }
})
