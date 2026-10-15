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

    /** Turns the Display section's "Show example team" toggle on or off, as a user flipping it would. */
    async function setExampleTeamShown(isShown) {
        await page.evaluate(shown => {
            const toggle = document.getElementById('show-example-team')
            if (toggle.checked === shown) return
            toggle.checked = shown
            toggle.dispatchEvent(new Event('change'))
        }, isShown)
    }

    /** With a candidate panel open: every empty roster slot holds a hypothetical pick whose label matches
     *  labelPattern. When the panel opened on a team not found yet, its empty slots show spinners first, and the panel
     *  is already at its full height while they do. */
    async function expectHypotheticalTeamFills(labelPattern, { expectSpinners }) {
        const grid = page.locator('[data-testid="roster-assignments-table"]')
        await grid.waitFor({ state: 'visible', timeout: 15000 })
        const panel = page.locator('#hscoretable td.panel-cell').first()
        const pendingSlots = await grid.locator('td.rosterpending').count()
        const heightWhilePending = (await panel.boundingBox()).height
        if (expectSpinners) assert.ok(pendingSlots > 0, 'the empty roster slots should show spinners at first')
        await page.waitForFunction(() =>
            document.querySelectorAll('[data-testid="roster-assignments-table"] td.rosterhypothetical').length > 0,
            null, { timeout: 30000 })
        assert.equal(await grid.locator('td.rosterpending').count(), 0, 'no spinner should remain')
        assert.equal(await grid.locator('td.rosterabsent').count(), 0, 'every empty slot should be filled')
        if (expectSpinners) {
            assert.equal(await grid.locator('td.rosterhypothetical').count(), pendingSlots,
                         'each spinner slot should hold one hypothetical pick')
            const heightWhenFilled = (await panel.boundingBox()).height
            assert.ok(Math.abs(heightWhenFilled - heightWhilePending) <= 1,
                      `the panel should not change height as the team arrives (${heightWhilePending} -> ${heightWhenFilled})`)
        }
        const labels = await grid.locator('.roster-hypothetical-detail').allTextContents()
        for (const label of labels) assert.match(label, labelPattern)
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
            // Show example team is off by default: the empty slots stay empty.
            const grid = page.locator('[data-testid="roster-assignments-table"]')
            assert.equal(await grid.locator('td.rosterpending, td.rosterhypothetical').count(), 0,
                         'no example team should be shown with the toggle off')
            assert.ok(await grid.locator('td.rosterabsent').count() > 0, 'the empty slots should stay empty')
            expectCleanSession(app, 'draft candidate expanded')

            await page.locator('#hscoretable .playerheaderdiv').first().click()   // collapse again
            await page.locator('[data-testid="gscore-expectations-table"]')
                .waitFor({ state: 'hidden', timeout: 5000 })
        })

        await t.test('opening a draft candidate fills its empty roster slots with a hypothetical team', async () => {
            await setExampleTeamShown(true)
            // The second candidate: one not asked about yet, so its panel opens on spinners.
            await page.locator('#hscoretable .playerheaderdiv').nth(1).click()
            await expectHypotheticalTeamFills(/^Rd \d+$/, { expectSpinners: true })
            expectCleanSession(app, 'draft hypothetical team')
            await page.locator('#hscoretable .playerheaderdiv').nth(1).click()   // collapse again
            await page.locator('[data-testid="roster-assignments-table"]').waitFor({ state: 'hidden', timeout: 5000 })
        })

        await t.test('the hypothetical team works on the default rankings shown before a league connects', async () => {
            await setSelect(page, 'ls-platform', 'Retrieve from Yahoo')
            await waitAppSettled(app)
            await page.locator('#hscoretable .playerheaderdiv').first().click()
            await expectHypotheticalTeamFills(/^Rd \d+$/, { expectSpinners: true })
            expectCleanSession(app, 'unconnected-platform hypothetical team')
            await page.locator('#hscoretable .playerheaderdiv').first().click()   // collapse again
            await page.locator('[data-testid="roster-assignments-table"]').waitFor({ state: 'hidden', timeout: 5000 })
            await setSelect(page, 'ls-platform', 'Enter your own data')
            await waitAppSettled(app)
        })

        await t.test('expanding an auction candidate adds the auction values table', async () => {
            await setSelect(page, 'ls-mode', 'Auction Mode')
            await waitAppSettled(app)
            await page.locator('#hscoretable .playerheaderdiv').first().click()
            await expectStyledTable('auction-values-table', { expectStyledCells: false })
            await expectStyledTable('gscore-expectations-table')
            expectCleanSession(app, 'auction candidate expanded')
        })

        await t.test('the auction hypothetical team labels its picks with predicted prices', async () => {
            // The panel opened in the step above, so its team may already be in.
            await expectHypotheticalTeamFills(/^\$\d+$/, { expectSpinners: false })
            expectCleanSession(app, 'auction hypothetical team')
        })
    } finally {
        await app.close()
    }
})
