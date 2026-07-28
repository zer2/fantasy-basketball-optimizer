// scripts/e2e/formats_and_categories.test.mjs
// Scoring formats and category selection: the candidate table must re-render across
// EC / MC / Rotisserie switches, category columns must track the selection while staying
// in CANONICAL order (the sport config's order, not the order the user added them), and
// the page must keep working with every available category selected at once.
// Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect,
} from './helpers.mjs'

test('scoring formats and categories', async t => {
    const app = await launchAppPage()
    const { page } = app

    const candidateRowCount = () => page.locator('#hscoretable .playerheaderdiv').count()

    // The Format & Categories sidebar section's multiselect (season trading has its own
    // .ms-containers in the main area, so scope to the section holding the format select).
    const categoryPicker = () =>
        page.locator('details.sidebar-section:has(#fc-scoring-format) .ms-container').first()

    /** The candidate table's per-category column headers, in display order. */
    async function readCategoryHeaders() {
        const headerTexts = await page.locator('#hscoretable thead th').allTextContents()
        const scoreColumnLabels = new Set(['Player', 'H-Score', 'Diff.', 'Your $', 'Gnrc. $', 'Orig. $'])
        return headerTexts.filter(text => !scoreColumnLabels.has(text))
    }

    async function removeCategory(name) {
        await categoryPicker().locator('.ms-chip', { hasText: name }).first()
            .locator('.ms-chip-remove').click()
        await waitAppSettled(app)
    }

    /** Adds the named category — or, with no argument, the first available option.
     *  Returns false when every category is already selected. */
    async function addCategory(name) {
        await categoryPicker().locator('.ms-input').click()
        const options = categoryPicker().locator('.ms-dropdown .ms-option')
        try {
            await options.first().waitFor({ state: 'visible', timeout: 2000 })
        } catch {
            return false   // dropdown shows "All categories selected"
        }
        const option = name ? options.filter({ hasText: name }).first() : options.first()
        await option.click()
        await waitAppSettled(app)
        return true
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('the candidate table re-renders across all three scoring formats', async () => {
            await setSelect(page, 'fc-scoring-format', 'Most Categories')
            await waitAppSettled(app)
            assert.ok(await candidateRowCount() > 0, 'Most Categories should render candidates')
            expectCleanSession(app, 'switch to Most Categories')

            await setSelect(page, 'fc-scoring-format', 'Rotisserie')
            await waitAppSettled(app)
            assert.ok(await candidateRowCount() > 0, 'Rotisserie should render candidates')
            assert.ok(await page.locator('#hscoretable td.categoricalRotoHscore').count() > 0,
                      'Rotisserie should render its converted category values')
            expectCleanSession(app, 'switch to Rotisserie')

            await setSelect(page, 'fc-scoring-format', 'Each Category')
            await waitAppSettled(app)
            assert.ok(await candidateRowCount() > 0, 'Each Category should render candidates')
            assert.ok(await page.locator('#hscoretable td.categoricalhscore').count() > 0,
                      'Each Category should render win-rate cells')
            expectCleanSession(app, 'switch back to Each Category')
        })

        await t.test('category columns track the selection and stay in canonical order', async () => {
            const originalHeaders = await readCategoryHeaders()
            assert.ok(originalHeaders.length > 0, 'the table should have category columns')

            const removedCategory = originalHeaders[0]   // first canonical category
            await removeCategory(removedCategory)
            const headersAfterRemoval = await readCategoryHeaders()
            assert.deepEqual(headersAfterRemoval, originalHeaders.slice(1),
                             'removing a category should drop exactly its column')
            assert.ok(await candidateRowCount() > 0, 'the table should re-render without the category')
            expectCleanSession(app, 'category removed')

            // Re-adding puts it LAST in insertion order — the column must come back FIRST,
            // because display order is canonical (the sport config's order), not insertion order.
            await addCategory(removedCategory)
            assert.deepEqual(await readCategoryHeaders(), originalHeaders,
                             're-adding a category must restore the canonical column order')
            expectCleanSession(app, 'category re-added')
        })

        await t.test('the page keeps working with every available category selected', async () => {
            let addedCount = 0
            while (await addCategory()) addedCount += 1
            assert.ok(addedCount > 0, 'the sport config should offer more than the default categories')

            const extremeHeaders = await readCategoryHeaders()
            const chipCount = await categoryPicker().locator('.ms-chip').count()
            assert.equal(extremeHeaders.length, chipCount,
                         'the table should show one column per selected category')

            // The default categories must appear as an in-order subsequence of the extreme
            // set — canonical ordering at work, not append-at-the-end.
            const defaultHeaders = ['Field Goal %', 'Free Throw %', 'Threes', 'Points', 'Rebounds']
            let searchFrom = 0
            for (const header of defaultHeaders) {
                const index = extremeHeaders.indexOf(header, searchFrom)
                assert.ok(index >= 0, `${header} should keep its canonical position among the extreme set`)
                searchFrom = index + 1
            }

            assert.ok(await candidateRowCount() > 0, 'candidates should render with every category selected')
            expectCleanSession(app, 'every category selected')
        })
    } finally {
        await app.close()
    }
})
