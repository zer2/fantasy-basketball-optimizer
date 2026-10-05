// scripts/e2e/input_guards.test.mjs
// Sidebar inputs with real logic behind them: marking a player injured removes them
// from the pool (and back), and invalid slot counts block the backend patch entirely
// while showing the validation message. Uses the 2025-26 historical season.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    readDropdownOptionLabels,
} from './helpers.mjs'

test('sidebar input guards', async t => {
    const app = await launchAppPage()
    const { page } = app

    const candidateRowsWith = (text) => page.locator('#hscoretable .playerheaderdiv', { hasText: text }).count()

    // The injured list is a multiselect of the pool's players: searched by typing, picked from its dropdown.
    const injuredSearch = page.locator('#ps-injured .ms-input')
    async function searchInjuredList(text) {
        await injuredSearch.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        await injuredSearch.click()
        await injuredSearch.fill(text)
    }
    async function pickInjuredPlayer(name) {
        await searchInjuredList(name)
        await page.locator('#ps-injured .ms-option', { hasText: name }).first().click()
        await injuredSearch.evaluate(el => el.blur())
        await waitAppSettled(app)
    }
    async function unpickInjuredPlayer(name) {
        await page.locator('#ps-injured .ms-chip', { hasText: name }).locator('.ms-chip-remove').click()
        await waitAppSettled(app)
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('an injured player leaves the candidate pool and returns when cleared', async () => {
            assert.ok(await candidateRowsWith('Nikola Jokic') > 0, 'Jokic should start in the candidate pool')

            await pickInjuredPlayer('Nikola Jokic')
            assert.equal(await candidateRowsWith('Nikola Jokic'), 0,
                         'an injured player should leave the candidate table')
            const pickOptions = await readDropdownOptionLabels(page, 'draft-pick-select-wrapper')
            assert.ok(!pickOptions.some(label => label.includes('Nikola Jokic')),
                      'an injured player should leave the pick dropdown too')
            expectCleanSession(app, 'player marked injured')

            // Still offered after leaving the pool: the list is built from the registry, which keeps injured players.
            assert.equal(await page.locator('#ps-injured .ms-chip', { hasText: 'Nikola Jokic' }).count(), 1)
            await unpickInjuredPlayer('Nikola Jokic')
            assert.ok(await candidateRowsWith('Nikola Jokic') > 0,
                      'clearing the injured list should restore the player')
            expectCleanSession(app, 'injured list cleared')
        })

        await t.test('searching the injured list changes nothing', async () => {
            // The Player Stats section rebuilds the session on any input inside it; typing a search is not a change.
            const requestsBefore = app.sessionRequestLog.length
            await searchInjuredList('Jok')
            await page.waitForTimeout(1500)   // longer than the section's 800ms debounce
            await injuredSearch.fill('')
            await injuredSearch.evaluate(el => el.blur())
            await waitAppSettled(app)
            assert.deepEqual(app.sessionRequestLog.slice(requestsBefore), [], 'a search must not reach the backend')
            expectCleanSession(app, 'injured list searched')
        })

        await t.test('invalid slot counts show the validation message and block the patch', async () => {
            const utilInput = page.locator('#sc-util')
            await utilInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            const originalUtil = await utilInput.inputValue()

            await utilInput.fill(String(parseInt(originalUtil) + 1))   // sum now exceeds picks/drafter
            await utilInput.dispatchEvent('input')
            const validationText = await page.locator('#sc-validation').textContent()
            assert.match(validationText, /exceeds picks per drafter/,
                         'the mismatch should surface in the validation message')

            const requestsBeforeApply = app.sessionRequestLog.length
            await page.locator('details.sidebar-section:has(#sc-validation) .section-apply-btn').click()
            await waitAppSettled(app)
            const requestsDuringApply = app.sessionRequestLog.slice(requestsBeforeApply)
            assert.deepEqual(requestsDuringApply, [],
                             'applying invalid slot counts must not send anything to the backend')
            expectCleanSession(app, 'invalid slot counts blocked')

            await utilInput.fill(originalUtil)
            await utilInput.dispatchEvent('input')
            assert.equal(await page.locator('#sc-validation').textContent(), '',
                         'restoring the count should clear the validation message')
            await page.locator('details.sidebar-section:has(#sc-validation) .section-apply-btn').click()
            await waitAppSettled(app)
            assert.ok(app.sessionRequestLog.length > requestsBeforeApply,
                      'a valid apply should patch the backend')
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'the candidate table should re-render after the valid apply')
            expectCleanSession(app, 'valid slot counts applied')
        })
    } finally {
        await app.close()
    }
})
