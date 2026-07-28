// scripts/e2e/error_recovery.test.mjs
// Graceful error handling: inputs that get PAST the frontend's validation must fail
// visibly-but-recoverably, and the app must ride out backend failures without wedging.
// Notable finding baked into these tests: a sum-valid-but-degenerate position structure
// (all thirteen slots at C) is a LEGAL configuration by design — the slot-sum gate
// (covered in input_guards) is the only position validation.
// Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, drainSessionFailures,
    waitAppSettled, APP,
} from './helpers.mjs'

const SLOT_INPUT_IDS = ['sc-pg', 'sc-sg', 'sc-sf', 'sc-pf', 'sc-c', 'sc-g', 'sc-f', 'sc-util']

test('error handling and recovery', async t => {
    const app = await launchAppPage()
    const { page } = app

    const candidateRowCount = () => page.locator('#hscoretable .playerheaderdiv').count()

    async function setNumberInput(id, value) {
        const input = page.locator(`#${id}`)
        await input.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        await input.fill(String(value))
        await input.evaluate(el => el.blur())
    }

    async function applySlotCounts() {
        await page.locator('details.sidebar-section:has(#sc-validation) .section-apply-btn').click()
        await waitAppSettled(app)
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        expectCleanSession(app, 'initial load')

        await t.test('nonsense injured names are ignored without disturbing the pool', async () => {
            const rowsBefore = await candidateRowCount()
            const injuredInput = page.locator('#ps-injured')
            await injuredInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            await injuredInput.fill('Notareal Player (C)\nAnother Fake (PG)')
            await injuredInput.evaluate(el => el.blur())
            await waitAppSettled(app)

            assert.equal(await candidateRowCount(), rowsBefore,
                         'unknown injured names should be ignored, leaving the pool intact')
            expectCleanSession(app, 'nonsense injured names')

            await injuredInput.fill('')
            await injuredInput.evaluate(el => el.blur())
            await waitAppSettled(app)
            expectCleanSession(app, 'injured names cleared')
        })

        await t.test('a degenerate but sum-valid position structure is a legal configuration', async () => {
            const originalSlotCounts = {}
            for (const id of SLOT_INPUT_IDS) {
                originalSlotCounts[id] = await page.locator(`#${id}`).inputValue()
            }

            for (const id of SLOT_INPUT_IDS) await setNumberInput(id, id === 'sc-c' ? 13 : 0)
            assert.equal(await page.locator('#sc-validation').textContent(), '',
                         'an all-C structure passes the sum validation')
            await applySlotCounts()
            assert.ok(await candidateRowCount() > 0, 'the app should evaluate an all-C league')
            expectCleanSession(app, 'all-C position structure')

            for (const [id, value] of Object.entries(originalSlotCounts)) await setNumberInput(id, value)
            await applySlotCounts()
            expectCleanSession(app, 'position structure restored')
        })

        await t.test('an expired backend session is recreated transparently', async () => {
            // Delete the live session out from under the app, then trigger a re-evaluate.
            const sessionUrl = app.sessionRequestLog.findLast(entry => entry.includes('/sessions/'))
            const sessionId = sessionUrl.match(/\/sessions\/([^/]+)/)[1]
            const deleteResponse = await fetch(`${APP}/sessions/${sessionId}`, { method: 'DELETE' })
            assert.equal(deleteResponse.status, 204, 'the probe delete should succeed')

            const upsilonInput = page.locator('#mp-upsilon')
            await upsilonInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            const currentValue = parseFloat(await upsilonInput.inputValue())
            await upsilonInput.fill(String(currentValue + 0.05))
            await upsilonInput.evaluate(el => el.blur())
            await waitAppSettled(app)

            // The 404 on the dead session is expected; everything after it must succeed.
            const { failures } = drainSessionFailures(app)
            assert.ok(failures.some(failure => failure.startsWith('404')),
                      `the dead session should 404 before recovery — saw: ${failures.join(' | ')}`)
            assert.ok(failures.every(failure => failure.startsWith('404')),
                      `only 404s are acceptable during recovery — saw: ${failures.join(' | ')}`)

            const latestSessionId = app.sessionRequestLog.findLast(entry => entry.includes('/sessions/'))
                .match(/\/sessions\/([^/]+)/)[1]
            assert.notEqual(latestSessionId, sessionId, 'a fresh session should replace the deleted one')
            assert.ok(await candidateRowCount() > 0, 'candidates should render from the fresh session')
            expectCleanSession(app, 'post-recovery state')
        })

        await t.test('a backend-rejected league setting fails visibly and the app recovers', async () => {
            // n_drafters=1 passes the frontend (the input's min is not enforced on change)
            // but the backend cannot build a 1-team league — today that surfaces as a 500.
            await setNumberInput('ls-n-drafters', 1)
            await waitAppSettled(app)

            const { failures } = drainSessionFailures(app)
            assert.ok(failures.length > 0, 'the impossible league should be rejected by the backend')
            assert.ok(await page.locator('#hscoretable .playerheaderdiv, #hscoretable').first().isVisible(),
                      'the app should not wedge after the rejection')

            await setNumberInput('ls-n-drafters', 12)
            await waitAppSettled(app)
            assert.ok(await candidateRowCount() > 0, 'restoring a valid league should recover fully')
            expectCleanSession(app, 'recovered from rejected league')
        })
    } finally {
        await app.close()
    }
})
