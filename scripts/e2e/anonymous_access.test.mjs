// scripts/e2e/anonymous_access.test.mjs
// Signing in is optional. A visitor with no account gets the whole app — the draft board
// evaluates, the sidebar works — and is asked to sign in only for the one thing that cannot
// work without an account: connecting a live platform, whose credentials are stored per account.
// The wall this replaced returned early from main.ts before any of the app was built, so the
// regression these guard against is total (a blank page), not cosmetic.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, expectCleanSession, waitAppSettled, setSelect,
} from './helpers.mjs'

test('the app is usable without signing in', async t => {
    const app = await launchAppPage({ signedIn: false })
    const { page } = app

    try {
        await loadApp(app)

        await t.test('no login wall — the candidate table evaluates anonymously', async () => {
            assert.equal(await page.locator('#login-overlay').count(), 0,
                         'a signed-out visitor must not be met by a login screen')
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'the board should evaluate with no account at all')
            expectCleanSession(app, 'anonymous first load')
        })

        await t.test('the sidebar offers sign-in rather than hiding it', async () => {
            assert.equal(await page.locator('.account-name').textContent(), 'Not signed in')
            const signInLink = page.locator('.sidebar-account .signin-link')
            assert.equal(await signInLink.count(), 1, 'signing in must stay one click away')
            assert.equal(await signInLink.getAttribute('href'), '/auth/login')
            assert.equal(await page.locator('.account-logout').count(), 0,
                         'there is nothing to sign out of')
        })

        await t.test('a parameter change still re-evaluates', async () => {
            // Exercises the PATCH + evaluate path, which is where an auth assumption would
            // surface as a 401 rather than as a missing control.
            await setSelect(page, 'ps-data-type', 'Projections')
            await waitAppSettled(app, { timeout: 120000 })
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'switching the data source should re-evaluate anonymously')
            expectCleanSession(app, 'anonymous data-source change')
        })

        await t.test('live platforms ask for sign-in instead of failing', async () => {
            await setSelect(page, 'ls-platform', 'Fantrax')
            const prompt = page.locator('#ls-connect-signin')
            await prompt.waitFor({ timeout: 5000 })
            assert.match(await prompt.textContent(), /needs an account/,
                         'the prompt should say why an account is needed')
            assert.equal(await prompt.locator('.signin-link').count(), 1)
            assert.ok(!(await page.locator('#ls-connect-btn').isVisible()),
                      'Connect must be hidden rather than left to fail at the first request')
            expectCleanSession(app, 'anonymous platform selection')
        })

        await t.test('choosing own data again clears the prompt', async () => {
            await setSelect(page, 'ls-platform', 'Enter your own data')
            assert.ok(!(await page.locator('#ls-connect-signin').isVisible()),
                      'own-data mode needs no account, so nothing should ask for one')
            expectCleanSession(app, 'back to own data')
        })
    } finally {
        await app.close()
    }
})
