// scripts/e2e/helpers.mjs
// Shared setup for the frontend e2e tests.
//
// Preconditions (same as the screenshots pipeline): the app must be running at
// http://localhost:8000 (or APP_URL), with backend data access working — evaluates
// hit the real backend. Tests pin historical seasons so assertions don't drift as
// live projections update.
//
// Run the suite with:  npm run test:e2e
// (node --test --test-concurrency=1 — the files share one backend, so they run serially)

import { chromium } from 'playwright'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import assert from 'node:assert/strict'
import {
    mintSessionCookie, setSelect, waitEval, lockInDraftPick, openSelectDropdown, chooseDropdownOption,
} from '../browser_helpers.mjs'

export { setSelect, waitEval, lockInDraftPick, openSelectDropdown, chooseDropdownOption }

export const APP = process.env.APP_URL ?? 'http://localhost:8000'
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..')

/**
 * Launches a headless browser page logged into the app, with monitors attached:
 * every failed backend call (status >= 400 on a /sessions route) and every console
 * error is recorded. Tests drain these via expectCleanSession after each step, so a
 * failure is attributed to the step that caused it.
 */
export async function launchAppPage() {
    try {
        await fetch(APP, { method: 'HEAD' })
    } catch {
        throw new Error(`App is not running at ${APP} — start it first (or set APP_URL)`)
    }

    const browser = await chromium.launch()
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
    // The season rosters grid pastes via navigator.clipboard — grant it up front.
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: APP })
    const cookie = mintSessionCookie(repoRoot)
    await context.addCookies([{ name: cookie.name, value: cookie.value, url: APP }])
    const page = await context.newPage()
    page.setDefaultTimeout(20000)

    const failedApiCalls = []
    page.on('response', response => {
        if (response.status() >= 400 && response.url().includes('/sessions')) {
            const failure = { status: response.status(), url: response.url(), body: '' }
            failedApiCalls.push(failure)
            // Body arrives async; it is filled in by the time a test reads the failure list.
            response.text().then(text => { failure.body = text.slice(0, 300) }).catch(() => {})
        }
    })

    // In-flight backend calls, for waitAppSettled: the eval indicator alone is not a
    // settledness signal — the app deliberately drops the spinner after the first
    // candidate batch while further batches (and a final re-render) are still landing.
    // sessionRequestLog records every backend call, so tests can assert an action did
    // NOT talk to the backend (e.g. an invalid input blocking its patch).
    const pendingSessionRequests = new Set()
    const sessionRequestLog = []
    page.on('request', request => {
        if (request.url().includes('/sessions')) {
            pendingSessionRequests.add(request)
            sessionRequestLog.push(`${request.method()} ${request.url()}`)
        }
    })
    page.on('requestfinished', request => pendingSessionRequests.delete(request))
    page.on('requestfailed', request => pendingSessionRequests.delete(request))

    const consoleErrors = []
    page.on('console', message => {
        // Generic resource-load lines duplicate what the response monitor already
        // captures with more detail — keep only app-emitted console errors.
        if (message.type() === 'error' && !message.text().startsWith('Failed to load resource')) {
            consoleErrors.push(message.text())
        }
    })
    page.on('pageerror', error => consoleErrors.push(String(error)))

    const app = {
        browser,
        page,
        failedApiCalls,
        consoleErrors,
        pendingSessionRequests,
        sessionRequestLog,
        async close() { await browser.close() },
    }
    return app
}

/**
 * Waits until the app is genuinely quiescent: eval indicator idle AND no in-flight
 * backend call, confirmed twice in a row. The double-check closes the promise-chain
 * gaps between a response landing and the next request being sent, and catches the
 * app's early-idle behaviour (spinner drops after the first candidate batch while the
 * remaining batches are still streaming in). Interacting before full quiescence is
 * unreliable: a late re-render can replace controls or reset in-progress board input.
 */
export async function waitAppSettled(app, { timeout = 60000 } = {}) {
    const { page } = app
    const deadline = Date.now() + timeout
    let consecutiveQuietChecks = 0
    while (consecutiveQuietChecks < 2) {
        if (Date.now() > deadline) throw new Error(`app did not settle within ${timeout}ms`)
        const indicatorIdle = await page.evaluate(() => {
            const el = document.querySelector('#eval-indicator')
            return !el || el.dataset.state === 'idle' || el.dataset.state === 'unconnected'
        })
        if (indicatorIdle && app.pendingSessionRequests.size === 0) {
            consecutiveQuietChecks += 1
            if (consecutiveQuietChecks < 2) await page.waitForTimeout(250)
        } else {
            consecutiveQuietChecks = 0
            await page.waitForTimeout(100)
        }
    }
}

/** Loads the app and waits for the first evaluate to render the candidate table. */
export async function loadApp(app) {
    await app.page.goto(APP, { waitUntil: 'domcontentloaded' })
    await app.page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })
    await waitAppSettled(app, { timeout: 120000 })
}

/** Pins the data source to a fixed historical season so assertions stay stable over time. */
export async function selectHistoricalSeason(app, season) {
    await setSelect(app.page, 'ps-data-type', 'Historical')
    await waitAppSettled(app)
    await setSelect(app.page, 'ps-season', season)
    await waitAppSettled(app)
}

/**
 * Asserts that no backend call failed and no console error fired since the last check,
 * then drains both monitors — so each test step owns exactly the failures it caused.
 */
export function expectCleanSession(app, stepLabel) {
    const failures = app.failedApiCalls.map(f => `${f.status} ${f.url} — ${f.body}`)
    const errors = [...app.consoleErrors]
    app.failedApiCalls.length = 0
    app.consoleErrors.length = 0
    assert.deepEqual(failures, [], `${stepLabel}: backend calls failed:\n  ${failures.join('\n  ')}`)
    assert.deepEqual(errors, [], `${stepLabel}: console errors:\n  ${errors.join('\n  ')}`)
}

/** Opens a custom select's dropdown, reads all option labels, and closes it again. */
export async function readDropdownOptionLabels(page, wrapperTestId) {
    const wrap = page.locator(`[data-testid="${wrapperTestId}"]`).first()
    await openSelectDropdown(page, wrap)
    const labels = await wrap.locator('.cs-dropdown .cs-option').allTextContents()
    await page.keyboard.press('Escape')
    await page.waitForTimeout(150)
    return labels
}

/** The draft/auction pick-control button with the given label ("Lock in selection", etc.). */
export function pickControlButton(page, label) {
    return page.locator('.pick-control-row .pick-btn', { hasText: label })
}

/** Sets the league's drafter count via the sidebar input and waits for the rebuild. A small
 *  count (e.g. 2) makes multi-round flows affordable — evaluates shrink with the league.
 *  Blur (not a synthetic change dispatch) fires the native change exactly once and clears
 *  the input's dirty flag, so a later focus shift can't fire a surprise second change. */
export async function setLeagueDrafterCount(app, count) {
    const { page } = app
    const draftersInput = page.locator('#ls-n-drafters')
    await draftersInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
    await draftersInput.fill(String(count))
    await draftersInput.evaluate(el => el.blur())
    await waitAppSettled(app)
}

/** Locks whichever player tops the pick dropdown — for driving a draft forward when the
 *  specific player doesn't matter (e.g. walking the pick order through several rounds). */
export async function lockInTopDraftPick(app) {
    const { page } = app
    const wrap = page.locator('[data-testid="draft-pick-select-wrapper"]').first()
    await chooseDropdownOption(page, wrap, wrapper => wrapper.locator('.cs-dropdown .cs-option').first())
    await pickControlButton(page, 'Lock in selection').click()
    await waitAppSettled(app)
}

/** Locks an auction pick: player + team + cost, then "Lock in selection". */
export async function lockInAuctionPick(app, playerName, teamLabel, cost) {
    const { page } = app
    await setSelect(page, 'auction-pick-player', playerName)
    await setSelect(page, 'auction-pick-team', teamLabel)
    await page.fill('.auction-cost-input', String(cost))
    await pickControlButton(page, 'Lock in selection').click()
    await waitAppSettled(app)
}
