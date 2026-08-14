// scripts/benchmark_autodraft.mjs
// Times a full 12-seat autodraft through the real browser and records it in the
// benchmark history — the end-to-end number the server-side evaluate probes can't
// capture (HTTP round-trips, board repaints, pick bookkeeping).
//
// Setup: the app must be running at http://127.0.0.1:8000 (or set APP_URL), and the
// machine should be otherwise quiet — concurrent test suites or builds inflate the
// number and defeat the comparison.
// Run:   node scripts/benchmark_autodraft.mjs

import { appendFileSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'
import { mintSessionCookie, waitEval } from './browser_helpers.mjs'

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const APP  = process.env.APP_URL ?? 'http://127.0.0.1:8000'
const HISTORY_PATH = path.join(REPO, 'testing_files', 'benchmark_history.jsonl')

const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: 1440, height: 900 } })
await context.addCookies([{ ...mintSessionCookie(REPO), url: APP }])
const page = await context.newPage()

await page.goto(APP)
await page.waitForFunction(() => {
    const table = document.getElementById('hscoretable')
    return table && table.querySelectorAll('.playerheaderdiv').length >= 8
}, { timeout: 120000 })

// All seats -> autodraft (seat 0 last: toggling the on-clock seat starts autopilot).
async function setSeatAutodraft(index, on) {
    const trigger = page.locator('.entry-table thead .method-dd-trigger').nth(index)
    const isOn = (await trigger.getAttribute('aria-pressed')) === 'true'
    if (isOn !== on) await trigger.click()
}
const seatCount = await page.locator('.entry-table thead .method-dd-trigger').count()
const started = Date.now()
for (let index = 1; index < seatCount; index++) await setSeatAutodraft(index, true)
await setSeatAutodraft(0, true)

// Autopilot hides the seat selector while running; done when it comes back.
await page.waitForFunction(
    () => getComputedStyle(document.getElementById('seat-selector-container')).visibility === 'hidden',
    { timeout: 15000 },
).catch(() => {})
await page.waitForFunction(
    () => getComputedStyle(document.getElementById('seat-selector-container')).visibility !== 'hidden',
    { timeout: 600000 },
)
await waitEval(page)
const elapsedSeconds = (Date.now() - started) / 1000
await browser.close()

const commit = execFileSync('git', ['rev-parse', '--short', 'HEAD'], { cwd: REPO, encoding: 'utf8' }).trim()
const label = `Full autodraft wall-clock (${seatCount} seats, browser)`
console.log(`[benchmark] ${label}: ${elapsedSeconds.toFixed(1)}s`)
appendFileSync(HISTORY_PATH, JSON.stringify({
    timestamp: new Date().toISOString().replace(/\.\d{3}Z$/, 'Z'),
    commit,
    label,
    seconds: Math.round(elapsedSeconds * 1000) / 1000,
}) + '\n')
