// One-off (temporary; delete with the Stage-D cleanup): drive a full 12-seat EC autodraft
// through the real browser and time it wall-clock — the end-to-end number the server-side
// probe (11s of evaluates) can't capture.
import { chromium } from 'playwright'
import { mintSessionCookie, waitEval } from './browser_helpers.mjs'

const REPO = 'C:/Users/zacha/Projects/FBBO/fantasy-basketball-optimizer'
const OUT  = 'C:/Users/zacha/AppData/Local/Temp/claude/c--Users-zacha-Projects-FBBO-fantasy-basketball-optimizer/3589b6b2-a765-4b7d-a32b-559c673bb2f5/scratchpad'
const APP  = 'http://127.0.0.1:8000'

const browser = await chromium.launch()
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, colorScheme: 'light' })
await context.addCookies([{ ...mintSessionCookie(REPO), url: APP }])
const page = await context.newPage()

await page.goto(APP)
await page.waitForFunction(() => {
    const table = document.getElementById('hscoretable')
    return table && table.querySelectorAll('.playerheaderdiv').length >= 8
}, { timeout: 120000 })

// Visual smoke of the Stage-C displays before the draft starts.
const tableBox = await page.locator('#hscoretable').boundingBox()
await page.screenshot({
    path: `${OUT}/stage_c_table.png`,
    clip: { x: tableBox.x, y: tableBox.y, width: tableBox.width, height: 520 },
})

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
    { timeout: 300000 },
)
await waitEval(page)
const elapsedSeconds = (Date.now() - started) / 1000
console.log(`full autodraft wall-clock: ${elapsedSeconds.toFixed(1)}s (${seatCount} seats)`)

await page.screenshot({ path: `${OUT}/stage_c_board.png`, fullPage: false })
await browser.close()
