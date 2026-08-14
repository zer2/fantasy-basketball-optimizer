// One-off (temporary): verify the dropdown headshot options, the board collapse, and the
// smaller board headshots after a short autodraft burst.
import { chromium } from 'playwright'
import { mintSessionCookie, waitEval, openSelectDropdown } from './browser_helpers.mjs'

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

// 1. Open the draft pick dropdown (retry-on-rebuild helper — an evaluate settling replaces
// the control and closes a freshly opened dropdown) and capture its headshot options.
await waitEval(page)
await page.waitForTimeout(2500)   // let the deep-bench batches finish so no rebuild closes the dropdown
const wrapperBox = await page.locator('[data-testid="draft-pick-select-wrapper"]').first().boundingBox()

// 0. Closed trigger: the rich value display (headshot + name + positions).
await page.waitForTimeout(400)
await page.screenshot({
    path: `${OUT}/design_trigger_closed.png`,
    clip: { x: wrapperBox.x - 140, y: wrapperBox.y - 6, width: wrapperBox.width + 150, height: wrapperBox.height + 12 },
})
await openSelectDropdown(page, page.locator('[data-testid="draft-pick-select-wrapper"]').first())
await page.waitForTimeout(250)   // let visible option headshots paint (proxy is warm)
await page.screenshot({
    path: `${OUT}/design_dropdown.png`,
    clip: { x: wrapperBox.x, y: wrapperBox.y, width: wrapperBox.width, height: 420 },
})
await page.keyboard.press('Escape')

// 2. Two quick autodraft rounds to fill some cells, then capture the board (small headshots).
async function setSeatAutodraft(index, on) {
    const trigger = page.locator('.entry-table thead .method-dd-trigger').nth(index)
    const isOn = (await trigger.getAttribute('aria-pressed')) === 'true'
    if (isOn !== on) await trigger.click()
}
const seatCount = await page.locator('.entry-table thead .method-dd-trigger').count()
for (let index = 1; index < seatCount; index++) await setSeatAutodraft(index, true)
await setSeatAutodraft(0, true)
await page.waitForFunction(
    () => getComputedStyle(document.getElementById('seat-selector-container')).visibility !== 'hidden',
    { timeout: 300000 },
)
await waitEval(page)
await page.waitForTimeout(1000)
await page.locator('.entry-table-scroll').first().screenshot({ path: `${OUT}/design_board_open.png` })

// 3. Collapse the board via the Round-corner toggle and capture the tucked-away state.
await page.locator('.board-toggle-header').first().click()
await page.waitForTimeout(300)
await page.screenshot({ path: `${OUT}/design_board_collapsed.png` })

await browser.close()
console.log('captured design_dropdown.png, design_board_open.png, design_board_collapsed.png')
