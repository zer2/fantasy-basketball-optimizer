// One-off visual check for the headshot experiment (temporary; not part of the pipeline):
// capture the candidate table's top rows (row avatars) and one expanded detail panel
// (corner headshot) to the session scratchpad.
import { chromium } from 'playwright'
import { mintSessionCookie } from './browser_helpers.mjs'

const REPO = 'C:/Users/zacha/Projects/FBBO/fantasy-basketball-optimizer'
const OUT  = 'C:/Users/zacha/AppData/Local/Temp/claude/c--Users-zacha-Projects-FBBO-fantasy-basketball-optimizer/3589b6b2-a765-4b7d-a32b-559c673bb2f5/scratchpad'
const APP  = 'http://localhost:8000'

const browser = await chromium.launch()
const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
    colorScheme: 'light',
})
await context.addCookies([{ ...mintSessionCookie(REPO), url: APP }])
const page = await context.newPage()

page.on('requestfailed', request => {
    if (request.url().includes('cdn.nba.com') || request.url().includes('nba-ids')) {
        console.log('FAILED:', request.url(), request.failure()?.errorText)
    }
})
page.on('console', message => {
    if (message.type() === 'error') console.log('CONSOLE ERROR:', message.text())
})

await page.goto(APP)
await page.waitForFunction(() => {
    const table = document.getElementById('hscoretable')
    return table && table.querySelectorAll('.playerheaderdiv').length >= 8
}, { timeout: 120000 })
await page.waitForTimeout(1500)   // let avatar images arrive from the CDN

const avatarDiagnostics = await page.evaluate(() => {
    const avatars = Array.from(document.querySelectorAll('.player-avatar'))
    return {
        avatarCount: avatars.length,
        firstSrc: avatars[0]?.src ?? null,
        firstLoaded: avatars[0] ? avatars[0].naturalWidth > 0 : null,
    }
})
console.log('avatar diagnostics:', JSON.stringify(avatarDiagnostics))

const tableBox = await page.locator('#hscoretable').boundingBox()
await page.screenshot({
    path: `${OUT}/headshots_table.png`,
    clip: { x: tableBox.x, y: tableBox.y, width: tableBox.width, height: 520 },
})

// Expand the top candidate and capture the roster-assignments grid with its headshot stack.
await page.locator('#hscoretable .playerheaderdiv').first().click()
await page.waitForTimeout(1200)
await page.locator('[data-testid="roster-assignments-table"]').first()
    .screenshot({ path: `${OUT}/headshots_roster.png` })

await browser.close()
console.log('captured headshots_table.png + headshots_roster.png')
