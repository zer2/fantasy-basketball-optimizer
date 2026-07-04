// scripts/screenshots.mjs
// Regenerate the documentation screenshots at a consistent size.
//
// Setup:  npm i -D playwright && npx playwright install chromium
//         App must be running at http://localhost:8000  (or set APP_URL)
// Run:    node scripts/screenshots.mjs                 # all non-skipped shots
//         node scripts/screenshots.mjs hec chi         # only these (runs even if skipped)
//
// Consistency comes from three things: a fixed browser (viewport + 2x scale +
// one theme), a deterministic app state per shot, and ELEMENT-level captures
// (locator.screenshot() crops tightly to the component). Selectors are the
// stable ids/testids surveyed from the frontend (data-testid="..." hooks were
// added for the components that lacked one).
//
// STATUS: first pass. Option labels in the STATES below (mode/format names) and
// a few interaction steps are best-effort and likely need a verification run —
// see the `skip` reasons in SHOTS for the ones known to be problematic.
//
// NOT captured here (paper/Wikipedia figures, not app screenshots):
//   roto_equations, normal, HistEC, crazyformula, assignmentproblem, savor

import { chromium } from 'playwright'
import { execFileSync } from 'node:child_process'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const APP  = process.env.APP_URL ?? 'http://localhost:8000'
// Defaults to docs/img; set SHOT_OUT to a scratch dir when debugging so real images aren't clobbered.
const IMG  = process.env.SHOT_OUT ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'docs', 'img')
const only = new Set(process.argv.slice(2))

const CONTEXT = {
    viewport:          { width: 1440, height: 900 },
    deviceScaleFactor: 2,        // retina-crisp PNGs
    colorScheme:       'light',  // one theme for every screenshot
}
const PAD = 18   // whitespace around each element/table shot so nothing sits tight against the edge

// ─── interaction helpers (from the frontend survey) ─────────────────────────────
// Sidebar sections are <details class="sidebar-section"> — expand by clicking <summary>.
async function expandSection(page, titleRegex) {
    const details = page.locator('details.sidebar-section', {
        has: page.locator('summary', { hasText: titleRegex }),
    }).first()
    if (!(await details.evaluate(d => d.open))) {
        await details.locator('summary').first().click()
    }
}

// Custom selects put the id on a hidden input; the wrapper carries data-testid="<id>-wrapper".
// Click .cs-trigger to open, then the matching .cs-option.
async function setSelect(page, id, optionText) {
    const wrap = page.locator(`[data-testid="${id}-wrapper"]`).first()
    // Open the containing collapsed <details> sidebar section, if any, so the trigger is visible.
    await wrap.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
    await wrap.locator('.cs-trigger').click()
    await page.locator('.cs-dropdown .cs-option', { hasText: optionText }).first().click()
    await page.waitForTimeout(100)
}

// Expand a candidate's detail drop-down (the expectation / strategy tables live inside it).
async function expandCandidate(page, i = 0) {
    await page.locator('#hscoretable .playerheaderdiv').nth(i).click()
    await page.waitForTimeout(250)
}

// Expand a specific candidate by (partial) player name — e.g. to feature Giannis's punt strategy.
async function expandCandidateByName(page, name) {
    await page.locator('#hscoretable .playerheaderdiv', { hasText: name }).first().click()
    await page.waitForTimeout(250)
}

// Draft a player for the current pick via the pick-control select + "Lock in selection".
async function lockInDraftPick(page, playerName) {
    const wrap = page.locator('[data-testid="draft-pick-select-wrapper"]').first()
    await wrap.locator('.cs-trigger').click()
    await wrap.locator('.cs-dropdown .cs-option').filter({ hasText: playerName }).first().click()
    await page.locator('.pick-control-row .pick-btn', { hasText: 'Lock in selection' }).click()
    await waitEval(page)
}

// Wait for an evaluate to finish (#eval-indicator reaches "idle"; it starts at "fetching").
async function waitEval(page) {
    await page.waitForFunction(() => {
        const el = document.querySelector('#eval-indicator')
        return el && el.dataset.state === 'idle'
    }, { timeout: 40000 }).catch(() => {})
    await page.waitForTimeout(200)
}

// ─── memoized app states ────────────────────────────────────────────────────────
// The page is loaded ONCE (STATES.load, before the loop). Every other state is a live UI
// transition — no reloads (a reload re-pulls Snowflake data and loses our mode/format).
let current = null
async function ensure(page, state) {
    if (current === state) return
    await STATES[state](page)
    current = state
}

async function setMode(page, mode) { await setSelect(page, 'ls-mode', mode);          await waitEval(page) }
async function setFmt(page, fmt)   { await setSelect(page, 'fc-scoring-format', fmt); await waitEval(page) }

// Establish the demonstration data source: historical stats for a completed season, so shots stay
// identical over time (live projections drift as the season updates). The H-scoring / draft page uses
// 2024-25 — where Giannis, the docs' punt exemplar, ranks top-6 by H-score; auction/season use 2025-26
// (Season needs it: its default rosters are 2025-26 players). Set after load and on auction/season
// entry; the projections/1984-85 shots switch the data source at the very end.
async function selectHistoricalSeason(page, season) {
    await setSelect(page, 'ps-data-type', 'Historical')
    await waitEval(page)
    await setSelect(page, 'ps-season', season)
    await waitEval(page)
    // setSelect opened the Player Stats <details> to reach the trigger — collapse it so `main` (and
    // the other early shots) show the default, un-expanded sidebar.
    await page.evaluate(() => document.querySelectorAll('details.sidebar-section').forEach(d => {
        if (/Player Stats/i.test(d.querySelector('summary')?.textContent ?? '')) d.open = false
    }))
}

const STATES = {
    async load(page) {
        await page.goto(APP, { waitUntil: 'domcontentloaded' })   // networkidle hangs on the polling SPA
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 120000 })   // first eval pulls from Snowflake
        await page.waitForTimeout(300)
    },
    async 'draft-EC'(page)   { await setMode(page, 'Draft Mode'); await setFmt(page, 'Each Category') },
    async 'draft-MC'(page)   { await setMode(page, 'Draft Mode'); await setFmt(page, 'Most Categories') },
    async 'draft-Roto'(page) { await setMode(page, 'Draft Mode'); await setFmt(page, 'Rotisserie') },

    async 'position'(page)     { await ensure(page, 'draft-EC'); await expandSection(page, /Position/i) },
    // Player Stats section open; the base data source (historical, set once after load) is unchanged.
    async 'player-stats'(page)      { await ensure(page, 'draft-EC'); await expandSection(page, /Player Stats/i) },
    // projections shot: switch to Projections to show that config panel. Runs at the END — leaks projections.
    async 'player-stats-proj'(page) { await ensure(page, 'player-stats'); await setSelect(page, 'ps-data-type', 'Projections'); await waitEval(page) },
    // 1984-85 shot: switch the historical season. Runs at the END — leaks the old season.
    async 'season-1984-85'(page)    {
        await ensure(page, 'player-stats')
        await setSelect(page, 'ps-season', '1984-85')
        await waitEval(page)
        // The season change clears #hscoretable and re-evaluates; wait for the new rows to render.
        await page.locator('#hscoretable .playerheaderdiv').first().waitFor({ timeout: 30000 })
        await page.waitForTimeout(500)
    },
    async 'league-settings'(page)   { await ensure(page, 'draft-EC'); await expandSection(page, /League Settings/i) },

    // hexp / hstrat feature Giannis — the docs' punt-strategy exemplar (he's a top-5 EC pick in 2024-25).
    // The detailed drop-down shots (hexp/hstrat/hflex/hroster) illustrate a team that took Giannis
    // in round 1 and is now considering Dyson Daniels in round 2. Team 1 drafts Giannis first; the
    // evaluation seat stays on Team 1 (runEvaluate keys off the seat, not the current pick), so
    // expanding Daniels shows his detail for a Giannis-owning team.
    async 'candidate'(page) {
        await ensure(page, 'draft-EC')
        await lockInDraftPick(page, 'Giannis')          // Team 1 takes Giannis with the first pick
        await setSelect(page, 'seat-select', 'Team 1')  // evaluate from Team 1's perspective
        await waitEval(page)
        await expandCandidateByName(page, 'Dyson Daniels')
    },

    // Toggle two drafters' autodraft "A" squares on so they highlight in the entry-table header.
    async 'draft-autodraft'(page) {
        await ensure(page, 'draft-EC')
        const toggles = page.locator('.entry-table thead .method-dd-trigger')
        await toggles.nth(1).click()
        await toggles.nth(3).click()
        await page.waitForTimeout(200)
    },

    // Auction / Season are not the H-scoring page, so they reset to 2025-26 (Season also *needs* it —
    // its default rosters are 2025-26 players, who have no stats in 2024-25).
    async 'auction'(page)           { await setMode(page, 'Auction Mode'); await selectHistoricalSeason(page, '2025-26') },
    async 'auction-candidate'(page) { await ensure(page, 'auction'); await expandCandidate(page, 0) },

    async 'league-fantrax'(page) { await ensure(page, 'league-settings'); await setSelect(page, 'ls-platform', 'Fantrax') },
    async 'league-espn'(page)    { await ensure(page, 'league-settings'); await setSelect(page, 'ls-platform', 'ESPN') },

    async 'season-waiver'(page)   {
        await setMode(page, 'Season Mode')
        await selectHistoricalSeason(page, '2025-26')
        await page.locator('.season-tab-btn[data-tab="waiver"]').click()
        await page.locator('.waiver-selector-row, .coming-soon').first().waitFor()
        // The waiver eval renders into the shared #hscoretable, but arriving from Auction Mode it can
        // keep the stale dollar view until the selection changes. Toggle the team select via Team 2 to
        // force a fresh runWaiver, so #hscoretable re-renders as the substitution table.
        await setSelect(page, 'waiver-team-select', 'Team 2')
        await waitEval(page)
        await setSelect(page, 'waiver-team-select', 'Team 1')
        await waitEval(page)
    },
    async 'season-waiver-exp'(page) { await ensure(page, 'season-waiver'); await expandCandidate(page, 0) },
    async 'season-trading'(page)  { await setMode(page, 'Season Mode'); await selectHistoricalSeason(page, '2025-26'); await page.locator('.season-tab-btn[data-tab="trading"]').click(); await page.locator('.trade-left-col .ms-container').first().waitFor() },
    // Pick one player to send and one to receive so the trade result panes populate.
    // The multiselect opens its dropdown on focus (async), so wait for the option before clicking.
    async 'season-trade'(page)    {
        await ensure(page, 'season-trading')
        const col = page.locator('.trade-left-col').first()
        for (const containerIndex of [0, 1]) {
            const container = col.locator('.ms-container').nth(containerIndex)
            await container.locator('.ms-input-area').click()
            const option = container.locator('.ms-option').first()
            await option.waitFor({ state: 'visible' })
            await option.click()
            // Close this select's dropdown — it expands over the select below and blocks its click.
            await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur())
            await page.waitForTimeout(200)
        }
        await waitEval(page)
    },
    async 'season-trade-g'(page)  { await ensure(page, 'season-trade'); await page.locator('.trade-tab-btn', { hasText: /G-score/i }).click(); await page.waitForTimeout(150) },
    async 'season-rosters'(page)  { await setMode(page, 'Season Mode'); await selectHistoricalSeason(page, '2025-26'); await page.locator('.season-tab-btn[data-tab="rosters"]').click() },
    async 'season-roster-insp'(page) {
        await ensure(page, 'season-rosters')
        // Team 1 is the default selection, so selecting it fires no change event and never re-triggers
        // the inspector's backend eval — whose initial run can come back empty under load, leaving the
        // async H-score row (rosterh) unrendered. Toggle via Team 2 to force a fresh, change-driven eval.
        await setSelect(page, 'sr-team-select', 'Team 2')
        await page.waitForTimeout(200)
        await setSelect(page, 'sr-team-select', 'Team 1')
        // The G-score table lists only players present in the G-score map; on a not-yet-warm session
        // that map fills in gradually, so wait for the full 13-player roster (not just Jokic) before
        // capturing. Then wait for the async backend H-score row (rosterh).
        await page.waitForFunction(() => {
            const table = document.querySelector('[data-testid="roster-inspection-gscore"]')
            return table && table.querySelectorAll('tbody tr').length >= 13
        }, { timeout: 30000 }).catch(() => {})
        await page.locator('[data-testid="roster-inspection-hscore"]').waitFor({ timeout: 30000 }).catch(() => {})
        await page.waitForTimeout(200)
    },
}

// ─── manifest / COVERAGE TRACKER (working toward 100% automated) ─────────────────
// The human-facing list of what still needs fixing lives in scripts/SCREENSHOTS_TODO.md — keep it
// in sync when shots move between states. `skip` bypasses a shot unless it's named on the CLI.
// Current status of every app screenshot:
//
//   Data-dependent shots use a fixed HISTORICAL season so they stay identical over time (live
//   projections drift). The H-scoring / draft page uses 2024-25 (Giannis, the docs' punt exemplar,
//   ranks top-6 there); auction & season use 2025-26 (Season needs it — its rosters are 2025-26
//   players). Set in the runner + auction/season states via selectHistoricalSeason().
//
//   AUTO (26, capture cleanly & verified light-theme): main hec hmc rototop positions hexp
//     hstrat hflex hroster projections 1984-85 lsettings fantraxsettings espnpop hdollars
//     auctiondetail hwaiver hwaiverexp tradeanalysis tradeanalysisg tradesuggestions tp3 rosters
//     rosterinspection rosterh autodraft
//
//   NOTE: removed as low-value-out-of-context: the single-parameter crops (chi aleph beth iterations
//     puntcontrol injury savorinput) and the settings dropdowns (formats categories historical =
//     season selector). Only the position-structure control (positions) is kept. tp1/tp2 and notbegun
//     were removed too (trade UI reworked to one inline row = tp3; no "auction not begun" state exists).
//
//   TODO (skipped; doable — need one extra interaction/data step, see each `skip`):
//     hec2 updating mdraft moreinfo mauction gteam rosterjokic
//
//   BLOCKED (skipped; external/live login — need real third-party credentials):
//     yahoopop yahoosettings livedraft
//
const SHOTS = [
    // Draft / H-scoring
    { name: 'main',        state: 'load',        selector: '#app-layout', viewport: true },
    { name: 'hec',         state: 'draft-EC',    selector: '#hscoretable', rows: 12 },
    { name: 'hmc',         state: 'draft-MC',    selector: '#hscoretable', rows: 12 },
    { name: 'rototop',     state: 'draft-Roto',  selector: '#hscoretable', rows: 12 },
    { name: 'positions',   state: 'position',    selector: '[data-testid="position-structure"]' },
    { name: 'hexp',        state: 'candidate',   selector: '[data-testid="gscore-expectations-table"]' },
    { name: 'hstrat',      state: 'candidate',   selector: '[data-testid="future-pick-strategy-table"]' },
    { name: 'hflex',       state: 'candidate',   selector: '[data-testid="flex-allocations-table"]' },
    { name: 'hroster',     state: 'candidate',   selector: '[data-testid="roster-assignments-table"]' },
    { name: 'hec2',        state: 'draft-EC',    selector: '#hscoretable', skip: 'needs the board driven to a round-7 pick first' },
    { name: 'updating',    state: 'draft-EC',    selector: '#eval-indicator', skip: 'transient — capture during an evaluate, not after' },
    { name: 'mdraft',      state: 'draft-EC',    selector: '.entry-table', skip: 'wants pick controls + board together; pick a container/region' },
    // autodraft is defined LAST (before the platform shots) — toggling the "A" autodrafters kicks
    // off autopilot, which mutates the persistent draft/candidate state for every shot after it.
    { name: 'livedraft',   state: 'load',        selector: '#hscoretable', skip: 'needs a live platform connection' },

    // Player Stats config panels (`projections` / `1984-85`) are defined near the END — they switch
    // the data source away from the base historical 2025-26, which would otherwise leak into the
    // auction/season shots. The `historical` shot needs no switch (2025-26 historical is the base).

    // League setup
    // NOTE: the platform-switching shots (league-fantrax / league-espn states) are run LAST,
    // at the bottom of this manifest — they set ls-platform to a live provider, which flips every
    // auction/season state into live-platform mode (blank rosters, no eval data). Keeping them
    // last means own-data states above capture correctly first.
    { name: 'lsettings',       state: 'league-settings', selector: '.ls-grid' },
    { name: 'moreinfo',        state: 'league-settings', selector: '.ls-grid', skip: 'manual-entry inputs span several controls; pick the right region' },
    { name: 'yahoopop',        state: 'load',          selector: 'body', skip: 'real external window.open to Yahoo login — needs popup-page handling, not deterministic' },
    { name: 'yahoosettings',   state: 'load',          selector: '#ls-connect-cell', skip: 'requires a live Yahoo auth session' },

    // Auction
    { name: 'auctiondetail', state: 'auction-candidate', selector: '[data-testid="auction-values-table"]' },
    { name: 'hdollars',      state: 'auction',           selector: '#hscoretable', rows: 12 },
    { name: 'mauction',      state: 'auction',           selector: '.pick-control-row', skip: 'auction manual-entry table + pick row; confirm the right container' },

    // Season
    { name: 'hwaiver',          state: 'season-waiver',     selector: '#hscoretable', rows: 12 },
    { name: 'hwaiverexp',       state: 'season-waiver-exp', selector: '[data-testid="gscore-expectations-table"]' },
    { name: 'tradeanalysis',    state: 'season-trade',      selector: '[data-testid="trade-hscore-pane"]' },
    { name: 'tradeanalysisg',   state: 'season-trade-g',    selector: '[data-testid="trade-gscore-pane"]' },
    { name: 'tradesuggestions', state: 'season-trading',    selector: '[data-testid="trade-suggestions"]' },
    { name: 'rosters',          state: 'season-rosters',    selector: '#rosters-left' },
    { name: 'rosterinspection', state: 'season-roster-insp', selector: '[data-testid="roster-inspection-gscore"]' },
    { name: 'rosterh',          state: 'season-roster-insp', selector: '[data-testid="roster-inspection-hscore"]' },
    { name: 'rosterjokic',      state: 'season-rosters',    selector: '.cs-dropdown', skip: 'double-click a roster cell and type "Jokic" to open the search' },
    // The old sidebar "trade parameters popover" (tp1/tp2) is gone — the trade-size combos and
    // both differential thresholds are now one inline control row, captured as tp3.
    { name: 'tp3', state: 'season-trading', selector: '.trade-combo-row' },

    // G-scores
    { name: 'gteam', state: 'draft-EC', selector: '[data-testid="team-gscore"]', skip: 'open the "Show team statistics" tab first (Draft/Auction only)' },

    // Data-source config panels — defined here (not in the middle) because `projections` / `1984-85`
    // switch the data source, which must not leak into the historical-2025-26 auction/season shots.
    // Order matters: `historical` (base 2025-26) and `1984-85` before `projections` flips the type.
    { name: 'projections', state: 'player-stats-proj', selector: '#ps-proj-section' },
    { name: '1984-85',     state: 'season-1984-85',    selector: '#hscoretable', rows: 12 },

    // Autodraft — runs late because toggling the "A" squares triggers autopilot, which mutates the
    // draft/candidate board for anything after it. Draft-mode state, so it precedes the platform shots.
    { name: 'autodraft',   state: 'draft-autodraft', selector: '.entry-table thead' },

    // Live-platform settings — MUST stay last: these switch ls-platform to a live provider, which
    // poisons every own-data auction/season state above (blank rosters, no eval). Nothing own-data
    // dependent may run after them.
    { name: 'fantraxsettings', state: 'league-fantrax', selector: '#ls-fantrax-wrap' },
    { name: 'espnpop',         state: 'league-espn',    selector: '.espn-modal-box' },
]

// Auth: the app UI is gated behind Google login. Inject a session cookie minted with the
// app's own SESSION_SECRET_KEY (scripts/mint_session_cookie.py) so it loads headlessly.
const repoRoot   = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const authCookie = JSON.parse(execFileSync('python', ['scripts/mint_session_cookie.py'],
                                           { cwd: repoRoot, encoding: 'utf8' }).trim())

const browser = await chromium.launch()
const ctx     = await browser.newContext(CONTEXT)
await ctx.addCookies([{ name: authCookie.name, value: authCookie.value, url: APP }])
// Force light theme (the app manages its own, keyed on this pref) so every shot matches.
await ctx.addInitScript(() => { try { localStorage.setItem('fbbo-light_mode', 'true') } catch {} })
const page    = await ctx.newPage()
page.setDefaultTimeout(20000)   // fail a stuck locator fast instead of hanging
// Capture one shot, dispatching on its kind. Returns true on success.
async function captureShot(page, s) {
    try {
        await ensure(page, s.state)
        if (s.viewport) await shootViewport(page, s.name)
        else if (s.rows) await shootRows(page, s.name, s.selector, s.rows)
        else await shoot(page, s.name, s.selector)
        return true
    } catch (err) {
        console.error(`✗ ${s.name}: ${err.message}`)
        return false
    }
}

let ok = 0, failed = 0
try {
    await STATES.load(page); current = 'load'   // one-time page load; states transition from here
    await selectHistoricalSeason(page, '2024-25')   // H-scoring/draft base: 2024-25 (Giannis ranks top-6); auction/season reset to 2025-26
    const failedShots = []
    for (const s of SHOTS) {
        if (only.size ? !only.has(s.name) : s.skip) continue
        if (await captureShot(page, s)) ok++
        else failedShots.push(s)
    }
    // Retry pass: some shots depend on a backend eval that returns empty only after many evals have
    // accumulated in one session (e.g. rosterh). A fresh page load resets the session, so retrying
    // the failures once from a clean load clears these cumulative-load casualties.
    if (failedShots.length) {
        console.log(`\nRetrying ${failedShots.length} failed shot(s) from a fresh load: ${failedShots.map(s => s.name).join(' ')}`)
        await STATES.load(page); current = 'load'
        await selectHistoricalSeason(page, '2024-25')   // re-establish the base after the reload
        for (const s of failedShots) {
            if (await captureShot(page, s)) ok++
            else failed++
        }
    }
} finally {
    await browser.close()
}
console.log(`\n${ok} captured, ${failed} failed${only.size ? '' : `, ${SHOTS.filter(s => s.skip).length} skipped (run by name to force)`}`)

// ─── element capture ──────────────────────────────────────────────────────────
async function shoot(page, name, selector) {
    const el = page.locator(selector).first()
    await el.waitFor({ state: 'visible' })
    await el.scrollIntoViewIfNeeded().catch(() => {})
    // Temporarily pad the element itself, so it captures with clean whitespace (its own background)
    // — reflow pushes neighbours out of the shot. Reverted right after.
    await el.evaluate((e, p) => { e.dataset._css = e.style.cssText; e.style.padding = p + 'px'; e.style.boxSizing = 'content-box' }, PAD)
    await page.waitForTimeout(120)
    await el.screenshot({ path: path.join(IMG, `${name}.png`) })
    await el.evaluate(e => { e.style.cssText = e.dataset._css || ''; delete e.dataset._css })
    console.log('✓', name)
}

// Whole-window shot (e.g. `main`) — the app-layout element is very tall, so capture the viewport.
async function shootViewport(page, name) {
    await page.waitForTimeout(150)
    await page.screenshot({ path: path.join(IMG, `${name}.png`) })
    console.log('✓', name, '(viewport)')
}

// Candidate/player tables render every player; clip to the header + first `rows` player rows.
async function shootRows(page, name, selector, rows) {
    const table = page.locator(selector).first()
    await table.waitFor({ state: 'visible' })
    await table.scrollIntoViewIfNeeded().catch(() => {})
    await page.waitForTimeout(150)
    const box = await table.boundingBox()
    const nth = page.locator(`${selector} .playerheaderdiv`).nth(rows - 1)
    const rowBox = await nth.boundingBox().catch(() => null)
    const bottom = rowBox ? rowBox.y + rowBox.height : box.y + box.height
    await page.screenshot({ path: path.join(IMG, `${name}.png`), clip: { x: box.x, y: box.y, width: box.width, height: bottom - box.y } })
    console.log('✓', name, `(top ${rows})`)
}
