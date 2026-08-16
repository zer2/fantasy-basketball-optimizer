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
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { mintSessionCookie, setSelect, waitEval, lockInDraftPick } from './browser_helpers.mjs'

const APP  = process.env.APP_URL ?? 'http://localhost:8000'
// Defaults to docs/img; set SHOT_OUT to a scratch dir when debugging so real images aren't clobbered.
const IMG  = process.env.SHOT_OUT ?? path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'docs', 'img')
const only = new Set(process.argv.slice(2))

const CONTEXT = {
    // 1280 rather than a roomier desktop width: the docs render every image at the text
    // column (~760px), so the wider the app lays out, the more each shot is shrunk and the
    // smaller the app's own labels end up — that shrink, not the PNG's pixel count, is what
    // reads as blur. The app's tables scale to the window, so a narrower capture yields the
    // same content in fewer CSS pixels and lands much closer to 1:1 on the page. 1280 is the
    // floor: at ~1100 the candidate table starts clipping its category headers.
    viewport:          { width: 1280, height: 900 },
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

// setSelect / waitEval / lockInDraftPick are shared with the e2e suite — see browser_helpers.mjs.

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

// Lock in the current top-ranked candidate for whoever is on the clock (name-agnostic, so it works on
// any season). Used to give a team its first pick before letting the rest of the field autodraft.
async function lockInTopPick(page) {
    const wrap = page.locator('[data-testid="draft-pick-select-wrapper"]').first()
    await wrap.locator('.cs-trigger').click()
    await wrap.locator('.cs-dropdown .cs-option').first().click()
    await page.locator('.pick-control-row .pick-btn', { hasText: 'Lock in selection' }).click()
    await waitEval(page)
}

// Set one seat's autodraft toggle to a desired on/off state, idempotently — the "A" trigger toggles on
// each click, so we click only when the current state (aria-pressed) differs. Deterministic regardless of
// what a prior shot left toggled.
async function setSeatAutodraft(page, index, on) {
    const trigger = page.locator('.entry-table thead .method-dd-trigger').nth(index)
    const isOn = (await trigger.getAttribute('aria-pressed')) === 'true'
    if (isOn !== on) await trigger.click()
}

// Turn every seat into an autodrafter. Seat 0 is toggled LAST: it is the current drafter, and toggling
// the current drafter to autodraft is what kicks off autopilot — by then all other seats are already
// autodrafters, so autopilot runs the entire board rather than stopping at the first manual seat.
async function setAllSeatsAutodraft(page) {
    const count = await page.locator('.entry-table thead .method-dd-trigger').count()
    for (let i = 1; i < count; i++) await setSeatAutodraft(page, i, true)
    await setSeatAutodraft(page, 0, true)
}

// Every seat EXCEPT seat 0 becomes an autodrafter. With seat 0 manual and on the clock, locking its pick
// lets autopilot fill the rest of round one and wrap back down to seat 0's next turn, then stop — a
// deterministic mid-draft board with seat 0 on the clock.
async function setLaterSeatsAutodraft(page) {
    const count = await page.locator('.entry-table thead .method-dd-trigger').count()
    for (let i = 1; i < count; i++) await setSeatAutodraft(page, i, true)
    await setSeatAutodraft(page, 0, false)
}

// Clear the draft board back to empty via the pick control, if the button is present.
async function clearDraftBoard(page) {
    const clearBtn = page.locator('.pick-control-row .pick-btn', { hasText: 'Clear draft board' })
    if (await clearBtn.count()) { await clearBtn.first().click(); await waitEval(page) }
}

// Autopilot hides the seat selector while it runs and restores it when done. Wait for it to go hidden
// (autopilot started) and then visible again (finished) — robust to a full-board run taking minutes.
async function waitAutopilotDone(page) {
    const hidden = () => getComputedStyle(document.getElementById('seat-selector-container')).visibility === 'hidden'
    await page.waitForFunction(hidden, { timeout: 8000 }).catch(() => {})
    await page.waitForFunction(() => getComputedStyle(document.getElementById('seat-selector-container')).visibility !== 'hidden',
                               { timeout: 240000 }).catch(() => {})
    await waitEval(page)
    await page.waitForTimeout(300)
}

// Wait for the team-statistics G-score table to fill in the full roster (13 players) before capturing.
async function waitTeamGscore(page) {
    await waitEval(page)
    await page.waitForFunction(() => {
        const table = document.querySelector('[data-testid="team-gscore"]')
        return table && table.querySelectorAll('tbody tr').length >= 13
    }, { timeout: 30000 }).catch(() => {})
    await page.waitForTimeout(200)
}

// A full 12-seat autodraft in 2025-26 EC, built ONCE and reused by both team-shot states (Team 5 / Team 2)
// so the expensive whole-board draft runs a single time. Switching the season first resets the board and
// the drafter methods, so this starts from a clean slate regardless of earlier shots.
let fullAutodraftReady = false
async function ensureFullAutodraft(page) {
    if (fullAutodraftReady) return
    await setMode(page, 'Draft Mode')
    await setFmt(page, 'Each Category')
    await selectHistoricalSeason(page, '2025-26')   // resets the board + methods
    await setAllSeatsAutodraft(page)
    await waitAutopilotDone(page)
    await page.locator('.draft-tab-bar .pick-btn[data-tab="my-team"]').click()   // Show team statistics
    fullAutodraftReady = true
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

/** Waits until the candidate table actually holds player rows.
 *
 *  waitEval only watches the eval indicator, which is NOT sufficient after a sidebar change:
 *  the Player Stats section debounces for 800ms, so the indicator still reads 'idle' from the
 *  previous evaluate when waitEval looks at it, and it returns while the table sits emptied by
 *  buildTableHeader and not yet refilled. That window is wide enough to photograph — it is how
 *  `main` was captured showing a header row and nothing under it. */
//  Not every state HAS a candidate list: Season Mode shows none until a waiver evaluation
//  runs, and selectHistoricalSeason is called on the way into season and auction states too.
//  So a hidden or absent table counts as "nothing to wait for", and the wait is advisory —
//  it times out quietly rather than failing the shot, since its job is to avoid photographing
//  a half-drawn table, not to assert that one exists.
async function waitCandidateRows(page, minimum = 8) {
    await page.waitForFunction(
        min => {
            const table = document.getElementById('hscoretable')
            if (!table || table.offsetParent === null) return true   // not on screen in this state
            return table.querySelectorAll('.playerheaderdiv').length >= min
        },
        minimum,
        { timeout: 30000 },
    ).catch(() => {})
    await page.waitForTimeout(200)
}

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
    await waitCandidateRows(page)
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
        await waitCandidateRows(page)
        await page.waitForTimeout(300)
    },
    async 'draft-EC'(page)   { await setMode(page, 'Draft Mode'); await setFmt(page, 'Each Category');  await waitCandidateRows(page) },
    async 'draft-MC'(page)   { await setMode(page, 'Draft Mode'); await setFmt(page, 'Most Categories'); await waitCandidateRows(page) },
    async 'draft-Roto'(page) { await setMode(page, 'Draft Mode'); await setFmt(page, 'Rotisserie');      await waitCandidateRows(page) },

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
        // The seat selector already sits on Team 1, and re-selecting the active seat fires no change
        // event — the table can then be captured showing whatever evaluate the lock-in triggered
        // (once observed as a different seat's perspective: a punt-3s build instead of Team 1's
        // punt-FT% build). Toggle via Team 2 so the final selection fires a change-driven evaluate,
        // the same trick auction-manual-team1 uses.
        await setSelect(page, 'seat-select', 'Team 2')
        await waitEval(page)
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

    // mdraft: the manual-entry draft view — pick controls + board captured together (#right-header
    // holds both). Clear any picks left by the candidate state (Giannis), then lock Nikola Jokic in
    // as the first overall pick so the board shows a real entry with Drafter 2 on the clock.
    async 'draft-manual'(page) {
        await ensure(page, 'draft-EC')
        await clearDraftBoard(page)
        await lockInDraftPick(page, 'Nikola Jok')
    },

    // Auction / Season are not the H-scoring page, so they reset to 2025-26 (Season also *needs* it —
    // its default rosters are 2025-26 players, who have no stats in 2024-25).
    async 'auction'(page)           { await setMode(page, 'Auction Mode'); await selectHistoricalSeason(page, '2025-26') },
    async 'auction-candidate'(page) { await ensure(page, 'auction'); await expandCandidate(page, 0) },

    // mauction: auction manual entry — record Nikola Jokic to Team 1 for $70 as the first board
    // entry, so the shot shows the pick controls plus a populated board (Team 1's remaining
    // budget drops to $130). Runs after auctiondetail, which wants the pristine empty board.
    async 'auction-manual'(page) {
        await ensure(page, 'auction')
        await setSelect(page, 'auction-pick-player', 'Nikola Jok')
        await setSelect(page, 'auction-pick-team', 'Team 1')
        await page.fill('.auction-cost-input', '70')
        await page.locator('.pick-control-row .pick-btn', { hasText: 'Lock in selection' }).click()
        await waitEval(page)
    },

    // hdollars: the $-value candidate table AFTER the first auction entry (Jokic to Team 1 for
    // $70), evaluated from Team 1's perspective — so the dollar values reflect Team 1's reduced
    // budget and filled roster spot. Toggle the seat via Team 2 so the final selection fires a
    // change-driven evaluate (selecting the already-active Team 1 directly fires no change event).
    async 'auction-manual-team1'(page) {
        await ensure(page, 'auction-manual')
        await setSelect(page, 'seat-select', 'Team 2')
        await waitEval(page)
        await setSelect(page, 'seat-select', 'Team 1')
        await waitEval(page)
    },

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

    // mid_draft: the candidate view partway through a draft. Clear any prior picks, make every seat but
    // Team 1 an autodrafter, then lock Team 1's first pick — autopilot fills round one and wraps back to
    // Team 1's next turn and stops, leaving a populated board with Team 1 on the clock. Stays on the
    // draft base season (2024-25). Runs late because it mutates the board heavily.
    async 'mid-draft'(page) {
        await ensure(page, 'draft-EC')
        await clearDraftBoard(page)
        await setLaterSeatsAutodraft(page)
        await lockInTopPick(page)          // Team 1 takes the top pick; autopilot then fills the field
        await waitAutopilotDone(page)
        await setSelect(page, 'seat-select', 'Team 1')   // view the candidate board from Team 1
        await waitEval(page)
    },

    // hec2: the candidate table at a round-7 pick — the docs' example of H-scores stabilising
    // across categories once a team's direction is set. Builds on mid-draft (Team 1 on the clock
    // in round 2): each lock-in hands the board to autopilot, which fills the field and stops at
    // Team 1's next turn (at the snake turns Team 1 picks back-to-back, so no autopilot fires).
    // Five more locks leave Team 1 on the clock for its round-7 pick. Expensive — two full
    // autopilot sweeps of ~22 picks each.
    async 'draft-round7'(page) {
        await ensure(page, 'mid-draft')
        for (let lockCount = 0; lockCount < 5; lockCount++) {
            await lockInTopPick(page)
            await waitAutopilotDone(page)
        }
    },

    // gteam: the team-statistics G-score table for a team mid-draft. Reuses the round-7 board —
    // Team 1 holds six players — and switches to the "Show team statistics" tab.
    async 'draft-round7-team'(page) {
        await ensure(page, 'draft-round7')
        await page.locator('.draft-tab-bar .pick-btn[data-tab="my-team"]').click()
        await page.waitForFunction(() => {
            const table = document.querySelector('[data-testid="team-gscore"]')
            return table && table.querySelectorAll('tbody tr').length >= 6
        }, { timeout: 30000 }).catch(() => {})
        await page.waitForTimeout(200)
    },

    // rosterjokic: the in-cell player search on the rosters grid. Double-click Team 2's first
    // roster cell and type "Niko" — the dropdown lists Nikola Jokic even though Team 1's roster
    // already has him (the caption's point: taken players remain selectable).
    async 'season-roster-search'(page) {
        await ensure(page, 'season-rosters')
        const cellWrapper = page.locator('[data-testid="sr-player-0-1-wrapper"]')
        await cellWrapper.locator('.cs-trigger').dblclick()
        await cellWrapper.locator('.cs-search-input').pressSequentially('Niko', { delay: 60 })
        await page.waitForTimeout(200)
    },

    // scottie_autodraft / sga_autodraft: two teams from ONE full 12-seat autodraft in 2025-26 EC, shown in
    // the team-statistics view. The whole-board draft is built once (ensureFullAutodraft) and reused; each
    // state just points the seat selector at its team. Team 5 / Team 2 match the doc captions' draft slots.
    async 'autodraft-team5'(page) {
        await ensureFullAutodraft(page)
        await setSelect(page, 'seat-select', 'Team 5')
        await waitTeamGscore(page)
    },
    async 'autodraft-team2'(page) {
        await ensureFullAutodraft(page)
        await setSelect(page, 'seat-select', 'Team 2')
        await waitTeamGscore(page)
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
//   AUTO (32, capture cleanly & verified light-theme): main hec hmc rototop positions hexp
//     hstrat hflex hroster projections 1984-85 lsettings fantraxsettings espnpop hdollars
//     auctiondetail hwaiver hwaiverexp tradeanalysis tradeanalysisg tradesuggestions tp3 rosters
//     rosterinspection rosterh autodraft mdraft (draft manual entry, Jokic locked in as pick 1)
//     mauction (auction manual entry, Jokic to Team 1 for $70) moreinfo (league-settings
//     manual-entry inputs, union capture) hec2 (candidate table at a round-7 pick) gteam
//     (team-statistics G-score table on the round-7 board) rosterjokic (rosters-grid player
//     search, union capture)
//
//   NEW (3, wired up but NOT yet verified against a live run — confirm framing/timing once):
//     mid_draft (candidate view mid-draft), scottie_autodraft (Team 5 team-stats after a full 2025-26
//     autodraft), sga_autodraft (Team 2, same draft). The two team shots reproduce the doc captions'
//     draft (SGA at seat 2 / Scottie at seat 5) using the current defaults, kappa included.
//
//   NOTE: removed as low-value-out-of-context: the single-parameter crops (chi aleph beth iterations
//     puntcontrol injury savorinput) and the settings dropdowns (formats categories historical =
//     season selector). Only the position-structure control (positions) is kept. tp1/tp2 and notbegun
//     were removed too (trade UI reworked to one inline row = tp3; no "auction not begun" state exists).
//     updating (the transient eval spinner) was dropped from the docs entirely.
//
//   BLOCKED (skipped; external/live login — need real third-party credentials):
//     yahoopop yahoosettings livedraft
//
const SHOTS = [
    // Draft / H-scoring
    // About half the viewport: the board plus the first few candidates, cut at a row edge.
    { name: 'main',        state: 'load',        selector: '#app-layout', viewport: true, throughRows: 3 },
    { name: 'hec',         state: 'draft-EC',    selector: '#hscoretable', rows: 12 },
    { name: 'hmc',         state: 'draft-MC',    selector: '#hscoretable', rows: 12 },
    { name: 'rototop',     state: 'draft-Roto',  selector: '#hscoretable', rows: 12 },
    { name: 'positions',   state: 'position',    selector: '[data-testid="position-structure"]' },
    { name: 'hexp',        state: 'candidate',   selector: '[data-testid="gscore-expectations-table"]' },
    { name: 'hstrat',      state: 'candidate',   selector: '[data-testid="future-pick-strategy-table"]' },
    { name: 'hflex',       state: 'candidate',   selector: '[data-testid="flex-allocations-table"]' },
    { name: 'hroster',     state: 'candidate',   selector: '[data-testid="roster-assignments-table"]' },
    // #right-header holds the pick-control row and the board grid together in own-data mode.
    { name: 'mdraft',      state: 'draft-manual', selector: '#right-header' },
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
    // The manual-entry inputs (drafters / picks-per-drafter / third-round reversal) have no shared
    // wrapper — the toggle sits outside .ls-grid — so this is a union capture of the three regions,
    // clamped below the platform select so the padding never bleeds into it.
    { name: 'moreinfo',        state: 'league-settings',
      union:       ['.ls-cell:has(#ls-n-drafters)', '.ls-cell:has(#ls-n-picks)', '#ls-trr-row'],
      clampAbove:  '[data-testid="ls-platform-wrapper"]',
      clampWithin: 'details.sidebar-section:has(#ls-trr-row)' },
    { name: 'yahoopop',        state: 'load',          selector: 'body', skip: 'real external window.open to Yahoo login — needs popup-page handling, not deterministic' },
    { name: 'yahoosettings',   state: 'load',          selector: '#ls-connect-cell', skip: 'requires a live Yahoo auth session' },

    // Auction — auctiondetail first (pristine empty board), then the Jokic entry, then the
    // dollar table from Team 1's post-pick perspective.
    { name: 'auctiondetail', state: 'auction-candidate',    selector: '[data-testid="auction-values-table"]' },
    { name: 'mauction',      state: 'auction-manual',       selector: '#right-header' },
    { name: 'hdollars',      state: 'auction-manual-team1', selector: '#hscoretable', rows: 12 },

    // Season
    { name: 'hwaiver',          state: 'season-waiver',     selector: '#hscoretable', rows: 12 },
    { name: 'hwaiverexp',       state: 'season-waiver-exp', selector: '[data-testid="gscore-expectations-table"]' },
    { name: 'tradeanalysis',    state: 'season-trade',      selector: '[data-testid="trade-hscore-pane"]' },
    { name: 'tradeanalysisg',   state: 'season-trade-g',    selector: '[data-testid="trade-gscore-pane"]' },
    { name: 'tradesuggestions', state: 'season-trading',    selector: '[data-testid="trade-suggestions"]' },
    { name: 'rosters',          state: 'season-rosters',    selector: '#rosters-left' },
    { name: 'rosterinspection', state: 'season-roster-insp', selector: '[data-testid="roster-inspection-gscore"]' },
    { name: 'rosterh',          state: 'season-roster-insp', selector: '[data-testid="roster-inspection-hscore"]' },
    // Union capture with no padding: the region is inside the roster table, and padding would
    // show slivers of the pick-number column and Team 3's cells. Framing: the Team 1 + Team 2
    // column headers, five rows of Team 1's roster (Jokic on top), and the open search dropdown.
    { name: 'rosterjokic',      state: 'season-roster-search', pad: 0,
      union: ['#rosters-left .entry-table thead th:nth-child(2)',
              '#rosters-left .entry-table thead th:nth-child(3)',
              '[data-testid="sr-player-4-0-wrapper"]',
              '[data-testid="sr-player-0-1-wrapper"] .cs-dropdown'] },
    // The old sidebar "trade parameters popover" (tp1/tp2) is gone — the trade-size combos and
    // both differential thresholds are now one inline control row, captured as tp3.
    { name: 'tp3', state: 'season-trading', selector: '.trade-combo-row' },

    // Data-source config panels — defined here (not in the middle) because `projections` / `1984-85`
    // switch the data source, which must not leak into the historical-2025-26 auction/season shots.
    // Order matters: `historical` (base 2025-26) and `1984-85` before `projections` flips the type.
    { name: 'projections', state: 'player-stats-proj', selector: '#ps-proj-section' },
    { name: '1984-85',     state: 'season-1984-85',    selector: '#hscoretable', rows: 12 },

    // Autodraft — runs late because toggling the "A" squares triggers autopilot, which mutates the
    // draft/candidate board for anything after it. Draft-mode state, so it precedes the platform shots.
    { name: 'autodraft',   state: 'draft-autodraft', selector: '.entry-table thead' },

    // Mid-draft candidate view + the two full-autodraft team shots. All three heavily mutate the board,
    // so they run after every clean draft shot. mid_draft clears + rebuilds on the 2024-25 base; the two
    // team shots switch to 2025-26 for a single full autodraft (which resets the board on season change).
    { name: 'mid_draft',         state: 'mid-draft',       selector: '#hscoretable', rows: 12 },
    // hec2 + gteam extend mid_draft's board (five more lock/autopilot cycles → round 7), so they
    // run directly after it; gteam then reuses the round-7 board on the team-statistics tab.
    { name: 'hec2',              state: 'draft-round7',      selector: '#hscoretable', rows: 12 },
    { name: 'gteam',             state: 'draft-round7-team', selector: '[data-testid="team-gscore"]' },
    // #draft-gscore wraps both the roster table (team-gscore) and the separate H-Score (est. win rate)
    // summary table below it — capture the container so the team's H-score row is in frame.
    { name: 'scottie_autodraft', state: 'autodraft-team5', selector: '#draft-gscore' },
    { name: 'sga_autodraft',     state: 'autodraft-team2', selector: '#draft-gscore' },

    // Live-platform settings — MUST stay last: these switch ls-platform to a live provider, which
    // poisons every own-data auction/season state above (blank rosters, no eval). Nothing own-data
    // dependent may run after them.
    { name: 'fantraxsettings', state: 'league-fantrax', selector: '#ls-fantrax-wrap' },
    { name: 'espnpop',         state: 'league-espn',    selector: '.espn-modal-box' },
]

// Auth: the app UI is gated behind Google login. Inject a session cookie minted with the
// app's own SESSION_SECRET_KEY (scripts/mint_session_cookie.py) so it loads headlessly.
const repoRoot   = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const authCookie = mintSessionCookie(repoRoot)

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
        if (s.viewport) await shootViewport(page, s.name, s.throughRows ?? 0)
        else if (s.union) await shootUnion(page, s)
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
// `throughRows` keeps only the top of it, down to the bottom edge of that many candidate rows:
// the full 1440x900 frame is scaled down to the docs' text width, which leaves every label too
// small to read, and cropping instead of shrinking keeps the part that carries the point legible.
// Measured from the row rather than set as a fixed fraction so the cut always lands on a row
// boundary instead of slicing one in half whenever the layout shifts.
async function shootViewport(page, name, throughRows = 0) {
    await page.waitForTimeout(150)
    const { width, height } = CONTEXT.viewport
    let clipHeight = height
    if (throughRows > 0) {
        const rowBox = await page.locator('#hscoretable .playerheaderdiv').nth(throughRows - 1)
            .boundingBox().catch(() => null)
        if (rowBox) clipHeight = Math.min(height, Math.ceil(rowBox.y + rowBox.height))
    }
    await page.screenshot({
        path: path.join(IMG, `${name}.png`),
        clip: { x: 0, y: 0, width, height: clipHeight },
    })
    console.log('✓', name, throughRows > 0 ? `(viewport, through row ${throughRows})` : '(viewport)')
}

// Union capture: some doc regions span sibling elements with no shared wrapper (e.g. the
// league-settings manual-entry inputs, whose toggle sits outside .ls-grid). Clip to the union of
// the `union` selectors' boxes plus padding (`pad` on the shot, PAD by default — use 0 for regions
// inside a table, where padding would show slivers of neighbouring cells), clamped (a) below the
// `clampAbove` element so the padding never bleeds into the control directly above, and (b) inside
// the `clampWithin` element so the shot never crosses the containing section's edge.
async function shootUnion(page, shot) {
    const pad = shot.pad ?? PAD
    for (const selector of shot.union) {
        await page.locator(selector).first().scrollIntoViewIfNeeded().catch(() => {})
    }
    await page.waitForTimeout(120)
    const boxes = []
    for (const selector of shot.union) {
        const box = await page.locator(selector).first().boundingBox()
        if (box) boxes.push(box)
    }
    if (!boxes.length) throw new Error(`no visible elements for union: ${shot.union.join(' ')}`)
    let left   = Math.min(...boxes.map(box => box.x)) - pad
    let top    = Math.min(...boxes.map(box => box.y)) - pad
    let right  = Math.max(...boxes.map(box => box.x + box.width)) + pad
    let bottom = Math.max(...boxes.map(box => box.y + box.height)) + pad
    if (shot.clampAbove) {
        const aboveBox = await page.locator(shot.clampAbove).first().boundingBox()
        if (aboveBox) top = Math.max(top, aboveBox.y + aboveBox.height + 2)
    }
    if (shot.clampWithin) {
        const withinBox = await page.locator(shot.clampWithin).first().boundingBox()
        if (withinBox) {
            left   = Math.max(left, withinBox.x)
            top    = Math.max(top, withinBox.y)
            right  = Math.min(right, withinBox.x + withinBox.width)
            bottom = Math.min(bottom, withinBox.y + withinBox.height)
        }
    }
    const viewport = page.viewportSize()
    left   = Math.max(0, left)
    top    = Math.max(0, top)
    right  = Math.min(viewport.width, right)
    bottom = Math.min(viewport.height, bottom)
    await page.screenshot({ path: path.join(IMG, `${shot.name}.png`),
                            clip: { x: left, y: top, width: right - left, height: bottom - top } })
    console.log('✓', shot.name, '(union)')
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
