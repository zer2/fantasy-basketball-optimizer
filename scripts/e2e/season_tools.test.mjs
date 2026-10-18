// scripts/e2e/season_tools.test.mjs
// Season Mode's tools: roster-grid editing (paste + in-cell search), the roster
// inspector, the waiver wire, and trade analysis/suggestions.
//
// The tests build their OWN roster dataset: player names are read from the live pool
// and pasted into the grid (the app ships no rosters of its own). A 3-drafter league keeps evaluates fast and still
// gives the trade suggestions more than one team to search.
// Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect, setLeagueDrafterCount, readDropdownPlayerNames, setObjectiveWeight,
} from './helpers.mjs'

const TEAM_COUNT = 3
const PICKS_PER_TEAM = 13

test('season tools on a self-built roster dataset', async t => {
    const app = await launchAppPage()
    const { page } = app

    // Rosters used by every subtest: column 0 = Team 1, column 1 = Team 2.
    let teamRosters = []
    let spareBenchPlayer = ''   // a pool player left off both rosters, for the edit subtest

    async function openSeasonTab(tabId) {
        await page.locator(`.season-tab-btn[data-tab="${tabId}"]`).click()
        await waitAppSettled(app)
    }

    /** The inspector's G-score table rows (bare player names only — the rich row label's
     *  .playername leading text node, positions span excluded; totals row drops out
     *  because its plain-text th has no .playername). */
    function readInspectorPlayers() {
        return page.evaluate(() =>
            [...document.querySelectorAll('[data-testid="roster-inspection-gscore"] tbody tr')]
                .map(row => row.querySelector('th .playername')?.childNodes[0]?.textContent?.trim() ?? '')
                .filter(name => name !== ''))
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        await setLeagueDrafterCount(app, TEAM_COUNT)

        // Build the dataset from the live pool: the draft pick dropdown lists every player.
        // Bare names — the paste path resolves registry names, and the inspector shows them.
        const playerPool = await readDropdownPlayerNames(page, 'draft-pick-select-wrapper')
        assert.ok(playerPool.length > TEAM_COUNT * PICKS_PER_TEAM, 'player pool should cover the rosters')
        teamRosters = Array.from({ length: TEAM_COUNT }, (_, teamIndex) =>
            playerPool.slice(teamIndex * PICKS_PER_TEAM, (teamIndex + 1) * PICKS_PER_TEAM))
        spareBenchPlayer = playerPool[TEAM_COUNT * PICKS_PER_TEAM]
        expectCleanSession(app, 'initial load')

        await t.test('pasting a roster block fills the grid and drives the inspector', async () => {
            await setSelect(page, 'ls-mode', 'Season Mode')
            await waitAppSettled(app)
            await openSeasonTab('rosters')

            // Paste rows = picks, tab-separated columns = teams, anchored at the top-left cell.
            const pasteBlock = Array.from({ length: PICKS_PER_TEAM }, (_, row) =>
                teamRosters.map(roster => roster[row]).join('\t')).join('\n')
            await page.evaluate(text => navigator.clipboard.writeText(text), pasteBlock)
            await page.locator('[data-testid="sr-player-0-0-wrapper"]').click()
            await page.keyboard.press('Control+v')
            await waitAppSettled(app)

            // Toggle the inspected team so the selection fires a fresh change-driven eval.
            await setSelect(page, 'sr-team-select', 'Team 2')
            await setSelect(page, 'sr-team-select', 'Team 1')
            await page.locator('[data-testid="roster-inspection-hscore"]').waitFor({ timeout: 30000 })

            const inspectorPlayers = await readInspectorPlayers()
            assert.deepEqual(inspectorPlayers.sort(), [...teamRosters[0]].sort(),
                             "the inspector should list exactly Team 1's pasted roster")
            expectCleanSession(app, 'paste rosters + inspect')
        })

        await t.test('an in-cell search edit updates the roster and the inspector', async () => {
            // Replace Team 2's first player via the double-click search dropdown.
            const editedCell = page.locator('[data-testid="sr-player-0-1-wrapper"]')
            await editedCell.locator('.cs-trigger').dblclick()
            await editedCell.locator('.cs-search-input').pressSequentially(spareBenchPlayer.slice(0, 12), { delay: 30 })
            await editedCell.locator('.cs-dropdown .cs-option', { hasText: spareBenchPlayer }).first().click()
            await waitAppSettled(app)

            await setSelect(page, 'sr-team-select', 'Team 2')
            await page.locator('[data-testid="roster-inspection-gscore"]').waitFor({ timeout: 30000 })
            const inspectorPlayers = await readInspectorPlayers()
            assert.ok(inspectorPlayers.includes(spareBenchPlayer),
                      'the swapped-in player should appear in the inspector')
            assert.ok(!inspectorPlayers.includes(teamRosters[1][0]),
                      'the swapped-out player should leave the inspector')
            expectCleanSession(app, 'in-cell roster edit')
        })

        await t.test('the waiver wire renders a substitution table for the selected team', async () => {
            await openSeasonTab('waiver')
            await setSelect(page, 'waiver-team-select', 'Team 2')
            await waitAppSettled(app)
            await setSelect(page, 'waiver-team-select', 'Team 1')
            await waitAppSettled(app)

            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'the waiver evaluation should render candidate rows')
            expectCleanSession(app, 'waiver wire')
        })

        await t.test('a settings change redraws the waiver results', async () => {
            // A settings change rebuilt the candidate table's header (emptying it) and never re-ran the waiver
            // evaluate: headers over no rows until a dropdown was touched (2026-10-13).
            const countRows = () => page.locator('#hscoretable .playerheaderdiv').count()
            assert.ok(await countRows() > 0, 'the waiver table has rows before the change')
            await setObjectiveWeight(page, 0.3)
            await waitAppSettled(app)
            await page.waitForFunction(() => document.querySelectorAll('#hscoretable .playerheaderdiv').length > 0,
                                       null, { timeout: 30000 })
            assert.ok(await countRows() > 0, 'the waiver table has rows after the change')
            expectCleanSession(app, 'waiver redrawn after a settings change')
        })

        await t.test('trade analysis panes and suggestions populate', async () => {
            await openSeasonTab('trading')
            await page.locator('.trade-left-col .ms-container').first().waitFor()

            // Pick one player to send and one to receive (first option in each multiselect).
            for (const containerIndex of [0, 1]) {
                const container = page.locator('.trade-left-col .ms-container').nth(containerIndex)
                await container.locator('.ms-input-area').click()
                const option = container.locator('.ms-option').first()
                await option.waitFor({ state: 'visible' })
                await option.click()
                await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur())
            }
            await waitAppSettled(app)

            await page.locator('[data-testid="trade-hscore-pane"]').waitFor({ timeout: 30000 })
            await page.locator('.trade-tab-btn', { hasText: /G-score/i }).click()
            await page.locator('[data-testid="trade-gscore-pane"]').waitFor({ timeout: 30000 })
            await page.locator('[data-testid="trade-suggestions"]').waitFor({ timeout: 60000 })
            expectCleanSession(app, 'trade analysis + suggestions')
        })

        await t.test('suggested trades span every other team, ranked, searchable, and open in the inspection', async () => {
            // Suggestions used to cover only the team chosen to trade with; now every other team is searched, in bands
            // of value gap (one request per band, across every team), each batch merged in by your score.
            await openSeasonTab('trading')
            await setSelect(page, 'trade-your-team', 'Team 1')
            await waitAppSettled(app)
            // Let every trade through, so each team contributes rows: the thresholds are what a user would loosen.
            for (const id of ['ts-your-threshold', 'ts-their-threshold']) {
                await page.locator(`#${id}`).fill('-100')
                await page.locator(`#${id}`).blur()   // the change lands as the user leaves the box
            }
            await waitAppSettled(app)
            const rows = page.locator('[data-testid="trade-suggestions"] tbody tr')
            const teams = await rows.evaluateAll(trs => [...new Set(trs.map(tr => tr.dataset.team))].sort())
            assert.deepEqual(teams, ['Team 2', 'Team 3'], 'suggestions come from every other team')
            const yourScores = await rows.evaluateAll(trs => trs.map(tr => parseFloat(tr.cells[3].textContent)))
            assert.deepEqual(yourScores, [...yourScores].sort((a, b) => b - a), 'merged batches stay ranked by your score')

            const otherTeam = (await page.locator('input#trade-their-team').inputValue()) === 'Team 2' ? 'Team 3' : 'Team 2'
            await page.locator(`[data-testid="trade-suggestions"] tbody tr[data-team="${otherTeam}"]`).first().click()
            await waitAppSettled(app)
                        assert.equal(await page.locator('input#trade-their-team').inputValue(), otherTeam,
                         'inspecting a suggestion switches to the team it is made with')
            await page.locator('[data-testid="trade-hscore-pane"] table').first().waitFor({ timeout: 30000 })

            const searchedName = (await rows.first().locator('td').nth(1).locator('.player-minimal span').first().innerText()).trim()
            await page.locator('#trade-suggestion-search').fill(searchedName)
            const searchedRows = await rows.evaluateAll(trs => trs.map(tr => tr.innerText))
            assert.ok(searchedRows.length > 0 && searchedRows.every(text => text.includes(searchedName)),
                      'the search keeps only trades involving the player')
            await page.locator('#trade-suggestion-search').fill('zzzz')
            assert.match(await page.locator('[data-testid="trade-suggestions"]').innerText(), /No suggested trade matches/)
            await page.locator('#trade-suggestion-search').fill('')

            await page.locator('#trade-team-search').fill('team 3')
            assert.deepEqual(await rows.evaluateAll(trs => [...new Set(trs.map(tr => tr.dataset.team))]), ['Team 3'],
                             'the team search keeps only trades with that team')
            await page.locator('#trade-team-search').fill('')
            for (const id of ['ts-your-threshold', 'ts-their-threshold']) {
                await page.locator(`#${id}`).fill('0')
                await page.locator(`#${id}`).blur()
            }
            await waitAppSettled(app)
            expectCleanSession(app, 'suggestions across teams')
        })

        await t.test('thresholds filter the suggestions found, never below -2, without searching again', async () => {
            // Every search runs at the -2% floor and the thresholds filter what it found: a threshold change searched
            // every team again (2026-10-13).
            const suggestRequestsSince = start => app.sessionRequestLog.slice(start).filter(entry => entry.includes('/trade/suggest')).length
            const setThreshold = async (id, value) => {
                await page.locator(`#${id}`).fill(String(value))
                await page.locator(`#${id}`).blur()
                await waitAppSettled(app)
            }
            const readYourScores = () => page.locator('[data-testid="trade-suggestions"] tbody tr')
                .evaluateAll(trs => trs.map(tr => parseFloat(tr.cells[3].textContent)))
            const start = app.sessionRequestLog.length
            await setThreshold('ts-your-threshold', -100)
            assert.equal(await page.locator('#ts-your-threshold').inputValue(), '-2', 'nothing below the floor')
            const atFloor = await readYourScores()
            await setThreshold('ts-your-threshold', 0.2)
            const stricter = await readYourScores()
            assert.ok(stricter.length < atFloor.length && stricter.every(score => score > 0.2), 'a stricter threshold hides trades below it')
            await setThreshold('ts-your-threshold', 0)
            assert.equal(suggestRequestsSince(start), 0, 'no threshold change searches again')
            expectCleanSession(app, 'threshold filtering')
        })

        await t.test('the chosen trade sizes are remembered', async () => {
            const sizesInput = page.locator('.trade-combo-select')
            const chooseTradeSize = async label => {
                await sizesInput.locator('.ms-input-area').click()
                await sizesInput.locator('.ms-option', { hasText: label }).click()
                await page.evaluate(() => document.activeElement instanceof HTMLElement && document.activeElement.blur())
            }
            await chooseTradeSize('2 for 2')
            await waitAppSettled(app)
            await openSeasonTab('waiver')
            await openSeasonTab('trading')
            assert.match(await sizesInput.innerText(), /2 for 2/, 'the Trading tab reopens with 2 for 2 still chosen')
            expectCleanSession(app, 'trade sizes remembered')
        })

        await t.test('hand-entered rosters survive the grid being drawn again', async () => {
            // Every visit to the Rosters tab redraws the grid. It used to refill built-in sample rosters, which hid
            // that it dropped whatever had been entered; with those removed it came back empty (2026-10-13).
            const countFilledCells = () => page.evaluate(() =>
                [...document.querySelectorAll('input[id^="sr-player-"]')].filter(input => input.value !== '').length)
            await openSeasonTab('rosters')
            assert.equal(await countFilledCells(), TEAM_COUNT * PICKS_PER_TEAM, 'every pasted cell is still filled')
            expectCleanSession(app, 'rosters kept through a redraw')
        })

        await t.test('your team is remembered across the Waiver and Trading tabs', async () => {
            // The Waiver tab opened on the first team every time it was drawn, and the rosters arriving draw it: it
            // scored someone else's team (2026-10-13).
            const readChosenTeam = id => page.locator(`input#${id}`).inputValue()
            await openSeasonTab('waiver')
            await setSelect(page, 'waiver-team-select', 'Team 2')
            await waitAppSettled(app)
            await openSeasonTab('trading')
            assert.match(await readChosenTeam('trade-your-team'), /Team 2/, 'Trading opens on the team chosen in Waiver')
            await openSeasonTab('waiver')
            assert.match(await readChosenTeam('waiver-team-select'), /Team 2/, 'Waiver keeps it when drawn again')
            expectCleanSession(app, 'remembered team')
        })

        await t.test('the inspected team is kept when the Rosters tab is drawn again', async () => {
            // Every Season Mode layout pass and every visit to the Rosters tab redraws it, and the inspector opened on the
            // first team each time: the inspection switched to another team while it was being read (2026-10-17).
            const readInspectedTeam = () => page.locator('input#sr-team-select').inputValue()
            await openSeasonTab('rosters')
            await setSelect(page, 'sr-team-select', 'Team 3')
            await waitAppSettled(app)
            await openSeasonTab('waiver')
            await openSeasonTab('rosters')
            assert.match(await readInspectedTeam(), /Team 3/, 'the inspector keeps the chosen team through a redraw')
            expectCleanSession(app, 'inspected team kept')
        })

        await t.test('the indicator beside the tabs shows the backend working', async () => {
            const readIndicatorState = () => page.locator('#season-indicator').getAttribute('data-state')
            await openSeasonTab('waiver')
            let releaseEvaluate
            const evaluateHeld = new Promise(resolve => { releaseEvaluate = resolve })
            await page.route(url => url.pathname.endsWith('/evaluate'), async route => {
                await evaluateHeld
                await route.fallback()
            })
            await setSelect(page, 'waiver-team-select', 'Team 1')
            await page.waitForFunction(() => document.querySelector('#season-indicator')?.dataset.state === 'evaluating')
            assert.match(await page.locator('#season-indicator').innerText(), /Updating/)
            releaseEvaluate()
            await waitAppSettled(app)
            await page.waitForFunction(() => document.querySelector('#season-indicator')?.dataset.state === 'idle')
            assert.equal(await readIndicatorState(), 'idle')
            await page.unroute(url => url.pathname.endsWith('/evaluate'))
            expectCleanSession(app, 'season indicator')
        })
    } finally {
        await app.close()
    }
})
