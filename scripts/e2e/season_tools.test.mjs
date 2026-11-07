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

        await t.test('suggested trades are cached: asked for again only when the session or an input changes', async () => {
            // Every visit to the Trading tab used to re-run the whole search (2026-11-06).
            let suggestRequests = 0
            const countSuggestRequest = request => { if (request.url().includes('/trade/suggest')) suggestRequests += 1 }
            page.on('request', countSuggestRequest)
            const rows = page.locator('[data-testid="trade-suggestions"] tbody tr')
            const readRowCount = () => rows.count()
            /** How many requests an action took, once the app has settled. */
            async function countRequestsDuring(action) {
                suggestRequests = 0
                await action()
                await waitAppSettled(app)
                return suggestRequests
            }
            await openSeasonTab('trading')
            await setSelect(page, 'trade-your-team', 'Team 1')
            for (const id of ['ts-your-threshold', 'ts-their-threshold']) {
                await page.locator(`#${id}`).fill('-100')
                await page.locator(`#${id}`).blur()
            }
            await waitAppSettled(app)
            const rowsFound = await readRowCount()
            assert.ok(rowsFound > 0, 'the search finds trades')

            // Back to the tab: nothing asked for, and the rows are there at once, not after a search.
            assert.equal(await countRequestsDuring(async () => {
                await page.locator('.season-tab-btn[data-tab="waiver"]').click()
                await waitAppSettled(app)
                await page.locator('.season-tab-btn[data-tab="trading"]').click()
                assert.equal(await readRowCount(), rowsFound, 'the rows are drawn immediately')
            }), 0, 'switching back asks for nothing')

            // What the search boxes hold survives the switch.
            await page.locator('#trade-team-search').fill('Team 3')
            await openSeasonTab('waiver')
            await openSeasonTab('trading')
            assert.equal(await page.locator('#trade-team-search').inputValue(), 'Team 3', 'the search box is kept')
            await page.locator('#trade-team-search').fill('')
            await waitAppSettled(app)

            // A threshold only filters.
            assert.equal(await countRequestsDuring(async () => {
                await page.locator('#ts-your-threshold').fill('-99')
                await page.locator('#ts-your-threshold').blur()
            }), 0, 'a threshold asks for nothing')

            // Your team is an input; going back to one already searched asks for nothing.
            assert.ok(await countRequestsDuring(() => setSelect(page, 'trade-your-team', 'Team 2')) > 0,
                      'another team searches')
            assert.equal(await countRequestsDuring(() => setSelect(page, 'trade-your-team', 'Team 1')), 0,
                         'a team already searched asks for nothing')

            // The position check is an input.
            assert.ok(await countRequestsDuring(() => page.locator('#ts-check-positions').evaluate(toggle => toggle.click())) > 0,
                      'the position check searches')
            assert.equal(await countRequestsDuring(() => page.locator('#ts-check-positions').evaluate(toggle => toggle.click())), 0,
                         'its earlier value is already searched')

            // A session change rebuilds the model under the same session id: nothing found before it may be reused.
            assert.ok(await countRequestsDuring(async () => {
                await setObjectiveWeight(page, 0.5)
                await waitAppSettled(app)
                await openSeasonTab('waiver')
                await openSeasonTab('trading')
            }) > 0, 'a session change searches again')

            // Leaving mid-search neither restarts it nor runs it twice: one visit's requests, however many visits.
            const oneSearch = await countRequestsDuring(async () => {
                await setObjectiveWeight(page, 0.25)
                await waitAppSettled(app)
                await openSeasonTab('waiver')
                await openSeasonTab('trading')
            })
            assert.equal(await countRequestsDuring(async () => {
                await setObjectiveWeight(page, 0)
                await waitAppSettled(app)
                await page.locator('.season-tab-btn[data-tab="waiver"]').click()
                await page.locator('.season-tab-btn[data-tab="trading"]').click()
                await page.locator('.season-tab-btn[data-tab="waiver"]').click()
                await page.locator('.season-tab-btn[data-tab="trading"]').click()
            }), oneSearch, 'switching during a search does not repeat its requests')

            // The rosters are an input: a swap between two teams searches; swapping back is already searched.
            async function pasteRosters(rosters) {
                await openSeasonTab('rosters')
                const pasteBlock = Array.from({ length: PICKS_PER_TEAM }, (_, row) =>
                    rosters.map(roster => roster[row]).join('\t')).join('\n')
                await page.evaluate(text => navigator.clipboard.writeText(text), pasteBlock)
                await page.locator('[data-testid="sr-player-0-0-wrapper"]').click()
                await page.keyboard.press('Control+v')
                await waitAppSettled(app)
                await openSeasonTab('trading')
            }
            await pasteRosters(teamRosters)   // the baseline: an earlier subtest edited a cell by hand
            const swapped = teamRosters.map(roster => [...roster])
            ;[swapped[0][0], swapped[1][0]] = [swapped[1][0], swapped[0][0]]
            assert.ok(await countRequestsDuring(() => pasteRosters(swapped)) > 0, 'changed rosters search')
            assert.equal(await countRequestsDuring(() => pasteRosters(teamRosters)), 0,
                         'rosters already searched ask for nothing')

            page.off('request', countSuggestRequest)
            expectCleanSession(app, 'suggestion cache')
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
            await page.waitForFunction(() => document.querySelector('#season-indicator')?.textContent === 'Updating...')
            releaseEvaluate()
            await waitAppSettled(app)
            await page.waitForFunction(() => document.querySelector('#season-indicator')?.dataset.state === 'idle')
            assert.equal(await readIndicatorState(), 'idle')
            await page.unroute(url => url.pathname.endsWith('/evaluate'))
            expectCleanSession(app, 'season indicator')
        })

        await t.test('neither short work nor the gaps in a series of requests flicker the indicator', async () => {
            // Opening the Rosters tab sends one quick evaluate, which blinked "Updated" to "Updating..." and back.
            const shownStates = await page.evaluate(async () => {
                const { applyIndicatorState } = await import('/dist/api/session.js')
                const indicator = document.getElementById('season-indicator')
                const read = () => ({ state: indicator.dataset.state, label: indicator.textContent })
                const wait = milliseconds => new Promise(resolve => setTimeout(resolve, milliseconds))
                const seen = {}
                applyIndicatorState('season-indicator', 'evaluating')
                seen.atOnce = read()
                await wait(50)
                applyIndicatorState('season-indicator', 'idle')
                await wait(300)
                seen.afterShortWork = read()
                applyIndicatorState('season-indicator', 'evaluating')
                await wait(300)
                seen.duringLongWork = read()
                // A gap between two requests of a series does not show "Updated" either.
                applyIndicatorState('season-indicator', 'idle')
                seen.inTheGap = read()
                await wait(50)
                applyIndicatorState('season-indicator', 'evaluating')
                await wait(300)
                seen.afterTheGap = read()
                applyIndicatorState('season-indicator', 'idle')
                await wait(300)
                seen.afterLongWork = read()
                return seen
            })
            assert.deepEqual(shownStates.atOnce, { state: 'evaluating', label: 'Updated' }, 'the state is immediate, the label waits')
            assert.deepEqual(shownStates.afterShortWork, { state: 'idle', label: 'Updated' }, 'short work never shows')
            assert.deepEqual(shownStates.duringLongWork, { state: 'evaluating', label: 'Updating...' }, 'long work shows')
            assert.deepEqual(shownStates.inTheGap, { state: 'idle', label: 'Updating...' }, 'a gap keeps the spinner')
            assert.deepEqual(shownStates.afterTheGap, { state: 'evaluating', label: 'Updating...' }, 'work resumed')
            assert.deepEqual(shownStates.afterLongWork, { state: 'idle', label: 'Updated' }, 'a real end shows')
        })
    } finally {
        await app.close()
    }
})
