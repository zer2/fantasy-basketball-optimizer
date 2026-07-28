// scripts/e2e/season_tools.test.mjs
// Season Mode's tools: roster-grid editing (paste + in-cell search), the roster
// inspector, the waiver wire, and trade analysis/suggestions.
//
// The tests build their OWN roster dataset: player names are read from the live pool
// and pasted into the grid, so nothing here depends on the app's built-in default
// rosters (which are slated for removal). A 2-drafter league keeps evaluates fast.
// Uses the 2025-26 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    setSelect, setLeagueDrafterCount, readDropdownOptionLabels,
} from './helpers.mjs'

const TEAM_COUNT = 2
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

    /** The inspector's G-score table rows (player labels only, totals row excluded). */
    function readInspectorPlayers() {
        return page.evaluate(() =>
            [...document.querySelectorAll('[data-testid="roster-inspection-gscore"] tbody tr')]
                .map(row => row.querySelector('th')?.textContent ?? '')
                .filter(label => label !== 'Team Total'))
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2025-26')
        await setLeagueDrafterCount(app, TEAM_COUNT)

        // Build the dataset from the live pool: the draft pick dropdown lists every player.
        const playerPool = await readDropdownOptionLabels(page, 'draft-pick-select-wrapper')
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
    } finally {
        await app.close()
    }
})
