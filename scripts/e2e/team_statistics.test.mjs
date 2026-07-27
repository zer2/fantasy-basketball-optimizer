// scripts/e2e/team_statistics.test.mjs
// The "Show team statistics" view in Draft Mode: one G-score row per player the seat has
// drafted, a totals row that actually sums them, tracking of roster changes, and the
// H-score (est. win rate) row once the roster is full. A 2-drafter league keeps a full
// 13-round draft affordable (autodraft fills the other seat).
// Uses the 2024-25 historical season for stable data.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, selectHistoricalSeason, expectCleanSession, waitAppSettled,
    lockInTopDraftPick, pickControlButton, setLeagueDrafterCount,
} from './helpers.mjs'

test('draft team statistics track the roster', async t => {
    const app = await launchAppPage()
    const { page } = app

    // Clicking the tab re-renders the table from current state — a deterministic refresh.
    async function openTeamStatisticsTab() {
        await page.locator('.draft-tab-bar .pick-btn[data-tab="my-team"]').click()
        await page.locator('[data-testid="team-gscore"]').waitFor({ timeout: 30000 })
    }

    /** Team 1's players as shown on the board (column 1 of the entry grid). */
    function readTeamOneBoardPlayers() {
        return page.evaluate(() =>
            [...document.querySelectorAll('.entry-table tbody tr')]
                .map(row => row.cells[1]?.textContent ?? '')
                .filter(text => text !== ''))
    }

    /** The team table's rows as { label, total } (the totals row included). Each row is a
     *  th label followed by the Total cell — row.cells counts the th, so Total is cells[1]. */
    function readTeamTableRows() {
        return page.evaluate(() =>
            [...document.querySelectorAll('[data-testid="team-gscore"] tbody tr')].map(row => ({
                label: row.querySelector('th')?.textContent ?? '',
                total: parseFloat(row.cells[1]?.textContent ?? ''),
            })))
    }

    try {
        await loadApp(app)
        await selectHistoricalSeason(app, '2024-25')
        await setLeagueDrafterCount(app, 2)
        expectCleanSession(app, 'initial load')

        await t.test('one row per drafted player, and the totals row sums them', async () => {
            for (let lockCount = 0; lockCount < 4; lockCount++) await lockInTopDraftPick(app)
            await openTeamStatisticsTab()

            const boardPlayers = await readTeamOneBoardPlayers()
            assert.equal(boardPlayers.length, 2, 'Team 1 should hold two picks after two rounds')

            const tableRows = await readTeamTableRows()
            assert.equal(tableRows.length, 3, 'two player rows plus the Team Total row')
            for (const player of boardPlayers) {
                assert.ok(tableRows.some(row => row.label === player),
                          `drafted player ${player} should have a row in the team table`)
            }

            const totalsRow = tableRows.find(row => row.label === 'Team Total')
            assert.ok(totalsRow, 'the team table should end with a Team Total row')
            const playerTotalSum = tableRows
                .filter(row => row.label !== 'Team Total')
                .reduce((sum, row) => sum + row.total, 0)
            assert.ok(Math.abs(playerTotalSum - totalsRow.total) < 0.05,
                      `Team Total (${totalsRow.total}) should equal the sum of player totals (${playerTotalSum.toFixed(2)})`)
            expectCleanSession(app, 'team table after two rounds')
        })

        await t.test('the table tracks roster changes', async () => {
            await pickControlButton(page, 'Undo previous selection').click()
            await waitAppSettled(app)
            await openTeamStatisticsTab()

            const tableRows = await readTeamTableRows()
            assert.equal(tableRows.length, 2, 'one player row plus the Team Total row after the undo')
            expectCleanSession(app, 'team table after undo')
        })

        await t.test('a full roster shows the H-score row', async () => {
            // Fill Team 1's remaining twelve rounds; autodraft handles Team 2's picks.
            await page.locator('.entry-table thead .method-dd-trigger').nth(1).click()
            for (let lockCount = 0; lockCount < 12; lockCount++) await lockInTopDraftPick(app)

            assert.match(await page.locator('.pick-control-row .pick-control-label').first().textContent(),
                         /Draft complete/, 'the draft should run to completion')

            await openTeamStatisticsTab()
            const tableRows = await readTeamTableRows()
            assert.equal(tableRows.length, 14, '13 player rows plus the Team Total row')
            await page.locator('#draft-gscore', { hasText: 'H-Score (est. win rate)' })
                .waitFor({ timeout: 30000 })
            expectCleanSession(app, 'full-roster team statistics')
        })
    } finally {
        await app.close()
    }
})
