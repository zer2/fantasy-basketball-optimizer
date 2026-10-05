// scripts/e2e/candidate_search.test.mjs
// The player search in the top row narrows the candidate table by name (case and accents ignored), keeps its
// search through a re-ranking, and restores every candidate when cleared.

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { launchAppPage, loadApp, expectCleanSession, waitAppSettled, lockInTopDraftPick } from './helpers.mjs'

const candidateNames = page => page.locator('#hscoretable .playerheaderdiv .playername').allInnerTexts()

test('the player search narrows the candidate table', async t => {
    const app = await launchAppPage()
    const { page } = app
    const search = page.locator('#candidate-search')
    try {
        await loadApp(app)
        const everyone = (await candidateNames(page)).length

        await t.test('typing a name without its accent finds the player', async () => {
            assert.ok(await search.isVisible(), 'the search box shows beside the candidate table')
            await search.fill('jokic')
            const names = await candidateNames(page)
            assert.ok(names.length >= 1, 'Jokić must be found')
            assert.ok(names.every(name => name.normalize('NFD').replace(/\p{Diacritic}/gu, '').toLowerCase().includes('jokic')),
                      `only matching players may show, saw ${JSON.stringify(names)}`)
            expectCleanSession(app, 'searched')
        })

        await t.test('the search survives the table being re-ranked', async () => {
            await search.fill('a')
            const before = (await candidateNames(page)).length
            await lockInTopDraftPick(app)
            await waitAppSettled(app, { timeout: 120000 })
            const after = await candidateNames(page)
            assert.equal(await search.inputValue(), 'a')
            assert.ok(after.every(name => name.toLowerCase().includes('a')), 'the re-ranked table must still be filtered')
            assert.ok(after.length <= before, 'a pick can only remove matching candidates')
            expectCleanSession(app, 're-ranked while searching')
        })

        await t.test('clearing the search restores every candidate', async () => {
            await search.fill('')
            assert.ok((await candidateNames(page)).length >= Math.min(everyone, 20) - 1)
            expectCleanSession(app, 'search cleared')
        })
    } finally {
        await app.close()
    }
})
