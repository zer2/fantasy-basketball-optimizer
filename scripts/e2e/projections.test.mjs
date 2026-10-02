// scripts/e2e/projections.test.mjs
// Projection blend weights: changing a weight must re-weight the SAME player pool —
// the draft board survives it (only pool-identity changes reset the boards), and the
// results must actually move (regression for the v0 cache serving a stale blend when
// a weight excluded from its key changed — no error anywhere, just identical output).

import { test } from 'node:test'
import assert from 'node:assert/strict'
import {
    launchAppPage, loadApp, expectCleanSession, drainSessionFailures, waitAppSettled,
    setSelect, lockInDraftPick, pickControlButton, countBoardCellsWithPlayer,
} from './helpers.mjs'

test('projection blend weights', async t => {
    const app = await launchAppPage()
    const { page } = app

    async function setBlendWeight(inputId, value) {
        const weightInput = page.locator(`#${inputId}`)
        await weightInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
        await weightInput.fill(String(value))
        await weightInput.evaluate(el => el.blur())
        await waitAppSettled(app, { timeout: 120000 })
    }

    /** Snapshot of the top candidates as (name, h_score) pairs. */
    function readCandidateSnapshot() {
        return page.evaluate(() =>
            [...document.querySelectorAll('#hscoretable tbody tr')].slice(0, 50)
                .map(row => `${row.querySelector('.playername')?.textContent}:${row.querySelector('.overallhscore')?.textContent}`)
                .filter(entry => !entry.startsWith('undefined')))
    }

    try {
        await loadApp(app)
        await setSelect(page, 'ps-data-type', 'Projections')
        await waitAppSettled(app, { timeout: 120000 })
        assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                  'projections mode should render candidates')
        expectCleanSession(app, 'switch to projections')

        await t.test('a blend-weight change keeps the draft board intact', async () => {
            await lockInDraftPick(page, 'Nikola Jokic')
            await waitAppSettled(app)
            assert.equal(await countBoardCellsWithPlayer(page, 'Nikola Jokic'), 1)

            await setBlendWeight('ps-w-darko', 0.9)

            assert.equal(await countBoardCellsWithPlayer(page, 'Nikola Jokic'), 1,
                         'the locked pick must survive a blend-weight change')
            assert.match(await page.locator('.pick-control-row .pick-control-label').first().textContent(),
                         /Select Pick 1 for Team 2/, 'the pick position must survive too')
            expectCleanSession(app, 'weight change with a live board')

            await pickControlButton(page, 'Clear draft board').click()
            await waitAppSettled(app)
            expectCleanSession(app, 'board cleared')
        })

        await t.test('blend weights actually change the results', async () => {
            // Order matters within each transition: raise the new source before zeroing
            // the old one, so the blend never passes through an all-zero state.
            await setBlendWeight('ps-w-espn', 1)
            await setBlendWeight('ps-w-darko', 0)
            const espnOnlySnapshot = await readCandidateSnapshot()
            assert.ok(espnOnlySnapshot.length > 10, 'the ESPN-only blend should render candidates')
            expectCleanSession(app, 'ESPN-only blend')

            await setBlendWeight('ps-w-darko', 1)
            await setBlendWeight('ps-w-espn', 0)
            const darkoOnlySnapshot = await readCandidateSnapshot()
            assert.ok(darkoOnlySnapshot.length > 10, 'the DARKO-only blend should render candidates')
            expectCleanSession(app, 'DARKO-only blend')

            assert.notDeepEqual(darkoOnlySnapshot, espnOnlySnapshot,
                                'opposite blends must produce different candidate results')
        })

        await t.test('all-zero blend weights are rejected clearly and the app recovers', async () => {
            await setBlendWeight('ps-w-darko', 0)   // ESPN is already 0 — blend is now empty

            const { failures } = drainSessionFailures(app)
            assert.ok(failures.length > 0, 'an empty blend should be rejected by the backend')
            assert.ok(failures.some(failure => failure.startsWith('400') && failure.includes('blend weights')),
                      `the rejection should be a clear 400, not an opaque 500 — saw: ${failures.join(' | ')}`)

            await setBlendWeight('ps-w-espn', 0.5)
            await setBlendWeight('ps-w-darko', 0.5)
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'restoring the weights should recover fully')
            expectCleanSession(app, 'recovered from empty blend')
        })

        await t.test('an unreadable upload is rejected at upload time, visibly', async () => {
            const firstSlider = page.locator('#ps-w-custom-1')
            assert.ok(await firstSlider.isDisabled(), 'a custom weight starts locked at zero')
            assert.equal(await page.locator('#ps-upload-custom-1 ~ .sidebar-file-name').textContent(),
                         'No file chosen', 'an empty custom slot should say so where the filename goes')

            const uploadInput = page.locator('#ps-upload-custom-1')
            await uploadInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            // Headers the alias table cannot interpret at all. (A file with only a few
            // recognizable stats is NOT rejected — sparse projection sets are legitimate.)
            await uploadInput.setInputFiles({
                name: 'not-projections.csv',
                mimeType: 'text/csv',
                buffer: Buffer.from('Ticker,Sector,Close,Volume\nABC,Tech,101.5,20000\n'),
            })

            const uploadStatus = page.locator('#ps-upload-custom-1 ~ .sidebar-caption')
            await uploadStatus.filter({ hasText: 'Upload failed' }).waitFor({ timeout: 15000 })
            assert.match(await uploadStatus.textContent(), /does not read as a projection set/,
                         'the status should say WHY the file was rejected')
            await waitAppSettled(app)
            assert.ok(await firstSlider.isDisabled(), 'a rejected upload must leave the weight locked')
            assert.equal(await page.locator('#ps-w-custom-2').count(), 0,
                         'a failed upload must not open the next slot')

            const { errors } = drainSessionFailures(app)
            assert.ok(errors.some(error => error.includes('upload failed')),
                      'the rejection is an expected failure, logged for diagnosis')
            expectCleanSession(app, 'after rejected upload')
        })

        await t.test('a well-formed upload is auto-detected, named, and changes results', async () => {
            // Build a valid projection CSV from real pool players with uniform stat lines —
            // blended in at full weight, it must visibly move the results.
            // Candidate rows render the rich display: the bare name is the .playername
            // span's leading text node; positions ride in the nested .player-positions span.
            const pooledPlayers = await page.evaluate(() =>
                [...document.querySelectorAll('#hscoretable .playername')].slice(0, 30)
                    .map(span => ({
                        name:     span.childNodes[0].textContent.trim(),
                        position: span.querySelector('.player-positions')?.textContent ?? '',
                    })))
            // Includes unmapped junk columns (Rank, Value) like real exports carry — the
            // parser must drop them; historically one junk column in an upload silently
            // wiped the entire pool ("0 players available") at ANY weight.
            const csvRows = pooledPlayers.map(({ name, position }, playerIndex) =>
                // Multi-position values contain commas (e.g. "C,PF") — quote them, as real exports do.
                `${playerIndex + 1},${name},"${position}",9.9,70,20.0,8.0,4.0,1.0,1.0,2.0,2.0,0.5,15.0,0.8,5.0`)
            const csvText = 'Rank,Name,Pos,Value,g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n' + csvRows.join('\n')

            const beforeUploadSnapshot = await readCandidateSnapshot()
            await page.locator('#ps-upload-custom-1').setInputFiles({
                name: 'generated-projections.csv', mimeType: 'text/csv', buffer: Buffer.from(csvText),
            })
            const uploadStatus = page.locator('#ps-upload-custom-1 ~ .sidebar-caption')
            await uploadStatus.filter({ hasText: 'players loaded' }).waitFor({ timeout: 15000 })
            await waitAppSettled(app)
            assert.match(await uploadStatus.textContent(), /\d+ players loaded/,
                         'the status should report how many players were read')
            assert.ok(!(await page.locator('#ps-w-custom-1').isDisabled()),
                      'a successful upload should unlock the weight')
            assert.equal(await page.locator('#ps-w-custom-2').count(), 1,
                         'a successful upload should open the next empty slot')
            assert.ok(await page.locator('#ps-w-custom-2').isDisabled(),
                      'the next slot starts locked')
            expectCleanSession(app, 'valid upload accepted')

            // The slot is identified by the file behind it. The browser's own file-input text
            // cannot be read back or restored, so the app renders the name itself.
            assert.equal(await page.locator('#ps-upload-custom-1 ~ .sidebar-file-name').textContent(),
                         'generated-projections.csv', 'the slot should name the file it holds')

            await setBlendWeight('ps-w-custom-1', 1)
            const withUploadSnapshot = await readCandidateSnapshot()
            assert.notDeepEqual(withUploadSnapshot, beforeUploadSnapshot,
                                'blending the uploaded projections must change the results')
            expectCleanSession(app, 'uploaded blend evaluated')

            // Back to zero with the upload still attached: the upload must drop out of the
            // blend entirely (historically it kept participating and could 400 every patch).
            await setBlendWeight('ps-w-custom-1', 0)
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'a zero-weight upload must leave the blend fully working')
            expectCleanSession(app, 'upload weight back to zero')

            // Uploads survive a reload: the file is kept server-side on a day-long clock that
            // resets whenever a session uses it, and the slot it fills (id, filename, weight,
            // status line) is remembered in preferences.
            await loadApp(app)
            assert.equal(await page.locator('#ps-upload-custom-1 ~ .sidebar-file-name').textContent(),
                         'generated-projections.csv',
                         'the restored source should come back under the filename it was uploaded as')
            assert.ok(!(await page.locator('#ps-w-custom-1').isDisabled()),
                      'a restored upload keeps its weight unlocked — the file is still there')
            assert.equal(await page.locator('#ps-w-custom-1').inputValue(), '0',
                         'the restored weight is the one last set, not a default')
            assert.equal(await page.locator('#ps-w-custom-2').count(), 1,
                         'the empty next slot is still open after the restore')
            assert.ok(await page.locator('#hscoretable .playerheaderdiv').count() > 0,
                      'the reloaded page should evaluate cleanly with the restored upload')
            expectCleanSession(app, 'reload with the restored upload')
        })

        await t.test('re-uploading an edited file under the same name refreshes the data', async () => {
            // The regression: a user uploads a file missing a stat, adds the stat, and uploads the
            // file again under the same name. Two things kept the old data. The file input kept the
            // file after a successful upload, so the browser fired no change event for the same
            // name again; and when an upload did fire, its change event rebuilt the session at once,
            // while the upload was still in flight, with the slot's PREVIOUS data id. (Playwright's
            // setInputFiles dispatches change itself, so the first cause is checked by its
            // mechanism: the input must be empty after an upload.)
            const pooledPlayers = await page.evaluate(() =>
                [...document.querySelectorAll('#hscoretable .playername')].slice(0, 30)
                    .map(span => ({
                        name:     span.childNodes[0].textContent.trim(),
                        position: span.querySelector('.player-positions')?.textContent ?? '',
                    })))
            const withoutBlocks = 'Name,Pos,g,p/g,r/g,a/g,s/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
                + pooledPlayers.map(({ name, position }) =>
                    `${name},"${position}",70,20.0,8.0,4.0,1.0,2.0,2.0,0.5,15.0,0.8,5.0`).join('\n')
            const withBlocks = 'Name,Pos,g,p/g,r/g,a/g,s/g,b/g,to/g,3/g,fg%,fga/g,ft%,fta/g\n'
                + pooledPlayers.map(({ name, position }) =>
                    `${name},"${position}",70,25.0,9.0,5.0,1.5,2.5,2.0,2.5,0.5,18.0,0.8,6.0`).join('\n')

            const sentDataIds = []
            page.on('request', request => {
                if (!request.url().includes('/sessions') || !['POST', 'PATCH'].includes(request.method())) return
                const ids = request.postDataJSON()?.data_source?.custom_data_ids
                if (Array.isArray(ids)) sentDataIds.push(ids)
            })
            const uploadInput = page.locator('#ps-upload-custom-1')
            const uploadStatus = page.locator('#ps-upload-custom-1 ~ .sidebar-caption')
            // the reload in the previous test collapses the Player Stats section, hiding the status line
            await uploadInput.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
            async function uploadAs(csvText) {
                const responsePromise = page.waitForResponse(r => r.url().includes('/data/upload'))
                await uploadInput.setInputFiles({ name: 'edited-projections.csv', mimeType: 'text/csv', buffer: Buffer.from(csvText) })
                const response = await responsePromise
                assert.ok(response.ok(), `the upload was rejected: ${response.status()} ${await response.text()}`)
                const dataId = (await response.json()).data_id
                await uploadStatus.filter({ hasText: 'players loaded' }).waitFor({ timeout: 15000 })
                await waitAppSettled(app, { timeout: 120000 })
                return dataId
            }

            const firstId = await uploadAs(withoutBlocks)
            assert.match(await uploadStatus.textContent(), /\(no [^)]*\)/, 'the first file is missing a stat, and says so')
            assert.equal(await uploadInput.inputValue(), '',
                         'after an upload the input must be empty, or choosing the same file again fires nothing')
            await setBlendWeight('ps-w-custom-1', 1)
            const firstSnapshot = await readCandidateSnapshot()

            sentDataIds.length = 0
            const secondId = await uploadAs(withBlocks)
            assert.notEqual(secondId, firstId, 'each upload is stored under its own id')
            assert.doesNotMatch(await uploadStatus.textContent(), /\(no [^)]*\)/,
                                'the edited file has every stat, so the missing-stats note must go')
            assert.ok(sentDataIds.length > 0, 'the re-upload must rebuild the session')
            assert.ok(sentDataIds.every(ids => ids.includes(secondId) && !ids.includes(firstId)),
                      `every rebuild after the re-upload must carry the new id, never the old one — sent: ${JSON.stringify(sentDataIds)}`)
            assert.notDeepEqual(await readCandidateSnapshot(), firstSnapshot,
                                'the edited projections must change the results')
            expectCleanSession(app, 're-upload of an edited file')

            await setBlendWeight('ps-w-custom-1', 0)
        })
    } finally {
        await app.close()
    }
})
