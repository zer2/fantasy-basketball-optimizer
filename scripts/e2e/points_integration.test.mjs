// Points UI contracts use fixed platform/evaluation fixtures; the real API builds sessions.
// Live numerical validation is recorded separately in the PR's local validation report.
import { test } from 'node:test'
import assert from 'node:assert/strict'
import { launchAppPage, loadApp, setSelect, waitAppSettled, expectCleanSession } from './helpers.mjs'

test('Points scoring controls, native rules, reconnect, and category restoration', async t => {
    const app = await launchAppPage()
    const { page } = app
    const patches=[]
    try {
        await loadApp(app)
        await setSelect(page,'ls-mode','Season Mode')
        await waitAppSettled(app)
        page.on('request',r=>{if(r.method()==='PATCH'&&new URL(r.url()).pathname.startsWith('/sessions/'))patches.push(r.postDataJSON())})
        await t.test('a category conversion requires an explicit scoring choice',async()=>{
            const count=patches.length
            await setSelect(page,'fc-scoring-format','Points League')
            await page.waitForTimeout(1100)
            assert.match(await page.locator('#fc-points-error').innerText(),/Choose/)
            assert.equal(patches.length,count)
            await setSelect(page,'fc-points-preset','ESPN standard')
            await page.waitForTimeout(1100);await waitAppSettled(app)
            assert.equal(patches.at(-1).league.points_config.scoring_rules.TO,-2)
            expectCleanSession(app,'Points preset')
        })
        await t.test('editing a weight selects Custom and invalid counts block requests',async()=>{
            await page.locator('#fc-points-weight-PTS').fill('2')
            await page.locator('#fc-points-weight-PTS').press('Tab')
            await page.waitForTimeout(1100);await waitAppSettled(app)
            assert.equal(await page.locator('#fc-points-preset').inputValue(),'custom')
            assert.equal(patches.at(-1).league.points_config.scoring_rules.PTS,2)
            const count=patches.length
            await page.locator('#fc-points-bench').fill('100')
            await page.locator('#fc-points-bench').press('Tab')
            await page.waitForTimeout(1100)
            assert.match(await page.locator('#fc-points-error').innerText(),/exceed/)
            assert.equal(patches.length,count)
            await page.locator('#fc-points-bench').fill('0')
            await page.locator('#fc-points-bench').press('Tab')
            await page.waitForTimeout(1100);await waitAppSettled(app)
            expectCleanSession(app,'custom scoring')
        })
        const ids=await page.evaluate(async()=> (await import('/dist/player_registry.js')).getAllPlayerIdentities().filter(p=>p.player_id>0&&p.player_id!==201939&&p.positions.length).slice(0,28).map(p=>p.player_id))
        const assignments={Ann:ids.slice(0,14),Bob:ids.slice(14,28)}
        await page.route(url=>url.pathname.endsWith('/connect'),route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({
            team_names:['Ann','Bob'],n_drafters:2,n_picks:14,available_modes:['Season Mode'],is_auction_draft:false,
            slot_counts:{PG:1,SG:1,SF:1,PF:1,C:1,G:1,F:1,Util:7},points_metadata:{scoring_system:'points',scoring_rules:{PTS:1,TO:-3},
                active_slots:{PG:1,SG:1,SF:1,PF:1,C:1,G:1,F:1,Util:3},bench_slots:4,ir_slots:3,lineup_lock_mode:'pregame_daily'},
        })}))
        await page.route(url=>/^\/sessions(\/[^/]+)?$/.test(url.pathname),async route=>{
            if(!['POST','PATCH'].includes(route.request().method()))return route.continue()
            const body=route.request().postDataJSON()
            delete body.platform;delete body.platform_config
            const response=await route.fetch({postData:JSON.stringify(body)})
            await route.fulfill({response})
        })
        await page.route(url=>url.pathname.endsWith('/draft-state'),route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({
            player_assignments:assignments,injured_players:[],unmatched_player_names:{},empty_slots:{},remaining_cash:null,status:'Success',
        })}))
        await page.route(url=>url.pathname.endsWith('/evaluate'),async route=>{
            if(await page.locator('#fc-scoring-format').inputValue()!=='Points League')return route.continue()
            const req=route.request().postDataJSON()
            await route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({
                scoring_system:'points',roster_fp:120,has_more:false,total_candidates:1,warnings:[],
                candidates:req.candidate_limit===0?[]:[{player_id:201939,rank:1,projected_fp_per_game:0,projected_fp_total:0,
                    marginal_fp:0,delta_fp:0,candidate_games:0,dropped_games:0,daily_deltas:{'2026-10-22':0}}],
            })})
        })
        await t.test('native Points rules import starters separately from bench and survive reconnect',async()=>{
            await setSelect(page,'ls-platform','Retrieve from Yahoo')
            await waitAppSettled(app)
            const id=page.locator('#ls-yahoo-league-id')
            await id.evaluate(el=>{const d=el.closest('details');if(d)d.open=true})
            await id.fill('points-fixture');await id.press('Enter')
            await page.waitForFunction(()=>document.querySelector('#ls-n-drafters')?.value==='2')
            await page.keyboard.press('Escape');await waitAppSettled(app)
            assert.equal(await page.locator('#fc-points-preset').inputValue(),'native')
            assert.equal(await page.locator('#fc-points-weight-TO').inputValue(),'-3')
            assert.equal(await page.locator('#fc-points-slot-Util').inputValue(),'3')
            assert.equal(await page.locator('#fc-points-bench').inputValue(),'4')
            await page.locator('.season-tab-btn[data-tab="waiver"]').click()
            await waitAppSettled(app)
            const values=page.locator('#hscoretable .points-value')
            await values.first().waitFor()
            assert.equal(await values.first().innerText(),'0.0')
            assert.ok(await values.evaluateAll(cells=>cells.every(c=>!getComputedStyle(c,'::after').content.includes('%'))))
            await page.locator('#hscoretable .playerpopup').first().click()
            await page.locator('[data-testid="points-candidate-details"]').waitFor()
            expectCleanSession(app,'native Points connection')
        })
        await t.test('custom rules remain scoped to the connected league',async()=>{
            await page.locator('#fc-points-weight-PTS').fill('2');await page.locator('#fc-points-weight-PTS').press('Tab')
            await page.waitForTimeout(1100);await waitAppSettled(app)
            const id=page.locator('#ls-yahoo-league-id')
            await id.evaluate(el=>{const d=el.closest('details');if(d)d.open=true})
            await id.fill('another-points-fixture');await id.press('Enter');await waitAppSettled(app)
            assert.equal(await page.locator('#fc-points-preset').inputValue(),'native')
            assert.equal(await page.locator('#fc-points-weight-PTS').inputValue(),'1')
            await id.fill('points-fixture');await id.press('Enter');await waitAppSettled(app)
            assert.equal(await page.locator('#fc-points-preset').inputValue(),'custom')
            assert.equal(await page.locator('#fc-points-weight-PTS').inputValue(),'2')
            expectCleanSession(app,'league-specific Points preferences')
        })
        await t.test('category mode is restored after a Points-only build',async()=>{
            await setSelect(page,'fc-scoring-format','Head to Head')
            await page.waitForTimeout(1100);await waitAppSettled(app)
            await page.locator('.season-tab-btn[data-tab="rosters"]').click()
            await waitAppSettled(app)
            await page.locator('[data-testid="roster-inspection-gscore"]').waitFor()
            expectCleanSession(app,'category restoration')
        })
    } finally { await app.close() }
})
