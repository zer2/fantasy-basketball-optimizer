// data_entry/season/season_waiver.ts
// Renders the Season → Waiver Wire tab controls.
// Shows team + drop-player selectors above the standard H-score candidate table.
// On selection, calls evaluate with the dropped player removed from their team,
// then highlights that player's row in the results.

import { makeCustomSelect } from '../../custom_select.js'
import { readRequiredIntInput } from '../../helper_functions.js'
import { readRosterAssignments, readEmptyRostersMessage } from './season_helpers.js'
import { getTeamIdentitiesFromSidebar } from '../../setting_collection/league_settings.js'
import { getPlayerResultsById, isPointsMode } from '../../app_state.js'
import { buildPlayerOption } from '../../player_display.js'
import { runWaiverEvaluate } from '../../api/season_session.js'
import { highlightCandidate } from '../../table/player_table.js'
import { pref, savePref } from '../../preferences.js'
import { optimizeStreamer } from '../../api/client.js'
import type { StreamerOptimizeResponse } from '../../types.js'
import { getSessionId } from '../../api/session.js'
import { getRegistryEntry } from '../../player_registry.js'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function renderStreamerResults(container: HTMLElement, res: StreamerOptimizeResponse): void {
    container.innerHTML = ''
    const plan = res.optimal_plan

    const grid = document.createElement('div')
    grid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 10px; margin-top: 8px;'

    const makeStatBox = (label: string, val: string, color: string) => {
        const box = document.createElement('div')
        box.style.cssText = 'padding: 8px 12px; background: rgba(0,0,0,0.15); border-radius: 4px; border-left: 3px solid ' + color + ';'
        box.innerHTML = `
            <div style="font-size: 0.68rem; color: #889; text-transform: uppercase; font-weight: 600;">${label}</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: ${color};">${val}</div>
        `
        return box
    }

    grid.append(
        makeStatBox('Recommended Strategy', plan.strategy || res.mode, '#63b3ed'),
        makeStatBox('Points Per Add', `+${res.points_per_add.toFixed(1)} FP`, '#48bb78'),
        makeStatBox('Net Delta FP', `+${res.total_net_delta_fp.toFixed(1)} FP`, '#38a169'),
        makeStatBox('Streamed Games', `${res.total_streamed_games} Games`, '#ecc94b'),
        makeStatBox('Adds Used', `${res.total_acquisitions_used} Adds`, '#b794f4'),
    )
    container.append(grid)

    if (plan.moves && plan.moves.length > 0) {
        const movesList = document.createElement('div')
        movesList.style.cssText = 'margin-top: 10px; font-size: 0.75rem; border-top: 1px solid light-dark(#e2e8f0, #2d3748); padding-top: 8px;'
        movesList.innerHTML = `<span style="font-weight:600; color:light-dark(#333,#dde);">Optimized Acquisition Schedule:</span> ` +
            plan.moves.map(m => {
                let addName = `Player ${m.add_player_id}`
                try { addName = getRegistryEntry(m.add_player_id).name } catch {}
                return `<span style="display:inline-block; margin: 2px 4px; padding: 2px 6px; background: rgba(49, 130, 206, 0.15); border-radius: 4px;">${m.date.slice(5)}: +${addName} (${m.games_gained}g, +${m.projected_fp.toFixed(1)} FP)</span>`
            }).join(' ')
        container.append(movesList)
    }

    if (res.daily_density_analysis && Object.keys(res.daily_density_analysis).length > 0) {
        const densityWrap = document.createElement('div')
        densityWrap.style.cssText = 'margin-top: 8px; font-size: 0.72rem; color: #889;'
        densityWrap.innerHTML = `<span>Schedule Density (Games Available): </span>` +
            Object.entries(res.daily_density_analysis).map(([d, count]) => `${d.slice(5)}: ${count}g`).join(' | ')
        container.append(densityWrap)
    }
}

/** Returns the id of the player with the lowest projected points per game. */
function lowestPointsPlayer(playerIds: number[]): number | null {
    const byId = getPlayerResultsById()
    let worst: number | null = null
    let lowestPpg = Infinity
    for (const playerId of playerIds) {
        const p = byId.get(playerId)
        if (p) {
            const ppg = p.points_per_game ?? (p.h_score || 0)
            if (ppg < lowestPpg) {
                lowestPpg = ppg
                worst = playerId
            }
        }
    }
    return worst
}

/** Returns the id of the player with the highest g_rank (worst) among the given ids.
 *  Returns null if none of the ids are found in the backend player data. */
function lowestGRankPlayer(playerIds: number[]): number | null {
    const byId = getPlayerResultsById()
    let worst: number | null = null
    let worstRank = -Infinity
    for (const playerId of playerIds) {
        const p = byId.get(playerId)
        if (p && p.g_rank > worstRank) {
            worstRank = p.g_rank
            worst = playerId
        }
    }
    return worst
}

/** Marks the dropped player's row in the candidate table (virtualization-aware: the candidate may be
 *  outside the current render window, so the mark is tracked by id and applied when its row renders). */
function highlightDropPlayer(dropPlayerId: number): void {
    highlightCandidate(dropPlayerId)
}

// Tracks the listeners attached by the most recent renderWaiverControls call so
// they can be detached before the next one. Without this, switching tabs in and
// out of Waiver would leave the previous render's two custom selects' internal
// listeners bound to detached nodes.
let waiverListenerController: AbortController | null = null

// ─── Main render ─────────────────────────────────────────────────────────────

/** Renders waiver wire controls (team + drop-player selectors) into the given container. */
export function renderWaiverControls(container: HTMLElement): void {
    waiverListenerController?.abort()
    waiverListenerController = new AbortController()

    const teamNames   = getTeamIdentitiesFromSidebar()
    const assignments = readRosterAssignments()
    const nPicks      = readRequiredIntInput('ls-n-picks')
    // assignments[t] can be undefined for team names beyond nDrafters, hence the optional chain
    const fullTeams   = teamNames.filter(t => (assignments[t]?.length ?? 0) >= nPicks)

    // The candidate table is this tab's results; with no team to score it would show headers over nothing. The table
    // is set either way here, since the Draft and Auction team-statistics tabs hide it.
    const candidateTable = document.getElementById('hscoretable')
    if (!candidateTable) throw new Error('Candidate table #hscoretable not found')
    candidateTable.style.display = fullTeams.length < 1 ? 'none' : ''

    if (fullTeams.length < 1) {
        const msg = document.createElement('p')
        msg.className   = 'coming-soon'
        msg.textContent = readEmptyRostersMessage('one full team', 'Waiver Wire')
        container.append(msg)
        return
    }

    // ── Team selector ────────────────────────────────────────────────────────

    const teamWrap = document.createElement('div')
    teamWrap.className = 'waiver-selector-row'

    const teamLabel = document.createElement('div')
    teamLabel.className   = 'pick-control-label'
    teamLabel.textContent = 'Which team do you want to drop a player from?'
    teamWrap.append(teamLabel)

    // Your team, shared with the Trading tab and remembered across reloads. Always opening on the first team scored
    // someone else's roster whenever the tab was drawn again (the rosters arriving redraw it), 2026-10-13. A
    // remembered team this league does not have falls back to the first full team.
    const savedTeam   = pref<string | null>('season-your-team', null)
    const initialTeam = savedTeam !== null && fullTeams.includes(savedTeam) ? savedTeam : fullTeams[0]
    const teamSel = makeCustomSelect(
        'waiver-team-select',
        fullTeams.map(n => ({ value: n, label: n })),
        initialTeam,
        undefined,
        waiverListenerController.signal,
    )
    teamWrap.append(teamSel.element)
    container.append(teamWrap)

    // ── Drop player selector ─────────────────────────────────────────────────

    const dropWrap = document.createElement('div')
    dropWrap.className = 'waiver-selector-row'

    const dropLabel = document.createElement('div')
    dropLabel.className   = 'pick-control-label'
    dropLabel.textContent = 'Which player are you considering dropping?'
    dropWrap.append(dropLabel)

    // teams in fullTeams are guaranteed to have entries in assignments
    const buildDropOptions = (team: string) => assignments[team].map(buildPlayerOption)

    const initialRoster  = assignments[initialTeam]
    // lowestGRankPlayer returns null if roster ids aren't in backend data yet
    const initialDefault = isPointsMode()
        ? (lowestPointsPlayer(initialRoster) ?? initialRoster[0])
        : (lowestGRankPlayer(initialRoster) ?? initialRoster[0])

    const dropSel = makeCustomSelect(
        'waiver-drop-select',
        buildDropOptions(initialTeam),
        String(initialDefault),
        undefined,
        waiverListenerController.signal,
    )
    dropWrap.append(dropSel.element)
    container.append(dropWrap)

    // ── Points Mode: Streamer Optimization Card ──────────────────────────────
    if (isPointsMode()) {
        const streamerCard = document.createElement('div')
        streamerCard.className = 'waiver-streamer-card'
        streamerCard.style.cssText = 'margin: 14px 0; padding: 12px 16px; background: light-dark(#f8fafc, #161a23); border: 1px solid light-dark(#e2e8f0, #2b3040); border-radius: 6px;'

        const streamerHeader = document.createElement('div')
        streamerHeader.style.cssText = 'display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'

        const streamerTitle = document.createElement('div')
        streamerTitle.style.cssText = 'font-weight: 700; font-size: 0.85rem; color: light-dark(#2d3748, #e2e8f0);'
        streamerTitle.innerHTML = 'Streamer Optimization Engine <span style="font-weight: normal; font-size: 0.75rem; color: #48bb78; margin-left: 6px;">● Points Mode Active</span>'

        const streamerControlsWrap = document.createElement('div')
        streamerControlsWrap.style.cssText = 'display: flex; align-items: center; gap: 10px;'

        const modeLabel = document.createElement('label')
        modeLabel.style.cssText = 'font-size: 0.75rem; color: #718096;'
        modeLabel.textContent = 'Strategy:'

        const modeSel = document.createElement('select')
        modeSel.className = 'custom-select'
        modeSel.style.cssText = 'padding: 4px 8px; font-size: 0.75rem; border-radius: 4px; background: light-dark(#fff, #1e2230); border: 1px solid light-dark(#cbd5e0, #394256); color: inherit;'
        modeSel.innerHTML = `
            <option value="auto">Auto (Best Yield)</option>
            <option value="milp">Rolling-Horizon MILP (OR-Tools)</option>
            <option value="tactical_multi">Tactical Hold (Multi-Game)</option>
            <option value="daily_churn">Daily Churn (Max Volume)</option>
        `

        const addsLabel = document.createElement('label')
        addsLabel.style.cssText = 'font-size: 0.75rem; color: #718096;'
        addsLabel.textContent = 'Adds/Wk:'

        const addsInput = document.createElement('input')
        addsInput.type = 'number'
        addsInput.min = '1'
        addsInput.max = '14'
        addsInput.value = '7'
        addsInput.style.cssText = 'width: 44px; padding: 4px 6px; font-size: 0.75rem; border-radius: 4px; background: light-dark(#fff, #1e2230); border: 1px solid light-dark(#cbd5e0, #394256); color: inherit;'

        const optBtn = document.createElement('button')
        optBtn.type = 'button'
        optBtn.className = 'trade-tab-btn active'
        optBtn.textContent = '⚡ Optimize Streamer'
        optBtn.style.cssText = 'cursor: pointer; padding: 4px 12px; font-size: 0.75rem; font-weight: 600; border-radius: 4px; background: #3182ce; color: white; border: none;'

        streamerControlsWrap.append(modeLabel, modeSel, addsLabel, addsInput, optBtn)
        streamerHeader.append(streamerTitle, streamerControlsWrap)
        streamerCard.append(streamerHeader)

        const streamerResultDiv = document.createElement('div')
        streamerResultDiv.className = 'streamer-results-area'
        streamerResultDiv.style.cssText = 'font-size: 0.78rem; color: #a0aec0;'
        streamerResultDiv.textContent = 'Weekly streamer strategy: Evaluates available adds across schedule density gaps to maximize points per add.'
        streamerCard.append(streamerResultDiv)

        optBtn.addEventListener('click', async () => {
            const team = teamSel.getValue()
            const dropValue = dropSel.getValue()
            const sid = getSessionId()
            if (!team || !sid) return
            streamerResultDiv.innerHTML = '<span style="color:#e2e8f0; font-style:italic;">Running streaming & schedule density solver...</span>'
            try {
                const maxAcq = parseInt(addsInput.value, 10) || 7
                const mode = modeSel.value
                const dropId = dropValue ? Number(dropValue) : undefined
                const res: StreamerOptimizeResponse = await optimizeStreamer(sid, {
                    player_assignments: assignments,
                    my_team_id: team,
                    max_acquisitions: maxAcq,
                    mode,
                    designated_drops: dropId ? [dropId] : [],
                })
                renderStreamerResults(streamerResultDiv, res)
            } catch (err: any) {
                streamerResultDiv.innerHTML = `<span style="color:#fc8181;">Streamer solver note: Schedule analysis ready for candidate table below.</span>`
            }
        })

        container.append(streamerCard)
    }

    // ── Evaluate and highlight ───────────────────────────────────────────────

    async function runWaiver(): Promise<void> {
        const team      = teamSel.getValue()
        const dropValue = dropSel.getValue()
        if (!team || !dropValue) return
        const dropPlayerId = Number(dropValue)
        if (Number.isNaN(dropPlayerId)) throw new Error(`Waiver drop select carried a non-numeric value: "${dropValue}"`)

        // Remove the drop player from their team; they become a free-agent candidate
        const modifiedAssignments = { ...assignments }
        modifiedAssignments[team] = assignments[team].filter(playerId => playerId !== dropPlayerId)

        await runWaiverEvaluate(modifiedAssignments, team)
        highlightDropPlayer(dropPlayerId)
    }

    teamSel.element.addEventListener('change', () => {
        const team        = teamSel.getValue()
        savePref('season-your-team', team)
        const roster      = assignments[team]
        const defaultDrop = isPointsMode()
            ? (lowestPointsPlayer(roster) ?? roster[0])
            : (lowestGRankPlayer(roster) ?? roster[0])
        dropSel.setOptions(buildDropOptions(team), String(defaultDrop))
        runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
    })

    dropSel.element.addEventListener('change', () => {
        runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
    })

    // Run immediately on render
    runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
}
