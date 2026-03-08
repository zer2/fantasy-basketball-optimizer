// data_entry/season/season_waiver.ts
// Renders the Season → Waiver Wire tab controls.
// Shows team + drop-player selectors above the standard H-score candidate table.
// On selection, calls evaluate with the dropped player removed from their team,
// then highlights that player's row in the results.

import { makeCustomSelect } from '../../custom_select.js'
import { getCandidatePlayers, getPlayerByName } from '../../app_state.js'
import { runWaiverEvaluate } from '../../api/session.js'

// ─── Helpers ─────────────────────────────────────────────────────────────────

function readTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

function readRosterAssignments(): Record<string, string[]> {
    const teamNames = readTeamNames()
    const nDrafters = parseInt((document.getElementById('ls-n-drafters') as HTMLInputElement).value) || 12
    const nPicks    = parseInt((document.getElementById('ls-n-picks')    as HTMLInputElement).value) || 13

    const assignments: Record<string, string[]> = {}
    for (let d = 0; d < nDrafters; d++) {
        const team = teamNames[d] ?? `Team ${d + 1}`   // d can exceed teamNames.length
        const players: string[] = []
        for (let r = 0; r < nPicks; r++) {
            const input = document.getElementById(`sr-player-${r}-${d}`) as HTMLInputElement | null
            const val = input?.value ?? ''              // getElementById can return null
            if (val) players.push(val)
        }
        assignments[team] = players
    }
    return assignments
}

/** Returns the name of the player with the highest g_rank (worst) among the given names.
 *  Returns null if none of the names are found in the backend player data. */
function lowestGRankPlayer(playerNames: string[]): string | null {
    const byName = getPlayerByName()
    let worst: string | null = null
    let worstRank = -Infinity
    for (const name of playerNames) {
        const p = byName.get(name)
        if (p && p.g_rank > worstRank) {
            worstRank = p.g_rank
            worst = name
        }
    }
    return worst
}

/** Adds a highlight class to the dropped player's row in the candidate table. */
function highlightDropPlayer(dropPlayer: string): void {
    const players = getCandidatePlayers()
    const idx = players.findIndex(p => p.name === dropPlayer)
    const table = document.getElementById('realtable') as HTMLTableElement
    // table.rows[0] is the thead header; player rows start at 1, interleaved with expand rows
    table.rows[1 + idx * 2].cells[0].classList.add('waiver-drop-highlight')
}

// ─── Main render ─────────────────────────────────────────────────────────────

/** Renders waiver wire controls (team + drop-player selectors) into the given container. */
export function renderWaiverControls(container: HTMLElement): void {
    const teamNames   = readTeamNames()
    const assignments = readRosterAssignments()
    const nPicks      = parseInt((document.getElementById('ls-n-picks') as HTMLInputElement).value) || 13
    // assignments[t] can be undefined for team names beyond nDrafters, hence the optional chain
    const fullTeams   = teamNames.filter(t => (assignments[t]?.length ?? 0) >= nPicks)

    if (fullTeams.length < 1) {
        const msg = document.createElement('p')
        msg.className   = 'coming-soon'
        msg.textContent = 'Fill out at least one full team on the Rosters tab to use Waiver Wire.'
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

    const teamSel = makeCustomSelect(
        'waiver-team-select',
        fullTeams.map(n => ({ value: n, label: n })),
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
    const buildDropOptions = (team: string) => assignments[team].map(n => ({ value: n, label: n }))

    const initialTeam    = fullTeams[0]
    const initialRoster  = assignments[initialTeam]
    // lowestGRankPlayer returns null if roster names aren't in backend data yet
    const initialDefault = lowestGRankPlayer(initialRoster) ?? initialRoster[0]

    const dropSel = makeCustomSelect(
        'waiver-drop-select',
        buildDropOptions(initialTeam),
        initialDefault,
    )
    dropWrap.append(dropSel.element)
    container.append(dropWrap)

    // ── Evaluate and highlight ───────────────────────────────────────────────

    async function runWaiver(): Promise<void> {
        const team       = teamSel.getValue()
        const dropPlayer = dropSel.getValue()
        if (!team || !dropPlayer) return

        // Remove the drop player from their team; they become a free-agent candidate
        const modifiedAssignments = { ...assignments }
        modifiedAssignments[team] = assignments[team].filter(p => p !== dropPlayer)
        
        await runWaiverEvaluate(modifiedAssignments, team)
        highlightDropPlayer(dropPlayer)
    }

    teamSel.element.addEventListener('change', () => {
        const team        = teamSel.getValue()
        const roster      = assignments[team]
        const defaultDrop = lowestGRankPlayer(roster) ?? roster[0]
        dropSel.setOptions(buildDropOptions(team), defaultDrop)
        runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
    })

    dropSel.element.addEventListener('change', () => {
        runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
    })

    // Run immediately on render
    runWaiver().catch(err => console.error('Waiver evaluate failed:', err))
}
