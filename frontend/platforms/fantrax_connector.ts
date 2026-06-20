// platforms/fantrax_connector.ts
// Fantrax connect UX: a manual league-id input plus an optional division dropdown.
// No authentication.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { pref, savePref } from '../preferences.js'
import { fetchDivisions } from '../api/client.js'
import { getClientId } from '../api/client_id.js'
import { PlatformConnector } from './connector.js'

const PLATFORM = 'Retrieve from Fantrax'

export function makeFantraxConnector(setStatus: (message: string) => void): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-fantrax-wrap'

    element.append(makeLabel('ls-league-id', 'League ID'))
    const leagueIdInput = document.createElement('input')
    leagueIdInput.type      = 'text'
    leagueIdInput.id        = 'ls-league-id'
    leagueIdInput.className = 'team-name-input'
    leagueIdInput.value     = pref('platform_league_id', '')
    element.append(leagueIdInput)

    const divisionWrap = document.createElement('div')
    divisionWrap.id = 'ls-division-wrap'
    divisionWrap.style.display = 'none'
    divisionWrap.append(makeLabel('ls-division', 'Division'))
    const divisionSelect = makeCustomSelect('ls-division', [{ value: '', label: '(none)' }])
    divisionWrap.append(divisionSelect.element)
    element.append(divisionWrap)

    /** Loads the league's divisions into the division select (hidden when none). */
    async function loadDivisions(): Promise<void> {
        const leagueId = leagueIdInput.value.trim()
        if (!leagueId) return
        const divisions = await fetchDivisions(PLATFORM, leagueId)
        if (divisions.length === 0) {
            divisionWrap.style.display = 'none'
            divisionSelect.setOptions([{ value: '', label: '(none)' }])
        } else {
            divisionWrap.style.display = ''
            divisionSelect.setOptions(divisions.map(division => ({ value: division.id, label: division.name })))
        }
    }

    leagueIdInput.addEventListener('change', () => {
        savePref('platform_league_id', leagueIdInput.value)
        loadDivisions().catch(err => setStatus(`Could not load divisions: ${err.message}`))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            const leagueId = leagueIdInput.value.trim()
            if (!leagueId) return null
            return { league_id: leagueId, division_id: divisionSelect.getValue() || null, client_id: getClientId() }
        },
    }
}
