// platforms/fantrax_connector.ts
// Fantrax connect UX: a manual league-id input plus an optional division dropdown.
// No authentication.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { pref, savePref } from '../preferences.js'
import { fetchDivisions } from '../api/client.js'
import { PlatformConnector, ConnectStatus, PLATFORM_LEAGUE_CHOSEN } from './connector.js'

const PLATFORM = 'Retrieve from Fantrax'

export function makeFantraxConnector(status: ConnectStatus): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-fantrax-wrap'

    element.append(makeLabel('ls-league-id', 'League ID'))
    const leagueIdInput = document.createElement('input')
    leagueIdInput.type      = 'text'
    leagueIdInput.id        = 'ls-league-id'
    leagueIdInput.className = 'sidebar-input'
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

    /** Loads the league's divisions, then tells the host the league is named, which connects. The divisions come first
     *  because a league with divisions connects to one of them. */
    function loadDivisionsThenAnnounceLeague(): void {
        if (leagueIdInput.value.trim() === '') return
        loadDivisions()
            .then(() => element.dispatchEvent(new Event(PLATFORM_LEAGUE_CHOSEN, { bubbles: true })))
            .catch(err => status.showError(`Could not load divisions: ${err.message}`))
    }

    leagueIdInput.addEventListener('change', () => {
        savePref('platform_league_id', leagueIdInput.value)
        loadDivisionsThenAnnounceLeague()
    })
    divisionSelect.element.addEventListener('change', () => {
        element.dispatchEvent(new Event(PLATFORM_LEAGUE_CHOSEN, { bubbles: true }))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            const leagueId = leagueIdInput.value.trim()
            if (!leagueId) return null
            return { league_id: leagueId, division_id: divisionSelect.getValue() || null }
        },
        // The remembered league id (restored into the box above) connects as soon as Fantrax is the platform.
        onSelected() { loadDivisionsThenAnnounceLeague() },
    }
}
