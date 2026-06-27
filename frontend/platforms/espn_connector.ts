// platforms/espn_connector.ts
// ESPN connect UX: paste the espn_s2 + SWID cookies (found via a browser plugin),
// save them, then pick a league. No OAuth. ESPN is Season-only.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, submitEspnCredentials } from '../api/client.js'
import { getClientId } from '../api/client_id.js'
import { PlatformConnector } from './connector.js'

const PLATFORM = 'Retrieve from ESPN'

export function makeEspnConnector(setStatus: (message: string) => void): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-espn-wrap'

    element.append(makeLabel('ls-espn-s2', 'ESPN s2 cookie'))
    const s2Input = document.createElement('input')
    s2Input.type      = 'text'
    s2Input.id        = 'ls-espn-s2'
    s2Input.className = 'team-name-input'
    s2Input.placeholder = 'Paste espn_s2'
    element.append(s2Input)

    element.append(makeLabel('ls-espn-swid', 'ESPN SWID'))
    const swidInput = document.createElement('input')
    swidInput.type      = 'text'
    swidInput.id        = 'ls-espn-swid'
    swidInput.className = 'team-name-input'
    swidInput.placeholder = 'Paste SWID'
    element.append(swidInput)

    const saveButton = document.createElement('button')
    saveButton.type        = 'button'
    saveButton.className   = 'section-apply-btn'
    saveButton.textContent = 'Save credentials'
    element.append(saveButton)

    element.append(makeLabel('ls-espn-league', 'League'))
    const leagueSelect = makeCustomSelect('ls-espn-league', [{ value: '', label: '(save credentials first)' }])
    element.append(leagueSelect.element)

    /** Loads the user's ESPN leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM, getClientId())
        if (leagues.length === 0) {
            leagueSelect.setOptions([{ value: '', label: '(no leagues found)' }])
            setStatus('Saved, but no leagues were found.')
        } else {
            leagueSelect.setOptions(leagues.map(league => ({ value: league.id, label: league.name })))
            setStatus('Credentials saved. Pick a league and click Connect.')
        }
    }

    saveButton.addEventListener('click', () => {
        const s2   = s2Input.value.trim()
        const swid = swidInput.value.trim()
        if (!s2 || !swid) { setStatus('Enter both the s2 and SWID cookies first.'); return }
        setStatus('Saving credentials...')
        submitEspnCredentials(getClientId(), s2, swid)
            .then(() => loadLeagues())
            .catch(err => setStatus(`Could not save credentials: ${err.message}`))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            const leagueId = leagueSelect.getValue() ?? ''
            if (!leagueId) return null
            return { league_id: leagueId, division_id: null, client_id: getClientId() }
        },
    }
}
