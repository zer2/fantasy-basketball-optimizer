// platforms/yahoo_connector.ts
// Yahoo connect UX: OAuth (authenticate → paste authorization code) then pick a league
// from the user's leagues. Tokens are persisted server-side, keyed by the client id.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, fetchYahooAuthUrl, submitYahooToken } from '../api/client.js'
import { getClientId } from '../api/client_id.js'
import { PlatformConnector } from './connector.js'

const PLATFORM = 'Retrieve from Yahoo'

export function makeYahooConnector(setStatus: (message: string) => void): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-yahoo-wrap'

    const authButton = document.createElement('button')
    authButton.type        = 'button'
    authButton.className   = 'section-apply-btn'
    authButton.textContent = 'Authenticate with Yahoo'

    const authLink = document.createElement('a')
    authLink.target      = '_blank'
    authLink.rel         = 'noopener'
    authLink.textContent = 'Open Yahoo authorization page'
    authLink.style.display = 'none'

    const codeInput = document.createElement('input')
    codeInput.type        = 'text'
    codeInput.id          = 'ls-yahoo-code'
    codeInput.className   = 'team-name-input'
    codeInput.placeholder = 'Paste authorization code'
    codeInput.style.display = 'none'

    const codeButton = document.createElement('button')
    codeButton.type        = 'button'
    codeButton.className   = 'section-apply-btn'
    codeButton.textContent = 'Submit code'
    codeButton.style.display = 'none'

    element.append(authButton, authLink, codeInput, codeButton)
    element.append(makeLabel('ls-yahoo-league', 'League'))
    const leagueSelect = makeCustomSelect('ls-yahoo-league', [{ value: '', label: '(authenticate first)' }])
    element.append(leagueSelect.element)

    authButton.addEventListener('click', () => {
        fetchYahooAuthUrl()
            .then(url => {
                authLink.href = url
                authLink.style.display  = ''
                codeInput.style.display  = ''
                codeButton.style.display = ''
                window.open(url, '_blank', 'noopener')
                setStatus('Authorize on Yahoo, then paste the code below.')
            })
            .catch(err => setStatus(`Yahoo auth failed: ${err.message}`))
    })

    /** Loads the authenticated user's Yahoo leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM, getClientId())
        if (leagues.length === 0) {
            leagueSelect.setOptions([{ value: '', label: '(no leagues found)' }])
            setStatus('Authenticated, but no NBA leagues were found.')
        } else {
            leagueSelect.setOptions(leagues.map(league => ({ value: league.id, label: league.name })))
            setStatus('Authenticated. Pick a league and click Connect.')
        }
    }

    codeButton.addEventListener('click', () => {
        const code = codeInput.value.trim()
        if (!code) { setStatus('Paste the authorization code first.'); return }
        setStatus('Exchanging code...')
        submitYahooToken(getClientId(), code)
            .then(() => loadLeagues())
            .catch(err => setStatus(`Token exchange failed: ${err.message}`))
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
