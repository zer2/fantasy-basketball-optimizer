// platforms/yahoo_connector.ts
// Yahoo connect UX: OAuth (authenticate → paste authorization code) then pick a league
// from the user's leagues. Tokens are persisted server-side, keyed by the signed-in user.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, fetchYahooAuthUrl, submitYahooToken } from '../api/client.js'
import { PlatformConnector } from './connector.js'

const PLATFORM = 'Retrieve from Yahoo'

/** The league id out of whatever the user pasted: a bare id, or a Yahoo URL like
 *  https://basketball.fantasysports.yahoo.com/nba/12345 whose last numeric segment is the league.
 *  Anything else is returned trimmed and unchanged, so a wrong value fails at Yahoo with a message
 *  about that value rather than being silently reinterpreted here. */
function extractLeagueId(raw: string): string {
    const trimmed = raw.trim()
    if (!trimmed.includes('/')) return trimmed
    const numericSegments = trimmed.split(/[/?#]/).filter(segment => /^\d+$/.test(segment))
    return numericSegments.length > 0 ? numericSegments[numericSegments.length - 1] : trimmed
}


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

    // Yahoo's API lists only the leagues a user has joined, and a mock draft is not one of them —
    // so the dropdown can never offer it. The id typed here is passed to exactly the same query as
    // a picked one, which is why a mock works at all: it is a real league Yahoo just does not list.
    element.append(makeLabel('ls-yahoo-league-id', 'Or enter a league ID'))
    const leagueIdInput = document.createElement('input')
    leagueIdInput.type        = 'text'
    leagueIdInput.id          = 'ls-yahoo-league-id'
    leagueIdInput.className   = 'team-name-input'
    leagueIdInput.placeholder = 'e.g. 12345, or paste the draft URL'
    element.append(leagueIdInput)

    authButton.addEventListener('click', () => {
        fetchYahooAuthUrl()
            .then(url => {
                authLink.href = url
                authLink.style.display  = ''
                codeInput.style.display  = ''
                codeButton.style.display = ''
                window.open(url, '_blank', 'noopener')
                const scope = new URL(url).searchParams.get('scope') ?? 'none'
                setStatus(`Authorize on Yahoo (requesting scope: ${scope}), then paste the code below.`)
            })
            .catch(err => setStatus(`Yahoo auth failed: ${err.message}`))
    })

    /** Loads the authenticated user's Yahoo leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM)
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
        submitYahooToken(code)
            .then(() => loadLeagues())
            .catch(err => setStatus(`Token exchange failed: ${err.message}`))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            // A typed id wins over the dropdown: it is the only way to reach a mock draft, and
            // someone who has just typed one means it.
            const typed = extractLeagueId(leagueIdInput.value)
            if (typed) return { league_id: typed, division_id: null }
            const leagueId = leagueSelect.getValue() ?? ''
            if (!leagueId) return null
            return { league_id: leagueId, division_id: null }
        },
    }
}
