// platforms/yahoo_connector.ts
// Yahoo connect UX: OAuth (authenticate → paste authorization code) then pick a league from the
// user's leagues. The OAuth handshake runs inside an in-page dialog that closes itself once the
// token is stored, so the sidebar is left holding only what still matters afterwards — which
// league, and Connect. Tokens are persisted server-side, keyed by the signed-in user.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, fetchYahooAuthUrl, submitYahooToken } from '../api/client.js'
import { PlatformConnector, ConnectStatus } from './connector.js'
import { makeConnectorDialog } from './connector_dialog.js'

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


export function makeYahooConnector(status: ConnectStatus): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-yahoo-wrap'

    const authButton = document.createElement('button')
    authButton.type        = 'button'
    authButton.className   = 'section-apply-btn'
    authButton.textContent = 'Authenticate with Yahoo'
    element.append(authButton)

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
    leagueIdInput.className   = 'sidebar-input'
    leagueIdInput.placeholder = 'e.g. 12345, or paste the draft URL'
    element.append(leagueIdInput)

    // ── The authorization dialog ──────────────────────────────────────────────────────
    // Everything below here is the handshake, and none of it means anything once the token
    // exists — which is why it is in a dialog that dismisses itself rather than in the sidebar.

    const dialog = makeConnectorDialog('ls-yahoo-dialog', 'Authenticate with Yahoo')

    const instructions = document.createElement('p')
    instructions.className = 'ls-dialog-text'
    instructions.textContent =
        'Open Yahoo\'s authorization page and approve access. Yahoo will show you a short code — '
        + 'paste it below and it is exchanged straight away.'

    const openButton = document.createElement('button')
    openButton.type        = 'button'
    openButton.className   = 'section-apply-btn'
    openButton.textContent = 'Open Yahoo authorization page'

    // Popup-blocker fallback. The window.open below happens after an await, so the browser no
    // longer counts it as user-initiated and may refuse it silently; a real link always works.
    // Revealed only once there is a URL to point it at.
    const authLink = document.createElement('a')
    authLink.target        = '_blank'
    authLink.rel           = 'noopener'
    authLink.className     = 'ls-dialog-text'
    authLink.textContent   = 'If no tab opened, use this link'
    authLink.style.display = 'none'

    const codeInput = document.createElement('input')
    codeInput.type        = 'text'
    codeInput.id          = 'ls-yahoo-code'
    codeInput.className   = 'sidebar-input'
    codeInput.placeholder = 'Paste the code from Yahoo'

    // The dialog carries its own line: the sidebar's is behind the overlay while this is open,
    // and these are step-by-step messages that belong beside the step they are about.
    const dialogStatus = document.createElement('p')
    dialogStatus.className = 'ls-dialog-text ls-dialog-status'
    function setDialogProgress(message: string): void {
        dialogStatus.textContent = message
        dialogStatus.classList.remove('sidebar-error')
    }
    function setDialogError(message: string): void {
        dialogStatus.textContent = message
        dialogStatus.classList.add('sidebar-error')
    }

    dialog.body.append(instructions, openButton, authLink, codeInput, dialogStatus)

    /** Loads the authenticated user's Yahoo leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM)
        if (leagues.length === 0) {
            leagueSelect.setOptions([{ value: '', label: '(no leagues found)' }])
            // Not a failure, and not worth a sentence: a mock draft is never listed, and the
            // select itself now says so. The league-ID box below is the way through.
        } else {
            leagueSelect.setOptions(leagues.map(league => ({ value: league.id, label: league.name })))
        }
        status.clear()
    }

    /**
     * Trades the pasted code for a stored token, then gets out of the way.
     *
     * There is no confirmation button in front of this. The code arrives by being copied off
     * Yahoo's page, so a paste IS the user's decision, and a failed exchange stores nothing and
     * can simply be retried — there is no destructive step for a button to guard.
     */
    let exchangeInFlight = false

    function exchangeCode(code: string): void {
        const trimmed = code.trim()
        if (!trimmed) { setDialogError('Paste the authorization code first.'); return }
        // An authorization code is single-use, and Yahoo rejects the second exchange of one with
        // a 400 that reads exactly like a failed authentication -- even though the first attempt
        // SUCCEEDED and the token is already stored. Both a paste and an Enter land here, and
        // pressing Enter right after pasting is the natural thing to do while the first exchange
        // is still in flight, so without this the normal way of using the dialog reports failure
        // for a handshake that worked.
        if (exchangeInFlight) return
        exchangeInFlight = true
        setDialogProgress('Exchanging code...')
        submitYahooToken(trimmed)
            .then(() => loadLeagues())
            .then(() => {
                codeInput.value = ''
                authButton.textContent = 'Re-authenticate with Yahoo'
                dialog.close()
            })
            .catch(err => setDialogError(`Token exchange failed: ${err.message}`))
            .finally(() => { exchangeInFlight = false })
    }

    authButton.addEventListener('click', () => {
        setDialogProgress('')
        dialog.open()
    })

    openButton.addEventListener('click', () => {
        setDialogProgress('Requesting an authorization URL...')
        fetchYahooAuthUrl()
            .then(url => {
                // Opened in its own tab rather than framed in the dialog: Yahoo refuses to be
                // embedded, so the approval page cannot live in here however much tidier that
                // would be. The dialog is what the user comes back to with the code.
                authLink.href          = url
                authLink.style.display = ''
                window.open(url, '_blank', 'noopener')
                setDialogProgress('Approve access in the Yahoo tab, then paste the code here.')
                codeInput.focus()
            })
            .catch(err => setDialogError(`Yahoo auth failed: ${err.message}`))
    })

    codeInput.addEventListener('paste', event => {
        // Read from the clipboard rather than the input: on paste the value has not been updated
        // yet, so codeInput.value here is still whatever was there before.
        const pasted = event.clipboardData?.getData('text') ?? ''
        if (!pasted.trim()) return
        event.preventDefault()
        codeInput.value = pasted.trim()
        exchangeCode(pasted)
    })
    codeInput.addEventListener('keydown', event => {
        if (event.key === 'Enter') exchangeCode(codeInput.value)
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
        // Switching away from Yahoo closes the handshake dialog; it is meaningless over another
        // platform's controls, and a modal left open would block them.
        onDeselected() { dialog.close() },
    }
}
