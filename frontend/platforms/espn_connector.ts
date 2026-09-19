// platforms/espn_connector.ts
// ESPN connect UX: paste the espn_s2 + SWID cookies (found via a browser plugin),
// save them, then pick a league. No OAuth. ESPN is Season-only.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, submitEspnCredentials } from '../api/client.js'
import { PlatformConnector, ConnectStatus } from './connector.js'
import { makeConnectorDialog } from './connector_dialog.js'

const PLATFORM = 'Retrieve from ESPN'

export function makeEspnConnector(status: ConnectStatus): PlatformConnector {
    const element = document.createElement('div')
    element.id = 'ls-espn-wrap'

    element.append(makeLabel('ls-espn-s2', 'ESPN s2 cookie'))
    const s2Input = document.createElement('input')
    s2Input.type      = 'text'
    s2Input.id        = 'ls-espn-s2'
    s2Input.className = 'sidebar-input'
    s2Input.placeholder = 'Paste espn_s2'
    element.append(s2Input)

    element.append(makeLabel('ls-espn-swid', 'ESPN SWID'))
    const swidInput = document.createElement('input')
    swidInput.type      = 'text'
    swidInput.id        = 'ls-espn-swid'
    swidInput.className = 'sidebar-input'
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

    // Instructions dialog. ESPN has no OAuth, so the user has to fetch two cookies by hand and
    // this explains how. Shown when ESPN becomes the selected platform (see onSelected) and
    // dismissed once read.
    const dialog = buildInstructionsDialog()

    /** Loads the user's ESPN leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM)
        if (leagues.length === 0) {
            leagueSelect.setOptions([{ value: '', label: '(no leagues found)' }])
        } else {
            leagueSelect.setOptions(leagues.map(league => ({ value: league.id, label: league.name })))
        }
        // The select says which of those two happened; the line stays empty either way.
        status.clear()
    }

    saveButton.addEventListener('click', () => {
        const s2   = s2Input.value.trim()
        const swid = swidInput.value.trim()
        if (!s2 || !swid) { status.showError('Enter both the s2 and SWID cookies first.'); return }
        status.showProgress('Saving credentials...')
        submitEspnCredentials(s2, swid)
            .then(() => loadLeagues())
            .catch(err => status.showError(`Could not save credentials: ${err.message}`))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            const leagueId = leagueSelect.getValue() ?? ''
            if (!leagueId) return null
            return { league_id: leagueId, division_id: null }
        },
        onSelected()   { dialog.open() },
        onDeselected() { dialog.close() },
    }
}

/** Builds the ESPN auth-instructions dialog. Text carried over from the original Streamlit
 *  dialog this replaces. */
function buildInstructionsDialog() {
    const dialog = makeConnectorDialog('ls-espn-dialog', 'Connecting to ESPN')

    const body = document.createElement('p')
    body.className = 'ls-dialog-text'
    body.innerHTML =
        'Find your ESPN <code>s2</code> and <code>SWID</code> by opening a tab with '
        + '<a href="https://www.espn.com/fantasy/" target="_blank" rel="noopener">ESPN</a>, logging into '
        + 'your account, and using '
        + '<a href="https://chromewebstore.google.com/detail/espn-cookie-finder/oapfffhnckhffnpiophbcmjnpomjkfcj" '
        + 'target="_blank" rel="noopener">this web plug-in</a>. Paste them into the fields under the '
        + 'League Settings sidebar to connect. SWID can be copy-pasted with or without brackets.'

    const closeButton = document.createElement('button')
    closeButton.type        = 'button'
    closeButton.className   = 'section-apply-btn'
    closeButton.textContent = 'Got it'
    closeButton.addEventListener('click', () => dialog.close())

    dialog.body.append(body, closeButton)
    return dialog
}
