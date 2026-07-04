// platforms/espn_connector.ts
// ESPN connect UX: paste the espn_s2 + SWID cookies (found via a browser plugin),
// save them, then pick a league. No OAuth. ESPN is Season-only.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { fetchLeagues, submitEspnCredentials } from '../api/client.js'
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

    // Instructions pop-up (mirrors the Streamlit @st.dialog "Authenticate with ESPN"). ESPN has no
    // OAuth, so the user has to fetch two cookies by hand — this pop-up explains how. It is shown
    // when ESPN becomes the selected platform (see onSelected) and dismissed once read.
    const modal = buildInstructionsModal()

    /** Loads the user's ESPN leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM)
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
        submitEspnCredentials(s2, swid)
            .then(() => loadLeagues())
            .catch(err => setStatus(`Could not save credentials: ${err.message}`))
    })

    return {
        platform: PLATFORM,
        element,
        getSelection() {
            const leagueId = leagueSelect.getValue() ?? ''
            if (!leagueId) return null
            return { league_id: leagueId, division_id: null }
        },
        onSelected()   { modal.style.display = 'flex' },
        onDeselected() { modal.style.display = 'none' },
    }
}

/** Builds the ESPN auth-instructions pop-up (appended to <body>, hidden until onSelected).
 *  Text copied from the original Streamlit dialog. */
function buildInstructionsModal(): HTMLElement {
    // Rebuilt on every renderLeagueSettings — drop any previous instance so it can't accumulate.
    document.getElementById('ls-espn-modal')?.remove()

    const overlay = document.createElement('div')
    overlay.id        = 'ls-espn-modal'
    overlay.className = 'espn-modal-overlay'
    overlay.style.display = 'none'

    const box = document.createElement('div')
    box.className = 'espn-modal-box'

    const title = document.createElement('div')
    title.className   = 'espn-modal-title'
    title.textContent = 'Connecting to ESPN'

    const body = document.createElement('p')
    body.className = 'espn-modal-body'
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
    closeButton.addEventListener('click', () => { overlay.style.display = 'none' })

    box.append(title, body, closeButton)
    overlay.append(box)
    // Dismiss when clicking the backdrop (but not the box itself).
    overlay.addEventListener('click', event => { if (event.target === overlay) overlay.style.display = 'none' })
    document.body.append(overlay)
    return overlay
}
