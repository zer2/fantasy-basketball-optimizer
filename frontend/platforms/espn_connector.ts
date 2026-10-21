// platforms/espn_connector.ts
// ESPN connect UX: paste the espn_s2 + SWID cookies (found via a browser plugin),
// save them, then pick a league. No OAuth. ESPN is Season-only.
// A public league needs no cookies: its id (or URL) typed into the league-ID box connects directly.

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { pref, savePref } from '../preferences.js'
import { fetchEspnLeagueSeasons, fetchLeagues, submitEspnCredentials } from '../api/client.js'
import { PlatformConnector, ConnectStatus, PLATFORM_LEAGUE_CHOSEN } from './connector.js'
import { makeConnectorDialog } from './connector_dialog.js'
import { extractEspnLeagueId, formatEspnSeason, splitEspnLeagueId } from './espn_league_id.js'

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
    const NO_CREDENTIALS = [{ value: '', label: '(save credentials first)' }]
    const leagueSelect = makeCustomSelect('ls-espn-league', NO_CREDENTIALS)
    element.append(leagueSelect.element)

    // ESPN serves a public league to anyone, so its id needs no cookies; the dropdown, which lists the account's
    // leagues, does. The id typed here goes to the same connect call as a picked league.
    element.append(makeLabel('ls-espn-league-id', 'Or enter a public league ID'))
    const leagueIdInput = document.createElement('input')
    leagueIdInput.type        = 'text'
    leagueIdInput.id          = 'ls-espn-league-id'
    leagueIdInput.className   = 'sidebar-input'
    leagueIdInput.placeholder = 'e.g. 1801165221, or paste the league URL'
    element.append(leagueIdInput)

    // An ESPN league keeps its id from season to season, so a typed id needs a season too: once the id is entered, the
    // league's seasons are listed here, newest first, with the one a pasted URL named (or the newest) chosen. A
    // season that has not drafted yet has empty rosters, which is why it is chosen here rather than assumed.
    element.append(makeLabel('ls-espn-season', 'Season'))
    const NO_TYPED_ID = [{ value: '', label: '(enter a league ID)' }]
    const seasonSelect = makeCustomSelect('ls-espn-season', NO_TYPED_ID)
    element.append(seasonSelect.element)

    // The two ways of naming a league are exclusive, and the screen says which one is in use (as in the Yahoo
    // connector): a typed id wins (see getSelection); while one is typed the dropdown says so, clearing it brings back
    // the league the dropdown showed, and picking a league from the dropdown clears it. Every switch is silent
    // (setOptions), so neither control's handler sets off the other.
    const USING_TYPED_ID = { value: '', label: '(using the league ID below)' }
    let listedLeagueOptions: { value: string, label: string }[] = NO_CREDENTIALS
    let leagueBeforeTypedId: string | null = null
    leagueIdInput.addEventListener('input', () => {
        const isIdTyped = leagueIdInput.value.trim() !== ''
        if (isIdTyped && leagueBeforeTypedId === null) {
            leagueBeforeTypedId = leagueSelect.getValue() ?? ''
            leagueSelect.setOptions([USING_TYPED_ID, ...listedLeagueOptions], USING_TYPED_ID.value)
        } else if (!isIdTyped && leagueBeforeTypedId !== null) {
            leagueSelect.setOptions(listedLeagueOptions, leagueBeforeTypedId)
            leagueBeforeTypedId = null
        }
    })
    // A typed or pasted id is finished when the box is left or Enter is pressed (the change event), not on each key.
    leagueIdInput.addEventListener('change', () => {
        const typed = leagueIdInput.value.trim()
        savePref('espn_typed_league_id', typed === '' ? null : typed)
        if (typed === '') { seasonSelect.setOptions(NO_TYPED_ID); return }
        loadTypedLeagueSeasons(splitEspnLeagueId(extractEspnLeagueId(typed)).season)
    })
    seasonSelect.element.addEventListener('change', () => {
        if (!seasonSelect.getValue()) return
        savePref('espn_typed_league_season', seasonSelect.getValue())
        element.dispatchEvent(new Event(PLATFORM_LEAGUE_CHOSEN, { bubbles: true }))
    })

    /** Lists the typed league's seasons, chooses `preferredSeason` when the league has it (else the newest), and
     *  connects. The choice is made silently (setOptions), and the connection announced once. */
    async function loadTypedLeagueSeasons(preferredSeason: number | null): Promise<void> {
        const { league } = splitEspnLeagueId(extractEspnLeagueId(leagueIdInput.value))
        seasonSelect.setOptions([{ value: '', label: '(finding the seasons...)' }])
        status.showProgress("Finding the league's seasons...")
        let seasons: number[]
        try {
            seasons = await fetchEspnLeagueSeasons(league)
        } catch (error) {
            seasonSelect.setOptions(NO_TYPED_ID)
            status.showError(error instanceof Error ? error.message : String(error))
            return
        }
        const chosen = preferredSeason !== null && seasons.includes(preferredSeason) ? preferredSeason : seasons[0]
        seasonSelect.setOptions(seasons.map(season => ({ value: String(season), label: formatEspnSeason(season) })),
                                String(chosen))
        savePref('espn_typed_league_season', String(chosen))
        status.clear()
        element.dispatchEvent(new Event(PLATFORM_LEAGUE_CHOSEN, { bubbles: true }))
    }

    // Instructions dialog. ESPN has no OAuth, so the user has to fetch two cookies by hand and
    // this explains how. Shown when ESPN becomes the selected platform (see onSelected) and
    // dismissed once read.
    const dialog = buildInstructionsDialog()

    /** Loads the user's ESPN leagues into the league select. */
    async function loadLeagues(): Promise<void> {
        const leagues = await fetchLeagues(PLATFORM)
        listedLeagueOptions = leagues.length === 0
            ? [{ value: '', label: '(no leagues found)' }]
            : leagues.map(league => ({ value: league.id, label: league.name }))
        if (leagueBeforeTypedId !== null) {
            // An id is typed: the dropdown keeps saying so, and the listed leagues wait behind it.
            leagueSelect.setOptions([USING_TYPED_ID, ...listedLeagueOptions], USING_TYPED_ID.value)
            status.clear()
            return
        }
        leagueSelect.setOptions(listedLeagueOptions)
        // The select says which of those two happened; the line stays empty either way.
        status.clear()
        // The league picked last time, when it is listed, is named again and connects; otherwise nothing connects
        // until one is picked (connecting to whichever is listed first would guess).
        const rememberedLeagueId = pref<string | null>('espn_league_id', null)
        if (rememberedLeagueId !== null && leagues.some(league => league.id === rememberedLeagueId)) {
            leagueSelect.setValue(rememberedLeagueId)
        }
    }

    // A picked league is remembered and connects (the host connects on PLATFORM_LEAGUE_CHOSEN).
    leagueSelect.element.addEventListener('change', () => {
        if (!leagueSelect.getValue()) return
        if (leagueBeforeTypedId !== null) {
            leagueIdInput.value = ''
            leagueBeforeTypedId = null
            savePref('espn_typed_league_id', null)
            seasonSelect.setOptions(NO_TYPED_ID)
            leagueSelect.setOptions(listedLeagueOptions, leagueSelect.getValue() ?? undefined)
        }
        savePref('espn_league_id', leagueSelect.getValue())
        element.dispatchEvent(new Event(PLATFORM_LEAGUE_CHOSEN, { bubbles: true }))
    })

    /** Types the remembered public-league id back into the box and connects to it. Needs no credentials, so it runs
     *  when ESPN is selected rather than after the leagues load. */
    function restoreTypedLeague(): void {
        const remembered = pref<string | null>('espn_typed_league_id', null)
        if (remembered === null || leagueIdInput.value.trim() !== '') return
        leagueIdInput.value = remembered
        leagueIdInput.dispatchEvent(new Event('input', { bubbles: true }))   // the dropdown says an id is in use
        const rememberedSeason = pref<string | null>('espn_typed_league_season', null)
        loadTypedLeagueSeasons(rememberedSeason === null ? null : Number(rememberedSeason))
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
            // A typed id wins over the dropdown: someone who has just typed one means it. It connects in the season
            // chosen for it, and not before one is (the seasons are still being listed).
            const typed = extractEspnLeagueId(leagueIdInput.value)
            if (typed) {
                const season = seasonSelect.getValue()
                if (!season) return null
                return { league_id: `${splitEspnLeagueId(typed).league}::${season}`, division_id: null }
            }
            const leagueId = leagueSelect.getValue() ?? ''
            if (!leagueId) return null
            return { league_id: leagueId, division_id: null }
        },
        // A remembered public league reconnects without the dialog: it needs no cookies.
        onSelected() {
            if (pref<string | null>('espn_typed_league_id', null) !== null) restoreTypedLeague()
            else dialog.open()
        },
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
        + 'League Settings sidebar to connect. SWID can be copy-pasted with or without brackets. '
        + 'A public league needs no cookies: paste its league ID or URL into the league ID box instead.'

    const closeButton = document.createElement('button')
    closeButton.type        = 'button'
    closeButton.className   = 'section-apply-btn'
    closeButton.textContent = 'Got it'
    closeButton.addEventListener('click', () => dialog.close())

    dialog.body.append(body, closeButton)
    return dialog
}
