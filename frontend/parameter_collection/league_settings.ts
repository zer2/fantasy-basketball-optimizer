// Collects: sport, platform, mode, n_drafters, n_picks,
//           cash_per_team (Auction Mode only), third_round_reversal, team_names, my_team_id
// Mirrors league_settings_popover() in src/parameter_collection/league_settings.py

import { makeCustomSelect } from '../custom_select.js'
import { makeLabel, makeNumberInput, makeSidebarToggle, readRequiredIntInput } from '../helper_functions.js'
import { getSportConfig } from '../app_state.js'
import { pref, savePref } from '../preferences.js'
import { connectPlatform } from '../api/client.js'
import { makeConnectors, connectorPlatforms } from '../platforms/registry.js'
import { PlatformConnector } from '../platforms/connector.js'
import { isSignedIn, makeSignInPrompt } from '../api/auth.js'

const DRAFT_MODE_OPTIONS = ['Draft Mode', 'Auction Mode', 'Season Mode'] as const
export type DraftMode = typeof DRAFT_MODE_OPTIONS[number]

// Own-data plus every platform we have a connector for. Deriving the live options from
// the connector registry means the dropdown can't offer a platform the frontend can't
// connect (main.ts asserts these match the backend's /platforms).
const PLATFORM_OPTIONS = ['Enter your own data', ...connectorPlatforms()]
export type Platform = string

// The active connectors, keyed by platform, so the module-level getPlatformConfig() can
// delegate to whichever platform is selected.
let connectorsByPlatform: Map<string, PlatformConnector> = new Map()

// True once a live platform has been connected successfully (Connect succeeded); reset
// when the platform changes. Gates the live-layout "Refresh Analysis" button.
let platformConnected = false

/** Whether a live platform connection has been established (Connect succeeded). */
export function isPlatformConnected(): boolean {
    return platformConnected
}


/**
 * Renders the League Settings section into `container`.
 * Layout: 2-column grid for compact items, followed by full-width items.
 */
export function renderLeagueSettings(container: HTMLElement): void {
    const config = getSportConfig()
    if (!config) throw new Error('renderLeagueSettings called before sport config loaded')
    const nDraftersDefault = config.options.n_drafters.default
    const nPicksDefault = config.options.n_picks.default

    const grid = document.createElement('div')
    grid.className = 'ls-grid'
    container.append(grid)

    // ── Sport (left col) ──────────────────────────────────────────────────
    // Only NBA is supported now; structured as a selector for future expansion.
    const sportCell = makeCell()
    sportCell.append(makeLabel('ls-sport', 'Sport'))
    const sportSelect = makeCustomSelect(
        'ls-sport',
        [
            { value: 'NBA', label: 'NBA' },
            { value: 'WNBA', label: 'WNBA' }
        ],
        pref('sport', 'NBA'),
    )
    sportSelect.element.addEventListener('change', () => {
        savePref('sport', sportSelect.getValue())
        window.location.reload()
    })
    sportCell.append(sportSelect.element)
    grid.append(sportCell)

    // ── Mode (right col) ──────────────────────────────────────────────────
    const modeCell = makeCell()
    modeCell.append(makeLabel('ls-mode', 'Mode'))
    const modeSelect = makeCustomSelect(
        'ls-mode',
        DRAFT_MODE_OPTIONS.map(m => ({ value: m, label: m })),
        pref('mode', 'Draft Mode'),
    )
    modeSelect.element.addEventListener('change', () => savePref('mode', modeSelect.getValue()))
    modeCell.append(modeSelect.element)
    grid.append(modeCell)

    // ── Platform (full-width) ─────────────────────────────────────────────
    // Controls whether league data is entered manually or pulled from a platform.
    const platformCell = makeCell('ls-cell-full')
    platformCell.append(makeLabel('ls-platform', 'Fantasy Platform'))
    const platformSelect = makeCustomSelect(
        'ls-platform',
        PLATFORM_OPTIONS.map(p => ({ value: p, label: p })),
        pref('platform', 'Enter your own data'),
    )
    platformSelect.element.addEventListener('change', () => {
        savePref('platform', platformSelect.getValue())
        platformConnected = false   // changing platform invalidates the previous connection
    })
    platformCell.append(platformSelect.element)
    grid.append(platformCell)

    // ── Number of drafters (left col) ─────────────────────────────────────
    const draftersCell = makeCell()
    draftersCell.append(makeLabel('ls-n-drafters', 'Drafters'))
    const nDraftersInput = makeNumberInput('ls-n-drafters', pref('n_drafters', nDraftersDefault), 2)
    nDraftersInput.addEventListener('change', () => savePref('n_drafters', parseInt(nDraftersInput.value)))
    draftersCell.append(nDraftersInput)
    grid.append(draftersCell)

    // ── Picks per drafter (right col) ─────────────────────────────────────
    const picksCell = makeCell()
    picksCell.append(makeLabel('ls-n-picks', 'Picks / drafter'))
    const nPicksInput = makeNumberInput('ls-n-picks', pref('n_picks', nPicksDefault), 1)
    nPicksInput.addEventListener('change', () => savePref('n_picks', parseInt(nPicksInput.value)))
    picksCell.append(nPicksInput)
    grid.append(picksCell)

    // ── Budget per team (left col, Auction Mode only) ─────────────────────
    const cashCell = makeCell()
    cashCell.style.display = 'none'
    cashCell.append(makeLabel('ls-cash-per-team', 'Budget / team ($)'))
    const cashInput = makeNumberInput('ls-cash-per-team', pref('cash_per_team', 200), 1)
    cashInput.addEventListener('change', () => savePref('cash_per_team', parseInt(cashInput.value)))
    cashCell.append(cashInput)
    grid.append(cashCell)

    // ── Third round reversal toggle (full-width, Draft Mode only) ─────────
    const trrToggle = makeSidebarToggle('ls-third-round-reversal', 'Third round reversal')
    trrToggle.id = 'ls-trr-row'

    const trrCheckbox = trrToggle.querySelector('input') as HTMLInputElement
    trrCheckbox.checked = pref('third_round_reversal', false)
    trrCheckbox.addEventListener('change', () => savePref('third_round_reversal', trrCheckbox.checked))

    // ── Team identities (hidden; "Team 1".."Team N", derived from drafter count) ──────
    // Team *identity* is fixed "Team N" and drives all logic (my_team_id, draft/auction
    // state, evaluate). The hidden #ls-team-names textarea keeps it in the DOM so existing
    // readers (draft_board, auction, main.ts seat selector) work unchanged. Editable display
    // labels live separately (team_labels.ts) and are shown/edited in the draft headers.
    const hiddenNamesTextarea = document.createElement('textarea')
    hiddenNamesTextarea.id = 'ls-team-names'
    hiddenNamesTextarea.style.display = 'none'
    container.append(hiddenNamesTextarea)
    container.append(trrToggle)

    /** Reset #ls-team-names to "Team 1".."Team N" for the current drafter count, then notify
     *  readers (main.ts rebuilds the seat selector from getTeamNames + getTeamLabel). */
    function refreshTeamIdentities(): void {
        const nDrafters = parseInt(nDraftersInput.value)
        if (isNaN(nDrafters) || nDrafters <= 0) return
        hiddenNamesTextarea.value = Array.from({ length: nDrafters }, (_, i) => `Team ${i + 1}`).join('\n')
        hiddenNamesTextarea.dispatchEvent(new Event('input', { bubbles: true }))
    }
    refreshTeamIdentities()
    nDraftersInput.addEventListener('change', refreshTeamIdentities)

    // ── Live-platform connect (shown only when a live platform is selected) ───
    const connectCell = makeCell('ls-cell-full')
    connectCell.id = 'ls-connect-cell'

    const connectStatus = document.createElement('div')
    connectStatus.id = 'ls-connect-status'
    connectStatus.className = 'pick-control-label'
    const setConnectStatus = (message: string): void => { connectStatus.textContent = message }

    // Each platform's connect/auth controls live in its own connector module. Build them
    // all once, append them, and show the active one — no per-platform branching here.
    const connectors = makeConnectors(setConnectStatus)
    connectorsByPlatform = new Map(connectors.map(connector => [connector.platform, connector]))
    for (const connector of connectors) {
        connector.element.style.display = 'none'
        connectCell.append(connector.element)
    }

    const connectButton = document.createElement('button')
    connectButton.type = 'button'
    connectButton.id = 'ls-connect-btn'
    connectButton.className = 'section-apply-btn'
    connectButton.textContent = 'Connect'
    connectCell.append(connectButton)
    connectCell.append(connectStatus)

    // Connecting a league stores that platform's credentials against a Google account, so it is
    // the one part of the app that cannot work anonymously. Signed out, the controls are replaced
    // by this rather than left to fail at the first request.
    const signInPrompt = makeSignInPrompt(
        'Connecting a live league needs an account, so your platform credentials can be stored '
        + 'against it. Entering your own data works without signing in.')
    signInPrompt.id = 'ls-connect-signin'
    connectCell.append(signInPrompt)
    container.append(connectCell)

    /** Show only the selected platform's connector controls — or, signed out, the sign-in
     *  prompt in their place. */
    let lastConnectorPlatform: string | null = null
    function refreshConnectControls(): void {
        const platform = platformSelect.getValue()
        const canConnect = isSignedIn()
        signInPrompt.style.display = canConnect ? 'none' : ''
        connectButton.style.display = canConnect ? '' : 'none'
        connectStatus.style.display = canConnect ? '' : 'none'
        for (const connector of connectors) {
            connector.element.style.display =
                canConnect && connector.platform === platform ? '' : 'none'
        }
        if (!canConnect) return
        // Fire onSelected/onDeselected only on the transition to a new platform (not on mode-only
        // re-renders), so e.g. the ESPN instructions pop-up appears when ESPN is picked and closes
        // when the user switches away.
        if (platform !== lastConnectorPlatform) {
            connectorsByPlatform.get(lastConnectorPlatform ?? '')?.onDeselected?.()
            lastConnectorPlatform = platform
            connectorsByPlatform.get(platform)?.onSelected?.()
        }
    }

    connectButton.addEventListener('click', () => {
        const connector = connectorsByPlatform.get(platformSelect.getValue())
        if (!connector) return
        const selection = connector.getSelection()
        if (selection === null) {
            setConnectStatus('Select or enter a league first.')
            return
        }
        setConnectStatus('Connecting...')
        connectPlatform(connector.platform, selection.league_id, selection.division_id)
            .then(resp => {
                // Restrict the mode selector to what this platform supports.
                modeSelect.setOptions(resp.available_modes.map(m => ({ value: m, label: m })))
                if (!resp.available_modes.includes(modeSelect.getValue())) {
                    modeSelect.setValue(resp.available_modes[0])
                }
                // Set n_drafters / n_picks directly — NOT via a change event, which
                // would reset the team-name rows — and drive the seat selector via
                // the hidden textarea's input event (main.ts repopulates it).
                nDraftersInput.value = String(resp.n_drafters)
                nPicksInput.value = String(resp.n_picks)
                hiddenNamesTextarea.value = resp.team_names.join('\n')
                hiddenNamesTextarea.dispatchEvent(new Event('input', { bubbles: true }))
                platformConnected = true   // enables the live-layout Refresh Analysis button
                // Patch the session with the platform's config (drives the draft-state poll +
                // name lookup) and counts. Routed through an event so this module doesn't import
                // the session layer (which imports this one — would be a cycle).
                document.dispatchEvent(new Event('platform-connected'))
                setConnectStatus(`Connected — ${resp.team_names.length} teams, ${resp.n_picks} picks. Click Refresh Analysis.`)
            })
            .catch(err => { setConnectStatus(`Connect failed: ${err.message}`) })
    })

    // ── Own-data-dependent and mode-dependent visibility ──────────────────
    function updateVisibility(): void {
        const isOwnData = platformSelect.getValue() === 'Enter your own data'
        const mode = modeSelect.getValue()
        cashCell.style.display = mode === 'Auction Mode' ? '' : 'none'
        trrToggle.style.display = isOwnData && mode === 'Draft Mode' ? '' : 'none'
        connectCell.style.display = isOwnData ? 'none' : ''
        refreshConnectControls()
        if (!isOwnData || mode !== 'Draft Mode') trrCheckbox.checked = false
    }

    updateVisibility()
    modeSelect.element.addEventListener('change', updateVisibility)
    platformSelect.element.addEventListener('change', updateVisibility)
}

export function getTeamNames(): string[] {
    return (document.getElementById('ls-team-names') as HTMLTextAreaElement)
        .value.split('\n').map(s => s.trim()).filter(Boolean)
}

/**
 * Reads all League Settings values from the DOM and returns them as a plain object.
 */
export function getLeagueSettings(): {
    sport: string
    platform: Platform
    mode: DraftMode
    n_drafters: number
    n_picks: number
    cash_per_team: number
    third_round_reversal: boolean
    team_names: string[]
} {
    const mode = (document.getElementById('ls-mode') as HTMLInputElement).value as DraftMode
    return {
        sport: (document.getElementById('ls-sport') as HTMLInputElement).value,
        platform: (document.getElementById('ls-platform') as HTMLInputElement).value as Platform,
        mode,
        n_drafters: readRequiredIntInput('ls-n-drafters'),
        n_picks: readRequiredIntInput('ls-n-picks'),
        cash_per_team: readRequiredIntInput('ls-cash-per-team'),
        third_round_reversal: (document.getElementById('ls-third-round-reversal') as HTMLInputElement).checked,
        team_names: (document.getElementById('ls-team-names') as HTMLTextAreaElement)
            .value.split('\n').map(s => s.trim()).filter(s => s.length > 0),
    }
}

/**
 * Returns the live-platform connection for the session request, or null for
 * 'Enter your own data' (or a live platform with no league ID entered yet).
 */
export function getPlatformConfig(): { league_id: string; division_id?: string | null } | null {
    const platform = (document.getElementById('ls-platform') as HTMLInputElement).value
    return connectorsByPlatform.get(platform)?.getSelection() ?? null
}

/** Creates a grid cell `<div>` that stacks its label and input vertically. */
function makeCell(extraClass?: string): HTMLDivElement {
    const cell = document.createElement('div')
    cell.className = extraClass ? `ls-cell ${extraClass}` : 'ls-cell'
    return cell
}
