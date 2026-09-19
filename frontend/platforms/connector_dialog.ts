// platforms/connector_dialog.ts
// An in-page dialog for the setup steps a connector needs before it can offer a league — the
// counterpart to Streamlit's @st.dialog, which is what the Streamlit version of this app used
// for exactly these flows.
//
// Connect flows are multi-step (read instructions, authenticate, paste a code back), and every
// one of those controls is noise in the sidebar the moment its step is done. They live in here
// instead, and the dialog closes as soon as the step it exists for has succeeded — leaving the
// sidebar holding only what is still true afterwards: which league to connect.
//
// The dialog attaches to <body>, not to the sidebar that built it, so it is not clipped by the
// sidebar's own scrolling and overflow.

export interface ConnectorDialog {
    /** Where the caller appends the dialog's own controls. */
    readonly body: HTMLElement
    open(): void
    close(): void
}

/**
 * Builds a dialog, hidden, appended to <body>.
 *
 * `id` must be stable per connector: renderLeagueSettings rebuilds its connectors, and a dialog
 * parented to <body> outlives that rebuild, so each new one drops the previous instance by id
 * rather than letting them stack up invisibly.
 */
export function makeConnectorDialog(id: string, title: string): ConnectorDialog {
    document.getElementById(id)?.remove()

    const overlay = document.createElement('div')
    overlay.id            = id
    overlay.className     = 'ls-dialog-overlay'
    overlay.style.display = 'none'

    const box = document.createElement('div')
    box.className = 'ls-dialog-box'

    const header = document.createElement('div')
    header.className = 'ls-dialog-header'

    const titleElement = document.createElement('div')
    titleElement.className   = 'ls-dialog-title'
    titleElement.textContent = title

    const closeButton = document.createElement('button')
    closeButton.type        = 'button'
    closeButton.className   = 'ls-dialog-close'
    closeButton.textContent = '×'
    closeButton.setAttribute('aria-label', 'Close')

    const body = document.createElement('div')
    body.className = 'ls-dialog-body'

    header.append(titleElement, closeButton)
    box.append(header, body)
    overlay.append(box)
    document.body.append(overlay)

    // Declared before the open/close pair that adds and removes it.
    function closeOnEscape(event: KeyboardEvent): void {
        if (event.key === 'Escape') close()
    }

    function open(): void {
        overlay.style.display = 'flex'
        document.addEventListener('keydown', closeOnEscape)
    }

    function close(): void {
        overlay.style.display = 'none'
        // Attached only while open, so a closed dialog never swallows the key from anything else.
        document.removeEventListener('keydown', closeOnEscape)
    }

    closeButton.addEventListener('click', close)
    // The backdrop dismisses; a click on the box itself must not bubble up as one.
    overlay.addEventListener('click', event => { if (event.target === overlay) close() })

    return { body, open, close }
}
