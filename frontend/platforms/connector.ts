// platforms/connector.ts
// A frontend "connector" owns one platform's connect/auth UX — the counterpart to the
// backend PlatformIntegration. It builds the platform-specific controls and produces the
// {league_id, division_id} the shared connect/session flow needs. The host (league_settings)
// renders the active connector and reuses one generic Connect path for every platform, so
// per-platform auth differences stay encapsulated here. The signed-in Google user identifies
// the credentials server-side, so no client identifier is sent from here.

export interface PlatformSelection {
    league_id: string
    division_id: string | null
}

export interface PlatformConnector {
    /** The exact platform label — matches the backend registry key. */
    readonly platform: string
    /** This connector's controls; the host appends it and toggles its visibility. */
    readonly element: HTMLElement
    /** The selection to connect with, or null if the user hasn't supplied enough yet. */
    getSelection(): PlatformSelection | null
    /** Optional: called when this platform becomes the selected one (e.g. to pop up
     *  auth instructions). Fires only on the transition to this platform, not on every refresh. */
    onSelected?(): void
    /** Optional: called when the selection moves away from this platform (e.g. to close a pop-up
     *  this connector opened). Fires once, on the transition away. */
    onDeselected?(): void
}

/** Builds a connector; `setStatus` writes to the shared connect status line. */
export type ConnectorFactory = (setStatus: (message: string) => void) => PlatformConnector
