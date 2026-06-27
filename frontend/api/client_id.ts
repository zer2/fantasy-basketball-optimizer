// api/client_id.ts
// A stable per-browser client id, minted once and kept in localStorage. Sent with
// auth-based platform requests so the backend can key persisted OAuth credentials
// to this browser (the app has no user login). Constrained to the backend's
// allowed character set ([A-Za-z0-9_-], <=64 chars).

const CLIENT_ID_KEY = 'fbbo-client-id'

function generateClientId(): string {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) {
        return crypto.randomUUID().replace(/-/g, '')
    }
    return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join('')
}

/** Returns this browser's client id, minting and persisting one on first use. */
export function getClientId(): string {
    let id = localStorage.getItem(CLIENT_ID_KEY)
    if (id === null) {
        id = generateClientId()
        localStorage.setItem(CLIENT_ID_KEY, id)
    }
    return id
}
