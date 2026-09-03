// preferences.ts
// Persists individual sidebar settings in localStorage so they survive page refreshes.
// Distinct from setting_collection (which gathers values to send to the backend).

const PREFIX = 'fbbo-'

/** Returns the saved value for `key` if one exists in localStorage, otherwise `fallback`. */
export function pref<T>(key: string, fallback: T): T {
    try {
        const raw = localStorage.getItem(PREFIX + key)
        return raw !== null ? JSON.parse(raw) : fallback
    } catch { return fallback }
}

/** Saves a single preference value to localStorage. */
export function savePref(key: string, value: unknown): void {
    try {
        localStorage.setItem(PREFIX + key, JSON.stringify(value))
    } catch { /* storage full or unavailable — silently ignore */ }
}
