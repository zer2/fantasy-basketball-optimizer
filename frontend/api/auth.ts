// api/auth.ts
// Google login gate: checks the server session and renders a sign-in screen when the
// user isn't authenticated. The session rides a same-origin httpOnly cookie, so there is
// no token for JS to hold.

import { BASE_URL } from './client.js'

export interface CurrentUser {
    email: string
}

/** Returns the signed-in user, or null when not authenticated (401). */
export async function fetchCurrentUser(): Promise<CurrentUser | null> {
    const response = await fetch(`${BASE_URL}/auth/me`, { credentials: 'same-origin' })
    if (response.status === 401) return null
    if (!response.ok) throw new Error(`Auth check failed: ${response.status}`)
    return response.json()
}

/** Clears the server session, then reloads back to the login screen. */
export async function logout(): Promise<void> {
    await fetch(`${BASE_URL}/auth/logout`, { method: 'POST', credentials: 'same-origin' })
    location.reload()
}

/** Replaces the app with a centered "Sign in with Google" screen. */
export function renderLoginScreen(): void {
    document.getElementById('app-layout')?.remove()
    const overlay = document.createElement('div')
    overlay.id = 'login-overlay'
    overlay.innerHTML = `
        <div class="login-card">
            <div class="login-logo">🏀</div>
            <div class="login-title">Fantasy Sports Optimizer</div>
            <p class="login-sub">Sign in to continue</p>
            <a class="login-btn" href="/auth/login">Sign in with Google</a>
        </div>`
    document.body.append(overlay)
}
