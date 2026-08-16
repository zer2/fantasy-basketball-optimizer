// api/auth.ts
// Google login: optional, not a gate. The app works signed out — signing in is what
// attaches a durable identity, which is what the live-platform credential store needs
// (Yahoo/ESPN/Fantrax tokens are stored per account server-side) and what gives a visitor
// their own rate-limit budget instead of one shared with everyone on the same IP.
// The session rides a same-origin httpOnly cookie, so there is no token for JS to hold.

import { BASE_URL } from './client.js'

export interface CurrentUser {
    email: string
    name: string             // Google given name, falling back to email
    picture: string | null   // Google profile picture URL, when available
}

export const SIGN_IN_PATH = '/auth/login'

// The signed-in user for this page load, or null when browsing anonymously. Set once during
// startup and read wherever a feature has to know — chiefly the live-platform controls, which
// cannot work without an account to store credentials against.
let signedInUser: CurrentUser | null = null

export function setSignedInUser(user: CurrentUser | null): void {
    signedInUser = user
}

export function isSignedIn(): boolean {
    return signedInUser !== null
}

/** Returns the signed-in user, or null when not authenticated (401). */
export async function fetchCurrentUser(): Promise<CurrentUser | null> {
    const response = await fetch(`${BASE_URL}/auth/me`, { credentials: 'same-origin' })
    if (response.status === 401) return null
    if (!response.ok) throw new Error(`Auth check failed: ${response.status}`)
    return response.json()
}

/** Clears the server session, then reloads — landing on the app signed out, not on a wall. */
export async function logout(): Promise<void> {
    await fetch(`${BASE_URL}/auth/logout`, { method: 'POST', credentials: 'same-origin' })
    location.reload()
}

/** A "Sign in with Google" link, styled as a button. Used both in the sidebar account row and
 *  wherever a signed-out visitor reaches a feature that needs an account. */
export function makeSignInLink(label: string): HTMLAnchorElement {
    const link = document.createElement('a')
    link.className   = 'signin-link'
    link.href        = SIGN_IN_PATH
    link.textContent = label
    return link
}

/** The block shown in place of a feature that requires an account: why it is unavailable,
 *  followed by the sign-in link. */
export function makeSignInPrompt(reason: string): HTMLElement {
    const prompt = document.createElement('div')
    prompt.className = 'signin-prompt'

    const explanation = document.createElement('div')
    explanation.className   = 'sidebar-caption'
    explanation.textContent = reason
    prompt.append(explanation, makeSignInLink('Sign in with Google'))
    return prompt
}
