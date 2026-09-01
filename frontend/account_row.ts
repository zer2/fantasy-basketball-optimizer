// account_row.ts
// The sidebar's account row: avatar + name, with sign in / sign out. Rendered once at
// startup by main.ts, between the sidebar title and the option sections.

import { CurrentUser, makeSignInLink, logout } from './api/auth.js'

/** Builds the account row and its divider into the sidebar, above the option sections. */
export function renderAccountRow(
    sidebar: HTMLElement
  , sidebarSections: HTMLElement
  , currentUser: CurrentUser | null
): void {
    const accountRow = document.createElement('div')
    accountRow.className = 'sidebar-account'

    const identity = document.createElement('div')
    identity.className = 'account-identity'

    const avatar = document.createElement('span')
    avatar.className = 'account-avatar'

    /** Default person icon — shown when no Google picture is available, and swapped in when a
     *  stored picture URL stops resolving (Google's profile-photo URLs are tokenized and expire;
     *  the session stores the URL from login, so a long-lived login eventually holds a dead link). */
    function showDefaultAccountIcon(): void {
        avatar.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">'
            + '<path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4.4 0-8 2.2-8 5v1h16v-1c0-2.8-3.6-5-8-5Z"/></svg>'
    }

    if (currentUser?.picture) {
        const avatarImg = document.createElement('img')
        avatarImg.src = currentUser.picture
        avatarImg.alt = ''
        avatarImg.referrerPolicy = 'no-referrer'   // Google pic URLs 403 without this
        avatarImg.addEventListener('error', showDefaultAccountIcon)
        avatar.append(avatarImg)
    } else {
        showDefaultAccountIcon()
    }

    const accountName = document.createElement('span')
    accountName.className   = 'account-name'
    // Signed out, the row still shows the account slot — with an invitation rather than a name,
    // so signing in stays one click away instead of being something the visitor has to look for.
    accountName.textContent = currentUser ? currentUser.name : 'Not signed in'
    identity.append(avatar, accountName)

    if (currentUser) {
        const logoutBtn = document.createElement('button')
        logoutBtn.type        = 'button'
        logoutBtn.className    = 'account-logout'
        logoutBtn.textContent  = 'Sign out'
        logoutBtn.addEventListener('click', () => { logout().catch(err => console.error('Logout failed:', err)) })
        accountRow.append(identity, logoutBtn)
    } else {
        accountRow.append(identity, makeSignInLink('Sign in'))
    }
    // Place it below the title's divider line, above the sidebar options, with its own
    // divider separating it from the first option.
    sidebar.insertBefore(accountRow, sidebarSections)
    const accountDivider = document.createElement('hr')
    accountDivider.className = 'sidebar-divider'
    sidebar.insertBefore(accountDivider, sidebarSections)
}
