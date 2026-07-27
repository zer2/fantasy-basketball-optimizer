// scripts/browser_helpers.mjs
// Browser-driving helpers shared by the screenshots pipeline (screenshots.mjs) and the
// e2e test suite (e2e/). They assume the app's markup conventions: custom selects carry
// data-testid="<id>-wrapper" on their wrapper element, and #eval-indicator exposes its
// state through the data-state attribute.

import { execFileSync } from 'node:child_process'

/** Mints an app session cookie via scripts/mint_session_cookie.py (signed with the app's
 *  own SESSION_SECRET_KEY), so headless pages load behind the Google login gate. */
export function mintSessionCookie(repoRoot) {
    return JSON.parse(execFileSync('python', ['scripts/mint_session_cookie.py'],
                                   { cwd: repoRoot, encoding: 'utf8' }).trim())
}

/**
 * Opens a custom select's dropdown and waits for its options to render, retrying the click.
 * The retry matters: the app rebuilds these controls after an evaluate settles (applyLayout),
 * so a click can land on a control that is replaced an instant later — the fresh control never
 * saw the click and stays closed. Re-clicking the (re-resolved) input recovers.
 */
export async function openSelectDropdown(page, wrapperLocator) {
    for (let attempt = 0; attempt < 5; attempt++) {
        await wrapperLocator.locator('.cs-search-input').click()
        try {
            await wrapperLocator.locator('.cs-dropdown .cs-option').first()
                .waitFor({ state: 'visible', timeout: 1000 })
            return
        } catch {
            // Control likely replaced mid-interaction — click the fresh one.
        }
    }
    throw new Error('custom-select dropdown did not open after 5 attempts')
}

/**
 * Opens a custom select's dropdown and clicks the option produced by `resolveOption`,
 * retrying the whole open-and-click sequence: a late app render can close or replace an
 * already-open dropdown, leaving Playwright's own click retry waiting forever on a node
 * that is now hidden — reopening is the only recovery.
 */
export async function chooseDropdownOption(page, wrapperLocator, resolveOption) {
    for (let attempt = 0; attempt < 5; attempt++) {
        await openSelectDropdown(page, wrapperLocator)
        try {
            await resolveOption(wrapperLocator).click({ timeout: 2000 })
            await page.waitForTimeout(100)
            return
        } catch {
            // Dropdown was closed or replaced mid-click — reopen and try again.
        }
    }
    throw new Error('could not click a custom-select option after 5 attempts')
}

/** Custom selects put the id on a hidden input; the wrapper carries data-testid="<id>-wrapper".
 *  Opens the containing collapsed <details> sidebar section, if any, so the control is visible,
 *  then opens the dropdown and clicks the matching option. */
export async function setSelect(page, id, optionText) {
    const wrap = page.locator(`[data-testid="${id}-wrapper"]`).first()
    await wrap.evaluate(el => { const d = el.closest('details'); if (d && !d.open) d.open = true })
    await chooseDropdownOption(page, wrap,
        wrapper => wrapper.locator('.cs-dropdown .cs-option', { hasText: optionText }).first())
}

/** Waits for an evaluate to finish (#eval-indicator reaches "idle"; it starts at "fetching"). */
export async function waitEval(page) {
    await page.waitForFunction(() => {
        const el = document.querySelector('#eval-indicator')
        return el && el.dataset.state === 'idle'
    }, { timeout: 40000 }).catch(() => {})
    await page.waitForTimeout(200)
}

/** Drafts a player for the current pick via the pick-control select + "Lock in selection". */
export async function lockInDraftPick(page, playerName) {
    const wrap = page.locator('[data-testid="draft-pick-select-wrapper"]').first()
    await chooseDropdownOption(page, wrap,
        wrapper => wrapper.locator('.cs-dropdown .cs-option').filter({ hasText: playerName }).first())
    await page.locator('.pick-control-row .pick-btn', { hasText: 'Lock in selection' }).click()
    await waitEval(page)
}
