// data_entry/method_dropdown.ts
// Compact per-drafter method picker for the draft-board header. The trigger shows a single
// bold letter (M / H / G); clicking opens a small menu of the full labels (first letter bold).
// The stored value is the full DrafterMethod string. Deliberately NOT built on the searchable
// makeCustomSelect (a text-input combobox) — this needs a divergent trigger vs. menu rendering.

import {
    DRAFTER_METHOD_OPTIONS, DrafterMethod, getDrafterMethod, setDrafterMethod,
} from './drafter_methods.js'

/** Builds the compact method dropdown for a drafter. `onChange` runs after a selection. */
export function makeMethodDropdown(
    drafterIndex: number
  , onChange: () => void
  , signal?: AbortSignal
): HTMLElement {
    const wrap = document.createElement('div')
    wrap.className = 'method-dd'

    const trigger = document.createElement('button')
    trigger.type      = 'button'
    trigger.className = 'method-dd-trigger'

    const menu = document.createElement('div')
    menu.className = 'method-dd-menu'
    menu.hidden    = true

    function refreshTrigger(): void {
        const method = getDrafterMethod(drafterIndex)
        trigger.textContent = method.charAt(0).toUpperCase()
        trigger.title       = method
    }

    function onDocClick(event: MouseEvent): void {
        if (!wrap.contains(event.target as Node)) close()
    }
    function onKey(event: KeyboardEvent): void {
        if (event.key === 'Escape') close()
    }
    function open(): void {
        menu.querySelectorAll('.method-dd-item').forEach((el, i) =>
            el.classList.toggle('selected', DRAFTER_METHOD_OPTIONS[i] === getDrafterMethod(drafterIndex)))
        menu.hidden = false
        document.addEventListener('click', onDocClick, { signal })
        document.addEventListener('keydown', onKey, { signal })
    }
    function close(): void {
        menu.hidden = true
        document.removeEventListener('click', onDocClick)
        document.removeEventListener('keydown', onKey)
    }

    for (const method of DRAFTER_METHOD_OPTIONS) {
        const item = document.createElement('button')
        item.type      = 'button'
        item.className = 'method-dd-item'
        item.innerHTML = `<b>${method.charAt(0)}</b>${method.slice(1)}`   // first letter bold
        item.addEventListener('click', () => {
            setDrafterMethod(drafterIndex, method as DrafterMethod)
            refreshTrigger()
            close()
            onChange()
        }, { signal })
        menu.append(item)
    }

    trigger.addEventListener('click', (event) => {
        event.stopPropagation()
        if (menu.hidden) open()
        else close()
    }, { signal })

    refreshTrigger()
    wrap.append(trigger, menu)
    return wrap
}
