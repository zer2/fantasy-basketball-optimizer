// custom_select.ts
// A fully browser-rendered <select> replacement.
// Avoids the OS-native dropdown white-flash on Windows/Chrome, where
// native <select> dropdowns are composited by the OS rather than the browser,
// making them immune to CSS styling and prone to a rendering delay.
//
// Usage:
//   const sel = makeCustomSelect('my-id', [{ value: 'a', label: 'Option A' }])
//   container.append(sel.element)
//   sel.element.addEventListener('change', () => console.log(sel.getValue()))
//
// Getter compatibility: a hidden <input id={id}> is kept in sync so that
// existing getter code like:
//   (document.getElementById('my-id') as HTMLInputElement).value
// continues to work without modification.
//
// The hidden value carrier is <input type="text" hidden>, not <select>.
// Two constraints had to be satisfied at once:
//   (1) it must be a labelable element so <label for={id}> resolves cleanly
//       and Chrome doesn't warn — that rules out <input type="hidden">, which
//       is explicitly excluded from the labelable list.
//   (2) it must not be a heavy native form element with many children. A
//       <select> populated with hundreds of <option>s creates internal
//       shadow DOM (slot + label container + ShadowRoot per option) that
//       browsers don't release cleanly on detach, producing massive detached-
//       DOM leaks when the widget is recreated repeatedly (~half a gigabyte
//       observed with 156 selects × 300 options × N renders).
// <input type="text" hidden> satisfies both: it's labelable per the spec, and
// it has no internal rendering or option children.

export interface CustomSelectOption {
    value: string
    label: string
}

export interface CustomSelect {
    element:    HTMLElement
    getValue:   () => string
    setValue:   (value: string) => void
    setOptions: (options: CustomSelectOption[], preferredValue?: string) => void
}

/**
 * Creates a fully browser-rendered custom select widget.
 *
 * An `<input type="text" id={id} hidden>` is kept in sync inside the wrapper
 * so that existing getter code like `(document.getElementById(id) as HTMLInputElement).value`
 * continues to work without modification, and so that a sibling `<label for={id}>`
 * references a labelable element. See the file header for why this is a hidden
 * text input rather than <select> or <input type="hidden">.
 *
 * The wrapper element dispatches a native `'change'` event (bubbling) whenever the
 * selected value changes, so `element.addEventListener('change', cb)` works as expected.
 *
 * Cleanup: every internal listener (trigger / search input / dropdown options)
 * is registered with the optional `signal`. Callers that build widgets in a
 * re-render path should provide an AbortSignal from a controller they tear
 * down before rebuilding — that detaches all internal listeners in one call
 * and lets the wrapper's closures be garbage-collected. Without a signal the
 * listeners persist for the page's lifetime, which is fine for one-shot
 * widgets (sidebar/seat selector) but accumulates if the widget is recreated
 * repeatedly.
 *
 * @param id           - DOM id; also used for the hidden <input> that exposes `.value`
 * @param options      - Initial option list
 * @param defaultValue - Initially selected value; falls back to `options[0]` if omitted
 */
export function makeCustomSelect(
    id:           string
  , options:      CustomSelectOption[]
  , defaultValue?: string
  , doubleClickToOpen?: boolean
  , signal?:      AbortSignal
): CustomSelect {

    const listenerOpts = signal ? { signal } : undefined

    let currentOptions = [...options]
    let currentValue   = defaultValue ?? currentOptions[0]?.value ?? ''

    // ── DOM structure ──────────────────────────────────────────────────────

    // Root element — callers append this, and attach 'change' listeners to it.
    const wrapper = document.createElement('div')
    wrapper.className = 'cs-wrapper'

    // Hidden value carrier — exposes the current value via
    // (document.getElementById(id) as HTMLInputElement).value, and gives a
    // sibling <label for={id}> a labelable target.
    //
    // Uses <input type="text" hidden> rather than <select>: a native <select>
    // populated with N options creates N pieces of internal shadow DOM (slot +
    // label container + ShadowRoot per option) that the browser keeps alive
    // even after the element is detached. With ~300 options × 156 selects per
    // renderSeasonRosters call, that produced ~half a gigabyte of detached DOM
    // that GC could not reclaim.
    //
    // <input type="hidden"> would be even simpler but is explicitly excluded
    // from the HTML spec's labelable-element list — <label for={id}> against
    // it triggers Chrome's "label's for attribute doesn't match any element"
    // warning. <input type="text"> IS labelable; combined with the `hidden`
    // attribute it's invisible and non-interactive, and it has no internal
    // children to leak.
    const hiddenValueInput = document.createElement('input')
    hiddenValueInput.type = 'text'
    hiddenValueInput.id = id
    hiddenValueInput.hidden = true
    hiddenValueInput.tabIndex = -1
    hiddenValueInput.autocomplete = 'off'

    // Visible trigger
    const trigger = document.createElement('div')
    trigger.className = 'cs-trigger'

    const searchInput = document.createElement('input')
    searchInput.type      = 'text'
    searchInput.name      = `${id}-search`
    searchInput.className = 'cs-search-input'
    searchInput.autocomplete = 'off'
    searchInput.spellcheck   = false

    const arrow = document.createElement('span')
    arrow.className   = 'cs-arrow'
    arrow.textContent = '▾'

    trigger.append(searchInput, arrow)

    // Dropdown panel
    const dropdown = document.createElement('div')
    dropdown.className = 'cs-dropdown'
    dropdown.hidden = true

    wrapper.append(hiddenValueInput, trigger, dropdown)

    // ── Internal helpers ───────────────────────────────────────────────────

    function getLabelFor(value: string): string {
        return currentOptions.find(o => o.value === value)?.label ?? value
    }

    function commit(value: string, silent = false): void {
        currentValue       = value
        hiddenValueInput.value = value
        searchInput.value  = getLabelFor(value)
        if (!silent) wrapper.dispatchEvent(new Event('change', { bubbles: true }))
    }

    /** Returns the options that match the current search text (case-insensitive). */
    function filteredOptions(): CustomSelectOption[] {
        const filter = searchInput.value.toLowerCase()
        if (!filter || filter === getLabelFor(currentValue).toLowerCase()) return currentOptions
        return currentOptions.filter(o => o.label.toLowerCase().includes(filter))
    }

    function renderDropdown(): void {
        dropdown.replaceChildren()
        const visible = filteredOptions()
        if (visible.length === 0) {
            const empty = document.createElement('div')
            empty.className = 'cs-option cs-no-matches'
            empty.textContent = 'No matches'
            dropdown.append(empty)
            return
        }
        for (const opt of visible) {
            const item = document.createElement('div')
            item.className = 'cs-option'
            if (opt.value === currentValue) item.classList.add('cs-option-selected')
            item.textContent = opt.label
            item.addEventListener('mousedown', e => {
                e.preventDefault()   // prevent blur from firing before click
                close()
                commit(opt.value)
            }, listenerOpts)
            dropdown.append(item)
        }
    }

    function open(): void {
        searchInput.value = ''
        searchInput.placeholder = getLabelFor(currentValue)
        renderDropdown()
        dropdown.hidden = false
        wrapper.classList.add('cs-open')
    }

    function close(): void {
        dropdown.hidden = true
        wrapper.classList.remove('cs-open')
        searchInput.value       = getLabelFor(currentValue)
        searchInput.placeholder = ''
        if (doubleClickToOpen) searchInput.readOnly = true
    }

    // ── Event wiring ───────────────────────────────────────────────────────

    if (doubleClickToOpen) {
        // Double-click mode: single click just focuses (for copy/paste);
        // double-click opens the dropdown for editing.
        searchInput.readOnly = true
        trigger.addEventListener('dblclick', () => {
            searchInput.readOnly = false
            searchInput.focus()
            open()
        }, listenerOpts)
    } else {
        // Default: single click toggles the dropdown.
        trigger.addEventListener('mousedown', e => {
            if (e.target === searchInput) return
            e.preventDefault()
            if (dropdown.hidden) { searchInput.focus(); open() } else { close(); searchInput.blur() }
        }, listenerOpts)

        searchInput.addEventListener('focus', () => {
            if (dropdown.hidden) open()
        }, listenerOpts)

        searchInput.addEventListener('click', () => {
            if (dropdown.hidden) open()
        }, listenerOpts)
    }

    searchInput.addEventListener('input', () => {
        if (dropdown.hidden) open()
        renderDropdown()
    }, listenerOpts)

    searchInput.addEventListener('keydown', e => {
        if (e.key === 'Escape') {
            close()
            searchInput.blur()
        } else if (e.key === 'Enter') {
            // Select the first visible option
            const visible = filteredOptions()
            if (visible.length > 0) {
                close()
                commit(visible[0].value)
            }
        } else if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            if (dropdown.hidden) return
            e.preventDefault()
            const visible = filteredOptions()
            const idx = visible.findIndex(o => o.value === currentValue)
            const next = e.key === 'ArrowDown' ? visible[idx + 1] : visible[idx - 1]
            if (next) { commit(next.value); renderDropdown() }
        }
    }, listenerOpts)

    searchInput.addEventListener('blur', () => {
        // Delay so the mousedown handler on a dropdown option fires first.
        setTimeout(close, 150)
    }, listenerOpts)

    // ── Public API ─────────────────────────────────────────────────────────

    function setOptions(opts: CustomSelectOption[], preferredValue?: string): void {
        currentOptions = [...opts]
        const keep = preferredValue ?? currentValue
        const resolved = currentOptions.find(o => o.value === keep)?.value
                      ?? currentOptions[0]?.value ?? ''
        commit(resolved, /* silent */ true)
        if (!dropdown.hidden) renderDropdown()
    }

    // Initialise display
    commit(currentValue, /* silent */ true)

    return {
        element:  wrapper,
        getValue: () => currentValue,
        setValue: (value: string) => commit(value),
        setOptions,
    }
}
