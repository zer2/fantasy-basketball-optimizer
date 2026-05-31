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
// Getter compatibility: a hidden <select id={id}> is kept in sync so that
// existing getter code like:
//   (document.getElementById('my-id') as HTMLSelectElement).value
// continues to work without modification.
//
// <select> is used (rather than <input type="hidden">) because a sibling
// <label for={id}> requires a labelable element per the HTML spec, and
// <input type="hidden"> is explicitly excluded from that list. Using <select>
// silences Chrome's "label's for attribute doesn't match any element id" warning.

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
 * A `<select id={id} hidden>` is kept in sync inside the wrapper so that
 * existing getter code like `(document.getElementById(id) as HTMLSelectElement).value`
 * continues to work without modification, and so that a sibling `<label for={id}>`
 * references a labelable element (which <input type="hidden"> is not).
 *
 * The wrapper element dispatches a native `'change'` event (bubbling) whenever the
 * selected value changes, so `element.addEventListener('change', cb)` works as expected.
 *
 * @param id           - DOM id; also used for the hidden <select> that exposes `.value`
 * @param options      - Initial option list
 * @param defaultValue - Initially selected value; falls back to `options[0]` if omitted
 */
export function makeCustomSelect(
    id:           string
  , options:      CustomSelectOption[]
  , defaultValue?: string
  , doubleClickToOpen?: boolean
): CustomSelect {

    let currentOptions = [...options]
    let currentValue   = defaultValue ?? currentOptions[0]?.value ?? ''

    // ── DOM structure ──────────────────────────────────────────────────────

    // Root element — callers append this, and attach 'change' listeners to it.
    const wrapper = document.createElement('div')
    wrapper.className = 'cs-wrapper'

    // Hidden <select> — exposes the current value via (document.getElementById(id) as HTMLSelectElement).value,
    // and gives a sibling <label for={id}> a labelable target so Chrome doesn't warn.
    const hiddenSelect = document.createElement('select')
    hiddenSelect.id = id
    hiddenSelect.hidden = true
    hiddenSelect.tabIndex = -1

    function syncHiddenSelectOptions(opts: CustomSelectOption[]): void {
        hiddenSelect.replaceChildren(...opts.map(o => {
            const optionEl = document.createElement('option')
            optionEl.value = o.value
            optionEl.textContent = o.label
            return optionEl
        }))
    }
    syncHiddenSelectOptions(currentOptions)

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

    wrapper.append(hiddenSelect, trigger, dropdown)

    // ── Internal helpers ───────────────────────────────────────────────────

    function getLabelFor(value: string): string {
        return currentOptions.find(o => o.value === value)?.label ?? value
    }

    function commit(value: string, silent = false): void {
        currentValue       = value
        hiddenSelect.value = value
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
            })
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
        })
    } else {
        // Default: single click toggles the dropdown.
        trigger.addEventListener('mousedown', e => {
            if (e.target === searchInput) return
            e.preventDefault()
            if (dropdown.hidden) { searchInput.focus(); open() } else { close(); searchInput.blur() }
        })

        searchInput.addEventListener('focus', () => {
            if (dropdown.hidden) open()
        })

        searchInput.addEventListener('click', () => {
            if (dropdown.hidden) open()
        })
    }

    searchInput.addEventListener('input', () => {
        if (dropdown.hidden) open()
        renderDropdown()
    })

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
    })

    searchInput.addEventListener('blur', () => {
        // Delay so the mousedown handler on a dropdown option fires first.
        setTimeout(close, 150)
    })

    // ── Public API ─────────────────────────────────────────────────────────

    function setOptions(opts: CustomSelectOption[], preferredValue?: string): void {
        currentOptions = [...opts]
        syncHiddenSelectOptions(currentOptions)
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
