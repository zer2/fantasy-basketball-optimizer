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
//   (document.getElementById('my-id') as HTMLSelectElement).value
// continues to work without modification.

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
 * A `<input type="hidden" id={id}>` is kept in sync inside the wrapper so that
 * existing getter code like `(document.getElementById(id) as HTMLInputElement).value`
 * continues to work without modification.
 *
 * The wrapper element dispatches a native `'change'` event (bubbling) whenever the
 * selected value changes, so `element.addEventListener('change', cb)` works as expected.
 *
 * @param id           - DOM id; also used for the hidden input that exposes `.value`
 * @param options      - Initial option list
 * @param defaultValue - Initially selected value; falls back to `options[0]` if omitted
 */
export function makeCustomSelect(
    id:           string,
    options:      CustomSelectOption[],
    defaultValue?: string,
): CustomSelect {

    let currentOptions = [...options]
    let currentValue   = defaultValue ?? currentOptions[0]?.value ?? ''

    // ── DOM structure ──────────────────────────────────────────────────────

    // Root element — callers append this, and attach 'change' listeners to it.
    const wrapper = document.createElement('div')
    wrapper.className = 'cs-wrapper'

    // Hidden input — keeps (document.getElementById(id) as HTMLSelectElement).value working.
    const hiddenInput = document.createElement('input')
    hiddenInput.type = 'hidden'
    hiddenInput.id   = id

    // Visible trigger
    const trigger = document.createElement('div')
    trigger.className = 'cs-trigger'
    trigger.tabIndex  = 0

    const triggerText = document.createElement('span')
    triggerText.className = 'cs-trigger-text'

    const arrow = document.createElement('span')
    arrow.className   = 'cs-arrow'
    arrow.textContent = '▾'

    trigger.append(triggerText, arrow)

    // Dropdown panel
    const dropdown = document.createElement('div')
    dropdown.className = 'cs-dropdown'
    dropdown.hidden = true

    wrapper.append(hiddenInput, trigger, dropdown)

    // ── Internal helpers ───────────────────────────────────────────────────

    function getLabelFor(value: string): string {
        return currentOptions.find(o => o.value === value)?.label ?? value
    }

    function commit(value: string, silent = false): void {
        currentValue            = value
        hiddenInput.value       = value
        triggerText.textContent = getLabelFor(value)
        if (!silent) wrapper.dispatchEvent(new Event('change', { bubbles: true }))
    }

    function renderDropdown(): void {
        dropdown.replaceChildren()
        for (const opt of currentOptions) {
            const item = document.createElement('div')
            item.className = 'cs-option'
            if (opt.value === currentValue) item.classList.add('cs-option-selected')
            item.textContent = opt.label
            item.addEventListener('mousedown', e => {
                e.preventDefault()   // prevent trigger's blur from firing before click
                close()
                commit(opt.value)
            })
            dropdown.append(item)
        }
    }

    function open(): void {
        renderDropdown()
        dropdown.hidden = false
        wrapper.classList.add('cs-open')
    }

    function close(): void {
        dropdown.hidden = true
        wrapper.classList.remove('cs-open')
    }

    // ── Event wiring ───────────────────────────────────────────────────────

    trigger.addEventListener('click', () => {
        dropdown.hidden ? open() : close()
    })

    trigger.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            dropdown.hidden ? open() : close()
        } else if (e.key === 'Escape') {
            close()
        } else if (!dropdown.hidden) {
            const idx = currentOptions.findIndex(o => o.value === currentValue)
            if (e.key === 'ArrowDown') {
                e.preventDefault()
                const next = currentOptions[idx + 1]
                if (next) { commit(next.value); renderDropdown() }
            } else if (e.key === 'ArrowUp') {
                e.preventDefault()
                const prev = currentOptions[idx - 1]
                if (prev) { commit(prev.value); renderDropdown() }
            }
        }
    })

    trigger.addEventListener('blur', () => {
        // Delay so the mousedown handler on a dropdown option fires first.
        setTimeout(close, 150)
    })

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
