// helper_functions.ts
// Shared UI building blocks used across parameter_collection modules.
// Table-specific helpers (ExpandView and friends) live in expand_view.ts.

// ─── Global JS tooltip ────────────────────────────────────────────────────────
// Uses position:fixed so it escapes the sidebar's overflow-y:auto clipping.
// Any element with a `data-tooltip` attribute automatically gets a hover tooltip.

const _tooltip = document.createElement('div')
_tooltip.className = 'js-tooltip'
document.body.append(_tooltip)

document.addEventListener('mouseover', (e: MouseEvent) => {
    const target = (e.target as Element).closest<HTMLElement>('[data-tooltip]')
    if (!target) {
        _tooltip.style.display = 'none'
        return
    }
    const rect = target.getBoundingClientRect()
    _tooltip.textContent = target.dataset.tooltip!
    _tooltip.style.left = (rect.left + rect.width / 2) + 'px'
    _tooltip.style.top = rect.top + 'px'
    _tooltip.style.display = 'block'
})

/** Creates a `<label>` with class `sidebar-label`, linked to the given input id. */
export function makeLabel(forId: string, text: string): HTMLLabelElement {
    const label = document.createElement('label')
    label.className = 'sidebar-label'
    label.htmlFor = forId
    label.textContent = text
    return label
}

/**
 * Creates a `<input type="number">` with class `sidebar-input`.
 * @param min - Optional minimum value; omit to allow any value (e.g. for signed thresholds).
 */
export function makeNumberInput(id: string, defaultValue: number, min?: number): HTMLInputElement {
    const input = document.createElement('input')
    input.type = 'number'
    input.id = id
    input.className = 'sidebar-input'
    if (min !== undefined) input.min = String(min)
    input.value = String(defaultValue)
    return input
}

/**
 * Creates a toggle switch row (mirrors Streamlit's `st.toggle`).
 * The underlying `<input type="checkbox">` retains the given id so that
 * `getElementById(id).checked` still works for reading the value.
 *
 * @returns The outer `<label>` element — append it directly to the container.
 */
export function makeSidebarToggle(id: string, labelText: string): HTMLLabelElement {
    const row = document.createElement('label')
    row.className = 'sidebar-toggle-row'

    const input = document.createElement('input')
    input.type = 'checkbox'
    input.className = 'sidebar-toggle-input'
    input.id = id

    const track = document.createElement('span')
    track.className = 'sidebar-toggle-track'

    const textSpan = document.createElement('span')
    textSpan.textContent = labelText

    row.append(input, track, textSpan)
    return row
}

/**
 * Renders a chip-style multiselect widget (mirrors Streamlit's `st.multiselect`).
 * Selected items are shown as removable chips; clicking the input area opens a
 * filtered dropdown of remaining options.
 *
 * Returns the **live** selected-items array that the widget mutates in place.
 * Callers should store this reference so their getter always sees the current state:
 * ```ts
 * let _selected = renderMultiselect(container, ALL, DEFAULT)
 * // later: _selected always reflects current chip state
 * ```
 */
export function renderMultiselect(
    container:       HTMLElement,
    allOptions:      string[],
    defaultSelected: string[],
): string[] {

    const selected: string[] = [...defaultSelected]

    const wrapper = document.createElement('div')
    wrapper.className = 'ms-container'

    const inputArea = document.createElement('div')
    inputArea.className = 'ms-input-area'

    const textInput = document.createElement('input')
    textInput.type = 'text'
    textInput.className = 'ms-input'
    textInput.placeholder = 'Add…'
    textInput.autocomplete = 'off'

    const dropdown = document.createElement('div')
    dropdown.className = 'ms-dropdown'
    dropdown.hidden = true

    function renderChips(): void {
        Array.from(inputArea.children).forEach(child => {
            if (child !== textInput) child.remove()
        })
        for (const cat of selected) {
            const chip = document.createElement('span')
            chip.className = 'ms-chip'

            const chipText = document.createElement('span')
            chipText.textContent = cat
            chip.append(chipText)

            const removeBtn = document.createElement('button')
            removeBtn.type = 'button'
            removeBtn.className = 'ms-chip-remove'
            removeBtn.textContent = '×'
            removeBtn.addEventListener('mousedown', e => {
                e.preventDefault()
                selected.splice(selected.indexOf(cat), 1)
                renderChips()
                renderDropdown()
            })
            chip.append(removeBtn)
            inputArea.insertBefore(chip, textInput)
        }
    }

    function renderDropdown(): void {
        dropdown.replaceChildren()
        const filter = textInput.value.toLowerCase()
        const available = allOptions.filter(
            opt => !selected.includes(opt) && opt.toLowerCase().includes(filter)
        )
        if (available.length === 0) {
            const empty = document.createElement('div')
            empty.className = 'ms-empty'
            empty.textContent = filter ? 'No matches' : 'All categories selected'
            dropdown.append(empty)
            return
        }
        for (const opt of available) {
            const item = document.createElement('div')
            item.className = 'ms-option'
            item.textContent = opt
            item.addEventListener('mousedown', e => {
                e.preventDefault()
                selected.push(opt)
                textInput.value = ''
                renderChips()
                renderDropdown()
            })
            dropdown.append(item)
        }
    }

    textInput.addEventListener('focus', () => {
        renderDropdown()
        dropdown.hidden = false
    })

    textInput.addEventListener('blur', () => {
        setTimeout(() => { dropdown.hidden = true }, 150)
    })

    textInput.addEventListener('input', renderDropdown)
    inputArea.addEventListener('click', () => textInput.focus())

    inputArea.append(textInput)
    wrapper.append(inputArea, dropdown)
    container.append(wrapper)

    renderChips()
    return selected
}
