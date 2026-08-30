// helper_functions.ts
// Shared UI building blocks used across sidebar and parameter_collection modules.
// Table-specific helpers (ExpandView and friends) live in table/expand_view.ts.

import { pref, savePref } from './preferences.js'

// ─── Viewport breakpoint ──────────────────────────────────────────────────────
// Mirrors the 768px breakpoint used by the @media (max-width: 768px) blocks in
// styles.css. CSS handles layout-level switches; this is for the few cases
// where rendering code (e.g. inline column widths, short vs full labels) has
// to make the same mobile/desktop choice in JS.

const MOBILE_BREAKPOINT_PX = 768

export function isMobileViewport(): boolean {
    return window.innerWidth <= MOBILE_BREAKPOINT_PX
}

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


/**
 * Reads a required positive integer from a sidebar input. Throws if the element
 * is missing or the value does not parse to a positive integer — both indicate
 * a programming error rather than user choice, so we surface them instead of
 * substituting a silent default.
 */
export function readRequiredIntInput(elementId: string): number {
    const element = document.getElementById(elementId) as HTMLInputElement | null
    if (!element) throw new Error(`Input element #${elementId} not found`)
    const value = parseInt(element.value)
    if (isNaN(value) || value <= 0) {
        throw new Error(`Input #${elementId} must be a positive integer, got "${element.value}"`)
    }
    return value
}


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
export function makeSidebarToggle(id: string, rightText: string, leftText?: string): HTMLLabelElement {
    const row = document.createElement('label')
    row.className = 'sidebar-toggle-row'

    const input = document.createElement('input')
    input.type = 'checkbox'
    input.className = 'sidebar-toggle-input'
    input.id = id

    if (leftText) {
        const leftSpan = document.createElement('span')
        leftSpan.textContent = leftText
        row.append(leftSpan)
    }

    const track = document.createElement('span')
    track.className = 'sidebar-toggle-track'

    const rightSpan = document.createElement('span')
    rightSpan.textContent = rightText

    row.append(input, track, rightSpan)
    return row
}

/** One selectable entry of a multiselect: `value` is what getSelected returns and what
 *  setSelected takes; `label` is what the chip and dropdown show (e.g. a player id value
 *  with a registry-name label). `html` optionally enriches the DROPDOWN row only (e.g. a
 *  player headshot via the display builders — keep imgs lazy); chips stay label text. */
export interface MultiSelectOption {
    value: string
    label: string
    html?: string
}

/**
 * Renders a chip-style multiselect widget (mirrors Streamlit's `st.multiselect`).
 * Selected items are shown as removable chips; clicking the input area opens a
 * filtered dropdown of remaining options.
 *
 * Returns the **live** selected-values array that the widget mutates in place.
 * Callers should store this reference so their getter always sees the current state:
 * ```ts
 * let _selected = renderMultiselect(container, ALL, DEFAULT)
 * // later: _selected always reflects current chip state
 * ```
 */
export function renderMultiselect(
    container:       HTMLElement
    , allOptions:      MultiSelectOption[]
    , defaultSelected: string[]
): string[] {

    const selected: string[] = [...defaultSelected]

    function getLabelFor(value: string): string {
        const option = allOptions.find(o => o.value === value)
        if (!option) throw new Error(`Multiselect value "${value}" is not among its options`)
        return option.label
    }

    const wrapper = document.createElement('div')
    wrapper.className = 'ms-container'

    const inputArea = document.createElement('div')
    inputArea.className = 'ms-input-area'

    const textInput = document.createElement('input')
    textInput.type = 'text'
    textInput.name = 'multiselect-search'
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
        for (const value of selected) {
            const chip = document.createElement('span')
            chip.className = 'ms-chip'

            const chipText = document.createElement('span')
            chipText.textContent = getLabelFor(value)
            chip.append(chipText)

            const removeBtn = document.createElement('button')
            removeBtn.type = 'button'
            removeBtn.className = 'ms-chip-remove'
            removeBtn.textContent = '×'
            removeBtn.addEventListener('mousedown', e => {
                e.preventDefault()
                selected.splice(selected.indexOf(value), 1)
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
            opt => !selected.includes(opt.value) && opt.label.toLowerCase().includes(filter)
        )
        if (available.length === 0) {
            const empty = document.createElement('div')
            empty.className = 'ms-empty'
            empty.textContent = filter ? 'No matches' : 'All options selected'
            dropdown.append(empty)
            return
        }
        for (const opt of available) {
            const item = document.createElement('div')
            item.className = 'ms-option'
            if (opt.html !== undefined) item.innerHTML = opt.html
            else item.textContent = opt.label
            item.addEventListener('mousedown', e => {
                e.preventDefault()
                selected.push(opt.value)
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

/**
 * A self-contained multiselect widget: labeled container, chip UI, and reactive
 * change notification via MutationObserver.  Use when you need push-style updates
 * (no Apply button) rather than the pull-style `renderMultiselect` + Apply pattern.
 */
export interface MultiSelectWidget {
    element:             HTMLElement
    getSelected:         () => string[]
    setSelected:         (values: string[]) => void
    setSelectedSilently: (values: string[]) => void
    onChange:            (cb: () => void) => void
}

/**
 * Wraps `renderMultiselect` with a labeled container and change detection.
 * Each chip add/remove fires all registered `onChange` callbacks immediately.
 */
export function makeMultiSelectWidget(
    label:   string
    , options: MultiSelectOption[]
    , wrapperClass = 'ms-widget'
): MultiSelectWidget {
    const wrap = document.createElement('div')
    wrap.className = wrapperClass

    if (label) {
        const lbl = document.createElement('div')
        lbl.className = 'pick-control-label'
        lbl.textContent = label
        wrap.append(lbl)
    }

    let selected = renderMultiselect(wrap, options, [])

    const callbacks: (() => void)[] = []

    // Reuses a single MutationObserver across re-renders. Without this, every
    // call to observeInputArea (initial + each replaceSelection) created a new
    // observer without disconnecting the previous one — the old observer kept
    // watching its (now-detached) input area, with its callback closure rooted
    // by V8's observer registry. Same re-render-leak pattern as the prior
    // <select> bug.
    let inputAreaObserver: MutationObserver | null = null

    function observeInputArea(): void {
        inputAreaObserver?.disconnect()
        const inputArea = wrap.querySelector('.ms-input-area')
        if (inputArea) {
            inputAreaObserver = new MutationObserver(() => callbacks.forEach(cb => cb()))
            inputAreaObserver.observe(inputArea, { childList: true })
        }
    }
    observeInputArea()

    function replaceSelection(values: string[]): void {
        const oldContainer = wrap.querySelector('.ms-container')
        if (oldContainer) oldContainer.remove()
        selected = renderMultiselect(wrap, options, values)
        observeInputArea()
    }

    return {
        element:             wrap,
        getSelected:         () => [...selected],
        setSelected:         (values: string[]) => {
            replaceSelection(values)
            callbacks.forEach(cb => cb())
        },
        setSelectedSilently: (values: string[]) => {
            replaceSelection(values)
        },
        onChange:            (cb) => callbacks.push(cb),
    }
}

// ─── Sidebar section helpers ────────────────────────────────────────────────
// Generic utilities for building collapsible sidebar sections with Apply buttons.

/**
 * Creates a collapsible `<details>` sidebar section and returns its content div.
 * The returned element is the container that `render*` functions should populate.
 */
export function createSection(parent: HTMLElement, title: string): HTMLElement {
    const details = document.createElement('details')
    details.className = 'sidebar-section'
    const summary = document.createElement('summary')
    summary.textContent = title
    details.append(summary)
    const content = document.createElement('div')
    content.className = 'sidebar-section-content'
    details.append(content)
    parent.append(details)
    return content
}

/** Builds a board table's corner header cell (the 'Round' cell) as the collapse toggle:
 *  the familiar rotating arrow rides beside the label, and clicking the cell hides the
 *  table's pick rows (tbody/tfoot via the 'board-collapsed' class) without giving the
 *  collapse a row of its own. The team header row stays visible, so team names and
 *  autodraft toggles remain usable while collapsed. The state persists under the given
 *  preference key. */
export function makeBoardToggleHeaderCell(
    table: HTMLTableElement
    , preferenceKey: string
    , labelText: string
): HTMLTableCellElement {
    const cornerHeader = document.createElement('th')
    cornerHeader.className = 'board-toggle-header'
    cornerHeader.title = 'Collapse or expand the board'

    const arrow = document.createElement('span')
    arrow.className = 'board-toggle-arrow'
    arrow.textContent = '▶'
    cornerHeader.append(arrow, labelText)

    table.classList.toggle('board-collapsed', !pref(preferenceKey, 1))
    cornerHeader.addEventListener('click', () => {
        const collapsed = table.classList.toggle('board-collapsed')
        savePref(preferenceKey, collapsed ? 0 : 1)
    })
    return cornerHeader
}

/**
 * Appends a small "Apply" button to a sidebar section content div.
 * The callback may be sync or async; errors are caught and logged.
 */
export function addApplyBtn(container: HTMLElement, onClick: () => void | Promise<void>): void {
    const btn = document.createElement('button')
    btn.className   = 'section-apply-btn'
    btn.textContent = 'Apply'
    btn.addEventListener('click', () => {
        const result = onClick()
        if (result instanceof Promise) {
            btn.disabled = true
            result.finally(() => { btn.disabled = false })
                  .catch(err => console.error('Apply failed:', err))
        }
    })
    container.append(btn)
}
