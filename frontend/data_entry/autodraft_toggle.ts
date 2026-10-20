// data_entry/autodraft_toggle.ts
// Compact per-drafter autodraft toggle for the draft-board header: a small square button showing
// "A". Clicking toggles the drafter between manual entry and being an H-scoring autodrafter; the
// highlighted (active) state means autodrafting is on.

import {
    getDrafterMethod,
    setDrafterMethod,
    getAuctionDrafterMethod,
    setAuctionDrafterMethod,
} from './drafter_methods.js'

/** Builds the compact autodraft toggle for a drafter. `onChange` runs after each toggle. */
export function makeAutodraftToggle(
    drafterIndex: number
  , onChange: () => void
  , signal?: AbortSignal
  , isAuction: boolean = false
): HTMLElement {
    const wrap = document.createElement('div')
    wrap.className = 'method-dd'

    const getMethod = isAuction ? getAuctionDrafterMethod : getDrafterMethod
    const setMethod = isAuction ? setAuctionDrafterMethod : setDrafterMethod

    const button = document.createElement('button')
    button.type        = 'button'
    button.className   = 'method-dd-trigger'
    button.textContent = 'A'

    function checkIsAuto(): boolean {
        return getMethod(drafterIndex) !== 'Manual input'
    }

    function refresh(): void {
        const autodrafting = checkIsAuto()
        button.classList.toggle('is-autodrafting', autodrafting)
        button.title = autodrafting
            ? 'H-scoring autodrafter (click to turn off)'
            : 'Autodraft off (click to make this an H-scoring autodrafter)'
        button.setAttribute('aria-pressed', String(autodrafting))
    }

    button.addEventListener('click', (event) => {
        event.stopPropagation()
        const autodrafting = checkIsAuto()
        setMethod(drafterIndex, autodrafting ? 'Manual input' : 'H-scoring')
        refresh()
        onChange()
    }, { signal })

    refresh()
    wrap.append(button)
    return wrap
}
