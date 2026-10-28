// table/failure_banner.ts
// A failure shown above the content, for when the candidate table it would otherwise replace is hidden.
//
// failure_message.ts writes a failed call into the table's place. Several views hide that table — Season Mode's
// Rosters, Waiver and Trading tabs, and the "Show team statistics" views — and there the sentence landed in a hidden
// cell, leaving the failure visible only in the browser console. Measured 2026-10-27 with every projection blend
// weight at zero. This banner carries the sentence in those views. Its own module so failure_message.ts (which shows
// it) and player_table.ts (which clears it whenever the table is refilled) can both use it without importing each
// other.

const banner = document.getElementById('failure-banner') as HTMLDivElement

/** Shows the failure above the content. */
export function showFailureBanner(message: string): void {
    banner.textContent = message
    banner.style.display = ''
}

/** Hides the banner: a new result or message is replacing whatever failed. */
export function clearFailureBanner(): void {
    banner.textContent = ''
    banner.style.display = 'none'
}
