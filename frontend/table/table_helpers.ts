// table/table_helpers.ts
// Shared helpers for the panel tables (expand view, G-score table).

/** Creates an invisible `<th>` spacer used to lock column widths in panel tables. */
export function makeSpacerTh(extraClass?: string): HTMLTableCellElement {
    const spacer = document.createElement('th')
    spacer.className = extraClass ? `panel-colspacer ${extraClass}` : 'panel-colspacer'
    return spacer
}
