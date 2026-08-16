# Spec: transpose the draft board on mobile (teams as rows, rounds as columns)

## Goal
On mobile, render the own-data Draft board **transposed** so team names live in a **frozen first
column** (always visible) and rounds run left-to-right as columns — so the user scrolls right as the
draft progresses and never loses track of whose row they're on. Desktop keeps the current
orientation (rounds as rows, teams as columns), which suits wide screens.

## Current behavior (as-is)
- `draft_board.ts` renders rounds-as-rows: `<thead>` = `Round | Team 1 | Team 2 | … | Team N` (each
  team `th` holds the `.team-header` widget = editable label input + compact method dropdown);
  `<tbody>` = one row per round (`round# | pick | pick | …`).
- Render split (built earlier): the scaffold (pick-control slot + table **header**) is built once per
  config via `buildScaffold`; the team-header widgets persist (so editing a label keeps focus and the
  method-dropdown listeners don't churn). Only the pick control and `rebuildBoardBody` (which replaces
  the whole `<tbody>`) rebuild per pick. Two controllers: `pickListenerController` (per render),
  `headerListenerController` (per scaffold). Module state: `boardTable`, `pickControlSlot`.
- Mobile shows the same orientation → many team columns overflow horizontally, and the team names
  (column headers) scroll off the top when you scroll right, so you can't tell which column is whom.
- `isMobileViewport()` = `window.innerWidth <= 768`. There are **no resize listeners** in the app —
  no component reflows on raw resize today; orientation is effectively chosen at render time.

## Target behavior (to-be)
- **Desktop (> 768px):** unchanged — rounds as rows, teams as columns.
- **Mobile (≤ 768px):** transposed —
  - `<thead>` = `[corner] | 1 | 2 | … | nPicks` (round numbers as columns).
  - one `<tbody>` row per team: `[sticky team header] | pick | pick | …` across rounds.
  - the team-header column is **`position: sticky; left: 0`** so names stay visible while scrolling
    right through rounds.
  - the current pick is highlighted at its `(team row, round column)` cell, and the board
    **auto-scrolls** that round column into view as picks advance.

## Design decisions
- **Mobile-only alternate layout**, chosen at render time via `isMobileViewport()`. Desktop is
  untouched in appearance/behavior.
- **Keep the existing build-once-header + rebuild-data model; desktop is untouched.** The one wrinkle
  is that HTML tables are row-major: `<thead>`/`<tbody>`/`<tr>` are structural units but a *column* is
  not — a column's cells are scattered one-per-row. So the desktop trick (team headers in a separate
  `<thead>`, replace the whole `<tbody>`) has no exact column equivalent. The symmetric move in the
  transposed layout is simply **per-row**: each team row's *first cell* is the persistent team header
  (built once, sticky), and per pick we rebuild only that row's *trailing* (pick) cells. Same idea —
  header persists, data rebuilt — applied to each row's tail instead of the whole body. No desktop
  changes, no cell-patching.
- **No raw-resize reflow required for v1.** Consistent with the rest of the app (which doesn't reflow
  on resize), orientation is picked on the next natural render. A debounced resize re-render that
  flips orientation live is an **optional** add-on (§7).

## Architecture / changes (`frontend/data_entry/draft_board.ts`)

### 1. Orientation state + scaffold trigger
- Module state: `let transposed = false` (the orientation the current scaffold was built for).
- In `renderDraftBoard`: `const wantTransposed = isMobileViewport()`. Rebuild the scaffold when
  `configChanged || !boardTable || !container.contains(boardTable) || wantTransposed !== transposed`.
  On (re)build, set `transposed = wantTransposed`.
- State is index-based and orientation-agnostic (`drafted[row][drafter]`, `getPickRow/Drafter`), so
  flipping orientation never disturbs picks.

### 2. Body rebuild — desktop unchanged; mobile rebuilds each row's tail
- **Team headers built once**, either way: `buildScaffold` builds
  `teamHeaderCells: HTMLTableCellElement[]` (index = drafter) from the existing `buildTeamHeader(d)`
  (label input + method dropdown), and reuses them across picks.
- **Desktop — unchanged.** `<thead>` = `Round` th + `teamHeaderCells[d]`; `rebuildBoardBody` replaces
  the whole `<tbody>` per pick (round rows). Exactly as today.
- **Mobile (transposed).** `<thead>` = corner th + round-number th (`1..nPicks`). One `<tbody>` row
  per team, whose **first cell is `teamHeaderCells[d]`** (sticky). The mobile branch of
  `rebuildBoardBody` leaves each row's first cell alone and rebuilds only its trailing pick cells:
  `for each row: while (row.cells.length > 1) row.deleteCell(-1)`, then append a cell per round from
  state (`drafted[r][d]` → text + `.drafted`; current `(r,d)` → `.current-pick`; else empty). The
  team-header cell — and any focus in its label input — is never removed, so the no-churn guarantee
  holds.
- `renderDraftBoard` per-render flow is unchanged: (maybe) `buildScaffold` → rebuild pick control →
  `rebuildBoardBody` → autopilot check. `rebuildBoardBody` just branches on `transposed`.

### 3. `buildTeamHeader` unchanged
- Same widget (`makeTeamLabelInput` + `makeMethodDropdown`, on `headerListenerController.signal`).
  Reused verbatim; only its *placement* (thead row vs first column) differs. The mobile header
  stacking we already added (`.team-header { flex-direction: column }`) fits the narrow first column.

### 4. Round labels / header
- Desktop round label = first cell of each body row (existing `entry-cell-label`).
- Mobile: round numbers become the `<thead>` columns (`1..nPicks`); the top-left is an empty corner
  cell. Reuse `entry-cell-label`-style text.

### 5. Sticky first column (mobile) + auto-scroll
- Add a `transposed` class to the table (or `.entry-table-scroll`) to scope mobile CSS.
- `.entry-table.transposed` first-column cells (`th.team-header-cell` and the corner): `position:
  sticky; left: 0; z-index: 2;` with an **opaque background** (team-header cells already have the
  `.entry-table th` background) so scrolled pick cells pass underneath, not through.
- Optional: make the round header row `position: sticky; top: 0` too (frozen header), corner cell
  sticky on both axes — spreadsheet-style. Mark optional.
- Auto-scroll: after the mobile body rebuild, if the draft is in progress, bring the current round
  column into view — keep a reference to the cell tagged `.current-pick` during the rebuild and call
  `cell.scrollIntoView({ inline: 'nearest', block: 'nearest' })` (or compute `scrollLeft` on
  `.entry-table-scroll`). Only nudge when the current round actually changed, so it doesn't fight
  manual scrolling.

### 6. CSS (`styles.css`)
- New `.entry-table.transposed` rules: sticky first column (+ optional sticky header row), z-index,
  background. Keep existing `.entry-table` cell styling (borders use `var(--line-grid)`, etc.).
- Reuse the mobile `.team-header { flex-direction: column }` already added.

### 7. (Optional) live resize flip
- The app has no resize handling, so v1 flips orientation only on the next natural board render. If we
  want it to flip live when the viewport crosses 768px, add one debounced `window` resize listener
  (in `main.ts`/layout) that re-renders the board when `isMobileViewport()` changes. Out of scope for
  v1 unless requested; called out so the limitation is explicit.

## Edge cases & behaviors to preserve
- **Editing a team label / changing a method** never resets the board or loses focus, in either
  orientation (the team-header cell is never removed). This is the key invariant — verify it still
  holds on mobile; desktop is unchanged.
- **Autopilot**: fires identically; `refreshPickCells` + auto-scroll follow the advancing pick.
  `headerListenerController` must not abort mid-autopilot (unchanged).
- **Undo / Clear**: repaint via `refreshPickCells`; current-pick highlight + auto-scroll update.
- **n_drafters / n_picks change**: config key changes → scaffold rebuild → grid resized (mobile: more
  team rows / more round columns); labels + methods restored per index from prefs.
- **Orientation flip** (load on mobile, or resize/rotate if §7 added): scaffold rebuilds in the new
  orientation; picks preserved (index-based); a focused label input loses focus on the flip
  (acceptable, rare).
- **Sticky column legibility**: team-header cells need an opaque background in both themes; current
  `.entry-table th` background covers it.
- **Live platform / auction / season**: out of scope — this spec covers the own-data Draft board
  only. (Live draft uses a different layout; auction/season boards could get the same treatment in a
  follow-up.)

## Verification
- `tsc --noEmit` + `npm run build` clean.
- **Desktop unchanged**: the desktop render path isn't modified, so board looks/behaves exactly as
  before — draft picks, autopilot, undo, clear, n_drafters change, label edit (no focus loss / no
  reset), method change all still work.
- **Mobile (devtools ≤ 768px / real device)**: board is transposed; team names in the sticky first
  column stay put while scrolling right through rounds; current pick highlighted; board auto-scrolls
  to the current round as picks advance; label input + method dropdown usable in the first column;
  picks survive an orientation flip.
- Backend suite still green (frontend-only change).

## Out of scope / non-goals
- Transposing the auction or season boards (possible follow-up).
- A desktop transpose toggle.
- Full app-wide resize reflow (only the optional draft-board resize flip in §7).
