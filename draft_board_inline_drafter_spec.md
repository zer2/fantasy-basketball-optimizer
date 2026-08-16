# Spec: inline drafter method in draft-board headers + editable team labels

## Goal
For **own-data Draft mode**, move per-drafter configuration out of the sidebar and into the
draft-board column headers. Each header becomes `[editable label] [▾]`, where `[▾]` is a **compact
one-letter method dropdown** (M / H / G) that opens to the full labels. The label defaults to
`Team N` but is an **editable display label** the user can overwrite in place. **Internally, team
identity stays `Team 1…Team N` (constant)** — a custom label is purely presentation, so renaming
touches nothing in logic. As part of this, **split the draft-board render** so the header (and its
dropdowns) is built once and only the body updates per pick.

## Current behavior (as-is)
- Sidebar "League Settings" → "Teams" section: hidden `#ls-team-names` textarea (source of truth
  for names) + a visible `.team-names-list` of rows, each `[editable name #ls-team-name-i]
  [method select #ls-drafter-mode-i]`. `DRAFTER_METHOD_OPTIONS = ['Manual input','H-scoring','G-scoring']`.
- `getTeamNames()` reads the textarea → consumed by draft_board, draft_state, auction_entry,
  auction_state, main.ts (seat selector). (Season modules do NOT read it — verified.)
- `getDrafterMethodByIndex(i)` reads `#ls-drafter-mode-i` → consumed ONLY by draft_board.ts.
  `getDrafterMethods()` (plural) is defined but unused (dead).
- `renderDraftBoard(container)` does `container.innerHTML=''` then `buildPickControl()` +
  `buildDraftBoard()` (full table: thead `Round|name|name…` + tbody). Called after **every** pick,
  undo, clear, autopilot step, and applyLayout → the entire UI incl. header rebuilds each pick.
- Column header: `th.textContent = name`.

## Target behavior (to-be)
- Own-data Draft header: `Round | Team 1 [▾] | Team 2 [▾] | …`. The `[▾]` shows the current method
  as a single **bold letter** (M/H/G); clicking opens a 3-item menu with full labels, first letter
  bolded (e.g. **M**anual input). Underlying value stays the full string.
- Team names fixed to `Team 1…Team N` (from n_drafters), not editable. "Select team" dropdown and
  auction columns show `Team N` automatically.
- Sidebar "Teams" section removed entirely.
- Draft board: header built once (and on config change); per pick only body + pick control update —
  no header/dropdown churn.

## Design decisions (settled with the user)
- **Identity vs. label split**: internal team identity is ALWAYS `Team 1…Team N` (constant, derived
  from n_drafters) and is what all logic uses (`getTeamNames`, `my_team_id`, draft/auction state,
  config key, backend). A **display label** (default `Team N`) is a separate per-drafter, pref-backed
  value the user edits in the header — purely presentation. Renaming changes only the label, so it
  never resets the draft, never touches the config key, and never reaches the backend.
- Editing a label = one tiny callback: `setTeamLabel(i, …)` + persist + relabel that one option in
  the "Select team" dropdown. (The header input shows the text live; other label readers — e.g. the
  pick-control "Select Pick X for …" text — pick it up on their next natural render.)
- Method control = compact one-letter dropdown (M/H/G trigger; full-label menu, first letter bold).
- Methods are Draft-only (auction has no method dropdown).
- Refactor render so header builds once, body updates per pick.
- OPEN: auction board column headers — show identity `Team N`, or also the display labels? (Default
  in this spec: show labels for consistency; the seat dropdown shows labels regardless.)

## Architecture / changes

### 1. Drafter-method state module (decouple method from the DOM)
Today method state lives in `#ls-drafter-mode-i` and is read cross-module by id. Moving the dropdown
into the header (built by draft_board) while the reader is also draft_board would create
DOM-ordering / cross-module-by-id fragility (e.g. `buildPickControl` reads the method before the
header exists). Fix: hold method state independent of the DOM.

New `frontend/data_entry/drafter_methods.ts`:
- `export type DrafterMethod = 'Manual input' | 'H-scoring' | 'G-scoring'`
- `export const DRAFTER_METHOD_OPTIONS: DrafterMethod[]`
- `getDrafterMethod(index): DrafterMethod` — stored method, default `'Manual input'`, backed by pref
  `drafter_mode_${index}` (index-keyed; survives reloads; no fixed-size array).
- `setDrafterMethod(index, method): void` — persist pref.

Migrate consumers:
- draft_board.ts: import `getDrafterMethod` from the new module instead of `getDrafterMethodByIndex`
  from league_settings.
- league_settings.ts: remove `DRAFTER_METHOD_OPTIONS`, `DrafterMethod`, `getDrafterMethodByIndex`,
  and the dead `getDrafterMethods`. Audit any `DrafterMethod` type imports and repoint to the new
  module (re-export from league_settings if needed to avoid churn).

### 2. Compact method dropdown component
New `frontend/data_entry/method_dropdown.ts` →
`makeMethodDropdown(drafterIndex, onChange: () => void, signal?: AbortSignal): HTMLElement`.
- Button trigger (`.method-dd-trigger`) shows one **bold letter** = first char of
  `getDrafterMethod(index)`.
- Click toggles a small popup (`.method-dd-menu`) of the 3 options; each item HTML = `<b>X</b>rest`
  (first letter bold); current selection highlighted.
- Select → `setDrafterMethod(index, method)`, update trigger letter, close, call `onChange()`.
- Click-outside / Escape close; listeners registered with `signal`.
- NOT built on `makeCustomSelect` (a searchable combobox with a text input — wrong tool for a
  3-option non-searchable picker, and it can't show a divergent trigger vs. menu rendering). Small
  dedicated component (~40–60 lines). [Alternative considered: extend makeCustomSelect with
  renderTrigger/renderOption — rejected as heavier + the search input is inappropriate.]
- Header passes `onChange = () => renderDraftBoard(container)` so changing the *current* drafter's
  method to an autopilot value immediately re-evaluates autopilot (improvement over today's
  "save pref, no effect until next render").

### 3. Draft board render split (header once, body per pick)
Make the table persist across picks.
- Container layout (stable across picks):
  - `div.draft-pick-control` (rebuilt every render)
  - `div.entry-table-scroll > table.entry-table` (thead built once / on config change; tbody rebuilt
    each render)
- Module state: `let boardTable: HTMLTableElement | null`. Reuse `getConfigKey()` to detect config
  changes.
- Two listener controllers:
  - `pickListenerController` — aborted/recreated every render (candidate select in pick control).
  - `headerListenerController` — aborted/recreated only when the table is (re)built (method
    dropdowns). MUST NOT be aborted during autopilot picks.
- `renderDraftBoard(container)`:
  1. `cfg = readDraftConfig(); configChanged = cfg.key !== getConfigKey(); if (configChanged) applyDraftConfig(cfg)`.
  2. If first render OR `configChanged` OR `boardTable` detached → (re)build scaffold: pick-control
     slot + new table (`buildHeaderRow()` incl. method dropdowns, empty tbody); abort+recreate
     `headerListenerController`; store `boardTable`.
  3. Always: rebuild pick control into its slot (abort+recreate `pickListenerController`) via existing
     `buildPickControl`.
  4. Always: `rebuildBody(boardTable)` — swap a fresh `<tbody>` built from state (drafted cells +
     current-pick highlight). tbody is text/classes only (no components) → cheap, no dropdown churn.
  5. Autopilot check (unchanged) using `getDrafterMethod(getPickDrafter())`.
- `buildHeaderRow()`: `Round` th + per drafter a th containing an **editable label input**
  (`.team-label-input`, value = `getTeamLabel(i)`, placeholder `Team ${i+1}`) + the method dropdown
  `makeMethodDropdown(i, () => renderDraftBoard(container), headerListenerController.signal)`. The
  label input's debounced `input` handler = `setTeamLabel(i, value)` + persist + notify the seat
  selector to relabel (see §4) — NOT `applyLayout`, and it does not rebuild the header (the input
  keeps focus across per-pick body updates because the header is built once).
- `rebuildBody(table)`: replace tbody; same cell logic as today's body (drafted text+`drafted`
  class; `current-pick` class). Keep ROUND_W / TEAM_W / min-width.
- Effectively splits `buildDraftBoard()` → `buildHeaderRow()` (once) + `rebuildBody()` (per pick).
  Pick control stays fully rebuilt per pick (its candidate dropdown changes every pick).

### 4. Identity (`Team N`, constant) + editable display labels + remove sidebar Teams section

**Identity (logic) — constant.** `#ls-team-names` is ALWAYS `Team 1…Team N`, derived from
n_drafters, never user-edited:
- league_settings.ts: remove the entire Teams UI (`teamNamesWrap` label, `.team-names-list`,
  `rebuildTeamNameRows`, `syncTeamNames`, per-row name inputs + method selects).
- Keep the hidden `#ls-team-names` textarea (other readers depend on it). On init and whenever
  `n_drafters` changes: set value = `Array.from({length:n},(_,i)=>`Team ${i+1}`).join('\n')`.
- `getTeamNames()` therefore returns the stable identities; used by draft_board (drafter index →
  `Team N` for `my_team_id`), draft_state, auction_state, evaluate, and the config key (which is now
  effectively constant w.r.t. names → renaming can never reset the draft).
- Live platform still overrides `#ls-team-names` on connect (unchanged). Season independent.

**Display labels (presentation) — editable.** New `frontend/data_entry/team_labels.ts`:
- `getTeamLabel(index): string` — saved label or default `Team ${index+1}` (pref `team_label_${index}`).
- `setTeamLabel(index, text): void` — persist; empty/whitespace clears back to the default.
- A lightweight change signal so the seat selector relabels without a layout rebuild: on
  `setTeamLabel`, `document.dispatchEvent(new Event('team-labels-changed'))`.

**Seat selector ("Select team", main.ts).** Build options as
`{ value: getTeamNames()[i] (identity "Team N"), label: getTeamLabel(i) }` — so `getCurrentSeat()`
still returns the identity, only the visible text differs. Add a listener for `team-labels-changed`
that re-applies the option **labels** (preserving the current selection by value). Also: simplify the
existing `#ls-team-names` input handler so it no longer needs `applyLayout` for name display (names
are identities now; labels drive display via the event).

**Where labels are shown:** the header input, the seat dropdown label, and the pick-control
"Select Pick X for {label}" / autopilot "{label} (method)" text (which read `getTeamLabel(pickDrafter)`
and rebuild per pick). **Auction:** board column headers use `getTeamLabel(i)` too (display), while
auction_state keys stay on identity `getTeamNames()` (see OPEN decision — flip to identity `Team N`
if preferred).

### 5. Renames (Drafter → Team) for fallbacks
The `?? `Drafter ${...}`` fallbacks become `?? getTeamLabel(...)` (display) or `Team ${...}`
(identity), as appropriate per call site:
- draft_board.ts:113 (setCurrentSeat → identity `Team ${...}`), 161/162 (pick label → `getTeamLabel`).
- draft_state.ts:137, auction_state.ts:99 (identity `Team ${...}`).
NOT changing `default_season_rosters.ts` ('Drafter N' keys) — season is out of scope and does not
read `getTeamNames()`. Optional follow-up: align season naming.

### 6. CSS
- Header th: lay out `[label input][dropdown]` inline (inline-flex, small gap, centered, compact).
- New `.team-label-input` (borderless/transparent input that reads as header text — inherits the
  header font/color, no visible box until focus; centered; ellipsis on overflow).
- New `.method-dd-trigger` (small button, bold letter, muted, hover) and `.method-dd-menu`
  (absolutely-positioned popup; 3 items; hover/selected; first letter bold via `<b>`). Colors from
  existing `light-dark(...)` tokens per CLAUDE.md.
- Remove now-unused sidebar Teams styles: `.team-names-list`, `.team-name-row`,
  `.team-name-row:not(:last-child)`, `.team-name-input`, `.drafter-mode-cell`,
  `.drafter-mode-cell.cs-wrapper`, `.drafter-mode-cell .cs-trigger`.

## Edge cases & behaviors to preserve
- **Renaming (editing a label)** changes ONLY the display label: identity `Team N` is unchanged → no
  config-key change → no draft reset, no header rebuild (input keeps focus), no backend effect. Just
  persists + fires `team-labels-changed` → seat dropdown relabels.
- **n_drafters change** → config key changes (n_drafters is in it) → full table rebuild (new dropdowns
  + label inputs, methods + labels restored per index from prefs) + identities recomputed to
  `Team 1…N`; seat selector rebuilds.
- **Autopilot** still fires from renderDraftBoard when current method ≠ Manual; reads
  `getDrafterMethod`. Mid-autopilot re-renders keep the header stable (only body + pick control
  update). `headerListenerController` must NOT abort during autopilot picks. Verify the "Running
  autopilot" indicator + advancement. (Pick-control label uses `getTeamLabel`.)
- **Undo / Clear** still call renderDraftBoard; body refresh handles cleared cells; undo rewinds to a
  Manual drafter via `getDrafterMethod`.
- **Change current drafter's method to autopilot** → onChange → renderDraftBoard → autopilot picks.
- **Live platform mode**: no manual draft board; header inputs/dropdowns only exist in own-data draft;
  identities overridden by platform names (labels feature is own-data only).
- **Auction (own-data)**: columns show `getTeamLabel(i)`; no method dropdown; `auction_state` keys on
  identity `getTeamNames()` (unchanged).
- **Season mode**: independent (doesn't read getTeamNames) → unaffected.
- **First render before session ready**: `getDrafterMethod` / `getTeamLabel` read prefs (no DOM
  dependency), so `buildPickControl` reading them before the header is built is safe.
- **Listener cleanup**: header listeners (label inputs + method dropdowns) freed on table rebuild;
  pick-control listeners freed each render. No leaks.

## Verification
- `tsc --noEmit` + `npm run build` clean.
- Manual (Draft, own data): header shows `[Team N][M]`; method menu opens with bolded first letters;
  selecting H shows `H` and (if current drafter) auto-picks; **edit a header label → it persists, the
  "Select team" dropdown relabels, the draft board does NOT reset, and focus isn't lost while
  typing**; draft several picks (body updates, header inputs/dropdowns stable, no flicker); undo;
  clear; change n_drafters (headers rebuild, methods + labels persist per index); reload (everything
  persists).
- Confirm `my_team_id` / evaluate still use identity `Team N` (custom label never sent to backend).
- Manual (Auction, own data): columns show labels; no method dropdown; bidding works.
- Manual: "Select team" shows labels; selecting one still drives evaluate via identity.
- Regression: live platform mode shows platform team names; season mode unaffected.
- Run backend test suite (frontend-only change, but confirm nothing imports moved symbols).

## Out of scope / non-goals
- Season-mode roster naming (stays 'Drafter N').
- Custom labels reaching the backend / affecting logic (identity stays `Team N` by design).
- Method dropdown in auction.
- Other styling passes (accent button, etc.).
