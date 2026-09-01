// styles/styler_functions.ts
// Dual-mode stat-cell stylers.  Every public function emits a CSS string
// containing light-dark() pairs for both themes, so toggling color-scheme
// on :root instantly updates every cell with zero DOM rebuilds.

type RGB = [number, number, number]

// ─── Dark styler (original) ──────────────────────────────────────────────────

function darkPrimary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const intensity = Math.min(Math.round(Math.abs(raw)), 110)
    return [
        raw > 0 ? 55 : 55 + intensity
      , raw > 0 ? 55 + intensity : 55
      , 70 + Math.round(intensity * 0.7)
    ]
}

function darkSecondary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const intensity = Math.min(Math.round(Math.abs(raw)), 150)
    if (raw > 0) {
        return [80 + Math.round(intensity * 0.53)
            , 80 + Math.round(intensity * 0.5)
            , 80]
    } else {
        return [80 + Math.round(intensity * 0.67)
            , 80 + Math.round(intensity * 0.2)
            , 80]
    }
}

function darkTertiary(value: number, multiplier: number, middle: number): RGB {
    const raw = Math.round((value - middle) * multiplier)
    const intensity = Math.min(Math.abs(raw), 130)
    return [
        raw > 0 ? 28 + Math.round(intensity / 6) : 28 - Math.round(intensity / 60)
      , raw > 0 ? 34 + Math.round(intensity / 2) : 34 - Math.round(intensity / 20)
      , raw > 0 ? 46 + Math.round(intensity * 0.7) : 46 - Math.round(intensity / 10)
    ]
}

// ─── Light styler ────────────────────────────────────────────────────────────
// Muted pastels on a white base. The original ported Streamlit's "full-range
// subtraction", which drove strong cells to fully-saturated primary red/green —
// garish against white. Instead we blend white toward a soft target colour by a
// capped factor, so even the strongest cell stays pastel. This mirrors the dark
// styler's restraint (which likewise caps intensity and never reaches a pure primary).

// How far a max-intensity cell travels from white toward its target colour
// (0 = white, 1 = the full target). Kept well below 1 so cells stay soft.
// This is the main knob: raise for punchier colour, lower for subtler.
const LIGHT_BLEND_MAX = 0.7

function blendFromWhite(target: RGB, intensity: number, cap: number): RGB {
    const t = Math.min(intensity, cap) / cap * LIGHT_BLEND_MAX
    return [
        Math.round(255 + t * (target[0] - 255))
      , Math.round(255 + t * (target[1] - 255))
      , Math.round(255 + t * (target[2] - 255))
    ]
}

function lightPrimary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const target: RGB = raw > 0 ? [70, 160, 100] : [205, 80, 80]   // soft green (good) / soft red (bad)
    return blendFromWhite(target, Math.abs(raw), 110)
}

function lightSecondary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const target: RGB = raw > 0 ? [215, 195, 90] : [200, 100, 175]   // soft gold / soft magenta
    return blendFromWhite(target, Math.abs(raw), 150)
}

function lightTertiary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    return blendFromWhite([120, 150, 215], Math.abs(raw), 100)   // soft blue (unsigned magnitude)
}

// ─── Public exports (light-dark dual output) ─────────────────────────────────

// ── Display multipliers ───────────────────────────────────────────────────────
// The stat stylers compute raw = (value - middle) * multiplier and saturate as |raw|
// approaches the blend divisor (~110 for primary). These are the two multipliers with
// app-wide meaning; one-off multipliers stay at their call sites.
export const G_SCORE_MULTIPLIER = 60   // G-score cells (middle 0): full colour ~ +-1.8 G
export const H_MULTIPLIER = 3          // per-category H cells: win % (middle 50); Rotisserie
                                       // standings points scale this by the (n_drafters - 1) range

export function stat_styler_primary(value: number
                                    , multiplier: number
                                    , middle: number): string {
    return formatDual(lightPrimary(value, multiplier, middle)
                      , darkPrimary(value, multiplier, middle))
}

export function stat_styler_secondary(value: number
                                    , multiplier: number
                                    , middle: number): string {
    return formatDual(lightSecondary(value, multiplier, middle)
                      , darkSecondary(value, multiplier, middle))
}

export function stat_styler_tertiary(value: number
                                    , multiplier: number
                                    , middle: number): string {
    return formatDual(lightTertiary(value, multiplier, middle)
                      , darkTertiary(value, multiplier, middle))
}

// ─── Theme switching ─────────────────────────────────────────────────────────

export function setTheme(theme: 'dark' | 'light'): void {
    document.documentElement.style.colorScheme = theme
}

// ─── Formatting helpers ──────────────────────────────────────────────────────

function pickTextColor(r: number, g: number, b: number): string {
    return (r * 0.299 + g * 0.587 + b * 0.114) > 150 ? 'black' : 'white'
}

function formatRgb(c: RGB): string { return `rgb(${c[0]},${c[1]},${c[2]})` }

function formatDual(light: RGB, dark: RGB): string {
    const lightTextColor = pickTextColor(...light)
    const darkTextColor  = pickTextColor(...dark)
    return `color:light-dark(${lightTextColor},${darkTextColor});background-color:light-dark(${formatRgb(light)},${formatRgb(dark)});`
}
