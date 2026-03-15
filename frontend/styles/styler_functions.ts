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
        raw > 0 ? 55 : 55 + intensity,
        raw > 0 ? 55 + intensity : 55,
        70 + Math.round(intensity * 0.7),
    ]
}

function darkSecondary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const intensity = Math.min(Math.round(Math.abs(raw)), 150)
    if (raw > 0) {
        return [80 + Math.round(intensity * 0.53), 80 + Math.round(intensity * 0.5), 80]
    } else {
        return [80 + Math.round(intensity * 0.67), 80 + Math.round(intensity * 0.2), 80]
    }
}

function darkTertiary(value: number, multiplier: number, middle: number): RGB {
    const raw = Math.round((value - middle) * multiplier)
    const intensity = Math.min(Math.abs(raw), 130)
    return [
        raw > 0 ? 28 + Math.round(intensity / 6) : 28 - Math.round(intensity / 60),
        raw > 0 ? 34 + Math.round(intensity / 2) : 34 - Math.round(intensity / 20),
        raw > 0 ? 46 + Math.round(intensity * 0.7) : 46 - Math.round(intensity / 10),
    ]
}

// ─── Light styler ────────────────────────────────────────────────────────────
// SAT controls how vivid the colors get (0 = greyscale, 1 = full saturation)
const SAT = 0.7

function lightPrimary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const i = Math.round(Math.min(Math.round(Math.abs(raw)), 130) * SAT)
    return [
        raw > 0 ? 255 - i : 255,
        raw > 0 ? 255 : 255 - i,
        raw > 0 ? 255 - i : 255 - i,
    ]
}

function lightSecondary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const i = Math.round(Math.min(Math.round(Math.abs(raw)), 140) * SAT)
    return [255, raw > 0 ? 255 : 255 - i, raw > 0 ? 255 - i : 255]
}

function lightTertiary(value: number, multiplier: number, middle: number): RGB {
    const raw = (value - middle) * multiplier
    const i = Math.round(Math.min(Math.round(Math.abs(raw)), 100) * SAT)
    return [
        raw > 0 ? 240 - i : 240 + Math.round(i / 10),
        raw > 0 ? 240 - i : 240 + Math.round(i / 10),
        240,
    ]
}

// ─── Public exports (light-dark dual output) ─────────────────────────────────

//ZR: The '-999' thing was a hack for streamlit, originally. We can probably clean it up by 
//explicitly setting a different class for the ineligible values, and not calling these functions
export function stat_styler_primary(value: number, multiplier: number, middle: number): string {
    if (value == -999) return sentinel('#F6F6F6', '#8D8D9E')
    return formatDual(lightPrimary(value, multiplier, middle), darkPrimary(value, multiplier, middle))
}

export function stat_styler_secondary(value: number, multiplier: number, middle: number): string {
    if (value == -999) return sentinel('#F6F6F6', '#8D8D9E')
    return formatDual(lightSecondary(value, multiplier, middle), darkSecondary(value, multiplier, middle))
}

export function stat_styler_tertiary(value: number, multiplier: number, middle: number): string {
    if (value == -999) return sentinel('#F6F6F6', '#555566')
    return formatDual(lightTertiary(value, multiplier, middle), darkTertiary(value, multiplier, middle))
}

// ─── Theme switching ─────────────────────────────────────────────────────────

export function setTheme(theme: 'dark' | 'light'): void {
    document.documentElement.style.colorScheme = theme
}

// ─── Formatting helpers ──────────────────────────────────────────────────────

function textColor(r: number, g: number, b: number): string {
    return (r * 0.299 + g * 0.587 + b * 0.114) > 150 ? 'black' : 'white'
}

function rgb(c: RGB): string { return `rgb(${c[0]},${c[1]},${c[2]})` }

function formatDual(light: RGB, dark: RGB): string {
    const ltc = textColor(...light)
    const dtc = textColor(...dark)
    return `color:light-dark(${ltc},${dtc});background-color:light-dark(${rgb(light)},${rgb(dark)});`
}

function sentinel(light: string, dark: string): string {
    return `background-color:light-dark(${light},${dark});color:light-dark(${light},${dark});`
}
