/**
* Determines a CSS style for a cell based on the difference between value and middle.
* Color scheme: muted teal for positive, muted magenta-purple for negative.
* Dark blue-grey baseline, intensity capped to stay cohesive with the sidebar palette.
* Used for category-level win rates and G-score category cells.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value that maps to the default dark color
* @returns {string} A CSS style string
*/
export function stat_styler_primary(value: number, multiplier: number, middle: number): string {
    if (value == -999) {
        return 'background-color:#8D8D9E;color:#8D8D9E;';
    }
    let raw_intensity = (value - middle) * multiplier;
    let intensity = Math.min(Math.round(Math.abs(raw_intensity)), 110);

    let r = raw_intensity > 0 ? 55 : 55 + intensity;
    let g = raw_intensity > 0 ? 55 + intensity : 55;
    let b = 70 + Math.round(intensity * 0.7);

    return final_formatter(r, g, b);
}

/**
* Color scheme: muted gold for positive, muted amber-orange for negative.
* Used for overall totals and differences.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value
* @returns {string} A CSS style string
*/
export function stat_styler_secondary(value: number, multiplier: number, middle: number): string {
    if (value == -999) {
        return 'background-color:#8D8D9E;color:#8D8D9E;';
    }
    let raw_intensity = (value - middle) * multiplier;
    let intensity = Math.min(Math.round(Math.abs(raw_intensity)), 150);

    let r, g, b;
    if (raw_intensity > 0) {
        r = 80 + Math.round(intensity * 0.53);
        g = 80 + Math.round(intensity * 0.5);
        b = 80;
    } else {
        r = 80 + Math.round(intensity * 0.67);
        g = 80 + Math.round(intensity * 0.2);
        b = 80;
    }
    return final_formatter(r, g, b);
}

/**
* Color scheme: muted steel-blue shades.
* Used for algorithm decisions like category weights.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value. Values below middle have minimal color effect.
* @returns {string} A CSS style string
*/
export function stat_styler_tertiary(value: number, multiplier: number, middle: number): string {
    if (value == -999) {
        return 'background-color:#555566;color:#555566;';
    }
    let raw_intensity = Math.round((value - middle) * multiplier);
    let intensity = Math.min(Math.abs(raw_intensity), 130);

    let r = raw_intensity > 0 ? 28 + Math.round(intensity / 6) : 28 - Math.round(intensity / 60);
    let g = raw_intensity > 0 ? 34 + Math.round(intensity / 2) : 34 - Math.round(intensity / 20);
    let b = raw_intensity > 0 ? 46 + Math.round(intensity * 0.7) : 46 - Math.round(intensity / 10);

    return final_formatter(r, g, b);
}

/**
* Returns a CSS string for an RGB background, choosing black or white text for contrast.
*
* @param {number} r - Red channel (0–255)
* @param {number} g - Green channel (0–255)
* @param {number} b - Blue channel (0–255)
* @returns {string} A CSS style string
*/
function final_formatter(r: number, g: number, b: number): string {
    // Formula adapted from:
    // https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    let darkness_value = r * 0.299 + g * 0.587 + b * 0.114;
    let tc = (darkness_value > 150) ? 'black' : 'white';
    return `color:${tc};background-color:rgb(${r},${g},${b});`;
}
