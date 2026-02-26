/**
* Determines a CSS style for a cell based on the difference between value and middle.
* Color scheme: cyan/teal for positive, magenta for negative.
* Used for category-level win rates and G-score category cells.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value that maps to the default dark color
* @returns {string} A CSS style string
*/
export function stat_styler_primary(value, multiplier, middle) {
    if (value == -999) {
        return 'background-color:#8D8D9E;color:#8D8D9E;';
    }
    let raw_intensity = (value - middle) * multiplier;
    let intensity = Math.min(Math.round(Math.abs(raw_intensity)), 165);

    let r = raw_intensity > 0 ? 90 : 90 + intensity;
    let g = raw_intensity > 0 ? 90 + intensity : 90;
    let b = 90 + intensity;

    return final_formatter(r, g, b);
}

/**
* Color scheme: yellow for positive, orange for negative.
* Used for overall totals and differences.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value
* @returns {string} A CSS style string
*/
export function stat_styler_secondary(value, multiplier, middle) {
    if (value == -999) {
        return 'background-color:#8D8D9E;color:#8D8D9E;';
    }
    let raw_intensity = (value - middle) * multiplier;
    let intensity = Math.min(Math.round(Math.abs(raw_intensity)), 255);

    let r, g, b;
    if (raw_intensity > 0) {
        r = 130 + Math.round(2 * intensity / 3);
        g = 130 + Math.round(2 * intensity / 3);
        b = 130;
    } else {
        r = 130 + intensity;
        g = 130 + Math.round(intensity / 3);
        b = 130;
    }
    return final_formatter(r, g, b);
}

/**
* Color scheme: varying shades of blue.
* Used for algorithm decisions like category weights.
*
* @param {number} value - The value of the cell
* @param {number} multiplier - Scales the intensity of color relative to (value - middle)
* @param {number} middle - The neutral value. Values below middle have minimal color effect.
* @returns {string} A CSS style string
*/
export function stat_styler_tertiary(value, multiplier, middle) {
    if (value == -999) {
        return 'background-color:#555566;color:#555566;';
    }
    let raw_intensity = Math.round((value - middle) * multiplier);
    let intensity = Math.min(Math.abs(raw_intensity), 185);

    let r = raw_intensity > 0 ? 28 + Math.round(intensity/6) : 28 - Math.round(intensity / 60);
    let g = raw_intensity > 0 ? 34 + Math.round(intensity/2) : 34 - Math.round(intensity / 20);
    let b = raw_intensity > 0 ? 46 + intensity : 46 - Math.round(intensity / 10);

    return final_formatter(r, g, b);
}

/**
* Flat dark background for overall H-score / aggregate cells.
* @returns {string} A CSS style string
*/
export function styler_a() {
    return 'background-color:#2a2a33;color:white;';
}

/**
* Slightly brighter background for summary / total-row cells.
* @returns {string} A CSS style string
*/
export function styler_b() {
    return 'background-color:#38384A;color:white;';
}

/**
* Returns a CSS string for an RGB background, choosing black or white text for contrast.
*
* @param {number} r - Red channel (0–255)
* @param {number} g - Green channel (0–255)
* @param {number} b - Blue channel (0–255)
* @returns {string} A CSS style string
*/
function final_formatter(r, g, b) {
    // Formula adapted from:
    // https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
    let darkness_value = r * 0.299 + g * 0.587 + b * 0.114;
    let tc = (darkness_value > 150) ? 'black' : 'white';
    return `color:${tc};background-color:rgb(${r},${g},${b});`;
}
