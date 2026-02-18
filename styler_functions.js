/**
* Determines a CSS style for a cell in the main display table, based on the difference between the value and a pre-defined middle
* 
* @param {number} value - The value of the cell
* @param {number} multiplier - The multiplier applied to (value - middle)
* @param {number} middle - The value to which the cell's value is compared
* @returns {str} A CSS style for the table cell
*/
export function stat_styler_primary(value
    , multiplier
    , middle) {

    if (value == -999) {
            return 'background-color:#8D8D9E;color:#8D8D9E;';
    }
    else {

        let raw_intensity = (value-middle)*multiplier;
        let intensity = Math.min(Math.round(Math.abs(raw_intensity)), 165);

        let r =  raw_intensity > 0 ? 90 : 90 + intensity
        let g =  raw_intensity > 0 ? 90 + intensity : 90
        let b = 90 + intensity; 

        return final_formatter(r,g,b);
    }
}

/**
* Returns a CSS string for an RGB code. Determines whether text color should be black or white 
* 
* @param {number} r - r value from 0 to 255
* @param {number} g - g value from 0 to 255
* @param {number} b - b value from 0 to 255
* @returns {str} A CSS style
*/
function final_formatter(r,g,b) {
  //formula adapted from
  //https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  let darkness_value = r * 0.299 + g * 0.587 + b * 0.114;
  let tc = (darkness_value > 150) ? 'black' : 'white';
  return `color:${tc};background-color:rgb(${r},${g},${b});`;
}