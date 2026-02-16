const title = 'Fantasy Basketball Optimizer 2';

let titleArea = document.getElementById('header');

titleArea.textContent = title;

/**
* Determines a CSS style for a cell in the main display table, based on the difference between the value and a pre-defined middle
* 
* @param {number} value - The value of the cell
* @param {number} multiplier - The multiplier applied to (value - middle)
* @param {number} middle - The value to which the cell's value is compared
* @returns {str} A CSS style for the table cell
*/
function stat_styler_primary(value
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
  return `color:${tc};background-color:rgb(${r},${g},${b});font-weight:500;text-align:right;padding:5px`;
}

let table = document.getElementById('realtable')

for (let i=1; i< table.rows.length; i++) {
    let row = table.rows[i]

    first_cell = row.cells[0]
    first_cell.style.cssText = "background-color:rgb(14,17,23);color:darkgrey;font-weight:400;text-align:left" 

    second_cell = row.cells[1]
    second_cell.style.cssText = "background-color:#2a2a33;color:white;;font-weight:400;text-align:right;padding:5px" 

    for (let j=2; j< row.cells.length; j++) {
        let cell = row.cells[j]
        let value = Number(cell.textContent)
        cell.style.cssText = stat_styler_primary(value, 50, 0)
    }
}

first_row = table.rows[0]
for (let cell of first_row.cells){
    cell.style.cssText = "background-color:#161721;color:darkgrey;font-weight:500;text-align:left" 
}
