import { stat_styler_primary } from './styler_functions.js'
import { ExpandView } from './helper_functions.js'

//set up the table 
let table = document.getElementById('realtable')

for (let i=1; i< table.rows.length; i++) {

    if (i % 2 == 1) {

        let row = table.rows[i]

        for (let j=2; j< row.cells.length; j++) {
            let cell = row.cells[j]
            let value = Number(cell.textContent)
            cell.style.cssText += stat_styler_primary(value, 5, 50)
                                                }
                    }
    else {
    }

}

let button = document.querySelector(`#PP1.playerpopup`);
button.addEventListener("click",() => ExpandView(1));