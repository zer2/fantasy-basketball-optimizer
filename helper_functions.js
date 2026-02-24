import { stat_styler_tertiary } from './styler_functions.js'

export function ExpandView(i, percentage_weighting_data){

    let evpopup = document.querySelector(`#PP${i}.playerpopup`);

    let evrows = document.querySelectorAll(`.EV${i}`);

    if (evrows[0].style.display == 'table-row') { 

        // remove the content from the rows
        for (let evrow of evrows ) {
            evrow.style.display = 'none'
            evrow.innerHTML = "" //remove the content so that it is ensured to be fresh next time it is rendered. Likely inefficient 
            }

        // reset the expansion marker
        evpopup.textContent = '▼'

        }

    else {
        
        // switch the style of the rows to display
        for (let evrow of evrows ) {
            evrow.style.display = 'table-row'
        }

        evpopup.textContent = '▲'

        // set up the dummy rows
        let dummyrows = document.querySelectorAll(`.EV${i}.dummyrow`);

        for (let drow of dummyrows ) {

            for (let j = 1; j < 12; j ++){
                var dummycell = drow.insertCell(-1)
                dummycell.className = 'mockheader'
                dummycell.textContent = '-'
            }

        }

        // set the main one
        let ev = document.querySelector(`.EV${i}.expandedview`);

        // create blankoverall score cell
        var dummyfirstcell = ev.insertCell(-1); 
        dummyfirstcell.className = 'mockheader'
        // add the row to the table 
        var dummyheadercell =  document.createElement('th');
        dummyheadercell.className = 'mockheader'
        dummyheadercell.textContent = 'Category weighting'
        ev.append(dummyheadercell);

        // create categorical cells
        var extra_data = Object.values(percentage_weighting_data)[i]

        console.log(extra_data)

        for (let value of extra_data) {
            var next_cell = ev.insertCell(-1); 
    
            next_cell.textContent = value.toFixed(1); 
            next_cell.style.cssText += stat_styler_tertiary(value, 5, 90)
            next_cell.className = 'categoricalhscore'; 
    
        }



    }

}