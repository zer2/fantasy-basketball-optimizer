import { stat_styler_primary } from './styler_functions.js'
import { ExpandView } from './helper_functions.js'


// Load player data

//set up the table 
let table = document.getElementById('realtable')

// this should be incorporated into the second loop so we only loop once 
const player_table_data = {
    "Nikola Jokic (C)" : [53.7, 66.2, 14.2, 33.9, 66.3, 73.4, 72.3, 59.7, 67.7, 29.7]
    ,"Shai Gilgeous-Alexander (PG)" : [53.0, 40.8,71.9,65.4,58.4,10.8, 55.2, 59.1, 57.2, 58.2]
    ,"Victor Wembanyama (C)" : [52.1, 51.3, 54.2, 66.2, 41.7, 57.4, 9.6, 39.2, 73.2, 76.2]
}

for (const [i, [player_name, player_data]] 
     of Object.entries(player_table_data).entries()){

    var row = table.insertRow(-1);

    // create header cell
    var player_header_cell = document.createElement('th');
    player_header_cell.innerHTML = `
                    <div class = 'playerheaderdiv'>
                    <div style = "width:80%">
                                    ${player_name}
                    </div>
                    <div style = "width:20%">
                        <button class = 'playerpopup' id = 'PP${i}'>
                            ▼
                        </button>
                    </div>
                </div>`
    player_header_cell.className = 'playerheader'
    row.append(player_header_cell);

    var button = player_header_cell.querySelector(`#PP${i}.playerpopup`);
    button.addEventListener("click",() => ExpandView(i));

    // create overall score cell
    var first_cell = row.insertCell(-1); 
    first_cell.className = 'overallhscore'
    first_cell.textContent = player_data[0].toFixed(1); 

    // create categorical cells
    for (let value of player_data.slice(1)) {
        var next_cell = row.insertCell(-1); 

        next_cell.textContent = value.toFixed(1); 
        next_cell.style.cssText += stat_styler_primary(value, 5, 50)
        next_cell.className = 'categoricalhscore'; 

    }

    // add expansion row
    var expanded_row = table.insertRow(-1);
    expanded_row.className = "expandedview"
    expanded_row.id = `EV${i}`
    expanded_row.innerHTML = "Boo"

}


