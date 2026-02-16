const title = 'Fantasy Basketball Optimizer 2';

let titleArea = document.getElementById('header');

titleArea.textContent = title;

let table_dummy = [-1,-2,3,-1,2]
let table_dummy_super = []
table_dummy.forEach(item => {table_dummy_super.push(stat_styler_primary(item, 50,0))})

let tableArea = document.getElementById('table');
tableArea.style = table_dummy_super[0]

function stat_styler_primary(value, multiplier, middle) {

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

function final_formatter(r,g,b) {
  //formula adapted from
  //https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  let darkness_value = r * 0.299 + g * 0.587 + b * 0.114;
  let tc = (darkness_value > 150) ? 'black' : 'white';
  return `color:${tc};background-color:rgb(${r},${g},${b})`;
}

