export function ExpandView(i){

    console.log(i)
    //this should also toggle the button from down to up 
    let ev = document.querySelector(`#EV${i}.expandedview`);
    let evpopup = document.querySelector(`#PP${i}.playerpopup`);

    if (ev.style.display == 'block') { 
        ev.style.display = 'none'
        evpopup.textContent = '▼'
    }
    else {

        ev.style.display = 'block'
        evpopup.textContent = '▲'
    }

}