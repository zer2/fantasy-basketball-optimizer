export function ExpandView(i){

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