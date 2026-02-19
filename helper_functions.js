export function ExpandView(i){
    //this should also toggle the button from down to up 
    let ev = document.querySelector(`#EV${i}.expandedview`);

    console.log(ev.style.display)
    if (ev.style.display == '') { 
        ev.style.display = 'block'
    }
    else {
        ev.style.display = ''
    }
    console.log(ev.style.display)

}