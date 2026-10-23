// Points inputs follow the same sidebar controls and apply flow as category settings.
import { makeCustomSelect } from '../custom_select.js'
import { makeLabel } from '../helper_functions.js'
import { pref, savePref } from '../preferences.js'
import { isPointsMode } from '../app_state.js'
import { getLeagueSettings, getPlatformConfig, getPointsMetadata } from './league_settings.js'
import { getSlotCounts } from './slot_counts.js'

export interface PointsConfig {
    points_input_mode: string
    projection_basis: string
    scoring_rules: Record<string, number>
    active_slots: Record<string, number>
    bench_slots: number
    ir_slots: number
    lineup_lock_mode: string
    evaluation_window: [string, string] | null
    time_zone: string
}
export interface PointsMetadata {
    scoring_system: string
    scoring_rules: Record<string, number>
    active_slots: Record<string, number>
    bench_slots: number
    ir_slots: number
    lineup_lock_mode: string
}
const ESPN = {PTS:1, REB:1, AST:2, STL:4, BLK:4, TO:-2, '3PM':1, FGM:2, FGA:-1, FTM:1, FTA:-1}
const SLOTS = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'Util']
let host: HTMLElement | null = null
let panel: HTMLElement | null = null
let inputs = new Map<string, HTMLInputElement>()
let selectValues = new Map<string, () => string>()
let selectSetters = new Map<string, (value:string)=>void>()

function preferenceKey(): string {
    const config = getPlatformConfig()
    return `points-settings:${config ? getLeagueSettings().platform + ':' + config.league_id : 'manual'}`
}
function defaults(): PointsConfig {
    const meta = getPointsMetadata()
    const slots = meta?.active_slots ?? getSlotCounts()
    return {points_input_mode:'raw_stats', projection_basis:'per_game', scoring_rules: {},
        active_slots: {...slots}, bench_slots: meta?.bench_slots ?? Math.max(0, getLeagueSettings().n_picks - Object.values(slots).reduce((a,b)=>a+b,0)),
        ir_slots:meta?.ir_slots ?? 0, lineup_lock_mode:meta?.lineup_lock_mode ?? 'pregame_daily',
        evaluation_window:null, time_zone:Intl.DateTimeFormat().resolvedOptions().timeZone}
}
function localDate(date: Date): string {
    return `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`
}
function input(id: string, label: string, value: string | number, type = 'number'): HTMLInputElement {
    panel!.append(makeLabel(id, label))
    const el = document.createElement('input')
    el.id=id;el.type=type;el.value=String(value);el.className='sidebar-input'
    if(type==='number')el.step='any'
    inputs.set(id,el);panel!.append(el)
    return el
}
function select(id: string, label: string, options: [string,string][], value: string): void {
    panel!.append(makeLabel(id,label))
    const widget=makeCustomSelect(id,options.map(([value,label])=>({value,label})),value)
    selectValues.set(id,widget.getValue);selectSetters.set(id,widget.setValue);panel!.append(widget.element)
}
function readConfig(): PointsConfig {
    const config=defaults()
    const choice=selectValues.get('fc-points-preset')?.() ?? ''
    if(!choice)throw new Error('Choose a Points scoring preset, Custom weights, or Direct fantasy points.')
    const n=(id:string):number=>{
        const el=inputs.get(id)!
        if(el.value.trim()===''||!Number.isFinite(Number(el.value)))throw new Error('Enter a valid number for every Points setting.')
        return Number(el.value)
    }
    config.points_input_mode=choice==='direct'?'projected_points':'raw_stats'
    config.projection_basis=choice==='direct'?(selectValues.get('fc-points-basis')?.()??'per_game'):'per_game'
    config.scoring_rules=choice==='direct'?{}:Object.fromEntries(Object.keys(ESPN).map(k=>[k,n('fc-points-weight-'+k)]).filter(([,v])=>v!==0))
    if(choice==='native')config.scoring_rules={...getPointsMetadata()!.scoring_rules}
    if(choice!=='direct'&&!Object.keys(config.scoring_rules).length)throw new Error('Set at least one nonzero scoring weight.')
    config.active_slots=Object.fromEntries(SLOTS.map(k=>[k,n('fc-points-slot-'+k)]))
    config.bench_slots=n('fc-points-bench');config.ir_slots=n('fc-points-ir')
    const counts=[...Object.values(config.active_slots),config.bench_slots,config.ir_slots]
    if(counts.some(v=>!Number.isInteger(v)||v<0))throw new Error('Lineup counts must be nonnegative whole numbers.')
    if(Object.values(config.active_slots).reduce((a,b)=>a+b,0)+config.bench_slots>getLeagueSettings().n_picks)throw new Error('Starters plus bench exceed this league’s roster size.')
    if(!Object.values(config.active_slots).some(v=>v>0))throw new Error('Choose at least one active scoring slot.')
    config.lineup_lock_mode=selectValues.get('fc-points-lock')!()
    const start=inputs.get('fc-points-start')!.value,end=inputs.get('fc-points-end')!.value
    if(!start||!end||end<start||(Date.parse(end)-Date.parse(start))/86400000>365)throw new Error('Choose a valid analysis date range of up to 366 days.')
    config.evaluation_window=[start,end]
    return config
}
export function getPointsConfig(): PointsConfig | undefined {
    return isPointsMode()?readConfig():undefined
}
export function isPointsSettingsValid(): boolean {
    if(!isPointsMode())return true
    try{readConfig();return true}catch{return false}
}
export function refreshPointsVisibility(): void {
    if(panel)panel.hidden=!isPointsMode()
}
export function renderPointsSettings(container: HTMLElement): void {
    host=container
}
export function refreshPointsControls(): void { draw() }
function draw(): void {
    if(!host)return
    panel?.remove();inputs=new Map();selectValues=new Map();selectSetters=new Map()
    panel=document.createElement('div');panel.id='fc-points-settings';host.append(panel)
    const stored=pref<(PointsConfig & {preset:string})|null>(preferenceKey(),null)
    const meta=getPointsMetadata()
    const config=stored??{...defaults(),scoring_rules:meta?.scoring_system==='points'?meta.scoring_rules:{}}
    const choice=stored?.preset??(meta?.scoring_system==='points'?'native':'')
    let customWindow=stored?.evaluation_window!=null
    select('fc-points-preset','Points scoring',[['','Choose scoring…'],...(meta?.scoring_system==='points'?[['native','Imported league rules'] as [string,string]]:[]),['espn','ESPN standard'],['custom','Custom weights'],['direct','Direct fantasy points']],choice)
    const rules=config.scoring_rules
    const weights=document.createElement('div');weights.id='fc-points-weights';weights.className='slot-grid';panel.append(weights)
    const weightColumns=[document.createElement('div'),document.createElement('div')];weights.append(...weightColumns)
    // Keep native league coefficients visible, including any unsupported rule warning.
    for(const [index,key] of Object.keys(ESPN).entries()){
        const el=input('fc-points-weight-'+key,key,rules[key]??0)
        weightColumns[index%2].append(el.previousSibling!,el)
    }
    if(meta?.scoring_system==='points'&&!stored){
        const note=document.createElement('p');note.className='sidebar-caption'
        note.textContent='Weights imported from this ESPN Points league.';panel.append(note)
        const unsupported=Object.keys(meta.scoring_rules).filter(k=>!(k in ESPN))
        if(unsupported.length){note.textContent+=' Unsupported rules: '+unsupported.join(', ')+'. Use supported custom rules explicitly.'}
    }
    select('fc-points-basis','Direct points basis',[['per_game','Per game'],['season_total','Season total'],['rest_of_season_total','Rest-of-season total']],config.projection_basis)
    const slotGrid=document.createElement('div');slotGrid.className='slot-grid';panel.append(slotGrid)
    const slotColumns=[document.createElement('div'),document.createElement('div')];slotGrid.append(...slotColumns)
    for(const [index,key] of SLOTS.entries()){const el=input('fc-points-slot-'+key, key+' starters',config.active_slots[key]??0);el.min='0';el.step='1';slotColumns[index%2].append(el.previousSibling!,el)}
    for(const [id,label,value] of [['fc-points-bench','Bench spots',config.bench_slots],['fc-points-ir','IR spots',config.ir_slots]] as const){const el=input(id,label,value);el.min='0';el.step='1'}
    select('fc-points-lock','Lineup changes',[['pregame_daily','Daily'],['weekly','Weekly']],config.lineup_lock_mode)
    const today=new Date(),last=new Date();last.setDate(today.getDate()+6)
    input('fc-points-start','Analysis starts',config.evaluation_window?.[0]??localDate(today),'date')
    input('fc-points-end','Analysis ends',config.evaluation_window?.[1]??localDate(last),'date')
    const zone=document.createElement('p');zone.className='sidebar-caption';zone.textContent='Dates use '+config.time_zone+'.';panel.append(zone)
    const error=document.createElement('div');error.id='fc-points-error';error.className='sidebar-error';error.setAttribute('role','status');panel.append(error)
    const basisWrap=panel.querySelector<HTMLElement>('[data-testid="fc-points-basis-wrapper"]')!
    const basisLabel=panel.querySelector<HTMLElement>('label[for="fc-points-basis"]')!
    const showBasis=()=>{const display=selectValues.get('fc-points-preset')!()==='direct'?'':'none';basisWrap.style.display=display;basisLabel.style.display=display}
    showBasis()
    const validate=()=>{try{readConfig();error.textContent=''}catch(e){error.textContent=(e as Error).message}}
    panel.addEventListener('change',event=>{
        const target=event.target as HTMLElement
        if(target.id.startsWith('fc-points-weight-')&&selectValues.get('fc-points-preset')!()!=='custom')selectSetters.get('fc-points-preset')!('custom')
        const preset=selectValues.get('fc-points-preset')!()
        if((event.target as HTMLElement).closest('[data-testid="fc-points-preset-wrapper"]')){
            if(preset==='espn')for(const [key,value] of Object.entries(ESPN))inputs.get('fc-points-weight-'+key)!.value=String(value)
            weights.style.display=preset==='direct'||preset===''?'none':''
        }
        showBasis();validate()
        if((event.target as HTMLElement).id==='fc-points-start'||(event.target as HTMLElement).id==='fc-points-end')customWindow=true
        if(isPointsSettingsValid()){const config=readConfig();savePref(preferenceKey(),{...config,evaluation_window:customWindow?config.evaluation_window:null,preset})}
    })
    weights.style.display=choice==='direct'||choice===''?'none':'';validate();refreshPointsVisibility()
}
document.addEventListener('points-league-metadata',()=>{
    draw()
    const metadata=getPointsMetadata()
    if(metadata?.scoring_system==='points')document.dispatchEvent(new Event('native-points-league'))
})
