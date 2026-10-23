"""Boundary and numerical regressions for production Points data plumbing."""
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests
from pydantic import ValidationError

from backend.api.schemas import PointsConfigRequest
from backend.points import live
from backend.points.config import PointsScoringConfig, build_points_scoring_config, PointsConfigError
from backend.points.projections import normalise_projection_file
from backend.points.scoring import ESPN_STANDARD_RULES
from backend.points.schedule import ScheduleCalendar
from backend.points.roster_schedule import compute_multi_day_lineup_fp
from backend.platform_integration.helpers import PlatformPlayerLookup
from backend.platform_integration.integrations.espn import ESPNIntegration
from backend.platform_integration.base import PlatformConfig
from backend.services.build_agent import _run_points_pipeline
from backend.services.projection_parsing import parse_projection_upload
from backend.parameters import load_all_params
from backend.state.session import create_session, delete_session
from test_points_api_integration import client, sample_schedule, sample_points_session


def test_real_projection_normalizer_imports_and_scores_all_espn_components():
    stats = pd.DataFrame({'Points':[20.], 'Rebounds':[10.], 'Assists':[5.],
        'Steals':[2.], 'Blocks':[1.], 'Turnovers':[3.], 'Threes':[2.],
        'Field Goals Made':[8.], 'Field Goal Attempts':[15.],
        'Free Throws Made':[4.], 'Free Throw Attempts':[5.], 'Games Played':[70.]}, index=[10])
    config = PointsScoringConfig(points_input_mode='raw_stats', scoring_rules=ESPN_STANDARD_RULES)
    result = normalise_projection_file(stats, config)
    # 20+10+10+8+4-6+2+16-15+4-5: independent box-score calculation.
    assert result.per_game_fp[10] == 48
    assert result.total_fp[10] == 3360
    assert result.raw_stats_per_game is not None


@pytest.mark.parametrize('basis,values,expected', [
    ('per_game',[30.,-4.,5.],[30.,-4.,np.nan]),
    ('season_total',[1500.,-200.,5.],[30.,-4.,np.nan]),
    ('rest_of_season_total',[1500.,-200.,5.],[30.,-4.,np.nan]),
])
def test_direct_points_never_reweighted_and_zero_games_stay_missing(basis,values,expected):
    frame=pd.DataFrame({'FantasyPts':values,'Games Played':[50,50,0]},index=[1,2,3])
    result=normalise_projection_file(frame,PointsScoringConfig(projection_basis=basis))
    np.testing.assert_allclose(result.per_game_fp,expected,equal_nan=True)


def test_raw_totals_and_derived_misses_are_converted_once():
    frame=pd.DataFrame({'PTS':[1000.], 'FGA':[800.], 'FGM':[400.], 'Games Played':[50]},index=[1])
    config=build_points_scoring_config('raw_stats','season_total',{'PTS':1,'FGMiss':-1})
    result=normalise_projection_file(frame,config)
    assert result.per_game_fp[1] == 12
    assert result.total_fp[1] == 600


@pytest.mark.parametrize('rules', [{'DOUBLE_DOUBLE':5}, {'ESPN stat 42':3}])
def test_unsupported_scoring_cannot_silently_disappear(rules):
    with pytest.raises(ValueError,match='not present'):
        normalise_projection_file(pd.DataFrame({'PTS':[20],'Games Played':[82]}),
            PointsScoringConfig(points_input_mode='raw_stats',scoring_rules=rules))


def test_scoring_nan_is_not_a_zero_projection():
    result=normalise_projection_file(pd.DataFrame({'PTS':[np.nan],'Games Played':[82]}),
        PointsScoringConfig(points_input_mode='raw_stats',scoring_rules={'PTS':1}))
    assert np.isnan(result.per_game_fp.iloc[0])


@pytest.mark.parametrize('value',[float('inf'),float('nan')])
def test_nonfinite_scoring_rules_rejected(value):
    with pytest.raises(PointsConfigError,match='finite'):
        build_points_scoring_config('raw_stats','per_game',{'PTS':value})


def test_points_only_csv_reaches_direct_points_normalizer():
    parsed=parse_projection_upload(b'Player,Position,GP,FP\nNikola Jokic,C,50,1500\n',load_all_params()['NBA'])
    assert parsed.loc['Nikola Jokic','FantasyPts'] == 1500
    result=normalise_projection_file(parsed,PointsScoringConfig(projection_basis='season_total'))
    assert result.per_game_fp['Nikola Jokic'] == 30


@pytest.mark.parametrize('kwargs',[
    {'time_zone':'invalid/timezone'}, {'evaluation_window':['2026-10-28','2026-10-22']},
    {'evaluation_window':['2026-01-01','2027-02-01']}, {'bench_slots':-1},
    {'ir_slots':-1}, {'active_slots':{'PG':-1}}, {'lineup_lock_mode':'hourly'},
])
def test_invalid_analysis_settings_are_validation_errors(kwargs):
    with pytest.raises(ValidationError): PointsConfigRequest(**kwargs)


def lookup():
    return PlatformPlayerLookup({'Player One':1,'Player Two':2},
        {'Player One':1,'Player Two':2},{'Player One':1,'Player Two':2})


def professional_data():
    # A 00:30 UTC game belongs to the prior local date in Chicago.
    stamp=int(datetime(2026,10,23,0,30,tzinfo=timezone.utc).timestamp()*1000)
    game={'id':99,'homeProTeamId':1,'awayProTeamId':2,'date':stamp}
    teams=[{'id':1,'abbrev':'BOS','proGamesByScoringPeriod':{'3':[game]}},
           {'id':2,'abbrev':'PHO','proGamesByScoringPeriod':{'3':[game]}}]
    players=[{'id':100,'fullName':'Player One','proTeamId':1},
             {'id':101,'fullName':'Player Two','proTeamId':0}]
    return teams,players


def test_real_calendar_deduplicates_games_localizes_dates_and_maps_free_agents():
    teams,players=professional_data()
    calendar=live.calendar_from_espn(teams,players,lookup(),'America/Chicago')
    assert len(calendar._games) == 1
    assert calendar.get_team_game_dates('BOS','2026-10-22','2026-10-23') == ['2026-10-22']
    assert calendar.get_team_game_dates('PHX','2026-10-22','2026-10-23') == ['2026-10-22']
    assert calendar.get_player_team(1) == 'BOS'
    assert calendar.get_player_team(2) == 'FA'
    assert calendar.get_player_game_count(2,'2026-10-22','2026-10-23') == 0


def test_unusable_calendar_is_unavailable():
    with pytest.raises(live.PointsDataUnavailable,match='no usable games'):
        live.calendar_from_espn([{'id':1,'abbrev':'BOS'}],[],lookup(),'UTC')


def test_successful_raw_data_cached_without_session_ids(monkeypatch):
    monkeypatch.setattr(live,'_CACHE',{})
    teams,players=professional_data();calls=[]
    def get(url,**kwargs):
        calls.append((url,kwargs))
        assert kwargs['timeout'] == 20
        return SimpleNamespace(raise_for_status=lambda:None,
            json=lambda:players if url.endswith('/players') else {'settings':{'proTeams':teams}})
    monkeypatch.setattr(live.requests,'get',get)
    assert live.fetch_professional_data('NBA',2027) == (teams,players)
    assert live.fetch_professional_data('NBA',2027) == (teams,players)
    assert len(calls) == 2
    live.fetch_professional_data('NBA',2026)
    assert len(calls) == 4


def test_upstream_failure_never_generates_a_synthetic_calendar(monkeypatch):
    monkeypatch.setattr(live,'_CACHE',{})
    monkeypatch.setattr(live.requests,'get',lambda *a,**k:(_ for _ in ()).throw(requests.Timeout()))
    with pytest.raises(live.PointsDataUnavailable):live.fetch_professional_data('NBA',2027)
    assert not live._CACHE


def test_rolling_default_uses_selected_local_day(monkeypatch,sample_points_session):
    session=sample_points_session
    session.current_settings.pop('evaluation_window')
    session.current_settings['points_config']={'time_zone':'America/Chicago'}
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime(2026,10,23,0,30,tzinfo=timezone.utc).astimezone(tz)
    monkeypatch.setattr(live,'datetime',Clock)
    _,dates=live.resolve_points_calendar(session)
    assert dates == ['2026-10-22','2026-10-23','2026-10-24','2026-10-25','2026-10-26','2026-10-27','2026-10-28']


def test_custom_window_is_shared_by_both_engines(sample_points_session):
    session=sample_points_session
    session.current_settings['points_config']={'evaluation_window':['2024-11-05','2024-11-07']}
    _,dates=live.resolve_points_calendar(session)
    assert dates == ['2024-11-05','2024-11-06','2024-11-07']


@pytest.mark.parametrize('raw_names,expected',[
    ([' Team Grant ','League Unc\t'],['Team Grant','League Unc']),
    (['Team Grant ','Team Grant'],['Team Grant','Team Grant 2']),
    ([' ','Team 5'],['Team 5','Team 5 2']),
])
def test_whitespace_fix_matches_rosters_by_noncontiguous_team_ids(monkeypatch,raw_names,expected):
    integration=ESPNIntegration()
    settings={'rosterSettings':{'lineupSlotCounts':{'0':1,'11':1,'12':4,'13':3}},
              'draftSettings':{'type':'SNAKE'}, 'scoringSettings':{'scoringType':'H2H_CATEGORY'}}
    league=SimpleNamespace(teams=[SimpleNamespace(team_id=tid,team_name=name,
        roster=[SimpleNamespace(name=player,lineupSlot='PG')])
        for tid,name,player in zip([5,12],raw_names,['Player One','Player Two'])],
        espn_request=SimpleNamespace(get_league=lambda:{'settings':settings}))
    monkeypatch.setattr(integration,'_make_league',lambda _:league)
    shape=integration.fetch_league_shape('987::2027',None)
    assert shape.team_names == expected
    config=PlatformConfig('Retrieve from ESPN','987::2027',None,shape.teams_dict,'ESPN_NAME',shape.team_names)
    result=integration.get_draft_results(config,'Season Mode',lookup())
    assert result.player_assignments == dict(zip(expected,[[1],[2]]))
    assert shape.slot_counts['Util'] == 5  # Existing category convention retained.
    assert shape.points_metadata['active_slots']['Util'] == 1
    assert shape.points_metadata['bench_slots'] == 4
    assert shape.points_metadata['ir_slots'] == 3


def test_native_rules_import_real_weights_and_retain_unsupported_bonus():
    settings={'rosterSettings':{'lineupSlotCounts':{'0':1,'12':4,'13':3}},
        'scoringSettings':{'scoringType':'H2H_POINTS','scoringItems':[
            {'statId':1,'points':4},{'statId':2,'points':3},{'statId':42,'points':5}]}}
    metadata=ESPNIntegration._points_metadata(settings)
    assert metadata['scoring_system']=='points'
    assert metadata['scoring_rules']=={'BLK':4,'STL':3,'ESPN stat 42':5}


def test_weekly_lineup_cannot_rotate_bench_daily():
    schedule=ScheduleCalendar.from_dict({'BOS':['2026-10-26'],'LAL':['2026-10-27']},{1:'BOS',2:'LAL'})
    args=([1,2],schedule,['2026-10-26','2026-10-27'],{1:['PG'],2:['PG']},{1:20,2:30},{'Util':1})
    assert compute_multi_day_lineup_fp(*args)[0] == 50
    schedule.lineup_lock_mode='weekly'
    total,daily=compute_multi_day_lineup_fp(*args)
    assert total == 30
    assert daily['2026-10-26'].total_expected_fp == 0


def test_points_gscores_endpoint_is_compatible_and_empty(client,sample_points_session):
    response=client.get(f'/sessions/{sample_points_session.id}/g-scores')
    assert response.status_code == 200
    assert response.json()['g_scores'] == []


def test_explicit_drop_is_one_transaction(client,sample_points_session):
    session=sample_points_session
    assignments={'A':[1,2,3,4,5,6],'B':[11,12,13,14,15,16]}
    response=client.post(f'/sessions/{session.id}/evaluate',json={
        'player_assignments':assignments,'my_team_id':'A','drop_player_id':1,'candidate_limit':4})
    assert response.status_code == 200,response.text
    result=response.json()
    for candidate in result['candidates']:
        assert candidate['drop_player_id'] == 1
        assert candidate['projected_fp_total'] == pytest.approx(candidate['projected_fp_per_game']*candidate['candidate_games'])
    assert assignments['A']==[1,2,3,4,5,6]
    invalid=client.post(f'/sessions/{session.id}/evaluate',json={
        'player_assignments':assignments,'my_team_id':'A','drop_player_id':101})
    assert invalid.status_code == 400


def test_schedule_failure_is_recoverable_http503(client,sample_points_session,monkeypatch):
    monkeypatch.setattr(live,'resolve_points_calendar',lambda *a:(_ for _ in ()).throw(live.PointsDataUnavailable('Retry schedule retrieval.')))
    sid=sample_points_session.id
    r=client.post(f'/sessions/{sid}/evaluate',json={'player_assignments':{'A':[1],'B':[11]},'my_team_id':'A'})
    assert r.status_code == 503
    trade={'player_assignments':{'A':[1],'B':[11]},'my_team':'A','their_team':'B','my_trade':[1],'their_trade':[11]}
    assert client.post(f'/sessions/{sid}/trade/analyze',json=trade).status_code == 503


def test_invalid_patch_does_not_destroy_last_valid_session(client,sample_points_session):
    session=sample_points_session;old_agent=session.agent;old_settings=session.current_settings
    session.v1_clean=pd.DataFrame({"Points":[20],"Games Played":[82]},index=[1])
    response=client.patch(f'/sessions/{session.id}',json={'from_step':4,'league':{'points_config':{
        'points_input_mode':'raw_stats','scoring_rules':{'ESPN stat 42':5}}}})
    assert response.status_code == 400
    assert session.agent is old_agent
    assert session.current_settings is old_settings
    assert client.get(f'/sessions/{session.id}/g-scores').status_code == 200


def test_invalid_roster_capacity_is_rejected_before_scoring():
    session=create_session()
    try:
        session.current_settings={'sport':'NBA','n_picks':2,'slot_counts':{'Util':2},
            'points_config':{'points_input_mode':'raw_stats','scoring_rules':{'PTS':1},'bench_slots':1}}
        with pytest.raises(ValueError,match='exceed'):_run_points_pipeline(session,1)
    finally:delete_session(session.id)


@pytest.mark.parametrize('send,receive',[([1],[11]),([1,2],[11]),([1,2],[11,12])])
def test_trade_totals_match_independent_daily_top_starter(client,sample_points_session,send,receive):
    session=sample_points_session
    session.agent.config=PointsScoringConfig(active_slots={'Util':1},bench_slots=3)
    session.current_settings.update(n_picks=4,waiver_baseline_fp_per_game=0.0)
    rosters={'A':[1,2],'B':[11,12]}
    dates=session.current_settings['evaluation_window']
    def expected(roster):
        total=0.0
        for day in dates:
            playing=[p for p in roster if session.schedule_calendar.is_player_playing_on_date(p,day)]
            total+=max([0.0]+[session.agent.per_game_fp[p] for p in playing])
        return total
    result=client.post(f'/sessions/{session.id}/trade/analyze',json={'player_assignments':rosters,
        'my_team':'A','their_team':'B','my_trade':send,'their_trade':receive,'position_check':False})
    assert result.status_code==200,result.text
    data=result.json();assert data['is_legal'],data
    post_a=[p for p in rosters['A'] if p not in send]+receive
    post_b=[p for p in rosters['B'] if p not in receive]+send
    assert data['team_a_pre_fp']==expected(rosters['A'])
    assert data['team_b_pre_fp']==expected(rosters['B'])
    assert data['team_a_post_fp']==expected(post_a)
    assert data['team_b_post_fp']==expected(post_b)
    assert data['team_a_net_delta_fp']==expected(post_a)-expected(rosters['A'])


@pytest.mark.parametrize('basis,fga,fta,games',[('per_game',20.,10.,50),('season_total',1000.,500.,50)])
def test_scoring_derives_makes_from_source_percentages_without_double_scaling(basis,fga,fta,games):
    frame=pd.DataFrame({'Field Goal %':[0.5],'Field Goal Attempts':[fga],
        'Free Throw %':[0.8],'Free Throw Attempts':[fta],'Games Played':[games]})
    result=normalise_projection_file(frame,PointsScoringConfig(points_input_mode='raw_stats',
        projection_basis=basis,scoring_rules={'FGM':2,'FTM':1}))
    assert result.per_game_fp.iloc[0] == 28


def test_live_rostered_missing_mapping_is_not_misreported_as_no_games(sample_points_session):
    import time
    session=sample_points_session
    session.points_calendar_key=('NBA',live.resolve_season(session),'America/New_York',session.player_pool_hash)
    session.points_calendar_loaded_at=time.monotonic()
    session.schedule_calendar.remove_player(1)
    with pytest.raises(live.PointsDataUnavailable,match='Player 1'):
        live.resolve_points_calendar(session,[1])



def test_mode_switch_rebuilds_category_intermediates_after_a_points_only_session(sample_points_session,monkeypatch):
    from backend.services import session_management
    session=sample_points_session;calls=[]
    def build(session,from_step,**kwargs):calls.append(from_step)
    monkeypatch.setattr(session_management,'build_agent',build)
    session_management.apply_patch(session,{'scoring_system':'categories','scoring_format':'Head to Head',
        'most_categories_weight':0,'tiebreaker_category':None,'categories':['Points','Rebounds','Assists']},4)
    assert calls == [3]


def test_multiple_empty_projection_slots_do_not_count_as_duplicate_real_players():
    schedule=ScheduleCalendar.from_dict({'BOS':['2026-10-26']},{1:'BOS'})
    assert compute_multi_day_lineup_fp([1,-1,-1],schedule,['2026-10-26'],{1:['PG']},{1:30},{'Util':1})[0] == 30



def test_raw_season_csv_is_converted_exactly_once_even_below_heuristic_ceilings(monkeypatch):
    from backend.services import build_agent as module
    monkeypatch.setattr(module,'attach_player_ids_by_name',lambda df:df.assign(player_id=100))
    session=create_session()
    try:
        session.current_settings={'sport':'NBA','n_picks':1,'n_drafters':2,'slot_counts':{'Util':1},
            'scoring_system':'points','data_source_type':'csv','points_config':{
                'points_input_mode':'raw_stats','projection_basis':'season_total','scoring_rules':{'PTS':1,'REB':1,'AST':1}}}
        module.build_agent(session,csv_bytes=b'Player,Position,GP,PTS,REB,AST\nExample,PG,2,80,20,10\n')
        assert session.agent.per_game_fp[100] == 55
        assert session.points_projections.total_fp[100] == 110
    finally:delete_session(session.id)



def test_explicit_total_basis_is_honored_for_uploaded_blend_sources(monkeypatch):
    from backend.api.routers import sessions
    monkeypatch.setattr(sessions,'_require_upload',lambda _: {'bytes':b'Player,Position,GP,PTS,REB,AST\nExample,PG,2,80,20,10\n'})
    frame=sessions._resolve_uploaded_dfs(['test-source'],load_all_params()['NBA'],
        PointsConfigRequest(points_input_mode='raw_stats',projection_basis='season_total',scoring_rules={'PTS':1}))['test-source']
    assert frame.loc['Example','Points'] == 40



def test_uploaded_pool_cache_key_changes_when_source_units_change():
    from backend.services.build_agent import _build_v0_cache_key
    settings={'sport':'NBA','data_source_type':'projections','blend_weights':{'file-a':1},'custom_data_ids':['file-a'],
        'points_config':{'points_input_mode':'raw_stats','projection_basis':'per_game'}}
    per_game=_build_v0_cache_key(settings)
    settings['points_config']['projection_basis']='season_total'
    assert _build_v0_cache_key(settings) != per_game
