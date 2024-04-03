
import numpy as np
import pandas as pd
from src.run_algorithm import HAgent, SimpleAgent
from src.helper_functions import get_categories, stat_styler, styler_a, rotate
from src.process_player_data import process_player_data
from src.get_data import get_player_metadata
from src.tabs import make_team_tab
import copy
import datetime
import streamlit as st

import statsmodels.api as sm
import plotly.express as px

#ZR: this can be cleaned up probably
cols = ['Free Throws','Free Throw Attempts','Field Goals','Field Goal Attempts'
        ,'Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
        

def make_weekly_df(season_df : pd.DataFrame):
    """Prepares a stat dataframe and a position series for a season"""

    season_df['Date'] = pd.to_datetime(season_df['Date'])
    season_df['Week'] = season_df['Date'].dt.isocalendar()['week']

    #make sure we aren't missing any weeks when a player didn't play
    weekly_df_index = pd.MultiIndex.from_product([pd.unique(season_df['Player'])
                                                 ,pd.unique(season_df['Week'])]
                                                 ,names = ['Player','Week'])
    weekly_df = season_df.groupby(['Player','Week'])[cols].sum()
    season_df = pd.DataFrame(weekly_df, index = weekly_df_index ).fillna(0)
    
    return season_df

def round_robin_opponent(t : int
                         , w : int
                         , n : int =12) -> int: 
    """Calculates the opposing team number based on a round robin schedule

    Based on the circle method as defined by wikipedia
    https://en.wikipedia.org/wiki/Round-robin_tournament#Circle_method
    
    Args:
        t: team number, from 0
        w: week number, from 0
        n: number of teams - must be an even number
        
    Returns:
        The opposing team number for team t during week w
    """
    if t == 0: #position 0 remains fixed, and the other teams rotate around their (n - 1) spots
        return ((n - 2 - w) % (n - 1) ) + 1
    elif ((t + w) % (n-1) ==0): # in spot (n-1) of the non-zero spots, the opponent is 0 
        return 0 
    else: #we calculate the current position of team, infer the opponent's position, then calculate the opposing team
        res = (((n - 1 - (t + w) % (n - 1)) % (n - 1))- w) % (n - 1)
        return (n - 1) if res == 0 else res

def run_draft(agents
              , n_rounds : int) -> tuple:
    """Run a snake draft

    Snake drafts wrap around like 1 -> 2 -> 3 -> 3 -> 2 -> 1 -> 1 -> 2 -> 3 etc. 
    
    Args:
        agents: list of Agents, which are required to have make_pick() methods
        n_rounds: number of rounds to do of the snake draft. Each drafter will get n_rounds * 2 players
        
    Returns:
        dictionary of player assignments with the structure
         {team_number : list of players } 
        
    """
    
    player_assignments = {n : [] for n in range(len(agents))}

    times = {n : [] for n in range(len(agents))}
    
    for i in range(n_rounds):
        
        if i % 2 ==0:
            for j in range(len(agents)):

                agent = agents[j]

                start = datetime.datetime.now()
                chosen_player = agent.make_pick(player_assignments, j)
                timediff = datetime.datetime.now() - start

                times[j] = times[j] + [timediff]
                player_assignments[j] = player_assignments[j] + [chosen_player]
        
        else:
           
            for j in reversed(range(len(agents))):
                agent = agents[j]

                start = datetime.datetime.now()
                chosen_player = agent.make_pick(player_assignments, j)
                timediff = datetime.datetime.now() - start

                times[j] = times[j] + [timediff]
                player_assignments[j] = player_assignments[j] + [chosen_player]
                
    return player_assignments, agents, times

def run_multiple_seasons(teams : dict[list]
                         , season_df : pd.DataFrame
                         , n_seasons : int = 100 
                         , n_weeks : int = 20
                         , scoring_format : str = 'Head to Head: Each Category'
                         , return_detailed_results : bool = False ):
    """Simulate multiple seasons with the same drafters 
    
    Weekly performances are sampled from a dataframe of real season performance
    Teams win weeks by winning more categories than their opponents. They win seasons by winning the
    most weeks of all players 
    
    Args:
        teams: player assignment dict, as produced by the run_draft() functoin
        season_df: dataframe of weekly numbers per players. These will be sampled to simulate seasons
        n_seasons: number of seasons to simulate
        n_weeks: number of weeks per season
        scoring_format: Each Category, Most Categories, or Rotisserie 
        return_cat_results: If True, return detailed results on category wins 
        
    Returns:
        Series of winning percentages with the structure
         team_number : winning_fraction  
    """

    #reverse the mapping
    t = pd.Series(teams).explode()
    team_dict = dict(zip(t,t.index))

    #create a version of the essential info dataframe which incorporate team information for this season
    season_df = season_df.reset_index()
    season_df.loc[:,'Team'] = season_df['Player'].map(team_dict) 
    season_df = season_df.dropna(subset = ['Team'])

    #use sampling to simulate many seasons at the same time
    #assuming each season has 11 weeks, we need 11 * n total rows of data per player
    #ZR: Note that for now a "week" of data is just one game per player
    #in real basketball multiple games are played per week, so we need to adjust for that 
    performances = season_df.groupby('Player').sample(n_weeks*n_seasons, replace = True)
    performances.loc[:,'Week'] = performances.groupby('Player').cumcount()
    performances.loc[:,'Season'] = performances['Week'] // n_weeks #integer division seperates weeks in groups 

    if scoring_format in ('Head to Head: Most Categories','Head to Head: Each Category'):

        #total team performances are simply the sum of statistics for each player 
        team_performances = performances.groupby(['Season','Team','Week']).sum()
        team_performances['Free Throw %'] = (team_performances['Free Throws']/team_performances['Free Throw Attempts']).fillna(0)
        team_performances['Field Goal %'] = (team_performances['Field Goals']/team_performances['Field Goal Attempts']).fillna(0)
        
        #for all categories except turnovers, higher numbers are better. So we invert turnovers 
        team_performances['Turnovers'] = - team_performances['Turnovers'] 
        
        team_performances = team_performances[get_categories()] #only want category columns
    
        #we need to map each team to its opponent for the week. We do that with a formula for round robin pairing
        opposing_team_schedule = [(s,round_robin_opponent(t,w),w) for s, t, w in team_performances.index]
        opposing_team_performances = team_performances.loc[opposing_team_schedule]

        cat_wins = np.greater(team_performances.values,opposing_team_performances.values)
        cat_ties = np.equal(team_performances.values,opposing_team_performances.values)
        
        tot_cat_wins = cat_wins.sum(axis = 1)
        tot_cat_ties = cat_ties.sum(axis = 1)
        
        if scoring_format == 'Head to Head: Most Categories':
            team_performances.loc[:,'Tie'] = tot_cat_wins + tot_cat_ties/2 == len(get_categories())/2
            team_performances.loc[:,'Win'] = tot_cat_wins + tot_cat_ties/2 > len(get_categories())/2
        elif scoring_format == 'Head to Head: Each Category':
            team_performances.loc[:,'Tie'] = tot_cat_ties
            team_performances.loc[:,'Win'] = tot_cat_wins
            
        team_results = team_performances.groupby(['Team','Season']).agg({'Win' : sum, 'Tie' : sum})

        #a team cannot win the season if it has fewer wins than any other team 
        most_wins = team_results.groupby('Season')['Win'].transform('max')
        winners = team_results[team_results['Win'] == most_wins]

        #among the teams with the most wins, ties are a tiebreaker 
        most_ties = winners.groupby('Season')['Tie'].transform('max')
        winners_after_ties = winners[winners['Tie'] == most_ties]

    elif scoring_format == 'Rotisserie':

        team_performances = performances.drop(columns = ['Player','Week']) \
                                            .groupby(['Season','Team']).sum()

        team_performances.loc[:,'Free Throw %'] = (team_performances['Free Throws']/team_performances['Free Throw Attempts']).fillna(0)
        team_performances.loc[:,'Field Goal %'] = (team_performances['Field Goals']/team_performances['Field Goal Attempts']).fillna(0)
        
        #for all categories except turnovers, higher numbers are better. So we invert turnovers 
        team_performances['Turnovers'] = - team_performances['Turnovers'] 
        season_points_by_category = team_performances.groupby('Season')[get_categories()].rank()

        season_points = season_points_by_category.sum(axis = 1)
        winners_after_ties = season_points[season_points == season_points.groupby('Season').transform('max')]

        winners_after_ties = pd.DataFrame({'Score' : winners_after_ties})

    #assuming that payouts are divided when multiple teams are exactly tied, we give fractional points 
    winners_after_ties.loc[:,'Winner Points'] = 1
    season_counts = winners_after_ties.groupby('Season')['Winner Points'].transform('count')
    winners_after_ties.loc[:,'Winner Points Adjusted'] = 1/season_counts
        
    wins_by_teams = winners_after_ties.groupby('Team')['Winner Points Adjusted'].sum()/winners_after_ties['Winner Points Adjusted'].sum()
        
    if not return_detailed_results:
        return wins_by_teams
    else:
        if scoring_format in ('Head to Head: Most Categories','Head to Head: Each Category'):

            cat_win_df = pd.DataFrame(cat_wins, columns = get_categories(), index = team_performances.index)
            cat_tie_df = pd.DataFrame(cat_ties, columns = get_categories(), index = team_performances.index)
            
            cat_wins_agg = cat_win_df.groupby('Team').mean()
            cat_wins_agg = pd.concat({'Win' : cat_wins_agg}, names = ['Result'])
            
            cat_ties_agg = cat_tie_df.groupby('Team').mean()
            cat_ties_agg = pd.concat({'Tie' : cat_ties_agg}, names = ['Result'])

            results_agg = pd.concat([cat_wins_agg, cat_ties_agg])
            results_agg = results_agg.reorder_levels(['Team','Result'])

        elif scoring_format == 'Rotisserie':

            results_agg = pd.DataFrame(season_points_by_category.groupby('Team').mean())
            results_agg.loc[:,'Result'] = 'Mean Points'
            results_agg = results_agg.reset_index().set_index(['Team','Result'])

        return wins_by_teams, results_agg

@st.cache_data()
def try_strategy(_primary_agent
                 , _default_agent
                 , n_drafters : int
                 , n_picks : int
                 , season_df : pd.DataFrame
                 , n_seasons : int
                 , n_primary : int
                 , scoring_format : str
                 ) -> tuple:
    """Try a particular strategy (enacted by the primary agent) against a field of default agents
    
    The strategy is tried with the primary agent at each draft seat. If n_primary > 1, successive
    agents also use the primary agent strategy. For example if n_primary is 2, the first two drafters 
    use the primary agent strategy, then the second and third drafters, etc. 
    
    Args:
        default_agent: default strategy 
        primary_agent: strategy to test
        n_drafters: int, number of drafters
        n_picks: int, number of picks each drafter gets 
        season_df: dataframe with rows for (week,player) pairs. Used to simulate an actual season
        n_seasons: number of seasons to simulate
        n_primary: number of successive seats to use the primary strategy
        scoring_format:
        
    Returns:
        victory_res: list of average overall win rates
        detailed_res: Dataframe of win/tie rates by category and draft seat (from the primary agent perspective)
        team_dict: dict of form {seat -> list of players chosen by the drafter in that seat}
    """

    #should really record all of the drafted teams too 

    victory_res = [[] for i in range(n_drafters)]
    detailed_res = pd.DataFrame()
    team_dict = {}
    all_times = {}
     
    for i in range(n_drafters): 

        #we need to deepcopy the agents so that they don't share references with each other
        agents =  [copy.deepcopy(_primary_agent) for x in range(n_primary)] + \
                    [copy.deepcopy(_default_agent) for x in range(n_drafters-n_primary)]
        
        primary = [True] * n_primary + [False] * (n_drafters-n_primary)

        agents = rotate(agents, i)
        primary = rotate(primary, i)
        
        teams, team_details, times = run_draft(agents,n_picks)
        res, details = run_multiple_seasons(teams = teams
                                   , season_df = season_df
                                   , n_seasons = n_seasons
                                   , scoring_format = scoring_format
                                   , return_detailed_results = True)
        detailed_res = pd.concat([detailed_res, details.loc[i,:]])
        
        victory_res[i] = np.mean([(res.get(n)) if (res.get(n)) is not None else 0 for n in range(n_drafters) if primary[n]])
        team_dict[i] = teams[i]
        all_times[i] = times[i]
        
    return victory_res, detailed_res, team_dict, all_times

def make_detailed_results_tab(res, overall_win_rate, info, n_drafters, roto = False):

    c1, c2 = st.columns([0.5,0.5])

    with c1: 

        if roto:
            detailed_rates_collapsed = res[1]
            detailed_rates_collapsed.columns = get_categories()

        else: 
            detailed_rates_collapsed = res[1].loc['Win'].reset_index(drop = True) + \
                                        res[1].loc['Tie'].reset_index(drop = True)/2
        overall_win_rate.columns = ['Overall Win %']

        detailed_rates_collapsed.index = ['Seat ' + str(x) for x in range(n_drafters)]

        detailed_rates_collapsed = overall_win_rate.merge(detailed_rates_collapsed
                                                                            , left_index = True
                                                                            , right_index = True)

        if roto:
            detailed_rate_df = detailed_rates_collapsed.style.format("{:.1%}", subset = ['Overall Win %']) \
                                        .format("{:.3}"
                                                , subset = get_categories()) \
                                        .map(stat_styler
                                            , middle = 6.5
                                            , multiplier = 40
                                            , subset = get_categories()) \
                                        .map(styler_a
                                            , subset = ['Overall Win %'])             
        else: 
            detailed_rate_df = detailed_rates_collapsed.style.format("{:.1%}") \
                                        .map(stat_styler
                                            , middle = 0.5
                                            , multiplier = 150
                                            , subset = get_categories()) \
                                        .map(styler_a
                                            , subset = ['Overall Win %']) 

        st.dataframe(detailed_rate_df
                , use_container_width = True
                , height = len(detailed_rates_collapsed) * 35 + 38)
    with c2:
        seats = ['Seat ' + str(x) for x in range(n_drafters)]
        team_tabs = st.tabs(seats)

        for team_tab, seat in zip(team_tabs, range(n_drafters)):
            with team_tab:
                make_team_tab(info['G-scores'] 
                            , res[2][seat]
                            , n_drafters 
                            , st.session_state.params['g-score-player-multiplier']
                            , st.session_state.params['g-score-team-multiplier']
                            , None
                            ) 


def validate() -> None:

    season_df = pd.read_csv('data/2022-23_complete.csv')

    weekly_df = make_weekly_df(season_df)

    input_tab, analysis_tab, results_tab, timing_tab = st.tabs(['Inputs'
                                                ,'Analysis'
                                                ,'Results'
                                                ,'Timing'])

    with input_tab:
        weekly_tab, averages_tab = st.tabs(['Weekly Data','Averages'])

        with weekly_tab:
            st.dataframe(weekly_df)

        with averages_tab:

            player_averages = weekly_df.groupby('Player')[cols].mean()

            player_averages.loc[:,'Free Throw %'] = player_averages['Free Throws']/player_averages['Free Throw Attempts']
            player_averages.loc[:,'Field Goal %'] = player_averages['Field Goals']/player_averages['Field Goal Attempts']

            metadata = get_player_metadata()

            player_averages = player_averages.merge(metadata, left_index = True, right_index = True)
            conversion_factors = pd.read_csv('./coefficients.csv', index_col = 0)

            st.markdown('Weekly player averages ')
            st.dataframe(player_averages)

    multipliers = pd.DataFrame({'Multiplier' : [1,1,1,1,1,1,1,1,1]}
                            , index = conversion_factors.index)

    #these should all be part of the params file
    psi = st.session_state.params['options']['psi']['default']
    nu = st.session_state.params['options']['nu']['default']
    n_drafters = st.session_state.params['options']['n_drafters']['default']
    n_picks = st.session_state.params['options']['n_picks']['default']
    omega = st.session_state.params['options']['omega']['default']
    gamma = st.session_state.params['options']['gamma']['default']
    alpha = st.session_state.params['options']['alpha']['default']
    beta = st.session_state.params['options']['beta']['default']
    chi = 0.05
    n_seasons = 1000

    info = process_player_data(player_averages
                        , conversion_factors
                        , multipliers 
                        , psi
                        , nu 
                        , n_drafters
                        , n_picks
                        , None
                        )

    g_scores = info['G-scores']
    g_scores = g_scores.sort_values('Total', ascending = False)
    default_agent_hth = SimpleAgent(order = list(g_scores.index))

    z_scores = info['Z-scores']
    z_scores = z_scores.sort_values('Total', ascending = False)
    default_agent_roto = SimpleAgent(order = list(z_scores.index))

    with analysis_tab: 
        omega, gamma = get_estimate_of_omega_gamma(info)

    primary_agent_roto = HAgent(
            info = info
            , omega = omega
            , gamma = gamma
            , alpha = alpha
            , beta = beta
            , n_picks = n_picks
            , n_drafters = n_drafters
            , scoring_format = 'Rotisserie'
            , punting = True
            , chi = chi
            )
                
    res_roto =  try_strategy(primary_agent_roto
            , default_agent_roto
            , n_drafters
            , n_picks
            , weekly_df
            , n_seasons + 2
            , n_primary = 1
            , scoring_format = 'Rotisserie')

    primary_agent_ec = HAgent(
                info = info
                , omega = omega
                , gamma = gamma
                , alpha = alpha
                , beta = beta
                , n_picks = n_picks
                , n_drafters = n_drafters
                , scoring_format = 'Head to Head: Each Category'
                , punting = True
                , chi = None
                )

    res_ec =  try_strategy(primary_agent_ec
            , default_agent_hth
            , n_drafters
            , n_picks
            , weekly_df
            , n_seasons
            , n_primary = 1
            , scoring_format = 'Head to Head: Each Category')

    primary_agent_wta = HAgent(
                info = info
                , omega = omega
                , gamma = gamma
                , alpha = alpha
                , beta = beta
                , n_picks = n_picks
                , n_drafters = n_drafters
                , scoring_format = 'Head to Head: Most Categories'
                , punting = True
                , chi = None
                )

    res_wta =  try_strategy(primary_agent_wta
            , default_agent_hth
            , n_drafters
            , n_picks
            , weekly_df
            , n_seasons
            , n_primary = 1
            , scoring_format = 'Head to Head: Most Categories')



    with results_tab:

        t1, t2, t3, t4 = st.tabs(['Overall'
                            ,'Head to Head: Most Categories Details'
                            ,'Head to Head: Each Category Details'
                            ,'Rotisserie'])

        with t1: 
            win_rate_df = pd.DataFrame({'Head to Head: Most Categories' : res_wta[0]
                                        ,'Head to Head: Each Category' : res_ec[0] 
                                        ,'Rotisserie' : res_roto[0]}
                                        , index = ['Seat ' + str(x) for x in range(n_drafters)])

            averages_df = pd.DataFrame({'Head to Head: Most Categories' : \
                                                    [win_rate_df['Head to Head: Most Categories'].mean()]
                                                ,'Head to Head: Each Category' : \
                                                    [win_rate_df['Head to Head: Each Category'].mean()] 
                                                ,'Rotisserie' : \
                                                    [win_rate_df['Rotisserie'].mean()]}
                                                        , index = ['Aggregate'])
            win_rate_df = pd.concat([averages_df, win_rate_df])

            win_rate_df_styled = win_rate_df.style.format("{:.1%}") \
                                            .map(stat_styler
                                            ,middle = 0.08333, multiplier = 300) \
                                            .map(styler_a, subset = pd.IndexSlice['Aggregate',:])

            st.subheader('Ultimate win rates')
            st.dataframe(win_rate_df_styled
                    , height = len(win_rate_df) * 35 + 38)

        with t2: 
            overall_win_rate = win_rate_df[['Head to Head: Most Categories']]

            make_detailed_results_tab(res_wta, overall_win_rate, info, n_drafters)
            
        with t3: 
            overall_win_rate = win_rate_df[['Head to Head: Each Category']]

            make_detailed_results_tab(res_ec, overall_win_rate, info, n_drafters)

        with t4:
            overall_win_rate = win_rate_df[['Rotisserie']]
            make_detailed_results_tab(res_roto, overall_win_rate, info, n_drafters, True)

    with timing_tab:
        
        wta_tab, ec_tab = st.tabs(['Head to Head: Most Categories','Head to Head: Each Category']) 

        with wta_tab:
            time_df = pd.DataFrame(res_wta[3])
            for col in time_df.columns:
                time_df[col] = time_df[col].dt.total_seconds()

            timing_df_wta = time_df.style.format("{:.2f} s").map(stat_styler
                                                , middle = 1
                                                , multiplier = -50
            )
            st.dataframe(timing_df_wta)

        with ec_tab:
            time_df = pd.DataFrame(res_ec[3])
            for col in time_df.columns:
                time_df[col] = time_df[col].dt.total_seconds()

            timing_df_ec = time_df.style.format("{:.2f} s").map(stat_styler
                                                , middle = 1
                                                , multiplier = -50
            )
            st.dataframe(timing_df_ec)

#add function here to sim
#check for actual values of beta, gamma 
def get_preds_c(v,c,L):    
    factor = v.dot(v.T).dot(L).dot(c)/v.T.dot(L).dot(v)
    sigma = np.sqrt(((c-factor).T.dot(L).dot((c-factor))))    
    #also check out actual correlation, in addition to the theoretical correlation?
        
    return sigma[0][0], sigma[0][0]
    
def get_actuals_c(c, v, x_scores):

        unweighted = x_scores.dot(v)
        weighted = x_scores.dot(c)
                
        selected = weighted == weighted.max()
                
        m = weighted.max() - unweighted.max()
        
        k = unweighted.max() - unweighted[selected]

        return m,k
    
def estimate_values(c, v, x_scores, g_scores):
    
    x_scores = x_scores.loc[g_scores.sum(axis = 1).sort_values(ascending = False).index]
    observations = [get_actuals_c(c, v, x_scores.iloc[i:]) for i in range(0,160,10)]

    m_avg = np.mean([x[0] for x in observations])
    k_avg = np.mean([x[1].values for x in observations])
    return m_avg, k_avg

def get_estimate_of_omega_gamma(info):

    x_scores = info['X-scores']
    g_scores = info['G-scores']
    mov = info['Mov']
    vom = info['Vom']
    v = np.sqrt(mov/(mov + vom))

    ##maybe we should modify the X-scores in the same way as L got modified? Or just retain X-scores-as-diff

    v = np.array(v/v.sum()).reshape(9,1)

    cs = - np.random.random(size = (100,9)) * (np.random.random(size = (100,9)) < 0.2)/20 + v.reshape(1,9)
    cs = cs/cs.sum(axis = 1).reshape(-1,1)

    L = info['L']

    res = [(get_preds_c(v, c.reshape(9,1),L), estimate_values(c, v, x_scores, g_scores)) for c in cs]
    
    x_m = [v[0][0] for v in res]    
    x_k = [v[0][1] for v in res]     

    y_m = [v[1][0] for v in res]   
    y_k = [v[1][1] for v in res]  

    s = pd.DataFrame(cs/v.T, columns = get_categories())

    s.loc[:,'sigma'] = x_k
    s.loc[:,'Actual_k'] = y_k
    s.loc[:,'Ratio_k'] = s['Actual_k']/s['sigma']

    s.loc[:,'Actual_m'] = y_m
    s.loc[:,'Ratio_m'] = s['Actual_m']/s['sigma']

    st.dataframe(s)

    k_res = sm.OLS(y_k, x_k).fit()
    m_res = sm.OLS(y_m, x_m).fit()
    gamma = np.round(k_res.params[0],3)
    st.subheader('gamma: ' + str(gamma))
    st.write(k_res.summary())

    fig1 = px.scatter(x = x_k, y = y_k)
    st.plotly_chart(fig1)

    omega = np.round(m_res.params[0],3)
    st.subheader('omega: ' + str(omega))
    st.write(m_res.summary())

    fig2 = px.scatter(x = x_m, y = y_m)
    st.plotly_chart(fig2)
    
    return omega, gamma