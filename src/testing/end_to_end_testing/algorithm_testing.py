import numpy as np
import pandas as pd
from src.run_algorithm import HAgent, SimpleAgent
from src.helper_functions import get_categories
from src.process_player_data import process_player_data
import streamlit as st
import yaml
#pip install scipy, numpy, pandas, plotly, streamlit

if 'params' not in st.session_state:
  with open("parameters.yaml", "r") as stream:
    f = yaml.safe_load(stream)
  st.session_state = {'params' : f }

print(st.session_state['params'])

def round_robin_opponent(t
                         , w
                         , n =12): 
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
              , n_rounds):
    """Run a snake draft

    Snake drafts wrap around like 1 -> 2 -> 3 -> 3 -> 2 -> 1 -> 1 -> 2 -> 3 etc. 
    
    Args:
        agents: list of Agents, which are required to have make_pick() methods
        n_rounds: number of rounds to do of the snake draft. Each drafter will get n_rounds * 2 players
        
    Returns:
        dictionary of player assignments with the structure
         {'player name' : team_number } 
    """
    
    player_assignments = {}
    
    for i in range(n_rounds):
        
        if i % 2 ==0:
            for j in range(len(agents)):

                agent = agents[j]

                chosen_player = agent.make_pick(player_assignments, j)
                player_assignments[j] = player_assignments[j] + chosen_player
                all_players_chosen = all_players_chosen + chosen_player 
        
        else:
           
            for j in reversed(range(len(agents))):
                agent = agents[j]

                chosen_player = agent.make_pick(player_assignments, j)
                player_assignments[j] = player_assignments[j] + chosen_player
                all_players_chosen = all_players_chosen + chosen_player 
        j

    return player_assignments, agents

def run_multiple_seasons(teams
                         , season_df
                         , n_seasons = 100 
                         , n_weeks = 25
                         , winner_take_all = True
                         , return_detailed_results = False ):
    """Simulate multiple seasons with the same drafters 
    
    Weekly performances are sampled from a dataframe of real season performance
    Teams win weeks by winning more categories than their opponents. They win seasons by winning the
    most weeks of all players 
    
    Args:
        teams: player assignment dict, as produced by the run_draft() functoin
        season_df: dataframe of weekly numbers per players. These will be sampled to simulate seasons
        n_seasons: number of seasons to simulate
        n_weeks: number of weeks per season
        winner_take_all: If True, the winner of a majority of categories in a week gets a point.
                         If false, each player gets a point for each category won 
        return_cat_results: If True, return detailed results on category wins 
        
    Returns:
        Series of winning percentages with the structure
         team_number : winning_fraction  
    """
    #create a version of the essential info dataframe which incorporate team information for this season
    season_df = season_df.reset_index().drop(columns = 'week')
    season_df.loc[:,'Team'] = season_df['Player'].map(teams) 
    season_df = season_df.dropna(subset = ['Team'])

    #use sampling to simulate many seasons at the same time
    #assuming each season has 11 weeks, we need 11 * n total rows of data per player
    #ZR: Note that for now a "week" of data is just one game per player
    #in real basketball multiple games are played per week, so we need to adjust for that 
    performances = season_df.groupby('Player').sample(n_weeks*n_seasons, replace = True)
    performances.loc[:,'Week'] = performances.groupby('Player').cumcount()
    performances.loc[:,'Season'] = performances['Week'] // n_weeks #integer division seperates weeks in groups 

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
    
    if winner_take_all:
        team_performances.loc[:,'Tie'] = tot_cat_wins + tot_cat_ties/2 == len(get_categories())/2
        team_performances.loc[:,'Win'] = tot_cat_wins + tot_cat_ties/2 > len(get_categories())/2
    else:
        team_performances.loc[:,'Tie'] = tot_cat_ties
        team_performances.loc[:,'Win'] = tot_cat_wins
        
    team_results = team_performances.groupby(['Team','Season']).agg({'Win' : sum, 'Tie' : sum})

    #a team cannot win the season if it has fewer wins than any other team 
    most_wins = team_results.groupby('Season')['Win'].transform('max')
    winners = team_results[team_results['Win'] == most_wins]

    #among the teams with the most wins, ties are a tiebreaker 
    most_ties = winners.groupby('Season')['Tie'].transform('max')
    winners_after_ties = winners[winners['Tie'] == most_ties]
    
    #assuming that payouts are divided when multiple teams are exactly tied, we give fractional points 
    winners_after_ties.loc[:,'Winner Points'] = 1
    season_counts = winners_after_ties.groupby('Season')['Winner Points'].transform('count')
    winners_after_ties.loc[:,'Winner Points Adjusted'] = 1/season_counts
    
    wins_by_teams = winners_after_ties.groupby('team')['Winner Points Adjusted'].sum()/winners_after_ties['Winner Points Adjusted'].sum()
    
    if not return_detailed_results:
        return wins_by_teams
    else:
        cat_win_df = pd.DataFrame(cat_wins, columns = categories, index = team_performances.index)
        cat_tie_df = pd.DataFrame(cat_ties, columns = categories, index = team_performances.index)
        
        cat_wins_agg = cat_win_df.groupby('team').mean()
        cat_wins_agg = pd.concat({'Win' : cat_wins_agg}, names = ['Result'])
        
        cat_ties_agg = cat_tie_df.groupby('team').mean()
        cat_ties_agg = pd.concat({'Tie' : cat_ties_agg}, names = ['Result'])

        results_agg = pd.concat([cat_wins_agg, cat_ties_agg])
        return wins_by_teams, results_agg.reorder_levels(['Team','Result'])
                              
    
def rotate(l, n):
    return l[-n:] + l[:-n]

def try_strategy(primary_agent
                 , default_agent
                 , season_df
                 , n_seasons
                 , n_primary
                 , categories
                 , winner_take_all):
    
    #should really record all of the drafted teams too 

    victory_res = [[] for i in range(12)]
    detailed_res = pd.DataFrame()
    primary_team_details = {}
    
    for i in range(12):
        #we need to deepcopy the agents so that they don't share references with each other
        agents =  [copy.deepcopy(primary_agent) for x in range(n_primary)] + \
                    [copy.deepcopy(default_agent) for x in range(12-n_primary)]
        
        primary = [True] * n_primary + [False] * (12-n_primary)

        agents = rotate(agents, i)
        primary = rotate(primary, i)
        
        teams, team_details = run_draft(agents,13)
        res, details = run_multiple_seasons(teams = teams
                                   , season_df = season_df
                                   , n_seasons = n_seasons
                                   , winner_take_all = winner_take_all
                                   , return_detailed_results = True)
        detailed_res = pd.concat([detailed_res, details.loc[i,:]])
        
        
        victory_res[i] = np.mean([(res.get(n)) if (res.get(n)) is not None else 0 for n in range(12) if primary[n]])
        primary_team_details[i] = team_details[i]
        
    return victory_res, detailed_res, primary_team_details


def get_results_of_strategy(primary_agent
                 , default_agent
                 , season_df : pd.DataFrame
                 , n_seasons
                 , n_primary = 1
                 , winner_take_all = True):
    
    victory_res, detailed_res, primary_team_details = try_strategy(primary_agent
                                             , default_agent
                                             , season_df
                                             , n_seasons
                                             , n_primary
                                             , winner_take_all)
    return victory_res, detailed_res, primary_team_details


def validate() -> None:

    season_df = pd.read_csv('data/2022-23_complete.csv')
    cols = ['Free Throws','Free Throw Attempts','Field Goals','Field Goal Attempts'
            ,'Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
    player_game_averages = season_df.groupby('Player')[cols].sum()/82 

    conversion_factors = pd.read_csv('./coefficients.csv', index_col = 0)

    multipliers = pd.DataFrame({'Multiplier' : [1,1,1,1,1,1,1,1,1]}
                            , index = conversion_factors.index).T

    psi = 0
    nu = 0.77
    n_drafters = 12
    n_picks = 13
    omega = 1
    gamma = 0.1
    alpha = 0.01
    beta = 0.25
    n_seasons = 1000

    info = process_player_data(player_game_averages
                        , conversion_factors
                        , multipliers 
                        , psi
                        , nu 
                        , n_drafters
                        , n_picks
                        , False
                        )


    g_scores = info['G-scores']
    g_scores = g_scores.sort_values('Total', ascending = False)
    default_agent = SimpleAgent(order = list(g_scores.index))

    primary_agent_wta = HAgent(
                info = info
                , omega = omega
                , gamma = gamma
                , alpha = alpha
                , beta = beta
                , n_picks = n_picks
                , winner_take_all = True
                , punting = False
                )

    primary_agent_ec = HAgent(
                info = info
                , omega = omega
                , gamma = gamma
                , alpha = alpha
                , beta = beta
                , n_picks = n_picks
                , winner_take_all = False
                , punting = False
                )

    res_wta =  get_results_of_strategy(primary_agent_wta
                 , default_agent
                 , season_df
                 , n_seasons
                 , n_primary = 1
                 , winner_take_all = True)

    print(res_wta[0])

    res_ec =  get_results_of_strategy(primary_agent_ec
                 , default_agent
                 , season_df
                 , n_seasons
                 , n_primary = 1
                 , winner_take_all = False)

    print(res_ec[0])

validate()