import numpy as np
from scipy.optimize import linear_sum_assignment
from src.helper_functions import get_position_structure, get_position_numbers, get_position_ranges, get_position_indices
import streamlit as st
import pandas as pd 

def get_future_player_rows(position_rewards):
    """Takes an array of rewards by simplified position (5 columns) and translates them to rewards per slot (13) by player"""

    position_numbers = get_position_numbers()

    #The reshaping is necessary to handle the case when a position number is zero
    position_structure = get_position_structure()

    base_list = position_structure['base_list']

    base_rewards = {position_code : np.array([position_rewards[:,i]] * position_numbers[position_code]) \
                                                            .reshape(position_numbers[position_code] , len(position_rewards))
                                    for i, position_code in zip(range(len(base_list)), base_list)
                                                            }
    
    position_indices = get_position_indices(position_structure)

    #add a small bonus to bias towards more flexible positions
    flex_rewards = {position_code : 
                    np.array([np.max(position_rewards[:,position_indices[position_code]],axis = 1) + 0.0001 * \
                                                                            len(position_indices[position_code])] * \
                                                            position_numbers[position_code]) \
                                                        .reshape(position_numbers[position_code] , len(position_rewards))
                        for position_code in position_structure['flex_list']}


    row = np.concatenate([base_rewards[position_code] for position_code in position_structure['base_list']] + \
                          [flex_rewards[position_code] for position_code in position_structure['flex_list']]
                        , axis = 0).T

    return row

def get_player_rows(players):
    """Turns a list of player eligibilities into a array of rows that can be input to the matching problem"""
    n_players = len(players)

    position_structure = get_position_structure()
    position_numbers = get_position_numbers()
    base_list = position_structure['base_list']
    flex_list = position_structure['flex_list']

    is_base_position = pd.DataFrame(
        {position_code : np.array([position_code in x for x in players]) for position_code in base_list}
    )

    base_slots = {}
    for position_code in position_structure['base_list']:
        base_slots[position_code] = np.where(is_base_position[position_code].values.reshape(-1,1)
                            , [[0] * position_numbers[position_code]] * n_players
                            , [[-np.inf] * position_numbers[position_code]] * n_players)

    flex_slots = {position_code : 
                  np.where(is_base_position[position_structure['flex'][position_code]['bases']].any(axis = 1).values.reshape(-1,1)
                            , [[0] * position_numbers[position_code]] * n_players
                            , [[-np.inf] * position_numbers[position_code]] * n_players)
                for position_code in flex_list}
    
    res = np.concatenate([base_slots[position_code] for position_code in base_list]\
                          + [flex_slots[position_code] for position_code in flex_list], axis = 1)

    return res

def optimize_positions_for_prospective_player(candidate_player_row : np.array
                                              , reward_vector : np.array
                                              , team_so_far_array : np.array
                                              , n_remaining_players : int) -> list[int]:
    
    """Optimizes positions of future draft picks for all candidate players and associated position rewards 

    The function works by setting up an optimization problem for assigning players to team positions
    If the optimization problem is infeasible, the team is not eligible
    
    Args:
        candidate_player_row: Row representing the candidate player for the assignment problem
        reward_vector: Array of length N with rewards for future players for each slot, where N is the number of slots  
        team_so_far_array: Rows representing players already on the team for the assignment problem
        n_remaining_

    Returns:
        List of ints, representing which slots the future players will take

    """

    future_player_rows = np.array([reward_vector] * n_remaining_players).reshape(n_remaining_players, reward_vector.shape[0])
    full_array = np.concatenate([team_so_far_array, [candidate_player_row], future_player_rows], axis = 0)    

    try:
        res = linear_sum_assignment(full_array, maximize = True)
        return res[1] 
    except: 
        return np.array([-1] * len(reward_vector))

def get_position_array_from_res(res :np.array
                                 , position_shares : dict[pd.DataFrame]
                                 , n_remaining_players : int):
    """Takes the result of the assignment problem from integers to the associated positions
    
    Args:
        res: Array with one row representing the solution to the assignment problem for each player 
        position_rewards: Array of length 13 with rewards for future players for each slot  
        team_so_far_array: Rows representing players already on the team for the assignment problem
        n_remaining_

    Returns:
        List of ints, representing which slots the future players will take

    """

    position_ranges = get_position_ranges()
    position_structure = get_position_structure()

    future_positions = res[:,-n_remaining_players:]

    position_sums = {}

    for position_code, position_range in position_ranges.items():

        position_sums[position_code] = ((future_positions >= position_range['start']) & \
               (future_positions < position_range['end'])).sum(axis = 1).astype(float)

    for position_code in position_structure['flex_list']:
        flex_split = position_shares[position_code].mul(position_sums[position_code].reshape(-1,1))

        #add the split-up flex positions into base positions 
        for base_position_code in position_structure['flex'][position_code]['bases']:
            position_sums[base_position_code] += flex_split.loc[:,base_position_code]


    res_main = np.concatenate([[position_sums[position_code]] for position_code in position_structure['base_list']]
                              , axis = 0).T

    flex_shares = {position_code: position_sums[position_code] for position_code in position_structure['flex_list']}

    return res_main, flex_shares

def optimize_positions_all_players(candidate_players : list[list[str]]
                                   , position_rewards : np.array
                                   , team_so_far : list[list[str]]
                                   , position_shares : dict[pd.DataFrame]
                                   , scale_down : bool = True):
    """Optimizes positions of future draft picks for all candidate players and associated position rewards 

    The function works by setting up an optimization problem for assigning players to team positions
    If the optimization problem is infeasible, the team is not eligible
    
    Args:
        candidate_players: List of candidate players, which are themselves lists of eligible positions. E.g. 
                [['SF','PF'],['C'],['SF']]
        position_rewards: Array with a column for each main slots, and a row for each candidate player.
                          Each row represents rewards for positions of future picks  
        team_so_far: List of players already chosen for the team
        scale_down: If True, scale result so that each row adds to 1

    Returns:
        Array, one column per position and one row per candidate player. 

    """

    position_numbers = get_position_numbers()
    n_total_picks = sum([v for k, v in position_numbers.items()])
    n_remaining_players = n_total_picks -1 - len(team_so_far)
    reward_array = get_future_player_rows(position_rewards)
    team_so_far_array = get_player_rows(team_so_far) if len(team_so_far) > 0 else np.empty((0,n_total_picks))
    candidate_player_array = get_player_rows(candidate_players)

    rosters = np.concatenate([[optimize_positions_for_prospective_player(player, reward_vector, team_so_far_array, n_remaining_players)
                                for player, reward_vector in zip(candidate_player_array,reward_array)]
                                ]
                                , axis = 0)
    
    final_positions, flex_shares = get_position_array_from_res(rosters
                                                  ,position_shares
                                                  , n_remaining_players)
    

    
    if scale_down:
        final_positions = final_positions/n_remaining_players
    
    return rosters, final_positions, flex_shares

def check_eligibility_alternate(player, team_so_far):
   
    position_numbers = get_position_numbers()
    n_total_picks = sum([v for k, v in position_numbers.items()])
    n_base_positions = len(get_position_structure()['base_list'])

    position_rewards = np.array([[0] * n_base_positions])
    n_remaining_players = n_total_picks -1 - len(team_so_far)
    reward_vector = get_future_player_rows(position_rewards)[0]
    team_so_far_array = get_player_rows(team_so_far) if len(team_so_far) > 0 else np.empty((0,n_total_picks))

    candidate_player_vector = get_player_rows([player])[0]

    all_res = optimize_positions_for_prospective_player(candidate_player_vector
                                                        , reward_vector
                                                        , team_so_far_array
                                                        , n_remaining_players)
    
    if all(all_res >= 0):
        return True
    else: 
       return False