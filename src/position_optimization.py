import numpy as np
from scipy.optimize import linear_sum_assignment
from src.helper_functions import get_position_numbers, get_position_ends
import pandas as pd 

def get_future_player_rows(position_rewards):
    """Takes an array of rewards by simplified position (5 columns) and translates them to rewards per slot (13) by player"""

    position_numbers = get_position_numbers()

    util_rewards = np.array([np.max(position_rewards, axis = 1) + 0.002] * position_numbers['Util'])
    center_rewards = np.array([position_rewards[:,0]] * position_numbers['C'])
    guard_rewards = np.array([np.max(position_rewards[:,1:3],axis = 1) + 0.001] * position_numbers['G'])
    pg_reward = np.array([position_rewards[:,1]] * position_numbers['PG'] ) 
    sg_reward = np.array([position_rewards[:,2]] * position_numbers['SG'] ) 
    forward_rewards = np.array([np.max(position_rewards[:,3:5], axis = 1)+ 0.001]  * position_numbers['F'] )
    pf_reward = np.array([position_rewards[:,3]]  * position_numbers['PF'] ) 
    sf_reward =  np.array([position_rewards[:,4]]  * position_numbers['SF'] ) 

    row = np.concatenate([util_rewards, center_rewards, guard_rewards, pg_reward, sg_reward, forward_rewards, pf_reward, sf_reward]
                        , axis = 0).T

    return row

def get_player_rows(players):
    """Turns a list of player eligibilities into a array of rows that can be input to the matching problem"""
    n_players = len(players)

    position_numbers = get_position_numbers()

    is_center = np.array([['C' in x] for x in players])
    is_pg = np.array([['PG' in x] for x in players])
    is_sg = np.array([['SG' in x] for x in players])
    is_pf = np.array([['PF' in x] for x in players])
    is_sf = np.array([['SF' in x] for x in players])

    util_slots = np.array([[0] * position_numbers['Util']] * n_players)


    center_slots = np.where(is_center
                            , [[0] * position_numbers['C']] * n_players
                            , [[-np.inf] * position_numbers['C']] * n_players)

    guard_slots = np.where(is_pg | is_sg
                            , [[0] * position_numbers['G']] * n_players
                            , [[-np.inf] * position_numbers['G']] * n_players)
    
    pg_slot = np.where(is_pg
                            , [[0] * position_numbers['PG']] * n_players
                            , [[-np.inf] * position_numbers['PG']] * n_players)
    
    sg_slot = np.where(is_sg
                            , [[0] * position_numbers['SG']] * n_players
                            , [[-np.inf] * position_numbers['SG']] * n_players)
    
    forward_slots = np.where(is_pf | is_sf
                            , [[0] * position_numbers['F']] * n_players
                            , [[-np.inf] * position_numbers['F']] * n_players)
    
    pf_slot = np.where(is_pf
                            , [[0] * position_numbers['PF']] * n_players
                            , [[-np.inf] * position_numbers['PF']] * n_players)
    
    sf_slot = np.where(is_sf
                            , [[0] * position_numbers['SF']] * n_players
                            , [[-np.inf] * position_numbers['SF']] * n_players)

    res = np.concatenate([util_slots, center_slots, guard_slots, pg_slot, sg_slot, forward_slots, pf_slot, sf_slot], axis = 1)

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
        reward_vector: Array of length 13 with rewards for future players for each slot  
        team_so_far_array: Rows representing players already on the team for the assignment problem
        n_remaining_

    Returns:
        List of ints, representing which slots the future players will take

    """

    future_player_rows = np.array([reward_vector] * n_remaining_players)
    full_array = np.concatenate([team_so_far_array, [candidate_player_row], future_player_rows], axis = 0)    
    try:
        res = linear_sum_assignment(full_array, maximize = True)
        return res[1] 
    except: 
        return np.array([0] * 13)

def get_position_array_from_res(res :np.array
                                 , utility_shares : pd.DataFrame
                                 , guard_shares : pd.DataFrame
                                 , forward_shares : pd.DataFrame
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

    position_ends = get_position_ends()

    future_positions = res[:,-n_remaining_players:]
    utils = (future_positions <= position_ends['Util']).sum(axis = 1).astype(float)
    centers = ((future_positions > position_ends['Util']) & (future_positions <=position_ends['C'])).sum(axis = 1).astype(float)
    guards = ((future_positions > position_ends['C']) & (future_positions <= position_ends['G'])).sum(axis = 1).astype(float)
    pg = ((future_positions > position_ends['G']) & (future_positions <= position_ends['PG'])).sum(axis = 1).astype(float)
    sg = ((future_positions > position_ends['PG']) & (future_positions <= position_ends['SG'])).sum(axis = 1).astype(float)
    
    forwards = ((future_positions > position_ends['SG']) & (future_positions <= position_ends['F'])).sum(axis = 1).astype(float)
    pf = ((future_positions > position_ends['F']) & (future_positions <= position_ends['PF'])).sum(axis = 1).astype(float)
    sf = ((future_positions > position_ends['PF']) & (future_positions <= position_ends['SF'])).sum(axis = 1).astype(float)

    #add flex spots based on computed shares 
    utils_split = utility_shares.mul(utils.reshape(-1,1))
    guards_split = guard_shares.mul(guards.reshape(-1,1))
    forwards_split = forward_shares.mul(forwards.reshape(-1,1))

    centers += utils_split.loc[:,'C']
    pg += utils_split.loc[:,'PG'] + guards_split.loc[:,'PG']
    sg += utils_split.loc[:,'SG'] + guards_split.loc[:,'SG']
    pf += utils_split.loc[:,'PF'] + forwards_split.loc[:,'PF']
    sf += utils_split.loc[:,'SF'] + forwards_split.loc[:,'SF']

    res_main = np.concatenate([[centers],[pg],[sg],[pf],[sf]], axis = 0).T

    flex_shares = np.concatenate([[utils],[guards],[forwards]], axis = 0).T

    return res_main, flex_shares

def optimize_positions_all_players(candidate_players : list[list[str]]
                                   , position_rewards : np.array
                                   , team_so_far : list[list[str]]
                                   , utility_shares : pd.DataFrame
                                   , guard_shares : pd.DataFrame
                                   , forward_shares : pd.DataFrame
                                   , scale_down : bool = True):
    """Optimizes positions of future draft picks for all candidate players and associated position rewards 

    The function works by setting up an optimization problem for assigning players to team positions
    If the optimization problem is infeasible, the team is not eligible
    
    Args:
        candidate_players: List of candidate players, which are themselves lists of eligible positions. E.g. 
                [['SF','PF'],['C'],['SF']]
        position_rewards: Array with a column for each of the 13 slots, and a row for each candidate player.
                          Each row represents rewards for positions of future picks  
        team_so_far: List of players already chosen for the team
        scale_down: If True, scale result so that each row adds to 1

    Returns:
        Array, one column per position and one row per candidate player. 

    """

    n_remaining_players = 12 - len(team_so_far)
    reward_array = get_future_player_rows(position_rewards)
    team_so_far_array = get_player_rows(team_so_far) if len(team_so_far) > 0 else np.empty((0,13))
    candidate_player_array = get_player_rows(candidate_players)

    all_res = np.concatenate([[optimize_positions_for_prospective_player(player, reward_vector, team_so_far_array, n_remaining_players)
                                for player, reward_vector in zip(candidate_player_array,reward_array)]
                                ]
                                , axis = 0)
    
    final_positions, flex_shares = get_position_array_from_res(all_res
                                                  ,utility_shares
                                                  ,guard_shares
                                                  ,forward_shares
                                                  , n_remaining_players)
    

    
    if scale_down:
        return final_positions/n_remaining_players, flex_shares
    else: 
        return final_positions, flex_shares

