import numpy as np
from scipy.optimize import linear_sum_assignment

def get_future_player_rows(position_rewards):
    """Takes an array of rewards by simplified position (5 columns) and translates them to rewards per slot (13) by player"""

    util_rewards = np.array([np.max(position_rewards, axis = 1)] * 3)
    center_rewards = np.array([position_rewards[:,0]] * 2)
    guard_rewards = np.array([np.max(position_rewards[:,1:3],axis = 1)] * 2)
    pg_reward = np.array([position_rewards[:,1]]) 
    sg_reward = np.array([position_rewards[:,2]]) 
    forward_rewards = np.array([np.max(position_rewards[:,3:5], axis = 1)] * 2)
    pf_reward = np.array([position_rewards[:,3]]) 
    sf_reward =  np.array([position_rewards[:,4]]) 

    row = np.concatenate([util_rewards, center_rewards, guard_rewards, pg_reward, sg_reward, forward_rewards, pf_reward, sf_reward]
                        , axis = 0).T

    return row

def get_player_rows(players):
    """Turns a list of player eligibilities into a array of rows that can be input to the matching problem"""

    two_zeros = [[0,0]] * len(players)
    two_infs =  [[-np.inf, -np.inf] ]* len(players)

    one_zero = [[0]]  * len(players)
    one_inf =  [[-np.inf] ]* len(players)

    n_players = len(players)
    util_slots = np.array([[0,0,0]] * n_players)

    is_center = np.array([['C' in x] for x in players])
    is_pg = np.array([['PG' in x] for x in players])
    is_sg = np.array([['SG' in x] for x in players])
    is_pf = np.array([['PF' in x] for x in players])
    is_sf = np.array([['SF' in x] for x in players])

    center_slots = np.where(is_center
                            , two_zeros
                            , two_infs)

    guard_slots = np.where(is_pg | is_sg
                            , two_zeros
                            , two_infs)
    
    pg_slot = np.where(is_pg
                            , one_zero
                            , one_inf)
    
    sg_slot = np.where(is_sg
                            , one_zero
                            , one_inf)
    
    forward_slots = np.where(is_pf | is_sf
                            , two_zeros
                            , two_infs)
    
    pf_slot = np.where(is_pf
                            , one_zero
                            , one_inf)
    
    sf_slot = np.where(is_sf
                            , one_zero
                            , one_inf)  

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
    full_array = np.concatenate([team_so_far_array, future_player_rows, [candidate_player_row]], axis = 0)    
    try:
        res = linear_sum_assignment(full_array, maximize = True)
        return res[1] 
    except: 
        return np.array([0] * 13)

def get_position_array_from_res(res :np.array
                                 , position_rewards : np.array
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

    future_positions = res[:,-n_remaining_players:]
    utils = (future_positions <= 2).sum(axis = 1)
    centers = ((future_positions > 2) & (future_positions <=4)).sum(axis = 1)
    guards = ((future_positions > 4) & (future_positions <=6)).sum(axis = 1)
    pg = (future_positions ==7).sum(axis =1)
    sg = (future_positions ==8).sum(axis = 1)
    
    forwards = ((future_positions > 8) & (future_positions <=10)).sum(axis = 1)
    pf = (future_positions ==11).sum(axis = 1)
    sf = (future_positions ==12).sum(axis = 1)
    
    max_reward = np.argmax(position_rewards, axis = 1)
    
    centers += (max_reward == 0) * utils
    pg += (max_reward == 1) * utils
    sg += (max_reward == 2) * utils
    pf += (max_reward == 3) * utils
    sf += (max_reward == 4) * utils

    pg_better = (position_rewards[:,1] > position_rewards[:,2])

    pg += pg_better * guards
    sg += ~pg_better * guards
   
    pf_better = (position_rewards[:,3] > position_rewards[:,4])
    
    pf += pf_better * forwards 
    sf += ~pf_better * forwards

    return np.concatenate([[centers],[pg],[sg],[pf],[sf]], axis = 0).T

def optimize_positions_all_players(candidate_players : list[list[str]]
                                   , position_rewards : np.array
                                   , team_so_far : list[list[str]]
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
    final_positions = get_position_array_from_res(all_res, position_rewards, n_remaining_players)

    if scale_down:
        return final_positions/n_remaining_players
    else: 
        return final_positions

