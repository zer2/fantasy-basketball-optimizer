import pandas as pd
from scipy.stats import norm
import numpy as np

def savor_calculation(raw_values_unselected : pd.Series
                    , n_remaining_players : int
                    , remaining_cash : int
                    , noise = 1) -> pd.Series:
    """Calculate SAVOR- Streaming-adjusted value over replacement

    SAVOR estimates the probability that a player will be replaced by a streamer, and adjusts 
    auction value accordingly

    Args:
      raw_values_unselected: raw value by Z-score, G-score, etc. 
      n_remaining_players: number of players left to be picked
      remaining_cash: amount of cash remaining to spend on players, from all teams
      noise: parameter for the SAVOR function. Controls how noisy we expect player performance to be
             and therefore how likely it is a player will be replaced by a streamer

    Returns:
      Series, SAVOR 
    """

    replacement_value = raw_values_unselected.iloc[n_remaining_players]
    value_above_replacement = np.clip(raw_values_unselected - replacement_value,0,None)

    probability_of_non_streaming = norm.cdf(value_above_replacement/noise)
    adjustment_factor = noise/(2 * np.pi)**(0.5) * (1 - np.exp((-value_above_replacement**2)/(2 * noise)))
    adjusted_value = value_above_replacement * probability_of_non_streaming - adjustment_factor

    remaining_value = adjusted_value.iloc[0:n_remaining_players].sum()
    dollar_per_value = remaining_cash/remaining_value

    savor = adjusted_value * dollar_per_value

    return savor 

def combinatorial_calculation(c : np.array
                              , c_comp : np.array
                              , data = 1 #the latest probabilities. Defaults to 1 at start
                              , level : int = 0 #the number of categories that have been worked into the probability
                              , n_false : int = 0 #the number of category losses that have been tracked so far
                             ):
    """This recursive functions enumerates winning probabilities for the Gaussian optimizer

    The function's recursive structure creates a binary tree, where each split is based on whether the next category is 
    won or lost. At the high level it looks like 
    
                                            (start) 
                                    |                   |
                                won rebounds      lost rebounds
                             |          |           |            |
                          won pts    lost pts   won pts     lost pts
                          
    The probabilities of winning scenarios are then added along the tree. This is more efficient than brute force calculation
    of each possibility, because it doesn't repeat multiplication steps for similar scenarios like (won 9) and (won 8 then 
    lost the last 1). Ultimately it is about five times faster than the equivalent with list comprehension
    
    Args:
        c: Array of all category winning probabilities. Dimensions are (player, category, opponent)
        c_comp: 1 - c
        data: probability of the node's scenario. Defaults to 1 because no categories are required at first
        level: the number of categories that have been worked into the probability
        n_false: the number of category losses that have been tracked so far. When it gets high enough 
                 we write off the node; the remaining scenarios do not contribute to winning chances

    Returns:
        DataFrame with probability of winning a majority of categories for each player 

        axis 0: player 
        axis 1: opponent

    """
    if n_false > (c.shape[1] -1)/2: #scenarios where a majority of categories are losses are overall losses
        return 0 
    elif level < c.shape[1] :
        #find the total winning probability of both branches from this point- if we win or lose the current category 
        return combinatorial_calculation(c, c_comp, data * c[:,level,:], level + 1, n_false) + \
                combinatorial_calculation(c, c_comp, data * c_comp[:,level,:], level + 1, n_false + 1)
    else: #a series where all 9 categories has been processed, and n_false <= the cutoff, can be added to the total %
        return data

@st.cache_data()
def get_grid():
    #create a grid representing 126 scenarios where 5 categories are won and 4 are lost

    which = np.array([list(combinations(range(9), 5))] )
    grid = np.zeros((126, 9), dtype="bool")     
    grid[np.arange(126)[None].T, which] = True

    grid = np.expand_dims(grid, axis = 2)

    return grid

def calculate_tipping_points(x : np.array) -> np.array:
    """Calculate the probability of each category being a tipping point, assuming independence

    Args:
        x: Array of shape (n,9,m) representing probabilities of winning each of the 9 categories 

    Returns:
        DataFrame of shape (n,9,m) representing probabilities of each category being a tipping point
        m is number of opponents
    """

    grid = get_grid()

    #copy grid for each row in x 
    grid = np.array([grid] * x.shape[0])

    x = x.reshape(x.shape[0],1,9, x.shape[2])

    #get the probabilities of the scenarios and filter them by which categories they apply to
    #the categories that are won all become tipping points

    first_part = ne.evaluate('grid * x + (1-grid) * (1-x)') \
                                   .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    positive_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    #do the same but for the inverse scenarios, where 5 categories are lost and 4 are won
    #in this case the lost categories become tipping points 
    first_part = ne.evaluate('(1 - grid) * x + grid * (1-x)') \
                                  .prod(axis = 2).reshape(x.shape[0],126,1,x.shape[3])
    negative_case_probabilities = ne.evaluate('first_part * grid').sum(axis = 1)

    final_probabilities = ne.evaluate('positive_case_probabilities + negative_case_probabilities')

    return final_probabilities