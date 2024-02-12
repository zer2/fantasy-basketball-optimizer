import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import itertools
from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
  
def stat_styler(value
                , multiplier = 50
                , middle = 0 ):
  #styler function used for coloring stat values red/green with varying intensities 
                    
  intensity = min(int(abs(value-middle)*multiplier), 255)
  if value != value:
    return f"background-color:white;color:white;" 
  elif value - middle > 0:
    rgb = (255 -  intensity,255 , 255 -  intensity)
  else:
    rgb = (255, 255 - intensity, 255 - intensity)
      
  bgc = '#%02x%02x%02x' % rgb

  #formula adapted from
  #https://stackoverflow.com/questions/3942878/how-to-decide-font-color-in-white-or-black-depending-on-background-color
  darkness_value = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
  tc = 'black' if darkness_value > 150 else 'white'

  return f"background-color: " + str(bgc) + ";color:" + tc + ";" 

def styler_a(value):
    return f"background-color: grey; color:white;" 

def styler_b(value):
    return f"background-color: lightgrey; color:black;" 

def styler_c(value):
    return f"background-color: darkgrey; color:black;" 
    
def listify(x):
    #get all values from a dataframe into a list. Useful for listing all chosen players 

    x = x.values.tolist()
    return [item for row in x for item in row]

def combinatorial_calculation(c
                              , c_comp
                              , categories
                              , data = 1 #the latest probabilities. Defaults to 1 at start
                              , level = 0 #the number of categories that have been worked into the probability
                              , n_false = 0 #the number of category losses that have been trackes so far
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
        c: Dataframe of all category winning probabilities. One column per category, one row per player
        c_comp: 1 - c
        categories: list of the relevant categories
        data: probability of the node's scenario. Defaults to 1 because no categories are required at first
        level: the number of categories that have been worked into the probability
        n_false: the number of category losses that have been tracked so far. When it gets high enough 
                 we write off the node; the remaining scenarios do not contribute to winning chances

    Returns:
        Probability of winning a majority of categories for each player 

    """
    if n_false > (len(categories) -1)/2: #scenarios where a majority of categories are losses are overall losses
        return 0 
    elif level < len(categories):
        #find the total winning probability of both branches from this point- if we win or lose the current category 
        return combinatorial_calculation(c, c_comp, categories, data * c[categories[level]], level + 1, n_false) + \
                combinatorial_calculation(c, c_comp, categories, data * c_comp[categories[level]], level + 1, n_false + 1)
    else: #a series where all 9 categories has been processed, and n_false <= the cutoff, can be added to the total %
        return data

def calculate_tipping_points(x):
    """Calculate the probability of each category being a tipping point, assuming independence

    Args:
        x: DataFrame of shape (n,9) representing probabilities of winning each of the 9 categories 

    Returns:
        PDataFrame of shape (n,9) representing probabilities of each category being a tipping point

    """

    #create a grid representing 126 scenarios where 5 categories are won and 4 are lost
    
    which = np.array([list(itertools.combinations(range(9), 5))] )
    grid = np.zeros((126, 9), dtype="bool")     
    grid[np.arange(126)[None].T, which] = True

    #copy grid for each row in x 
    grid = np.array([grid] * x.shape[0])

    x = x.reshape(x.shape[0],1,9)

    #get the probabilities of the scenarios and filter them by which categories they apply to
    #the categories that are won all become tipping points
    positive_case_probabilities = ((grid * x + (1-grid) * (1-x)) \
                                   .prod(axis = 2).reshape(x.shape[0],126,1) * grid).sum(axis = 1)

    #do the same but for the inverse scenarios, where 5 categories are lost and 4 are won
    #in this case the lost categories become tipping points 
    negative_case_probabilities = (((1 - grid) * x + grid * (1-x)) \
                                   .prod(axis = 2).reshape(x.shape[0],126,1) * grid).sum(axis = 1)
    final_probabilities = positive_case_probabilities + negative_case_probabilities
    
    return final_probabilities

def make_progress_chart(res):
    """Chart the progress of gradient descent in action, for the top 10 players 

    Args:
        res: List of result objects 

    Returns:
        Line chart showing scores per player by iteration

    """
    #
    #input res is the the log from H-scoring after some number of iterations
    
    data = pd.concat([pd.DataFrame({'H-score' : [r.loc[player] for r in res]
                                , 'Player' : player
                               , 'Iteration' : list(range(len(res)))})
        for player in res[-1].sort_values(ascending = False).index[0:10]])
    
    fig = px.line(data
                  , x = "Iteration"
                  , y = 'H-score'
                  , color = "Player")
    
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
                ))

    fig.update_layout(yaxis={'visible': False, 'showticklabels': True})

    fig.update_layout(margin=go.layout.Margin(
                                    l=0, #left margin
                                    r=0, #right margin
                                    b=0, #bottom margin
                                    t=0, #top margin
                                            )
                     )

    return fig
