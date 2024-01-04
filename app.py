import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from pandas.api.types import CategoricalDtype
from process_player_data import process_player_data
from run_algorithm import HAgent
from helper_functions import listify, make_progress_chart

st.title('Optimization for fantasy basketball: based on [this paper](https://arxiv.org/abs/2307.02188)') 

color_map = {'C' : 'yellow'
             ,'PF' : 'green'
             ,'SF' : 'green'
             ,'SG' : 'red'
             ,'PG' : 'red'}

def stat_styler(value):
  if value != value:
    return f"background-color:white;color:white;" 
  elif value > 0:
    bgc = '#%02x%02x%02x' % (255 -  int(value*50),255 , 255 -  int(value*50))
  else:
    bgc = '#%02x%02x%02x' % (255, 255 + int(value*50), 255 + int(value*50))

  tc = 'black' if abs(value) > 1 else 'black'
  
  return f"background-color: " + str(bgc) + ";color:" + tc + ";" 
  
  if value > 1.5:
    return f"background-color: darkgreen;color:white;" 
  elif value > 0.5:
    return f"background-color: green;color:white;" 
  elif value > -0.5: 
    return f"background-color: yellow;color:black;" 
  elif value > -1.5:
    return f"background-color: red;color:white;" 
  else:
    return f"background-color: darkred;color:white;" 

def other_styler(value):
    return f"background-color: grey; color:white;" 

counting_statistics = ['Points','Rebounds','Assists','Steals','Blocks','Threes','Turnovers']
percentage_statistics = ['Free Throw %','Field Goal %']
volume_statistics = ['Free Throw Attempts','Field Goal Attempts']

@st.cache_data
def get_full_data():
  full_df = pd.read_csv('./stat_df.csv').set_index(['Season','Player']).sort_index()  
  full_df[counting_statistics + volume_statistics ] = full_df[counting_statistics + volume_statistics]/3
  
   #adjust for the display
  full_df[r'Free Throw %'] = full_df[r'Free Throw %'] * 100
  full_df[r'Field Goal %'] = full_df[r'Field Goal %'] * 100
  full_df[r'No Play %'] = full_df[r'No Play %'] * 100
  return full_df

@st.cache_data
def get_partial_data(full_df, dataset_name):
  return full_df, full_df.loc[dataset_name]

full_df = get_full_data()

coefficient_df = pd.read_csv('./coefficients.csv', index_col = 0)

tab1, tab2, tab3 = st.tabs(["Parameters", "Player Stats", "Draft"])

with tab1: 
  left, middle, right = st.columns([0.25,0.25,0.5])

  with left: 
    st.header('General')
    
    format = st.selectbox(
      'Which format are you playing?',
      ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories')
      , index = 2)
  
    if format == 'Rotisserie':
      st.caption('Note that it is recommended to use Z-scores rather than G-scores to evaluate players for Rotisserie. Also, Rotisserie H-scores are experimental and have not been tested')
    else:
      st.caption('Note that it is recommended to use G-scores rather than Z-scores to evaluate players for Head to Head')

    unique_datasets = pd.unique(full_df.index.get_level_values('Season'))
                                
    dataset_name = st.selectbox(
      'Which dataset do you want to default to?'
      ,unique_datasets
      ,index = len(unique_datasets)-1
    )

    df = get_partial_data(full_df,dataset_name)

    n_drafters = st.number_input(r'How many drafters are in your league?'
                    , min_value = 2
                    , value = 12)

    n_picks = st.number_input(r'How many players will each drafter choose?'
                , min_value = 1
                , value = 13)
  
  with middle: 
      st.header('Player Statistics')

      psi = st.number_input(r'Select a $\psi$ value'
                        , min_value = 0.0
                        , value = 0.85
                       , max_value = 1.0)
      psi_str = r'''$\psi$ controls how injury rates are dealt with. If $\psi$ is $X\%$, and a player is expected to miss $Y\%$ of weeks
                    entirely, their counting statistics will be multiplied by $(1-Y*X)$. So for example is if $\psi$ is $50\%$ and a 
                    player is expected to miss $20\%$ of weeks, their counting statistics will be multiplied by $(1-0.5*0.2) =  90\%$'''
    
      st.caption(psi_str)

      st.subheader(f"Coefficients")
      coefficients = st.data_editor(coefficient_df
                                   , column_config = {'Mean of Means' :  'μ'
                                                      ,'Variance of Means' : 'σ²'
                                                      ,'Mean of Variances' : '𝜏²'}
                                                      )

      st.caption('μ, σ² and 𝜏² are defined in the paper. If you believe e.g. steals will be relatively unpredictable next year, you can increase 𝜏² for it. But the default values should be reasonable')


  
  with right:
    st.header('H-scoring Algorithm')

    left_col, right_col = st.columns(2)

    with left_col:
      omega = st.number_input(r'Select a $\omega$ value', value = 1.5)
      omega_str = r'''The higher $\omega$ is, the more aggressively the algorithm will try to punt. Slightly more technically, 
                      it quantifies how much better the optimal player choice will be compared to the player that would be 
                      chosen with baseline weights'''
      st.caption(omega_str)
    
      gamma = st.number_input(r'Select a $\gamma$ value', value = 0.1)
      gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
              tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
               players that  work best for its strategy'''
      st.caption(gamma_str)
  
      nu = st.number_input(r'Select a $\nu$ value', value = 0.77, min_value = 0.0, max_value = 1.0)
      nu_str = r'''Covariance matrix is calculated with position averages multiplied by $\nu$ subtracted out. $\nu=0$ is appropriate 
                  if there are no position requirements, $\nu=1$ is appropriate if position requirements are fully strict '''
      st.caption(nu_str)

    with right_col:
      alpha = st.number_input(r'Select a $\alpha$ value', value = 0.01, min_value = 0.0)
      alpha_str = r'''$\alpha$ is the initial step size for gradient descent. Tuning $\alpha$ is not recommended'''
      st.caption(alpha_str)
  
      beta = st.number_input(r'Select a $\beta$ value', value = 0.25, min_value = 0.0)
      beta_str = r'''$\beta$ is the degree of step size decay. Tuning $\beta$ is not recommended'''
      st.caption(beta_str)
  
      n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run', value = 30, min_value = 10, max_value = 10000)
      n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
      st.caption(n_iterations_str)

with tab2:
  st.markdown(f"Per-game player projections below: feel free to edit. Converted to weekly by multiplying by three")

  player_stats_editable = st.data_editor(df) # 👈 An editable dataframe
  player_stats = player_stats_editable.copy()

  #re-adjust from user inputs
  player_stats[r'Free Throw %'] = player_stats[r'Free Throw %']/100
  player_stats[r'Field Goal %'] = player_stats[r'Field Goal %']/100
  player_stats[r'No Play %'] = player_stats[r'No Play %']/100
  player_stats[counting_statistics + volume_statistics] = player_stats[counting_statistics + volume_statistics] * 3
  
with tab3:

  rotisserie = format == 'Rotisserie'
  
  info = process_player_data(player_stats, coefficients, psi, nu, n_drafters, n_picks, rotisserie)

  #perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
  selections = pd.DataFrame({'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)})

  #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
  player_category_type = CategoricalDtype(categories=list(player_stats.index), ordered=True)
  selections = selections.astype(player_category_type)

  z_scores = info['Z-scores']
  categories = z_scores.columns

  z_scores.loc[:,'Total'] = z_scores.sum(axis = 1)
  z_scores.sort_values('Total', ascending = False, inplace = True)

  g_scores = info['G-scores']
  g_scores.loc[:,'Total'] = g_scores.sum(axis = 1)
  g_scores.sort_values('Total', ascending = False, inplace = True)
  

  left, right = st.columns(2)

  with left:

    st.subheader('Draft board')
    selections_editable = st.data_editor(selections, hide_index = True)

    #figure out which drafter is next
    i = 0
    default_seat = None
    while (i < n_drafters * n_picks) and (default_seat is None):
      round = i // n_drafters 
      pick = i - round * n_drafters
      drafter = pick + 1 if round % 2 == 0 else n_drafters - pick

      #the condition should only trigger when the seat to check is blank 
      check_seat = selections_editable.loc[round, 'Drafter ' + str(drafter)]
      if check_seat != check_seat:
        default_seat = drafter

      i += 1 
        
    seat =  st.number_input(r'Analyze for which drafter?'
                    , min_value = 1
                    , value = default_seat
                   , max_value = n_drafters)

  with right:

    team_tab, cand_tab = st.tabs(["Team", "Candidates"])

    with team_tab:
    
      team_selections = selections_editable['Drafter ' + str(seat)].dropna()
  
      z_tab, g_tab = st.tabs(["Z-score", "G-score"])
        
      with z_tab:
        team_stats = z_scores[z_scores.index.isin(team_selections)]
        team_players = list(team_stats.index)

        n_players_on_team = team_stats.shape[0]
        expected = z_scores[0:n_players_on_team*n_drafters].mean() * n_players_on_team
        team_stats.loc['Total', :] = team_stats.sum(axis = 0)
        team_stats.loc['Expected', :] = expected

        team_stats = team_stats.style.format("{:.2}").applymap(other_styler).applymap(stat_styler, subset = pd.IndexSlice[team_players, counting_statistics + percentage_statistics])

        z_display = st.dataframe(team_stats)        
        
      with g_tab:
        team_stats = g_scores[g_scores.index.isin(team_selections)]

        n_players_on_team = team_stats.shape[0]
        expected = z_scores[0:n_players_on_team*n_drafters].mean() * n_players_on_team
        team_stats.loc['Total', :] = team_stats.sum(axis = 0)
        team_stats.loc['Expected', :] = expected

        team_stats = team_stats.style.format("{:.2}").applymap(other_styler).applymap(stat_styler, subset = pd.IndexSlice[team_players, counting_statistics + percentage_statistics])


        g_display = st.dataframe(team_stats)
        
    with cand_tab:

      subtab1, subtab2, subtab3 = st.tabs(["Z-score", "G-score", "H-score"])
    
      with subtab1:
        z_scores_unselected = z_scores[~z_scores.index.isin(listify(selections_editable))]
        z_scores_unselected = z_scores_unselected.style.format("{:.2}").applymap(other_styler).applymap(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics])
        z_scores_display = st.dataframe(z_scores_unselected)
        
      with subtab2:
        g_scores_unselected = g_scores[~g_scores.index.isin(listify(selections_editable))]
        g_scores_unselected = g_scores_unselected.style.format("{:.2}").applymap(other_styler).applymap(stat_styler, subset = pd.IndexSlice[:,counting_statistics + percentage_statistics])
        g_scores_display = st.dataframe(g_scores_unselected)
    
      with subtab3:

        winner_take_all = format == 'Head to Head: Most Categories'
        n_players = n_drafters * n_picks
      
        H = HAgent(info = info
                   , omega = omega
                   , gamma = gamma
                   , alpha = alpha
                   , beta = beta
                   , n_players = n_players
                   , winner_take_all = winner_take_all)
    
        players_chosen = [x for x in listify(selections_editable) if x ==x]
        my_players = [p for p in selections_editable['Drafter ' + str(seat)].dropna()]
    
        generator = H.get_h_scores(player_stats, my_players, players_chosen)
  
        placeholder = st.empty()
        all_res = []
        
        for i in range(n_iterations):
    
          c, res = next(generator)
          all_res = all_res + [res]
          #normalize weights by what we expect from other drafters
          c = pd.DataFrame(c, index = res.index, columns = categories)/info['v'].T
          c = (c * 100).round()
            
          with placeholder.container():

            score_tab, weight_tab = st.tabs(['Scores','Weights'])

            with score_tab:
              c1, c2 = st.columns([0.25,0.75])
    
              with c1:
                res = res.sort_values(ascending = False)
                res.name = 'H-score'
    
    
                st.dataframe(pd.DataFrame(res))
    
              with c2:
                st.plotly_chart(make_progress_chart(all_res))
  
            with weight_tab:
              c_df = c.loc[res.index].dropna().round().astype(int)
              c_df = c_df.style.background_gradient(axis = None)
              st.dataframe(c_df)
          

 

#below: use this for the color of results
#def color(pos):
#    col = color_map[pos]
#    return f'background-color: {color}'
#
#df = df.style.applymap(color, subset=pd.IndexSlice[:, ['pos']])
