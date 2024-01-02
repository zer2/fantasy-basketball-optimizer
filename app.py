import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from pandas.api.types import CategoricalDtype
from process_player_data import process_player_data
from run_algorithm import HAgent
from helper_functions import listify, make_progress_chart

st.title('Optimization for fantasy basketball: based on [this paper](https://arxiv.org/abs/2307.02188)') 
tab1, tab2, tab3 = st.tabs(["Player Stats", "Parameters", "Draft"])

color_map = {'C' : 'yellow'
             ,'PF' : 'green'
             ,'SF' : 'green'
             ,'SG' : 'red'
             ,'PG' : 'red'}

df = pd.read_csv('./predictions.csv').set_index('Player').sort_index()
df = df.drop(columns = ['ft','fg'])

df[r'Free Throw %'] = df[r'Free Throw %'] * 100
df[r'Field Goal %'] = df[r'Field Goal %'] * 100
df[r'No Play %'] = df[r'No Play %'] * 100

df = df.round(1)

coefficient_df = pd.read_csv('./coefficients.csv', index_col = 0)

with tab1:
  st.markdown(f"Weekly player projections below: feel free to edit")

  player_stats = st.data_editor(df) # 👈 An editable dataframe

with tab2: 
  left, middle, right = st.columns([0.25,0.25,0.5])

  with left: 
    st.header('General')
    
    format = st.selectbox(
      'Which format are you playing?',
      ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories'))
  
    if format == 'Rotisserie':
      st.caption('Note that only Z-scores are available for Rotisserie. No advanced algorithms for Rotisserie have been implemented')
    else:
      st.caption('Head to head formats are supported with G-scores and H-scores. Z-scores are also available but not advisable to use')

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

with tab3:

  rotisserie = format == 'Rotisserie'
  
  info = process_player_data(player_stats, coefficients, psi, nu, n_drafters, n_picks, rotisserie)

  #perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
  selections = pd.DataFrame({'Drafter ' + str(n+1) : [None] * n_picks for n in range(n_drafters)})

  #make the selection df use a categorical variable for players, so that only players can be chosen, and it autofills
  player_category_type = CategoricalDtype(categories=list(player_stats.index), ordered=True)
  selections = selections.astype(player_category_type)

  z_scores = info['Z-scores']
  g_scores = info['G-scores']
  categories = z_scores.columns

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
      
      metric = 'Z-score' if format == 'Rotisserie' else 'G-score'
      st.caption('For ' + format + ', it is recommended to evaluate your team by ' + metric)
    
      team_selections = selections_editable['Drafter ' + str(seat)].dropna()
  
      z_tab, g_tab = st.tabs(["Z-score", "G-score"])
      with z_tab:
        team_stats = z_scores[z_scores.index.isin(team_selections)]
        team_stats.loc['Total', :] = team_stats.sum(axis = 0)
        team_stats = team_stats.round(2)
        z_display = st.dataframe(team_stats)
      with g_tab:
        team_stats = g_scores[g_scores.index.isin(team_selections)]
        team_stats.loc['Total', :] = team_stats.sum(axis = 0)
        team_stats = team_stats.round(2)
        g_display = st.dataframe(team_stats)

    with cand_tab:

      if format == 'Rotisserie': 
        st.caption('For Rotisserie, it is recommended to select players based on ' + metric + ' or H-score. However, keep in mind that H-score for Rotisserie is experimental')
      else: 
        st.caption('For ' + format + ', it is recommended to select players based on H-score' )

      for i in range(n_iterations):
  
        c, res = next(generator)
        all_res = all_res + [res]
        c = pd.DataFrame(c, index = res.index, columns = categories)
          
      with placeholder.container():
        
        subtab1, subtab2, subtab3, subtab5 = st.tabs(["Z-scores", "G-scores", "H-score","H-weight"])
      
        with subtab1:
          z_scores.loc[:,'Total'] = z_scores.sum(axis = 1)
          z_scores.sort_values('Total', ascending = False, inplace = True)
          z_scores = z_scores.round(2)
          z_scores_unselected = st.dataframe(z_scores[~z_scores.index.isin(listify(selections_editable))])
          
        with subtab2:
          g_scores.loc[:,'Total'] = g_scores.sum(axis = 1)
          g_scores.sort_values('Total', ascending = False, inplace = True)
          g_scores = g_scores.round(2)
          g_scores_unselected = st.dataframe(g_scores[~g_scores.index.isin(listify(selections_editable))])
      
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
      
          players_chosen = listify(selections_editable)
          my_players = [p for p in selections_editable['Drafter ' + str(seat)].dropna()]
      
          generator = H.get_h_scores(player_stats, my_players, players_chosen)
    
          placeholder = st.empty()
          all_res = []
          
          with subtab3:
            c1, c2 = st.columns([0.25,0.75])

            with c1:
              res = res.sort_values(ascending = False)
              res.name = 'H-score'

              st.dataframe(pd.DataFrame(res))

            with c2:
              st.plotly_chart(make_progress_chart(all_res))

          with subtab4:
            st.dataframe(c.loc[res.index])
            

 

#below: use this for the color of results
#def color(pos):
#    col = color_map[pos]
#    return f'background-color: {color}'
#
#df = df.style.applymap(color, subset=pd.IndexSlice[:, ['pos']])
