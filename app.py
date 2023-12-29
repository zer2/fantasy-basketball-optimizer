import streamlit as st
st.set_page_config(layout="wide")

import pandas as pd
from process_player_data import process_player_data
from run_algorithm import run_algorithm
from helper_functions import listify

tab1, tab2, tab3 = st.tabs(["Player Stats", "Parameters", "Draft"])

color_map = {'C' : 'yellow'
             ,'PF' : 'green'
             ,'SF' : 'green'
             ,'SG' : 'red'
             ,'PG' : 'red'}

df = pd.read_csv('./predictions.csv').set_index('Player')
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
  c1, c2, c3 = st.columns(3)

  with c1: 
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

  
  with c2: 
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
      coefficients = st.data_editor(coefficient_df)

  
  with c3:
    st.header('Algorithm')

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

    alpha = st.number_input(r'Select a $\alpha$ value', value = 0.01, min_value = 0.0)
    alpha_str = r'''$\alpha$ is the initial step size for gradient descent. Tuning $\alpha$ is not recommended'''
    st.caption(alpha_str)

    beta = st.number_input(r'Select a $\beta$ value', value = 0.25, min_value = 0.0)
    beta_str = r'''$\beta$ is the degree of step size decay. Tuning $\beta$ is not recommended'''
    st.caption(beta_str)

    n_iterations = st.number_input(r'Select a number of iterations for gradient descent to run', value = 30, min_value = 0, max_value = 1000)
    n_iterations_str = r'''More iterations take more computational power, but theoretically achieve better convergence'''
    st.caption(n_iterations_str)

with tab3:
  g_scores, z_scores, x_scores, positions, v, L = process_player_data(player_stats, coefficients, psi, nu, n_drafters, n_picks)

  #perhaps the dataframe should be uneditable, and users just get to enter the next players picked? With an undo button?
  selections = pd.DataFrame({'Drafter ' + str(n+1) : [''] * n_picks for n in range(n_drafters)})
  selections_editable = st.data_editor(selections)

  subtab1, subtab2, subtab3 = st.tabs(["Z-scores", "G-scores", "H-score Algorithm"])

  with subtab1:
    z_scores.loc[:,'Total'] = z_scores.sum(axis = 1)
    z_scores.sort_values('Total', ascending = False, inplace = True)
    z_scores_unselected = st.dataframe(z_scores[~z_scores.index.isin(listify(selections_editable))])
    
  with subtab2:
    g_scores.loc[:,'Total'] = g_scores.sum(axis = 1)
    g_scores.sort_values('Total', ascending = False, inplace = True)
    g_scores_unselected = st.dataframe(g_scores[~g_scores.index.isin(listify(selections_editable))])

  with subtab3:
    st.header('H scoring')

 

#below: use this for the color of results
#def color(pos):
#    col = color_map[pos]
#    return f'background-color: {color}'
#
#df = df.style.applymap(color, subset=pd.IndexSlice[:, ['pos']])
