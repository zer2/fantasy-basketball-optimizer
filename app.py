import streamlit as st
import pandas as pd
from process_player_data import process_player_data
from run_algorithm import run_algorithm

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

with tab1:
  st.markdown(f"Weekly player projections below: feel free to edit")

  edited_df = st.data_editor(df) # 👈 An editable dataframe

with tab2: 

  st.header('General options')
  
  format = st.selectbox(
    'Which format are you playing?',
    ('Rotisserie', 'Head to Head: Each Category', 'Head to Head: Most Categories'))

  if format == 'Rotisserie':
    st.write('Note that only Z-scores are available for Rotisserie. No advanced algorithms for Rotisserie have been implemented')

  st.header('Algorithm parameters')

  omega = st.number_input(r'$\omega$', value = 1.5)
  omega_str = r'''The higher $\omega$ is, the more aggressively the algorithm will try to punt. Slightly more technically, 
                  it quantifies how much better the optimal player choice will be compared to the player that would be 
                  chosen with baseline weights'''
  st.write(omega_str)

  gamma = st.number_input(r'$\gamma$ value', value = 0.1)
  gamma_str = r'''$\gamma$ also influences the level of punting, complementing omega. Tuning gamma is not suggested but you can 
          tune it if you want. Higher values imply that the algorithm will have to give up more general value to find the
           players that  work best for its strategy'''
  st.write(gamma_str)

  psi = st.number_input(r'$\psi$'
                        , min_value = 0.0
                        , value = 0.85
                       , max_value = 1.0)
  psi_str = r'''$\psi$ controls how injury rates are dealt with. If $\psi$ is $X\%$, and a player is expected to miss $Y\%$ of weeks
                entirely, their counting statistics will be multplied by $(1-Y*X)$. So for example is if $\psi$ is $50\%$ and a 
                player is expected to miss $20\%$ of weeks, their counting statistics will be multplied by $(1-0.5*0.2) =  90\%$'''

  st.write(psi_str)



with tab3:
  st.markdown(process_player_data())
  st.markdown(run_algorithm())


#below: use this for the color of results
#def color(pos):
#    col = color_map[pos]
#    return f'background-color: {color}'
#
#df = df.style.applymap(color, subset=pd.IndexSlice[:, ['pos']])
