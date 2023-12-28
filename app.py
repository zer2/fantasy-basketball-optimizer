import streamlit as st
import pandas as pd
from process_player_data import process_player_data
from run_algorithm import run_algorithm

st.markdown(f"Weekly player projections below: feel free to edit")

color_map = {'C' : 'yellow'
             ,'PF' : 'green'
             ,'SF' : 'green'
             ,'SG' : 'red'
             ,'PG' : 'red'}
             
def color(pos):
    col = color_map[pos]
    return f'background-color: {color}'

df = pd.read_csv('./predictions.csv').set_index('player')
df = df.style.applymap(color, subset=['pos'])

edited_df = st.data_editor(df) # 👈 An editable dataframe

st.markdown(process_player_data())
st.markdown(run_algorithm())


