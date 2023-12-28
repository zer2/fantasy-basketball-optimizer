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

df = pd.read_csv('./predictions.csv').set_index('Player')
df = df.drop(columns = ['ft','fg'])

df[r'Free Throw %'] = df[r'Free Throw %'] * 100
df[r'Field Goal %'] = df[r'Field Goal %'] * 100
df[r'No Play %'] = df[r'No Play %'] * 100

df = df.round(1)

edited_df = st.data_editor(df) # 👈 An editable dataframe

st.markdown(process_player_data())
st.markdown(run_algorithm())


#below: use this for the color of results
#def color(pos):
#    col = color_map[pos]
#    return f'background-color: {color}'
#
#df = df.style.applymap(color, subset=pd.IndexSlice[:, ['pos']])
