import streamlit as st
import pandas as pd

df = pd.read_csv('./predictions.csv').set_index('player')

edited_df = st.data_editor(df) # 👈 An editable dataframe
