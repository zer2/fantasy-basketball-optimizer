import streamlit as st
import pandas as pd

st.markdown(f"Weekly player projections below: feel free to edit")

df = pd.read_csv('./predictions.csv').set_index('player')

edited_df = st.data_editor(df) # 👈 An editable dataframe
