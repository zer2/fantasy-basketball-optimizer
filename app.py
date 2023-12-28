import streamlit as st
import pandas as pd

df = pd.read_csv('./predictions.csv')

edited_df = st.data_editor(df) # 👈 An editable dataframe
