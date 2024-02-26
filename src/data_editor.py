#this code is adapted from 
#https://medium.com/@serranocarlosd/visualize-dataframe-changes-when-using-st-data-editor-in-streamlit-39f23e45fbb7

import pandas as pd
import streamlit as st

def reset():
    st.session_state.player_stats_key += 1

def highlight_changes(val):
    color = f"color: black;" if val else "color:lightgray;"
    background = f"background-color:lightgray;" if val else ""
    return f"{color} {background}"

@st.cache_data()
def show_diff(
    source_df: pd.DataFrame
    , modified_df: pd.DataFrame
    , editor_key: dict
) -> None:
    target = pd.DataFrame(editor_key.get("edited_rows")).transpose()

    modified_columns = [i for i in target.notna().columns if i != "index"]
    source = source_df.iloc[target.index]
    source.index = pd.Series(source_df.index).iloc[target.index]

    target = target[modified_columns]
    target.index = pd.Series(source_df.index).iloc[target.index]

    changes = pd.merge(
        source[modified_columns],
        target,
        how="outer",
        left_index = True,
        right_index = True,
        suffixes=["_BEFORE", "_AFTER"],
    )
    after_columns = [i for i in changes.columns if "_AFTER" in i]
    for cl in changes:
        if cl in after_columns:
            new_col = cl.replace("_AFTER", "_BEFORE")
            changes[cl] = changes[cl].fillna(changes[new_col])

    change_markers = changes.copy()

    print(change_markers)
    for cl in change_markers:
        if cl in after_columns:
            new_col = cl.replace("_AFTER", "_BEFORE")
            change_markers[cl] = change_markers[cl] != change_markers[new_col]
            change_markers[new_col] = change_markers[cl]

    if len(changes) > 0:
        st.subheader("Modified")
        st.caption("Showing only modified columns")

        st.dataframe(
            changes.style.apply(
                lambda _: change_markers.applymap(highlight_changes), axis=None
            ).format("{:.1f}"),
            use_container_width=True
        )

    return changes

def make_data_editor(data):

    with st.form("Edit your data ⬇️"):
        editor_df = st.data_editor(
            data, key=st.session_state.player_stats_key
                    , num_rows="dynamic"
                    , use_container_width=True
        )
        submitted = st.form_submit_button("Lock in Player Stats"
                                        , use_container_width = True
                                        , type = 'primary')

    changes = show_diff(
                    source_df=data
                    , modified_df=editor_df
                    , editor_key=st.session_state[st.session_state.player_stats_key]
    )

    if len(changes) > 0:
        st.button('Undo changes'
            , on_click=reset
            , use_container_width = True
            , type = 'primary')
    
    return editor_df