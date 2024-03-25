#this code is adapted from 
#https://medium.com/@serranocarlosd/visualize-dataframe-changes-when-using-st-data-editor-in-streamlit-39f23e45fbb7

import pandas as pd
import streamlit as st

def reset(key_name):
    st.session_state[key_name] += 1
    diff_key = key_name + '_diff'
    st.session_state[diff_key] = None

def highlight_changes(val):
    color = f"color: white;" if val else "color:lightgrey;"
    background = f"background-color:#3580BB;" if val else ""
    return f"{color} {background}"

@st.cache_data(show_spinner = False)
def show_diff(
    source_df: pd.DataFrame
    , modified_df: pd.DataFrame
    , key_name
) -> None:

    editor_key = st.session_state[st.session_state[key_name]]

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

    for cl in change_markers:
        if cl in after_columns:
            new_col = cl.replace("_AFTER", "_BEFORE")
            change_markers[cl] = change_markers[cl] != change_markers[new_col]
            change_markers[new_col] = change_markers[cl]

    st.session_state[key_name + '_diff'] = changes.style.apply(
                                            lambda _: change_markers.applymap(highlight_changes), axis=None
                                        ).format("{:.1f}")
    st.session_state[key_name + '_diff_len'] = len(changes)

    version_key = key_name + '_version'
    st.session_state[version_key] += 1

def make_data_editor(data
                    , key_name
                    , lock_in_button_str):

    with st.form("Edit your data ⬇️ " + key_name):
        editor_df = st.data_editor(
            data, key=st.session_state[key_name]
                    , num_rows="dynamic"
                    , use_container_width=True
        )

        submitted = st.form_submit_button(lock_in_button_str
                                        , use_container_width = True
                                        , type = 'primary'
                                        , on_click = show_diff
                                        , args = (data, editor_df,key_name))

        diff_key = key_name + '_diff'

    #changes = show_diff(
    #                source_df=data
    #                , modified_df=editor_df
    #                , editor_key=st.session_state[st.session_state[key_name]]
    #)

    if diff_key in st.session_state:

        if st.session_state[diff_key] is not None:

            if st.session_state[key_name + '_diff_len']  > 0:
                st.subheader("Modified")
                st.caption("Showing only modified columns")

                st.dataframe(st.session_state[key_name + '_diff']
                                , use_container_width = True)

                st.button('Undo changes'
                    , on_click=reset
                    , use_container_width = True
                    , type = 'primary'
                    , args = (key_name,))
    
    return editor_df