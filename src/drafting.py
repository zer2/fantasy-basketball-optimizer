  
import streamlit as st 
from src.helper_functions import listify, move_back_one_pick, move_forward_one_pick
from src.tabs import *
from src import yahoo_connect

def run_autodraft():
  while (st.session_state.selections_df.columns[st.session_state.drafter] in st.session_state.autodrafters) and (st.session_state.row < st.session_state.n_picks):
    selection_list = listify(st.session_state.selections_df)
    g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]
    select_player_from_draft_board(g_scores_unselected.index[0])

def select_player_from_draft_board(p = None):

  if not p:
    p = st.session_state.selected_player

  st.session_state.selections_df.iloc[st.session_state.row, st.session_state.drafter] = p

  st.session_state.row, st.session_state.drafter = move_forward_one_pick(st.session_state.row
                                                                          ,st.session_state.drafter
                                                                          ,st.session_state.selections_df.shape[1])
  
  run_autodraft()

def undo_selection():
  st.session_state.row, st.session_state.drafter = move_back_one_pick(st.session_state.row
                                    , st.session_state.drafter
                                    , st.session_state.selections_df.shape[1])
  
  st.session_state.selections_df.iloc[st.session_state.row, st.session_state.drafter] = None

  run_autodraft()

def clear_board():
  st.session_state.selections_df = st.session_state.selections_default
  st.session_state.drafter = 0
  st.session_state.row = 0

def make_drafting_tab_own_data(H):

    left, right = st.columns(2)

    with left:

        selection_list = listify(st.session_state.selections_df)
        g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]

        with st.form("pick form", border = False):
            st.selectbox('Select Pick ' + str(st.session_state.row) + ' for ' + \
                                    st.session_state.selections_df.columns[st.session_state.drafter]
                                    ,key = 'selected_player'
                                    ,options = g_scores_unselected.index)

            button_col1, button_col2, button_col3 = st.columns(3)

            with button_col1:
                st.form_submit_button("Lock in selection"
                                        , on_click = select_player_from_draft_board
                                        , use_container_width = True)
            
            with button_col2:
                st.form_submit_button("Undo last selection"
                                        , on_click = undo_selection
                                        , use_container_width = True)
            with button_col3:
                st.form_submit_button("Clear draft board"
                                        , on_click = clear_board
                                        , use_container_width = True)
            
        player_assignments = st.session_state.selections_df.to_dict('list')

        st.dataframe(st.session_state.selections_df
                                    ,key = 'selections_df'
                                        , hide_index = True
                                        , height = st.session_state.n_picks * 35 + 50)

    with right:

        non_autodrafters = [i for i,c in zip(range(st.session_state.selections_default.shape[1])
                                             ,st.session_state.selections_default.columns) 
                            if c not in st.session_state.autodrafters]

        draft_seat = st.selectbox(f'Which drafter are you?'
            , st.session_state.selections_df.columns
            , index = 0 if len(non_autodrafters) ==0 else non_autodrafters[0])

        my_players = st.session_state.selections_df[draft_seat].dropna()

        cand_tab, team_tab = st.tabs(["Candidates","Team"])

        with cand_tab:

            z_cand_tab, g_cand_tab, h_cand_tab = st.tabs(["Z-score", "G-score", "H-score"])
                    
            with z_cand_tab:
            
                make_cand_tab(st.session_state.z_scores
                                ,selection_list
                                , st.session_state.params['z-score-player-multiplier']
                                , info_key = st.session_state.info_key)

            with g_cand_tab:

                make_cand_tab(st.session_state.g_scores
                                , selection_list
                                , st.session_state.params['g-score-player-multiplier']
                                , info_key = st.session_state.info_key)

            with h_cand_tab:

                if st.session_state.selections_default.columns[st.session_state.drafter] == draft_seat:

                    make_h_cand_tab(H
                        ,st.session_state.g_scores
                        ,st.session_state.z_scores
                        ,player_assignments
                        ,draft_seat
                        ,st.session_state.n_iterations
                        ,st.session_state.v
                        ,30)
                else:
                    st.write('It is not your turn, so H-scoring will not run')

        with team_tab:

            if len(my_players) == st.session_state.n_picks:
                base_h_res = get_base_h_score(st.session_state.info
                                                ,st.session_state.omega
                                                ,st.session_state.gamma
                                                ,st.session_state.n_picks
                                                ,st.session_state.n_drafters
                                                ,st.session_state.scoring_format
                                                ,st.session_state.chi
                                                ,player_assignments
                                                ,draft_seat
                                                ,st.session_state.info_key)

                base_h_score = base_h_res['Scores']
                base_win_rates = base_h_res['Rates']

            else:
                base_h_score = None
                base_win_rates = None

            make_full_team_tab(st.session_state.z_scores
                            ,st.session_state.g_scores
                            ,my_players
                            ,st.session_state.n_drafters
                            ,st.session_state.n_picks
                            ,base_h_score
                            ,base_win_rates
                            ,st.session_state.info_key
                            )

        




def make_drafting_tab_live_data(H):

    st.session_state.player_metadata = st.session_state.player_stats['Position']

    def refresh_analysis():
        yahoo_league_id = st.session_state.yahoo_league_id
        auth_dir = st.session_state.auth_dir
        player_metadata = st.session_state.player_metadata.copy()

        player_metadata.index = [player.split('(')[0][0:-1] for player in player_metadata.index]

        st.session_state.draft_results = yahoo_connect.get_draft_results(yahoo_league_id
                                                                            , auth_dir
                                                                            , player_metadata)
        

    if 'team_names' in st.session_state:

        c1, c2 = st.columns([0.1,0.9])

        with c1:
            st.button('Refresh Analysis', on_click = refresh_analysis)

        with c2:
            draft_seat = st.selectbox(f'Which drafter are you?'
            , st.session_state.team_names
            , key = 'draft_seat'
            , index = 0)
       
    if st.session_state.draft_results is None:

        st.write('Draft has not yet begun')

    else:
        
        selection_list = listify(st.session_state.draft_results) 
        player_assignments = st.session_state.draft_results.to_dict('list')

        my_players = st.session_state.draft_results[st.session_state.draft_seat].dropna()
        
        cand_tab, team_tab = st.tabs(["Candidates","Team"])

        with cand_tab:

            if len(my_players) < st.session_state.n_picks:

                make_h_cand_tab(H
                    ,st.session_state.g_scores
                    ,st.session_state.z_scores
                    ,player_assignments
                    ,draft_seat
                    ,st.session_state.n_iterations
                    ,st.session_state.v
                    ,100)

        with team_tab:

            if len(my_players) == st.session_state.n_picks:
                base_h_res = get_base_h_score(st.session_state.info
                                                ,st.session_state.omega
                                                ,st.session_state.gamma
                                                ,st.session_state.n_picks
                                                ,st.session_state.n_drafters
                                                ,st.session_state.scoring_format
                                                ,st.session_state.chi
                                                ,player_assignments
                                                ,draft_seat
                                                ,st.session_state.info_key)

                base_h_score = base_h_res['Scores']
                base_win_rates = base_h_res['Rates']

            else:
                base_h_score = None
                base_win_rates = None

            make_full_team_tab(st.session_state.z_scores
                            ,st.session_state.g_scores
                            ,my_players
                            ,st.session_state.n_drafters
                            ,st.session_state.n_picks
                            ,base_h_score
                            ,base_win_rates
                            ,st.session_state.info_key
                            )        

