  
import streamlit as st 
from pandas.api.types import CategoricalDtype
from src.helpers.helper_functions import listify, move_back_one_pick, move_forward_one_pick, increment_player_stats_version
from src.tabs.team_subtabs import *
from src.tabs.candidate_subtabs import *
from src.helpers.helper_functions import move_forward_one_pick, increment_default_key

import pandas as pd 
import numpy as np
from src.math.algorithm_helpers import savor_calculation
from src.math.algorithm_agents import get_base_h_score

def run_h_score():
    st.session_state.run_h_score = True

def stop_run_h_score():
    st.session_state.run_h_score = False

def lock_in():
   st.session_state.have_locked_in = True

def clear_draft_board():
  if 'draft_results' in st.session_state:
    st.session_state.draft_results = None

  if 'selections_df' in st.session_state:
    st.session_state.selections_df = st.session_state.selections_default

def run_autodraft():
  
  if 'selections_df' not in st.session_state:
    st.session_state.selections_df = st.session_state.selections_default

  while (st.session_state.team_names[st.session_state.drafter] in st.session_state.autodrafters) and (st.session_state.row < st.session_state.n_starters):
    selection_list = listify(st.session_state.selections_df)
    g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]
    select_player_from_draft_board(g_scores_unselected.index[0])

#run_autodraft_and_increment
def increment_and_reset_draft():
    increment_player_stats_version()
    clear_draft_board()

    if ('autodrafters' in st.session_state) and ('drafter' in st.session_state):
      run_autodraft()

    st.session_state.live_draft_active = False

    if 'selections_df' in st.session_state:
        del st.session_state.selections_df

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
  
  st.session_state.selections_df.iloc[st.session_state.row, st.session_state.drafter] = np.nan

  run_autodraft()

def clear_board():
  st.session_state.selections_df = st.session_state.selections_default
  st.session_state.drafter = 0
  st.session_state.row = 0

@st.fragment
def make_drafting_tab_own_data(H):

    left, right = st.columns([0.47,0.53])

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
            
        player_assignments = st.session_state.selections_df[0:st.session_state.n_starters].to_dict('list')

        st.dataframe(st.session_state.selections_df
                                    ,key = 'selections_df'
                                        , hide_index = True
                                        , height = st.session_state.n_starters * 35 + 50)

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

            if st.session_state.run_h_score:
                st.session_state.run_h_score = False

                make_h_cand_tab(H
                    ,st.session_state.g_scores
                    ,st.session_state.z_scores
                    ,player_assignments
                    ,draft_seat
                    ,st.session_state.n_iterations
                    ,st.session_state.v
                    ,5)
            else:

                def run():
                   st.session_state.run_h_score = True

                button = st.button('Run H-score', on_click = run)


        with team_tab:

            if len(my_players) >= st.session_state.n_starters:
                base_h_res = get_base_h_score(st.session_state.info
                                                ,st.session_state.omega
                                                ,st.session_state.gamma
                                                ,st.session_state.n_starters
                                                ,st.session_state.n_drafters
                                                ,st.session_state.scoring_format
                                                ,st.session_state.chi
                                                ,player_assignments
                                                ,draft_seat
                                                ,st.session_state.info_key)

            else:
                base_h_res = None

            make_full_team_tab(st.session_state.z_scores
                            ,st.session_state.g_scores
                            ,my_players
                            ,st.session_state.n_drafters
                            ,st.session_state.n_starters
                            ,base_h_res
                            ,st.session_state.info_key
                            ,draft_seat
                            )

        
def refresh_analysis():

    player_metadata = st.session_state.player_metadata.copy()

    player_metadata.index = [' '.join(player.split('(')[0].split(' ')[0:2]) for player in player_metadata.index]

    if st.session_state.mode == 'Draft Mode':

        draft_results, error_string = st.session_state.integration.get_draft_results(player_metadata)
                
    else:

        draft_results, error_string  = st.session_state.integration.get_auction_results(player_metadata)
            

    st.session_state.draft_results = draft_results

    if error_string == 'Success':
        st.session_state.live_draft_active = True
    else:
        if st.session_state.live_draft_active:
            st.error(error_string)
            st.stop()
   
@st.fragment
def make_drafting_tab_live_data(H):

    st.session_state.player_metadata = st.session_state.player_stats['Position']

    if 'team_names' not in st.session_state:
        st.write('No league info has been passed')
    else:

        if st.session_state.draft_results is None:
            refresh_analysis()

        c1, c2 = st.columns([0.1,0.9])

        with c1:
            st.button('Refresh Analysis', on_click = refresh_analysis)

        with c2:
            draft_seat = st.selectbox(f'Which drafter are you?'
            , st.session_state.integration.get_team_names(st.session_state.integration.league_id)
            , key = 'draft_seat'
            , on_change = refresh_analysis)

        if not st.session_state.live_draft_active:

            st.write('Draft has not yet begun')

        else:
            
            player_assignments = st.session_state.draft_results[0:st.session_state.n_starters].to_dict('list')

            my_players = st.session_state.draft_results[st.session_state.draft_seat].dropna()
            
            cand_tab, team_tab = st.tabs(["Candidates","Team"])

            with team_tab:

                if len(my_players) >= st.session_state.n_starters:
                    base_h_res = get_base_h_score(st.session_state.info
                                                    ,st.session_state.omega
                                                    ,st.session_state.gamma
                                                    ,st.session_state.n_starters
                                                    ,st.session_state.n_drafters
                                                    ,st.session_state.scoring_format
                                                    ,st.session_state.chi
                                                    ,player_assignments
                                                    ,draft_seat
                                                    ,st.session_state.info_key)

                else:
                    base_h_res = None
                    

                make_full_team_tab(st.session_state.z_scores
                                ,st.session_state.g_scores
                                ,my_players
                                ,st.session_state.n_drafters
                                ,st.session_state.n_starters
                                ,base_h_res
                                ,st.session_state.info_key
                                ,draft_seat
                                )        

            with cand_tab:

                if len(my_players) < st.session_state.n_starters:

                    make_h_cand_tab(H
                        ,st.session_state.g_scores
                        ,st.session_state.z_scores
                        ,player_assignments
                        ,draft_seat
                        ,st.session_state.n_iterations
                        ,st.session_state.v
                        ,5)
                else:
                    st.write('You have selected all of your players')

@st.fragment 
def make_auction_tab_own_data(H):
      left, right = st.columns([0.4,0.6])

      with left:

        cash_per_team = st.number_input(r'How much cash does each team have to pick players?'
                  , key = 'cash_per_team'
                  , min_value = 1
                  , value = 200)

        auction_selections_default = pd.DataFrame([[None] * 3] * st.session_state.n_picks * st.session_state.n_drafters
                                          ,columns = ['Player','Team','Cost'])

        player_category_type = CategoricalDtype(categories=list(st.session_state.player_stats.index), ordered=True)

        auction_selections_default.loc[:'Player'] = \
            auction_selections_default.loc[:'Player'].astype(player_category_type)

        st.caption("""Enter which players have been selected by which teams, and for how much, below""")

        with st.form('manual_auctions'):

            auction_selections = st.data_editor(auction_selections_default
                        ,column_config = 
                        {"Player" : st.column_config.SelectboxColumn(options = list(st.session_state.player_stats.index))
                        ,"Team" : st.column_config.SelectboxColumn(options = st.session_state.team_names)
                        ,'Cost' : st.column_config.NumberColumn(min_value = 0
                                                                , step = 1)}
                        , hide_index = True
                        , use_container_width = True
                        )
            
            c1, c2 = st.columns([0.2,0.8])
            
            with c1: 
              submit = st.form_submit_button("Lock in",on_click = lock_in)
            with c2:
              st.warning('Lock in to update candidate display')

        selection_list = auction_selections['Player'].dropna()

        total_cash = cash_per_team * st.session_state.n_drafters

        amount_spent = auction_selections['Cost'].dropna().sum()

        remaining_cash = total_cash - amount_spent
        
        st.caption(r'\$' + str(remaining_cash) + r' remains out of \$' + str(total_cash) + ' originally available' )

      with right: 
        auction_seat = st.selectbox(f'Which team are you?'
            , st.session_state.team_names
            , index = 0)
        
        cash_spent_per_team = auction_selections.dropna().groupby('Team', observed = False)['Cost'].sum()
        cash_remaining_per_team = cash_per_team - cash_spent_per_team
        player_assignments = auction_selections.dropna().groupby('Team', observed = False)['Player'].apply(list)

        for team in st.session_state.team_names:
          if not team in cash_remaining_per_team.index:
            cash_remaining_per_team.loc[team] = cash_per_team

          if not team in player_assignments.index:
            player_assignments.loc[team] = []

        my_players = player_assignments[auction_seat]
        n_my_players = len(my_players)

        my_remaining_cash = cash_remaining_per_team[auction_seat]

        st.caption(r'You have \$' + str(my_remaining_cash) + r' remaining out of \$' + str(cash_per_team) \
                + ' to select ' + str(st.session_state.n_picks - n_my_players) + ' of ' + str(st.session_state.n_picks) + ' players')

        cand_tab, team_tab = st.tabs(["Candidates","Team"])
              
        with cand_tab:

          z_cand_tab, g_cand_tab, h_cand_tab = st.tabs(["Z-score", "G-score", "H-score"])
                    
          with z_cand_tab:
            
            make_cand_tab(st.session_state.z_scores
                          ,selection_list
                          , st.session_state.params['z-score-player-multiplier']
                          ,remaining_cash
                          ,st.session_state.n_drafters * st.session_state.n_picks
                          ,st.session_state.info_key)

          with g_cand_tab:

            make_cand_tab(st.session_state.g_scores
                          , selection_list
                          , st.session_state.params['g-score-player-multiplier']
                          ,remaining_cash
                          ,st.session_state.n_drafters * st.session_state.n_picks
                          ,st.session_state.info_key)

          with h_cand_tab:

            if not st.session_state.have_locked_in:
               st.markdown('Lock in to run H-score')

            elif len(my_players) == st.session_state.n_picks:
              st.markdown('Team is complete!')
                        
            else:

              h_ranks_unselected = st.session_state.h_ranks[~st.session_state.h_ranks.index.isin(selection_list)]
              h_defaults_savor = savor_calculation(h_ranks_unselected['H-score']
                                                            , st.session_state.n_picks * st.session_state.n_drafters - len(selection_list)
                                                            , remaining_cash
                                                            , st.session_state['streaming_noise_h'])
              
              h_defaults_savor = pd.Series(h_defaults_savor.values, index = h_ranks_unselected['Player'])

              make_h_cand_tab(H
                    ,st.session_state.g_scores
                    ,st.session_state.z_scores
                    ,player_assignments.to_dict()
                    ,auction_seat
                    ,st.session_state.n_iterations
                    ,st.session_state.v
                    ,5 #display frequency
                    ,cash_remaining_per_team.to_dict()
                    ,h_defaults_savor
                    ,st.session_state.n_drafters * st.session_state.n_picks)

        with team_tab:

            if len(my_players) >= st.session_state.n_starters:
                base_h_res = get_base_h_score(st.session_state.info
                                                ,st.session_state.omega
                                                ,st.session_state.gamma
                                                ,st.session_state.n_starters
                                                ,st.session_state.n_drafters
                                                ,st.session_state.scoring_format
                                                ,st.session_state.chi
                                                ,player_assignments.to_dict()
                                                ,auction_seat
                                                ,st.session_state.info_key)

            else:
                base_h_res = None

            make_full_team_tab(st.session_state.z_scores
                                ,st.session_state.g_scores
                                ,my_players
                                ,st.session_state.n_drafters
                                ,st.session_state.n_picks
                                ,base_h_res
                                ,st.session_state.info_key
                                ,auction_seat
                                )
@st.fragment
def make_auction_tab_live_data(H):

    st.session_state.player_metadata = st.session_state.player_stats['Position']

    if 'team_names' not in st.session_state:
        st.write('No league info has been passed')
    else:

        if st.session_state.draft_results is None:
            refresh_analysis()

        c1, c2 = st.columns([0.1,0.9])

        with c1:
            st.button('Refresh Analysis'
                        , on_click = refresh_analysis)
            
        with c2:
            auction_seat = st.selectbox(f'Which drafter are you?'
            , st.session_state.team_names
            , key = 'auction_seat'
            , on_change = refresh_analysis)
        
        if not st.session_state.live_draft_active:

            st.write('Auction has not begun, or just began. There is sometimes a lag of ~1 minute before data can be fetched')

        else:

            cand_tab, team_tab = st.tabs(["Candidates","Team"])

            cash_per_team = 200
            
            selection_list = st.session_state.draft_results['Player'].dropna()
            player_assignments = st.session_state.draft_results.dropna()  \
                                .groupby('Team', observed = False)['Player'].apply(list)
            
            cash_spent_per_team = st.session_state.draft_results.dropna().groupby('Team', observed = False)['Cost'].sum()
            cash_remaining_per_team = cash_per_team - cash_spent_per_team

            total_cash = cash_per_team * st.session_state.n_drafters

            amount_spent = st.session_state.draft_results['Cost'].dropna().sum()

            remaining_cash = total_cash - amount_spent

            for team in st.session_state.team_names:
                if not team in cash_remaining_per_team.index:
                    cash_remaining_per_team.loc[team] = cash_per_team

                if not team in player_assignments.index:
                    player_assignments.loc[team] = []

            my_players = player_assignments[auction_seat]
            n_my_players = len(my_players)

            my_remaining_cash = cash_remaining_per_team[auction_seat]

            h_ranks_unselected = st.session_state.h_ranks[~st.session_state.h_ranks['Player'].isin(selection_list)]

            h_defaults_savor = savor_calculation(h_ranks_unselected['H-score']
                                                            , st.session_state.n_picks * st.session_state.n_drafters - len(selection_list)
                                                            , remaining_cash
                                                            , st.session_state['streaming_noise_h'])
                                
            h_defaults_savor = pd.Series(h_defaults_savor.values, index = h_ranks_unselected['Player'])

            #For when the rank page gets out of synch with the number of drafters and therefore the amount of cash available
            h_defaults_savor = h_defaults_savor * np.sum([v for k, v in cash_remaining_per_team.items()])/h_defaults_savor.sum()

            with team_tab:

                if len(my_players) >= st.session_state.n_picks:

                    base_h_res = get_base_h_score(st.session_state.info
                                                    ,st.session_state.omega
                                                    ,st.session_state.gamma
                                                    ,st.session_state.n_picks
                                                    ,st.session_state.n_drafters
                                                    ,st.session_state.scoring_format
                                                    ,st.session_state.chi
                                                    ,player_assignments.to_dict()
                                                    ,auction_seat
                                                    ,st.session_state.info_key)


                else:
                    base_h_res = None

                make_full_team_tab(st.session_state.z_scores
                                    ,st.session_state.g_scores
                                    ,my_players
                                    ,st.session_state.n_drafters
                                    ,st.session_state.n_picks
                                    ,base_h_res
                                    ,st.session_state.info_key
                                    ,auction_seat
                                    )
                
            with cand_tab:

                if len(my_players) < st.session_state.n_picks:

                    make_h_cand_tab(H
                        ,st.session_state.g_scores
                        ,st.session_state.z_scores
                        ,player_assignments.to_dict()
                        ,auction_seat
                        ,st.session_state.n_iterations
                        ,st.session_state.v
                        ,5 #display frequency
                        ,cash_remaining_per_team.to_dict()
                        ,h_defaults_savor
                        ,st.session_state.n_drafters * st.session_state.n_picks)
                    
                else:
                    st.write('Your team is full')