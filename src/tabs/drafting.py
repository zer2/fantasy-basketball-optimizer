  
import streamlit as st 
import pandas as pd 
from pandas.api.types import CategoricalDtype
from src.math import process_player_data
import numpy as np

from src.tabs.season_mode import make_team_display
from src.tabs.candidate_subtabs import make_cand_tab
from src.helpers.helper_functions import get_beth, get_chi, get_data_from_session_state, get_data_key \
                                            , get_default_draft_seat_index, get_draft_position \
                                            , get_gamma, get_n_iterations, get_n_picks, get_n_starters \
                                            , get_omega, get_psi, get_selected_players, get_selections_df \
                                            , get_style_base, h_score_is_running, initialize_selections_df \
                                            , modify_selections_df, move_back_one_pick, move_forward_one_pick \
                                            , get_n_drafters, remove_selections_df, run_h_score, set_draft_position \
                                            , stop_run_h_score, store_dataset_in_session_state, get_team_names \
                                            , get_n_picks, get_streaming_noise, get_scoring_format, get_params \
                                            , get_selected_categories, get_mode

from src.math.algorithm_agents import build_h_agent, get_default_h_values
from src.math.algorithm_helpers import auction_value_adjuster
from src.math.process_player_data import process_player_data
#from wfork_streamlit_profiler import Profiler

@st.fragment
def make_drafting_tab_own_data():
    """Create a page for drafting based on manual input
    This requires a significantly different UI from drafting based on a live connection, because the user has to enter players

    Args:
        H: H-scoring Agent for performing calculations

    Returns:
        None
    """

    info = get_data_from_session_state('info')
    info_key = get_data_key('info')

    left, right = st.columns([0.47,0.53])

    selections_df = get_selections_df()

    with left:

        selection_list = get_selected_players()
        
        g_scores_unselected = info['G-scores'][~info['G-scores'].index.isin(selection_list)]

        with st.form("pick form", border = False):
            row, drafter = get_draft_position()

            st.selectbox('Select Pick ' + str(row) + ' for ' + \
                                    get_team_names()[drafter]
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
                                        , on_click = clear_draft_board
                                        , use_container_width = True)
                            

        st.dataframe(selections_df
                                        , hide_index = True
                                        , height = get_n_starters() * 35 + 50)
        
        player_assignments = selections_df[0:get_n_starters()].to_dict('list')


    with right:

        draft_seat = st.selectbox(f'Which drafter are you?'
            , get_team_names())


    with left:

        my_players = selections_df[draft_seat].dropna()

        make_team_display(info_key
                        ,my_players
                        ,get_style_base()
                        )
    with right:

        if h_score_is_running():
            if len(my_players) < get_n_starters():
                make_cand_tab(player_assignments
                    ,draft_seat
                    ,get_n_iterations())
                stop_run_h_score()
            else:
                st.write('You have selected all of your players')    
            
        else:

            st.button('Run algorithm', on_click = run_h_score)

        

@st.fragment
def make_drafting_tab_live_data():
    """Create a page for drafting based on a live connection e.g. from Yaho

    Args:
        H: H-scoring Agent for performing calculations

    Returns:
        None
    """
    update_data_and_info()
    info_key = get_data_key('info')

    if 'team_names' not in st.session_state:
        st.write('No league info has been passed')
    else:

        if st.session_state.draft_results is None:
            refresh_analysis()

        c1, c2 = st.columns([0.1,0.9])
        
        c1.button('Refresh Analysis', on_click = refresh_analysis)

        team_names = get_team_names()

        default_index = get_default_draft_seat_index()

        draft_seat = c2.selectbox(f'Which drafter are you?'
                , team_names
                , key = 'draft_seat'
                , index = default_index
                , on_change = refresh_analysis)
                
        if st.session_state.live_draft_active:
            player_assignments = st.session_state.draft_results[0:get_n_starters()].to_dict('list')
            my_players = st.session_state.draft_results[st.session_state.draft_seat].dropna()

        else:
            player_assignments = {team : [] for team in team_names}  
            my_players = []

            
        candidate_evaluation = st.container(height = 625, border = False)

        make_team_display(info_key
                        ,my_players
                        ,get_style_base()
                        )

        with candidate_evaluation:
            if len(my_players) < get_n_starters():

                make_cand_tab(player_assignments
                    ,draft_seat
                    ,get_n_iterations())
                
            else:
                st.write('You have selected all of your players')    


@st.fragment 
def make_auction_tab_own_data():
      """Create a page for an aunction based on user input data
      This is a bit different from drafting because user information about cash needs to be collected

      Args:
        H: H-scoring Agent for performing calculations

      Returns:
        None
      """
      n_drafters = get_n_drafters()
      n_picks = get_n_picks()
      info_key = get_data_key('info')

      left, right = st.columns([0.4,0.6])

      with left:

        cash_per_team = st.number_input(r'How much cash does each team have to pick players?'
                  , key = 'cash_per_team'
                  , min_value = 1
                  , value = 200)

        auction_selections_default = pd.DataFrame([[None] * 3] * n_picks * n_drafters
                                          ,columns = ['Player','Team','Cost'])
        
        player_stats = get_data_from_session_state('player_stats_v2')

        player_category_type = CategoricalDtype(categories=list(player_stats.index), ordered=True)

        auction_selections_default.loc[:'Player'] = \
            auction_selections_default.loc[:'Player'].astype(player_category_type)

        st.caption("""Enter which players have been selected by which teams, and for how much, below""")

        with st.form('manual_auctions'):

            auction_selections = st.data_editor(auction_selections_default
                        ,column_config = 
                        {"Player" : st.column_config.SelectboxColumn(options = list(player_stats.index))
                        ,"Team" : st.column_config.SelectboxColumn(options = get_team_names())
                        ,'Cost' : st.column_config.NumberColumn(min_value = 0
                                                                , step = 1)}
                        , hide_index = True
                        , use_container_width = True
                        )
            
            c1, c2 = st.columns([0.2,0.8])
            
            with c1: 
              submit = st.form_submit_button("Lock in")
            with c2:
              st.warning('Lock in to update candidate display')

            #check that every row is either fully filled out or not filled out 
            na_counts = auction_selections.isna().sum(axis = 1)
            na_counts_3 = na_counts == 3
            na_counts_0 = na_counts == 0

            if not all (na_counts_3 | na_counts_0):
               st.error('''Some rows are partially filled in. If all rows look filled in, this error may have beeen triggered by 
                        hitting the 'Lock in' button while still editing a cell, which prevents that cell from being saved. Hit 
                        'Lock in' again to remove this error''')
               st.stop()


        selection_list = auction_selections['Player'].dropna()

        total_cash = cash_per_team * n_drafters

        amount_spent = auction_selections['Cost'].dropna().sum()

        remaining_cash = total_cash - amount_spent
        
        st.caption(r'\$' + str(remaining_cash) + r' remains out of \$' + str(total_cash) + ' originally available' )

      with right: 
        
        draft_seat = st.selectbox(f'Which team are you?'
            , get_team_names()
            , index = 0)
        
        cash_spent_per_team = auction_selections.dropna().groupby('Team', observed = False)['Cost'].sum()
        cash_remaining_per_team = cash_per_team - cash_spent_per_team
        player_assignments = auction_selections.dropna().groupby('Team', observed = False)['Player'].apply(list)

        for team in get_team_names():
          if not team in cash_remaining_per_team.index:
            cash_remaining_per_team.loc[team] = cash_per_team

          if not team in player_assignments.index:
            player_assignments.loc[team] = []

        my_players = player_assignments[draft_seat]
    
      with left: 

        make_team_display(info_key
                            ,my_players
                            ,get_style_base()
                            )     
      with right:

        if len(my_players) == n_picks:
            st.markdown('Team is complete!')
                    
        else:
            streaming_noise = get_streaming_noise()

            h_ranks = get_default_h_values(info_key = info_key
                                        , omega = get_omega()
                                        , gamma = get_gamma()
                                        , n_picks = n_picks
                                        , n_drafters = n_drafters
                                        , n_iterations = get_n_iterations()
                                        , beth = get_beth()
                                        , scoring_format = get_scoring_format())

            h_ranks_unselected = h_ranks[~h_ranks.index.isin(selection_list)]
            h_defaults_savor = auction_value_adjuster(h_ranks_unselected['H-score']
                                                        , n_picks * n_drafters - len(selection_list)
                                                        , remaining_cash
                                                        , streaming_noise)
            h_defaults_savor = pd.Series(h_defaults_savor.values, index = h_ranks_unselected['Player'])

            h_original_savor = auction_value_adjuster(h_ranks['H-score']
                                                        , n_picks * n_drafters
                                                        , cash_per_team * n_drafters
                                                        , streaming_noise)
            h_original_savor = pd.Series(h_original_savor.values, index = h_ranks_unselected['Player'])

            make_cand_tab(player_assignments.to_dict()
                ,draft_seat
                ,get_n_iterations()
                ,cash_remaining_per_team.to_dict()
                ,h_defaults_savor
                ,h_original_savor
                ,n_drafters * n_picks)


@st.fragment
def make_auction_tab_live_data():
    """Create a page for an aunction based on a live connection e.g. from Yahoo
    This is a bit different from drafting because user information about cash needs to be collected

    Args:
        H: H-scoring Agent for performing calculations

    Returns:
        None
    """
    update_data_and_info()

    n_drafters = get_n_drafters()
    n_picks = get_n_picks()
    info_key = get_data_key('info')

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
            team_names = get_team_names()

            default_index = get_default_draft_seat_index()

            draft_seat = st.selectbox(f'Which drafter are you?'
            , team_names
            , key = 'draft_seat'
            , index = default_index
            , on_change = refresh_analysis)
        
        if not st.session_state.live_draft_active:

            st.write('''Auction has not begun, or just began. There is sometimes a lag of ~1 minute before data can be fetched
                     from Yahoo, so player selections from the beginning of the auction may not be reflected''')
            selection_list = []
            player_assignments = pd.Series({team : [] for team in team_names} ) 
            cash_spent_per_team = pd.Series()
            amount_spent = 0

        else:
            selection_list = st.session_state.draft_results['Player'].dropna()
            player_assignments = st.session_state.draft_results.dropna()  \
                                .groupby('Team', observed = False)['Player'].apply(list)
            
            cash_spent_per_team = st.session_state.draft_results.dropna().groupby('Team', observed = False)['Cost'].sum()
            amount_spent = st.session_state.draft_results['Cost'].dropna().sum()

        #ZR: This is bad
        cash_per_team = 200
        st.session_state.cash_per_team = 200

        cash_remaining_per_team = cash_per_team - cash_spent_per_team

        total_cash = cash_per_team * n_drafters

        remaining_cash = total_cash - amount_spent

        #this is what needs to get updated
        for team in team_names:
            if not team in cash_remaining_per_team.index:
                cash_remaining_per_team.loc[team] = cash_per_team

            if not team in player_assignments.index:
                player_assignments.loc[team] = []

        my_players = player_assignments[draft_seat]
        n_picks = get_n_picks()

        #ZR: I think this could be improved
        h_ranks = get_default_h_values(info_key = info_key
                                    , omega = get_omega()
                                    , gamma = get_gamma()
                                    , n_picks = n_picks
                                    , n_drafters = n_drafters
                                    , n_iterations = get_n_iterations()
                                    , beth = get_beth()
                                    , scoring_format = get_scoring_format()).set_index('Player')

        h_ranks_unselected = h_ranks[~h_ranks.index.isin(selection_list)]
        h_defaults_savor = auction_value_adjuster(h_ranks_unselected['H-score']
                                                        , n_picks * n_drafters - len(selection_list)
                                                        , remaining_cash
                                                        , get_streaming_noise())
                            
        h_defaults_savor = pd.Series(h_defaults_savor.values, index = h_ranks_unselected.index)

        #For when the rank page gets out of synch with the number of drafters and therefore the amount of cash available
        #h_defaults_savor = h_defaults_savor * np.sum([v for k, v in cash_remaining_per_team.items()])/h_defaults_savor.sum()

        h_original_savor = auction_value_adjuster(h_ranks['H-score']
                                                    , n_picks * n_drafters
                                                    , cash_per_team * n_drafters
                                                    , get_streaming_noise())
        
        h_original_savor = pd.Series(h_original_savor.values, index = h_ranks.index)

        candidate_evaluation = st.container(height = 645, border = False)

        make_team_display(info_key
                        ,my_players
                        ,get_style_base()
                        )
            
        with candidate_evaluation:
            if len(my_players) < n_picks:

                make_cand_tab(player_assignments.to_dict()
                    ,draft_seat
                    ,get_n_iterations()
                    ,cash_remaining_per_team.to_dict()
                    ,h_defaults_savor
                    ,h_original_savor
                    ,n_drafters * n_picks)
                
            else:
                st.write('Your team is full')


def clear_draft_board():
  set_draft_position(0,0)

  if 'draft_results' in st.session_state:
    st.session_state.draft_results = None

  remove_selections_df()
  initialize_selections_df()

  st.session_state.live_draft_active = False

def select_player_from_draft_board(p = None):

  if not p:
    p = st.session_state.selected_player

  row, drafter = get_draft_position()

  if (row < get_n_picks()):

    modify_selections_df(row, drafter, p)
    row, drafter = move_forward_one_pick(row, drafter, get_n_drafters())
    set_draft_position(row, drafter)

def undo_selection():

  row, drafter = get_draft_position()

  if not (row == 0) & (drafter == 0):

    row, drafter = move_back_one_pick(row
                                        , drafter
                                        , get_n_drafters())

    modify_selections_df(row, drafter, np.nan)
    set_draft_position(row, drafter)

def refresh_analysis():

    if get_mode() == 'Draft Mode':

        draft_results, error_string = st.session_state.integration.get_draft_results()
    else:

        draft_results, error_string  = st.session_state.integration.get_auction_results()
            
    st.session_state.draft_results = draft_results

    if error_string == 'Success':
        st.session_state.live_draft_active = True
    else:
        if st.session_state.live_draft_active:
            st.error(error_string)
            st.stop()

#this function calls process_player_data and updates the result into session state
#process_player_data will be stored in the cache most of the time, but sometimes n_picks or n_drafters 
#can change if there is a live connection
def update_data_and_info():
    info, key = process_player_data(None
                          ,get_data_key('player_stats_v2')
                          ,get_psi()
                          ,get_chi()
                          ,get_scoring_format()
                          ,get_n_drafters()
                          ,get_n_starters()
                          ,get_params()
                          ,get_selected_categories())
    store_dataset_in_session_state(info, 'info', key)

    H, key = build_h_agent(get_data_key('info')
                    ,get_omega()
                    ,get_gamma()
                    ,get_n_starters()
                    ,get_n_drafters()
                    ,get_beth()
                    ,get_scoring_format()
                    ,get_n_iterations() > 0)
    store_dataset_in_session_state(H, 'H',key)