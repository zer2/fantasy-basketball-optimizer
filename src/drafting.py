  
import streamlit as st 
from src.helper_functions import listify, move_back_one_pick, move_forward_one_pick, increment_player_stats_version
from src.tabs import *
from src import yahoo_connect, fantrax_connect
from src.helper_functions import move_forward_one_pick, adjust_teams_dict_for_duplicate_names, increment_default_key

def clear_draft_board():
  if 'draft_results' in st.session_state:
    del st.session_state.draft_results 

  if 'live_draft_active' in st.session_state:
    del st.session_state.live_draft_active

  if 'selections_df' in st.session_state:
    del st.session_state.selections_df

def run_autodraft():
  
  if 'selections_df' not in st.session_state:
    st.session_state.selections_df = st.session_state.selections_default

  while (st.session_state.selections_df.columns[st.session_state.drafter] in st.session_state.autodrafters) and (st.session_state.row < st.session_state.n_picks):
    selection_list = listify(st.session_state.selections_df)
    g_scores_unselected = st.session_state.g_scores[~st.session_state.g_scores.index.isin(selection_list)]
    select_player_from_draft_board(g_scores_unselected.index[0])

#run_autodraft_and_increment
def increment_and_reset_draft():
    increment_player_stats_version()
    clear_draft_board()

    if 'autodrafters' in st.session_state:
      run_autodraft()

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
            
        player_assignments = st.session_state.selections_df.to_dict('list')
        print(player_assignments)

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

            make_h_cand_tab(H
                ,st.session_state.g_scores
                ,st.session_state.z_scores
                ,player_assignments
                ,draft_seat
                ,st.session_state.n_iterations
                ,st.session_state.v
                ,5)

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

        
def refresh_analysis():

    player_metadata = st.session_state.player_metadata.copy()

    player_metadata.index = [' '.join(player.split('(')[0].split(' ')[0:2]) for player in player_metadata.index]

    if st.session_state.data_source == 'Retrieve from Yahoo Fantasy':

        yahoo_league_id = st.session_state.yahoo_league_id
        auth_dir = st.session_state.auth_dir

        if st.session_state.mode == 'Draft Mode':

            draft_result = yahoo_connect.get_draft_results(yahoo_league_id
                                                        , auth_dir
                                                        , player_metadata)
                    
        else:

            draft_result = yahoo_connect.get_auction_results(yahoo_league_id
                                                                        , auth_dir
                                                                        , player_metadata)
            
    else:

        if st.session_state.mode == 'Draft Mode':

            draft_result = fantrax_connect.get_draft_results(st.session_state.fantrax_league
                                                        , player_metadata)
                                
        else:

            draft_result = fantrax_connect.get_auction_results(st.session_state.fantrax_league
                                                                        , player_metadata)
                        
    st.session_state.draft_results, st.session_state.live_draft_active = draft_result
   
@st.fragment
def make_drafting_tab_live_data(H):

    st.session_state.player_metadata = st.session_state.player_stats['Position']

    if 'team_names' not in st.session_state:
        st.write('No league info has been passed')
        st.stop()

    if st.session_state.draft_results is None:
        refresh_analysis()

    c1, c2 = st.columns([0.1,0.9])

    with c1:
        st.button('Refresh Analysis', on_click = refresh_analysis)

    with c2:
        draft_seat = st.selectbox(f'Which drafter are you?'
        , st.session_state.team_names
        , key = 'draft_seat'
        , on_change = refresh_analysis)

    if not st.session_state.live_draft_active:

        st.write('Draft has not yet begun')

    else:
        
        player_assignments = st.session_state.draft_results.to_dict('list')

        my_players = st.session_state.draft_results[st.session_state.draft_seat].dropna()
        
        cand_tab, team_tab = st.tabs(["Candidates","Team"])

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

        with cand_tab:

            if len(my_players) < st.session_state.n_picks:

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
def make_auction_tab_live_data(H):

    with st.container():

        st.session_state.player_metadata = st.session_state.player_stats['Position']

        if 'team_names' not in st.session_state:
            st.write('No league info has been passed')
            st.stop()

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
            player_assignments = st.session_state.draft_results.dropna().groupby('Team', observed = False)['Player'].apply(list)

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

                
            with cand_tab:

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