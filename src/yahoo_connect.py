from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import streamlit as st

import requests
from requests_oauthlib import OAuth2Session

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

def get_yahoo_info(league_id):
    yahoo_client_id = st.secrets["YAHOO_CLIENT_ID"]
    yahoo_client_secret = st.secrets["YAHOO_CLIENT_SECRET"]

    # Replace these values with your Yahoo API credentials
    client_id = st.secrets["YAHOO_CLIENT_ID"]
    client_secret = st.secrets["YAHOO_CLIENT_SECRET"]
    redirect_uri = 'https://fantasy-basketball-optimizer-yleby8jedyrzhfwoycwuc8.streamlit.app/oath/callback'
    
    authorization_base_url = 'https://api.login.yahoo.com/oauth2/request_auth'
    token_url = 'https://api.login.yahoo.com/oauth2/get_token'

    # Step 1: Obtain authorization URL
    yahoo = OAuth2Session(client_id, redirect_uri=redirect_uri, scope=['openid', 'profile', 'email'])
    authorization_url, state = yahoo.authorization_url(authorization_base_url)

    print(f'Please go to {authorization_url} and authorize the application.')

    # Step 2: Get the authorization response
    redirect_response = input('Paste the full redirect URL here: ')

    # Step 3: Get the access token
    yahoo.fetch_token(token_url, authorization_response=redirect_response, client_secret=client_secret)

    # Now you can use yahoo's session for authenticated requests

    # Example: Get user's profile information
    profile_url = 'https://api.login.yahoo.com/openid/v1/userinfo'
    response = yahoo.get(profile_url)
    user_data = response.json()

    print('User Profile Information:')
    print(user_data)
    
    redirect = 'https://api.login.yahoo.com/oauth2/request_auth?redirect_uri=oob&response_type=code&client_id=' + yahoo_client_id
    st.markdown("check out this [link](" + redirect + ")", unsafe_allow_html = True)
    oauth = OAuth2(yahoo_client_id, yahoo_client_secret)


    league_id = '418.1.' + league_id 
    
    league = gm.to_league(league_id)
  
    rosters = { t['name'] : league.to_team(team_id).roster(21) for team_id, t in league.teams().items()}
  
    roster_dict = {team: [p['name'] for p in roster if p['selected_position'] != 'IL' ] for team, roster in rosters.items()}
    roster_df = pd.DataFrame.from_dict(roster_dict, orient='index')
    roster_df = roster_df.transpose()
  
    il_dict = {team: [p['name'] for p in roster if p['selected_position'] == 'IL' ] for team, roster in rosters.items()}
    all_il_players = [x for v in il_dict.values() for x in v]

    return rosters, all_il_players



