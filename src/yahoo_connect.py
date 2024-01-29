from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import streamlit as st

import requests
from requests_oauthlib import OAuth1Session

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

def get_yahoo_info(league_id):
    yahoo_client_id = st.secrets["YAHOO_CLIENT_ID"]
    yahoo_client_secret = st.secrets["YAHOO_CLIENT_SECRET"]

    # Replace these values with your Yahoo API credentials
    consumer_key = st.secrets["YAHOO_CLIENT_ID"]
    consumer_secret = st.secrets["YAHOO_CLIENT_SECRET"]
    callback_url = 'https://localhost:8000'
    
    # Step 1: Get a request token
    request_token_url = 'https://api.login.yahoo.com/oauth/v2/get_request_token'
    yahoo = OAuth1Session(consumer_key, client_secret=consumer_secret, callback_uri=callback_url)
    fetch_response = yahoo.fetch_request_token(request_token_url)
    
    resource_owner_key = fetch_response.get('oauth_token')
    resource_owner_secret = fetch_response.get('oauth_token_secret')
    
    # Step 2: Redirect the user to the authorization URL
    authorization_url = 'https://api.login.yahoo.com/oauth/v2/request_auth?oauth_token='
    authorization_url += resource_owner_key
    
    print(f'Please go to {authorization_url} and authorize the application.')
    
    # Step 3: Get the access token
    verifier = input('Enter the verification code: ')
    access_token_url = 'https://api.login.yahoo.com/oauth/v2/get_token'
    yahoo = OAuth1Session(consumer_key,
                         client_secret=consumer_secret,
                         resource_owner_key=resource_owner_key,
                         resource_owner_secret=resource_owner_secret,
                         verifier=verifier)
    
    yahoo_tokens = yahoo.fetch_access_token(access_token_url)
    
    # Now you can use yahoo_tokens['oauth_token'] and yahoo_tokens['oauth_token_secret'] for authenticated requests
    
    # Example: Get user's profile information
    profile_url = 'https://api.login.yahoo.com/openid/v1/userinfo'
    yahoo = OAuth1Session(consumer_key,
                         client_secret=consumer_secret,
                         resource_owner_key=yahoo_tokens['oauth_token'],
                         resource_owner_secret=yahoo_tokens['oauth_token_secret'])
    
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



