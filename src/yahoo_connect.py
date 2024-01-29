from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa
import streamlit as st

def nav_to(url):
    nav_script = """
        <meta http-equiv="refresh" content="0; url='%s'">
    """ % (url)
    st.write(nav_script, unsafe_allow_html=True)

def get_yahoo_info(league_id):
  yahoo_client_id = st.secrets["YAHOO_CLIENT_ID"]
  yahoo_client_secret = st.secrets["YAHOO_CLIENT_SECRET"]
    
  uri = 'https://api.login.yahoo.com/oauth2/code?client_id=' + yahoo_client_id
  st.markdown("check out this [link](uri)", unsafe_allow_html = True)
  oauth = OAuth2(yahoo_client_id, yahoo_client_secret)


  #convert league ID to the right format, or grab the first team id if none is listed
  if league_idea is None:
    gm = yfa.Game(oauth, 'nba')
    league_id = gm.league_ids()[0]
  else:
    league_id = '418.1.' + league_id 
    
  league = gm.to_league(league_id)
  
  rosters = { t['name'] : league.to_team(team_id).roster(21) for team_id, t in league.teams().items()}
  
  roster_dict = {team: [p['name'] for p in roster if p['selected_position'] != 'IL' ] for team, roster in rosters.items()}
  roster_df = pd.DataFrame.from_dict(roster_dict, orient='index')
  roster_df = roster_df.transpose()
  
  il_dict = {team: [p['name'] for p in roster if p['selected_position'] == 'IL' ] for team, roster in rosters.items()}
  all_il_players = [x for v in il_dict.values() for x in v]

  return rosters, all_il_players
