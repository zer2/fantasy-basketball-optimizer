from yahoo_oauth import OAuth2
import yahoo_fantasy_api as yfa

oauth = OAuth2('consumer_key', 'consumer_secret')

gm = yfa.Game(oauth, 'nba')
league_id = gm.league_ids(year=2023)[0]
league = gm.to_league(league_id)

rosters = { t['name'] : league.to_team(team_id).roster(21) for team_id, t in league.teams().items()}

roster_dict = {team: [p['name'] for p in roster if p['selected_position'] != 'IL' ] for team, roster in rosters.items()}
roster_df = pd.DataFrame.from_dict(roster_dict, orient='index')
roster_df = roster_df.transpose()

il_dict = {team: [p['name'] for p in roster if p['selected_position'] == 'IL' ] for team, roster in rosters.items()}
all_il_players = [x for v in il_dict.values() for x in v]
