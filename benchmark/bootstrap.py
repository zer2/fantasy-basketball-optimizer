# benchmark/bootstrap.py
"""Populate Streamlit session state headlessly so zer2's engine runs outside the app.
Sets runtime state only — never edits src/."""
import os
import yaml
import streamlit as st

from src.helpers.helper_functions import gen_key, store_dataset_in_session_state
from src.math.process_player_data import process_player_data
from benchmark.config import LeagueConfig

def _load_all_params():
    with open('parameters.yaml', 'r') as f:
        return yaml.safe_load(f)

def bootstrap_session(averages, cfg: LeagueConfig):
    os.environ['SPORT'] = cfg.league
    ss = st.session_state
    all_params = _load_all_params()
    params = all_params[cfg.league]

    ss['all_params'] = all_params
    ss['params'] = params
    ss['data_source'] = 'Enter your own data'
    ss['mode'] = 'Draft Mode'
    ss['league'] = cfg.league
    ss['scoring_format'] = cfg.scoring_format
    ss['omega'] = cfg.omega
    ss['gamma'] = cfg.gamma
    ss['psi'] = cfg.psi
    ss['chi'] = cfg.chi
    ss['beth'] = cfg.beth
    ss['n_iterations'] = cfg.n_iterations
    ss['aleph'] = cfg.aleph
    ss['third_round_reversal'] = cfg.third_round_reversal
    ss['selected_categories'] = list(cfg.selected_categories)
    ss['n_picks'] = cfg.n_starters
    ss['n_bench'] = 0
    ss['team_names'] = ['Drafter ' + str(i + 1) for i in range(cfg.n_drafters)]
    ss['styler'] = None
    ss['base'] = 'light'
    ss['data_dictionary'] = {}

    # Position slot counts the getters read as n_<code>. Standard 9-cat single-slot layout.
    position_counts = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1,
                       'G': 1, 'F': 1, 'Util': 2}
    assert sum(position_counts.values()) == cfg.n_starters, \
        f'position slots ({sum(position_counts.values())}) must sum to n_starters ({cfg.n_starters})'
    for code, n in position_counts.items():
        ss['n_' + code] = n

    # Inject projections as player_stats_v2.
    store_dataset_in_session_state(averages, 'player_stats_v2', gen_key())

    # Build info via the engine (weekly_df=None -> uses averages path).
    info, key = process_player_data(
        None, gen_key(), cfg.psi, cfg.chi, cfg.scoring_format,
        cfg.n_drafters, cfg.n_starters, params, list(cfg.selected_categories))
    store_dataset_in_session_state(info, 'info', key)
    return info
