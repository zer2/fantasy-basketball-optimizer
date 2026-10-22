# backend/benchmark/bootstrap.py
"""Build the runtime dictionary headlessly so zer2's engine runs natively.
Constructs `info` equivalent to Step 4 of the FastApi build_agent module."""
from backend.services.build_agent import _load_mean_of_variances
from backend.math.process_player_data import process_player_data
from backend.benchmark.config import LeagueConfig, get_params

def bootstrap_session(averages, cfg: LeagueConfig):
    all_params = get_params()
    params = all_params[cfg.league]

    # Build info via the engine natively bypassing Streamlit Session stores.
    info, _ = process_player_data(
        player_stats_v2   = averages,
        weekly_df         = None,
        mean_of_variances = _load_mean_of_variances(cfg.league),
        psi               = cfg.psi,
        chi               = cfg.chi,
        scoring_format    = cfg.scoring_format,
        n_drafters        = cfg.n_drafters,
        n_starters        = cfg.n_starters,
        params            = params,
        categories        = list(cfg.selected_categories),
        sport             = cfg.league
    )
    return info
