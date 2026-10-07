# benchmark/config.py
from dataclasses import dataclass, field
from functools import lru_cache

# Ratio categories first, then counting — matches process_player_data output ordering
# (calculate_scores_from_coefficients concatenates ratio_statistics + counting_statistics,
#  then reindexes to get_selected_categories()).
NINE_CAT = ['Field Goal %', 'Free Throw %',
            'Threes', 'Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers']

# Default 9-cat standard lineup slots.
DEFAULT_SLOT_COUNTS = {'PG': 1, 'SG': 1, 'SF': 1, 'PF': 1, 'C': 1, 'G': 1, 'F': 1, 'Util': 2}

def get_slot_counts(cfg):
    """Return position-slot mapping appropriate for the league config."""
    return dict(DEFAULT_SLOT_COUNTS) if cfg.n_starters == 9 else {'Util': cfg.n_starters}

# Pre-computed struct signature for the eligibility LRU cache key.
# Constant for any standard 9-cat league; avoids recomputing on every call.
DEFAULT_STRUCT_SIG = tuple(sorted(DEFAULT_SLOT_COUNTS.items()))

@lru_cache(maxsize=1)
def get_params():
    """Load parameters.yaml once per process (lazy, cached)."""
    import yaml
    with open('parameters.yaml', 'r') as f:
        return yaml.safe_load(f)

@dataclass(frozen=True)
class LeagueConfig:
    league: str = 'NBA'
    season: str = '2025-26'
    n_drafters: int = 12
    n_starters: int = 9
    scoring_format: str = 'Head to Head: Each Category'
    selected_categories: list = field(default_factory=lambda: list(NINE_CAT))
    # H-score engine params — verified against parameters.yaml['NBA'].
    # omega/gamma/n_iterations come from the "Moderate punting" level (the app default
    # punting_default), NOT options.*.default. beth/psi/chi/aleph are options.*.default.
    omega: float = 0.7
    gamma: float = 0.25
    beth: float = 3.0
    psi: float = 0.8
    chi: float = 0.6
    n_iterations: int = 30
    aleph: float = 0.2
    third_round_reversal: bool = False

@dataclass(frozen=True)
class ExperimentConfig:
    fields: tuple = ('gscore', 'hscore')
    formats: tuple = ('Head to Head: Each Category', 'Head to Head: Most Categories')
    temperatures: tuple = (0.0, 0.5, 1.0, 2.0)   # T sweep; 0.0 == chalk/deterministic
    n_drafts: int = 24            # drafts per (field, format, T) cell before aggregation
    n_season_sims: int = 500      # evaluator bootstrap seasons per draft
    seed: int = 12345
    # MCTS knobs
    mcts_top_k: int = 15
    mcts_simulations: int = 200
    c_puct: float = 1.4
