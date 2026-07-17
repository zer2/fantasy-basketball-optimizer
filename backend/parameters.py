"""Load the sport parameter definitions from parameters.yaml.

Shared leaf: both the API routers and the service tier read params, so this lives at
the backend top level and imports nothing internal.
"""

from __future__ import annotations

import yaml

_PARAMS_PATH = 'parameters.yaml'


def load_all_params() -> dict:
    """Return the full parameters.yaml as a dict, keyed by sport."""
    with open(_PARAMS_PATH) as params_file:
        return yaml.safe_load(params_file)
