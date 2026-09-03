"""Load the sport parameter definitions from parameters.yaml.

Shared leaf: both the API routers and the service tier read params, so this lives at the
backend top level and imports nothing internal. The parse is cached and reused until
parameters.yaml changes on disk (keyed on its mtime), so a single request doesn't re-read
and re-parse the file the handful of times it asks for params. Each call returns an
independent copy, so a caller can't mutate the shared cache.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

# Anchored on this file so the load works from any working directory.
_PARAMS_PATH = Path(__file__).parents[1] / 'parameters.yaml'
_cache: tuple[float, dict] | None = None   # (mtime, parsed params)


def load_all_params() -> dict:
    """Return the full parameters.yaml as a dict, keyed by sport (cached by file mtime)."""
    global _cache
    mtime = os.path.getmtime(_PARAMS_PATH)
    if _cache is None or _cache[0] != mtime:
        with open(_PARAMS_PATH) as params_file:
            _cache = (mtime, yaml.safe_load(params_file))
    return copy.deepcopy(_cache[1])
