"""Sessions for the experiment harnesses, built in whichever process runs them (spawned workers start fresh)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

TESTING_FILES = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTING_FILES.parent


def build_league_session(
    objective: str
    , season: str
    , model_settings: dict
) -> tuple:
    """(test client, session id, session) for the backend tests' league (12 x 13) in this format and season, with these
    model settings over the app's defaults."""
    os.environ.setdefault('SESSION_SECRET_KEY', 'experiment-only-session-secret')
    os.environ.setdefault('RATE_LIMITS_ENABLED', 'false')
    for path in (str(REPO_ROOT), str(TESTING_FILES)):
        if path not in sys.path:
            sys.path.insert(0, path)
    logging.disable(logging.CRITICAL)
    from benchmark_helpers import client, _build_session_request
    from backend.state.session import get_session
    request = _build_session_request(objective=objective)
    request['data_source']['season'] = season
    request['model_settings'].update(model_settings)
    response = client.post('/sessions', json=request)
    if response.status_code != 201:
        raise RuntimeError(f'Session creation failed: {response.text[:300]}')
    session_id = response.json()['session_id']
    return client, session_id, get_session(session_id)
