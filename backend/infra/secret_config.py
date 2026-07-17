"""Resolve configuration secrets: environment variables first, then .streamlit/secrets.toml.

The single place that knows HOW a secret resolves, so the rule lives in one spot rather than
being re-implemented per consumer (snowflake_connection, auth, and main all route through it).
It lives under infra/ because it reads the environment and filesystem — generic plumbing, not
fantasy-domain logic.

All local secrets (Snowflake, Google OAuth, Yahoo app keys, the session signing key) can live in
one gitignored .streamlit/secrets.toml during local dev, while deployment uses real environment
variables / a secret manager. Keys are flat, top-level TOML entries, e.g.:

    GOOGLE_OAUTH_CLIENT_ID = "1234-abc.apps.googleusercontent.com"
    GOOGLE_OAUTH_CLIENT_SECRET = "..."
    SESSION_SECRET_KEY = "..."
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Optional

# This file is backend/infra/secret_config.py, so the project root is three parents up.
_SECRETS_TOML = Path(__file__).parent.parent.parent / '.streamlit' / 'secrets.toml'


def get_secret(name: str) -> Optional[str]:
    """Return a secret by name: environment variable first, then .streamlit/secrets.toml,
    else None. Env always wins, so deployment can override the local file."""
    value = os.environ.get(name)
    if value:
        return value
    elif _SECRETS_TOML.exists():
        with open(_SECRETS_TOML, 'rb') as secrets_file:
            return tomllib.load(secrets_file).get(name)
    else:
        return None
