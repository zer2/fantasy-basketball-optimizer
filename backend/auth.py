"""
Google (OIDC) login and per-user identity.

A single Authlib OAuth client talks to Google's OIDC endpoints. After the redirect
dance the authenticated user's identity ({sub, email}) is stored in the signed session
cookie (Starlette SessionMiddleware). Per-platform credentials are keyed by a hash of
the Google `sub`, derived server-side here — never taken from client input — so each
Google account gets its own isolated credential store.

Required environment variables (set before running the server):
    GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET  - from Google Cloud Console
    SESSION_SECRET_KEY                                  - random secret signing the cookie
Optional:
    AUTH_ALLOWED_EMAILS  - comma-separated allowlist; if unset, any verified Google account
    OAUTH_REDIRECT_URI   - overrides the auto-derived <base>/auth/callback
    SESSION_HTTPS_ONLY   - 'true' marks the cookie Secure (set this in production / behind HTTPS)
"""

from __future__ import annotations

import hashlib
import os
from typing import Optional

from authlib.integrations.starlette_client import OAuth
from fastapi import HTTPException, Request

from backend.secret_config import get_secret

_GOOGLE_METADATA_URL = 'https://accounts.google.com/.well-known/openid-configuration'

# The OAuth client. client_id/secret are read at import (env or .streamlit/secrets.toml); if
# unset, import still succeeds and the failure surfaces noisily only on an actual login.
oauth = OAuth()
oauth.register(
    name='google',
    client_id=get_secret('GOOGLE_OAUTH_CLIENT_ID'),
    client_secret=get_secret('GOOGLE_OAUTH_CLIENT_SECRET'),
    server_metadata_url=_GOOGLE_METADATA_URL,
    client_kwargs={'scope': 'openid email profile'},
)

def session_secret_key() -> str:
    """The key that signs the login session cookie. Required, with no fallback: a per-process
    random key would silently break logins across restarts / multiple instances (a process
    that didn't sign the OAuth-state cookie can't verify it). Fail loudly if it's missing."""
    secret = get_secret('SESSION_SECRET_KEY')
    if not secret:
        raise RuntimeError(
            'SESSION_SECRET_KEY is not set (environment or .streamlit/secrets.toml). It signs '
            'the login session cookie and must be a stable secret value. Generate one with: '
            'python -c "import secrets; print(secrets.token_hex(32))"'
        )
    return secret


def session_https_only() -> bool:
    # Plain deployment flag, not a secret — read straight from the environment.
    return os.environ.get('SESSION_HTTPS_ONLY', '').lower() in ('1', 'true', 'yes')


def _allowed_emails() -> Optional[set[str]]:
    # A (non-secret) allowlist of emails; environment config, not a secret.
    raw = os.environ.get('AUTH_ALLOWED_EMAILS', '').strip()
    if not raw:
        return None   # no allowlist -> any verified Google account may sign in
    return {email.strip().lower() for email in raw.split(',') if email.strip()}


def email_is_allowed(email: str) -> bool:
    allow = _allowed_emails()
    return allow is None or email.lower() in allow


def get_current_user(request: Request) -> dict:
    """FastAPI dependency: the logged-in user ({'sub', 'email'}), or 401 if not signed in."""
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail='Authentication required.')
    return user


def _key_for(user: dict) -> str:
    return hashlib.sha256(user['sub'].encode()).hexdigest()[:32]


def current_user_key(request: Request) -> str:
    """Filesystem-safe per-user key (a hash of the Google sub) for the credential store.
    Derived from the session server-side, so a client can never name another user's key.
    401s when not signed in — use on endpoints that always require a logged-in user."""
    return _key_for(get_current_user(request))


def current_user_key_optional(request: Request) -> Optional[str]:
    """Like current_user_key but returns None instead of 401 when not signed in. Used by
    endpoints that work for 'Enter your own data' without auth but require it for a live
    platform (the caller enforces that)."""
    user = request.session.get('user')
    return _key_for(user) if user else None
