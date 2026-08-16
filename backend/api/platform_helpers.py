"""Platform-integration resolution helpers for the API layer.

These raise HTTPExceptions (401/400/502) — mapping platform + credential failures to status
codes — so they belong in the transport tier, shared by the sessions and platforms routers.
The HTTP-free session orchestration lives in services.session_management.
"""

from __future__ import annotations

from typing import Optional

from fastapi import HTTPException

from backend.api.errors import fail
from backend.infra.secret_config import get_secret
from backend.platform_integration.registry import get_integration, is_live_platform
from backend.platform_integration.base import PlatformConfig
from backend.platform_integration.credential_store import (
    yahoo_auth_dir, has_yahoo_credentials,
    get_espn_credentials, has_espn_credentials,
)


def credentials_for(platform: str, client_id: Optional[str]) -> Optional[dict]:
    """Build the credentials an integration needs, or None. Yahoo needs a persisted OAuth
    token directory keyed by client_id; an unauthenticated client_id fails noisily."""
    if platform == 'Retrieve from Yahoo' and client_id:
        if not has_yahoo_credentials(client_id):
            raise HTTPException(status_code=401, detail='Not authenticated with Yahoo; complete the auth flow first.')
        return {'auth_dir': yahoo_auth_dir(client_id)}
    if platform == 'Retrieve from ESPN' and client_id:
        if not has_espn_credentials(client_id):
            raise HTTPException(status_code=401, detail='Not authenticated with ESPN; save your s2/SWID cookies first.')
        return get_espn_credentials(client_id)   # {'s2': ..., 'swid': ...} -> spread into ESPNIntegration
    return None


def yahoo_app_credentials() -> tuple[str, str]:
    """The Yahoo *app* (developer) client id/secret, from the environment."""
    client_id = get_secret('YAHOO_CLIENT_ID')
    client_secret = get_secret('YAHOO_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise HTTPException(status_code=500, detail='YAHOO_CLIENT_ID / YAHOO_CLIENT_SECRET are not configured.')
    return client_id, client_secret


def resolve_live_integration(platform: str, client_id: Optional[str] = None):
    """Resolve a live integration, configured with credentials when the platform needs them
    (Yahoo). client_id=None gives an unauthenticated instance — fine for Fantrax and for Yahoo
    calls that don't hit the API (list_divisions)."""
    if not is_live_platform(platform):
        raise HTTPException(status_code=400, detail=f'No live integration for platform {platform!r}.')
    return get_integration(platform, credentials_for(platform, client_id))


def resolve_platform_config(
    platform: str
    , platform_config_request
    , user_key: Optional[str]
) -> Optional[PlatformConfig]:
    """For a live platform, fetch the league shape and build a PlatformConfig to store on the
    session. Returns None for 'Enter your own data'. Shared by the create and patch routes."""
    if not is_live_platform(platform):
        return None
    if user_key is None:
        raise HTTPException(status_code=401, detail='Sign in to connect a live platform.')
    if platform_config_request is None:
        raise HTTPException(
            status_code=400,
            detail=f'platform_config is required for platform {platform!r}.',
        )
    # Credentials are keyed by the authenticated user, never by anything in the request.
    integration = get_integration(platform, credentials_for(platform, user_key))
    try:
        shape = integration.fetch_league_shape(
            platform_config_request.league_id,
            platform_config_request.division_id,
        )
    except Exception:
        raise fail(502, 'Could not reach the fantasy platform for this league.')
    return PlatformConfig(
        platform           = platform,
        league_id          = platform_config_request.league_id,
        division_id        = platform_config_request.division_id,
        teams_dict         = shape.teams_dict,
        player_name_column = integration.player_name_column,
    )
