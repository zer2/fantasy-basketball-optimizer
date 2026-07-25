"""Live platform integration endpoints (ESPN / Yahoo / Fantrax).

The {platform} path segment is the exact platform label the frontend sends
(e.g. 'Retrieve from Fantrax'), URL-encoded; it is resolved via the registry.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends, status, Response

from backend.infra.auth import current_user_key
from backend.state.session import get_session
from backend.platform_integration.registry import get_integration
from backend.platform_integration.credential_store import yahoo_auth_dir, store_espn_credentials
from backend.platform_integration.integrations.yahoo import YahooIntegration
from backend.api.platform_helpers import (
    credentials_for, yahoo_app_credentials, resolve_live_integration,
)
from backend.api.schemas import (
    YahooAuthUrlResponse, YahooTokenRequest, EspnCredentialsRequest,
    LeaguesResponse, DivisionsResponse, ConnectRequest, ConnectResponse, DraftStateResponse,
)
from backend.api.errors import fail

router = APIRouter()


@router.get('/platforms/yahoo/auth-url', response_model=YahooAuthUrlResponse)
def yahoo_auth_url_route():
    client_id, _ = yahoo_app_credentials()
    return YahooAuthUrlResponse(auth_url=YahooIntegration.build_auth_url(client_id))


@router.post('/platforms/yahoo/token', status_code=status.HTTP_204_NO_CONTENT)
def yahoo_token_route(req: YahooTokenRequest, user_key: str = Depends(current_user_key)):
    client_id, client_secret = yahoo_app_credentials()
    try:
        YahooIntegration.exchange_auth_code(
            client_id, client_secret, req.auth_code, yahoo_auth_dir(user_key),
        )
    except Exception:
        raise fail(502, 'Yahoo authorization failed.')
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post('/platforms/espn/credentials', status_code=status.HTTP_204_NO_CONTENT)
def espn_credentials_route(req: EspnCredentialsRequest, user_key: str = Depends(current_user_key)):
    # SWID is stored with braces stripped (as the Streamlit integration did).
    swid = req.swid.replace('{', '').replace('}', '')
    store_espn_credentials(user_key, req.s2, swid)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get('/platforms/{platform}/leagues', response_model=LeaguesResponse)
def get_platform_leagues_route(platform: str, user_key: str = Depends(current_user_key)):
    integration = resolve_live_integration(platform, user_key)
    try:
        leagues = integration.list_leagues()
    except Exception:
        raise fail(502, 'Failed to list leagues from the platform.')
    return LeaguesResponse(leagues=leagues)


@router.get('/platforms/{platform}/divisions', response_model=DivisionsResponse)
def get_platform_divisions_route(platform: str, league_id: str):
    integration = resolve_live_integration(platform)
    try:
        divisions = integration.list_divisions(league_id)
    except Exception:
        raise fail(502, 'Failed to list divisions from the platform.')
    return DivisionsResponse(divisions=divisions)


@router.post('/platforms/{platform}/connect', response_model=ConnectResponse)
def connect_platform_route(platform: str, req: ConnectRequest, user_key: str = Depends(current_user_key)):
    integration = resolve_live_integration(platform, user_key)
    try:
        shape = integration.fetch_league_shape(req.league_id, req.division_id)
    except Exception:
        raise fail(502, 'Failed to connect to the platform league.')
    return ConnectResponse(
        team_names      = shape.team_names,
        n_drafters      = shape.n_drafters,
        n_picks         = shape.n_picks,
        available_modes = integration.available_modes,
    )


@router.get('/sessions/{session_id}/draft-state', response_model=DraftStateResponse)
def get_draft_state_route(session_id: str, mode: str, user_key: str = Depends(current_user_key)):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    config = session.platform_config
    if config is None:
        raise HTTPException(status_code=400, detail='Session is not connected to a live platform.')
    # Poll with the requesting user's own credentials, never the key stored on the session.
    integration = get_integration(config.platform, credentials_for(config.platform, user_key))
    # The lookup is built at session creation and rebuilt on data/injured/connect patches, so a
    # connected session always has it here; no defensive rebuild (its absence would be an upstream bug).
    try:
        if mode == 'Auction Mode':
            selections = integration.get_auction_results(config, mode, session.platform_name_lookup)
            if selections is None:
                raise HTTPException(status_code=400, detail=f'{config.platform} does not support auctions.')
        else:
            selections = integration.get_draft_results(config, mode, session.platform_name_lookup)
    except HTTPException:
        raise
    except Exception:
        raise fail(502, 'Failed to fetch the live draft state.')

    # Auction selections carry per-player costs; turn them into per-team remaining cash.
    if selections.costs is None:
        remaining_cash = None
    else:
        cash_per_team = session.current_params['cash_per_team']
        remaining_cash = {
            team: cash_per_team - sum(team_costs)
            for team, team_costs in selections.costs.items()
        }
    return DraftStateResponse(
        player_assignments = selections.player_assignments,
        injured_players    = selections.injured_players,
        status             = selections.status,
        remaining_cash     = remaining_cash,
    )
