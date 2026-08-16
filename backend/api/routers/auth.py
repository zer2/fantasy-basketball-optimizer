"""Google (OIDC) login routes."""

from __future__ import annotations

import os
import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response, RedirectResponse

from backend.infra.auth import oauth, email_is_allowed
from backend.api.errors import fail

router = APIRouter()
logger = logging.getLogger('fbbo.api')


@router.get('/auth/login')
async def auth_login_route(request: Request):
    """Kick off the Google sign-in redirect."""
    redirect_uri = os.environ.get('OAUTH_REDIRECT_URI') or str(request.url_for('auth_callback_route'))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get('/auth/callback', name='auth_callback_route')
async def auth_callback_route(request: Request):
    """Google redirects back here with a code; exchange it, verify, and start a session."""
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception:
        raise fail(401, 'Google sign-in failed.')
    userinfo = token.get('userinfo')
    if not userinfo or not userinfo.get('email_verified'):
        raise HTTPException(status_code=401, detail='Google account has no verified email.')
    email = userinfo['email']
    if not email_is_allowed(email):
        logger.warning('login denied (not allowlisted): %s', email)
        raise HTTPException(status_code=403, detail='This account is not permitted to sign in.')
    name = userinfo.get('given_name') or userinfo.get('name') or email
    request.session['user'] = {
        'sub': userinfo['sub'], 'email': email, 'name': name,
        'picture': userinfo.get('picture'),   # optional; Google provides it when available
    }
    logger.info('login: %s (%s)', email, name)
    return RedirectResponse(url='/')


@router.post('/auth/logout', status_code=status.HTTP_204_NO_CONTENT)
async def auth_logout_route(request: Request):
    request.session.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get('/auth/me')
async def auth_me_route(request: Request):
    """Lightweight auth check for the frontend; 401 when not signed in."""
    # A valid session always carries name (set at login); a session missing it is stale /
    # malformed, so treat it as unauthenticated rather than silently degrading to the email.
    user = request.session.get('user')
    if not user or 'name' not in user:
        raise HTTPException(status_code=401, detail='Not authenticated.')
    return {'email': user['email'], 'name': user['name'], 'picture': user.get('picture')}
