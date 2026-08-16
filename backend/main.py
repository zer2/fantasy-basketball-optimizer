"""
FastAPI application — Fantasy Basketball Optimizer backend (composition root).

Wires the app together: logging, middleware, routers, and the static frontend. Route handlers
live in backend.api.routers.*; application logic in backend.services.*; shared plumbing in
backend.infra.*.
"""

from __future__ import annotations

import os
import socket
import time
import logging

import urllib3.util.connection
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware


# Configure our loggers BEFORE importing the routers. The Yahoo integration (pulled in by the
# platforms router) imports yahoo_oauth, which calls logging.setLoggerClass to install a logger
# class that auto-attaches a handler to every logger created afterwards — duplicating and
# reformatting log lines process-wide. Creating 'fbbo' / 'fbbo.api' here means they already exist
# (clean) when the routers' getLogger calls run; the hijack only affects loggers created after it.
_fbbo_logger = logging.getLogger('fbbo')
_fbbo_logger.setLevel(logging.INFO)
_fbbo_handler = logging.StreamHandler()   # -> stderr, captured by Cloud Run / Cloud Logging
_fbbo_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s'))
_fbbo_logger.addHandler(_fbbo_handler)
_fbbo_logger.propagate = False   # this logger owns its handler; don't also emit via the root logger
logging.getLogger('fbbo.api')    # instantiate now (clean), before the hijack fires below

# Outbound HTTP resolves IPv4 only.
#
# Cloud Run's Direct VPC egress is IPv4-only unless the subnet is dual-stack, and cdn.nba.com
# publishes AAAA records alongside its single A record. When a lookup steers a connection to
# an AAAA address the kernel has no route for it and the request dies immediately with
# OSError [Errno 101] "Network is unreachable" — which is what made headshot fetches fail
# intermittently in production while the very same URL served fine from anywhere with IPv6.
# Nothing distinguished a good request from a bad one but which address family came back.
#
# Every host this app talks to — the NBA CDN, Snowflake, ESPN/Yahoo/Fantrax — is reachable
# over IPv4, so restricting the family costs nothing and removes the failure mode outright.
# This is a property of where the app runs, not of the app, so it belongs here at the
# composition root rather than at any one call site. Remove it if the deployment ever gains
# real IPv6 egress. (urllib3 consults this on every connection, so it covers requests too.)
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

from backend.infra.auth import session_secret_key, session_https_only
from backend.infra.rate_limit import configure_rate_limits
from backend.api.routers import auth, meta, data, sessions, ranking, trade, platforms

# Read the per-client request limits here, at startup: signing in is optional, so these are what
# cap how fast one caller can ask for the expensive work, and malformed configuration should stop
# the process rather than surface on an unlucky request.
configure_rate_limits()

# Undo yahoo_oauth's logging.setLoggerClass hijack (fired during the router imports above) so the
# rest of the process's loggers behave normally. Our 'fbbo'* loggers were created before it.
logging.setLoggerClass(logging.Logger)


app = FastAPI(title='Fantasy Basketball Optimizer', version='1.0.0')

# The app serves its own frontend (same-origin, BASE_URL=''), so normal use never triggers
# CORS. Restrict to a localhost allowlist (override with ALLOWED_ORIGINS, comma-separated, in
# deployment) so other origins can't script the API / credential endpoints.
_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        'ALLOWED_ORIGINS', 'http://localhost:8000,http://127.0.0.1:8000'
    ).split(',')
    if origin.strip()
]

# SessionMiddleware holds the signed login cookie (and Authlib's OAuth state). add_middleware
# prepends, so adding CORS last makes it outermost. Auth itself is enforced per-route via the
# current_user_key dependency, not globally — 'Enter your own data' usage needs no login.
app.add_middleware(
    SessionMiddleware,
    secret_key=session_secret_key(),
    https_only=session_https_only(),
    same_site='lax',
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.middleware('http')
async def _revalidate_app_assets(request: Request, call_next):
    """Force the browser to revalidate the app shell + static assets. StaticFiles sends no
    Cache-Control, so browsers heuristically cache styles.css / dist and serve stale copies —
    'no-cache' means 'revalidate before use' (cheap 304s when unchanged), so edits show up."""
    response = await call_next(request)
    path = request.url.path
    if path == '/' or path.startswith(('/dist', '/styles')):
        response.headers['Cache-Control'] = 'no-cache'
    return response


@app.middleware('http')
async def _server_timing(request: Request, call_next):
    """Append a `total` entry to Server-Timing for every request so each fetch's total server
    time shows in DevTools -> Network -> Timing. Instrumented handlers (e.g. /evaluate) set their
    own phase breakdown on the response first; this preserves it and adds the total."""
    start = time.perf_counter()
    response = await call_next(request)
    total_ms = (time.perf_counter() - start) * 1000
    existing = response.headers.get('Server-Timing')
    total_entry = f'total;dur={total_ms:.1f}'
    response.headers['Server-Timing'] = f'{existing}, {total_entry}' if existing else total_entry
    return response


# Routers — one module per concern; handlers live in backend.api.routers.*
app.include_router(auth.router)
app.include_router(meta.router)
app.include_router(data.router)
app.include_router(sessions.router)
app.include_router(ranking.router)
app.include_router(trade.router)
app.include_router(platforms.router)


# ── Static frontend ────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def serve_index():
    return FileResponse('frontend/app.html')

app.mount('/styles', StaticFiles(directory='frontend/styles'), name='styles')
app.mount('/dist',   StaticFiles(directory='frontend/dist'),   name='dist')
