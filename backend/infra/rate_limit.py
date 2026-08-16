"""
Per-client rate limiting for the endpoints that cost real resources.

The app is usable without signing in, so the login wall is no longer what stands between a
script and the expensive work: a session build runs Snowflake queries and the whole pipeline,
and evaluate burns CPU. This is that stand-in — not DDoS protection (a volumetric attack never
reaches this process), but a ceiling on how fast one client can ask for expensive work.

Deliberately NOT limited: headshots (the browser's background preloader legitimately fires
hundreds of them), /config, /seasons, and the static frontend. Limiting those would break normal
use while doing nothing for the cost that matters.

Counting is a sliding window per (policy, client) held in this process. That is the same
assumption the session store already makes — sessions live in memory, so a deployment running
several instances would already be losing sessions between them — and it means the effective
limit is per instance.

Configuration (all optional; every value has a documented default below):
    RATE_LIMITS_ENABLED   - 'false' turns limiting off entirely (test suites do this)
    RATE_LIMIT_BUILD      - 'count/seconds' for session builds, e.g. '40/300'
    RATE_LIMIT_COMPUTE    - 'count/seconds' for evaluate and trade analysis
    RATE_LIMIT_UPLOAD     - 'count/seconds' for projection uploads
    TRUSTED_PROXY_HOPS    - how many proxies append to X-Forwarded-For (see resolve_client_ip)
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from fastapi import HTTPException, Request

BUILD_POLICY   = 'build'
REBUILD_POLICY = 'rebuild'
COMPUTE_POLICY = 'compute'
UPLOAD_POLICY  = 'upload'

# Defaults sized against real usage rather than round numbers.
#
#   build    a fresh session: Snowflake queries plus the whole pipeline. One per page load or
#            data-source change, so 40 per five minutes sits far above any human pattern.
#   rebuild  a PATCH, which re-runs the pipeline from some step. Every parameter change makes
#            one, and someone tuning sliders generates them steadily — hence the much higher
#            ceiling, still well under what a script could ask for.
#   compute  evaluate and trade analysis: cheap per call but bursty, since a full autodraft
#            fires one evaluate per pick back to back. Sized to let two of those through.
#   upload   projection files, rare by nature.
_DEFAULT_POLICIES = {
    BUILD_POLICY:   (40, 300),
    REBUILD_POLICY: (200, 300),
    COMPUTE_POLICY: (900, 300),
    UPLOAD_POLICY:  (20, 300),
}


@dataclass(frozen=True)
class RateLimitPolicy:
    name: str
    max_requests: int
    window_seconds: int


def _parse_policy(
    name: str
    , raw_value: str
) -> RateLimitPolicy:
    """Parses a 'count/seconds' setting. Raises on anything malformed: a typo here would
    otherwise silently fall back to a default and leave the deployment limited differently
    than its configuration says."""
    parts = raw_value.split('/')
    if len(parts) != 2:
        raise RuntimeError(f'RATE_LIMIT_{name.upper()}={raw_value!r} is not in "count/seconds" form.')
    try:
        max_requests, window_seconds = int(parts[0]), int(parts[1])
    except ValueError:
        raise RuntimeError(f'RATE_LIMIT_{name.upper()}={raw_value!r} has non-integer parts.')
    if max_requests < 1 or window_seconds < 1:
        raise RuntimeError(f'RATE_LIMIT_{name.upper()}={raw_value!r} must be positive.')
    return RateLimitPolicy(name, max_requests, window_seconds)


_policies: dict[str, RateLimitPolicy] = {}
_limits_enabled = True


def configure_rate_limits() -> None:
    """Reads the limits from the environment. Called once at startup so malformed configuration
    fails at boot rather than on some unlucky request, and again by tests that change the
    environment."""
    global _limits_enabled
    _limits_enabled = os.environ.get('RATE_LIMITS_ENABLED', 'true').lower() not in ('0', 'false', 'no')
    _policies.clear()
    for name, (default_max, default_window) in _DEFAULT_POLICIES.items():
        raw_value = os.environ.get(f'RATE_LIMIT_{name.upper()}', '').strip()
        _policies[name] = (_parse_policy(name, raw_value) if raw_value
                           else RateLimitPolicy(name, default_max, default_window))


def rate_limits_enabled() -> bool:
    return _limits_enabled


def get_rate_limit_policy(name: str) -> RateLimitPolicy:
    policy = _policies.get(name)
    if policy is None:
        raise RuntimeError(f'Rate limit policy {name!r} was never configured — '
                           'configure_rate_limits() must run at startup.')
    return policy


def resolve_client_ip(
    forwarded_for: Optional[str]
    , direct_ip: Optional[str]
) -> str:
    """The caller's IP, read from X-Forwarded-For when a proxy set it.

    Which entry is the real client depends on how many proxies append to the header, and the
    leftmost entry is whatever the caller sent — trusting it lets one script rotate fake values
    and evade the limit entirely. So we count TRUSTED_PROXY_HOPS entries in from the right
    (default 1, which is Cloud Run appending the address it saw). Confirm the value for a given
    deployment with GET /health/rate-limit, which echoes the chain it received.
    """
    if not forwarded_for:
        return direct_ip or 'unknown'
    chain = [entry.strip() for entry in forwarded_for.split(',') if entry.strip()]
    if not chain:
        return direct_ip or 'unknown'
    hops = int(os.environ.get('TRUSTED_PROXY_HOPS', '1'))
    # With one trusted hop the client is the last entry; with two, the one before it, and so on.
    # A chain shorter than the configured hops means fewer proxies than expected, so the leftmost
    # entry is the closest thing to a real client we have.
    index = max(0, len(chain) - hops)
    return chain[index]


_LOOPBACK_ADDRESSES = frozenset({'127.0.0.1', '::1', 'localhost'})


def request_is_local(request: Request) -> bool:
    """Whether this request came from the machine running the server, which is exempt.

    Limiting localhost would mean the test suites and the screenshot runs — which drive the app
    far harder than any person does — tripping their own protection, and no attacker is on the
    loopback interface. The exemption requires the absence of X-Forwarded-For as well as a
    loopback peer: anything arriving through a proxy is remote traffic no matter what address the
    proxy sits at, and a forwarded header claiming to be 127.0.0.1 is exactly what a bypass
    attempt looks like.
    """
    if request.headers.get('x-forwarded-for'):
        return False
    return bool(request.client) and request.client.host in _LOOPBACK_ADDRESSES


def identify_rate_limit_client(request: Request) -> str:
    """Who this request is counted against: the signed-in account when there is one, otherwise
    the caller's IP. Signing in therefore gives someone their own budget rather than sharing one
    with everyone behind the same address (an office, a campus, a phone carrier's NAT)."""
    user = request.session.get('user')
    if user and 'sub' in user:
        return 'user:' + hashlib.sha256(user['sub'].encode()).hexdigest()[:16]
    return 'ip:' + resolve_client_ip(request.headers.get('x-forwarded-for'),
                                     request.client.host if request.client else None)


class SlidingWindowRateLimiter:
    """Counts recent hits per (policy, client) and reports how long an over-limit caller must
    wait. A sliding window rather than a fixed one so a burst cannot straddle a boundary and
    get through at twice the intended rate."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], deque] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def record_and_measure_wait(
        self
        , policy: RateLimitPolicy
        , client_key: str
        , now: float
    ) -> float:
        """Records this request and returns 0.0 when it is within the limit, otherwise the
        seconds until the window has room. A rejected request is NOT recorded — a client that
        keeps hammering would otherwise push its own wait out indefinitely."""
        with self._lock:
            hits = self._hits.setdefault((policy.name, client_key), deque())
            window_start = now - policy.window_seconds
            while hits and hits[0] <= window_start:
                hits.popleft()
            if len(hits) >= policy.max_requests:
                return hits[0] + policy.window_seconds - now
            hits.append(now)
            self._sweep_idle_clients(now)
            return 0.0

    def _sweep_idle_clients(self, now: float) -> None:
        """Drops clients whose windows have fully expired. Called under the lock, at most once a
        minute: without it the map would grow one entry per address seen, forever."""
        if now - self._last_sweep < 60.0:
            return
        self._last_sweep = now
        for key in [key for key, hits in self._hits.items() if not hits or hits[-1] <= now - 3600]:
            del self._hits[key]

    def describe_client(
        self
        , client_key: str
        , now: float
    ) -> dict:
        """Current usage for one client, for the /health/rate-limit view."""
        with self._lock:
            usage = {}
            for name, policy in _policies.items():
                hits = self._hits.get((name, client_key), deque())
                live = sum(1 for hit in hits if hit > now - policy.window_seconds)
                usage[name] = {
                    'used': live,
                    'limit': policy.max_requests,
                    'window_seconds': policy.window_seconds,
                }
            return usage


_limiter = SlidingWindowRateLimiter()


def get_rate_limiter() -> SlidingWindowRateLimiter:
    return _limiter


def enforce_rate_limit(policy_name: str) -> Callable[[Request], None]:
    """Builds the FastAPI dependency that applies one policy to a route:

        @router.post('/sessions', dependencies=[Depends(enforce_rate_limit(BUILD_POLICY))])

    Over-limit callers get a 429 carrying Retry-After, which is what tells the frontend to say
    'try again in N seconds' rather than showing a raw failure."""
    def check_request_rate(request: Request) -> None:
        if not _limits_enabled or request_is_local(request):
            return
        policy = get_rate_limit_policy(policy_name)
        wait_seconds = _limiter.record_and_measure_wait(
            policy, identify_rate_limit_client(request), time.monotonic())
        if wait_seconds > 0:
            raise HTTPException(
                status_code=429,
                detail=(f'Too many requests. This endpoint allows {policy.max_requests} per '
                        f'{policy.window_seconds // 60} minutes; try again in '
                        f'{max(1, round(wait_seconds))} seconds.'),
                headers={'Retry-After': str(max(1, round(wait_seconds)))},
            )
    return check_request_rate
