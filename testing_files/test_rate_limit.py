"""Rate limiting: the protection that replaced the login wall.

Signing in is optional now, so these limits are what stand between one script and the endpoints
that cost real work. The tests below pin the three things that decide whether that actually
holds: the window arithmetic, who a request is counted against (getting this wrong either lets a
caller rotate fake addresses past the limit or lumps every visitor into one bucket), and that the
limit reaches the routes at all.
"""

import os
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.infra import rate_limit
from backend.infra.rate_limit import (
    RateLimitPolicy, SlidingWindowRateLimiter, resolve_client_ip, _parse_policy,
    configure_rate_limits, BUILD_POLICY, COMPUTE_POLICY,
)

client = TestClient(app)
# TestClient's default peer is the literal string 'testclient'; this one presents itself the way
# a browser on the same machine does, which is what the local exemption keys on.
local_client = TestClient(app, client=('127.0.0.1', 45123))

# Any non-loopback address turns the local exemption off, so requests carrying this header are
# treated exactly like traffic from the internet.
REMOTE_HEADERS = {'x-forwarded-for': '203.0.113.7'}


@contextmanager
def rate_limiting_turned_on(**policies: str):
    """Turns limiting on (conftest disables it for the suite) with the given policy overrides,
    starting from an empty count, and puts everything back afterwards.

    The environment is saved and restored here rather than via monkeypatch: the limits are read
    into module state by configure_rate_limits(), so the re-read has to happen after the
    environment is back — and monkeypatch's undo runs after the test body, which is too late.
    Getting this wrong leaves limiting enabled for every test that follows, which is exactly how
    it first went wrong."""
    changes = {'RATE_LIMITS_ENABLED': 'true'}
    changes.update({f'RATE_LIMIT_{name.upper()}': value for name, value in policies.items()})
    saved = {name: os.environ.get(name) for name in changes}
    os.environ.update(changes)
    configure_rate_limits()
    rate_limit.get_rate_limiter()._hits.clear()
    try:
        yield
    finally:
        for name, previous in saved.items():
            if previous is None:
                del os.environ[name]
            else:
                os.environ[name] = previous
        configure_rate_limits()
        rate_limit.get_rate_limiter()._hits.clear()


# ── The window itself ──────────────────────────────────────────────────────────────────

def test_requests_are_allowed_up_to_the_limit_then_refused():
    limiter = SlidingWindowRateLimiter()
    policy = RateLimitPolicy('test', max_requests=3, window_seconds=60)

    assert [limiter.record_and_measure_wait(policy, 'client', 1000.0 + n) for n in range(3)] == [0.0, 0.0, 0.0]
    wait = limiter.record_and_measure_wait(policy, 'client', 1003.0)
    assert wait > 0, 'the fourth request inside the window must be refused'


def test_the_window_slides_rather_than_resetting_on_a_boundary():
    """A fixed window would let a caller fire the full allowance either side of the boundary —
    twice the intended rate in an instant. The oldest hit must expire on its own clock."""
    limiter = SlidingWindowRateLimiter()
    policy = RateLimitPolicy('test', max_requests=2, window_seconds=10)

    limiter.record_and_measure_wait(policy, 'client', 100.0)
    limiter.record_and_measure_wait(policy, 'client', 105.0)
    assert limiter.record_and_measure_wait(policy, 'client', 109.0) > 0

    # At 110.5 the first hit has aged out and exactly one slot is free — not two.
    assert limiter.record_and_measure_wait(policy, 'client', 110.5) == 0.0
    assert limiter.record_and_measure_wait(policy, 'client', 110.6) > 0


def test_a_refused_request_is_not_recorded():
    """Otherwise a caller that keeps hammering pushes its own recovery further away every try,
    turning a brief limit into an ever-growing lockout."""
    limiter = SlidingWindowRateLimiter()
    policy = RateLimitPolicy('test', max_requests=1, window_seconds=10)

    limiter.record_and_measure_wait(policy, 'client', 100.0)
    for attempt in range(5):
        limiter.record_and_measure_wait(policy, 'client', 101.0 + attempt)
    # The wait still counts from the single recorded hit at 100.0, not from the last attempt.
    assert limiter.record_and_measure_wait(policy, 'client', 106.0) == pytest.approx(4.0, abs=0.01)


def test_clients_are_counted_separately():
    limiter = SlidingWindowRateLimiter()
    policy = RateLimitPolicy('test', max_requests=1, window_seconds=60)

    assert limiter.record_and_measure_wait(policy, 'ip:1.2.3.4', 100.0) == 0.0
    assert limiter.record_and_measure_wait(policy, 'ip:5.6.7.8', 100.0) == 0.0, \
        'one busy client must not spend another client\'s allowance'


# ── Who the request is counted against ─────────────────────────────────────────────────

def test_client_ip_comes_from_the_trusted_end_of_the_forwarded_chain(monkeypatch):
    """The leftmost entry is whatever the caller sent. Trusting it would let one script rotate
    invented addresses and never be limited, so the address is taken from the proxy end."""
    monkeypatch.setenv('TRUSTED_PROXY_HOPS', '1')
    assert resolve_client_ip('198.51.100.9', None) == '198.51.100.9'
    # A caller claiming its own chain: the entry the proxy appended is the real one.
    assert resolve_client_ip('1.1.1.1, 198.51.100.9', None) == '198.51.100.9'

    monkeypatch.setenv('TRUSTED_PROXY_HOPS', '2')
    assert resolve_client_ip('1.1.1.1, 198.51.100.9, 10.0.0.1', None) == '198.51.100.9'


def test_client_ip_falls_back_to_the_peer_without_a_forwarded_header():
    assert resolve_client_ip(None, '203.0.113.5') == '203.0.113.5'
    assert resolve_client_ip('', '203.0.113.5') == '203.0.113.5'


def test_local_requests_are_exempt_but_forwarded_ones_never_are():
    """Localhost is exempt so the suites and screenshot runs do not trip their own protection —
    but a forwarded header means the request came through a proxy, whatever address it claims."""
    assert local_client.get('/health/rate-limit').json()['exempt_as_local'] is True

    spoofed = local_client.get('/health/rate-limit', headers={'x-forwarded-for': '127.0.0.1'})
    assert spoofed.json()['exempt_as_local'] is False, \
        'a forwarded header claiming loopback must not buy the local exemption'


# ── Configuration ──────────────────────────────────────────────────────────────────────

def test_policy_configuration_is_parsed_and_malformed_values_fail_loudly():
    assert _parse_policy('build', '15/60') == RateLimitPolicy('build', 15, 60)
    for bad_value in ('15', '15/0', 'fifteen/60', '15/60/300', '-1/60'):
        with pytest.raises(RuntimeError):
            _parse_policy('build', bad_value)


def test_environment_overrides_the_defaults():
    with rate_limiting_turned_on(build='7/120'):
        assert rate_limit.get_rate_limit_policy(BUILD_POLICY) == RateLimitPolicy('build', 7, 120)


def test_turning_limits_on_for_a_test_does_not_leak_into_the_next_one():
    """The limits live in module state, so a botched restore leaves every later test in this run
    subject to them — which looks like unrelated 429s in suites that never mention rate limiting."""
    with rate_limiting_turned_on(build='1/300'):
        assert rate_limit.rate_limits_enabled() is True
    assert rate_limit.rate_limits_enabled() is False, 'the suite default must be restored'
    assert rate_limit.get_rate_limit_policy(BUILD_POLICY).max_requests == 40, \
        'and so must the default policy'


# ── The limit actually reaching the routes ─────────────────────────────────────────────

def test_an_expensive_route_refuses_a_remote_caller_over_its_limit():
    """End to end through the app: past the limit the route answers 429 with Retry-After, and it
    does so before doing any of the work (this session body would otherwise 4xx on its contents)."""
    with rate_limiting_turned_on(build='2/300'):
        statuses = [client.post('/sessions', json={}, headers=REMOTE_HEADERS).status_code
                    for _ in range(4)]
        assert statuses[-1] == 429, f'the third call past the limit should be refused, saw {statuses}'
        refused = client.post('/sessions', json={}, headers=REMOTE_HEADERS)
        assert int(refused.headers['Retry-After']) >= 1
        assert 'try again' in refused.json()['detail'].lower()


def test_headshots_are_never_rate_limited():
    """The browser preloads hundreds in the background; limiting them would break normal use
    while doing nothing about the cost that matters."""
    with rate_limiting_turned_on(build='1/300', compute='1/300'):
        for _ in range(5):
            response = client.get('/players/headshots/203999.png', headers=REMOTE_HEADERS)
            assert response.status_code != 429


def test_the_health_view_reports_only_the_caller():
    with rate_limiting_turned_on(compute='5/300'):
        body = client.get('/health/rate-limit', headers=REMOTE_HEADERS).json()
        assert body['counted_as'] == 'ip:203.0.113.7'
        assert body['signed_in'] is False
        assert body['usage'][COMPUTE_POLICY] == {'used': 0, 'limit': 5, 'window_seconds': 300}
