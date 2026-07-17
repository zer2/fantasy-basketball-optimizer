"""
Per-request server-side timing, surfaced to the browser as a `Server-Timing`
response header (visible in DevTools -> Network -> the request -> Timing tab).

A route handler calls begin_timing() at the start; instrumented code wraps phases
in `with record_phase(name):`; the handler reads server_timing_header() and sets it
on the response. A request-total entry is added by the middleware in main.py.

Phase timings live in a ContextVar so instrumented functions (e.g. run_evaluate)
don't need a collector threaded through their signatures. A sync route handler and
the code it calls run in a single context, so set / record / read all see the same
dict. When begin_timing() was never called (tests, the trade routes, direct calls),
record_phase() is a harmless no-op.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_phase_durations_ms: ContextVar[Optional[dict[str, float]]] = ContextVar(
    'server_timing_phase_durations_ms', default=None,
)


def begin_timing() -> None:
    """Start a fresh phase collector for the current request."""
    _phase_durations_ms.set({})


@contextmanager
def record_phase(name: str) -> Iterator[None]:
    """Time the wrapped block and accumulate it under `name` (summed if it repeats).
    A no-op when begin_timing() was not called for this request."""
    start = time.perf_counter()
    try:
        yield
    finally:
        durations = _phase_durations_ms.get()
        if durations is not None:
            durations[name] = durations.get(name, 0.0) + (time.perf_counter() - start) * 1000


def server_timing_header() -> str:
    """Format the recorded phases as a Server-Timing header value (empty string if none)."""
    durations = _phase_durations_ms.get()
    if not durations:
        return ''
    return ', '.join(f'{name};dur={ms:.1f}' for name, ms in durations.items())
