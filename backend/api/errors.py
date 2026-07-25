"""Shared API error helper: log the failure server-side, return a client-safe error."""

from __future__ import annotations

import logging

from fastapi import HTTPException

logger = logging.getLogger('fbbo.api')


def fail(status_code: int, message: str) -> HTTPException:
    """Log the active exception server-side (with traceback) and return a client-safe error.

    Tracebacks / internal details must never be sent in the response body.
    """
    logger.exception(message)
    return HTTPException(status_code=status_code, detail=message)
