"""Shared FastAPI dependencies for the routers."""

from fastapi import HTTPException

from backend.state.session import Session, get_session


def require_session(session_id: str) -> Session:
    """The live session for the path's {session_id}, or a 404 when it is missing or expired.

    A dependency rather than a helper, so a route declares "this needs a live session" in
    its signature instead of repeating the lookup-and-404 block six times over.
    """
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail='Session not found or expired.')
    return session
