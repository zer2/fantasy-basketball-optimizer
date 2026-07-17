"""In-memory application state that persists between requests: the session store and the
uploaded-file store.

This is the app's ephemeral-state surface — none of it survives a process restart or is
shared across instances (relevant for horizontal scaling / Cloud Run). Operations *over*
this state (build/patch, pipeline, evaluate, trading) live in backend.services, which may
import state; state must not import services.
"""
