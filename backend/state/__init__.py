"""In-memory application state that persists between requests: the session store and the
uploaded-file store.

This is the app's per-process state surface — sessions do not survive a restart, while
the upload store persists its files to disk and rehydrates them; nothing here is shared
across instances (relevant for horizontal scaling / Cloud Run). Operations *over*
this state (build/patch, pipeline, evaluate, trading) live in backend.services, which may
import state; state must not import services.
"""
