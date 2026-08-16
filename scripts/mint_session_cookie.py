#!/usr/bin/env python
"""Mint a signed Starlette session cookie for a stub user, so the Playwright screenshot
script can load the app headlessly despite the Google-login gate.

It signs with the SAME SESSION_SECRET_KEY the app uses (via backend.auth), so the running
app accepts the cookie exactly as if the user had logged in. Local dev tooling only.

Run from the repo root:  python scripts/mint_session_cookie.py
Prints:  {"name": "session", "value": "<signed cookie>"}
"""
import json
import os
import sys
from base64 import b64encode

import itsdangerous

# Run-from-anywhere: put the repo root (parent of scripts/) on the path so `backend` imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.infra.auth import session_secret_key

# Matches the shape written by the OAuth callback (main.py: sub/email/name/picture).
# /auth/me requires 'name' to be present, else it treats the session as stale (401).
SESSION = {
    'user': {
        'sub':     'screenshot-bot',
        'email':   'screenshots@localhost',
        'name':    'Screenshot Bot',
        'picture': None,
    }
}

# Starlette's SessionMiddleware serialization: base64(json) signed by a TimestampSigner.
signer = itsdangerous.TimestampSigner(session_secret_key())
data   = b64encode(json.dumps(SESSION).encode('utf-8'))
value  = signer.sign(data).decode('utf-8')

print(json.dumps({'name': 'session', 'value': value}))
