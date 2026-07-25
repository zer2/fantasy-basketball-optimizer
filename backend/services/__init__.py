"""Application/service tier: the logic the API layer calls into.

Modules here take a Session (or a request) and orchestrate the domain work — the
session store + lifecycle, the projection pipeline, evaluation, and trading. The
rule that pins the layering: `api` may import `services`; `services` must never
import `api`.
"""
