"""HTTP boundary: FastAPI routers plus their request schemas and shared helpers.

The transport tier. It may import `services`; `services` must never import `api`.
Response/DTO types that a service builds live in the top-level `models` leaf (both
tiers import it); request bodies and router-built responses live in `schemas` here.
"""
