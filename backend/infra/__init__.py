"""Domain-free plumbing: Snowflake access, secret resolution, request timing, OAuth,
rate limiting, and the headshot cache.

The criterion for living here is being app-shaped rather than sport-shaped: these modules
would survive the app changing domains. headshot_cache bends the letter of that rule (it
hardcodes the NBA CDN) but not its structure — a generic fetch-and-cache with one
configured endpoint.
"""
