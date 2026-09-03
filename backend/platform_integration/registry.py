"""Resolve a platform label (as sent by the frontend) to its integration class."""

from __future__ import annotations

from typing import Optional

from backend.platform_integration.base import PlatformIntegration
from backend.platform_integration.integrations.fantrax import FantraxIntegration
from backend.platform_integration.integrations.yahoo import YahooIntegration
from backend.platform_integration.integrations.espn import ESPNIntegration

# Kept separate from base.py on purpose: this module imports the concrete
# integrations, and base.py is the dependency-free root they import — folding the
# registry into base.py would create a base <- subclass <- base import cycle.

# Keyed by the exact platform string the frontend sends (PLATFORM_OPTIONS).
_INTEGRATION_CLASSES_BY_PLATFORM: dict[str, type[PlatformIntegration]] = {
    'Retrieve from Fantrax': FantraxIntegration,
    'Retrieve from Yahoo':   YahooIntegration,
    'Retrieve from ESPN':    ESPNIntegration,
}


def is_live_platform(platform: str) -> bool:
    """True for platforms with a live integration (i.e. not 'Enter your own data')."""
    return platform in _INTEGRATION_CLASSES_BY_PLATFORM


def get_integration(platform: str, credentials: Optional[dict] = None) -> PlatformIntegration:
    """Construct the integration for a platform. The credentials bag is spread into the
    class's explicit constructor credentials (Yahoo's {'auth_dir': ...} -> auth_dir=...);
    Fantrax takes none, so an empty bag constructs it with no args. A bag key the
    constructor doesn't declare raises TypeError — fail-noisily."""
    if platform not in _INTEGRATION_CLASSES_BY_PLATFORM:
        raise ValueError(f'No platform integration registered for {platform!r}')
    return _INTEGRATION_CLASSES_BY_PLATFORM[platform](**(credentials or {}))
