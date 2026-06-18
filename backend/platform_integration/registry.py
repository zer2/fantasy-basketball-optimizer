"""Resolve a platform label (as sent by the frontend) to its integration class."""

from __future__ import annotations

from backend.platform_integration.base import PlatformIntegration
from backend.platform_integration.integrations.fantrax import FantraxIntegration

# Kept separate from base.py on purpose: this module imports the concrete
# integrations, and base.py is the dependency-free root they import — folding the
# registry into base.py would create a base <- subclass <- base import cycle.

# Keyed by the exact platform string the frontend sends (PLATFORM_OPTIONS).
_INTEGRATION_CLASSES_BY_PLATFORM: dict[str, type[PlatformIntegration]] = {
    'Retrieve from Fantrax': FantraxIntegration,
}


def is_live_platform(platform: str) -> bool:
    """True for platforms with a live integration (i.e. not 'Enter your own data')."""
    return platform in _INTEGRATION_CLASSES_BY_PLATFORM


def get_integration(platform: str) -> PlatformIntegration:
    if platform not in _INTEGRATION_CLASSES_BY_PLATFORM:
        raise ValueError(f'No platform integration registered for {platform!r}')
    return _INTEGRATION_CLASSES_BY_PLATFORM[platform]()
