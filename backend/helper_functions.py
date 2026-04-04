"""
Shared utility functions for the backend.

These are small, stateless helpers with no inter-module dependencies that are
useful across more than one backend module.
"""

from __future__ import annotations


def counting_stats(params: dict, categories: list[str]) -> list[str]:
    """Return counting statistics from params that are in the active categories."""
    return [c for c in params['counting-statistics'] if c in categories]


def ratio_stats(params: dict, categories: list[str]) -> list[str]:
    """Return ratio statistics from params that are in the active categories."""
    return [c for c in params['ratio-statistics'] if c in categories]


def extract_last_name(player_full_name: str) -> str:
    """Return the player's last name, stripping any trailing position suffix.

    For example: "Nikola Jokic (C, PF)" → "Jokic", "LeBron James (PF, SF)" → "James",
    "Nikola Jokic" → "Jokic".
    """
    return ' '.join(player_full_name.split(' (')[0].split(' ')[1:])
