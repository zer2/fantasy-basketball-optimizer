"""Probabilistic opportunity cost drop penalty calculator (c_p) for fantasy basketball.

Quantifies the expected Rest-of-Season (ROS) opportunity cost of dropping a bench hold
based on replacement baseline, weekly VORP, waiver wire snipe risk, survival horizon decay,
and current matchup urgency.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from fantasy_streamer.config import (
    DEFAULT_EFFECTIVE_GAMES_PER_WEEK,
    DEFAULT_MATCHUP_SIGMA,
    DEFAULT_RETENTION_RATE,
    DEFAULT_TOP_K_REPLACEMENT,
    DEFAULT_URGENCY_MAX_CLAMP,
)


@dataclass
class DropPenaltyResult:
    """Detailed breakdown of a player's drop penalty components."""

    player_id: int
    player_name: str
    mu_p: float
    mu_rep: float
    weekly_vorp: float
    p_snipe: float
    h_eff: float
    m_urgency: float
    c_p: float

    def __repr__(self) -> str:
        return (
            f"DropPenaltyResult(player='{self.player_name}' (ID={self.player_id}), "
            f"FPPG={self.mu_p:.1f}, VORP={self.weekly_vorp:.2f}, "
            f"P_snipe={self.p_snipe:.3f}, H_eff={self.h_eff:.2f}, "
            f"M_urgency={self.m_urgency:.2f} -> c_p={self.c_p:.2f})"
        )


def calculate_weekly_vorp(
    mu_p: float,
    mu_rep: float,
    effective_games: float = DEFAULT_EFFECTIVE_GAMES_PER_WEEK,
) -> float:
    """Compute Weekly Value Over Replacement Player (VORP).

    Weekly VORP = g_eff * max(0, mu_p - mu_rep)
    where g_eff ~= 3.0 usable starts/week accounting for busy nights.
    """
    return effective_games * max(0.0, mu_p - mu_rep)


def calculate_waiver_snipe_probability(mu_p: float, mu_rep: float) -> float:
    """Compute logistic probability that dropped player is claimed/sniped off waivers.

    P_snipe(mu_p) = 1 / (1 + exp(-0.9 * (mu_p - mu_rep - 2.5)))
    """
    z = -0.9 * (mu_p - mu_rep - 2.5)
    # Numerical stability guard against overflow
    if z > 700:
        return 0.0
    if z < -700:
        return 1.0
    return 1.0 / (1.0 + math.exp(z))


def calculate_effective_survival_horizon(
    weeks_remaining: int,
    gamma: float = DEFAULT_RETENTION_RATE,
) -> float:
    """Compute geometric decay over remaining weeks W_rem with weekly retention gamma.

    H_eff = gamma * (1 - gamma^W_rem) / (1 - gamma)
    """
    if weeks_remaining <= 0:
        return 0.0
    if math.isclose(gamma, 1.0):
        return float(weeks_remaining)
    return (gamma * (1.0 - math.pow(gamma, weeks_remaining))) / (1.0 - gamma)


def calculate_matchup_urgency(
    delta_s: float,
    sigma_s: float = DEFAULT_MATCHUP_SIGMA,
    max_clamp: float = DEFAULT_URGENCY_MAX_CLAMP,
) -> float:
    """Compute clamped exponential matchup urgency scalar based on point differential delta_S.

    M_urgency = min(max_clamp, exp(delta_S^2 / (2 * sigma_S^2)))
    """
    if sigma_s <= 0:
        return 1.0
    exponent = (delta_s**2) / (2.0 * (sigma_s**2))
    # Bound exponent to prevent float overflow
    if exponent > 50.0:
        return max_clamp
    urgency = math.exp(exponent)
    return min(max_clamp, urgency)


def calculate_drop_penalty(
    player_id: int,
    player_name: str,
    mu_p: float,
    mu_rep: float,
    weeks_remaining: int = 12,
    delta_s: float = 0.0,
    effective_games: float = DEFAULT_EFFECTIVE_GAMES_PER_WEEK,
    gamma: float = DEFAULT_RETENTION_RATE,
    sigma_s: float = DEFAULT_MATCHUP_SIGMA,
    max_urgency_clamp: float = DEFAULT_URGENCY_MAX_CLAMP,
) -> DropPenaltyResult:
    """Compute the full 4-component opportunity cost penalty c_p.

    c_p = Weekly_VORP * P_snipe * H_eff * M_urgency
    """
    vorp = calculate_weekly_vorp(mu_p, mu_rep, effective_games)
    p_snipe = calculate_waiver_snipe_probability(mu_p, mu_rep)
    h_eff = calculate_effective_survival_horizon(weeks_remaining, gamma)
    m_urgency = calculate_matchup_urgency(delta_s, sigma_s, max_urgency_clamp)

    c_p = vorp * p_snipe * h_eff * m_urgency

    return DropPenaltyResult(
        player_id=player_id,
        player_name=player_name,
        mu_p=mu_p,
        mu_rep=mu_rep,
        weekly_vorp=vorp,
        p_snipe=p_snipe,
        h_eff=h_eff,
        m_urgency=m_urgency,
        c_p=c_p,
    )


class DropPenaltyCalculator:
    """Stateful wrapper for batch evaluating drop penalties across a roster."""

    def __init__(
        self,
        mu_rep: float = 22.5,
        weeks_remaining: int = 12,
        delta_s: float = 0.0,
        effective_games: float = DEFAULT_EFFECTIVE_GAMES_PER_WEEK,
        gamma: float = DEFAULT_RETENTION_RATE,
        sigma_s: float = DEFAULT_MATCHUP_SIGMA,
        max_urgency_clamp: float = DEFAULT_URGENCY_MAX_CLAMP,
        replacement_baseline: Optional[float] = None,
    ):
        if replacement_baseline is not None:
            mu_rep = replacement_baseline
        self.mu_rep = mu_rep
        self.weeks_remaining = weeks_remaining
        self.delta_s = delta_s
        self.effective_games = effective_games
        self.gamma = gamma
        self.sigma_s = sigma_s
        self.max_urgency_clamp = max_urgency_clamp

    def evaluate_player(
        self, player_id: int, player_name: str, mu_p: float
    ) -> DropPenaltyResult:
        return calculate_drop_penalty(
            player_id=player_id,
            player_name=player_name,
            mu_p=mu_p,
            mu_rep=self.mu_rep,
            weeks_remaining=self.weeks_remaining,
            delta_s=self.delta_s,
            effective_games=self.effective_games,
            gamma=self.gamma,
            sigma_s=self.sigma_s,
            max_urgency_clamp=self.max_urgency_clamp,
        )

    def calculate_penalty(
        self,
        fppg: float,
        games_per_week: Optional[float] = None,
        player_id: int = 0,
        player_name: str = "",
    ) -> float:
        """Convenience method returning the scalar penalty value c_p."""
        eff_games = games_per_week if games_per_week is not None else self.effective_games
        res = calculate_drop_penalty(
            player_id=player_id,
            player_name=player_name,
            mu_p=fppg,
            mu_rep=self.mu_rep,
            weeks_remaining=self.weeks_remaining,
            delta_s=self.delta_s,
            effective_games=eff_games,
            gamma=self.gamma,
            sigma_s=self.sigma_s,
            max_urgency_clamp=self.max_urgency_clamp,
        )
        return res.c_p
