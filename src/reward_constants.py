"""Shared reward normalization constants for bandit updates."""

from __future__ import annotations

import math

REWARD_POSITIVE_MAX = 700_000.0
REWARD_NEGATIVE_MIN = -27_000_000.0
REWARD_SCALE_ABS = 27_000_000.0


def normalize_two_branch(reward: float) -> float:
    """
    Map rewards into [0, 1] with explicit sign separation.

    Positive rewards land in (0.5, 1.0], zero maps to 0.5, and
    negative rewards land in [0.0, 0.5).
    """
    reward_value = float(reward)
    if reward_value > 0.0:
        scaled = math.log1p(min(reward_value, REWARD_POSITIVE_MAX)) / math.log1p(
            REWARD_POSITIVE_MAX
        )
        return 0.5 + 0.5 * scaled
    if reward_value == 0.0:
        return 0.5
    scaled = math.log1p(min(abs(reward_value), abs(REWARD_NEGATIVE_MIN))) / math.log1p(
        abs(REWARD_NEGATIVE_MIN)
    )
    return 0.5 - 0.5 * scaled


def normalize_symmetric(reward: float) -> float:
    """Scale rewards into [-1, 1] with log compression for stable bandit means."""
    reward_value = float(reward)
    if reward_value == 0.0:
        return 0.0
    scaled = math.log1p(min(abs(reward_value), REWARD_SCALE_ABS)) / math.log1p(
        REWARD_SCALE_ABS
    )
    return math.copysign(scaled, reward_value)
