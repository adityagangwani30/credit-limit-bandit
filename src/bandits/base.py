"""Shared interface for contextual bandit policies."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ContextualBandit(ABC):
    """Abstract interface for user-aware bandit policies."""

    @abstractmethod
    def select_action(
        self, context: np.ndarray, user_id: str, actions: list[str]
    ) -> str:
        """Choose an action for the provided user and context."""

    @abstractmethod
    def update(
        self, user_id: str, action: str, reward: float, context: np.ndarray
    ) -> None:
        """Update internal state from observed delayed reward feedback."""

    @abstractmethod
    def get_stats(self) -> dict:
        """Return a logging-friendly snapshot of the policy state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the policy to its initial state."""
