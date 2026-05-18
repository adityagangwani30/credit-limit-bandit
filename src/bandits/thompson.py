"""Thompson sampling policy for credit-limit actions."""

from __future__ import annotations

import numpy as np

from src.bandits.base import ContextualBandit
from src.reward_constants import (
    REWARD_NEGATIVE_MIN,
    REWARD_POSITIVE_MAX,
    normalize_two_branch,
)


class ThompsonSampling(ContextualBandit):
    """Beta-Bernoulli Thompson Sampling keyed by (user_id, action)."""

    POSITIVE_MAX = REWARD_POSITIVE_MAX
    NEGATIVE_MIN = REWARD_NEGATIVE_MIN

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior and beta_prior must be > 0")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.rng = np.random.default_rng()
        self.reset()

    def _ensure_pair(self, user_id: str, action: str) -> tuple[float, float]:
        key = (str(user_id), str(action))
        if key not in self.params:
            self.params[key] = [self.alpha_prior, self.beta_prior]
        return self.params[key]

    def select_action(self, context: np.ndarray, user_id: str, actions: list[str]) -> str:
        if not actions:
            raise ValueError("actions must not be empty")
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")

        self.total_selections += 1
        sampled_scores: dict[str, float] = {}
        for action in actions:
            alpha, beta = self._ensure_pair(user_id, action)
            sampled_scores[action] = float(self.rng.beta(alpha, beta))

        best_action = max(actions, key=lambda action: sampled_scores[action])
        self.last_selected_action = best_action
        return best_action

    def update(self, user_id: str, action: str, reward: float, context: np.ndarray) -> None:
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")
        alpha_beta = self._ensure_pair(user_id, action)
        reward_norm = self.normalize_reward(reward)
        alpha_beta[0] = min(1000.0, alpha_beta[0] + reward_norm)
        alpha_beta[1] = min(1000.0, alpha_beta[1] + (1.0 - reward_norm))

        self.total_updates += 1

    @staticmethod
    def normalize_reward(reward: float) -> float:
        return normalize_two_branch(reward)

    def get_stats(self) -> dict:
        alpha_values = [params[0] for params in self.params.values()]
        beta_values = [params[1] for params in self.params.values()]
        return {
            "algorithm": "thompson_sampling",
            "total_updates": self.total_updates,
            "total_selections": self.total_selections,
            "tracked_pairs": len(self.params),
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "mean_alpha": float(np.mean(alpha_values)) if alpha_values else self.alpha_prior,
            "mean_beta": float(np.mean(beta_values)) if beta_values else self.beta_prior,
            "last_selected_action": self.last_selected_action,
        }

    def reset(self) -> None:
        self.params: dict[tuple[str, str], list[float]] = {}
        self.total_updates = 0
        self.total_selections = 0
        self.last_selected_action: str | None = None
