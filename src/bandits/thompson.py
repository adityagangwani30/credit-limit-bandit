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
    """Beta-Bernoulli Thompson Sampling keyed by contextual risk segment and action."""

    POSITIVE_MAX = REWARD_POSITIVE_MAX
    NEGATIVE_MIN = REWARD_NEGATIVE_MIN

    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0, update_weight: float = 2.0):
        if alpha_prior <= 0 or beta_prior <= 0:
            raise ValueError("alpha_prior and beta_prior must be > 0")
        if update_weight <= 0:
            raise ValueError("update_weight must be > 0")
        self.alpha_prior = float(alpha_prior)
        self.beta_prior = float(beta_prior)
        self.update_weight = float(update_weight)
        self.rng = np.random.default_rng()
        self.reset()

    @staticmethod
    def _risk_segment(context: np.ndarray) -> str:
        values = np.asarray(context, dtype=float)
        if values.size < 10:
            return "unknown"
        if float(np.std(values)) < 1e-6:
            return "unknown"

        safety_score = (
            0.30 * values[7]  # CIBIL
            + 0.20 * values[3]  # delinquency quality
            + 0.15 * values[1]  # payment streak
            + 0.15 * values[2]  # income percentile
            + 0.10 * (1.0 - values[0])  # utilization headroom
            + 0.10 * values[4]  # account age
        )
        if safety_score >= 0.78:
            return "prime"
        if safety_score >= 0.62:
            return "near_prime"
        if safety_score >= 0.45:
            return "subprime"
        return "deep_subprime"

    @staticmethod
    def _safety_score(context: np.ndarray) -> float:
        values = np.asarray(context, dtype=float)
        if values.size < 10:
            return 0.50
        return float(
            0.30 * values[7]
            + 0.20 * values[3]
            + 0.15 * values[1]
            + 0.15 * values[2]
            + 0.10 * (1.0 - values[0])
            + 0.10 * values[4]
        )

    def _eligible_actions(self, context: np.ndarray, actions: list[str]) -> list[str]:
        values = np.asarray(context, dtype=float)
        if values.size >= 10 and float(np.std(values)) < 1e-6:
            return actions

        safety = self._safety_score(context)
        if safety < 0.65:
            allowed = {"keep"}
        elif safety < 0.78:
            allowed = {"keep", "plus_10"}
        elif safety < 0.88:
            allowed = {"keep", "plus_10", "plus_20"}
        else:
            allowed = set(actions)

        eligible = [action for action in actions if action in allowed]
        return eligible or [actions[0]]

    def _initial_params(self, segment: str, action: str) -> list[float]:
        safety_by_segment = {
            "prime": 0.90,
            "near_prime": 0.70,
            "subprime": 0.50,
            "deep_subprime": 0.30,
            "unknown": 0.50,
        }
        safety = safety_by_segment.get(segment, 0.50)
        if segment == "unknown":
            return [self.alpha_prior, self.beta_prior]
        action_prior = {
            "keep": 0.68,
            "plus_10": 0.52 + 0.18 * safety,
            "plus_20": 0.38 + 0.30 * safety,
            "plus_50": 0.20 + 0.45 * safety,
        }.get(str(action), 0.50)
        prior_strength = 3.0
        return [
            self.alpha_prior + action_prior * prior_strength,
            self.beta_prior + (1.0 - action_prior) * prior_strength,
        ]

    def _ensure_pair(self, user_id: str, action: str, context: np.ndarray) -> tuple[float, float]:
        segment = self._risk_segment(context)
        key = (segment, str(action))
        if key not in self.params:
            self.params[key] = self._initial_params(segment, action)
        return self.params[key]

    def select_action(self, context: np.ndarray, user_id: str, actions: list[str]) -> str:
        if not actions:
            raise ValueError("actions must not be empty")
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")

        self.total_selections += 1
        sampled_scores: dict[str, float] = {}
        eligible_actions = self._eligible_actions(context, actions)
        for action in eligible_actions:
            alpha, beta = self._ensure_pair(user_id, action, context)
            sampled_scores[action] = float(self.rng.beta(alpha, beta))

        best_action = max(eligible_actions, key=lambda action: sampled_scores[action])
        self.last_selected_action = best_action
        return best_action

    def update(self, user_id: str, action: str, reward: float, context: np.ndarray) -> None:
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")
        alpha_beta = self._ensure_pair(user_id, action, context)
        reward_norm = self.normalize_reward(reward)
        alpha_beta[0] = min(1000.0, alpha_beta[0] + self.update_weight * reward_norm)
        alpha_beta[1] = min(1000.0, alpha_beta[1] + self.update_weight * (1.0 - reward_norm))

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
            "update_weight": self.update_weight,
            "mean_alpha": float(np.mean(alpha_values)) if alpha_values else self.alpha_prior,
            "mean_beta": float(np.mean(beta_values)) if beta_values else self.beta_prior,
            "last_selected_action": self.last_selected_action,
        }

    def reset(self) -> None:
        self.params: dict[tuple[str, str], list[float]] = {}
        self.total_updates = 0
        self.total_selections = 0
        self.last_selected_action: str | None = None
