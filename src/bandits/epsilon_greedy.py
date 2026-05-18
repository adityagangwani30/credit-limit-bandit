"""Epsilon-greedy bandit for credit-limit actions."""

from __future__ import annotations

import numpy as np

from src.bandits.base import ContextualBandit
from src.reward_constants import REWARD_SCALE_ABS, normalize_symmetric


class EpsilonGreedyBandit(ContextualBandit):
    """User-specific epsilon-greedy policy with multiplicative decay."""

    REWARD_SCALE = REWARD_SCALE_ABS

    def __init__(self, epsilon: float = 0.15, decay: float = 0.995, min_epsilon: float = 0.01):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        if not 0 < decay <= 1:
            raise ValueError("decay must be in (0, 1]")
        if not 0 <= min_epsilon <= 1:
            raise ValueError("min_epsilon must be in [0, 1]")
        if min_epsilon > epsilon:
            raise ValueError("min_epsilon must be <= epsilon")

        self.initial_epsilon = float(epsilon)
        self.decay = float(decay)
        self.min_epsilon = float(min_epsilon)
        self.rng = np.random.default_rng()
        self.reset()

    def _key(self, user_id: str, action: str) -> tuple[str, str]:
        return (str(user_id), str(action))

    def select_action(self, context: np.ndarray, user_id: str, actions: list[str]) -> str:
        if not actions:
            raise ValueError("actions must not be empty")
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")

        self.total_selections += 1
        should_explore = bool(self.rng.random() < self.epsilon)

        if should_explore:
            selected_action = str(self.rng.choice(actions))
        else:
            selected_action = max(
                actions,
                key=lambda action: self.rewards.get(self._key(user_id, action), 0.0),
            )

        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        self.last_selected_action = selected_action
        return selected_action

    def update(self, user_id: str, action: str, reward: float, context: np.ndarray) -> None:
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")
        key = self._key(user_id, action)
        current_count = self.counts.get(key, 0)
        current_mean = self.rewards.get(key, 0.0)
        reward_norm = self._normalize(reward)

        new_count = current_count + 1
        new_mean = current_mean + (reward_norm - current_mean) / new_count

        self.counts[key] = new_count
        self.rewards[key] = new_mean
        self.total_updates += 1

    @staticmethod
    def _normalize(reward: float) -> float:
        return normalize_symmetric(reward)

    def get_stats(self) -> dict:
        reward_values = list(self.rewards.values())
        count_values = list(self.counts.values())
        return {
            "algorithm": "epsilon_greedy",
            "total_updates": self.total_updates,
            "total_selections": self.total_selections,
            "tracked_pairs": len(self.counts),
            "current_epsilon": self.epsilon,
            "min_epsilon": self.min_epsilon,
            "decay": self.decay,
            "mean_pair_reward": float(np.mean(reward_values)) if reward_values else 0.0,
            "mean_pair_count": float(np.mean(count_values)) if count_values else 0.0,
            "last_selected_action": self.last_selected_action,
        }

    def reset(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.rewards: dict[tuple[str, str], float] = {}
        self.epsilon = self.initial_epsilon
        self.total_updates = 0
        self.total_selections = 0
        self.last_selected_action: str | None = None
