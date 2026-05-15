"""Upper confidence bound bandit for credit-limit actions."""

from __future__ import annotations

import math

import numpy as np

from src.bandits.base import ContextualBandit


class UCBBandit(ContextualBandit):
    """UCB1-style bandit keyed by (user_id, action)."""

    def __init__(self, c: float = 2.0):
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = float(c)
        self.reset()

    def _key(self, user_id: str, action: str) -> tuple[str, str]:
        return (str(user_id), str(action))

    def select_action(self, context: np.ndarray, user_id: str, actions: list[str]) -> str:
        if not actions:
            raise ValueError("actions must not be empty")
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")

        self.total_selections += 1

        for action in actions:
            if self.counts.get(self._key(user_id, action), 0) == 0:
                self.last_selected_action = action
                return action

        total_pulls = sum(self.counts[self._key(user_id, action)] for action in actions)
        total_pulls = max(total_pulls, 1)

        scores: dict[str, float] = {}
        for action in actions:
            key = self._key(user_id, action)
            count = self.counts[key]
            mean_reward = self.rewards[key]
            bonus = self.c * math.sqrt(math.log(total_pulls) / count)
            scores[action] = mean_reward + bonus

        best_action = max(actions, key=lambda action: scores[action])
        self.last_selected_action = best_action
        return best_action

    def update(self, user_id: str, action: str, reward: float, context: np.ndarray) -> None:
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")
        key = self._key(user_id, action)
        current_count = self.counts.get(key, 0)
        current_mean = self.rewards.get(key, 0.0)

        new_count = current_count + 1
        new_mean = current_mean + (float(reward) - current_mean) / new_count

        self.counts[key] = new_count
        self.rewards[key] = new_mean
        self.total_updates += 1

    def get_stats(self) -> dict:
        reward_values = list(self.rewards.values())
        count_values = list(self.counts.values())
        return {
            "algorithm": "ucb",
            "total_updates": self.total_updates,
            "total_selections": self.total_selections,
            "tracked_pairs": len(self.counts),
            "exploration_constant": self.c,
            "mean_pair_reward": float(np.mean(reward_values)) if reward_values else 0.0,
            "mean_pair_count": float(np.mean(count_values)) if count_values else 0.0,
            "last_selected_action": self.last_selected_action,
        }

    def reset(self) -> None:
        self.counts: dict[tuple[str, str], int] = {}
        self.rewards: dict[tuple[str, str], float] = {}
        self.total_updates = 0
        self.total_selections = 0
        self.last_selected_action: str | None = None
