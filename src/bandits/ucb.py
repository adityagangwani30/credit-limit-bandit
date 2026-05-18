"""Upper confidence bound bandit for credit-limit actions."""

from __future__ import annotations

import math

import numpy as np

from src.bandits.base import ContextualBandit
from src.reward_constants import REWARD_SCALE_ABS, normalize_symmetric


class UCBBandit(ContextualBandit):
    """UCB1-style bandit keyed by contextual risk segment and action."""

    REWARD_SCALE = REWARD_SCALE_ABS

    def __init__(self, c: float = 2.0):
        if c < 0:
            raise ValueError("c must be >= 0")
        self.c = float(c)
        self.reset()

    @staticmethod
    def _risk_segment(context: np.ndarray) -> str:
        values = np.asarray(context, dtype=float)
        if values.size < 10:
            return "unknown"
        if float(np.std(values)) < 1e-6:
            return "unknown"

        safety_score = (
            0.30 * values[7]
            + 0.20 * values[3]
            + 0.15 * values[1]
            + 0.15 * values[2]
            + 0.10 * (1.0 - values[0])
            + 0.10 * values[4]
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

    def _key(self, context: np.ndarray, action: str) -> tuple[str, str]:
        return (self._risk_segment(context), str(action))

    def select_action(
        self, context: np.ndarray, user_id: str, actions: list[str]
    ) -> str:
        if not actions:
            raise ValueError("actions must not be empty")
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")

        self.total_selections += 1
        eligible_actions = self._eligible_actions(context, actions)
        unseen = [
            action
            for action in eligible_actions
            if self.counts.get(self._key(context, action), 0) == 0
        ]
        if len(unseen) == len(eligible_actions):
            self.last_selected_action = eligible_actions[0]
            return eligible_actions[0]

        total_pulls = sum(
            self.counts.get(self._key(context, action), 0)
            for action in eligible_actions
        )
        total_pulls = max(total_pulls, 1)

        scores: dict[str, float] = {}
        for action in eligible_actions:
            key = self._key(context, action)
            count = self.counts.get(key, 0)
            if count == 0:
                scores[action] = float("inf")
            else:
                mean_reward = self.rewards[key]
                bonus = self.c * math.sqrt(math.log(total_pulls) / count)
                scores[action] = mean_reward + bonus

        best_action = max(eligible_actions, key=lambda action: scores[action])
        self.last_selected_action = best_action
        return best_action

    def update(
        self, user_id: str, action: str, reward: float, context: np.ndarray
    ) -> None:
        if context is None or not np.isfinite(np.asarray(context, dtype=float)).all():
            raise ValueError("context must contain only finite values")
        key = self._key(context, action)
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
