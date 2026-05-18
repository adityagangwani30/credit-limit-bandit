"""Reward calculation and delayed feedback utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any


INTERCHANGE_RATE = 0.018


class RewardEngine:
    """Business rules for monthly user-level reward computation."""

    @staticmethod
    def calculate_immediate_fee(amount_spent: float) -> float:
        if float(amount_spent) < 0:
            raise ValueError("amount_spent must be >= 0")
        return float(amount_spent) * INTERCHANGE_RATE

    @staticmethod
    def calculate_default_penalty(outstanding_amount: float, did_default: bool) -> float:
        if float(outstanding_amount) < 0:
            raise ValueError("outstanding_amount must be >= 0")
        return -float(outstanding_amount) if did_default else 0.0

    @classmethod
    def compute_net_reward(
        cls,
        amount_spent: float,
        outstanding_amount: float,
        did_default: bool,
    ) -> float:
        return cls.calculate_immediate_fee(amount_spent) + cls.calculate_default_penalty(
            outstanding_amount,
            did_default,
        )


@dataclass(slots=True)
class _RecordedAction:
    user_id: str
    action_month: int
    action: str
    context: Any


@dataclass(slots=True)
class _RecordedOutcome:
    amount_spent: float
    outstanding_amount: float
    did_default: bool


class RewardBuffer:
    """Tracks delayed rewards so bandit updates happen after the configured lag."""

    def __init__(self, delay_months: int = 3):
        if delay_months < 0:
            raise ValueError("delay_months must be >= 0")
        self.delay_months = delay_months
        self._pending_actions: list[_RecordedAction] = []
        self._outcomes: dict[tuple[str, int], _RecordedOutcome] = {}
        self._released_keys: set[tuple[str, int]] = set()

    def record_action(self, user_id, month, action, context) -> None:
        if int(month) < 1:
            raise ValueError("month must be >= 1")
        self._pending_actions.append(
            _RecordedAction(
                user_id=str(user_id),
                action_month=int(month),
                action=str(action),
                context=deepcopy(context),
            )
        )

    def receive_outcome(self, user_id, month, amount_spent, outstanding_amount, did_default) -> None:
        if int(month) < 1:
            raise ValueError("month must be >= 1")
        if float(outstanding_amount) < 0:
            raise ValueError("outstanding_amount must be >= 0")
        self._outcomes[(str(user_id), int(month))] = _RecordedOutcome(
            amount_spent=float(amount_spent),
            outstanding_amount=float(outstanding_amount),
            did_default=bool(did_default),
        )

    def get_ready_rewards(self, current_month: int) -> list[dict]:
        ready_rewards: list[dict] = []
        still_pending: list[_RecordedAction] = []

        for recorded_action in self._pending_actions:
            action_key = (recorded_action.user_id, recorded_action.action_month)
            reward_month = recorded_action.action_month + self.delay_months

            if current_month >= reward_month and action_key not in self._released_keys:
                outcome = self._outcomes.get(action_key)
                if outcome is None:
                    still_pending.append(recorded_action)
                    continue

                ready_rewards.append(
                    {
                        "user_id": str(recorded_action.user_id),
                        "action_month": recorded_action.action_month,
                        "action": str(recorded_action.action),
                        "context": deepcopy(recorded_action.context),
                        "reward": RewardEngine.compute_net_reward(
                            outcome.amount_spent,
                            outcome.outstanding_amount,
                            outcome.did_default,
                        ),
                        "is_default": outcome.did_default,
                    }
                )
                self._released_keys.add(action_key)
            else:
                still_pending.append(recorded_action)

        self._pending_actions = still_pending
        for reward in ready_rewards:
            assert isinstance(reward["user_id"], str), (
                f"user_id must be str, got {type(reward['user_id'])}"
            )
            assert isinstance(reward["action"], str), (
                f"action must be str, got {type(reward['action'])}"
            )
        return ready_rewards

    def pending_count(self) -> int:
        return len(self._pending_actions)
