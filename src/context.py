"""Context vector construction for monthly credit-limit decisions."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


INCOME_PERCENTILE_MAP: dict[str, float] = {
    "low": 0.25,
    "mid": 0.50,
    "high": 0.75,
    "very_high": 1.00,
}

ACTION_SPACE = ["keep", "plus_10", "plus_20", "plus_50"]
ACTION_MULTIPLIERS: dict[str, float] = {
    "keep": 1.0,
    "plus_10": 1.10,
    "plus_20": 1.20,
    "plus_50": 1.50,
}

MAX_CREDIT_LIMIT = 500000.0
MAX_PAYMENT_STREAK = 36.0
MAX_DELINQUENCY_COUNT = 5.0
MAX_ACCOUNT_AGE = 120.0
MAX_TRANSACTION_FREQUENCY = 40.0
MIN_CIBIL_SCORE = 300.0
CIBIL_SCORE_RANGE = 600.0
MAX_HISTORY_MONTHS = 12.0


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


class ContextBuilder:
    """Build normalized context vectors for a user at a monthly decision point."""

    def __init__(self, users_df: pd.DataFrame):
        if users_df.empty:
            raise ValueError("users_df must not be empty")
        if "user_id" not in users_df.columns:
            raise ValueError("users_df must include a user_id column")
        self.users_df = users_df.copy()
        self._users_by_id = self.users_df.set_index("user_id", drop=False)
        self.income_percentiles = self.users_df["income_bucket"].map(INCOME_PERCENTILE_MAP).fillna(0.0)

    def build_context(self, user_id: str, month_history: list[dict]) -> np.ndarray:
        if user_id not in self._users_by_id.index:
            raise KeyError(f"Unknown user_id: {user_id}")

        user = self._users_by_id.loc[user_id]
        current_limit = self._resolve_current_limit(user, month_history)
        current_outstanding = self._resolve_current_outstanding(month_history)

        credit_utilization = _clip01(current_outstanding / current_limit) if current_limit > 0 else 0.0
        payment_streak_norm = _clip01(float(user["payment_streak"]) / MAX_PAYMENT_STREAK)
        income_percentile = _clip01(INCOME_PERCENTILE_MAP.get(str(user["income_bucket"]), 0.0))
        delinquency_score = _clip01(1.0 - float(user["delinquency_count"]) / MAX_DELINQUENCY_COUNT)
        account_age_norm = _clip01(float(user["account_age_months"]) / MAX_ACCOUNT_AGE)
        spending_volatility = self._calculate_spending_volatility(month_history)
        transaction_freq_norm = _clip01(float(user["transaction_frequency"]) / MAX_TRANSACTION_FREQUENCY)
        cibil_norm = _clip01((float(user["cibil_score"]) - MIN_CIBIL_SCORE) / CIBIL_SCORE_RANGE)
        months_of_history_norm = _clip01(min(len(month_history), int(MAX_HISTORY_MONTHS)) / MAX_HISTORY_MONTHS)
        current_limit_norm = _clip01(np.log10(max(current_limit, 1.0)) / np.log10(MAX_CREDIT_LIMIT))

        context = np.array(
            [
                credit_utilization,
                payment_streak_norm,
                income_percentile,
                delinquency_score,
                account_age_norm,
                spending_volatility,
                transaction_freq_norm,
                cibil_norm,
                months_of_history_norm,
                current_limit_norm,
            ],
            dtype=np.float32,
        )
        if not np.isfinite(context).all():
            raise ValueError(f"Context vector for {user_id} contains NaN or Inf values")
        return np.clip(context, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def get_action_space() -> list[str]:
        return ACTION_SPACE.copy()

    @staticmethod
    def apply_action(current_limit: float, action: str) -> float:
        if action not in ACTION_MULTIPLIERS:
            raise ValueError(f"Unsupported action: {action}")
        if not np.isfinite(current_limit) or current_limit <= 0:
            raise ValueError("current_limit must be a positive finite number")
        updated_limit = float(current_limit) * ACTION_MULTIPLIERS[action]
        return round(min(updated_limit, MAX_CREDIT_LIMIT), 2)

    def _resolve_current_limit(self, user: pd.Series, month_history: list[dict]) -> float:
        if not month_history:
            return float(user["initial_credit_limit"])

        latest_record = month_history[-1]
        if "current_credit_limit" in latest_record:
            return float(latest_record["current_credit_limit"])
        if "credit_limit" in latest_record:
            return float(latest_record["credit_limit"])
        return float(user["initial_credit_limit"])

    @staticmethod
    def _resolve_current_outstanding(month_history: list[dict]) -> float:
        if not month_history:
            return 0.0

        latest_record = month_history[-1]
        for key in ("outstanding_amount", "current_outstanding"):
            if key in latest_record:
                return max(float(latest_record[key]), 0.0)
        return 0.0

    @staticmethod
    def _calculate_spending_volatility(month_history: Iterable[dict]) -> float:
        history_list = list(month_history)
        if len(history_list) < 3:
            return 0.0

        recent_spend = np.array([max(float(month.get("amount_spent", 0.0)), 0.0) for month in history_list[-3:]], dtype=float)
        mean_spend = float(np.mean(recent_spend))
        if mean_spend <= 0:
            return 0.0

        volatility = float(np.std(recent_spend) / mean_spend)
        return _clip01(volatility)
