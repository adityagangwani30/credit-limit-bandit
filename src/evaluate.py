"""Evaluation metrics and portfolio analysis utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


VALID_COHORT_COLUMNS = {"risk_tier", "income_bucket", "account_age_bucket"}
ACTION_MULTIPLIERS = {
    "keep": 1.0,
    "plus_10": 1.10,
    "plus_20": 1.20,
    "plus_50": 1.50,
}


def _load_users_frame() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_users.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic user file not found: {data_path}")
    return pd.read_csv(data_path)


def _with_user_features(results_df: pd.DataFrame) -> pd.DataFrame:
    if {"risk_tier", "income_bucket", "account_age_months"}.issubset(results_df.columns):
        enriched = results_df.copy()
    else:
        users_df = _load_users_frame()[
            ["user_id", "risk_tier", "income_bucket", "account_age_months", "initial_credit_limit"]
        ]
        enriched = results_df.merge(users_df, on="user_id", how="left")

    enriched["account_age_bucket"] = pd.cut(
        enriched["account_age_months"],
        bins=[-np.inf, 12, 60, np.inf],
        labels=["young", "mid", "old"],
        right=False,
    ).astype(str)
    enriched["limit_increase_pct"] = (
        enriched["action_taken"].map(ACTION_MULTIPLIERS).fillna(1.0) - 1.0
    ) * 100.0
    return enriched


def _monthly_rewards(frame: pd.DataFrame, reward_column: str = "reward_received") -> pd.Series:
    return frame.groupby("month")[reward_column].sum().sort_index()


def compute_regret(oracle_df, policy_df) -> pd.DataFrame:
    """
    Compute cumulative regret against the oracle month by month.
    
    Regret = Oracle cumulative reward - Policy cumulative reward.
    Both must use RAW INR rewards (not normalized).
    
    Returns dataframe with columns:
        month, oracle_cumulative, policy_cumulative, 
        regret_cumulative, regret_pct
    
    regret_pct = regret_cumulative / oracle_cumulative * 100
    (only meaningful when oracle_cumulative > 0)
    """
    oracle_by_month = (oracle_df.groupby("month")["reward_received"]
                       .sum().cumsum().reset_index()
                       .rename(columns={"reward_received": "oracle_cumulative"}))
    
    policy_by_month = (policy_df.groupby("month")["reward_received"]
                       .sum().cumsum().reset_index()
                       .rename(columns={"reward_received": "policy_cumulative"}))
    
    merged = oracle_by_month.merge(policy_by_month, on="month")
    merged["regret_cumulative"] = (merged["oracle_cumulative"] 
                                    - merged["policy_cumulative"])
    merged["regret_pct"] = (merged["regret_cumulative"] 
                             / merged["oracle_cumulative"] * 100).clip(0, 100)
    return merged[["month", "oracle_cumulative", "policy_cumulative", 
                   "regret_cumulative", "regret_pct"]]


def cohort_analysis(results_df, groupby: list[str]) -> pd.DataFrame:
    """Aggregate outcomes by user cohorts."""

    invalid_columns = sorted(set(groupby) - VALID_COHORT_COLUMNS)
    if invalid_columns:
        raise ValueError(f"Unsupported cohort columns: {invalid_columns}")
    if not groupby:
        raise ValueError("groupby must contain at least one cohort column")

    enriched = _with_user_features(results_df)
    cohort_df = (
        enriched.groupby(groupby, dropna=False)
        .agg(
            total_reward=("reward_received", "sum"),
            avg_reward_per_user=("reward_received", lambda s: s.sum() / max(s.index.nunique(), 1)),
            default_rate=("did_default", "mean"),
            avg_limit_increase_pct=("limit_increase_pct", "mean"),
            n_users=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values("total_reward", ascending=False)
    )
    cohort_df["avg_reward_per_user"] = cohort_df["total_reward"] / cohort_df["n_users"].clip(lower=1)
    return cohort_df


def compute_exploration_ratio(results_df) -> float:
    """Return the percentage of non-keep actions and attach action breakdown metadata."""

    action_counts = results_df["action_taken"].value_counts(normalize=True)
    exploration_ratio = float((results_df["action_taken"] != "keep").mean() * 100.0)
    compute_exploration_ratio.breakdown = {
        "plus_10_pct": float(action_counts.get("plus_10", 0.0) * 100.0),
        "plus_20_pct": float(action_counts.get("plus_20", 0.0) * 100.0),
        "plus_50_pct": float(action_counts.get("plus_50", 0.0) * 100.0),
    }
    return exploration_ratio


compute_exploration_ratio.breakdown = {"plus_10_pct": 0.0, "plus_20_pct": 0.0, "plus_50_pct": 0.0}


def shock_recovery_analysis(pre_shock_df, post_shock_df, n_months_post: int = 6) -> dict:
    """Measure reward and default behavior before and after a shock period."""

    pre_default_rate = float(pre_shock_df["did_default"].mean())
    post_default_by_month = (
        post_shock_df.groupby("month")["did_default"].mean().sort_index().head(n_months_post)
    )
    post_reward_by_month = (
        post_shock_df.groupby("month")["reward_received"].sum().sort_index().head(n_months_post)
    )

    pre_reward_baseline = float(pre_shock_df.groupby("month")["reward_received"].sum().mean())
    pre_reward_baseline = pre_reward_baseline if pre_reward_baseline != 0 else 1.0

    max_default_spike_pct = float(
        max(((post_default_by_month - pre_default_rate) * 100.0).max(), 0.0) if not post_default_by_month.empty else 0.0
    )

    months_to_recovery = n_months_post
    for month, monthly_reward in post_reward_by_month.items():
        if monthly_reward >= 0.99 * pre_reward_baseline:
            months_to_recovery = int(month - post_shock_df["month"].min() + 1)
            break

    reward_recovery_pct = float(post_reward_by_month.iloc[-1] / pre_reward_baseline * 100.0) if not post_reward_by_month.empty else 0.0
    return {
        "months_to_recovery": int(months_to_recovery),
        "max_default_spike_pct": max_default_spike_pct,
        "reward_recovery_pct": reward_recovery_pct,
    }


def _convergence_month(results_df: pd.DataFrame) -> int:
    monthly_rewards = _monthly_rewards(results_df)
    rolling_rewards = monthly_rewards.rolling(window=3, min_periods=3).mean()
    rolling_improvement_pct = rolling_rewards.pct_change().replace([np.inf, -np.inf], np.nan)

    for month, improvement in rolling_improvement_pct.dropna().items():
        if improvement < 0.01:
            return int(month)
    return int(monthly_rewards.index.max()) if not monthly_rewards.empty else 0


def policy_comparison_table(all_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create a cross-policy comparison table."""

    if "oracle" not in all_results:
        raise ValueError("all_results must include an 'oracle' entry")
    if "static_baseline" not in all_results:
        raise ValueError("all_results must include a 'static_baseline' entry")

    oracle_total_reward = float(all_results["oracle"]["reward_received"].sum())
    static_total_reward = float(all_results["static_baseline"]["reward_received"].sum())

    rows: list[dict] = []
    for policy_name, frame in all_results.items():
        total_reward = float(frame["reward_received"].sum())
        default_rate_pct = float(frame["did_default"].mean() * 100.0)
        exploration_ratio_pct = compute_exploration_ratio(frame)
        regret_vs_oracle_pct = float(
            ((oracle_total_reward - total_reward) / oracle_total_reward) * 100.0 if oracle_total_reward else 0.0
        )
        revenue_lift_vs_static_pct = float(
            ((total_reward - static_total_reward) / static_total_reward) * 100.0 if static_total_reward else 0.0
        )

        rows.append(
            {
                "policy": policy_name,
                "total_revenue_inr": total_reward,
                "default_rate_pct": default_rate_pct,
                "regret_vs_oracle_pct": regret_vs_oracle_pct,
                "exploration_ratio_pct": exploration_ratio_pct,
                "convergence_month": _convergence_month(frame),
                "revenue_lift_vs_static_pct": revenue_lift_vs_static_pct,
            }
        )

    return pd.DataFrame(rows).sort_values("total_revenue_inr", ascending=False).reset_index(drop=True)
