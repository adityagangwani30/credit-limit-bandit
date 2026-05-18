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


def _total_reward(frame: pd.DataFrame) -> float:
    reward_column = "reward_received" if "reward_received" in frame.columns else "reward"
    return float(frame[reward_column].sum())


def compute_regret(oracle_df, policy_df, label: str = "policy") -> dict:
    """
    Compute total regret against a chosen oracle.

    Returns:
        {
            "label": str,
            "oracle_total": float,
            "policy_total": float,
            "regret": float,
            "regret_pct": float,
        }
    """
    oracle_total = _total_reward(oracle_df)
    policy_total = _total_reward(policy_df)
    regret = oracle_total - policy_total
    regret_pct = (regret / oracle_total * 100.0) if oracle_total > 0 else 0.0
    return {
        "label": label,
        "oracle_total": oracle_total,
        "policy_total": policy_total,
        "regret": regret,
        "regret_pct": round(regret_pct, 2),
    }


def compute_cumulative_regret(oracle_df, policy_df) -> pd.DataFrame:
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
    if "month" not in oracle_df.columns or "month" not in policy_df.columns:
        raise ValueError("Cumulative regret requires month-level results with a 'month' column")

    reward_column_oracle = "reward_received" if "reward_received" in oracle_df.columns else "reward"
    reward_column_policy = "reward_received" if "reward_received" in policy_df.columns else "reward"

    oracle_by_month = (oracle_df.groupby("month")[reward_column_oracle]
                       .sum().cumsum().reset_index()
                       .rename(columns={reward_column_oracle: "oracle_cumulative"}))
    
    policy_by_month = (policy_df.groupby("month")[reward_column_policy]
                       .sum().cumsum().reset_index()
                       .rename(columns={reward_column_policy: "policy_cumulative"}))
    
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

    oracle_key = "oracle_practical" if "oracle_practical" in all_results else "oracle"
    if oracle_key not in all_results and "oracle_theoretical" in all_results:
        oracle_key = "oracle_theoretical"
    if oracle_key not in all_results:
        raise ValueError("all_results must include an oracle entry")
    if "static_baseline" not in all_results:
        raise ValueError("all_results must include a 'static_baseline' entry")

    oracle_total_reward = _total_reward(all_results[oracle_key])
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


# ============================================================================
# NEW METRICS: Convergence, Exploration, Cold Start, Shock Recovery, Regret %
# ============================================================================


def compute_convergence_month(results_df: pd.DataFrame, policy: str) -> int:
    """
    Find the first month where the bandit's 3-month rolling 
    average reward improvement drops below 1%.
    
    Logic:
    - Compute monthly total reward for the policy
    - Compute rolling 3-month mean
    - Find first month where (mean[t] - mean[t-1]) / mean[t-1] < 0.01
    - That is the convergence month
    - If never converges: return n_months (12)
    
    Args:
        results_df: Full results dataframe (all policies)
        policy: Policy name (e.g., "thompson_sampling", "ucb", "epsilon_greedy")
    
    Returns:
        Convergence month (int, 1-12)
    """
    policy_df = results_df[results_df["policy"] == policy].copy()
    
    if policy_df.empty:
        return 12
    
    monthly_rewards = _monthly_rewards(policy_df)
    rolling_rewards = monthly_rewards.rolling(window=3, min_periods=3).mean()
    rolling_improvement_pct = rolling_rewards.pct_change().replace([np.inf, -np.inf], np.nan)
    
    for month, improvement in rolling_improvement_pct.dropna().items():
        if improvement < 0.01:
            return int(month)
    
    return int(monthly_rewards.index.max()) if not monthly_rewards.empty else 12


def compute_exploration_ratio_dict(results_df: pd.DataFrame, policy: str) -> dict:
    """
    Returns breakdown of how often each action was chosen.
    
    exploration_ratio = % of decisions that were NOT "keep"
    
    Args:
        results_df: Full results dataframe (all policies)
        policy: Policy name (e.g., "thompson_sampling", "ucb", "epsilon_greedy")
    
    Returns:
        {
          "keep_pct": float,
          "plus_10_pct": float,
          "plus_20_pct": float,
          "plus_50_pct": float,
          "exploration_ratio": float   # = 100 - keep_pct
        }
    """
    policy_df = results_df[results_df["policy"] == policy].copy()
    
    if policy_df.empty:
        return {
            "keep_pct": 0.0,
            "plus_10_pct": 0.0,
            "plus_20_pct": 0.0,
            "plus_50_pct": 0.0,
            "exploration_ratio": 0.0,
        }
    
    action_counts = policy_df["action_taken"].value_counts()
    total_actions = len(policy_df)
    
    keep_pct = float((action_counts.get("keep", 0) / total_actions) * 100.0)
    plus_10_pct = float((action_counts.get("plus_10", 0) / total_actions) * 100.0)
    plus_20_pct = float((action_counts.get("plus_20", 0) / total_actions) * 100.0)
    plus_50_pct = float((action_counts.get("plus_50", 0) / total_actions) * 100.0)
    exploration_ratio = 100.0 - keep_pct
    
    return {
        "keep_pct": keep_pct,
        "plus_10_pct": plus_10_pct,
        "plus_20_pct": plus_20_pct,
        "plus_50_pct": plus_50_pct,
        "exploration_ratio": exploration_ratio,
    }


def compute_cold_start_performance(
    results_df: pd.DataFrame, 
    users_df: pd.DataFrame, 
    policy: str,
    cold_start_months: int = 3
) -> dict:
    """
    Cold start = user's first cold_start_months months (no history).
    Compare reward and default rate in months 1-3 vs months 4-12.
    
    Args:
        results_df: Full results dataframe (all policies)
        users_df: User information (with account_age_months)
        policy: Policy name
        cold_start_months: Number of months to consider "cold start" (default 3)
    
    Returns:
        {
          "cold_start_avg_reward": float,    # months 1-3
          "warm_avg_reward": float,          # months 4-12
          "cold_start_default_rate": float,
          "warm_default_rate": float,
          "reward_improvement_pct": float    # (warm - cold) / abs(cold) * 100
        }
    """
    policy_df = results_df[results_df["policy"] == policy].copy()
    
    if policy_df.empty:
        return {
            "cold_start_avg_reward": 0.0,
            "warm_avg_reward": 0.0,
            "cold_start_default_rate": 0.0,
            "warm_default_rate": 0.0,
            "reward_improvement_pct": 0.0,
        }
    
    # Split by cold start vs warm period
    cold_start_df = policy_df[policy_df["month"] <= cold_start_months]
    warm_df = policy_df[policy_df["month"] > cold_start_months]
    
    cold_start_avg_reward = float(cold_start_df["reward_received"].mean()) if not cold_start_df.empty else 0.0
    warm_avg_reward = float(warm_df["reward_received"].mean()) if not warm_df.empty else 0.0
    
    cold_start_default_rate = float(cold_start_df["did_default"].mean() * 100.0) if not cold_start_df.empty else 0.0
    warm_default_rate = float(warm_df["did_default"].mean() * 100.0) if not warm_df.empty else 0.0
    
    # Calculate improvement % (avoid division by zero)
    if cold_start_avg_reward != 0:
        reward_improvement_pct = ((warm_avg_reward - cold_start_avg_reward) / abs(cold_start_avg_reward)) * 100.0
    else:
        reward_improvement_pct = 0.0
    
    return {
        "cold_start_avg_reward": cold_start_avg_reward,
        "warm_avg_reward": warm_avg_reward,
        "cold_start_default_rate": cold_start_default_rate,
        "warm_default_rate": warm_default_rate,
        "reward_improvement_pct": reward_improvement_pct,
    }


def compute_shock_recovery(
    results_df: pd.DataFrame, 
    policy: str,
    shock_month: int = 6
) -> dict:
    """
    Economic shock happens at month 6 — default rates double.
    Recovery = months after shock until default rate returns to 
    within 10% of pre-shock average.
    
    Logic:
    - pre_shock_default_rate = mean default rate months 1-5
    - For each month after shock: check if default_rate <= 
      pre_shock_default_rate * 1.10
    - recovery_month = first month this condition is true
    - months_to_recovery = recovery_month - shock_month
    
    Args:
        results_df: Full results dataframe (all policies)
        policy: Policy name
        shock_month: Month when shock occurs (default 6)
    
    Returns:
        {
          "pre_shock_default_rate": float,
          "peak_default_rate": float,        # worst month after shock
          "peak_default_month": int,
          "recovery_month": int,             # None if never recovers
          "months_to_recovery": int,         # None if never recovers
          "max_spike_pct": float             # (peak - pre) / pre * 100
        }
    """
    policy_df = results_df[results_df["policy"] == policy].copy()
    
    if policy_df.empty:
        return {
            "pre_shock_default_rate": 0.0,
            "peak_default_rate": 0.0,
            "peak_default_month": None,
            "recovery_month": None,
            "months_to_recovery": None,
            "max_spike_pct": 0.0,
        }
    
    # Pre-shock average (months 1 to shock_month-1)
    pre_shock_df = policy_df[policy_df["month"] < shock_month]
    pre_shock_default_rate = float(pre_shock_df["did_default"].mean()) if not pre_shock_df.empty else 0.0
    
    # Post-shock analysis
    post_shock_df = policy_df[policy_df["month"] >= shock_month]
    if post_shock_df.empty:
        return {
            "pre_shock_default_rate": pre_shock_default_rate,
            "peak_default_rate": 0.0,
            "peak_default_month": None,
            "recovery_month": None,
            "months_to_recovery": None,
            "max_spike_pct": 0.0,
        }
    
    post_default_by_month = post_shock_df.groupby("month")["did_default"].mean().sort_index()
    
    # Peak default rate
    peak_default_rate = float(post_default_by_month.max())
    peak_default_month = int(post_default_by_month.idxmax())
    
    # Max spike %
    max_spike_pct = float(((peak_default_rate - pre_shock_default_rate) / max(pre_shock_default_rate, 0.001)) * 100.0)
    
    # Recovery month: first month where default rate <= pre_shock * 1.10
    recovery_threshold = pre_shock_default_rate * 1.10
    recovery_month = None
    months_to_recovery = None
    
    for month, default_rate in post_default_by_month.items():
        if default_rate <= recovery_threshold:
            recovery_month = int(month)
            months_to_recovery = int(month - shock_month)
            break
    
    return {
        "pre_shock_default_rate": pre_shock_default_rate,
        "peak_default_rate": peak_default_rate,
        "peak_default_month": peak_default_month,
        "recovery_month": recovery_month,
        "months_to_recovery": months_to_recovery,
        "max_spike_pct": max_spike_pct,
    }


def build_full_metrics_table(
    results_df: pd.DataFrame, 
    users_df: pd.DataFrame, 
    oracle_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Builds the COMPLETE metrics table for all policies.
    Columns:
      policy | total_revenue_inr | revenue_lift_pct | 
      default_rate_pct | regret_vs_oracle_pct | 
      convergence_month | exploration_pct | 
      cold_start_reward | shock_recovery_months
    
    Only compute exploration_pct and convergence_month for 
    thompson_sampling, ucb, epsilon_greedy (not static or oracle).
    Mark static_baseline and oracle as "N/A" for bandit-specific metrics.
    
    Args:
        results_df: Full results dataframe (all policies)
        users_df: User information
        oracle_df: Oracle results (if None, extracted from results_df)
    
    Returns:
        DataFrame with one row per policy
    """
    if oracle_df is None:
        oracle_df = results_df[results_df["policy"] == "oracle_practical"]
        if oracle_df.empty:
            oracle_df = results_df[results_df["policy"] == "oracle"]
        if oracle_df.empty:
            oracle_df = results_df[results_df["policy"] == "oracle_theoretical"]
    
    if oracle_df.empty:
        raise ValueError("No oracle results found")
    
    oracle_total_reward = _total_reward(oracle_df)
    oracle_total_reward = oracle_total_reward if oracle_total_reward != 0 else 1.0
    
    # Get unique policies
    policies = sorted(results_df["policy"].unique())
    
    rows = []
    for policy in policies:
        policy_df = results_df[results_df["policy"] == policy]
        
        if policy_df.empty:
            continue
        
        total_revenue_inr = float(policy_df["reward_received"].sum())
        default_rate_pct = float(policy_df["did_default"].mean() * 100.0)
        regret_vs_oracle_pct = float(((oracle_total_reward - total_revenue_inr) / oracle_total_reward) * 100.0)
        if policy == "oracle_theoretical":
            regret_vs_oracle_pct = np.nan
        
        # Bandit-specific metrics (skip for static_baseline and oracle)
        if policy in ["oracle", "oracle_theoretical", "oracle_practical", "static_baseline"]:
            convergence_month = "N/A"
            exploration_pct = "N/A"
        else:
            convergence_month = compute_convergence_month(results_df, policy)
            exp_dict = compute_exploration_ratio_dict(results_df, policy)
            exploration_pct = exp_dict["exploration_ratio"]
        
        # Cold start and shock recovery apply to all
        cold_start = compute_cold_start_performance(results_df, users_df, policy)
        cold_start_reward = cold_start["cold_start_avg_reward"]
        
        shock = compute_shock_recovery(results_df, policy)
        shock_recovery_months = shock["months_to_recovery"]
        
        rows.append({
            "policy": policy,
            "total_revenue_inr": total_revenue_inr,
            "default_rate_pct": default_rate_pct,
            "regret_vs_oracle_pct": regret_vs_oracle_pct,
            "convergence_month": convergence_month,
            "exploration_pct": exploration_pct,
            "cold_start_reward": cold_start_reward,
            "shock_recovery_months": shock_recovery_months,
        })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv("data/simulation_results.csv")
    users_df = pd.read_csv("data/synthetic_users.csv")
    oracle_df = df[df["policy"] == "oracle_practical"]
    if oracle_df.empty:
        oracle_df = df[df["policy"] == "oracle"]
    if oracle_df.empty:
        oracle_df = df[df["policy"] == "oracle_theoretical"]
    
    print("=" * 70)
    print("FULL METRICS REPORT")
    print("=" * 70)
    
    for policy in ["thompson_sampling", "ucb", "epsilon_greedy"]:
        policy_data = df[df["policy"] == policy]
        if policy_data.empty:
            print(f"\n--- {policy.upper()} ---")
            print("No data found")
            continue
            
        print(f"\n--- {policy.upper()} ---")
        print(f"Convergence month : {compute_convergence_month(df, policy)}")
        exp = compute_exploration_ratio_dict(df, policy)
        print(f"Exploration ratio : {exp['exploration_ratio']:.1f}%")
        print(f"  Keep: {exp['keep_pct']:.1f}%  "
              f"+10%: {exp['plus_10_pct']:.1f}%  "
              f"+20%: {exp['plus_20_pct']:.1f}%  "
              f"+50%: {exp['plus_50_pct']:.1f}%")
        cs = compute_cold_start_performance(df, users_df, policy)
        print(f"Cold start reward : ₹{cs['cold_start_avg_reward']:,.0f} "
              f"→ ₹{cs['warm_avg_reward']:,.0f} "
              f"({cs['reward_improvement_pct']:+.1f}%)")
        shock = compute_shock_recovery(df, policy)
        print(f"Shock recovery    : "
              f"{shock['months_to_recovery']} months "
              f"(peak default: {shock['peak_default_rate']:.1f}% "
              f"at month {shock['peak_default_month']})")
    
    print("\n\nFULL COMPARISON TABLE:")
    table = build_full_metrics_table(df, users_df, oracle_df)
    print(table.to_string(index=False))
