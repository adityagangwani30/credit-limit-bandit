import pandas as pd

from src.bandits.thompson import ThompsonSampling
from src.evaluate import (
    compute_convergence_month,
    compute_exploration_ratio_dict,
    compute_cold_start_performance,
    compute_shock_recovery,
    build_full_metrics_table,
)
from src.simulate_run import run_simulation
from src.simulator import generate_users


def test_thompson_integration_run_three_months():
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=3, seed=21)

    total_reward = float(results_df["reward_received"].sum())
    default_rate = float(results_df["did_default"].mean())
    action_types = set(results_df["action_taken"].unique())

    assert total_reward > 0
    assert default_rate < 0.20
    assert action_types == {"keep", "plus_10", "plus_20", "plus_50"}

    early_month_rewards = results_df.loc[results_df["month"].isin([1, 2]), "reward_received"]
    later_month_rewards = results_df.loc[results_df["month"] >= 3, "reward_received"]

    assert (early_month_rewards == 0.0).all()
    assert (later_month_rewards != 0.0).any()


def test_convergence_month_is_reasonable():
    """Convergence month should happen between month 1 and 12."""
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=12, seed=21)
    
    # Add a dummy policy column to match full results structure
    results_df["policy"] = "thompson_sampling"
    
    month = compute_convergence_month(results_df, "thompson_sampling")
    assert 1 <= month <= 12, f"Convergence month {month} out of range [1, 12]"


def test_exploration_ratio_sums_to_100():
    """Exploration ratio components should sum to 100%."""
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=12, seed=21)
    
    results_df["policy"] = "thompson_sampling"
    
    exp = compute_exploration_ratio_dict(results_df, "thompson_sampling")
    total = exp["keep_pct"] + exp["plus_10_pct"] + exp["plus_20_pct"] + exp["plus_50_pct"]
    assert abs(total - 100.0) < 0.01, f"Action percentages sum to {total}, not 100"


def test_shock_recovery_returns_valid_structure():
    """Shock recovery should return expected keys and valid relationships."""
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=12, seed=21)
    
    results_df["policy"] = "thompson_sampling"
    
    shock = compute_shock_recovery(results_df, "thompson_sampling")
    assert "months_to_recovery" in shock
    assert "pre_shock_default_rate" in shock
    assert "peak_default_rate" in shock
    # Peak default rate should be >= pre-shock default rate
    if shock["peak_default_rate"] is not None:
        assert shock["peak_default_rate"] >= shock["pre_shock_default_rate"]


def test_cold_start_warm_reward_is_higher_or_equal():
    """Warm reward should be >= cold start reward (learning improves over time)."""
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=12, seed=21)
    
    results_df["policy"] = "thompson_sampling"
    
    cs = compute_cold_start_performance(results_df, users_df, "thompson_sampling")
    # Warm reward should be >= cold start (learning should improve)
    assert (
        cs["warm_avg_reward"] >= cs["cold_start_avg_reward"]
    ), (
        f"Warm reward {cs['warm_avg_reward']} < cold start {cs['cold_start_avg_reward']} "
        "— model should learn over time"
    )


def test_full_metrics_table_has_expected_columns():
    """Build full metrics table and verify structure."""
    users_df = generate_users(n=100, seed=21)
    results_df = run_simulation(ThompsonSampling(), users_df, n_months=12, seed=21)
    
    results_df["policy"] = "thompson_sampling"
    
    # Add oracle for comparison (use same results as thompson for now)
    oracle_df = results_df.copy()
    oracle_df["policy"] = "oracle"
    full_results = pd.concat([results_df, oracle_df], ignore_index=True)
    
    table = build_full_metrics_table(full_results, users_df, oracle_df)
    
    # Check expected columns
    expected_columns = {
        "policy",
        "total_revenue_inr",
        "default_rate_pct",
        "regret_vs_oracle_pct",
        "convergence_month",
        "exploration_pct",
        "cold_start_reward",
        "shock_recovery_months",
    }
    assert expected_columns.issubset(set(table.columns)), f"Missing columns: {expected_columns - set(table.columns)}"
    assert len(table) >= 1, "Table should have at least one row"
