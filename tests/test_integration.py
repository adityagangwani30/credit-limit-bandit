import pandas as pd

from src.bandits.thompson import ThompsonSampling
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
