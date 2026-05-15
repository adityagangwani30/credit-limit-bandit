"""Main simulation loop for contextual credit-limit policies."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from src.bandits.epsilon_greedy import EpsilonGreedyBandit
from src.bandits.thompson import ThompsonSampling
from src.bandits.ucb import UCBBandit
from src.context import ContextBuilder
from src.reward import RewardBuffer
from src.simulator import simulate_month

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class _TqdmFallback:
        def __call__(self, iterable=None, total=None, desc=None, leave=True):
            return iterable

        @staticmethod
        def write(message: str) -> None:
            print(message)

    tqdm = _TqdmFallback()


def _policy_name(policy) -> str:
    if hasattr(policy, "get_stats"):
        stats = policy.get_stats()
        if isinstance(stats, dict) and "algorithm" in stats:
            return str(stats["algorithm"])
    return policy.__class__.__name__.lower()


def _seed_bandit(bandit, seed: int) -> None:
    if hasattr(bandit, "rng"):
        bandit.rng = np.random.default_rng(seed)


def _resolve_stress(
    month: int,
    economic_stress_fn: Callable[[int], float] | None,
) -> float:
    if economic_stress_fn is None:
        return 1.0

    stress = float(economic_stress_fn(month))
    if month == 6:
        stress = max(stress, 2.0)
    return max(stress, 0.01)


def _simulate_policy(
    users_df: pd.DataFrame,
    n_months: int,
    seed: int,
    economic_stress_fn: Callable[[int], float] | None,
    bandit=None,
    fixed_action: str | None = None,
    policy_label: str | None = None,
) -> pd.DataFrame:
    if bandit is None and fixed_action is None:
        raise ValueError("Either bandit or fixed_action must be provided")

    working_users = users_df.copy().reset_index(drop=True)
    context_builder = ContextBuilder(working_users)
    reward_buffer = RewardBuffer(delay_months=3)
    action_space = context_builder.get_action_space()
    user_records = working_users.to_dict("records")
    user_ids = [record["user_id"] for record in user_records]
    n_users = len(user_records)

    current_limits = {record["user_id"]: float(record["initial_credit_limit"]) for record in user_records}
    histories: dict[str, list[dict]] = {user_id: [] for user_id in user_ids}

    if bandit is not None:
        bandit.reset()
        _seed_bandit(bandit, seed)

    logs: list[dict] = []
    row_lookup: dict[tuple[str, int], int] = {}

    month_iterator = tqdm(range(1, n_months + 1), total=n_months, desc=f"{policy_label or 'policy'} months")
    for month in month_iterator:
        selected_actions: dict[str, str] = {}

        for user_index, record in enumerate(user_records, start=1):
            user_id = record["user_id"]
            context = context_builder.build_context(user_id, histories[user_id])

            if fixed_action is not None:
                action = fixed_action
            else:
                action = bandit.select_action(context, user_id, action_space)

            reward_buffer.record_action(user_id, month, action, context)

            current_limit = ContextBuilder.apply_action(current_limits[user_id], action)
            current_limits[user_id] = current_limit
            selected_actions[user_id] = action

            row_lookup[(user_id, month)] = len(logs)
            logs.append(
                {
                    "month": month,
                    "user_id": user_id,
                    "action_taken": action,
                    "reward_received": 0.0,
                    "current_limit": current_limit,
                    "did_default": False,
                    "amount_spent": 0.0,
                    "outstanding_amount": 0.0,
                    "reward_ready_month": np.nan,
                    "reward_is_default": np.nan,
                }
            )

            if user_index % 100 == 0 or user_index == n_users:
                tqdm.write(f"Month {month}: processed {user_index}/{n_users} users for {policy_label or 'policy'}")

        simulation_input = working_users.copy()
        simulation_input["initial_credit_limit"] = simulation_input["user_id"].map(current_limits).astype(float)
        stress = _resolve_stress(month, economic_stress_fn)
        outcomes = simulate_month(simulation_input, month_num=month, economic_stress=stress)
        outcome_map = outcomes.set_index("user_id").to_dict("index")

        for user_id in user_ids:
            outcome = outcome_map[user_id]
            histories[user_id].append(
                {
                    "month": month,
                    "amount_spent": float(outcome["amount_spent"]),
                    "outstanding_amount": float(outcome["outstanding_amount"]),
                    "did_default": bool(outcome["did_default"]),
                    "current_credit_limit": current_limits[user_id],
                }
            )

            row = logs[row_lookup[(user_id, month)]]
            row["did_default"] = bool(outcome["did_default"])
            row["amount_spent"] = float(outcome["amount_spent"])
            row["outstanding_amount"] = float(outcome["outstanding_amount"])

            reward_buffer.receive_outcome(
                user_id=user_id,
                month=month,
                amount_spent=outcome["amount_spent"],
                outstanding_amount=outcome["outstanding_amount"],
                did_default=outcome["did_default"],
            )

        ready_rewards = reward_buffer.get_ready_rewards(current_month=month)
        for ready_reward in ready_rewards:
            if bandit is not None:
                bandit.update(
                    user_id=ready_reward["user_id"],
                    action=ready_reward["action"],
                    reward=float(ready_reward["reward"]),
                    context=np.asarray(ready_reward["context"], dtype=np.float32),
                )
            log_index = row_lookup[(ready_reward["user_id"], ready_reward["action_month"])]
            logs[log_index]["reward_received"] = float(ready_reward["reward"])
            logs[log_index]["reward_ready_month"] = month
            logs[log_index]["reward_is_default"] = bool(ready_reward["is_default"])

    results = pd.DataFrame(logs)
    if policy_label is not None:
        results["policy"] = policy_label
    return results


def run_simulation(
    bandit,
    users_df: pd.DataFrame,
    n_months: int = 12,
    economic_stress_fn: Callable[[int], float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one contextual bandit policy across the simulation horizon."""

    return _simulate_policy(
        users_df=users_df,
        n_months=n_months,
        seed=seed,
        economic_stress_fn=economic_stress_fn,
        bandit=bandit,
        fixed_action=None,
        policy_label=_policy_name(bandit),
    )


def run_static_baseline(users_df: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    """Run the fixed keep-limit baseline."""

    return _simulate_policy(
        users_df=users_df,
        n_months=n_months,
        seed=42,
        economic_stress_fn=None,
        fixed_action="keep",
        policy_label="static_baseline",
    )


def run_oracle(users_df: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    """Build an oracle policy in hindsight from completed fixed-action runs."""

    fixed_policy_runs = []
    for action in ContextBuilder.get_action_space():
        fixed_policy_runs.append(
            _simulate_policy(
                users_df=users_df,
                n_months=n_months,
                seed=42,
                economic_stress_fn=None,
                fixed_action=action,
                policy_label=f"oracle_source_{action}",
            )
        )

    oracle_source = pd.concat(fixed_policy_runs, ignore_index=True)
    oracle_source["reward_rank"] = oracle_source["reward_received"]
    best_rows = (
        oracle_source.sort_values(["month", "user_id", "reward_rank"], ascending=[True, True, False])
        .groupby(["month", "user_id"], as_index=False)
        .head(1)
        .copy()
    )
    best_rows["policy"] = "oracle"
    return best_rows.drop(columns=["reward_rank"])


def _summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return (
        results_df.groupby("policy", as_index=False)
        .agg(
            total_reward=("reward_received", "sum"),
            default_rate=("did_default", "mean"),
        )
        .sort_values("total_reward", ascending=False)
    )


def _load_users() -> pd.DataFrame:
    data_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_users.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Synthetic user file not found: {data_path}")
    return pd.read_csv(data_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run contextual credit-limit simulations.")
    parser.add_argument(
        "--policy",
        default="all",
        choices=["all", "thompson", "ucb", "epsilon_greedy", "static", "oracle"],
        help="Run one policy or all policies.",
    )
    args = parser.parse_args()

    users_df = _load_users()
    start_time = time.perf_counter()
    all_results: list[pd.DataFrame] = []

    if args.policy in {"all", "thompson"}:
        all_results.append(run_simulation(ThompsonSampling(), users_df, n_months=12, seed=42))
    if args.policy in {"all", "ucb"}:
        all_results.append(run_simulation(UCBBandit(), users_df, n_months=12, seed=42))
    if args.policy in {"all", "epsilon_greedy"}:
        all_results.append(run_simulation(EpsilonGreedyBandit(), users_df, n_months=12, seed=42))
    if args.policy in {"all", "static"}:
        all_results.append(run_static_baseline(users_df, n_months=12))
    if args.policy in {"all", "oracle"}:
        all_results.append(run_oracle(users_df, n_months=12))

    combined_results = pd.concat(all_results, ignore_index=True)
    output_path = Path(__file__).resolve().parents[1] / "data" / "simulation_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_results.to_csv(output_path, index=False)

    summary = _summarize_results(combined_results)
    runtime_seconds = time.perf_counter() - start_time

    print("Simulation summary:")
    for row in summary.itertuples(index=False):
        print(
            f"  {row.policy}: total_reward={row.total_reward:.2f}, "
            f"default_rate={row.default_rate * 100:.2f}%"
        )
    print(f"Runtime: {runtime_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
