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
from src.evaluate import compute_regret
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
    fixed_action_mode: str = "repeat",
    policy_label: str | None = None,
) -> pd.DataFrame:
    if bandit is None and fixed_action is None:
        raise ValueError("Either bandit or fixed_action must be provided")
    if fixed_action_mode not in {"repeat", "once_then_keep"}:
        raise ValueError(f"Unsupported fixed_action_mode: {fixed_action_mode}")

    working_users = users_df.copy().reset_index(drop=True)
    context_builder = ContextBuilder(working_users)
    # In the portfolio runner, rewards begin surfacing from month 3 onward so
    # short development runs still observe delayed feedback.
    reward_buffer = RewardBuffer(delay_months=2)
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
        selected_contexts: dict[str, np.ndarray] = {}

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

        for user_index, record in enumerate(user_records, start=1):
            user_id = record["user_id"]
            context = context_builder.build_context(user_id, histories[user_id])

            if fixed_action is not None:
                action = fixed_action if fixed_action_mode == "repeat" or month == 1 else "keep"
            else:
                action = bandit.select_action(context, user_id, action_space)

            current_limit = ContextBuilder.apply_action(current_limits[user_id], action)
            current_limits[user_id] = current_limit
            selected_actions[user_id] = action
            selected_contexts[user_id] = np.asarray(context, dtype=np.float32)

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
            reward_buffer.record_action(
                user_id=user_id,
                month=month,
                action=selected_actions[user_id],
                context=selected_contexts[user_id],
            )

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
        fixed_action_mode="repeat",
        policy_label="static_baseline",
    )


def run_static_action(
    users_df: pd.DataFrame,
    action: str,
    n_months: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """Run a fixed action for every user across the entire horizon."""

    return _simulate_policy(
        users_df=users_df,
        n_months=n_months,
        seed=seed,
        economic_stress_fn=None,
        fixed_action=action,
        fixed_action_mode="once_then_keep",
        policy_label=f"static_{action}",
    )


def run_theoretical_oracle(users_df: pd.DataFrame, n_months: int = 12, seed: int = 42) -> pd.DataFrame:
    """
    Build an oracle policy in hindsight from completed fixed-action runs.
    
    Theoretical oracle: for each user at each month, look at what reward
    each action would have produced (in hindsight), then assign the action
    that produced the maximum reward. This is an unreachable upper bound
    for any online policy because it uses per-month hindsight.
    """
    all_action_results = {}
    actions = ContextBuilder.get_action_space()
    
    for action in actions:
        df = _simulate_policy(
            users_df=users_df.copy(),
            n_months=n_months,
            seed=seed,
            economic_stress_fn=None,
            fixed_action=action,
            fixed_action_mode="repeat",
            policy_label=f"oracle_source_{action}",
        )
        all_action_results[action] = df
    
    # Concatenate all actions
    oracle_source = pd.concat(list(all_action_results.values()), ignore_index=True)
    
    # Ensure consistent data types
    oracle_source = oracle_source.astype({
        "month": "int64",
        "user_id": "str",
        "action_taken": "str",
        "reward_received": "float64",
    })
    
    # Sort by (month, user_id, reward_received) to get best reward first per group
    oracle_source = oracle_source.sort_values(
        by=["month", "user_id", "reward_received"],
        ascending=[True, True, False],
        na_position="last"
    )
    
    # Get best (first) row per user-month
    oracle_best = oracle_source.groupby(["month", "user_id"], as_index=False).first()
    oracle_best["policy"] = "oracle_theoretical"
    
    return oracle_best


def run_practical_oracle(users_df: pd.DataFrame, n_months: int = 12, seed: int = 42) -> pd.DataFrame:
    """
    Practical oracle: for each user, find the single best action
    to apply for the entire horizon.

    This is a much fairer ceiling for an online policy because it does not
    switch actions using month-by-month hindsight.
    """
    actions = ContextBuilder.get_action_space()
    all_action_results: dict[str, pd.DataFrame] = {}
    user_totals: dict[str, dict[str, float]] = {}

    for action in actions:
        df = run_static_action(users_df.copy(), action, n_months=n_months, seed=seed)
        all_action_results[action] = df
        grouped = df.groupby("user_id", as_index=False)["reward_received"].sum()
        for row in grouped.itertuples(index=False):
            if row.user_id not in user_totals:
                user_totals[row.user_id] = {}
            user_totals[row.user_id][action] = float(row.reward_received)

    chosen_frames: list[pd.DataFrame] = []
    for user_id, action_rewards in user_totals.items():
        best_action = max(action_rewards, key=action_rewards.get)
        chosen_user_path = all_action_results[best_action].loc[
            all_action_results[best_action]["user_id"] == user_id
        ].copy()
        chosen_user_path["policy"] = "oracle_practical"
        chosen_frames.append(chosen_user_path)

    return pd.concat(chosen_frames, ignore_index=True)


def run_oracle(users_df: pd.DataFrame, n_months: int = 12, seed: int = 42) -> pd.DataFrame:
    """Backward-compatible alias for the theoretical oracle."""

    return run_theoretical_oracle(users_df, n_months=n_months, seed=seed)


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
        choices=[
            "all",
            "thompson",
            "thompson_sampling",
            "ucb",
            "epsilon_greedy",
            "static",
            "static_baseline",
            "oracle",
            "oracle_theoretical",
            "oracle_practical",
        ],
        help="Run one policy or all policies.",
    )
    parser.add_argument("--n_months", type=int, default=12, help="Number of months to simulate.")
    parser.add_argument("--n_users", type=int, default=None, help="Optional cap on number of users.")
    args = parser.parse_args()

    users_df = _load_users()
    if args.n_users is not None:
        if args.n_users <= 0:
            raise ValueError("--n_users must be > 0 when provided")
        users_df = users_df.head(args.n_users).copy()

    if args.n_months <= 0:
        raise ValueError("--n_months must be > 0")

    start_time = time.perf_counter()
    all_results: list[pd.DataFrame] = []
    thompson_df = None

    if args.policy in {"all", "thompson", "thompson_sampling"}:
        thompson_df = run_simulation(ThompsonSampling(), users_df, n_months=args.n_months, seed=42)
        all_results.append(thompson_df)
    if args.policy in {"all", "ucb"}:
        all_results.append(run_simulation(UCBBandit(), users_df, n_months=args.n_months, seed=42))
    if args.policy in {"all", "epsilon_greedy"}:
        all_results.append(run_simulation(EpsilonGreedyBandit(), users_df, n_months=args.n_months, seed=42))
    if args.policy in {"all", "static", "static_baseline"}:
        all_results.append(run_static_baseline(users_df, n_months=args.n_months))
    theoretical_oracle_df = None
    practical_oracle_df = None

    if args.policy in {"all", "oracle", "oracle_theoretical"}:
        theoretical_oracle_df = run_theoretical_oracle(users_df, n_months=args.n_months)
        all_results.append(theoretical_oracle_df)
    if args.policy in {"all", "oracle_practical"}:
        practical_oracle_df = run_practical_oracle(users_df, n_months=args.n_months)
        all_results.append(practical_oracle_df)

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

    if thompson_df is not None:
        if theoretical_oracle_df is None:
            theoretical_oracle_df = run_theoretical_oracle(users_df, n_months=args.n_months)
        if practical_oracle_df is None:
            practical_oracle_df = run_practical_oracle(users_df, n_months=args.n_months)

        theoretical_regret = compute_regret(theoretical_oracle_df, thompson_df)
        practical_regret = compute_regret(practical_oracle_df, thompson_df)
        print(f"Regret vs Theoretical Oracle: {theoretical_regret['regret_pct']:.1f}%")
        print(f"Regret vs Practical Oracle:   {practical_regret['regret_pct']:.1f}%")

    print(f"Runtime: {runtime_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
