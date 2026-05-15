"""Streamlit dashboard for the credit limit bandit project."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bandits.thompson import ThompsonSampling
from src.context import ContextBuilder
from src.evaluate import compute_regret, policy_comparison_table
from src.reward import RewardBuffer
from src.simulator import simulate_month


st.set_page_config(page_title="Credit Limit Bandit Dashboard", layout="wide", page_icon="💳")

POLICY_LABELS = {
    "thompson_sampling": "Thompson",
    "ucb": "UCB",
    "epsilon_greedy": "Epsilon-Greedy",
    "static_baseline": "Static",
    "oracle": "Oracle",
}


@st.cache_data
def load_simulation_results() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "simulation_results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["policy_display"] = df["policy"].map(POLICY_LABELS).fillna(df["policy"])
    return df


@st.cache_data
def load_synthetic_users() -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "synthetic_users.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def require_data(results_df: pd.DataFrame, users_df: pd.DataFrame) -> bool:
    if results_df.empty or users_df.empty:
        st.warning("Missing `data/simulation_results.csv` or `data/synthetic_users.csv`. Generate them first to use the dashboard.")
        return False
    return True


def monthly_cumulative_rewards(results_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        results_df.groupby(["policy_display", "month"], as_index=False)["reward_received"]
        .sum()
        .sort_values(["policy_display", "month"])
    )
    monthly["cumulative_reward"] = monthly.groupby("policy_display")["reward_received"].cumsum()
    return monthly


def default_rate_by_policy(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("policy_display", as_index=False)["did_default"]
        .mean()
        .rename(columns={"did_default": "default_rate"})
    )
    summary["default_rate_pct"] = summary["default_rate"] * 100.0
    return summary


def enrich_results(results_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["user_id", "risk_tier", "income_bucket", "cibil_score", "initial_credit_limit"]
    return results_df.merge(users_df[base_columns], on="user_id", how="left")


def build_custom_portfolio(users_df: pd.DataFrame, risk_distribution: dict[str, int], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    total_users = len(users_df)
    desired_counts = {
        "Prime": int(total_users * risk_distribution["Prime"] / 100),
        "Near-Prime": int(total_users * risk_distribution["Near-Prime"] / 100),
        "Subprime": int(total_users * risk_distribution["Subprime"] / 100),
    }
    desired_counts["Deep-Subprime"] = total_users - sum(desired_counts.values())

    sampled_frames: list[pd.DataFrame] = []
    for tier, count in desired_counts.items():
        tier_df = users_df.loc[users_df["risk_tier"] == tier]
        sampled = tier_df.sample(n=count, replace=True, random_state=int(rng.integers(0, 1_000_000)))
        sampled_frames.append(sampled)

    portfolio = pd.concat(sampled_frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    portfolio["user_id"] = [f"LIVE_{idx:05d}" for idx in range(1, len(portfolio) + 1)]
    return portfolio


def apply_fraud_spike(outcomes: pd.DataFrame, users_df: pd.DataFrame, month: int, rng: np.random.Generator) -> pd.DataFrame:
    if month != 9:
        return outcomes

    shocked = outcomes.merge(users_df[["user_id", "risk_tier"]], on="user_id", how="left")
    mask = shocked["risk_tier"] == "Deep-Subprime"
    tier_slice = shocked.loc[mask]
    if tier_slice.empty:
        return outcomes

    current_default_rate = float(tier_slice["did_default"].mean())
    if current_default_rate <= 0:
        current_default_rate = 0.10

    non_default_idx = tier_slice.index[~tier_slice["did_default"]]
    if len(non_default_idx) == 0:
        return outcomes

    extra_flip_prob = min(1.0, current_default_rate)
    flip_flags = rng.random(len(non_default_idx)) < extra_flip_prob
    flip_idx = non_default_idx[flip_flags]

    shocked.loc[flip_idx, "did_default"] = True
    shocked.loc[flip_idx, "outstanding_amount"] = np.round(
        shocked.loc[flip_idx, "amount_spent"] * rng.uniform(0.72, 1.0, size=len(flip_idx)),
        2,
    )
    return shocked[outcomes.columns]


def run_live_thompson_simulation(
    users_df: pd.DataFrame,
    economic_stress: float,
    fraud_spike: bool,
    risk_distribution: dict[str, int],
    n_months: int,
    progress_bar,
    status_placeholder,
    chart_placeholder,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    live_users = build_custom_portfolio(users_df, risk_distribution, seed=42)
    context_builder = ContextBuilder(live_users)
    reward_buffer = RewardBuffer(delay_months=3)
    bandit = ThompsonSampling()
    bandit.rng = np.random.default_rng(42)

    current_limits = {row.user_id: float(row.initial_credit_limit) for row in live_users.itertuples(index=False)}
    histories: dict[str, list[dict]] = {user_id: [] for user_id in live_users["user_id"]}
    logs: list[dict] = []
    row_lookup: dict[tuple[str, int], int] = {}
    action_space = ContextBuilder.get_action_space()

    for month in range(1, n_months + 1):
        status_placeholder.info(f"Running month {month} of {n_months}...")

        for user_id in live_users["user_id"]:
            context = context_builder.build_context(user_id, histories[user_id])
            action = bandit.select_action(context, user_id, action_space)
            reward_buffer.record_action(user_id, month, action, context)

            current_limits[user_id] = ContextBuilder.apply_action(current_limits[user_id], action)
            row_lookup[(user_id, month)] = len(logs)
            logs.append(
                {
                    "month": month,
                    "user_id": user_id,
                    "policy": "thompson_sim_live",
                    "action_taken": action,
                    "reward_received": 0.0,
                    "current_limit": current_limits[user_id],
                    "did_default": False,
                    "amount_spent": 0.0,
                    "outstanding_amount": 0.0,
                }
            )

        simulation_input = live_users.copy()
        simulation_input["initial_credit_limit"] = simulation_input["user_id"].map(current_limits)
        stress = economic_stress
        if month == 6:
            stress = max(stress, 2.0)

        outcomes = simulate_month(simulation_input, month_num=month, economic_stress=stress)
        if fraud_spike:
            outcomes = apply_fraud_spike(outcomes, live_users, month, rng)

        for outcome in outcomes.itertuples(index=False):
            histories[outcome.user_id].append(
                {
                    "month": month,
                    "amount_spent": float(outcome.amount_spent),
                    "outstanding_amount": float(outcome.outstanding_amount),
                    "did_default": bool(outcome.did_default),
                    "current_credit_limit": current_limits[outcome.user_id],
                }
            )
            row = logs[row_lookup[(outcome.user_id, month)]]
            row["did_default"] = bool(outcome.did_default)
            row["amount_spent"] = float(outcome.amount_spent)
            row["outstanding_amount"] = float(outcome.outstanding_amount)
            reward_buffer.receive_outcome(
                outcome.user_id,
                month,
                outcome.amount_spent,
                outcome.outstanding_amount,
                outcome.did_default,
            )

        ready_rewards = reward_buffer.get_ready_rewards(month)
        for ready_reward in ready_rewards:
            bandit.update(
                ready_reward["user_id"],
                ready_reward["action"],
                float(ready_reward["reward"]),
                np.asarray(ready_reward["context"], dtype=np.float32),
            )
            logs[row_lookup[(ready_reward["user_id"], month)]]["reward_received"] = float(
                ready_reward["reward"]
            )

        progress_bar.progress(month / n_months)
        partial_df = pd.DataFrame(logs)
        monthly = partial_df.groupby("month", as_index=False)["reward_received"].sum()
        monthly["cumulative_reward"] = monthly["reward_received"].cumsum()
        chart = px.line(monthly, x="month", y="cumulative_reward", markers=True, title="Live Thompson Learning Curve")
        chart_placeholder.plotly_chart(chart, use_container_width=True)

    status_placeholder.success("Simulation complete.")
    return pd.DataFrame(logs)


def render_portfolio_overview(results_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    thompson_df = results_df.loc[results_df["policy"] == "thompson_sampling"]
    static_df = results_df.loc[results_df["policy"] == "static_baseline"]

    total_revenue = float(thompson_df["reward_received"].sum())
    static_revenue = float(static_df["reward_received"].sum())
    revenue_lift = ((total_revenue - static_revenue) / static_revenue * 100.0) if static_revenue else 0.0
    default_rate = float(thompson_df["did_default"].mean() * 100.0)
    users_with_increases = int(
        thompson_df.loc[thompson_df["action_taken"] != "keep", "user_id"].nunique()
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue (Thompson)", f"₹{total_revenue:,.0f}")
    col2.metric("Revenue Lift vs Static", f"{revenue_lift:.2f}%")
    col3.metric("Portfolio Default Rate", f"{default_rate:.2f}%")
    col4.metric("Users with Limit Increases", f"{users_with_increases:,}")

    left, right = st.columns([2, 1])
    with left:
        cumulative = monthly_cumulative_rewards(results_df)
        fig = px.line(
            cumulative,
            x="month",
            y="cumulative_reward",
            color="policy_display",
            markers=True,
            title="Cumulative Revenue by Month",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        action_counts = thompson_df["action_taken"].value_counts().reset_index()
        action_counts.columns = ["action_taken", "count"]
        pie = px.pie(action_counts, names="action_taken", values="count", title="Thompson Action Mix")
        st.plotly_chart(pie, use_container_width=True)

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=default_rate,
            title={"text": "Current Default Rate (%)"},
            gauge={
                "axis": {"range": [0, max(8, default_rate + 1)]},
                "bar": {"color": "#cc5500"},
                "threshold": {"line": {"color": "red", "width": 4}, "value": 4.0},
                "steps": [
                    {"range": [0, 4], "color": "#d8f3dc"},
                    {"range": [4, max(8, default_rate + 1)], "color": "#ffd6a5"},
                ],
            },
        )
    )
    st.plotly_chart(gauge, use_container_width=True)


def render_user_deep_dive(results_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    user_ids = users_df["user_id"].tolist()
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = user_ids[0]

    input_col, button_col = st.sidebar.columns([3, 1])
    selected_input = input_col.text_input("User ID", value=st.session_state.selected_user_id)
    if button_col.button("Random"):
        st.session_state.selected_user_id = str(np.random.default_rng().choice(user_ids))
    elif selected_input in set(user_ids):
        st.session_state.selected_user_id = selected_input

    selected_user_id = st.session_state.selected_user_id
    user_row = users_df.loc[users_df["user_id"] == selected_user_id].iloc[0]
    st.subheader(f"User {selected_user_id}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk Tier", str(user_row["risk_tier"]))
    c2.metric("Income Bucket", str(user_row["income_bucket"]))
    c3.metric("CIBIL Score", f"{int(user_row['cibil_score'])}")
    c4.metric("Initial Limit", f"₹{float(user_row['initial_credit_limit']):,.0f}")

    user_results = results_df.loc[results_df["user_id"] == selected_user_id].copy()
    if user_results.empty:
        st.info("No simulation records found for this user.")
        return

    policy_options = sorted(user_results["policy_display"].unique())
    selected_policy = st.selectbox("Policy", options=policy_options, index=0)
    policy_user_results = user_results.loc[user_results["policy_display"] == selected_policy].sort_values("month")

    left, right = st.columns(2)
    with left:
        fig_limit = px.line(
            policy_user_results,
            x="month",
            y="current_limit",
            markers=True,
            title="Credit Limit Trajectory",
        )
        st.plotly_chart(fig_limit, use_container_width=True)

    with right:
        fig_reward = px.bar(
            policy_user_results,
            x="month",
            y="reward_received",
            title="Monthly Reward",
        )
        st.plotly_chart(fig_reward, use_container_width=True)

    st.dataframe(
        policy_user_results[["month", "action_taken", "amount_spent", "did_default", "reward_received"]]
        .rename(
            columns={
                "action_taken": "action",
                "amount_spent": "amount_spent_inr",
                "reward_received": "reward_inr",
            }
        ),
        use_container_width=True,
    )


def render_policy_comparison(results_df: pd.DataFrame) -> None:
    cumulative = monthly_cumulative_rewards(results_df)
    fig = px.line(
        cumulative,
        x="month",
        y="cumulative_reward",
        color="policy_display",
        markers=True,
        title="Cumulative Reward Curves",
    )
    st.plotly_chart(fig, use_container_width=True)

    oracle_df = results_df.loc[results_df["policy"] == "oracle"]
    regret_frames = []
    for policy in ["thompson_sampling", "ucb", "epsilon_greedy"]:
        regret_df = compute_regret(oracle_df, results_df.loc[results_df["policy"] == policy])
        regret_df["policy_display"] = POLICY_LABELS.get(policy, policy)
        regret_frames.append(regret_df)
    regret_plot_df = pd.concat(regret_frames, ignore_index=True)
    regret_fig = px.line(
        regret_plot_df,
        x="month",
        y="regret",
        color="policy_display",
        markers=True,
        title="Regret vs Oracle",
    )
    st.plotly_chart(regret_fig, use_container_width=True)

    policy_frames = {policy: frame.copy() for policy, frame in results_df.groupby("policy")}
    comparison = policy_comparison_table(policy_frames)
    comparison["policy"] = comparison["policy"].map(POLICY_LABELS).fillna(comparison["policy"])
    st.dataframe(comparison, use_container_width=True)

    default_df = default_rate_by_policy(results_df)
    default_fig = px.bar(
        default_df,
        x="policy_display",
        y="default_rate_pct",
        color="policy_display",
        title="Default Rate by Policy",
    )
    st.plotly_chart(default_fig, use_container_width=True)


def render_live_simulation(users_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    stress = st.sidebar.slider("Economic Stress", min_value=1.0, max_value=3.0, value=1.0, step=0.1)
    fraud_spike = st.sidebar.toggle("Fraud Spike", value=False)
    st.sidebar.caption("Risk distribution must sum to 100%.")

    prime = st.sidebar.slider("Prime %", 0, 100, 40)
    near_prime = st.sidebar.slider("Near-Prime %", 0, 100, 30)
    subprime = st.sidebar.slider("Subprime %", 0, 100, 20)
    deep_subprime = st.sidebar.slider("Deep-Subprime %", 0, 100, 10)
    total_pct = prime + near_prime + subprime + deep_subprime
    n_months = st.sidebar.select_slider("Months", options=[6, 9, 12], value=12)

    risk_distribution = {
        "Prime": prime,
        "Near-Prime": near_prime,
        "Subprime": subprime,
        "Deep-Subprime": deep_subprime,
    }

    if total_pct != 100:
        st.error(f"Risk distribution currently sums to {total_pct}. Adjust the sliders to total 100.")
        return

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()
    chart_placeholder = st.empty()

    if st.sidebar.button("Run Simulation", type="primary"):
        live_results = run_live_thompson_simulation(
            users_df=users_df,
            economic_stress=stress,
            fraud_spike=fraud_spike,
            risk_distribution=risk_distribution,
            n_months=n_months,
            progress_bar=progress_bar,
            status_placeholder=status_placeholder,
            chart_placeholder=chart_placeholder,
        )
        st.session_state.live_results = live_results

    live_results = st.session_state.get("live_results")
    if live_results is None:
        st.info("Use the sidebar controls and click `Run Simulation` to launch a live Thompson Sampling run.")
        return

    static_df = results_df.loc[results_df["policy"] == "static_baseline"]
    static_total = float(static_df["reward_received"].sum()) if not static_df.empty else 0.0
    live_total = float(live_results["reward_received"].sum())
    revenue_lift = ((live_total - static_total) / static_total * 100.0) if static_total else 0.0
    default_rate = float(live_results["did_default"].mean() * 100.0)

    monthly = live_results.groupby("month", as_index=False)["reward_received"].sum()
    monthly["cumulative_reward"] = monthly["reward_received"].cumsum()
    fig = px.line(monthly, x="month", y="cumulative_reward", markers=True, title="Live Simulation Learning Curve")
    st.plotly_chart(fig, use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("Final Revenue Lift vs Static", f"{revenue_lift:.2f}%")
    m2.metric("Final Default Rate", f"{default_rate:.2f}%")


def main() -> None:
    results_df = load_simulation_results()
    users_df = load_synthetic_users()

    st.title("💳 Credit Limit Bandit Dashboard")
    page = st.sidebar.radio(
        "Navigation",
        ["Portfolio Overview", "Per-User Deep Dive", "Policy Comparison", "Live Simulation"],
    )

    if page in {"Portfolio Overview", "Per-User Deep Dive", "Policy Comparison"} and not require_data(results_df, users_df):
        return
    if page == "Live Simulation" and users_df.empty:
        st.warning("Missing `data/synthetic_users.csv`. Generate it first to run the live simulation.")
        return

    if page == "Portfolio Overview":
        render_portfolio_overview(results_df, users_df)
    elif page == "Per-User Deep Dive":
        render_user_deep_dive(enrich_results(results_df, users_df), users_df)
    elif page == "Policy Comparison":
        render_policy_comparison(results_df)
    else:
        render_live_simulation(users_df, results_df)


if __name__ == "__main__":
    main()
