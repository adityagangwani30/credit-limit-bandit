"""Streamlit dashboard for the credit limit bandit project."""

from __future__ import annotations

from html import escape
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

SIDEBAR_WIDTH = 280
GITHUB_URL = "https://github.com/adityagangwani30/credit-limit-bandit"


def inject_global_styles() -> None:
    st.markdown(
        f"""
        <style>
            html, body, [class*="css"] {{
                font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
            }}

            .stApp {{
                background: #F8F9FA;
            }}

            [data-testid="stAppViewContainer"] {{
                background: #F8F9FA;
            }}

            [data-testid="stSidebar"] {{
                width: {SIDEBAR_WIDTH}px !important;
                min-width: {SIDEBAR_WIDTH}px !important;
                background: #FFFFFF;
            }}

            [data-testid="stHeader"] {{
                background: linear-gradient(180deg, rgba(55, 138, 221, 0.12), rgba(55, 138, 221, 0));
                border-top: 4px solid #378ADD;
            }}

            .app-hero {{
                background: #FFFFFF;
                border-top: 4px solid #378ADD;
                border-radius: 18px;
                padding: 1.2rem 1.4rem 1rem 1.4rem;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
                margin-bottom: 1rem;
            }}

            .app-hero h1 {{
                margin: 0;
                font-size: 2.25rem;
                color: #12324B;
                line-height: 1.1;
            }}

            .app-subtitle {{
                margin: 0.35rem 0 0 0;
                color: #5B6472;
                font-size: 1rem;
            }}

            .hero-pills {{
                display: flex;
                gap: 0.55rem;
                flex-wrap: wrap;
                margin-top: 0.95rem;
            }}

            .hero-pill {{
                display: inline-flex;
                align-items: center;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: #EAF3FD;
                color: #1E5E9A;
                font-size: 0.88rem;
                font-weight: 600;
            }}

            .section-rule {{
                border: 0;
                height: 1px;
                background: linear-gradient(90deg, rgba(55, 138, 221, 0.0), rgba(55, 138, 221, 0.45), rgba(55, 138, 221, 0.0));
                margin: 1rem 0 0 0;
            }}

            [data-testid="stMetric"] {{
                background: #FFFFFF;
                border: 1px solid #E4E8EE;
                border-radius: 16px;
                padding: 1rem 1rem 0.85rem 1rem;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            }}

            [data-testid="stMetric"] label {{
                color: #5B6472;
            }}

            .info-card {{
                background: #FFFFFF;
                border: 1px solid #E4E8EE;
                border-radius: 16px;
                padding: 1rem 1.1rem;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
                height: 100%;
            }}

            .info-card h4 {{
                margin: 0 0 0.7rem 0;
                color: #12324B;
                font-size: 1rem;
            }}

            .info-row {{
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                padding: 0.4rem 0;
                border-bottom: 1px solid #EEF2F6;
            }}

            .info-row:last-child {{
                border-bottom: none;
            }}

            .info-label {{
                color: #6B7280;
                font-weight: 600;
            }}

            .info-value {{
                color: #12324B;
                font-weight: 700;
                text-align: right;
            }}

            .risk-badge {{
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.25rem 0.75rem;
                border-radius: 999px;
                color: #FFFFFF;
                font-size: 0.82rem;
                font-weight: 700;
            }}

            .risk-prime {{ background: #2E9E4D; }}
            .risk-near-prime {{ background: #378ADD; }}
            .risk-subprime {{ background: #F28E2B; }}
            .risk-deep-subprime {{ background: #D64545; }}

            .winner-badge {{
                display: inline-flex;
                align-items: center;
                margin-left: 0.4rem;
                padding: 0.14rem 0.5rem;
                border-radius: 999px;
                background: #EAF7EF;
                color: #1F7A3F;
                font-size: 0.74rem;
                font-weight: 700;
                white-space: nowrap;
            }}

            table.summary-table {{
                width: 100%;
                border-collapse: collapse;
                background: #FFFFFF;
                border: 1px solid #E4E8EE;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            }}

            table.summary-table th {{
                background: #F2F6FB;
                color: #12324B;
                text-align: left;
                font-weight: 700;
                padding: 0.85rem 0.9rem;
                border-bottom: 1px solid #E4E8EE;
            }}

            table.summary-table td {{
                padding: 0.8rem 0.9rem;
                border-bottom: 1px solid #EEF2F6;
                color: #243447;
            }}

            table.summary-table tr:last-child td {{
                border-bottom: none;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <h1>Credit Limit Bandit</h1>
            <p class="app-subtitle">Contextual Multi-Armed Bandit for Dynamic Credit Limit Optimization</p>
            <div class="hero-pills">
                <span class="hero-pill">10,000 Users</span>
                <span class="hero-pill">12 Months</span>
                <span class="hero-pill">3 Policies</span>
            </div>
            <hr class="section-rule" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge_html(risk_tier: str) -> str:
    risk_class = {
        "Prime": "risk-prime",
        "Near-Prime": "risk-near-prime",
        "Subprime": "risk-subprime",
        "Deep-Subprime": "risk-deep-subprime",
    }.get(risk_tier, "risk-near-prime")
    return f'<span class="risk-badge {risk_class}">{escape(str(risk_tier))}</span>'


def metric_delta_text(current_value: float, baseline_value: float, suffix: str = "%") -> str:
    if baseline_value == 0:
        return "n/a vs static"
    delta = ((current_value - baseline_value) / baseline_value) * 100.0
    return f"{delta:+.2f}{suffix} vs static"


def render_caption(text: str) -> None:
    st.caption(text)


def render_summary_table(comparison: pd.DataFrame) -> None:
    display_df = comparison.copy()
    winner_idx = display_df["total_revenue_inr"].idxmax() if not display_df.empty else None
    if winner_idx is not None:
        display_df["policy"] = display_df["policy"].astype(str)
        display_df.loc[winner_idx, "policy"] = f"{display_df.loc[winner_idx, 'policy']} <span class='winner-badge'>Winner</span>"

    display_df["total_revenue_inr"] = display_df["total_revenue_inr"].map(lambda value: f"₹{value:,.0f}")
    display_df["default_rate_pct"] = display_df["default_rate_pct"].map(lambda value: f"{value:.2f}%")
    display_df["regret_vs_oracle_pct"] = display_df["regret_vs_oracle_pct"].map(lambda value: f"{value:.2f}%")
    display_df["exploration_ratio_pct"] = display_df["exploration_ratio_pct"].map(lambda value: f"{value:.2f}%")
    display_df["revenue_lift_vs_static_pct"] = display_df["revenue_lift_vs_static_pct"].map(lambda value: f"{value:.2f}%")

    display_df = display_df.rename(
        columns={
            "policy": "Policy",
            "total_revenue_inr": "Total Revenue (INR)",
            "default_rate_pct": "Default Rate (%)",
            "regret_vs_oracle_pct": "Regret vs Oracle (%)",
            "exploration_ratio_pct": "Exploration Ratio (%)",
            "convergence_month": "Convergence Month",
            "revenue_lift_vs_static_pct": "Revenue Lift vs Static (%)",
        }
    )

    st.markdown(display_df.to_html(classes="summary-table", index=False, escape=False), unsafe_allow_html=True)


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
        chart = px.line(
            monthly,
            x="month",
            y="cumulative_reward",
            markers=True,
            title="Live Thompson Learning Curve",
            labels={"month": "Month", "cumulative_reward": "Cumulative Revenue (₹)"},
        )
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
    static_default_rate = float(static_df["did_default"].mean() * 100.0) if not static_df.empty else 0.0
    static_increase_users = int(static_df.loc[static_df["action_taken"] != "keep", "user_id"].nunique()) if not static_df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue (Thompson)", f"₹{total_revenue:,.0f}", metric_delta_text(total_revenue, static_revenue, suffix="%"))
    col2.metric("Revenue Lift vs Static", f"{revenue_lift:.2f}%", f"{revenue_lift:+.2f}% vs static")
    col3.metric("Portfolio Default Rate", f"{default_rate:.2f}%", metric_delta_text(default_rate, static_default_rate))
    col4.metric(
        "Users with Limit Increases",
        f"{users_with_increases:,}",
        f"{users_with_increases - static_increase_users:+,} vs static",
    )

    st.info("Revenue lift means the extra reward generated by Thompson Sampling compared with the static baseline, expressed as a percentage of static revenue.")

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
            labels={"month": "Month", "cumulative_reward": "Cumulative Revenue (₹)", "policy_display": "Policy"},
        )
        st.plotly_chart(fig, use_container_width=True)
        render_caption("Thompson Sampling is tracked alongside the other policies to show how revenue compounds over time.")

    with right:
        action_counts = thompson_df["action_taken"].value_counts().reset_index()
        action_counts.columns = ["action_taken", "count"]
        pie = px.pie(action_counts, names="action_taken", values="count", title="Thompson Action Mix")
        st.plotly_chart(pie, use_container_width=True)
        render_caption("Action mix shows how often Thompson Sampling kept a limit steady or increased it during the run.")

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
    render_caption("The gauge highlights the portfolio default rate relative to a low-risk operating range.")


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

    user_results = results_df.loc[results_df["user_id"] == selected_user_id].copy()
    if user_results.empty:
        st.info("No simulation records found for this user.")
        return

    policy_options = sorted(user_results["policy_display"].unique())
    selected_policy = st.selectbox("Policy", options=policy_options, index=0)
    policy_user_results = user_results.loc[user_results["policy_display"] == selected_policy].sort_values("month")

    limit_increase_count = int((policy_user_results["action_taken"] != "keep").sum())
    default_count = int(policy_user_results["did_default"].sum())
    st.success(
        f"This user had {limit_increase_count} limit increases over 12 months and defaulted {default_count} times."
    )

    profile_col1, profile_col2 = st.columns(2)
    with profile_col1:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Profile</h4>
                    <div class="info-row"><span class="info-label">Risk Tier</span><span class="info-value">{risk_badge_html(str(user_row['risk_tier']))}</span></div>
                    <div class="info-row"><span class="info-label">Income Bucket</span><span class="info-value">{escape(str(user_row['income_bucket']))}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with profile_col2:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="info-card">
                    <h4>Credit Snapshot</h4>
                    <div class="info-row"><span class="info-label">CIBIL Score</span><span class="info-value">{int(user_row['cibil_score'])}</span></div>
                    <div class="info-row"><span class="info-label">Initial Limit</span><span class="info-value">₹{float(user_row['initial_credit_limit']):,.0f}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        fig_limit = px.line(
            policy_user_results,
            x="month",
            y="current_limit",
            markers=True,
            title="Credit Limit Trajectory",
            labels={"month": "Month", "current_limit": "Credit Limit (₹)"},
        )
        st.plotly_chart(fig_limit, use_container_width=True)
        render_caption("The trajectory shows how the selected policy moved this user’s limit over time.")

    with right:
        fig_reward = px.bar(
            policy_user_results,
            x="month",
            y="reward_received",
            title="Monthly Reward",
            labels={"month": "Month", "reward_received": "Monthly Reward (₹)"},
        )
        st.plotly_chart(fig_reward, use_container_width=True)
        render_caption("Monthly reward combines revenue earned from spend and the cost of defaults.")

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
    reward_tab, regret_tab, summary_tab = st.tabs(["Reward Curves", "Regret Curves", "Summary Table"])

    with reward_tab:
        cumulative = monthly_cumulative_rewards(results_df)
        fig = px.line(
            cumulative,
            x="month",
            y="cumulative_reward",
            color="policy_display",
            markers=True,
            title="Cumulative Reward Curves",
            labels={"month": "Month", "cumulative_reward": "Cumulative Reward (₹)", "policy_display": "Policy"},
        )
        st.plotly_chart(fig, use_container_width=True)
        render_caption("Higher cumulative reward means the policy generated more value over the full simulation horizon.")

    with regret_tab:
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
            labels={"month": "Month", "regret": "Regret (₹)", "policy_display": "Policy"},
        )
        st.plotly_chart(regret_fig, use_container_width=True)
        render_caption("Regret measures the cumulative gap between each policy and the oracle benchmark.")

    with summary_tab:
        policy_frames = {policy: frame.copy() for policy, frame in results_df.groupby("policy")}
        comparison = policy_comparison_table(policy_frames)
        comparison["policy"] = comparison["policy"].map(POLICY_LABELS).fillna(comparison["policy"])
        render_summary_table(comparison)

        st.divider()
        default_df = default_rate_by_policy(results_df)
        default_fig = px.bar(
            default_df,
            x="policy_display",
            y="default_rate_pct",
            color="policy_display",
            title="Default Rate by Policy",
            labels={"policy_display": "Policy", "default_rate_pct": "Default Rate (%)"},
        )
        st.plotly_chart(default_fig, use_container_width=True)
        render_caption("Default rate by policy provides the risk-side comparison for the same simulation data.")


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
        st.warning(f"Risk distribution currently sums to {total_pct}. Adjust the sliders to total 100.")
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
    fig = px.line(
        monthly,
        x="month",
        y="cumulative_reward",
        markers=True,
        title="Live Simulation Learning Curve",
        labels={"month": "Month", "cumulative_reward": "Cumulative Revenue (₹)"},
    )
    st.plotly_chart(fig, use_container_width=True)
    render_caption("The live learning curve updates as each month of the simulation completes.")

    st.success(
        f"Simulation complete. Revenue lift vs static: {revenue_lift:.2f}% | Final default rate: {default_rate:.2f}% | Months simulated: {n_months}"
    )
    m1, m2 = st.columns(2)
    m1.metric("Final Revenue Lift vs Static", f"{revenue_lift:.2f}%")
    m2.metric("Final Default Rate", f"{default_rate:.2f}%")


def main() -> None:
    results_df = load_simulation_results()
    users_df = load_synthetic_users()

    inject_global_styles()
    render_app_header()
    page = st.sidebar.radio(
        "Navigation",
        ["💼 Portfolio", "👤 Per-User", "📊 Policy Compare", "⚡ Live Sim"],
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Thompson Sampling · Streamlit · Plotly")
    st.sidebar.markdown(f"[Project on GitHub]({GITHUB_URL})")

    page_map = {
        "💼 Portfolio": "Portfolio Overview",
        "👤 Per-User": "Per-User Deep Dive",
        "📊 Policy Compare": "Policy Comparison",
        "⚡ Live Sim": "Live Simulation",
    }
    selected_page = page_map[page]

    if selected_page in {"Portfolio Overview", "Per-User Deep Dive", "Policy Comparison"} and not require_data(results_df, users_df):
        return
    if selected_page == "Live Simulation" and users_df.empty:
        st.warning("Missing `data/synthetic_users.csv`. Generate it first to run the live simulation.")
        return

    if selected_page == "Portfolio Overview":
        render_portfolio_overview(results_df, users_df)
    elif selected_page == "Per-User Deep Dive":
        render_user_deep_dive(enrich_results(results_df, users_df), users_df)
    elif selected_page == "Policy Comparison":
        render_policy_comparison(results_df)
    else:
        render_live_simulation(users_df, results_df)


if __name__ == "__main__":
    main()
