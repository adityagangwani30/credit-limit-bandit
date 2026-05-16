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

st.set_page_config(
    page_title="Credit Limit Bandit",
    page_icon="ðŸ’³",
    layout="wide",
    initial_sidebar_state="expanded",
)

# â”€â”€ Constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
POLICY_LABELS = {
    "thompson_sampling": "Thompson Sampling",
    "ucb": "UCB",
    "epsilon_greedy": "Epsilon-Greedy",
    "static_baseline": "Static Baseline",
    "oracle": "Oracle",
}
POLICY_COLORS = {
    "Thompson Sampling": "#3b82f6",
    "UCB": "#8b5cf6",
    "Epsilon-Greedy": "#f59e0b",
    "Static Baseline": "#374151",
    "Oracle": "#10b981",
}
ACTION_COLORS = {"keep": "#374151", "plus_10": "#3b82f6", "plus_20": "#8b5cf6", "plus_50": "#10b981"}
RISK_TIER_COLORS = {"Prime": "#10b981", "Near-Prime": "#3b82f6", "Subprime": "#f59e0b", "Deep-Subprime": "#ef4444"}
GITHUB_URL = "https://github.com/adityagangwani30/credit-limit-bandit"

# â”€â”€ Formatting helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def format_inr(amount):
    if amount >= 1e7:
        return f"â‚¹{amount/1e7:.1f}Cr"
    if amount >= 1e5:
        return f"â‚¹{amount/1e5:.1f}L"
    return f"â‚¹{amount:,.0f}"


def dark_chart(fig, title=None, height=320):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#161b2e",
        plot_bgcolor="#161b2e",
        font=dict(family="Inter, system-ui", color="#c9d1e0", size=11),
        title=dict(
            text=title,
            font=dict(size=13, color="#ffffff"),
            x=0, xanchor="left",
            pad=dict(l=4, b=12),
        ) if title else None,
        margin=dict(l=8, r=8, t=36 if title else 12, b=8),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor="#1e2a45", borderwidth=0.5,
            font=dict(size=10), orientation="h",
            yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        xaxis=dict(gridcolor="#1e2a45", linecolor="#1e2a45", tickcolor="#1e2a45",
                   tickfont=dict(size=10, color="#4a5568"), zeroline=False),
        yaxis=dict(gridcolor="#1e2a45", linecolor="#1e2a45", tickcolor="#1e2a45",
                   tickfont=dict(size=10, color="#4a5568"), zeroline=False),
        hoverlabel=dict(bgcolor="#1a1f35", bordercolor="#2a3050",
                        font=dict(size=11, color="#ffffff")),
    )
    return fig


def page_header(title, subtitle, badge_text=None, badge_color="#10b981"):
    badge_html = ""
    if badge_text:
        badge_html = f"""<span style="background:#0d2416;border:0.5px solid #0f3d22;
            border-radius:10px;padding:3px 10px;font-size:10px;color:{badge_color};
            font-weight:500;margin-left:10px">{badge_text}</span>"""
    st.markdown(f"""
    <div style="padding:8px 0 20px;border-bottom:1px solid #1e2a45;margin-bottom:20px">
      <div style="display:flex;align-items:center">
        <span style="font-size:20px;font-weight:600;color:#ffffff;letter-spacing:-0.02em">{title}</span>
        {badge_html}
      </div>
      <div style="font-size:12px;color:#4a5568;margin-top:4px">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# â”€â”€ Global CSS injection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def inject_global_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: #0e1117 !important;
    color: #c9d1e0 !important;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] {
    background-color: #0f1117 !important;
    border-right: 1px solid #1e2a45 !important;
    min-width: 240px !important; max-width: 240px !important;
}
section[data-testid="stSidebar"] .stRadio label {
    color: #8892a4 !important; font-size: 13px !important;
    padding: 6px 8px !important; border-radius: 6px !important;
    display: block; transition: all 0.15s;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: #1a1f35 !important; color: #fff !important;
}
[data-testid="stMetric"] {
    background-color: #161b2e !important; border: 0.5px solid #1e2a45 !important;
    border-radius: 10px !important; padding: 16px 18px !important;
}
[data-testid="stMetricLabel"] {
    font-size: 10px !important; text-transform: uppercase !important;
    letter-spacing: 0.06em !important; color: #4a5568 !important; font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-size: 22px !important; font-weight: 600 !important;
    color: #ffffff !important; letter-spacing: -0.02em !important;
}
[data-testid="stMetricDelta"] { font-size: 11px !important; }
.js-plotly-plot .plotly .modebar { display: none !important; }
.js-plotly-plot { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] {
    background: #161b2e !important; border: 0.5px solid #1e2a45 !important;
    border-radius: 10px !important;
}
.dataframe { color: #c9d1e0 !important; }
.dataframe th {
    background: #1a1f35 !important; color: #4a5568 !important;
    font-size: 10px !important; text-transform: uppercase !important;
    letter-spacing: 0.05em !important; border-bottom: 1px solid #1e2a45 !important;
}
.stButton > button {
    background: #1a1f35 !important; color: #c9d1e0 !important;
    border: 0.5px solid #2a3050 !important; border-radius: 8px !important;
    font-size: 13px !important; font-weight: 500 !important;
    padding: 8px 20px !important; transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #232a48 !important; color: #fff !important;
    border-color: #3b82f6 !important;
}
.stButton.primary > button {
    background: #3b82f6 !important; color: #fff !important;
    border-color: #3b82f6 !important;
}
.stSlider > div > div > div { background: #1e2a45 !important; }
.stSlider > div > div > div > div { background: #3b82f6 !important; }
.stSelectbox > div > div, .stTextInput > div > div {
    background: #161b2e !important; border: 0.5px solid #2a3050 !important;
    border-radius: 8px !important; color: #c9d1e0 !important;
}
.stAlert { border-radius: 8px !important; border-left-width: 3px !important; font-size: 13px !important; }
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important; border-bottom: 1px solid #1e2a45 !important; gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #4a5568 !important;
    font-size: 12px !important; font-weight: 500 !important;
    padding: 8px 16px !important; border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color: #3b82f6 !important; border-bottom-color: #3b82f6 !important; }
.stProgress > div > div { background: #3b82f6 !important; }
.streamlit-expanderHeader {
    background: #161b2e !important; border: 0.5px solid #1e2a45 !important;
    border-radius: 8px !important; color: #c9d1e0 !important; font-size: 13px !important;
}
hr { border-color: #1e2a45 !important; }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Data loading (unchanged logic) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        st.warning("Missing `data/simulation_results.csv` or `data/synthetic_users.csv`. Generate them first.")
        return False
    return True


# â”€â”€ Data helpers (unchanged logic) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def monthly_cumulative_rewards(results_df: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        results_df.groupby(["policy_display", "month"], as_index=False)["reward_received"]
        .sum().sort_values(["policy_display", "month"])
    )
    monthly["cumulative_reward"] = monthly.groupby("policy_display")["reward_received"].cumsum()
    return monthly


def default_rate_by_month(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.groupby(["policy_display", "month"], as_index=False)["did_default"].mean()


def default_rate_by_policy(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby("policy_display", as_index=False)["did_default"].mean().rename(
        columns={"did_default": "default_rate"})
    summary["default_rate_pct"] = summary["default_rate"] * 100.0
    return summary


def enrich_results(results_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    base_columns = ["user_id", "risk_tier", "income_bucket", "cibil_score",
                    "initial_credit_limit", "account_age_months", "employment_type"]
    available = [c for c in base_columns if c in users_df.columns]
    return results_df.merge(users_df[available], on="user_id", how="left")


# â”€â”€ Simulation helpers (unchanged logic) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        shocked.loc[flip_idx, "amount_spent"] * rng.uniform(0.72, 1.0, size=len(flip_idx)), 2)
    return shocked[outcomes.columns]


def run_live_thompson_simulation(
    users_df, economic_stress, fraud_spike_flag, risk_distribution, n_months,
    progress_bar, status_placeholder, chart_placeholder,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    live_users = build_custom_portfolio(users_df, risk_distribution, seed=42)
    context_builder = ContextBuilder(live_users)
    reward_buffer = RewardBuffer(delay_months=3)
    bandit = ThompsonSampling()
    bandit.rng = np.random.default_rng(42)
    current_limits = {row.user_id: float(row.initial_credit_limit) for row in live_users.itertuples(index=False)}
    histories: dict[str, list[dict]] = {uid: [] for uid in live_users["user_id"]}
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
            logs.append({"month": month, "user_id": user_id, "policy": "thompson_sim_live",
                         "action_taken": action, "reward_received": 0.0,
                         "current_limit": current_limits[user_id],
                         "did_default": False, "amount_spent": 0.0, "outstanding_amount": 0.0})
        simulation_input = live_users.copy()
        simulation_input["initial_credit_limit"] = simulation_input["user_id"].map(current_limits)
        stress = economic_stress
        if month == 6:
            stress = max(stress, 2.0)
        outcomes = simulate_month(simulation_input, month_num=month, economic_stress=stress)
        if fraud_spike_flag:
            outcomes = apply_fraud_spike(outcomes, live_users, month, rng)
        for outcome in outcomes.itertuples(index=False):
            histories[outcome.user_id].append({
                "month": month, "amount_spent": float(outcome.amount_spent),
                "outstanding_amount": float(outcome.outstanding_amount),
                "did_default": bool(outcome.did_default),
                "current_credit_limit": current_limits[outcome.user_id],
            })
            row = logs[row_lookup[(outcome.user_id, month)]]
            row["did_default"] = bool(outcome.did_default)
            row["amount_spent"] = float(outcome.amount_spent)
            row["outstanding_amount"] = float(outcome.outstanding_amount)
            reward_buffer.receive_outcome(outcome.user_id, month, outcome.amount_spent,
                                          outcome.outstanding_amount, outcome.did_default)
        ready_rewards = reward_buffer.get_ready_rewards(month)
        for rr in ready_rewards:
            bandit.update(rr["user_id"], rr["action"], float(rr["reward"]),
                          np.asarray(rr["context"], dtype=np.float32))
            logs[row_lookup[(rr["user_id"], month)]]["reward_received"] = float(rr["reward"])
        progress_bar.progress(month / n_months, text=f"Simulating month {month}/{n_months}...")
        partial_df = pd.DataFrame(logs)
        m_df = partial_df.groupby("month", as_index=False)["reward_received"].sum()
        m_df["cumulative_reward"] = m_df["reward_received"].cumsum()
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=m_df["month"], y=m_df["cumulative_reward"],
                                   mode="lines+markers", line=dict(color="#3b82f6", width=2),
                                   marker=dict(size=5), name="Thompson"))
        dark_chart(chart, title="Live Thompson Learning Curve", height=340)
        chart_placeholder.plotly_chart(chart, use_container_width=True)
    status_placeholder.success("âœ“ Simulation complete")
    return pd.DataFrame(logs)


# â”€â”€ PAGE 1: Portfolio Overview â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_portfolio_overview(results_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    page_header("Portfolio Overview",
                "12-month simulation Â· All policies vs static baseline",
                badge_text="â— Live")

    thompson_df = results_df.loc[results_df["policy"] == "thompson_sampling"]
    static_df = results_df.loc[results_df["policy"] == "static_baseline"]
    oracle_df = results_df.loc[results_df["policy"] == "oracle"]

    total_revenue = float(thompson_df["reward_received"].sum())
    static_revenue = float(static_df["reward_received"].sum())
    oracle_revenue = float(oracle_df["reward_received"].sum()) if not oracle_df.empty else total_revenue
    revenue_lift = ((total_revenue - static_revenue) / static_revenue * 100.0) if static_revenue else 0.0
    default_rate = float(thompson_df["did_default"].mean() * 100.0)
    regret_pct = ((oracle_revenue - total_revenue) / oracle_revenue * 100.0) if oracle_revenue else 0.0

    # Convergence month
    monthly_rew = thompson_df.groupby("month")["reward_received"].sum().sort_index()
    rolling = monthly_rew.rolling(window=3, min_periods=3).mean()
    conv_month = int(monthly_rew.index.max()) if not monthly_rew.empty else 12
    for m, imp in rolling.pct_change().replace([np.inf, -np.inf], np.nan).dropna().items():
        if imp < 0.01:
            conv_month = int(m)
            break

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue (Thompson)", format_inr(total_revenue),
              f"+{revenue_lift:.1f}% vs static")
    c2.metric("Default Rate", f"{default_rate:.2f}%",
              "Within 4% threshold" if default_rate <= 4 else "Above threshold",
              delta_color="normal" if default_rate <= 4 else "inverse")
    c3.metric("Regret vs Oracle", f"{regret_pct:.1f}%",
              "Below 20% target" if regret_pct < 20 else "Above 20% target",
              delta_color="normal" if regret_pct < 20 else "inverse")
    c4.metric("Convergence Month", f"Month {conv_month}",
              f"{conv_month} months of active learning")

    # Tabs
    tab_rev, tab_def, tab_act, tab_tier = st.tabs(["Revenue", "Default Rate", "Actions Taken", "By Risk Tier"])

    with tab_rev:
        cumulative = monthly_cumulative_rewards(results_df)
        fig = go.Figure()
        for policy in cumulative["policy_display"].unique():
            pdf = cumulative[cumulative["policy_display"] == policy]
            fig.add_trace(go.Scatter(x=pdf["month"], y=pdf["cumulative_reward"],
                                     mode="lines+markers", name=policy,
                                     line=dict(color=POLICY_COLORS.get(policy, "#888"), width=2),
                                     marker=dict(size=4)))
        fig.add_vline(x=6, line_dash="dash", line_color="#ef4444", line_width=1,
                      annotation_text="Economic shock", annotation_font_color="#ef4444",
                      annotation_font_size=10)
        dark_chart(fig, title="Cumulative Revenue by Month", height=360)
        st.plotly_chart(fig, use_container_width=True)

    with tab_def:
        def_df = default_rate_by_month(results_df)
        def_df["default_pct"] = def_df["did_default"] * 100.0
        fig2 = go.Figure()
        for policy in def_df["policy_display"].unique():
            pdf = def_df[def_df["policy_display"] == policy]
            fig2.add_trace(go.Scatter(x=pdf["month"], y=pdf["default_pct"],
                                      mode="lines+markers", name=policy,
                                      line=dict(color=POLICY_COLORS.get(policy, "#888"), width=2),
                                      marker=dict(size=4)))
        fig2.add_hline(y=4.0, line_dash="dash", line_color="#ef4444", line_width=1,
                       annotation_text="4% Threshold", annotation_font_color="#ef4444",
                       annotation_font_size=10)
        dark_chart(fig2, title="Default Rate by Month", height=360)
        st.plotly_chart(fig2, use_container_width=True)

    with tab_act:
        action_monthly = (thompson_df.groupby(["month", "action_taken"]).size()
                         .reset_index(name="count"))
        month_totals = action_monthly.groupby("month")["count"].transform("sum")
        action_monthly["pct"] = action_monthly["count"] / month_totals * 100
        fig3 = go.Figure()
        for action in ["keep", "plus_10", "plus_20", "plus_50"]:
            adf = action_monthly[action_monthly["action_taken"] == action]
            fig3.add_trace(go.Bar(x=adf["month"], y=adf["pct"], name=action,
                                  marker_color=ACTION_COLORS.get(action, "#888")))
        fig3.update_layout(barmode="stack")
        dark_chart(fig3, title="Thompson Sampling â€” Action Distribution (%)", height=360)
        st.plotly_chart(fig3, use_container_width=True)

    with tab_tier:
        enriched = enrich_results(results_df, users_df)
        thompson_enriched = enriched[enriched["policy"] == "thompson_sampling"]
        tier_names = ["Prime", "Near-Prime", "Subprime", "Deep-Subprime"]
        tc = st.columns(4)
        for i, tier in enumerate(tier_names):
            tier_data = thompson_enriched[thompson_enriched["risk_tier"] == tier]
            t_rev = float(tier_data["reward_received"].sum())
            t_def = float(tier_data["did_default"].mean() * 100) if not tier_data.empty else 0
            t_cnt = tier_data["user_id"].nunique()
            color = RISK_TIER_COLORS[tier]
            tc[i].markdown(f"""
            <div style="background:#161b2e;border:0.5px solid #1e2a45;border-radius:10px;padding:16px">
              <div style="font-size:13px;font-weight:600;color:{color}">{tier}</div>
              <div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em;margin-top:8px">Revenue</div>
              <div style="font-size:18px;font-weight:600;color:#fff">{format_inr(t_rev)}</div>
              <div style="font-size:10px;color:#4a5568;margin-top:6px">Default: <span style="color:#c9d1e0">{t_def:.1f}%</span> Â· Users: <span style="color:#c9d1e0">{t_cnt:,}</span></div>
            </div>""", unsafe_allow_html=True)

        # Grouped bar by tier
        tier_rev = []
        for policy in ["thompson_sampling", "ucb", "epsilon_greedy", "static_baseline"]:
            pdata = enriched[enriched["policy"] == policy]
            for tier in tier_names:
                tdata = pdata[pdata["risk_tier"] == tier]
                tier_rev.append({"Policy": POLICY_LABELS.get(policy, policy),
                                 "Risk Tier": tier,
                                 "Revenue": float(tdata["reward_received"].sum())})
        tier_rev_df = pd.DataFrame(tier_rev)
        fig4 = go.Figure()
        for policy in tier_rev_df["Policy"].unique():
            pdf = tier_rev_df[tier_rev_df["Policy"] == policy]
            fig4.add_trace(go.Bar(x=pdf["Risk Tier"], y=pdf["Revenue"], name=policy,
                                  marker_color=POLICY_COLORS.get(policy, "#888")))
        fig4.update_layout(barmode="group")
        dark_chart(fig4, title="Revenue by Risk Tier & Policy", height=360)
        st.plotly_chart(fig4, use_container_width=True)


# â”€â”€ PAGE 2: Per-User Deep Dive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_user_deep_dive(results_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    page_header("Per-User Deep Dive",
                "Inspect any user's full 12-month limit trajectory and decision history")

    user_ids = users_df["user_id"].tolist()
    if "selected_user_id" not in st.session_state:
        st.session_state.selected_user_id = user_ids[0]

    sel_col, btn_col = st.columns([3, 1])
    selected_input = sel_col.text_input("User ID", value=st.session_state.selected_user_id,
                                        placeholder="e.g. USER_00042")
    if btn_col.button("ðŸŽ² Random User"):
        st.session_state.selected_user_id = str(np.random.default_rng().choice(user_ids))
        st.rerun()
    elif selected_input in set(user_ids):
        st.session_state.selected_user_id = selected_input

    uid = st.session_state.selected_user_id
    user_row = users_df.loc[users_df["user_id"] == uid]
    if user_row.empty:
        st.markdown(f"""<div style="background:#161b2e;border:1px dashed #1e2a45;border-radius:10px;
            padding:32px;text-align:center;color:#4a5568">User <b>{uid}</b> not found</div>""",
            unsafe_allow_html=True)
        return
    user_row = user_row.iloc[0]

    user_results = results_df.loc[results_df["user_id"] == uid].copy()
    if user_results.empty:
        st.markdown("""<div style="background:#161b2e;border:1px dashed #1e2a45;border-radius:10px;
            padding:32px;text-align:center;color:#4a5568">No simulation records found</div>""",
            unsafe_allow_html=True)
        return

    # Only show Thompson data for this user
    thompson_user = user_results[user_results["policy"] == "thompson_sampling"].sort_values("month")
    if thompson_user.empty:
        thompson_user = user_results.sort_values("month")

    risk_tier = str(user_row.get("risk_tier", "Unknown"))
    rt_color = RISK_TIER_COLORS.get(risk_tier, "#4a5568")
    rt_bg = {"Prime": "#0d2416", "Near-Prime": "#0d1a2d", "Subprime": "#2a1d0a", "Deep-Subprime": "#2a0d0d"}.get(risk_tier, "#1a1f35")
    initials = uid[:2] if len(uid) >= 2 else "U"
    income = str(user_row.get("income_bucket", "â€”"))
    cibil = int(user_row["cibil_score"]) if "cibil_score" in user_row.index else "â€”"
    acct_age = int(user_row["account_age_months"]) if "account_age_months" in user_row.index else "â€”"
    init_limit = float(user_row["initial_credit_limit"]) if "initial_credit_limit" in user_row.index else 0
    emp_type = str(user_row.get("employment_type", "â€”"))

    st.markdown(f"""
    <div style="background:#161b2e;border:0.5px solid #1e2a45;border-radius:10px;padding:20px;
                display:flex;gap:20px;align-items:flex-start;margin-bottom:16px">
      <div style="width:48px;height:48px;border-radius:50%;background:{rt_color}20;
                  display:flex;align-items:center;justify-content:center;
                  font-size:16px;font-weight:600;color:{rt_color};flex-shrink:0">{escape(initials)}</div>
      <div style="flex:1;display:grid;grid-template-columns:repeat(3,1fr);gap:12px">
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Risk Tier</div>
          <span style="background:{rt_bg};border:0.5px solid {rt_color}33;border-radius:10px;
                padding:2px 8px;font-size:11px;color:{rt_color};font-weight:500;margin-top:4px;display:inline-block">{escape(risk_tier)}</span></div>
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Income</div>
          <div style="font-size:13px;color:#fff;margin-top:4px">{escape(income)}</div></div>
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">CIBIL Score</div>
          <div style="font-size:13px;color:#fff;margin-top:4px">{cibil}</div></div>
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Account Age</div>
          <div style="font-size:13px;color:#fff;margin-top:4px">{acct_age} months</div></div>
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Initial Limit</div>
          <div style="font-size:13px;color:#fff;margin-top:4px">{format_inr(init_limit)}</div></div>
        <div><div style="font-size:10px;color:#4a5568;text-transform:uppercase;letter-spacing:0.06em">Employment</div>
          <div style="font-size:13px;color:#fff;margin-top:4px">{escape(emp_type)}</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    limit_increases = int((thompson_user["action_taken"] != "keep").sum())
    default_count = int(thompson_user["did_default"].sum())
    total_user_rev = float(thompson_user["reward_received"].sum())
    st.markdown(f"""
    <div style="background:#0d1a2d;border:0.5px solid #1e2a45;border-radius:8px;
                padding:10px 16px;font-size:13px;color:#c9d1e0;margin-bottom:16px">
      This user received <b style="color:#3b82f6">{limit_increases}</b> limit increases over 12 months
      and defaulted <b style="color:#ef4444">{default_count}</b> times.
      Total revenue generated: <b style="color:#10b981">{format_inr(total_user_rev)}</b>.
    </div>""", unsafe_allow_html=True)

    # Charts
    left, right = st.columns(2)
    with left:
        fig_limit = go.Figure()
        fig_limit.add_trace(go.Scatter(x=thompson_user["month"], y=thompson_user["current_limit"],
                                        mode="lines", line=dict(color="#3b82f6", width=2), name="Limit"))
        actions_taken = thompson_user[thompson_user["action_taken"] != "keep"]
        if not actions_taken.empty:
            fig_limit.add_trace(go.Scatter(
                x=actions_taken["month"], y=actions_taken["current_limit"], mode="markers",
                marker=dict(size=8, color=[ACTION_COLORS.get(a, "#888") for a in actions_taken["action_taken"]]),
                name="Increase", showlegend=True))
        dark_chart(fig_limit, title="Credit Limit Trajectory", height=300)
        st.plotly_chart(fig_limit, use_container_width=True)

    with right:
        colors = ["#ef4444" if d else "#10b981" for d in thompson_user["did_default"]]
        fig_rew = go.Figure()
        fig_rew.add_trace(go.Bar(x=thompson_user["month"], y=thompson_user["reward_received"],
                                  marker_color=colors, name="Reward"))
        dark_chart(fig_rew, title="Monthly Reward", height=300)
        st.plotly_chart(fig_rew, use_container_width=True)

    # Table
    display_cols = thompson_user[["month", "action_taken", "current_limit", "amount_spent",
                                   "did_default", "reward_received"]].copy()
    display_cols.columns = ["Month", "Action Taken", "Credit Limit", "Amount Spent",
                            "Defaulted", "Reward"]
    st.dataframe(display_cols, use_container_width=True, hide_index=True)


# â”€â”€ PAGE 3: Policy Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_policy_comparison(results_df: pd.DataFrame) -> None:
    page_header("Policy Comparison",
                "Thompson Sampling vs UCB vs Epsilon-Greedy vs Oracle upper bound")

    # Winner banner
    policy_revenues = {}
    for policy_key, label in POLICY_LABELS.items():
        pdf = results_df[results_df["policy"] == policy_key]
        if not pdf.empty:
            policy_revenues[label] = float(pdf["reward_received"].sum())

    static_rev = policy_revenues.get("Static Baseline", 0)
    oracle_rev = policy_revenues.get("Oracle", 1)
    winner = max((k for k in policy_revenues if k not in ["Static Baseline", "Oracle"]),
                 key=lambda k: policy_revenues[k], default="Thompson Sampling")
    winner_rev = policy_revenues.get(winner, 0)
    lift = ((winner_rev - static_rev) / static_rev * 100) if static_rev else 0
    regret = ((oracle_rev - winner_rev) / oracle_rev * 100) if oracle_rev else 0

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0d1a2d,#1a1f35);
                border:0.5px solid #3b82f6;border-radius:10px;
                padding:14px 20px;margin-bottom:20px;
                display:flex;align-items:center;gap:12px">
      <span style="font-size:20px">ðŸ†</span>
      <div>
        <div style="font-size:13px;font-weight:600;color:#fff">
          {winner} wins with {format_inr(winner_rev)} total revenue</div>
        <div style="font-size:11px;color:#4a5568;margin-top:2px">
          {lift:.1f}% above static baseline Â· {regret:.1f}% regret vs oracle</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_reward, tab_regret, tab_summary = st.tabs(["Reward Curves", "Regret Analysis", "Summary Table"])

    with tab_reward:
        cumulative = monthly_cumulative_rewards(results_df)
        fig = go.Figure()
        for policy in cumulative["policy_display"].unique():
            pdf = cumulative[cumulative["policy_display"] == policy]
            fig.add_trace(go.Scatter(x=pdf["month"], y=pdf["cumulative_reward"],
                                     mode="lines+markers", name=policy,
                                     line=dict(color=POLICY_COLORS.get(policy, "#888"), width=2),
                                     marker=dict(size=4)))
        fig.add_vline(x=6, line_dash="dash", line_color="#ef4444", line_width=1,
                      annotation_text="Economic shock", annotation_font_color="#ef4444",
                      annotation_font_size=10)
        dark_chart(fig, title="Cumulative Reward â€” All Policies", height=360)
        st.plotly_chart(fig, use_container_width=True)

        # Monthly delta Thompson - Static
        t_monthly = results_df[results_df["policy"] == "thompson_sampling"].groupby("month")["reward_received"].sum()
        s_monthly = results_df[results_df["policy"] == "static_baseline"].groupby("month")["reward_received"].sum()
        delta_df = pd.DataFrame({"month": t_monthly.index, "delta": (t_monthly - s_monthly).values})
        colors_delta = ["#10b981" if d >= 0 else "#ef4444" for d in delta_df["delta"]]
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(x=delta_df["month"], y=delta_df["delta"],
                                   marker_color=colors_delta, name="Î” Revenue"))
        dark_chart(fig_delta, title="Monthly Revenue Delta (Thompson âˆ’ Static)", height=280)
        st.plotly_chart(fig_delta, use_container_width=True)

    with tab_regret:
        oracle_df = results_df.loc[results_df["policy"] == "oracle"]
        fig_reg = go.Figure()
        for policy_key, label in [("thompson_sampling", "Thompson Sampling"),
                                   ("ucb", "UCB"), ("epsilon_greedy", "Epsilon-Greedy")]:
            policy_df = results_df.loc[results_df["policy"] == policy_key]
            if policy_df.empty:
                continue
            regret_df = compute_regret(oracle_df, policy_df)
            fig_reg.add_trace(go.Scatter(
                x=regret_df["month"], y=regret_df["regret"],
                mode="lines+markers", name=label,
                line=dict(color=POLICY_COLORS.get(label, "#888"), width=2),
                marker=dict(size=4)))
            # Annotation at final month
            final = regret_df.iloc[-1] if not regret_df.empty else None
            if final is not None:
                fig_reg.add_annotation(x=final["month"], y=final["regret"],
                                       text=f"{final['regret_pct']:.1f}%",
                                       showarrow=True, arrowhead=2, arrowcolor="#4a5568",
                                       font=dict(size=10, color=POLICY_COLORS.get(label, "#888")))
        dark_chart(fig_reg, title="Cumulative Regret vs Oracle", height=360)
        st.plotly_chart(fig_reg, use_container_width=True)

    with tab_summary:
        policy_frames = {policy: frame.copy() for policy, frame in results_df.groupby("policy")}
        comparison = policy_comparison_table(policy_frames)
        comparison["policy"] = comparison["policy"].map(POLICY_LABELS).fillna(comparison["policy"])

        # Build HTML table
        best_rev = comparison["total_revenue_inr"].max()
        best_lift = comparison["revenue_lift_vs_static_pct"].max()
        best_def = comparison["default_rate_pct"].min()
        best_regret = comparison.loc[comparison["policy"] != "Oracle", "regret_vs_oracle_pct"].min() if len(comparison) > 1 else 0

        rows_html = ""
        for _, r in comparison.iterrows():
            is_oracle = r["policy"] == "Oracle"
            text_color = "#4a5568" if is_oracle else "#c9d1e0"
            def hl(val, best, fmt):
                c = "#3b82f6" if abs(val - best) < 0.01 and not is_oracle else text_color
                return f'<span style="color:{c};font-weight:{"600" if c=="#3b82f6" else "400"}">{fmt}</span>'

            rows_html += f"""<tr style="border-bottom:0.5px solid #1e2a45">
              <td style="padding:10px 12px;color:{text_color};font-weight:500">{r['policy']}</td>
              <td style="padding:10px 12px">{hl(r['total_revenue_inr'], best_rev, format_inr(r['total_revenue_inr']))}</td>
              <td style="padding:10px 12px">{hl(r['revenue_lift_vs_static_pct'], best_lift, f"{r['revenue_lift_vs_static_pct']:.1f}%")}</td>
              <td style="padding:10px 12px">{hl(r['default_rate_pct'], best_def, f"{r['default_rate_pct']:.2f}%")}</td>
              <td style="padding:10px 12px">{hl(r['regret_vs_oracle_pct'], best_regret, f"{r['regret_vs_oracle_pct']:.1f}%")}</td>
              <td style="padding:10px 12px;color:{text_color}">{int(r['convergence_month'])}</td>
              <td style="padding:10px 12px;color:{text_color}">{r['exploration_ratio_pct']:.1f}%</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#161b2e;border:0.5px solid #1e2a45;border-radius:10px;overflow:hidden">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead><tr style="border-bottom:1px solid #1e2a45">
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Policy</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Revenue</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Lift vs Static</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Default Rate</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Regret vs Oracle</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Convergence</th>
              <th style="padding:10px 12px;text-align:left;color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.05em;font-weight:500">Explore %</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)


# â”€â”€ PAGE 4: Live Simulation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_live_simulation(users_df: pd.DataFrame, results_df: pd.DataFrame) -> None:
    page_header("Live Simulation",
                "Adjust parameters and re-run Thompson Sampling in real time")

    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown("""<div style="background:#161b2e;border:0.5px solid #1e2a45;
            border-radius:10px;padding:20px">
            <div style="font-size:14px;font-weight:600;color:#fff;margin-bottom:16px">
            Simulation Parameters</div>
        </div>""", unsafe_allow_html=True)

        economic_stress = st.slider("Economic stress multiplier",
                                     min_value=1.0, max_value=3.0, value=1.0, step=0.1,
                                     help="1.0 = normal, 2.0 = default rates double (COVID-like shock)")
        fraud_spike_toggle = st.toggle("Fraud spike at month 9",
                                        help="Doubles Deep-Subprime defaults at month 9")

        st.markdown("""<div style="font-size:12px;font-weight:500;color:#c9d1e0;margin:12px 0 4px">
            Risk distribution (must sum to 100%)</div>""", unsafe_allow_html=True)
        prime_pct = st.slider("Prime %", 10, 70, 40)
        near_prime_pct = st.slider("Near-Prime %", 10, 60, 30)
        subprime_pct = st.slider("Subprime %", 5, 50, 20)
        deep_sub_pct = 100 - prime_pct - near_prime_pct - subprime_pct

        if deep_sub_pct < 0:
            st.warning("âš ï¸ Distribution exceeds 100% â€” adjust sliders")
        else:
            st.markdown(f"""
            <div style="background:#161b2e;border:0.5px solid #1e2a45;border-radius:6px;
                        padding:8px 12px;font-size:11px;color:#4a5568;margin-top:4px">
              Deep-Subprime auto-set to <b style="color:#ef4444">{deep_sub_pct}%</b>
            </div>""", unsafe_allow_html=True)

        n_months = st.select_slider("Horizon", options=[6, 9, 12], value=12)
        run_btn = st.button("â–¶ Run Simulation", use_container_width=True)

    risk_distribution = {"Prime": prime_pct, "Near-Prime": near_prime_pct,
                         "Subprime": subprime_pct, "Deep-Subprime": max(deep_sub_pct, 0)}

    with right_col:
        if run_btn and deep_sub_pct >= 0:
            progress_bar = st.progress(0, text="Initializing simulation...")
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            live_results = run_live_thompson_simulation(
                users_df=users_df, economic_stress=economic_stress,
                fraud_spike_flag=fraud_spike_toggle,
                risk_distribution=risk_distribution, n_months=n_months,
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
                chart_placeholder=chart_placeholder,
            )
            st.session_state.live_results = live_results
            st.session_state.live_n_months = n_months

        live_results = st.session_state.get("live_results")
        if live_results is None:
            st.markdown("""
            <div style="background:#161b2e;border:1px dashed #1e2a45;
                        border-radius:10px;padding:48px;text-align:center">
              <div style="font-size:32px;margin-bottom:12px">âš¡</div>
              <div style="color:#4a5568;font-size:13px">
                Configure parameters and click Run Simulation</div>
            </div>""", unsafe_allow_html=True)
        else:
            static_df = results_df.loc[results_df["policy"] == "static_baseline"]
            static_total = float(static_df["reward_received"].sum()) if not static_df.empty else 0.0
            live_total = float(live_results["reward_received"].sum())
            rev_lift = ((live_total - static_total) / static_total * 100) if static_total else 0
            def_rate = float(live_results["did_default"].mean() * 100)

            st.markdown(f"""
            <div style="background:#0d2416;border:0.5px solid #0f3d22;border-radius:8px;
                        padding:10px 16px;font-size:13px;color:#10b981;margin-bottom:12px">
              âœ“ Simulation complete â€” Thompson Sampling: {format_inr(live_total)} total revenue
            </div>""", unsafe_allow_html=True)

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Revenue vs Standard", f"{rev_lift:+.1f}%")
            mc2.metric("Default Rate", f"{def_rate:.2f}%")
            n_m = st.session_state.get("live_n_months", 12)
            # Convergence
            m_rew = live_results.groupby("month")["reward_received"].sum().sort_index()
            roll = m_rew.rolling(window=3, min_periods=3).mean()
            conv = int(m_rew.index.max()) if not m_rew.empty else n_m
            for m, imp in roll.pct_change().replace([np.inf, -np.inf], np.nan).dropna().items():
                if imp < 0.01:
                    conv = int(m)
                    break
            mc3.metric("Convergence Month", f"Month {conv}")

            # Cumulative chart
            monthly = live_results.groupby("month", as_index=False)["reward_received"].sum()
            monthly["cumulative_reward"] = monthly["reward_received"].cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["cumulative_reward"],
                                     mode="lines+markers", line=dict(color="#3b82f6", width=2),
                                     marker=dict(size=5), name="Thompson (Live)"))
            dark_chart(fig, title="Cumulative Revenue", height=320)
            st.plotly_chart(fig, use_container_width=True)

            # Side by side: default rate + action dist
            d1, d2 = st.columns(2)
            with d1:
                def_monthly = live_results.groupby("month", as_index=False)["did_default"].mean()
                def_monthly["pct"] = def_monthly["did_default"] * 100
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(x=def_monthly["month"], y=def_monthly["pct"],
                                           mode="lines+markers", line=dict(color="#ef4444", width=2),
                                           marker=dict(size=4), name="Default Rate"))
                fig_d.add_hline(y=4.0, line_dash="dash", line_color="#4a5568", line_width=1)
                dark_chart(fig_d, title="Default Rate by Month", height=280)
                st.plotly_chart(fig_d, use_container_width=True)

            with d2:
                act_m = live_results.groupby(["month", "action_taken"]).size().reset_index(name="count")
                mt = act_m.groupby("month")["count"].transform("sum")
                act_m["pct"] = act_m["count"] / mt * 100
                fig_a = go.Figure()
                for action in ["keep", "plus_10", "plus_20", "plus_50"]:
                    adf = act_m[act_m["action_taken"] == action]
                    fig_a.add_trace(go.Bar(x=adf["month"], y=adf["pct"], name=action,
                                           marker_color=ACTION_COLORS.get(action, "#888")))
                fig_a.update_layout(barmode="stack")
                dark_chart(fig_a, title="Action Distribution (%)", height=280)
                st.plotly_chart(fig_a, use_container_width=True)


# â”€â”€ SIDEBAR & MAIN â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 16px">
          <div style="font-size:15px;font-weight:600;color:#fff;letter-spacing:-0.01em">
            ðŸ’³ Credit Limit Bandit</div>
          <div style="font-size:10px;color:#4a5568;margin-top:3px;
                      text-transform:uppercase;letter-spacing:0.06em">
            Contextual RL Â· Fintech</div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        page = st.radio("Navigate",
                        ["ðŸ’¼  Portfolio Overview", "ðŸ‘¤  Per-User Deep Dive",
                         "ðŸ“Š  Policy Comparison", "âš¡  Live Simulation"],
                        label_visibility="collapsed")

        st.divider()

        st.markdown("""<div style="font-size:10px;color:#4a5568;text-transform:uppercase;
                    letter-spacing:0.06em;margin-bottom:10px">Simulation Config</div>""",
                    unsafe_allow_html=True)

        config_items = {"Users": "10,000", "Horizon": "12 months", "Reward delay": "3 months",
                        "Risk tiers": "4", "Actions": "Keep / +10% / +20% / +50%"}
        for k, v in config_items.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        padding:4px 0;border-bottom:0.5px solid #1e2a45">
              <span style="font-size:11px;color:#4a5568">{k}</span>
              <span style="font-size:11px;color:#c9d1e0;font-weight:500">{v}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#0d2416;border:0.5px solid #0f3d22;border-radius:6px;
                    padding:7px 10px;font-size:11px;color:#10b981;
                    display:flex;align-items:center;gap:6px">
          <span style="width:6px;height:6px;border-radius:50%;
                       background:#10b981;display:inline-block"></span>
          Results loaded</div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="position:absolute;bottom:16px;left:16px;right:16px;
                    font-size:10px;color:#2a3050;text-align:center">
          Thompson Sampling Â· UCB Â· Îµ-Greedy<br/>
          <a href="{GITHUB_URL}" style="color:#3b82f6;text-decoration:none">GitHub â†—</a>
        </div>""", unsafe_allow_html=True)

    return page


def main() -> None:
    results_df = load_simulation_results()
    users_df = load_synthetic_users()

    inject_global_styles()
    page = render_sidebar()

    page_map = {
        "ðŸ’¼  Portfolio Overview": "Portfolio Overview",
        "ðŸ‘¤  Per-User Deep Dive": "Per-User Deep Dive",
        "ðŸ“Š  Policy Comparison": "Policy Comparison",
        "âš¡  Live Simulation": "Live Simulation",
    }
    selected_page = page_map.get(page, "Portfolio Overview")

    if selected_page in {"Portfolio Overview", "Per-User Deep Dive", "Policy Comparison"} and not require_data(results_df, users_df):
        return
    if selected_page == "Live Simulation" and users_df.empty:
        st.warning("Missing `data/synthetic_users.csv`. Generate it first.")
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

