# Interactive Dashboard

Streamlit app with 4 pages: Portfolio Overview, Per-User Deep Dive, Policy Comparison, Live Simulation.

## Launch

```bash
streamlit run dashboard/app.py
# Opens http://localhost:8501
```

## Page 1: Portfolio Overview

Shows 5 policies (Thompson, UCB, Epsilon, Static, Oracle) over 12 months.

- **Revenue Curves:** Line chart showing cumulative ₹ over time. Thompson reaches ₹12.13Cr; static ₹8.72Cr.
- **Monthly Revenue:** Bar chart ₹/month by policy.
- **Default Rate Timeline:** Tracks 3.38% constraint across months, alerts if violated.
- **Action Distribution:** Stacked bar (% keep / plus_10 / plus_20 / plus_50 by month).

**What to look for:** Thompson curve should outpace others consistently. Month 6 shock should show default rate spike for all policies (economic shock) but Thompson recovers quickest.

## Page 2: Per-User Deep Dive

Select a specific user (dropdown 1-10,000) and view their 12-month trajectory.

- **User Info:** CIBIL, income, risk tier, starting limit
- **Monthly Actions:** Table with month, limit, spent, reward, default status
- **Limit Evolution:** Trajectory from starting ₹X to ending ₹Y after 12 months
- **Reward Trajectory:** Monthly rewards showing when defaults impact revenue

**What to look for:** Prime users should accumulate limits (plus_10/plus_20 actions), Subprime users should stay on keep. Defaults should be rare (3.38% probability).

## Page 3: Policy Comparison

Side-by-side comparison of Thompson (ours) vs UCB vs Epsilon vs Static vs Oracle.

- **Metrics Table:** Revenue (₹12.13Cr vs ₹10.83Cr vs ₹8.70Cr vs ₹8.72Cr vs ₹14.50Cr), lift (+39.09% vs +24.15% vs −0.26% vs 0% vs +66.39%), regret (16.34% vs 25.32% vs 40.01% vs 39.80% vs 0%)
- **Regret Curves:** Month-by-month regret showing convergence (Thompson: Month 5; Epsilon: Month 7)
- **Risk Tier Breakdown:** Revenue by Prime/Near-Prime/Subprime/Deep-Subprime for each policy

**What to look for:** Thompson dominant across all tiers. UCB solid but not best. Epsilon-Greedy underperforming (−0.26%).

## Page 4: Live Simulation

Re-run Thompson Sampling with custom parameters:

- **Stress Slider:** 1.0 (normal) to 3.0 (severe shock)
- **User Distribution:** Adjust Prime/Near-Prime/Subprime/Deep-Subprime split
- **Button:** "Run Simulation"

Output: Real-time results showing how Thompson adapts to your chosen scenario.

**What to look for:** Higher stress → default rates rise but stay <5%. Thompson adapts by shifting toward keep action. Changing user distribution should show tier-specific optimal actions.

## Related Docs

- [running-simulations.md](running-simulations.md)
- [../results/interpreting-results.md](../results/interpreting-results.md)
