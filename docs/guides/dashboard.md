---
title: Streamlit Dashboard Guide
category: guide
file_reference: dashboard/app.py
---
# Streamlit Dashboard Guide

How to launch
```bash
streamlit run dashboard/app.py
```
- Streamlit Cloud: push repository and set `streamlit run dashboard/app.py` as run command.

Page 1 — Portfolio Overview
- Top metric cards: cumulative revenue, default rate, regret vs static.
- Revenue chart: monthly revenue for each policy; read relative gaps and spikes at shock month.

Page 2 — Per-User Deep Dive
- Select a `user_id` to see limit trajectory, monthly spend, and default flags.
- Use the timeline to inspect actions taken and correlate with future defaults.

Page 3 — Policy Comparison
- Regret curves: compare cumulative regret vs the oracle.
- Convergence visual: 3-month rolling improvement to find `Convergence Month`.

Page 4 — Live Simulation
- Sliders: `n_users`, `n_months`, `economic_shock` toggle, `seed`.
- Run triggers a short simulation locally and updates the charts; useful for demos.

Tips for demo
- Start on Portfolio Overview to show business-level KPIs, then drill into Policy Comparison for algorithmic detail.
- Highlight how Thompson crosses UCB by month ~3 and how default rate behaves at the shock.

Related docs
- [docs/guides/getting-started.md](docs/guides/getting-started.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
