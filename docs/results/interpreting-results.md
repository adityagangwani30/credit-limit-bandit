---
title: Interpreting Simulation Results
category: results
file_reference: none
---
# How to Interpret Simulation Results

What good results look like
- Thompson Sampling should surpass UCB and Epsilon-Greedy by month 3–4 in cumulative revenue in typical parameterizations.
- Default rate should spike at month 6 (shock) and recover within ~2 months in adaptive policies.
- Exploration ratio should decline from ~25% to ~10% across 12 months.
- Regret curve: concave shape with faster improvement early and slower asymptotic gains.

Red flags and fixes
- Thompson ≈ Static baseline: likely reward normalization bug or action mapping issue. Check `src/reward.py` normalization pipeline.
- Default rate never recovers after month 6: ensure `update()` is called with released rewards and posterior counts are updated; check `RewardBuffer` release logic.
- Exploration ratio stuck at 100%: `select_action` may always sample exploratory branch; inspect epsilon parameters or sampling code.
- Oracle revenue == static baseline: oracle computation may be wrong—ensure it selects the best realized action per user-month, not a constant action.

Reading cohort heatmaps
- Rows: `risk_tier`, Columns: `income_bucket`. Cells show revenue lift vs static. High positive lift in middle-income, near-prime cohorts indicates profitable personalization window.

Practical interpretation
- Business decision: prefer a policy that yields positive Revenue Lift while keeping Default Rate below the risk cap. Smaller, sustained lift with tight risk control often beats larger but volatile gains.

Related docs
- [docs/results/metrics-explained.md](docs/results/metrics-explained.md)
- [docs/guides/dashboard.md](docs/guides/dashboard.md)
