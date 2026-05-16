---
title: Evaluation Module
category: component
file_reference: src/evaluate.py
---
# Evaluation Module

`src/evaluate.py` computes portfolio-level and cohort-level metrics to judge policy performance. The evaluator supports offline oracle computation, cohort heatmaps, and shock analysis.

Metrics (7 defined)
1. Cumulative Revenue: sum of net rewards across users and months (units: INR). Interpretation: absolute return from the policy.
2. Revenue Lift: (policy_revenue - static_revenue) / static_revenue × 100. Interpretation: percent improvement vs a static baseline.
3. Default Rate: defaults / total_users per month (percent). Interpretation: risk constraint; must be kept under the business cap (e.g., 4%).
4. Regret: oracle_revenue - policy_revenue (cumulative, INR). Interpretation: absolute opportunity cost vs hindsight oracle.
5. Regret %: regret / oracle_revenue × 100. Interpretation: relative performance gap.
6. Convergence Month: first month where 3-month rolling improvement < 1%. Interpretation: when learning stabilizes.
7. Exploration Ratio: percent of decisions that were not "keep" or default actions. Interpretation: how much the policy explores.

Cohort analysis
- Group by `risk_tier` × `income_bucket` and compute per-cohort revenue lift, default rate, and regret. Cohort heatmaps reveal where personalization helps or harms.

Oracle policy
- Computed in hindsight by selecting the action with maximum realized reward per user-month. It serves as a ceiling; algorithms cannot use its decisions online.

Economic shock analysis
- When an economic stress factor increases at month 6, compute recovery time as months required for the policy default rate to return within 10% of pre-shock baseline.

Implementation notes
- Functions: `compute_metrics(df)`, `compute_oracle(df)`, `cohort_heatmap(df, by=['risk_tier','income_bucket'])`.
- Tests in `tests/test_integration.py` assert metric calculations against deterministic synthetic subsets.

Related docs
- [docs/results/metrics-explained.md](docs/results/metrics-explained.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
