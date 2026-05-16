---
title: Metrics Explained
category: results
file_reference: src/evaluate.py
---
# Metrics Explained

This file defines the primary metrics used to evaluate policies. For each metric: Name · Formula · Unit · Interpretation · Target value · Bad value meaning.

1. Cumulative Revenue
- Formula: sum_t sum_u net_reward_{u,t}
- Unit: INR
- Interpretation: total economic gain achieved by the policy.
- Target: maximize; larger is better.
- Bad: low or negative → policy is destructive.

2. Revenue Lift
- Formula: (policy_revenue - static_revenue) / static_revenue × 100
- Unit: percent
- Interpretation: relative improvement vs a static baseline.
- Target: positive lift; >5% is meaningful in this prototypical setting.
- Bad: near zero or negative → policy not adding value.

3. Default Rate
- Formula: defaults / total_users per month
- Unit: percent
- Interpretation: portfolio risk; a constraint not an objective.
- Target: keep under business cap (e.g., 4%).
- Bad: rising above cap → pause or tighten policy.

4. Regret
- Formula: oracle_revenue - policy_revenue (cumulative)
- Unit: INR
- Interpretation: opportunity cost relative to hindsight-optimal policy.
- Target: minimize; lower is better.

5. Regret %
- Formula: regret / oracle_revenue × 100
- Unit: percent
- Interpretation: relative performance gap.

6. Convergence Month
- Formula: first month where 3-month rolling improvement < 1%
- Unit: month index
- Interpretation: when learning stabilizes; earlier is better for quick deployment.

7. Exploration Ratio
- Formula: % of decisions that were exploratory (not keep)
- Unit: percent
- Interpretation: how aggressively the policy experiments.

Why regret is relative to oracle
- The oracle is the upper bound computed in hindsight; using it as a baseline measures absolute headroom. Static baselines are useful too, but oracle quantifies the ceiling.

Default rate as a constraint
- The business objective is to maximize revenue subject to default rates remaining below a threshold; this makes default a constraint, not a direct objective to minimize.

Statistical significance
- Use bootstrapping across users or multiple seeds to estimate confidence intervals for revenue lift. A p-value < 0.05 or non-overlapping 95% CIs indicates a meaningful difference.

Related docs
- [docs/evaluation.md](docs/components/evaluation.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
